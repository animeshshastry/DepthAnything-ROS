#include <rclcpp/rclcpp.hpp>
#include <optional>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/synchronizer.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

class DepthAlignNode : public rclcpp::Node {
public:
    DepthAlignNode() : Node("depth_align_node") {

        // TF buffer and listener
        tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);

        int syncQueueSize = 100;
        syncQueueSize = this->declare_parameter("sync_queue_size", syncQueueSize);
        maxDepth = this->declare_parameter("max_depth", maxDepth);
        pub_cloud = this->declare_parameter("pub_cloud", pub_cloud);
        odom_frame_id = this->declare_parameter("odom_frame_id", odom_frame_id);
        duration = this->declare_parameter("duration", duration);
        decimation = this->declare_parameter("decimation", decimation);
        voxel_size = this->declare_parameter("voxel_size", voxel_size);

        decimation = std::max<uint8_t>(1, decimation);

        // Setup subscribers
        auto qos = rclcpp::QoS(10).reliable();
        cameraInfoSub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("depth_image/camera_info", qos, 
						std::bind(&DepthAlignNode::cameraInfo_callback, this, std::placeholders::_1));

        sparse_sub_.subscribe(this, "sparse_topic", rmw_qos_profile_services_default);
        dense_sub_.subscribe(this, "dense_topic", rmw_qos_profile_services_default);

        // Synchronizer
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(syncQueueSize), sparse_sub_, dense_sub_));
        sync_->registerCallback(std::bind(&DepthAlignNode::callback, this, std::placeholders::_1, std::placeholders::_2));

        // Publisher
        depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 10);
        if (pub_cloud) cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("depth/pointcloud", 10);
    }

private:
    typedef message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::Image> sparse_sub_, dense_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSub_;

    std::shared_ptr<tf2_ros::Buffer> tfBuffer_;
    std::shared_ptr<tf2_ros::TransformListener> tfListener_;

    bool pub_cloud = false;
    float maxDepth = 100.0;
    uint8_t decimation = 1;
    float voxel_size = 0.0f;
    std::string odom_frame_id = "odom";
    float duration = 0.1;

	bool camera_info_received_ = false;
	double fx, fy, cx, cy, k1, k2, p1, p2, k3;
	uint16_t width, height;
    cv::Mat undist_map_;   // stores normalized coordinates (x = X/Z, y = Y/Z)

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& sparse_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& dense_msg) {
        // Convert to OpenCV
        cv::Mat sparse = cv_bridge::toCvShare(sparse_msg, "16UC1")->image; // depth in mm
        cv::Mat disp8  = cv_bridge::toCvShare(dense_msg, "mono8")->image;

        sparse.convertTo(sparse, CV_32FC1, 1.0/1000.0); // convert to meters

        // Convert disparity -> depth (predicted)
        cv::Mat dense;
        disp8.convertTo(dense, CV_32FC1); // avoid integer division
        dense = 1.0f / (dense + 1e-6f); // dense_depth

        std::vector<float> xs, ys;
        for (uint16_t v = 0; v < sparse.rows; v++) {
            for (uint16_t u = 0; u < sparse.cols; u++) {
                float d_sparse = sparse.at<float>(v, u);
                float d_pred   = dense.at<float>(v, u);
                if (d_sparse < maxDepth && d_sparse > 0.0f && d_pred > 0.0f) { // consider valid sparse points only
                    ys.push_back(d_sparse);
                    xs.push_back(d_pred);
                }
            }
        }

        if (xs.size() < 4) {
            RCLCPP_WARN(this->get_logger(), "Not enough valid points (%zu) to compute scale/bias.",
                        xs.size());
            return;
        }

        // Mean scaling
        double sum = 0.0;
        for (size_t i = 0; i < xs.size(); i++) {
            sum += ys[i]/xs[i];
        }
        double s = sum/xs.size();

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Scale: %.4f, (using %zu points)",
                             s, xs.size());

        // Apply correction
        cv::Mat corrected = s * dense;
        corrected.convertTo(corrected, CV_32FC1);

        // Set values greater than max depth to infinity
        cv::Mat mask = corrected > maxDepth;
        corrected.setTo(std::numeric_limits<float>::infinity(), mask);

        // Publish
        auto out_msg = cv_bridge::CvImage(dense_msg->header, "32FC1", corrected).toImageMsg();
        depth_pub_->publish(*out_msg);

        if (!pub_cloud) return;
        auto cloud_msg = depthToPointCloud(corrected, dense_msg->header, odom_frame_id, duration, decimation, voxel_size);
        if (cloud_msg) cloud_pub_->publish(*cloud_msg);
    }

    std::optional<sensor_msgs::msg::PointCloud2> depthToPointCloud(
        const cv::Mat &depth,
        const std_msgs::msg::Header &header,
        const std::string &odom_frame_id,
        const float duration = 0.1,
        const int decimation = 2,
        const float voxel_size = 0.05f)
    {
        // --- Step 1: Get transform ---
        geometry_msgs::msg::TransformStamped transform_stamped;
        try {
            transform_stamped = tfBuffer_->lookupTransform(
                header.frame_id, odom_frame_id, header.stamp, tf2::durationFromSec(duration));
        }
        catch (tf2::TransformException &ex) {
            RCLCPP_WARN(rclcpp::get_logger("depthToPointCloud"),
                        "No transform %s -> %s: %s",
                        header.frame_id.c_str(), odom_frame_id.c_str(), ex.what());
            return std::nullopt; // nothing to publish
        }
    
        // Convert to tf2::Transform for fast math
        tf2::Transform tf_cam_to_odom;
        tf2::fromMsg(transform_stamped.transform, tf_cam_to_odom);
        tf_cam_to_odom = tf_cam_to_odom.inverse();

        // --- Step 2: Allocate cloud in odom frame ---
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header = header;
        cloud_msg.header.frame_id = odom_frame_id; // directly output in odom
    
        cloud_msg.height = (depth.rows + decimation - 1) / decimation;
        cloud_msg.width  = (depth.cols + decimation - 1) / decimation;
        cloud_msg.is_dense = false;
    
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(cloud_msg.height * cloud_msg.width);
    
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    
        // --- Step 3: Generate transformed points directly ---
        for (int v = 0; v < depth.rows; v += decimation) {
            for (int u = 0; u < depth.cols; u += decimation, ++iter_x, ++iter_y, ++iter_z) {
                float Z = depth.at<float>(v, u);
                if (!std::isfinite(Z) || Z <= 0.0f) {
                    *iter_x = *iter_y = *iter_z = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
    
                cv::Vec2f norm_xy = undist_map_.at<cv::Vec2f>(v, u);
                tf2::Vector3 pt_cam(norm_xy[0] * Z, norm_xy[1] * Z, Z);
    
                // Apply transform once
                tf2::Vector3 pt_odom = tf_cam_to_odom * pt_cam;
                // tf2::Vector3 pt_odom = pt_cam;
    
                *iter_x = pt_odom.x();
                *iter_y = pt_odom.y();
                *iter_z = pt_odom.z();
            }
        }

        if (!voxel_size || voxel_size <= 0.0f) {
            return cloud_msg; // skip filtering
        }

        // --- Step 2: Apply voxel grid filtering (PCL) ---
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(cloud_msg, *pcl_cloud);

        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(pcl_cloud);
        voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);

        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        voxel_filter.filter(*pcl_cloud_filtered);

        sensor_msgs::msg::PointCloud2 cloud_msg_filtered;
        pcl::toROSMsg(*pcl_cloud_filtered, cloud_msg_filtered);
        cloud_msg_filtered.header = cloud_msg.header;

        return cloud_msg_filtered;
    }

    void cameraInfo_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info) {
        if (!camera_info_received_) {
            // --- Camera intrinsics ---
            fx = cam_info->k[0];
            fy = cam_info->k[4];
            cx = cam_info->k[2];
            cy = cam_info->k[5];

            if (!cam_info->d.empty()) {
                k1 = cam_info->d[0];
                k2 = cam_info->d[1];
                p1 = cam_info->d[2];
                p2 = cam_info->d[3];
                k3 = cam_info->d.size() > 4 ? cam_info->d[4] : 0.0;
            }

            width  = cam_info->width;
            height = cam_info->height;

            if (width == 0 || height == 0) {
                RCLCPP_ERROR(this->get_logger(), "CameraInfo has invalid width/height: %u x %u", width, height);
                return;
            }

            // --- Build undistortion map ---
            std::vector<cv::Point2f> pts;
            pts.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));

            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {
                    pts.emplace_back(static_cast<float>(u), static_cast<float>(v));
                }
            }

            cv::Matx33d K(fx, 0, cx,
                        0, fy, cy,
                        0, 0, 1);

            cv::Vec<double, 5> D(k1, k2, p1, p2, k3);

            std::vector<cv::Point2f> undistorted_pts;
            cv::undistortPoints(pts, undistorted_pts, K, D);

            undist_map_ = cv::Mat(height, width, CV_32FC2);
            int idx = 0;
            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {
                    undist_map_.at<cv::Vec2f>(v, u) = cv::Vec2f(
                        undistorted_pts[idx].x,
                        undistorted_pts[idx].y
                    );
                    idx++;
                }
            }

            RCLCPP_INFO(this->get_logger(), "Received camera info and built undistortion map.");
            camera_info_received_ = true;
        }
    }

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthAlignNode>());
    rclcpp::shutdown();
    return 0;
}
