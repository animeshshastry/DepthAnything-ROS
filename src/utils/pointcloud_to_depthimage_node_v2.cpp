#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class pointcloud_to_depthimage_node : public rclcpp::Node {
public:
    pointcloud_to_depthimage_node() : Node("pointcloud_to_depthimage") {

        tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);

        fixedFrameId_ = this->declare_parameter("fixed_frame_id", fixedFrameId_);
		camFrameId_ = this->declare_parameter("cam_frame_id", camFrameId_);
        
		auto qos = rclcpp::QoS(10).reliable();

		// Setup subscribers
		pointCloudSub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("cloud", qos, 
						std::bind(&pointcloud_to_depthimage_node::pointCloud_callback, this, std::placeholders::_1));
		cameraInfoSub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", qos, 
						std::bind(&pointcloud_to_depthimage_node::cameraInfo_callback, this, std::placeholders::_1));
    	odomSub_ = this->create_subscription<nav_msgs::msg::Odometry>("synced_odom", qos, 
						std::bind(&pointcloud_to_depthimage_node::odom_callback, this, std::placeholders::_1));

        // Publisher
		depthImagePub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 10);// 16 bits unsigned in mm
		cameraInfoPub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("depth_image/camera_info", 10);
    }

private:
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
	rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointCloudSub_;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSub_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depthImagePub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoPub_;

	std::shared_ptr<tf2_ros::Buffer> tfBuffer_;
	std::shared_ptr<tf2_ros::TransformListener> tfListener_;
	double waitForTransform_ = 0.1;
	std::string fixedFrameId_ = "odom";
	std::string camFrameId_ = "cam0";

	tf2::Transform base_to_cam_;
	bool base_to_cam_received_= false;

	sensor_msgs::msg::CameraInfo cameraInfoMsg;
	bool camera_info_received_ = false;
	double fx, fy, cx, cy, k1, k2, p1, p2, k3;
	uint16_t width, height;

	pcl::PCLPointCloud2 cloud_aggregated;
	std::string cloud_frame_id;

	sensor_msgs::msg::PointCloud2 cloud_msg;

    void pointCloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointCloud2Msg){
		// if (!pointCloud2Msg_aggregated.data.empty()) {
		// 	// Append new point cloud to the aggregated point cloud
		// 	pcl::PCLPointCloud2 cloud_new;
		// 	pcl_conversions::toPCL(*pointCloud2Msg, *cloud_new);

		// 	pcl::PCLPointCloud2 cloud_result;
		// 	pcl::concatenatePointCloud(*cloud_aggregated, *cloud_new, *cloud_result);
		// 	*cloud_aggregated = *cloud_result;
		// } else {
		// 	// First point cloud, just copy
		// 	pcl_conversions::toPCL(*pointCloud2Msg, *cloud_aggregated);
		// 	cloud_frame_id = pointCloud2Msg->header.frame_id;
		// }
		pcl_conversions::toPCL(*pointCloud2Msg, cloud_aggregated);
		cloud_frame_id = pointCloud2Msg->header.frame_id;

		cloud_msg = *pointCloud2Msg;
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

			RCLCPP_INFO(this->get_logger(), "Received camera info.");
			camera_info_received_ = true;
		}
		cameraInfoMsg = *cam_info;
	}

	void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odomMsg) {
		
		// If we haven’t retrieved base_link->camera yet, try once
		if (!base_to_cam_received_)
		{
			try
			{
				auto tf_msg = tfBuffer_->lookupTransform(odomMsg->child_frame_id, camFrameId_, tf2::TimePointZero);

				tf2::fromMsg(tf_msg.transform, base_to_cam_);
				base_to_cam_received_ = true;

				RCLCPP_INFO(this->get_logger(), "Retrieved base_link->camera_link transform.");
			}
			catch (const tf2::TransformException & ex)
			{
				RCLCPP_WARN_THROTTLE(
				this->get_logger(), *this->get_clock(), 2000,
				"Waiting for base_link->camera_link TF: %s", ex.what());
				return;
			}
		}

		if(depthImagePub_->get_subscription_count() > 0)
		{
			// Build odom->base_link from odometry
			tf2::Transform odom_to_base;
			tf2::fromMsg(odomMsg->pose.pose, odom_to_base);

			// Compose odom->camera = odom->base_link * base_link->camera
			tf2::Transform odom_to_cam = odom_to_base * base_to_cam_;
			odom_to_cam = odom_to_cam.inverse();

			auto depthImage_msg = cloudToDepthImage(cloud_msg, odom_to_cam);
			depthImage_msg->header = odomMsg->header;
			depthImage_msg->header.frame_id = camFrameId_;
			depthImagePub_->publish(*depthImage_msg);
			if(cameraInfoPub_->get_subscription_count())
			{
				cameraInfoMsg.header = depthImage_msg->header;
				cameraInfoPub_->publish(cameraInfoMsg);
			}
		}
	}

	sensor_msgs::msg::Image::SharedPtr cloudToDepthImage(
		const sensor_msgs::msg::PointCloud2 & cloud_in_odom,
		const tf2::Transform & odom_to_cam)
	{
		// --- Initialize depth image (float32, meters) ---
		// cv::Mat depth_image(height, width, CV_32FC1, std::numeric_limits<float>::quiet_NaN());
		cv::Mat depth_image = cv::Mat::zeros(height, width, CV_16UC1); // depth in mm

		// --- Iterate through point cloud ---
		sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_in_odom, "x");
		sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud_in_odom, "y");
		sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud_in_odom, "z");

		for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
		{
			tf2::Vector3 p_odom(*iter_x, *iter_y, *iter_z);
			tf2::Vector3 p_cam = odom_to_cam * p_odom;

			if (p_cam.z() > 0.0)
			{
				double x = p_cam.x() / p_cam.z();
				double y = p_cam.y() / p_cam.z();
				double r2 = x*x + y*y;
				double x_distorted = x*(1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
				double y_distorted = y*(1.0 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) + p1*(r2 + 2.0*y*y) + 2.0*p2*x*y;

				uint16_t u = static_cast<uint16_t>(fx * x_distorted + cx);
				uint16_t v = static_cast<uint16_t>(fy * y_distorted + cy);

				// uint16_t u = static_cast<uint16_t>(fx * (p_cam.x() / p_cam.z()) + cx);
				// uint16_t v = static_cast<uint16_t>(fy * (p_cam.y() / p_cam.z()) + cy);

				if (u >= 0 && u < width && v >= 0 && v < height)
				{
					// float &depth = depth_image.at<float>(v, u);
					// // Keep the closest point if multiple map to same pixel
					// if (std::isnan(depth) || p_cam.z() < depth){
					// 	depth = static_cast<float>(p_cam.z()*1000);
					// }
					uint16_t &depth = depth_image.at<uint16_t>(v, u);
					if (depth == 0 || p_cam.z()*1000 < depth){
						depth = static_cast<uint16_t>(p_cam.z()*1000); //mm
					}
				}
			}
		}

		// --- Convert cv::Mat -> sensor_msgs::Image ---
		std_msgs::msg::Header header;
		cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::TYPE_16UC1, depth_image);
		return cv_img.toImageMsg();
	}
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pointcloud_to_depthimage_node>());
    rclcpp::shutdown();
    return 0;
}
