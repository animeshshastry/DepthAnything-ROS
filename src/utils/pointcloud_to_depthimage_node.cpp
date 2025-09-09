#include <rclcpp/rclcpp.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/synchronizer.h>
// #include <sensor_msgs/image_encodings.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rtabmap_conversions/MsgConversion.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util2d.h>
#include <rtabmap/utilite/ULogger.h>

class pointcloud_to_depthimage_node : public rclcpp::Node {
public:
    pointcloud_to_depthimage_node() : Node("pointcloud_to_depthimage") {

        tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_);

		int topicQueueSize = 1;
		int syncQueueSize = 10;
		bool approx = true;

        approx = this->declare_parameter("approx", approx);
        fixedFrameId_ = this->declare_parameter("fixed_frame_id", fixedFrameId_);
		camFrameId_ = this->declare_parameter("cam_frame_id", camFrameId_);
        waitForTransform_ = this->declare_parameter("wait_for_transform", waitForTransform_);
        syncQueueSize = this->declare_parameter("sync_queue_size", syncQueueSize);
        topicQueueSize = this->declare_parameter("topic_queue_size", topicQueueSize);
        
        // QoS profile: Reliable + Keep last
        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        qos_profile.depth = topicQueueSize;

        // Setup subscribers
        pointCloudSub_.subscribe(this, "cloud", qos_profile);
        cameraInfoSub_.subscribe(this, "camera_info", qos_profile);
        if(approx)
        {
            approxSync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(syncQueueSize), pointCloudSub_, cameraInfoSub_);
            approxSync_->registerCallback(std::bind(&pointcloud_to_depthimage_node::callback, this, std::placeholders::_1, std::placeholders::_2));
        }
        else
        {
            fixedFrameId_.clear();
			exactSync_ = std::make_shared<message_filters::Synchronizer<ExactSyncPolicy>>(ExactSyncPolicy(syncQueueSize), pointCloudSub_, cameraInfoSub_);
			exactSync_->registerCallback(std::bind(&pointcloud_to_depthimage_node::callback, this, std::placeholders::_1, std::placeholders::_2));
        };

        // Publisher
		depthImagePub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_image", 1);// 16 bits unsigned in mm
		cameraInfoPub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("depth_image/camera_info", 1);
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::CameraInfo> ApproxSyncPolicy;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::CameraInfo> ExactSyncPolicy;

    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> approxSync_;
    std::shared_ptr<message_filters::Synchronizer<ExactSyncPolicy>> exactSync_;
    
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> pointCloudSub_;
	message_filters::Subscriber<sensor_msgs::msg::CameraInfo> cameraInfoSub_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depthImagePub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoPub_;

	std::shared_ptr<tf2_ros::Buffer> tfBuffer_;
	std::shared_ptr<tf2_ros::TransformListener> tfListener_;
	double waitForTransform_ = 0.1;
	std::string fixedFrameId_ = "odom";
	std::string camFrameId_ = "cam0";

    void callback(
		const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointCloud2Msg,
		const sensor_msgs::msg::CameraInfo::ConstSharedPtr cameraInfoMsg) {

		if(depthImagePub_->get_subscription_count() > 0)
		{
			double cloudStamp = rtabmap_conversions::timestampFromROS(pointCloud2Msg->header.stamp);
			double infoStamp = rtabmap_conversions::timestampFromROS(cameraInfoMsg->header.stamp);

			rtabmap::Transform cloudDisplacement = rtabmap::Transform::getIdentity();
			if(!fixedFrameId_.empty())
			{
				// approx sync
				cloudDisplacement = rtabmap_conversions::getMovingTransform(
						pointCloud2Msg->header.frame_id,
						fixedFrameId_,
						pointCloud2Msg->header.stamp,
						cameraInfoMsg->header.stamp,
						*tfBuffer_,
						waitForTransform_);
			}

			if(cloudDisplacement.isNull())
			{
				RCLCPP_ERROR(this->get_logger(), "Could not find transform between %s and %s, accordingly to %s, aborting!",
					pointCloud2Msg->header.frame_id.c_str(), 
					camFrameId_.c_str(),
					fixedFrameId_.c_str());
				return;
			}

			rtabmap::Transform cloudToCamera = rtabmap_conversions::getTransform(
					pointCloud2Msg->header.frame_id,
					camFrameId_,
					cameraInfoMsg->header.stamp,
					*tfBuffer_,
					waitForTransform_);

			if(cloudToCamera.isNull())
			{
				RCLCPP_ERROR(this->get_logger(), "Could not find transform between %s and %s, aborting!",
					pointCloud2Msg->header.frame_id.c_str(), 
					camFrameId_.c_str());
				return;
			}
			rtabmap::Transform localTransform = cloudDisplacement*cloudToCamera;

			rtabmap::CameraModel model = rtabmap_conversions::cameraModelFromROS(*cameraInfoMsg, localTransform);
			sensor_msgs::msg::CameraInfo cameraInfoMsgOut = *cameraInfoMsg;

			UASSERT_MSG(pointCloud2Msg->data.size() == pointCloud2Msg->row_step*pointCloud2Msg->height,
							uFormat("data=%d row_step=%d height=%d", pointCloud2Msg->data.size(), pointCloud2Msg->row_step, pointCloud2Msg->height).c_str());

			pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
			pcl_conversions::toPCL(*pointCloud2Msg, *cloud);

			cv_bridge::CvImage depthImage;

			if(cloud->data.empty())
			{
				RCLCPP_WARN(this->get_logger(), "Received an empty cloud on topic \"%s\"! A depth image with all zeros is returned.", pointCloudSub_.getTopic().c_str());
				depthImage.image = cv::Mat::zeros(model.imageSize(), CV_32FC1);
			}
			else
			{
				depthImage.image = rtabmap::util3d::projectCloudToCamera(model.imageSize(), model.K(), cloud, model.localTransform());
			}

			depthImage.header = cameraInfoMsg->header;
			depthImage.header .frame_id = camFrameId_;

			if(depthImagePub_->get_subscription_count())
			{
				depthImage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
				depthImage.image = rtabmap::util2d::cvtDepthFromFloat(depthImage.image);
				// depthImage.image.convertTo(depthImage.image, CV_16UC1, 1000);
				sensor_msgs::msg::Image depthImage_msg;
				depthImage.toImageMsg(depthImage_msg);
				depthImagePub_->publish(depthImage_msg);
				if(cameraInfoPub_->get_subscription_count())
				{
					cameraInfoPub_->publish(cameraInfoMsgOut);
				}
			}

			if( cloudStamp != rtabmap_conversions::timestampFromROS(pointCloud2Msg->header.stamp) ||
				infoStamp != rtabmap_conversions::timestampFromROS(cameraInfoMsg->header.stamp))
			{
				RCLCPP_ERROR(this->get_logger(), "Input stamps changed between the beginning and the end of the callback! Make "
						"sure the node publishing the topics doesn't override the same data after publishing them. A "
						"solution is to use this node within another nodelet manager. Stamps: "
						"cloud=%f->%f info=%f->%f",
						cloudStamp, rtabmap_conversions::timestampFromROS(pointCloud2Msg->header.stamp),
						infoStamp, rtabmap_conversions::timestampFromROS(cameraInfoMsg->header.stamp));
			}
		}
	}
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pointcloud_to_depthimage_node>());
    rclcpp::shutdown();
    return 0;
}
