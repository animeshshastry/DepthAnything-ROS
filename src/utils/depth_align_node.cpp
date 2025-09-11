#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/synchronizer.h>

class DepthAlignNode : public rclcpp::Node {
public:
    DepthAlignNode() : Node("depth_align_node") {

        int syncQueueSize = 100;
        syncQueueSize = this->declare_parameter("sync_queue_size", syncQueueSize);

        // Setup subscribers
        sparse_sub_.subscribe(this, "sparse_topic", rmw_qos_profile_services_default);
        dense_sub_.subscribe(this, "dense_topic", rmw_qos_profile_services_default);

        // Synchronizer
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(syncQueueSize), sparse_sub_, dense_sub_));
        sync_->registerCallback(std::bind(&DepthAlignNode::callback, this, std::placeholders::_1, std::placeholders::_2));

        // Publisher
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("output_topic", 10);
    }

private:
    typedef message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::Image> sparse_sub_, dense_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

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
        // dense.convertTo(dense, CV_16U);      // back to 16-bit unsigned

        std::vector<float> xs, ys;
        // xs.reserve(sparse.rows * sparse.cols / 10); // heuristic
        // ys.reserve(xs.size());

        for (uint16_t v = 0; v < sparse.rows; v++) {
            for (uint16_t u = 0; u < sparse.cols; u++) {
                float d_sparse = sparse.at<float>(v, u);
                float d_pred   = dense.at<float>(v, u);
                if (d_sparse < 20.0f && d_sparse > 0.0f && d_pred > 0.0f) { // consider valid sparse points only
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

        // // Mean values
        // double mean_x = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
        // double mean_y = std::accumulate(ys.begin(), ys.end(), 0.0) / ys.size();
        // // Linear regression
        // double num = 0.0, den = 0.0;
        // for (size_t i = 0; i < xs.size(); i++) {
        //     num += (xs[i] - mean_x) * (ys[i] - mean_y);
        //     den += (xs[i] - mean_x) * (xs[i] - mean_x);
        // }
        // double s = (den > 1e-6) ? num / den : 1.0;
        // double b = mean_y - s * mean_x;

        // Mean scaling
        double sum = 0.0;
        for (size_t i = 0; i < xs.size(); i++) {
            sum += ys[i]/xs[i];
        }
        double s = sum/xs.size();
        double b = 0.0;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Scale: %.4f, Bias: %.4f (using %zu points)",
                             s, b, xs.size());

        // Apply correction
        cv::Mat corrected = s * dense + b;

        corrected.convertTo(corrected, CV_32FC1);

        // Publish
        auto out_msg = cv_bridge::CvImage(dense_msg->header, "32FC1", corrected).toImageMsg();
        pub_->publish(*out_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthAlignNode>());
    rclcpp::shutdown();
    return 0;
}
