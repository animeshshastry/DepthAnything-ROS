#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class DepthAlignNode : public rclcpp::Node {
public:
    DepthAlignNode() : Node("depth_align_node") {

        // QoS profile: Reliable + Keep last
        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        qos_profile.depth = 10;

        // Setup subscribers
        sparse_sub_.subscribe(this, "sparse_topic", qos_profile);
        dense_sub_.subscribe(this, "dense_topic", qos_profile);

        // Synchronizer
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), sparse_sub_, dense_sub_));
        sync_->registerCallback(std::bind(
            &DepthAlignNode::callback, this,
            std::placeholders::_1, std::placeholders::_2));

        // Publisher
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("output_topic", 1);
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::Image> sparse_sub_, dense_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& sparse_msg,
                  const sensor_msgs::msg::Image::ConstSharedPtr& dense_msg) {
        // Convert to OpenCV
        cv::Mat sparse = cv_bridge::toCvShare(sparse_msg, "32FC1")->image;
        cv::Mat disp8  = cv_bridge::toCvShare(dense_msg, "mono8")->image;
        cv::Mat disp;
        disp8.convertTo(disp, CV_32F);

        // Convert disparity -> depth (predicted)
        cv::Mat dense = 1.0f / (disp + 1e-6f);

        std::vector<float> xs, ys;
        // xs.reserve(sparse.rows * sparse.cols / 10); // heuristic
        // ys.reserve(xs.size());

        for (int v = 0; v < sparse.rows; v++) {
            for (int u = 0; u < sparse.cols; u++) {
                float d_sparse = sparse.at<float>(v, u);
                float d_pred   = dense.at<float>(v, u);
                if (d_sparse > 0.0f && d_pred > 0.0f &&
                    d_sparse < 20.0f && !std::isnan(d_sparse)) {
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
            sum += ys[i] / xs[i];
        }
        double s = sum/xs.size();
        double b = 0.0;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Scale: %.4f, Bias: %.4f (using %zu points)",
                             s, b, xs.size());

        // Apply correction
        cv::Mat corrected = s * dense + b;
        // Set corrected values > 10.0 to NaN
        // corrected.setTo(std::numeric_limits<float>::quiet_NaN(), corrected > 20.0f);

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
