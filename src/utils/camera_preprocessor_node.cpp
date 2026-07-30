#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

using std::placeholders::_1;

class CameraPreprocessor : public rclcpp::Node
{
public:
  CameraPreprocessor() : Node("camera_preprocessor")
  {
    calib_file_    = declare_parameter<std::string>("calibration_file", "");
    camera_key_    = declare_parameter<std::string>("camera_name", "cam0");
    image_topic_   = declare_parameter<std::string>("image_topic", "image_raw");
    info_topic_    = declare_parameter<std::string>("camera_info_topic", "camera_info");
    frame_id_override_ = declare_parameter<std::string>("frame_id", "");
    alpha_or_balance = declare_parameter<double>("alpha", 1.0);

    // ---- New: YUV conversion parameters ----
    convert_yuv_ = declare_parameter<bool>("convert_yuv422_to_bgr", false);
    yuv_variant_ = declare_parameter<std::string>("yuv_variant", "uyvy"); // "uyvy" or "yuyv"
    image_yuv_topic_ = declare_parameter<std::string>("image_yuv_topic", "image_raw_bgr");

	rectify_in_node_ = declare_parameter<bool>("rectify", false);
	image_rect_topic_ = declare_parameter<std::string>("image_rect_topic", "image_rect");

	  // --- New: crop bottom row(s) that contain sensor metadata, not scene data ---
  	crop_bottom_rows_ = declare_parameter<int>("crop_bottom_rows", 0);

    if (calib_file_.empty()) {
      RCLCPP_FATAL(get_logger(), "Parameter 'calibration_file' is required.");
      throw std::runtime_error("Missing calibration_file parameter");
    }

    loadKalibrYaml();

    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(info_topic_, qos);

    if (convert_yuv_) {
      	image_yuv_pub_ = create_publisher<sensor_msgs::msg::Image>(image_yuv_topic_, qos);
    }
	if (rectify_in_node_) {
		image_rect_pub_ = create_publisher<sensor_msgs::msg::Image>(image_rect_topic_, qos);
	}

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, qos,
      std::bind(&CameraPreprocessor::imageCallback, this, _1));

	  
	RCLCPP_INFO(get_logger(), "Loaded calibration for '%s' from %s", camera_key_.c_str(), calib_file_.c_str());
	std::ostringstream log_msg;
	log_msg << "Subscribing: " << image_sub_->get_topic_name() << "  ->  CameraInfo: " << info_pub_->get_topic_name();
	if (convert_yuv_) log_msg << "  ->  Image(bgr8): " << image_yuv_pub_->get_topic_name();
	if (rectify_in_node_) log_msg << "  ->  ImageRect: " << image_rect_pub_->get_topic_name();
	RCLCPP_INFO(get_logger(), "%s", log_msg.str().c_str());

	// ---- Mask publisher: latched (TRANSIENT_LOCAL), published once ----
	rclcpp::QoS mask_qos(1);
	mask_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
	mask_qos.durability(rclcpp::DurabilityPolicy::TransientLocal);
	mask_pub_ = create_publisher<sensor_msgs::msg::Image>("image_rect_mask", mask_qos);
	publishMask();
  }

private:
  void loadKalibrYaml()
  {
    // ... unchanged from your existing implementation ...
    YAML::Node root = YAML::LoadFile(calib_file_);
    if (!root[camera_key_]) {
      RCLCPP_FATAL(get_logger(), "Key '%s' not found in %s", camera_key_.c_str(), calib_file_.c_str());
      throw std::runtime_error("Camera key not found in calibration file");
    }
    YAML::Node cam = root[camera_key_];

    auto res = cam["resolution"].as<std::vector<int>>();
    width_  = res.at(0);
    height_ = res.at(1);

    auto intr = cam["intrinsics"].as<std::vector<double>>();
    double fu = intr.at(0), fv = intr.at(1), cu = intr.at(2), cy = intr.at(3);

    std::string kalibr_dist_model = cam["distortion_model"] ?
      cam["distortion_model"].as<std::string>() : "radtan";
    auto dist = cam["distortion_coeffs"].as<std::vector<double>>();

    if (kalibr_dist_model == "radtan" || kalibr_dist_model == "radial-tangential") {
      distortion_model_ = "plumb_bob";
      D_ = {dist.at(0), dist.at(1), dist.at(2), dist.at(3), 0.0};
    } else if (kalibr_dist_model == "equidistant" || kalibr_dist_model == "kannala-brandt4") {
      distortion_model_ = "equidistant";
      D_ = {dist.at(0), dist.at(1), dist.at(2), dist.at(3)};
    } else {
      RCLCPP_WARN(get_logger(),
        "Unrecognized distortion_model '%s', defaulting to plumb_bob with raw coeffs",
        kalibr_dist_model.c_str());
      distortion_model_ = "plumb_bob";
      D_ = dist;
      D_.resize(5, 0.0);
    }

    K_ = {fu, 0.0, cu,
          0.0, fv, cy,
          0.0, 0.0, 1.0};

    R_ = {1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0};

    cv::Mat cvK = (cv::Mat_<double>(3,3) <<
      fu, 0.0, cu,
      0.0, fv, cy,
      0.0, 0.0, 1.0);

    cv::Mat cvD;
    if (distortion_model_ == "equidistant") {
      cvD = (cv::Mat_<double>(4,1) << D_[0], D_[1], D_[2], D_[3]);
    } else {
      cvD = (cv::Mat_<double>(5,1) << D_[0], D_[1], D_[2], D_[3], D_[4]);
    }

    cv::Size image_size(width_, height_);
    cv::Mat new_K;

    if (distortion_model_ == "equidistant") {
      cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        cvK, cvD, image_size, cv::Mat::eye(3, 3, CV_64F), new_K,
        alpha_or_balance, image_size);
    } else {
      new_K = cv::getOptimalNewCameraMatrix(
        cvK, cvD, image_size, alpha_or_balance, image_size,
        nullptr, false);
    }

    double nfu = new_K.at<double>(0,0);
    double nfv = new_K.at<double>(1,1);
    double ncu = new_K.at<double>(0,2);
    double ncv = new_K.at<double>(1,2);

    P_ = {nfu, 0.0, ncu, 0.0,
          0.0, nfv, ncv, 0.0,
          0.0, 0.0, 1.0, 0.0};

	  // ---- Build undistortion maps + validity mask, matching what rectify_node will compute ----
	cv::Mat cvR = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat cvP = (cv::Mat_<double>(3,3) <<
		P_[0], P_[1], P_[2],
		P_[4], P_[5], P_[6],
		P_[8], P_[9], P_[10]);

	cv::Size mask_image_size(width_, height_ - crop_bottom_rows_); // match the cropped size you actually publish

	if (distortion_model_ == "equidistant") {
		cv::fisheye::initUndistortRectifyMap(
		cvK, cvD, cvR, cvP, mask_image_size, CV_32FC1, map1, map2);
	} else {
		cv::initUndistortRectifyMap(
		cvK, cvD, cvR, cvP, mask_image_size, CV_32FC1, map1, map2);
	}

	cv::Mat mask_src(mask_image_size, CV_8UC1, cv::Scalar(255));
	cv::remap(mask_src, valid_mask_, map1, map2, cv::INTER_NEAREST,
				cv::BORDER_CONSTANT, cv::Scalar(0));
	// valid_mask_: 255 where the rectified pixel has real source data, 0 where it doesn't
  }

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
	int effective_height = static_cast<int>(msg->height) - crop_bottom_rows_;

	if (effective_height <= 0 || static_cast<int>(msg->width) != width_) {
		RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
		"Image size mismatch: got %ux%u, expected width=%d after cropping %d rows",
		msg->width, msg->height, width_, crop_bottom_rows_);
		return;
	}

	// ---- CameraInfo: K/P/D unchanged (bottom crop doesn't shift cx, cy, or focal length) ----
	sensor_msgs::msg::CameraInfo info;
	info.header.stamp = msg->header.stamp;
	info.header.frame_id = frame_id_override_.empty() ? msg->header.frame_id : frame_id_override_;

	info.width  = width_;
	info.height = effective_height;   // <-- cropped height, matches what we actually publish below
	info.distortion_model = distortion_model_;
	info.d = D_;
	std::copy(K_.begin(), K_.end(), info.k.begin());
	std::copy(R_.begin(), R_.end(), info.r.begin());
	std::copy(P_.begin(), P_.end(), info.p.begin());

	info.binning_x = 0;
	info.binning_y = 0;
	info.roi.x_offset = 0;
	info.roi.y_offset = 0;
	info.roi.height = 0;
	info.roi.width = 0;
	info.roi.do_rectify = false;

	info_pub_->publish(info);

	cv::Mat yuv(effective_height, msg->width, CV_8UC2, const_cast<uint8_t*>(msg->data.data()), msg->step);
	cv::Mat bgr;

	// ---- Convert YUV422 -> BGR8, cropping the metadata row(s) first ----
	if (convert_yuv_) {
		if (yuv_variant_ == "yuyv") {
			cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_YUYV);
		} else {
			cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_UYVY);
		}

		auto out = std::make_unique<sensor_msgs::msg::Image>();
		out->header = msg->header;
		out->height = bgr.rows;
		out->width  = bgr.cols;
		out->encoding = sensor_msgs::image_encodings::BGR8;
		out->is_bigendian = false;
		out->step = bgr.cols * 3;
		out->data.assign(bgr.data, bgr.data + bgr.total() * bgr.elemSize());

		image_yuv_pub_->publish(std::move(out));
	}

	// ---- Rectify, regardless of whether YUV conversion happened above ----
	if (rectify_in_node_) {
		cv::Mat src_for_rectify;
		std::string out_encoding;

		if (convert_yuv_) {
			// Reuse the BGR image we just converted above — no need to redo work.
			src_for_rectify = bgr;
			out_encoding = sensor_msgs::image_encodings::BGR8;
		} else {
			// No conversion happened: wrap msg->data directly, according to its actual encoding.
			// This assumes the incoming image is ALREADY in a rectifiable pixel format
			// (e.g. mono8, bgr8, rgb8) rather than packed YUV422.
			if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
			src_for_rectify = cv::Mat(effective_height, msg->width, CV_8UC1,
										const_cast<uint8_t*>(msg->data.data()), msg->step);
			out_encoding = sensor_msgs::image_encodings::MONO8;
			} else if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
			src_for_rectify = cv::Mat(effective_height, msg->width, CV_8UC3,
										const_cast<uint8_t*>(msg->data.data()), msg->step);
			out_encoding = sensor_msgs::image_encodings::BGR8;
			} else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
			src_for_rectify = cv::Mat(effective_height, msg->width, CV_8UC3,
										const_cast<uint8_t*>(msg->data.data()), msg->step);
			out_encoding = sensor_msgs::image_encodings::RGB8;
			} else {
			RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
				"rectify_in_node is true but convert_yuv422_to_bgr is false, and incoming "
				"encoding '%s' isn't a supported pass-through format (mono8/bgr8/rgb8). "
				"Enable convert_yuv422_to_bgr if the source is packed YUV422.",
				msg->encoding.c_str());
			return;
			}
		}

		cv::Mat rect;
		cv::Scalar border_fill = (src_for_rectify.channels() == 1) ? cv::Scalar(0) : cv::Scalar(0, 0, 0);
		
		cv::remap(src_for_rectify, rect, map1, map2, cv::INTER_LINEAR,
					cv::BORDER_CONSTANT, border_fill);

		auto out = std::make_unique<sensor_msgs::msg::Image>();
		out->header = msg->header;
		out->height = rect.rows;
		out->width  = rect.cols;
		out->encoding = out_encoding;
		out->is_bigendian = false;
		out->step = rect.cols * rect.channels();
		out->data.assign(rect.data, rect.data + rect.total() * rect.elemSize());

		image_rect_pub_->publish(std::move(out));
	}
  }

	void publishMask()
	{
		auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
		mask_msg->header.stamp = now();
		mask_msg->header.frame_id = frame_id_override_.empty() ? "" : frame_id_override_;
		mask_msg->height = valid_mask_.rows;
		mask_msg->width  = valid_mask_.cols;
		mask_msg->encoding = sensor_msgs::image_encodings::MONO8;
		mask_msg->is_bigendian = false;
		mask_msg->step = valid_mask_.cols;
		mask_msg->data.assign(valid_mask_.data, valid_mask_.data + valid_mask_.total());

		mask_pub_->publish(std::move(mask_msg));
		RCLCPP_INFO(get_logger(), "Published rectified-image validity mask on %s", mask_pub_->get_topic_name());
	}

	// params
	std::string calib_file_, camera_key_, image_topic_, info_topic_, frame_id_override_;
	double alpha_or_balance;
	int crop_bottom_rows_ = 0;
	bool convert_yuv_, rectify_in_node_ = false;
	std::string yuv_variant_, image_yuv_topic_, image_rect_topic_;
	cv::Mat valid_mask_;
	cv::Mat map1, map2;

	// calibration
	int width_ = 0, height_ = 0;
	std::string distortion_model_;
	std::vector<double> D_;
	std::array<double, 9> K_{};
	std::array<double, 9> R_{};
	std::array<double, 12> P_{};

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

	rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_yuv_pub_;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_rect_pub_;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<CameraPreprocessor>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("camera_preprocessor"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}