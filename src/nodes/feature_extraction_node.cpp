// Feature extraction node: port of bruce_slam feature_extraction.py.
// CFAR target detection on the polar sonar image (GPU with CPU fallback),
// polar -> Cartesian remap, voxel downsampling and radius outlier removal,
// publishing a PointCloud2 stamped with the source ping time.
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "sonar_slam_cpp/cfar.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/gpu.hpp"
#include "sonar_slam_cpp/interp.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"

namespace sonar_slam {

class FeatureExtractionNode : public SlamNodeBase
{
public:
  FeatureExtractionNode() : SlamNodeBase("feature_extraction_node")
  {
    // CFAR parameters
    const int ntc = get_int("CFAR/Ntc");
    const int ngc = get_int("CFAR/Ngc");
    const double pfa = get_double("CFAR/Pfa");
    const int rank = get_int("CFAR/rank");
    alg_ = CFAR::alg_from_string(get_string("CFAR/alg", "SOCA"));
    threshold_ = get_int("filter/threshold");

    // point cloud filtering parameters
    resolution_ = get_double("filter/resolution");
    outlier_filter_radius_ = get_double("filter/radius");
    outlier_filter_min_points_ = get_int("filter/min_points");
    skip_ = get_int("filter/skip");

    compressed_images_ = get_bool("compressed_images");

    detector_ = std::make_unique<CFAR>(ntc, ngc, pfa, rank);

    // sonar driver (pluggable adapter + configurable topic); blank driver
    // auto-selects the Oculus adapter matching compressed_images
    std::string driver = get_string("sonar/driver", "");
    if (driver.empty())
      driver = compressed_images_ ? "oculus_compressed" : "oculus_uncompressed";
    std::string topic = get_string("sonar/topic", "");
    if (topic.empty()) {
      topic = default_sonar_topic(driver);
      if (topic.empty())
        throw std::runtime_error(
          "sonar/topic must be set explicitly for sonar driver '" + driver + "'");
    }

    sonar_sub_ = subscribe_sonar(
      this, driver, topic, rclcpp::SensorDataQoS(),
      [this](const SonarPing& ping) { callback(ping); });

    feature_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SONAR_FEATURE_TOPIC, 10);
    feature_img_pub_ = create_publisher<sensor_msgs::msg::Image>(
      SONAR_FEATURE_IMG_TOPIC, 10);

    RCLCPP_INFO(get_logger(), "Feature extraction node initialized (GPU: %s)",
                gpu::available() ? "on" : "off, CPU fallback");
  }

private:
  // build (or refresh) the polar -> Cartesian maps when geometry changes
  void generate_map_xy(const SonarPing& ping)
  {
    const double res = ping.range_resolution;
    const double range_min = ping.range_min;
    // Polar row 0 is at range_min, so the Cartesian fan spans
    // [0, range_min + num_ranges*res]. Add ceil(range_min/res) near-field output
    // rows (kept at cell size `res`) so a nonzero-min-range multibeam is placed
    // at its true range. For an Oculus (range_min == 0) extra_rows is 0 and this
    // reduces byte-for-byte to the original grid.
    const int extra_rows =
      res > 0.0 ? static_cast<int>(std::ceil(range_min / res)) : 0;
    const int rows = ping.num_ranges + extra_rows;
    const double height = rows * res;
    const double width =
      std::sin((ping.bearings.back() - ping.bearings.front()) / 2.0) * height * 2.0;
    const int cols = static_cast<int>(std::ceil(width / res));

    if (res == res_ && height == height_ && rows == rows_ && width == width_ &&
        cols == cols_)
      return;
    res_ = res;
    height_ = height;
    rows_ = rows;
    width_ = width;
    cols_ = cols;
    // new maps -> new version so the GPU remap re-uploads its cached copy
    ++map_version_;

    // bearing -> beam column, linear like feature_extraction.py
    std::vector<double> bx(ping.bearings.begin(), ping.bearings.end());
    std::vector<double> by(ping.bearings.size());
    for (std::size_t i = 0; i < by.size(); ++i) by[i] = static_cast<double>(i);
    const Interp1d f_bearings(bx, by, Interp1d::LINEAR, -1.0);

    map_x_.create(rows, cols, CV_32FC1);
    map_y_.create(rows, cols, CV_32FC1);
    for (int r = 0; r < rows; ++r) {
      float* px = map_x_.ptr<float>(r);
      float* py = map_y_.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const double x = res * (rows - r);
        const double y = res * (-cols / 2.0 + c + 0.5);
        const double b = std::atan2(y, x);
        const double range = std::sqrt(x * x + y * y);
        // polar row = (range - range_min)/res; near-field cells (range <
        // range_min) map to a negative row and are sampled as border.
        py[c] = static_cast<float>((range - range_min) / res);
        px[c] = static_cast<float>(f_bearings(b));
      }
    }
  }

  void publish_features(const builtin_interfaces::msg::Time& stamp,
                        const Matrix& points)
  {
    // Publish honest planar base_link coordinates: [x, y, 0]. The python
    // original packed the lateral coordinate into z ([x, 0, y] — a leftover
    // of bruce's roll-pi rviz static, which this stack does not use); that
    // drew the fan rolled 90 deg in an ENU viewer. The SLAM node's
    // consumption is adjusted in lockstep (slam_node.cpp slam_callback),
    // so the graph math is unchanged.
    Matrix xyz(points.rows(), 3);
    xyz.col(0) = points.col(0);
    xyz.col(1) = points.col(1);
    xyz.col(2).setZero();

    sensor_msgs::msg::PointCloud2 msg = make_cloud_xyz(xyz);
    // the stamp of the source sonar image is CRITICAL to downstream sync
    msg.header.stamp = stamp;
    msg.header.frame_id = "base_link";
    feature_pub_->publish(msg);
  }

  // interp: 0 = nearest, 1 = bilinear. Use nearest for a binary detection
  // mask (correct for a 0/1 image and bit-exact between the CPU and GPU remap
  // paths); use bilinear for the grayscale visualization image.
  cv::Mat remap_u8(const cv::Mat& src, int interp) const
  {
    const cv::Mat cont = src.isContinuous() ? src : src.clone();
#ifdef SONAR_SLAM_WITH_CUDA
    // maps are device-resident across calls (re-uploaded only when
    // map_version_ changes); a failing GPU falls through to cv::remap
    if (gpu::available()) {
      cv::Mat dst(map_x_.rows, map_x_.cols, CV_8UC1);
      if (gpu::remap_u8_cuda(cont.ptr<std::uint8_t>(), cont.rows, cont.cols,
                             map_x_.ptr<float>(), map_y_.ptr<float>(),
                             map_x_.rows, map_x_.cols, interp, map_version_,
                             dst.ptr<std::uint8_t>()))
        return dst;
    }
#endif
    cv::Mat dst;
    cv::remap(cont, dst, map_x_, map_y_,
              interp == 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
    return dst;
  }

  void callback(const SonarPing& ping)
  {
    // cheap skip test; skipped frames still publish an empty (NaN) cloud so
    // the SLAM node's time synchronizer keeps advancing
    ++frame_count_;
    const int ping_id = ping.ping_id != 0 ? ping.ping_id : frame_count_;
    if (skip_ > 0 && ping_id % skip_ != 0) {
      Matrix nan(1, 2);
      nan.setConstant(std::numeric_limits<float>::quiet_NaN());
      publish_features(ping.stamp, nan);
      return;
    }

    // a malformed adapter frame (empty / zero-range image) must not throw out
    // of the callback and terminate the node
    if (ping.image.empty() || ping.num_ranges <= 0) {
      RCLCPP_WARN(get_logger(),
                  "Dropping sonar ping: empty or zero-range image");
      return;
    }

    // isolate the OpenCV / detection pipeline so a bad frame drops rather than
    // crashing the node
    try {
      const cv::Mat& img = ping.image;
      generate_map_xy(ping);

      // CFAR detection with the intensity gate folded into the detector, so
      // the CPU/GPU twins apply it identically and no separate host-side
      // compare + bitwise-and pass over the whole image is needed
      cv::Mat peaks = detector_->detect(img, alg_, static_cast<float>(threshold_));

      // visualization image (Cartesian, JET colormap like cv2.applyColorMap);
      // same GPU remap path as the mask, skipped when nobody is subscribed
      if (feature_img_pub_->get_subscription_count() > 0) {
        cv::Mat vis = remap_u8(img, /*interp=*/1);  // bilinear for grayscale
        cv::applyColorMap(vis, vis, cv::COLORMAP_JET);
        std_msgs::msg::Header header;
        header.stamp = ping.stamp;
        header.frame_id = "base_link";
        feature_img_pub_->publish(
          *cv_bridge::CvImage(header, "bgr8", vis).toImageMsg());
      }

      // to Cartesian — nearest-neighbour for the binary mask
      const cv::Mat cart_peaks = remap_u8(peaks, /*interp=*/0);

      std::vector<cv::Point> locs;
      cv::findNonZero(cart_peaks, locs);

      // image coordinates -> meters (feature_extraction.py lines 254-258)
      Matrix points(static_cast<long>(locs.size()), 2);
      for (std::size_t i = 0; i < locs.size(); ++i) {
        double x = locs[i].x - cols_ / 2.0;
        x = -1.0 * ((x / (cols_ / 2.0)) * (width_ / 2.0));
        const double y =
          -1.0 * (locs[i].y / static_cast<double>(rows_)) * height_ + height_;
        points(static_cast<long>(i), 0) = static_cast<float>(y);
        points(static_cast<long>(i), 1) = static_cast<float>(x);
      }

      if (points.rows() > 0 && resolution_ > 0)
        points = downsample(points, static_cast<float>(resolution_));

      if (outlier_filter_min_points_ > 1 && points.rows() > 0)
        points = remove_outlier(points, outlier_filter_radius_,
                                outlier_filter_min_points_);

      publish_features(ping.stamp, points);
    } catch (const cv::Exception& e) {
      RCLCPP_WARN(get_logger(), "Dropping sonar ping: OpenCV error: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Dropping sonar ping: %s", e.what());
    }
  }

  std::unique_ptr<CFAR> detector_;
  CFAR::Alg alg_ = CFAR::SOCA;
  int threshold_ = 0;
  double resolution_ = 0.5;
  double outlier_filter_radius_ = 1.0;
  int outlier_filter_min_points_ = 5;
  int skip_ = 5;
  bool compressed_images_ = true;
  int frame_count_ = 0;

  // remap state
  double res_ = 0.0, height_ = 0.0, width_ = 0.0;
  int rows_ = 0, cols_ = 0;
  int map_version_ = 0;
  cv::Mat map_x_, map_y_;

  rclcpp::SubscriptionBase::SharedPtr sonar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr feature_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_img_pub_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::FeatureExtractionNode>());
  rclcpp::shutdown();
  return 0;
}
