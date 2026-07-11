#include "sonar_slam_cpp/sensors.hpp"
#include "sonar_slam_cpp/common.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <bar30_depth/msg/depth.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <kvh_gyro/msg/gyro.hpp>
#include <marine_acoustic_msgs/msg/projected_sonar_image.hpp>
#include <rti_dvl/msg/dvl.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sonar_oculus/msg/oculus_ping.hpp>
#include <sonar_oculus/msg/oculus_ping_uncompressed.hpp>

#include <cmath>
#include <stdexcept>

namespace sonar_slam {

namespace {

double get_double_param(rclcpp::Node* node, const std::string& name,
                        double default_value)
{
  if (!node->has_parameter(name)) node->declare_parameter(name, default_value);
  return node->get_parameter(name).as_double();
}

[[noreturn]] void unknown_driver(const std::string& kind, const std::string& driver)
{
  throw std::invalid_argument("Unknown " + kind + " driver '" + driver + "'");
}

// ------------------------------------------------------------ sonar helpers
FireMsg oculus_fire(const sonar_oculus::msg::OculusFire& fm)
{
  FireMsg out;
  out.mode = fm.mode;
  out.gamma = fm.gamma;
  out.flags = fm.flags;
  out.range = fm.range;
  out.gain = fm.gain;
  out.speed_of_sound = fm.speed_of_sound;
  out.salinity = fm.salinity;
  return out;
}

template <typename PingMsg>
SonarPing oculus_ping(const PingMsg& msg, cv::Mat image)
{
  SonarPing ping;
  ping.stamp = msg.header.stamp;
  ping.image = std::move(image);
  // Oculus bearings are int16 hundredths of a degree -> radians
  ping.bearings.reserve(msg.bearings.size());
  for (const auto b : msg.bearings)
    ping.bearings.push_back(static_cast<float>(b / 100.0 * M_PI / 180.0));
  ping.range_resolution = msg.range_resolution;
  ping.num_ranges = msg.num_ranges;
  ping.ping_id = msg.ping_id;
  ping.fire = oculus_fire(msg.fire_msg);
  ping.part_number = msg.part_number;
  return ping;
}

}  // namespace

// --------------------------------------------------------------------- DVL
rclcpp::SubscriptionBase::SharedPtr subscribe_dvl(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const DvlReading&)> cb)
{
  if (driver == "rti_dvl") {
    return node->create_subscription<rti_dvl::msg::DVL>(
      topic, qos, [cb](const rti_dvl::msg::DVL& msg) {
        DvlReading r;
        r.stamp = msg.header.stamp;
        r.velocity << msg.velocity.x, msg.velocity.y, msg.velocity.z;
        r.altitude = msg.altitude;
        cb(r);
      });
  }
  if (driver == "twist_stamped") {
    return node->create_subscription<geometry_msgs::msg::TwistStamped>(
      topic, qos, [cb](const geometry_msgs::msg::TwistStamped& msg) {
        DvlReading r;
        r.stamp = msg.header.stamp;
        r.velocity << msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z;
        cb(r);
      });
  }
  if (driver == "twist_cov") {
    return node->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      topic, qos, [cb](const geometry_msgs::msg::TwistWithCovarianceStamped& msg) {
        DvlReading r;
        r.stamp = msg.header.stamp;
        r.velocity << msg.twist.twist.linear.x, msg.twist.twist.linear.y,
          msg.twist.twist.linear.z;
        cb(r);
      });
  }
  unknown_driver("dvl", driver);
}

// ------------------------------------------------------------------- depth
rclcpp::SubscriptionBase::SharedPtr subscribe_depth(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const DepthReading&)> cb)
{
  if (driver == "bar30") {
    return node->create_subscription<bar30_depth::msg::Depth>(
      topic, qos, [cb](const bar30_depth::msg::Depth& msg) {
        DepthReading r;
        r.stamp = msg.header.stamp;
        r.depth = msg.depth;
        cb(r);
      });
  }
  if (driver == "fluid_pressure") {
    const double rho = get_double_param(node, "depth.water_density", 1025.0);
    const double g = get_double_param(node, "depth.gravity", 9.80665);
    return node->create_subscription<sensor_msgs::msg::FluidPressure>(
      topic, qos, [cb, rho, g](const sensor_msgs::msg::FluidPressure& msg) {
        DepthReading r;
        r.stamp = msg.header.stamp;
        r.depth = msg.fluid_pressure / (rho * g);
        cb(r);
      });
  }
  unknown_driver("depth", driver);
}

// -------------------------------------------------------------------- gyro
rclcpp::SubscriptionBase::SharedPtr subscribe_gyro(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const GyroReading&)> cb)
{
  if (driver == "kvh_gyro") {
    return node->create_subscription<kvh_gyro::msg::Gyro>(
      topic, qos, [cb](const kvh_gyro::msg::Gyro& msg) {
        GyroReading r;
        r.stamp = msg.header.stamp;
        r.delta << msg.delta[0], msg.delta[1], msg.delta[2];
        cb(r);
      });
  }
  if (driver == "vector3_stamped") {
    return node->create_subscription<geometry_msgs::msg::Vector3Stamped>(
      topic, qos, [cb](const geometry_msgs::msg::Vector3Stamped& msg) {
        GyroReading r;
        r.stamp = msg.header.stamp;
        r.delta << msg.vector.x, msg.vector.y, msg.vector.z;
        cb(r);
      });
  }
  unknown_driver("gyro", driver);
}

// --------------------------------------------------------------------- IMU
bool imu_driver_is_legacy(const std::string& driver) { return driver == "vn100"; }

rclcpp::SubscriptionBase::SharedPtr subscribe_imu(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const ImuReading&)> cb)
{
  const bool legacy = imu_driver_is_legacy(driver);
  const bool enu =
    driver == "enu" || driver == "3dm_gx5" || driver == "microstrain";
  if (!legacy && !enu) unknown_driver("imu", driver);

  return node->create_subscription<sensor_msgs::msg::Imu>(
    topic, qos, [cb, enu](const sensor_msgs::msg::Imu& msg) {
      ImuReading r;
      r.stamp = msg.header.stamp;
      const auto& q = msg.orientation;
      r.orientation = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
      if (enu) {
        // world-side ENU -> NED and body-side FLU -> FRD (both involutive),
        // delivering the pipeline's z-down convention
        static const Eigen::Quaterniond R_NED_ENU(
          (Eigen::Matrix3d() << 0, 1, 0, 1, 0, 0, 0, 0, -1).finished());
        static const Eigen::Quaterniond R_FLU_FRD(
          (Eigen::Matrix3d() << 1, 0, 0, 0, -1, 0, 0, 0, -1).finished());
        r.orientation = R_NED_ENU * r.orientation * R_FLU_FRD;
      }
      cb(r);
    });
}

// ------------------------------------------------------------------- sonar
std::string default_sonar_topic(const std::string& driver)
{
  if (driver == "oculus_compressed") return SONAR_TOPIC;
  if (driver == "oculus_uncompressed") return SONAR_TOPIC_UNCOMPRESSED;
  return "";
}

rclcpp::SubscriptionBase::SharedPtr subscribe_sonar(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const SonarPing&)> cb)
{
  if (driver == "oculus_compressed") {
    return node->create_subscription<sonar_oculus::msg::OculusPing>(
      topic, qos, [node, cb](const sonar_oculus::msg::OculusPing& msg) {
        const cv::Mat buf(1, static_cast<int>(msg.ping.data.size()), CV_8UC1,
                          const_cast<unsigned char*>(msg.ping.data.data()));
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        // a truncated/corrupt message must not take the node down
        if (img.empty()) {
          RCLCPP_WARN(node->get_logger(),
                      "Dropping sonar ping %d: compressed image failed to decode",
                      msg.ping_id);
          return;
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        cb(oculus_ping(msg, std::move(img)));
      });
  }
  if (driver == "oculus_uncompressed") {
    return node->create_subscription<sonar_oculus::msg::OculusPingUncompressed>(
      topic, qos, [cb](const sonar_oculus::msg::OculusPingUncompressed& msg) {
        cv::Mat img = cv_bridge::toCvCopy(msg.ping, "")->image;
        if (img.type() != CV_8UC1) img.convertTo(img, CV_8UC1);
        cb(oculus_ping(msg, std::move(img)));
      });
  }
  if (driver == "image") {
    // a bare image carries no acoustic geometry -> read it from parameters
    const double range_resolution =
      get_double_param(node, "sonar.range_resolution", 0.0);
    if (range_resolution <= 0.0)
      throw std::invalid_argument(
        "sonar/range_resolution (m per range bin) must be set for the "
        "generic 'image' sonar driver");
    const double horizontal_fov =
      get_double_param(node, "sonar.horizontal_fov", 130.0 * M_PI / 180.0);
    auto count = std::make_shared<int>(0);
    return node->create_subscription<sensor_msgs::msg::Image>(
      topic, qos,
      [cb, range_resolution, horizontal_fov, count](const sensor_msgs::msg::Image& msg) {
        cv::Mat img = cv_bridge::toCvCopy(msg, "mono8")->image;
        SonarPing ping;
        ping.stamp = msg.header.stamp;
        ping.num_ranges = img.rows;
        const int num_beams = img.cols;
        ping.image = std::move(img);
        ping.bearings.resize(num_beams);
        for (int i = 0; i < num_beams; ++i)
          ping.bearings[i] = static_cast<float>(
            -horizontal_fov / 2.0 +
            horizontal_fov * (num_beams > 1 ? i / double(num_beams - 1) : 0.5));
        ping.range_resolution = range_resolution;
        ping.ping_id = ++(*count);
        cb(ping);
      });
  }
  if (driver == "projected_sonar") {
    auto count = std::make_shared<int>(0);
    return node->create_subscription<marine_acoustic_msgs::msg::ProjectedSonarImage>(
      topic, qos,
      [cb, count](const marine_acoustic_msgs::msg::ProjectedSonarImage& msg) {
        // 8-bit images only (this vehicle's config)
        if (msg.image.dtype != 0)  // SonarImageData::DTYPE_UINT8
          throw std::runtime_error(
            "projected_sonar adapter supports DTYPE_UINT8 only, got dtype " +
            std::to_string(msg.image.dtype));
        const int num_beams = static_cast<int>(msg.image.beam_count);
        const int num_ranges = static_cast<int>(msg.ranges.size());

        SonarPing ping;
        ping.stamp = msg.header.stamp;
        ping.image.create(num_ranges, num_beams, CV_8UC1);
        std::memcpy(ping.image.ptr(), msg.image.data.data(),
                    static_cast<std::size_t>(num_ranges) * num_beams);
        // bearing = atan2(-y, z), the driver's declared convention
        ping.bearings.reserve(msg.beam_directions.size());
        for (const auto& d : msg.beam_directions)
          ping.bearings.push_back(static_cast<float>(std::atan2(-d.y, d.z)));
        ping.num_ranges = num_ranges;
        ping.range_resolution =
          num_ranges > 1
            ? static_cast<double>(msg.ranges[1] - msg.ranges[0])
            : static_cast<double>(msg.ranges.empty() ? 0.0 : msg.ranges[0] * 2.0);
        const double sos = msg.ping_info.sound_speed;
        ping.fire.speed_of_sound = sos > 0.0 ? sos : 1500.0;
        ping.fire.range = msg.ranges.empty() ? 0.0 : msg.ranges.back();
        ping.ping_id = ++(*count);
        cb(ping);
      });
  }
  unknown_driver("sonar", driver);
}

}  // namespace sonar_slam
