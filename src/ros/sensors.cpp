#include "sonar_slam_cpp/sensors.hpp"
#include "sonar_slam_cpp/common.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Standard messages — always available. The Deep Trekker Revolution (and any
// payload wired to REP-standard / marine_acoustic_msgs drivers) runs entirely
// on these, so a build needs none of the legacy vendor packages below.
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <marine_acoustic_msgs/msg/dvl.hpp>
#include <marine_acoustic_msgs/msg/projected_sonar_image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64.hpp>

// Optional legacy BlueROV-payload vendor messages. Each package is found
// QUIET by CMake; when present it defines HAVE_<PKG> and the matching driver
// compiles in, otherwise selecting that driver reports a graceful runtime
// error (driver_not_compiled) instead of failing the whole build.
#ifdef HAVE_BAR30_DEPTH
#include <bar30_depth/msg/depth.hpp>
#endif
#ifdef HAVE_KVH_GYRO
#include <kvh_gyro/msg/gyro.hpp>
#endif
#ifdef HAVE_RTI_DVL
#include <rti_dvl/msg/dvl.hpp>
#endif
#ifdef HAVE_SONAR_OCULUS
#include <sonar_oculus/msg/oculus_ping.hpp>
#include <sonar_oculus/msg/oculus_ping_uncompressed.hpp>
#endif
#ifdef HAVE_DVL_MSGS
#include <dvl_msgs/msg/dvl.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace sonar_slam {

namespace {

// SlamNodeBase auto-declares YAML overrides before the constructor body runs,
// so a value written without a decimal point (e.g. `atmospheric_pressure:
// 101325`) arrives typed INTEGER; convert instead of letting as_double() throw
// out of the node constructor — the same leniency SlamNodeBase::get_double
// guarantees for every other parameter.
double get_double_param(rclcpp::Node* node, const std::string& name,
                        double default_value)
{
  if (!node->has_parameter(name)) node->declare_parameter(name, default_value);
  const auto p = node->get_parameter(name);
  return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
           ? static_cast<double>(p.as_int())
           : p.as_double();
}

bool get_bool_param(rclcpp::Node* node, const std::string& name,
                    bool default_value)
{
  if (!node->has_parameter(name)) node->declare_parameter(name, default_value);
  const auto p = node->get_parameter(name);
  return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
           ? p.as_int() != 0
           : p.as_bool();
}

[[noreturn]] void unknown_driver(const std::string& kind, const std::string& driver)
{
  throw std::invalid_argument("Unknown " + kind + " driver '" + driver + "'");
}

// Per-sensor stamp offset (<kind>.stamp_offset, seconds, default 0): the
// DVL/IMU/gyro/depth/sonar drivers stamp from INDEPENDENT device clocks with
// no normalization point anywhere in the stack, and every synchronizer pairs
// on those stamps. A constant cross-device offset delta biases every pairing
// by delta (position error v*delta, heading error omega*delta baked into
// EVERY keyframe) or, past the sync slop, silently starves the consumer.
// Measure the offset (e.g. median stamp difference of concurrently-arriving
// messages) and correct it here, at the single adapter choke point.
template <typename Reading>
std::function<void(const Reading&)> with_stamp_offset(
  rclcpp::Node* node, const std::string& kind,
  std::function<void(const Reading&)> cb)
{
  const double off = get_double_param(node, kind + ".stamp_offset", 0.0);
  if (off == 0.0) return cb;
  const int64_t off_ns = static_cast<int64_t>(off * 1e9);
  return [cb = std::move(cb), off_ns](const Reading& r_in) {
    Reading r = r_in;
    int64_t t = static_cast<int64_t>(r.stamp.sec) * 1000000000LL +
                static_cast<int64_t>(r.stamp.nanosec) + off_ns;
    if (t < 0) t = 0;
    r.stamp.sec = static_cast<int32_t>(t / 1000000000LL);
    r.stamp.nanosec = static_cast<uint32_t>(t % 1000000000LL);
    cb(r);
  };
}

// A vendor driver was requested but its message package was not on the build.
[[noreturn]] void driver_not_compiled(const std::string& kind,
                                      const std::string& driver,
                                      const std::string& pkg)
{
  throw DriverNotCompiledError(
    kind + " driver '" + driver + "' needs the '" + pkg +
    "' message package, which was not present at build time. Rebuild with '" +
    pkg + "' on AMENT_PREFIX_PATH, or select a standard-message driver for "
    "this sensor (see docs/PAYLOADS.md).");
}

// The polar->Cartesian map builder (feature_extraction_node generate_map_xy)
// interpolates bearing->column with Interp1d, which needs bearings ascending
// in column order, and derives the fan width from bearings.front()/back().
// Fan and multibeam sonars enumerate beams monotonically, but some drivers
// sweep starboard-to-port (descending bearing). Normalize by reversing the
// beam axis — bearings and image columns together — so column i still maps to
// bearings[i]. Already-ascending pings are left untouched.
void normalize_bearing_order(SonarPing& ping)
{
  if (ping.bearings.size() < 2) return;
  if (ping.bearings.back() >= ping.bearings.front()) return;  // ascending
  std::reverse(ping.bearings.begin(), ping.bearings.end());
  if (!ping.image.empty()) cv::flip(ping.image, ping.image, /*flipCode=*/1);
}

#ifdef HAVE_SONAR_OCULUS
// ------------------------------------------------------------ Oculus helpers
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
#endif  // HAVE_SONAR_OCULUS

}  // namespace

// --------------------------------------------------------------------- DVL
rclcpp::SubscriptionBase::SharedPtr subscribe_dvl(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const DvlReading&)> cb)
{
  // Single choke point for every DVL adapter (present and future): a
  // non-finite velocity must never reach the consumer — NaN passes the
  // downstream magnitude gates (all NaN comparisons are false) and
  // permanently corrupts dead reckoning.
  cb = [inner = std::move(cb)](const DvlReading& r) {
    if (!r.velocity.allFinite()) return;
    inner(r);
  };
  cb = with_stamp_offset(node, "dvl", std::move(cb));

  if (driver == "rti_dvl") {
#ifdef HAVE_RTI_DVL
    return node->create_subscription<rti_dvl::msg::DVL>(
      topic, qos, [cb](const rti_dvl::msg::DVL& msg) {
        DvlReading r;
        r.stamp = msg.header.stamp;
        r.velocity << msg.velocity.x, msg.velocity.y, msg.velocity.z;
        r.altitude = msg.altitude;
        cb(r);
      });
#else
    driver_not_compiled("dvl", driver, "rti_dvl");
#endif
  }
  // Standard marine_acoustic_msgs/Dvl — what the official Water Linked A50/A125
  // ROS 2 driver (waterlinked_dvl) publishes, so it is the Revolution's DVL
  // path. A DVL that has lost bottom lock still emits a (garbage) velocity, so
  // gate on the message's own validity signal unless dvl.require_valid=false.
  if (driver == "dvl_standard" || driver == "marine_dvl") {
    const bool require_valid = get_bool_param(node, "dvl.require_valid", true);
    return node->create_subscription<marine_acoustic_msgs::msg::Dvl>(
      topic, qos,
      [node, cb, require_valid](const marine_acoustic_msgs::msg::Dvl& msg) {
        if (require_valid) {
          // The spec defines beam_velocities_valid as "the optional per-beam
          // data is populated", but the official waterlinked_dvl driver
          // publishes its bottom-lock validity in it, so false is treated as
          // invalid by default. A spec-conforming derived-velocity-only
          // publisher leaves the flag false on every message — never drop it
          // silently; tell the operator the remedy (dvl.require_valid: false).
          // Water-track velocity is relative to the water column, not the
          // ground; using it would corrupt dead reckoning.
          const bool water_track =
            msg.velocity_mode == marine_acoustic_msgs::msg::Dvl::DVL_MODE_WATER;
          if (!msg.beam_velocities_valid || water_track) {
            RCLCPP_WARN_THROTTLE(
              node->get_logger(), *node->get_clock(), 10000,
              "dvl_standard: dropping DVL reading (%s). If your driver never "
              "sets beam_velocities_valid, set dvl.require_valid: false "
              "(docs/PAYLOADS.md).",
              water_track ? "water-track velocity"
                          : "beam_velocities_valid = false");
            return;
          }
        }
        DvlReading r;
        r.stamp = msg.header.stamp;
        r.velocity << msg.velocity.x, msg.velocity.y, msg.velocity.z;
        r.altitude = msg.altitude;
        cb(r);
      });
  }
  // Native Water Linked message (paagutie/ndahn dvl_msgs/DVL), used by the
  // community A50 drivers. Optional: needs the dvl_msgs package at build time.
  if (driver == "dvl_a50" || driver == "dvl_msgs") {
#ifdef HAVE_DVL_MSGS
    const bool require_valid = get_bool_param(node, "dvl.require_valid", true);
    return node->create_subscription<dvl_msgs::msg::DVL>(
      topic, qos, [cb, require_valid](const dvl_msgs::msg::DVL& msg) {
        if (require_valid && !msg.velocity_valid) return;
        DvlReading r;
        r.stamp = msg.header.stamp;
        r.velocity << msg.velocity.x, msg.velocity.y, msg.velocity.z;
        r.altitude = msg.altitude;
        cb(r);
      });
#else
    driver_not_compiled("dvl", driver, "dvl_msgs");
#endif
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
  // same non-finite choke as the DVL adapter: one NaN depth permanently
  // corrupts the Kalman state through the gain
  cb = [inner = std::move(cb)](const DepthReading& r) {
    if (!std::isfinite(r.depth)) return;
    inner(r);
  };
  cb = with_stamp_offset(node, "depth", std::move(cb));
  if (driver == "bar30") {
#ifdef HAVE_BAR30_DEPTH
    return node->create_subscription<bar30_depth::msg::Depth>(
      topic, qos, [cb](const bar30_depth::msg::Depth& msg) {
        DepthReading r;
        r.stamp = msg.header.stamp;
        r.depth = msg.depth;
        cb(r);
      });
#else
    driver_not_compiled("depth", driver, "bar30_depth");
#endif
  }
  if (driver == "fluid_pressure") {
    // sensor_msgs/FluidPressure is absolute pressure in Pascals; subtract the
    // surface atmosphere before converting to depth. A driver already
    // reporting gauge pressure should set depth.atmospheric_pressure: 0.0.
    const double rho = get_double_param(node, "depth.water_density", 1025.0);
    const double g = get_double_param(node, "depth.gravity", 9.80665);
    const double p_atm =
      get_double_param(node, "depth.atmospheric_pressure", 101325.0);
    return node->create_subscription<sensor_msgs::msg::FluidPressure>(
      topic, qos, [cb, rho, g, p_atm](const sensor_msgs::msg::FluidPressure& msg) {
        DepthReading r;
        r.stamp = msg.header.stamp;
        r.depth = (msg.fluid_pressure - p_atm) / (rho * g);
        cb(r);
      });
  }
  if (driver == "odom") {
    // depth carried in a nav_msgs/Odometry z (e.g. a payload that fuses depth
    // internally). depth.z_sign maps the frame convention: -1 (default) for an
    // ENU/up odometry (depth = -z), +1 for a z-down odometry.
    const double z_sign = get_double_param(node, "depth.z_sign", -1.0);
    return node->create_subscription<nav_msgs::msg::Odometry>(
      topic, qos, [cb, z_sign](const nav_msgs::msg::Odometry& msg) {
        DepthReading r;
        r.stamp = msg.header.stamp;
        r.depth = z_sign * msg.pose.pose.position.z;
        cb(r);
      });
  }
  if (driver == "float") {
    // bare std_msgs/Float64 depth in metres, positive down. It carries no
    // timestamp; stamp it on arrival (the depth stamp is only used as a
    // latest-value cache downstream, never for sync).
    return node->create_subscription<std_msgs::msg::Float64>(
      topic, qos, [node, cb](const std_msgs::msg::Float64& msg) {
        DepthReading r;
        r.stamp = node->now();
        r.depth = msg.data;
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
  // one non-finite delta angle would NaN the cumulative FOG integration
  cb = [inner = std::move(cb)](const GyroReading& r) {
    if (!r.delta.allFinite()) return;
    inner(r);
  };
  cb = with_stamp_offset(node, "gyro", std::move(cb));
  if (driver == "kvh_gyro") {
#ifdef HAVE_KVH_GYRO
    return node->create_subscription<kvh_gyro::msg::Gyro>(
      topic, qos, [cb](const kvh_gyro::msg::Gyro& msg) {
        GyroReading r;
        r.stamp = msg.header.stamp;
        r.delta << msg.delta[0], msg.delta[1], msg.delta[2];
        cb(r);
      });
#else
    driver_not_compiled("gyro", driver, "kvh_gyro");
#endif
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
  // a non-finite quaternion would NaN every attitude consumer
  cb = [inner = std::move(cb)](const ImuReading& r) {
    if (!r.orientation.coeffs().allFinite()) return;
    inner(r);
  };
  cb = with_stamp_offset(node, "imu", std::move(cb));
  const bool enu =
    driver == "enu" || driver == "3dm_gx5" || driver == "microstrain";
  // 'ned' is an AHRS already reporting the pipeline's z-down orientation (no
  // conversion and none of the VN100 legacy roll offset); 'vn100' keeps the
  // historic VectorNav frame handling applied downstream.
  const bool ned = driver == "ned";
  const bool legacy = imu_driver_is_legacy(driver);
  if (!legacy && !enu && !ned) unknown_driver("imu", driver);

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
  rclcpp::Node* node, const std::string& driver_in, const std::string& topic_in,
  const rclcpp::QoS& qos, std::function<void(const SonarPing&)> cb)
{
  // Blank driver = auto-select the Oculus adapter from compressed_images, for
  // EVERY consumer — slam/mapping subscribe to the same ping as the feature
  // node, and previously only feature_extraction implemented this fallback,
  // so a blank driver killed the other nodes at startup. A topic left blank
  // alongside a blank driver resolves to the selected driver's default
  // (callers that resolved it against the unresolved "" got "" back).
  const std::string driver =
    !driver_in.empty()
      ? driver_in
      : (get_bool_param(node, "compressed_images", false)
           ? std::string("oculus_compressed")
           : std::string("oculus_uncompressed"));
  const std::string topic =
    !topic_in.empty() ? topic_in : default_sonar_topic(driver);
  cb = with_stamp_offset(node, "sonar", std::move(cb));
  if (driver == "oculus_compressed") {
#ifdef HAVE_SONAR_OCULUS
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
#else
    driver_not_compiled("sonar", driver, "sonar_oculus");
#endif
  }
  if (driver == "oculus_uncompressed") {
#ifdef HAVE_SONAR_OCULUS
    return node->create_subscription<sonar_oculus::msg::OculusPingUncompressed>(
      topic, qos,
      [node, cb](const sonar_oculus::msg::OculusPingUncompressed& msg) {
        try {
          cv::Mat img = cv_bridge::toCvCopy(msg.ping, "")->image;
          if (img.type() != CV_8UC1) img.convertTo(img, CV_8UC1);
          cb(oculus_ping(msg, std::move(img)));
        } catch (const std::exception& e) {
          RCLCPP_WARN(node->get_logger(),
                      "Dropping sonar ping: image conversion failed: %s",
                      e.what());
        }
      });
#else
    driver_not_compiled("sonar", driver, "sonar_oculus");
#endif
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
    // range of image row 0, for sonars whose first sample is not at ~zero
    // range (same convention as SonarPing::range_min)
    const double range_min = get_double_param(node, "sonar.range_min", 0.0);
    auto count = std::make_shared<int>(0);
    return node->create_subscription<sensor_msgs::msg::Image>(
      topic, qos,
      [node, cb, range_resolution, horizontal_fov, range_min, count](
        const sensor_msgs::msg::Image& msg) {
        cv::Mat img;
        try {
          img = cv_bridge::toCvCopy(msg, "mono8")->image;
        } catch (const std::exception& e) {
          RCLCPP_WARN(node->get_logger(),
                      "Dropping sonar ping: image conversion failed: %s",
                      e.what());
          return;
        }
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
        ping.range_min = range_min;
        ping.ping_id = ++(*count);
        cb(ping);
      });
  }
  if (driver == "projected_sonar") {
    auto count = std::make_shared<int>(0);
    return node->create_subscription<marine_acoustic_msgs::msg::ProjectedSonarImage>(
      topic, qos,
      [node, cb, count](const marine_acoustic_msgs::msg::ProjectedSonarImage& msg) {
        // 8-bit images only (this vehicle's config)
        if (msg.image.dtype != 0) {  // SonarImageData::DTYPE_UINT8
          RCLCPP_WARN(node->get_logger(),
                      "Dropping projected_sonar ping: unsupported dtype %u",
                      static_cast<unsigned>(msg.image.dtype));
          return;
        }
        const int num_beams = static_cast<int>(msg.image.beam_count);
        const int num_ranges = static_cast<int>(msg.ranges.size());
        // beam_count / ranges.size() are independent of the actual image
        // payload, so guard the copy or a malformed message reads past the
        // end of image.data
        if (num_beams <= 0 || num_ranges <= 0 ||
            msg.image.data.size() <
              static_cast<std::size_t>(num_ranges) * num_beams) {
          RCLCPP_WARN(node->get_logger(),
                      "Dropping projected_sonar ping: image data (%zu bytes) "
                      "smaller than %d ranges x %d beams",
                      msg.image.data.size(), num_ranges, num_beams);
          return;
        }
        // beam_directions is independent of beam_count in the message; the map
        // builder assumes one bearing per image column, so a missing/mismatched
        // array would leave ping.bearings empty (UB in generate_map_xy's
        // front()/back()) or mis-index every beam. Drop it here instead.
        if (msg.beam_directions.size() != static_cast<std::size_t>(num_beams)) {
          RCLCPP_WARN(node->get_logger(),
                      "Dropping projected_sonar ping: %zu beam_directions for "
                      "%d beams",
                      msg.beam_directions.size(), num_beams);
          return;
        }
        // The ranges array comes straight off the wire. Duplicate/descending
        // entries give res <= 0 and NaN/garbage first entries feed the map
        // sizing (NaN->int casts are UB; a bogus min range would demand a
        // multi-GB Cartesian grid). Validate before anything downstream sees it.
        const double res =
          num_ranges > 1
            ? static_cast<double>(msg.ranges[1] - msg.ranges[0])
            : static_cast<double>(msg.ranges[0]) * 2.0;
        const double rmin = static_cast<double>(msg.ranges.front());
        if (!std::isfinite(res) || res <= 0.0 || !std::isfinite(rmin) ||
            rmin < 0.0 || rmin > res * 1.0e6) {
          RCLCPP_WARN(node->get_logger(),
                      "Dropping projected_sonar ping: invalid ranges "
                      "(resolution %.6g m, min range %.6g m)",
                      res, rmin);
          return;
        }

        SonarPing ping;
        ping.stamp = msg.header.stamp;
        // SonarImageData.data is range-major (row = range bin, col = beam) as
        // read by the reference apl-ocean-engineering consumer:
        // index = range * beam_count + beam.
        ping.image.create(num_ranges, num_beams, CV_8UC1);
        std::memcpy(ping.image.ptr(), msg.image.data.data(),
                    static_cast<std::size_t>(num_ranges) * num_beams);
        // bearing = atan2(-y, z), the driver's declared convention
        ping.bearings.reserve(msg.beam_directions.size());
        for (const auto& d : msg.beam_directions)
          ping.bearings.push_back(static_cast<float>(std::atan2(-d.y, d.z)));
        ping.num_ranges = num_ranges;
        ping.range_resolution = res;
        // range of image row 0: for a multibeam whose first sample is at a
        // nonzero minimum range, the polar->Cartesian map must offset by this
        // (an Oculus reports ~one cell, so this is ~0 there).
        ping.range_min = rmin;
        const double sos = msg.ping_info.sound_speed;
        ping.fire.speed_of_sound = sos > 0.0 ? sos : 1500.0;
        ping.fire.range = msg.ranges.empty() ? 0.0 : msg.ranges.back();
        ping.ping_id = ++(*count);
        // some drivers sweep beams starboard-to-port; the map builder needs
        // bearings ascending in column order
        normalize_bearing_order(ping);
        cb(ping);
      });
  }
  unknown_driver("sonar", driver);
}

}  // namespace sonar_slam
