// Gyro integration node: port of bruce_slam gyro.py. Integrates the FOG
// delta angles (with earth-rate compensation) into euler angles published as
// an odometry message.
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <gtsam/geometry/Pose3.h>

#include <cmath>
#include <memory>

#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"

namespace sonar_slam {

class GyroNode : public SlamNodeBase
{
public:
  GyroNode() : SlamNodeBase("gyro_fusion")
  {
    // rotation offset between the gyro frame and the sonar frame (degrees)
    const double x = get_double("offset/x");
    const double y = get_double("offset/y");
    const double z = get_double("offset/z");
    offset_matrix_ = (Eigen::AngleAxisd(z * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(y * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(x * M_PI / 180.0, Eigen::Vector3d::UnitX()))
                       .toRotationMatrix();

    const double latitude = get_double("latitude") * M_PI / 180.0;
    // Earth rotation rate: 15.04107 deg/hr -> rad/s, so it is unit-consistent
    // with the radian delta angles the integration accumulates. gyro.py used
    // deg/s here (15.04107/3600) added to a radian integrator — a unit
    // mismatch. Intentional divergence from bruce_slam (see docs/DIVERGENCES.md).
    earth_rate_ = -15.04107 * std::sin(latitude) * M_PI / (180.0 * 3600.0);
    sensor_rate_ = get_double("sensor_rate");

    const std::string driver = get_string("gyro/driver", "kvh_gyro");
    const std::string topic = get_string("gyro/topic", GYRO_TOPIC);

    const int queue_depth = static_cast<int>(sensor_rate_) + 50;
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(GYRO_INTEGRATION_TOPIC,
                                                          queue_depth);
    // A payload without a FOG gyro (e.g. the Deep Trekker Revolution) either
    // omits this node via enable_gyro:=false or builds without kvh_gyro. Only
    // the not-compiled-in case may idle: any OTHER constructor failure (a
    // typo'd driver name, a bad topic) must stay a loud startup abort, or a
    // misconfigured FOG vehicle would silently lose all localization.
    try {
      gyro_sub_ = subscribe_gyro(this, driver, topic,
                                 rclcpp::SensorDataQoS(rclcpp::KeepLast(queue_depth)),
                                 [this](const GyroReading& r) { callback(r); });
      RCLCPP_INFO(get_logger(), "Gyro filtering node is initialized");
    } catch (const DriverNotCompiledError& e) {
      RCLCPP_WARN(get_logger(),
                  "Gyro node idle (no integration published): %s. Expected on a "
                  "payload with no FOG gyro; pass enable_gyro:=false to omit it.",
                  e.what());
    }
  }

private:
  void callback(const GyroReading& reading)
  {
    // apply the offset matrix (row vector · matrix, like gyro.py)
    const Eigen::Vector3d arr = offset_matrix_.transpose() * reading.delta;
    const double delta_yaw = arr[0];
    const double delta_pitch = arr[1];
    double delta_roll = arr[2];

    // Subtract the rotation of the earth over the MEASURED sample interval.
    // The nominal 1/sensor_rate made the compensation scale with message
    // count, not elapsed time (wrong under dropped best-effort samples or a
    // rate mismatch); dropped DELTA-ANGLE samples themselves are unrecoverable
    // — warn when a gap implies losses.
    double dt = 1.0 / sensor_rate_;
    const double t_now = to_sec(reading.stamp);
    if (last_stamp_ > 0.0) {
      const double measured = t_now - last_stamp_;
      if (measured >= 0.0 && measured < 1.0) {
        dt = measured;
        if (measured > 2.5 / sensor_rate_)
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "gyro sample gap %.1f ms (nominal %.1f ms): delta angles were "
            "dropped and their rotation is lost",
            measured * 1e3, 1e3 / sensor_rate_);
      } else if (measured >= 1.0) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "gyro stream gap %.2f s (nominal %.1f ms): every delta angle in "
          "the gap is lost — heading is now biased by any rotation during it",
          measured, 1e3 / sensor_rate_);
      }
    }
    last_stamp_ = t_now;
    delta_roll += earth_rate_ * dt;

    pitch_ += delta_pitch;
    yaw_ += delta_yaw;
    roll_ += delta_roll;

    const gtsam::Pose3 pose(gtsam::Rot3::Ypr(yaw_, pitch_, roll_),
                            gtsam::Point3(0, 0, 0));

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = reading.stamp;
    odom_msg.header.frame_id = "odom";
    odom_msg.pose.pose = g2r(pose);
    odom_msg.child_frame_id = "base_link";
    odom_pub_->publish(odom_msg);
  }

  Eigen::Matrix3d offset_matrix_ = Eigen::Matrix3d::Identity();
  double earth_rate_ = 0.0;
  double sensor_rate_ = 250.0;
  double last_stamp_ = 0.0;
  // Start attitude: 90 deg roll expressed in RADIANS (pi/2). gyro.py wrote the
  // literal 90. and passed it straight to Rot3.Ypr, which expects radians — a
  // unit bug (90 rad). Tracked in radians throughout, matching the radian delta
  // angles. Intentional divergence from bruce_slam (see docs/DIVERGENCES.md).
  double roll_ = M_PI / 2.0, yaw_ = 0.0, pitch_ = 0.0;

  rclcpp::SubscriptionBase::SharedPtr gyro_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::GyroNode>());
  rclcpp::shutdown();
  return 0;
}
