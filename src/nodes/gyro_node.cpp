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
    // earth rotation rate: 15.04107 deg/hr at the pole -> rad/s, matching the
    // radian delta angles the gyro adapters deliver
    earth_rate_ = -15.04107 * std::sin(latitude) * M_PI / (180.0 * 3600.0);
    sensor_rate_ = get_double("sensor_rate");

    const std::string driver = get_string("gyro/driver", "kvh_gyro");
    const std::string topic = get_string("gyro/topic", GYRO_TOPIC);

    const int queue_depth = static_cast<int>(sensor_rate_) + 50;
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(GYRO_INTEGRATION_TOPIC,
                                                          queue_depth);
    gyro_sub_ = subscribe_gyro(this, driver, topic, rclcpp::QoS(queue_depth),
                               [this](const GyroReading& r) { callback(r); });

    RCLCPP_INFO(get_logger(), "Gyro filtering node is initialized");
  }

private:
  void callback(const GyroReading& reading)
  {
    // apply the offset matrix (row vector · matrix, like gyro.py)
    const Eigen::Vector3d arr = offset_matrix_.transpose() * reading.delta;
    const double delta_yaw = arr[0];
    const double delta_pitch = arr[1];
    double delta_roll = arr[2];

    // subtract the rotation of the earth
    delta_roll += earth_rate_ / sensor_rate_;

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
  // start attitude per gyro.py (roll = 90 deg), tracked in RADIANS throughout:
  // the gyro adapters deliver radian delta angles and Rot3::Ypr expects radians
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
