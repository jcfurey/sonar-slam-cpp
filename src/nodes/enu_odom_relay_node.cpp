// Relay an ENU (REP-105) odometry topic into the pipeline's z-down convention
// (port of bruce_slam scripts/enu_odom_relay.py): conjugate pose and twist by
// the roll-pi transform, pass headers through untouched.
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <memory>

#include "sonar_slam_cpp/common.hpp"

namespace sonar_slam {

class EnuOdomRelay : public rclcpp::Node
{
public:
  EnuOdomRelay() : rclcpp::Node("enu_odom_relay")
  {
    declare_parameter("input_topic", "/localization/odometry/odom");
    const std::string input_topic = get_parameter("input_topic").as_string();

    pub_ = create_publisher<nav_msgs::msg::Odometry>(LOCALIZATION_ODOM_TOPIC, 10);
    sub_ = create_subscription<nav_msgs::msg::Odometry>(
      input_topic, 10,
      [this](const nav_msgs::msg::Odometry& msg) { callback(msg); });
  }

private:
  // sign signature of the roll-pi conjugation on (x, y, z, rot_x, rot_y, rot_z)
  static constexpr std::array<double, 6> FLIP = {1.0, -1.0, -1.0, 1.0, -1.0, -1.0};

  static std::array<double, 36> flip_covariance(const std::array<double, 36>& cov)
  {
    std::array<double, 36> out;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) out[i * 6 + j] = FLIP[i] * FLIP[j] * cov[i * 6 + j];
    return out;
  }

  void callback(const nav_msgs::msg::Odometry& msg)
  {
    nav_msgs::msg::Odometry out;
    out.header = msg.header;
    out.child_frame_id = msg.child_frame_id;

    out.pose.pose.position.x = msg.pose.pose.position.x;
    out.pose.pose.position.y = -msg.pose.pose.position.y;
    out.pose.pose.position.z = -msg.pose.pose.position.z;
    out.pose.pose.orientation.x = msg.pose.pose.orientation.x;
    out.pose.pose.orientation.y = -msg.pose.pose.orientation.y;
    out.pose.pose.orientation.z = -msg.pose.pose.orientation.z;
    out.pose.pose.orientation.w = msg.pose.pose.orientation.w;
    out.pose.covariance = flip_covariance(msg.pose.covariance);

    out.twist.twist.linear.x = msg.twist.twist.linear.x;
    out.twist.twist.linear.y = -msg.twist.twist.linear.y;
    out.twist.twist.linear.z = -msg.twist.twist.linear.z;
    out.twist.twist.angular.x = msg.twist.twist.angular.x;
    out.twist.twist.angular.y = -msg.twist.twist.angular.y;
    out.twist.twist.angular.z = -msg.twist.twist.angular.z;
    out.twist.covariance = flip_covariance(msg.twist.covariance);

    pub_->publish(out);
  }

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::EnuOdomRelay>());
  rclcpp::shutdown();
  return 0;
}
