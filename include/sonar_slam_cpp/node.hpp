#pragma once

#include <rclcpp/rclcpp.hpp>

namespace sonar_slam {

std::shared_ptr<rclcpp::Node> make_slam_node(
  const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

}  // namespace sonar_slam
