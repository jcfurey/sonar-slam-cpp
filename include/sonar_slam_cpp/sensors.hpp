// Pluggable sensor adapters: port of bruce_slam/sensors.py. Each sensor kind
// (imu, dvl, depth, gyro, sonar) has named driver adapters selected by the
// same `driver` parameter values as the Python stack; an adapter subscribes
// to the raw driver message and normalizes it into a small reading struct.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <builtin_interfaces/msg/time.hpp>
#include <functional>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "sonar_slam_cpp/sonar_geometry.hpp"

namespace sonar_slam {

// ------------------------------------------------------- normalized readings
struct DvlReading
{
  builtin_interfaces::msg::Time stamp;
  Eigen::Vector3d velocity;  // body frame m/s
  double altitude = 0.0;
};

struct DepthReading
{
  builtin_interfaces::msg::Time stamp;
  double depth = 0.0;  // m below surface
};

struct GyroReading
{
  builtin_interfaces::msg::Time stamp;
  Eigen::Vector3d delta;  // integrated delta angles (rad)
};

struct ImuReading
{
  builtin_interfaces::msg::Time stamp;
  Eigen::Quaterniond orientation;  // frame convention per adapter
};

// ------------------------------------------------------------- subscriptions
// Each factory creates the raw subscription for the named driver and invokes
// the callback with the normalized reading. Unknown driver names throw.
rclcpp::SubscriptionBase::SharedPtr subscribe_dvl(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const DvlReading&)> cb);

rclcpp::SubscriptionBase::SharedPtr subscribe_depth(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const DepthReading&)> cb);

rclcpp::SubscriptionBase::SharedPtr subscribe_gyro(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const GyroReading&)> cb);

rclcpp::SubscriptionBase::SharedPtr subscribe_imu(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const ImuReading&)> cb);

// Sonar adapters read geometry parameters off the node for the generic image
// driver, mirroring the Python GenericImageSonarAdapter.
rclcpp::SubscriptionBase::SharedPtr subscribe_sonar(
  rclcpp::Node* node, const std::string& driver, const std::string& topic,
  const rclcpp::QoS& qos, std::function<void(const SonarPing&)> cb);

// vn100 keeps the historic frame handling; enu/3dm_gx5/microstrain normalize
bool imu_driver_is_legacy(const std::string& driver);

// default sonar topic for a driver name ("" when it must be set explicitly)
std::string default_sonar_topic(const std::string& driver);

}  // namespace sonar_slam
