// ROS message conversions: port of bruce_slam utils/conversions.py and the
// message-building parts of utils/visualization.py.
#pragma once

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <gtsam/geometry/Pose3.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <utility>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

// ---------------------------------------------------------- pose conversions
gtsam::Pose3 r2g(const geometry_msgs::msg::Pose& pose);
geometry_msgs::msg::Pose g2r(const gtsam::Pose3& pose);

geometry_msgs::msg::TransformStamped make_transform(
  const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation,
  const builtin_interfaces::msg::Time& stamp, const std::string& parent_frame,
  const std::string& child_frame);

// ----------------------------------------------------------- PointCloud2 I/O
// XYZ float32 cloud from an Nx3 matrix (create_cloud_xyz32 equivalent)
sensor_msgs::msg::PointCloud2 make_cloud_xyz(const Matrix& points);

// float32 cloud with arbitrary field names, one column per field
sensor_msgs::msg::PointCloud2 make_cloud(const std::vector<std::string>& fields,
                                         const Matrix& data);

// extract the (x, y, z) channels as an Nx3 float matrix, NaN preserved
Matrix cloud_to_xyz(const sensor_msgs::msg::PointCloud2& msg);

// ---------------------------------------------------------------- constraint
// (point1, point2, color) with color "green" or "red", per visualization.py
using ConstraintLink =
  std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, std::string>;
visualization_msgs::msg::Marker ros_constraints(
  const std::vector<ConstraintLink>& links);

}  // namespace sonar_slam
