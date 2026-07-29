// Geometry at the sonar_proc -> pose-graph boundary.
#pragma once

#include <Eigen/Dense>

#include <cmath>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

// sonar_proc clouds use optical axes: +Z is boresight. Return its elevation
// relative to the base_link horizontal plane after applying base <- sensor.
inline double optical_boresight_pitch(const Eigen::Matrix3f& R_base_sensor)
{
  const Eigen::Vector3f forward =
    R_base_sensor * Eigen::Vector3f::UnitZ();
  return std::atan2(
    static_cast<double>(forward.z()),
    std::hypot(static_cast<double>(forward.x()),
               static_cast<double>(forward.y())));
}

// Transform every return in a flash ping with one sensor pose, then remove the
// vehicle's roll/pitch for planar registration. No per-point timing exists.
inline Matrix project_flash_ping(
  const Matrix& xyz_sensor,
  const Eigen::Matrix3f& R_base_sensor,
  const Eigen::RowVector3f& t_base_sensor,
  const Eigen::Matrix3f& R_horizontal_base)
{
  Matrix xyz_base = xyz_sensor * R_base_sensor.transpose();
  xyz_base.rowwise() += t_base_sensor;
  return xyz_base * R_horizontal_base.transpose();
}

}  // namespace sonar_slam
