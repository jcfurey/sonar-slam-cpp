// Shared small types and the public sonar-SLAM topic contract.
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <string>

namespace sonar_slam {

// sonar_proc owns image decoding, artifact suppression, feature selection and
// ping-time projection. Its candidate cloud is the only SLAM sensor input.
inline const char* SONAR_POINTS_TOPIC = "/sensor/sonar/sonar0/proc_points";

inline const char* SLAM_NS = "/bruce/slam/";
inline const std::string SLAM_POSE_TOPIC = std::string(SLAM_NS) + "slam/pose";
inline const std::string SLAM_ODOM_TOPIC = std::string(SLAM_NS) + "slam/odom";
inline const std::string SLAM_TRAJ_TOPIC = std::string(SLAM_NS) + "slam/traj";
inline const std::string SLAM_CONSTRAINT_TOPIC = std::string(SLAM_NS) + "slam/constraint";

inline double to_sec(const builtin_interfaces::msg::Time& t)
{
  return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
}

}  // namespace sonar_slam
