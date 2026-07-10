// Shared small types and constants for sonar_slam_cpp.
// Topic names mirror bruce_slam/utils/topics.py verbatim so the two stacks are
// drop-in interchangeable.
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <string>

namespace sonar_slam {

// ---------------------------------------------------------------- input topics
inline const char* IMU_TOPIC = "/vn100/imu/raw";
inline const char* IMU_TOPIC_MK_II = "/vectornav/IMU";
inline const char* IMU_TOPIC_ENU = "/imu/data";
inline const char* DVL_TOPIC = "/rti/body_velocity/raw";
inline const char* DEPTH_TOPIC = "/bar30/depth/raw";
inline const char* SONAR_TOPIC = "/sonar_oculus_node/M750d/ping";
inline const char* SONAR_TOPIC_UNCOMPRESSED = "/sonar_oculus_node/ping";
inline const char* GYRO_TOPIC = "/gyro";

// --------------------------------------------------------------- output topics
inline const char* SLAM_NS = "/bruce/slam/";
inline const std::string GYRO_INTEGRATION_TOPIC = std::string(SLAM_NS) + "gyro_integrated";
inline const std::string LOCALIZATION_ODOM_TOPIC = std::string(SLAM_NS) + "localization/odom";
inline const std::string SLAM_POSE_TOPIC = std::string(SLAM_NS) + "slam/pose";
inline const std::string SLAM_ODOM_TOPIC = std::string(SLAM_NS) + "slam/odom";
inline const std::string SLAM_TRAJ_TOPIC = std::string(SLAM_NS) + "slam/traj";
inline const std::string SLAM_CLOUD_TOPIC = std::string(SLAM_NS) + "slam/cloud";
inline const std::string SLAM_CONSTRAINT_TOPIC = std::string(SLAM_NS) + "slam/constraint";
inline const std::string SONAR_FEATURE_TOPIC = std::string(SLAM_NS) + "feature_extraction/feature";
inline const std::string SONAR_FEATURE_IMG_TOPIC = std::string(SLAM_NS) + "feature_extraction/feature_img";

inline double to_sec(const builtin_interfaces::msg::Time& t)
{
  return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
}

}  // namespace sonar_slam
