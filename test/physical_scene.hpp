#pragma once
#include "sonar_slam_cpp/keyframe.hpp"

namespace sonar_slam::test {
inline Matrix physical_scene()
{
  constexpr int n = 31;
  Matrix cloud(3 * n * n, 3);
  int row = 0;
  for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) {
    const float u = -3.0f + a * .2f, v = -3.0f + b * .2f;
    cloud.row(row++) << 4.0f, u, v;
    cloud.row(row++) << u, 4.5f, v;
    cloud.row(row++) << u, v, -3.5f;
  }
  return cloud;
}
inline KeyframePtr physical_frame(int seconds, const gtsam::Pose3& truth,
                                  const gtsam::Pose3& odometry, const Matrix& scene)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = seconds;
  auto frame = std::make_shared<Keyframe>(true, stamp, odometry);
  frame->chart = CloudChart::Body;
  frame->points = transform_body_points(scene, truth.inverse());
  return frame;
}
}  // namespace sonar_slam::test
