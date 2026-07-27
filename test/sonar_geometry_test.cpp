// Unit coverage for endpoint-inclusive sonar bearing geometry.
#include "sonar_slam_cpp/sonar_geometry.hpp"
#include "sonar_slam_cpp/sonar_projection.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main()
{
  const std::vector<std::uint8_t> beam_major{10, 11, 12, 20, 21, 22};
  const std::vector<std::uint8_t> expected_range_major{
    10, 20, 11, 21, 12, 22};
  std::vector<std::uint8_t> range_major(beam_major.size());
  if (!sonar_slam::beamMajorToRangeMajor(
          beam_major.data(), beam_major.size(), 3, 2, range_major.data()) ||
      range_major != expected_range_major) {
    std::printf("FAIL: ProjectedSonarImage layout conversion\n");
    return 1;
  }
  if (sonar_slam::beamMajorToRangeMajor(
          beam_major.data(), beam_major.size() - 1, 3, 2,
          range_major.data())) {
    std::printf("FAIL: truncated ProjectedSonarImage was accepted\n");
    return 1;
  }

  const double normal_gate = sonar_slam::effectiveCfarMinRange(
    0.6, 0.0, 637, 0.008, 25);
  if (std::abs(normal_gate - 0.6) > 1e-12) {
    std::printf("FAIL: valid configured minimum range was altered\n");
    return 1;
  }
  const double short_range_gate = sonar_slam::effectiveCfarMinRange(
    0.6, 0.0, 255, 0.002, 25);
  if (std::abs(short_range_gate - 0.05) > 1e-12) {
    std::printf("FAIL: impossible short-range gate was not bounded\n");
    return 1;
  }

  // The extracted fan stores [forward, lateral], while its physical optical
  // coordinates are [0, lateral, forward]. Lock in both that axis mapping and
  // the sensor translation: omitting the latter caused the deployed clouds to
  // be displaced by the full ~0.286 m head/sonar lever arm.
  Eigen::MatrixXf fan(2, 2);
  fan << 2.0f, 1.0f,
         4.0f, -0.5f;
  const Eigen::Vector3f lever_arm(0.286f, -0.1f, 0.2f);
  const Eigen::MatrixXf translated = sonar_slam::projectSonarPlane(
    fan, Eigen::Matrix3f::Identity(), lever_arm);
  Eigen::MatrixXf expected_translated(2, 3);
  expected_translated << 0.286f, 0.9f, 2.2f,
                         0.286f, -0.6f, 4.2f;
  if (!translated.isApprox(expected_translated, 1e-6f)) {
    std::printf("FAIL: optical-axis or lever-arm projection\n");
    return 1;
  }

  const Eigen::Matrix3f pitch_90 =
    Eigen::AngleAxisf(
      static_cast<float>(M_PI_2), Eigen::Vector3f::UnitY()).toRotationMatrix();
  const Eigen::MatrixXf pitched =
    sonar_slam::projectSonarPlane(fan, pitch_90, lever_arm);
  Eigen::MatrixXf expected_pitched(2, 3);
  expected_pitched << 2.286f, 0.9f, 0.2f,
                      4.286f, -0.6f, 0.2f;
  if (!pitched.isApprox(expected_pitched, 1e-5f)) {
    std::printf("FAIL: rotated optical-frame projection\n");
    return 1;
  }

  sonar_slam::SonarPing ping;
  ping.num_ranges = 100;
  ping.range_resolution = 0.1;
  ping.bearings = {-0.5f, 0.0f, 0.5f};

  sonar_slam::OculusProperty geometry;
  if (!geometry.configure(ping)) {
    std::printf("FAIL: first configure did not report a change\n");
    return 1;
  }
  if (std::abs(geometry.max_range - 10.0) > 1e-12 ||
      std::abs(geometry.horizontal_aperture - 1.0) > 1e-12 ||
      std::abs(geometry.angular_resolution - 0.5) > 1e-12) {
    std::printf("FAIL: range/aperture/spacing = %.17g/%.17g/%.17g\n",
                geometry.max_range, geometry.horizontal_aperture,
                geometry.angular_resolution);
    return 1;
  }
  if (geometry.configure(ping)) {
    std::printf("FAIL: identical geometry reported a change\n");
    return 1;
  }

  ping.bearings = {0.25f};
  if (!geometry.configure(ping) || geometry.angular_resolution != 0.0) {
    std::printf("FAIL: a single beam must have zero angular spacing\n");
    return 1;
  }

  std::printf(
    "PASS (sonar layout, projection, and endpoint-inclusive beam spacing)\n");
  return 0;
}
