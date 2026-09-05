#include "sonar_slam_cpp/slam_core.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <unistd.h>

int main()
{
  using sonar_slam::Keyframe;
  using sonar_slam::Matrix;
  using sonar_slam::Slam;
  int failures = 0;
  const auto check = [&](bool ok, const char* label) {
    if (!ok) { std::printf("FAIL: %s\n", label); ++failures; }
  };
  const auto dir = std::filesystem::temp_directory_path() /
                   ("sonar-slam-map-test-" + std::to_string(getpid()));
  std::filesystem::create_directory(dir);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove_all(path); }
  } cleanup{dir};
  const auto good_path = (dir / "good.ssm").string();
  const auto bad_path = (dir / "bad.ssm").string();
  Slam original;
  original.ssm_params.enable = false;
  original.configure();
  for (int k = 0; k < 2; ++k) {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = k == 0 ? -2 : 1;
    stamp.nanosec = 750000000;
    auto frame = std::make_shared<Keyframe>(
      true, stamp, gtsam::Pose3(gtsam::Rot3::Yaw(0.1 * k),
                               gtsam::Point3(k, 0, -2)));
    frame->points = Matrix::Identity(3, 3);
    if (k == 0) original.add_prior(frame);
    else original.add_odometry(frame);
    check(original.update_factor_graph(frame), "build fixture graph");
  }
  check(original.save_map(good_path), "save fixture map");
  std::ifstream in(good_path, std::ios::binary);
  const std::vector<char> good((std::istreambuf_iterator<char>(in)), {});
  Slam restored;
  restored.ssm_params.enable = false;
  restored.configure();
  check(restored.load_map(good_path), "load valid map");
  check(restored.loaded_keyframes() == 2, "restore both keyframes");
  if (restored.keyframes.size() == 2) {
    check(restored.keyframes[0]->time == original.keyframes[0]->time,
          "negative fractional timestamp round trip");
    check(restored.keyframes[1]->points.isApprox(original.keyframes[1]->points),
          "point payload round trip");
  }
  const auto rejects = [&](const std::vector<char>& bytes, const char* label) {
    { std::ofstream out(bad_path, std::ios::binary);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); }
    Slam s;
    s.ssm_params.enable = false;
    s.configure();
    bool loaded = false;
    try { loaded = s.load_map(bad_path); }
    catch (...) { check(false, "corrupt map must return failure, not throw"); }
    check(!loaded, label);
    check(s.keyframes.empty() && !s.awaiting_relocalization() &&
            s.loaded_keyframes() == 0, "failed load must leave an empty session");
    check(s.load_map(good_path), "valid load must work after a failed load");
  };
  auto empty = good;
  empty.resize(16);
  std::fill(empty.begin() + 12, empty.end(), 0);
  rejects(empty, "empty map must not arm permanent relocalization");
  for (std::size_t cut : {std::size_t(1), std::size_t(31), good.size() - 1}) {
    rejects(std::vector<char>(good.begin(), good.begin() + cut), "truncated map");
  }
  auto bad = good;
  bad.push_back(0);
  rejects(bad, "trailing map data");
  // Header=16, stamp=8, each pose=7 doubles, planar estimate=3 doubles.
  const auto corrupt_double = [&](std::size_t offset, double value, const char* label) {
    auto bytes = good;
    std::memcpy(bytes.data() + offset, &value, sizeof value);
    rejects(bytes, label);
  };
  corrupt_double(24, std::numeric_limits<double>::quiet_NaN(), "NaN pose rotation");
  corrupt_double(24 + 4 * sizeof(double),
                 std::numeric_limits<double>::infinity(), "infinite pose position");
  corrupt_double(136 + 2 * sizeof(double),
                 std::numeric_limits<double>::quiet_NaN(), "NaN optimized yaw");
  bad = good;
  std::fill(bad.begin() + 24, bad.begin() + 24 + 4 * sizeof(double), 0);
  rejects(bad, "zero quaternion");
  bad = good;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(bad.data() + 168, &nan, sizeof nan);
  rejects(bad, "non-finite cloud payload");
  bad = good;
  const std::int64_t oversized_rows = 50000000;
  std::memcpy(bad.data() + 160, &oversized_rows, sizeof oversized_rows);
  rejects(bad, "point count larger than the file payload");
  bad = good;
  const std::int64_t oversized_stamp = std::numeric_limits<std::int64_t>::max();
  std::memcpy(bad.data() + 16, &oversized_stamp, sizeof oversized_stamp);
  rejects(bad, "timestamp outside the ROS seconds range");
  return failures ? 1 : 0;
}
