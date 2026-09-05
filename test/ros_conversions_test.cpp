#include "sonar_slam_cpp/ros_conversions.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>

int main()
{
  using sonar_slam::Matrix;
  using sonar_slam::cloud_to_xyz;
  int failures = 0;
  const auto check = [&](bool ok, const char* label) {
    if (!ok) { std::printf("FAIL: %s\n", label); ++failures; }
  };
  Matrix expected(4, 3);
  expected << 1, 2, 3, -4, 5, 6, 7, -8, 9, 10, 11, -12;
  const auto packed = sonar_slam::make_cloud_xyz(expected);
  check(cloud_to_xyz(packed).isApprox(expected), "packed XYZ round trip");

  // Organized clouds may have row padding and unaligned fields/point strides.
  // Construct the wire bytes independently of make_cloud/cloud_to_xyz.
  for (bool big_endian : {false, true}) {
    auto msg = packed;
    msg.width = 2;
    msg.height = 2;
    msg.point_step = 17;
    msg.row_step = 39;
    msg.is_bigendian = big_endian;
    for (int c = 0; c < 3; ++c) msg.fields[c].offset = 1 + c * 5;
    msg.data.assign(msg.row_step * msg.height, 0xa5);
    for (int i = 0; i < 4; ++i) {
      for (int c = 0; c < 3; ++c) {
        std::uint32_t bits;
        const float value = expected(i, c);
        std::memcpy(&bits, &value, sizeof bits);
        const auto offset = (i / 2) * msg.row_step +
                            (i % 2) * msg.point_step + msg.fields[c].offset;
        for (int b = 0; b < 4; ++b)
          msg.data[offset + b] = static_cast<std::uint8_t>(
            bits >> (8 * (big_endian ? 3 - b : b)));
      }
    }
    const Matrix decoded = cloud_to_xyz(msg);
    check(decoded.rows() == expected.rows() && decoded.isApprox(expected),
          big_endian ? "big-endian padded XYZ" : "little-endian padded XYZ");
  }

  const auto rejects = [&](const auto& msg, const char* label) {
    const Matrix decoded = cloud_to_xyz(msg);
    check(decoded.rows() == 0 && decoded.cols() == 3, label);
  };
  auto bad = packed;
  bad.data.pop_back();
  rejects(bad, "truncated payload");
  bad = packed;
  bad.data.clear();
  rejects(bad, "missing payload");
  bad = packed;
  bad.fields[2].offset = std::numeric_limits<std::uint32_t>::max();
  rejects(bad, "field offset beyond point stride");
  bad = packed;
  bad.fields[2].count = 0;
  rejects(bad, "zero field count");
  bad = packed;
  bad.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT64;
  rejects(bad, "unsupported field type");
  bad = packed;
  bad.fields.push_back(bad.fields[0]);
  rejects(bad, "ambiguous duplicate XYZ field");
  bad = packed;
  bad.row_step = bad.point_step * bad.width - 1;
  rejects(bad, "row stride shorter than its points");
  bad = packed;
  bad.point_step = 0;
  rejects(bad, "zero point stride");
  bad = packed;
  bad.width = bad.height = std::numeric_limits<std::uint32_t>::max();
  rejects(bad, "overflowing dimensions with short payload");
  rejects(sensor_msgs::msg::PointCloud2{}, "empty cloud");
  return failures ? 1 : 0;
}
