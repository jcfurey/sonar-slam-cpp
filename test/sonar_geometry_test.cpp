// Unit coverage for endpoint-inclusive sonar bearing geometry.
#include "sonar_slam_cpp/sonar_geometry.hpp"

#include <cmath>
#include <cstdio>

int main()
{
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

  std::printf("PASS (endpoint-inclusive beam spacing)\n");
  return 0;
}
