#include "sonar_slam_cpp/sonar_geometry.hpp"

#include <cmath>

namespace sonar_slam {

bool OculusProperty::configure(const SonarPing& ping)
{
  bool changed = false;

  if (ping.num_ranges != num_ranges || ping.range_resolution != range_resolution) {
    num_ranges = ping.num_ranges;
    range_resolution = ping.range_resolution;
    max_range = range_resolution * num_ranges;
    changed = true;
  }

  if (static_cast<int>(ping.bearings.size()) != num_bearings) {
    num_bearings = static_cast<int>(ping.bearings.size());
    bearings = ping.bearings;
    horizontal_aperture = std::abs(bearings.back() - bearings.front());
    angular_resolution = horizontal_aperture / num_bearings;
    // OCULUS_VERTICAL_APERTURE: mode 1 -> 20 deg, mode 2 -> 12 deg,
    // default to the low-frequency aperture for unknown modes
    vertical_aperture =
      ping.fire.mode == 2 ? 12.0 * M_PI / 180.0 : 20.0 * M_PI / 180.0;

    std::vector<double> bx(bearings.begin(), bearings.end());
    std::vector<double> by(num_bearings);
    for (int i = 0; i < num_bearings; ++i) by[i] = i;
    b2c_ = Interp1d(bx, by, Interp1d::CUBIC, -1.0);
    changed = true;
  }

  if (changed && num_bearings > 0 && num_ranges > 0) {
    const double height = max_range;
    const int rows = num_ranges;
    const double width =
      std::sin((bearings.back() - bearings.front()) / 2.0) * height * 2.0;
    const int cols = static_cast<int>(std::ceil(width / range_resolution));

    remap_x.create(rows, cols, CV_32FC1);
    remap_y.create(rows, cols, CV_32FC1);
    for (int r = 0; r < rows; ++r) {
      float* px = remap_x.ptr<float>(r);
      float* py = remap_y.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const double x = range_resolution * (rows - r);
        const double y = range_resolution * (-cols / 2.0 + c + 0.5);
        const double b = std::atan2(y, x);
        const double range = std::sqrt(x * x + y * y);
        // ra2ro: round(range / resolution - 1)
        py[c] = static_cast<float>(std::round(range / range_resolution - 1.0));
        px[c] = static_cast<float>(b2c_(b));
      }
    }
  }

  return changed;
}

}  // namespace sonar_slam
