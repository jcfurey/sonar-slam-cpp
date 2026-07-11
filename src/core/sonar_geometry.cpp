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

  // compare values, not just the count — a frequency-mode switch keeps the
  // beam count but changes the aperture
  if (!ping.bearings.empty() && ping.bearings != bearings) {
    num_bearings = static_cast<int>(ping.bearings.size());
    bearings = ping.bearings;
    horizontal_aperture = std::abs(bearings.back() - bearings.front());
    angular_resolution = horizontal_aperture / num_bearings;
    // OCULUS_VERTICAL_APERTURE: mode 1 -> 20 deg, mode 2 -> 12 deg,
    // default to the low-frequency aperture for unknown modes
    vertical_aperture =
      ping.fire.mode == 2 ? 12.0 * M_PI / 180.0 : 20.0 * M_PI / 180.0;
    changed = true;
  }

  return changed;
}

}  // namespace sonar_slam
