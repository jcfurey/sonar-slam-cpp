// Sonar ping normalization + Oculus geometry: port of bruce_slam sensors.py
// (SonarPing) and sonar.py (OculusProperty, minus the matplotlib/shapely and
// deconvolution helpers, which the launched pipeline never calls).
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <opencv2/core.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sonar_slam {

struct FireMsg
{
  int mode = 1;
  int gamma = 255;       // raw gamma byte; 0/0xff mean gamma = 1.0
  int flags = 0;
  double range = 0.0;
  double gain = 0.0;
  double speed_of_sound = 1500.0;
  double salinity = 0.0;
};

// Normalized sonar ping: polar grayscale image (rows = range bins, cols =
// beams) + acoustic geometry. Bearings are always radians.
struct SonarPing
{
  builtin_interfaces::msg::Time stamp;
  cv::Mat image;                  // CV_8UC1
  std::vector<float> bearings;    // radians, ascending
  double range_resolution = 0.0;  // m per range bin
  double range_min = 0.0;         // range of image row 0 (0 for Oculus/image)
  int num_ranges = 0;
  int ping_id = 0;
  FireMsg fire;
  int part_number = 0;
};

// Decode the beam-major marine_acoustic_msgs/SonarImageData payload into the
// range-major polar image used internally by the feature and mapping code.
inline bool beamMajorToRangeMajor(
    const std::uint8_t* input,
    std::size_t input_size,
    std::size_t range_count,
    std::size_t beam_count,
    std::uint8_t* output) noexcept
{
  if (input == nullptr || output == nullptr ||
      range_count == 0 || beam_count == 0 ||
      range_count > input_size / beam_count) {
    return false;
  }
  for (std::size_t beam = 0; beam < beam_count; ++beam) {
    for (std::size_t range = 0; range < range_count; ++range) {
      output[range * beam_count + beam] =
          input[beam * range_count + range];
    }
  }
  return true;
}

// Preserve an explicitly configured minimum range unless it would reject the
// entire range interval CFAR can possibly emit. Short-range inspection bags
// can have a maximum range below a navigation-tuned body-exclusion radius
// (observed: 0.50 m bag versus 0.60 m gate). In that one impossible case,
// retain CFAR's own inner blanking boundary instead of producing an
// unconditionally empty cloud.
inline double effectiveCfarMinRange(
    double configured_min_range,
    double range_min,
    int range_count,
    double range_resolution,
    int cfar_border_bins) noexcept
{
  if (configured_min_range <= 0.0 || range_count <= 0 ||
      range_resolution <= 0.0 || cfar_border_bins < 0) {
    return configured_min_range;
  }
  const double usable_min =
      range_min + cfar_border_bins * range_resolution;
  const double usable_max =
      range_min + (range_count - cfar_border_bins) * range_resolution;
  if (usable_max > usable_min && configured_min_range >= usable_max)
    return usable_min;
  return configured_min_range;
}

class OculusProperty
{
public:
  // updates the geometry from a ping; returns true when it changed. Cheap
  // when nothing changed, so it can run on every ping — the operator can
  // change the sonar range/frequency mid-mission (per sonar.py configure())
  bool configure(const SonarPing& ping);

  double max_range = 30.0;
  double horizontal_aperture = 2.2689280275926285;  // radians(130)
  double vertical_aperture = 0.0;
  double range_resolution = 0.0;
  double angular_resolution = 0.0;
  int num_ranges = 0;
  int num_bearings = 0;
  std::vector<float> bearings;
};

}  // namespace sonar_slam
