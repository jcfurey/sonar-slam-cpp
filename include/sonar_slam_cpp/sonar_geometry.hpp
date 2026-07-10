// Sonar ping normalization + Oculus geometry: port of bruce_slam sensors.py
// (SonarPing) and sonar.py (OculusProperty, minus the matplotlib/shapely and
// deconvolution helpers, which the launched pipeline never calls).
#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "sonar_slam_cpp/interp.hpp"

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
  int num_ranges = 0;
  int ping_id = 0;
  FireMsg fire;
  int part_number = 0;
};

class OculusProperty
{
public:
  // returns true when the geometry changed (per sonar.py configure())
  bool configure(const SonarPing& ping);

  double max_range = 30.0;
  double horizontal_aperture = 2.2689280275926285;  // radians(130)
  double vertical_aperture = 0.0;
  double range_resolution = 0.0;
  double angular_resolution = 0.0;
  int num_ranges = 0;
  int num_bearings = 0;
  std::vector<float> bearings;

  // polar <-> Cartesian remap maps (CV_32FC1), built on configure()
  cv::Mat remap_x, remap_y;

private:
  Interp1d b2c_;  // bearing -> column (cubic, like sonar.py)
};

}  // namespace sonar_slam
