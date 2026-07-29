// Geometry at the sonar_proc -> pose-graph boundary.
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <set>
#include <utility>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

struct ScanAdmissionParams
{
  int min_points = 40;
  double cell_size = 0.5;
  int min_occupied_cells = 8;
  double azimuth_bin_size = 0.08726646259971647;  // 5 deg
  int min_azimuth_bins = 3;
};

enum class ScanAdmissionReason
{
  ACCEPTED,
  NOT_ENOUGH_POINTS,
  NOT_ENOUGH_CELLS,
  NOT_ENOUGH_AZIMUTH,
};

struct ScanAdmission
{
  bool informative = false;
  ScanAdmissionReason reason = ScanAdmissionReason::NOT_ENOUGH_POINTS;
  int finite_points = 0;
  int occupied_cells = 0;
  int occupied_azimuth_bins = 0;
};

inline const char* scan_admission_reason(ScanAdmissionReason reason)
{
  switch (reason) {
    case ScanAdmissionReason::ACCEPTED: return "accepted";
    case ScanAdmissionReason::NOT_ENOUGH_POINTS: return "too few points";
    case ScanAdmissionReason::NOT_ENOUGH_CELLS: return "too few occupied cells";
    case ScanAdmissionReason::NOT_ENOUGH_AZIMUTH: return "too little azimuth coverage";
  }
  return "unknown";
}

// sonar_proc clouds use optical axes: +Z is boresight. Return its elevation
// relative to the base_link horizontal plane after applying base <- sensor.
inline double optical_boresight_pitch(const Eigen::Matrix3f& R_base_sensor)
{
  const Eigen::Vector3f forward =
    R_base_sensor * Eigen::Vector3f::UnitZ();
  return std::atan2(
    static_cast<double>(forward.z()),
    std::hypot(static_cast<double>(forward.x()),
               static_cast<double>(forward.y())));
}

// Transform every return in a flash ping with one sensor pose, then remove the
// vehicle's roll/pitch for planar registration. No per-point timing exists.
inline Matrix project_flash_ping(
  const Matrix& xyz_sensor,
  const Eigen::Matrix3f& R_base_sensor,
  const Eigen::RowVector3f& t_base_sensor,
  const Eigen::Matrix3f& R_horizontal_base)
{
  Matrix xyz_base = xyz_sensor * R_base_sensor.transpose();
  xyz_base.rowwise() += t_base_sensor;
  return xyz_base * R_horizontal_base.transpose();
}

// Discard malformed returns before either admission or registration. A single
// NaN must not poison ICP, and it must not count toward the open-water gate.
inline Matrix finite_points(const Matrix& points)
{
  int count = 0;
  for (int i = 0; i < points.rows(); ++i)
    if (points.row(i).allFinite()) ++count;

  Matrix out(count, points.cols());
  int at = 0;
  for (int i = 0; i < points.rows(); ++i)
    if (points.row(i).allFinite()) out.row(at++) = points.row(i);
  return out;
}

// Admission is deliberately independent of raw detector density. A scan needs
// enough finite returns, enough distinct horizontal cells, and returns across
// multiple azimuth bins. Thus a dense blob or one water-column ray cannot
// become graph evidence merely by carrying many neighboring points.
inline ScanAdmission assess_scan(
  const Matrix& points, const ScanAdmissionParams& params)
{
  ScanAdmission out;
  std::set<std::pair<std::int64_t, std::int64_t>> cells;
  std::set<std::int64_t> azimuth_bins;

  for (int i = 0; i < points.rows(); ++i) {
    if (points.cols() < 2 || !std::isfinite(points(i, 0)) ||
        !std::isfinite(points(i, 1)))
      continue;
    ++out.finite_points;
    const double x = static_cast<double>(points(i, 0));
    const double y = static_cast<double>(points(i, 1));
    cells.emplace(
      static_cast<std::int64_t>(std::floor(x / params.cell_size)),
      static_cast<std::int64_t>(std::floor(y / params.cell_size)));
    azimuth_bins.insert(static_cast<std::int64_t>(
      std::floor(std::atan2(y, x) / params.azimuth_bin_size)));
  }

  out.occupied_cells = static_cast<int>(cells.size());
  out.occupied_azimuth_bins = static_cast<int>(azimuth_bins.size());
  if (out.finite_points < params.min_points) {
    out.reason = ScanAdmissionReason::NOT_ENOUGH_POINTS;
  } else if (out.occupied_cells < params.min_occupied_cells) {
    out.reason = ScanAdmissionReason::NOT_ENOUGH_CELLS;
  } else if (out.occupied_azimuth_bins < params.min_azimuth_bins) {
    out.reason = ScanAdmissionReason::NOT_ENOUGH_AZIMUTH;
  } else {
    out.informative = true;
    out.reason = ScanAdmissionReason::ACCEPTED;
  }
  return out;
}

}  // namespace sonar_slam
