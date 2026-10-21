// Global scan-match initialization: port of slam.py's
// get_matching_cost_subroutine1 + scipy.optimize.shgo (sobol sampling +
// local refinement). The cost of a candidate delta is the negated count of
// transformed source points landing on a dilated occupancy grid built from
// the target points. Candidate evaluation is GPU-batched with an OpenMP CPU
// fallback.
#pragma once

#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>
#include <cstddef>
#include <vector>

#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/deadline.hpp"

namespace sonar_slam {

// The raster is a proposal generator. Coarsen its cells on large footprints;
// final registration, observability and ambiguity checks use the XYZ geometry.
inline constexpr int GLOBAL_INIT_MAX_GRID_AXIS = 16384;
inline constexpr std::size_t GLOBAL_INIT_MAX_GRID_CELLS = 16 * 1024 * 1024;
inline constexpr int GLOBAL_INIT_MAX_SAMPLES = 1000000;

struct GlobalInitResult
{
  bool success = false;
  bool timed_out = false;
  gtsam::Pose2 delta;   // optimal perturbation of the source pose
  double best_cost = 0.0;
  // every evaluated sample as [x, y, theta, cost] of the *sampled source
  // pose* (source_pose.compose(delta)), like slam.py's pose_samples
  std::vector<Eigen::Vector4d> pose_samples;
  // Actual proposal-raster resolution and extent, including any coarsening.
  double grid_resolution = 0.0;
  int grid_rows = 0, grid_cols = 0;
};

// bounds: 3x2 [lo, hi] per dof; n/iters/ftol mirror the shgo
// initialization_params (n sobol samples per iteration, iters iterations,
// local-refine tolerance). Local refinement also stays inside bounds. Invalid
// geometry/settings throw std::invalid_argument; empty clouds return failure.
// A success only establishes nonzero coarse overlap, not a valid registration.
GlobalInitResult global_scan_match_init(
  const Matrix& source_points, const gtsam::Pose2& source_pose,
  const Matrix& target_points, const gtsam::Pose2& target_pose,
  double point_noise, const Eigen::Matrix<double, 3, 2>& bounds, int n,
  int iters, double ftol, Deadline deadline = Deadline::max());

// Per-delta matching costs on the dilated target grid — the exact cost the
// initializer optimizes, exposed for parity checking and debugging. Uses the
// same GPU/CPU dispatch as the initializer.
std::vector<float> global_match_costs(
  const Matrix& source_points, const gtsam::Pose2& source_pose,
  const Matrix& target_points, const gtsam::Pose2& target_pose,
  double point_noise, const std::vector<Eigen::Vector3d>& deltas);

}  // namespace sonar_slam
