// Global scan-match initialization: port of slam.py's
// get_matching_cost_subroutine1 + scipy.optimize.shgo (sobol sampling +
// local refinement). The cost of a candidate delta is the negated count of
// transformed source points landing on a dilated occupancy grid built from
// the target points. Candidate evaluation is GPU-batched with an OpenMP CPU
// fallback.
#pragma once

#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>
#include <vector>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

struct GlobalInitResult
{
  bool success = false;
  gtsam::Pose2 delta;   // optimal perturbation of the source pose
  double best_cost = 0.0;
  // every evaluated sample as [x, y, theta, cost] of the *sampled source
  // pose* (source_pose.compose(delta)), like slam.py's pose_samples
  std::vector<Eigen::Vector4d> pose_samples;
};

// bounds: 3x2 [lo, hi] per dof; n/iters/ftol mirror the shgo
// initialization_params (n sobol samples per iteration, iters iterations,
// local-refine tolerance).
GlobalInitResult global_scan_match_init(
  const Matrix& source_points, const gtsam::Pose2& source_pose,
  const Matrix& target_points, const gtsam::Pose2& target_pose,
  double point_noise, const Eigen::Matrix<double, 3, 2>& bounds, int n,
  int iters, double ftol);

// Per-delta matching costs on the dilated target grid — the exact cost the
// initializer optimizes, exposed for parity checking and debugging. Uses the
// same GPU/CPU dispatch as the initializer.
std::vector<float> global_match_costs(
  const Matrix& source_points, const gtsam::Pose2& source_pose,
  const Matrix& target_points, const gtsam::Pose2& target_pose,
  double point_noise, const std::vector<Eigen::Vector3d>& deltas);

}  // namespace sonar_slam
