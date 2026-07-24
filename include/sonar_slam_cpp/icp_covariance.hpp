// Retained point-to-point Censi covariance math, separately unit-tested but
// NOT selectable by the runtime SLAM pipeline. The shipped ICP is
// point-to-plane, so Slam::configure rejects CENSI until this implementation
// is replaced by a covariance Hessian for the same objective.
#pragma once

#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>

#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/icp_covariance_math.hpp"

namespace sonar_slam {

struct CensiCovResult
{
  bool success = false;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();  // (x, y, theta) in the TARGET frame - the consumer must conjugate the translation block into the body frame (slam_core.cpp does)
  int n_correspondences = 0;
};

// Covariance of the estimated relative transform aligning `source` onto
// `target`. Correspondences are each source point's nearest target neighbour
// within max_dist (after applying `transform`, mirroring the ICP data
// association); sensor_var is the per-point coordinate noise variance on each
// cloud. Returns success=false when fewer than min_correspondences pairs
// remain or the geometry is degenerate, so the caller can fall back to the
// fixed ICP noise model.
CensiCovResult censi_icp_covariance(const Matrix& source, const Matrix& target,
                                    const gtsam::Pose2& transform,
                                    double max_dist, double sensor_var,
                                    int min_correspondences = 6);

}  // namespace sonar_slam
