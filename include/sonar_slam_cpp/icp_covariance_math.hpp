// Pure-math core of the Censi (2007) closed-form ICP covariance estimate,
// kept free of GTSAM/libpointmatcher so it can be unit-tested standalone.
//
// For 2D point-to-point registration each correspondence contributes a
// residual r_i(x) = R(theta) p_i + t - q_i with pose x = (tx, ty, theta) and
// Jacobian J_i = d r_i / d x = [ I2 | R'(theta) p_i ]  (2x3), where
// R'(theta) = d/dtheta R(theta). The Gauss-Newton information is the sum of
// J_i^T J_i over correspondences.
#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

// Sum_i J_i^T J_i (3x3) for the matched source points p_i at the registration
// angle theta. Only the source geometry and theta enter the Jacobian.
inline Eigen::Matrix3d censi_information(const Matrix& src_matched, double theta)
{
  const double c = std::cos(theta), s = std::sin(theta);
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (int i = 0; i < src_matched.rows(); ++i) {
    const double px = src_matched(i, 0), py = src_matched(i, 1);
    // R'(theta) p_i = [[-s, -c], [c, -s]] p_i
    const double d0 = -s * px - c * py;
    const double d1 = c * px - s * py;
    Eigen::Matrix<double, 2, 3> J;
    J << 1.0, 0.0, d0,
         0.0, 1.0, d1;
    H.noalias() += J.transpose() * J;
  }
  return H;
}

// Closed-form covariance of the estimated relative pose (tx, ty, theta) under
// an isotropic, independent per-point sensor-noise model with variance
// sensor_var on both point clouds. Linearising the registration at its
// optimum makes it a weighted least-squares fit whose residual noise is
// (var_source + var_target) I per correspondence, giving
//     cov = 2 * sensor_var * (Sum_i J_i^T J_i)^-1.
// Returns false when the information matrix is singular (degenerate geometry).
inline bool censi_covariance(const Matrix& src_matched, double theta,
                             double sensor_var, Eigen::Matrix3d& cov_out)
{
  const Eigen::Matrix3d H = censi_information(src_matched, theta);
  Eigen::FullPivLU<Eigen::Matrix3d> lu(H);
  if (!lu.isInvertible()) return false;
  cov_out = 2.0 * sensor_var * lu.inverse();
  return true;
}

}  // namespace sonar_slam
