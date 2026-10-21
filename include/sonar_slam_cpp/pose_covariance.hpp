#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace sonar_slam {

inline bool valid_pose_covariance(const Eigen::Matrix3d& covariance)
{
  if (!covariance.allFinite() ||
      (covariance.diagonal().array() <= 0.0).any()) return false;
  // Eigen's relative symmetry test and an unscaled Frobenius norm square
  // entries. At extreme finite scales they overflow and can admit asymmetric
  // or indefinite matrices. Normalize before both comparisons and the solve.
  const Eigen::Matrix3d scaled = covariance / covariance.cwiseAbs().maxCoeff();
  if (!scaled.isApprox(scaled.transpose(), 1e-9)) return false;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(scaled);
  return eigen.info() == Eigen::Success &&
    eigen.eigenvalues().minCoeff() >= -1e-12 * std::max(1.0, scaled.norm());
}

// Unknown uncertainty must never become a small, confident diagonal. The
// caller can continue publishing TF while a fusing consumer discounts the pose.
inline Eigen::Matrix3d publication_covariance(
  const Eigen::Matrix3d& covariance, bool available)
{
  if (!available || !valid_pose_covariance(covariance))
    return Eigen::Matrix3d::Identity() * 1e6;
  Eigen::Matrix3d out = covariance * 0.5 + covariance.transpose() * 0.5;
  for (int i = 0; i < 3; ++i) out(i, i) = std::max(1e-3, out(i, i));
  return out;
}

// Propagate fixed-map [x,y,yaw] covariance through a relative DR increment.
// Its noise uses right perturbations [dx,dy,dyaw] at the destination body.
// Parent heading uncertainty moves the extrapolated position through the
// map-frame displacement; it also creates position/heading cross covariance.
inline Eigen::Matrix3d propagate_planar_covariance(
  const Eigen::Matrix3d& map_covariance, const Eigen::Vector2d& map_displacement,
  double destination_yaw, const Eigen::Matrix3d& increment_covariance)
{
  Eigen::Matrix3d parent_jacobian = Eigen::Matrix3d::Identity();
  parent_jacobian(0, 2) = -map_displacement.y();
  parent_jacobian(1, 2) = map_displacement.x();
  const double c = std::cos(destination_yaw), s = std::sin(destination_yaw);
  Eigen::Matrix3d increment_jacobian;
  increment_jacobian << c, -s, 0, s, c, 0, 0, 0, 1;
  const Eigen::Matrix3d result =
    parent_jacobian * map_covariance * parent_jacobian.transpose() +
    increment_jacobian * increment_covariance * increment_jacobian.transpose();
  return result * .5 + result.transpose() * .5;
}

}  // namespace sonar_slam
