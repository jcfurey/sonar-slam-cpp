// Projection of the 2D sonar fan into a Cartesian target frame.
#pragma once

#include <Eigen/Core>

namespace sonar_slam {

// `points` stores [forward, lateral] in the sonar image convention. The
// corresponding optical-frame point is [0, lateral, forward]: optical +z is
// the acoustic boresight and optical +y is the fan's lateral axis.
inline Eigen::MatrixXf projectSonarPlane(
  const Eigen::MatrixXf& points,
  const Eigen::Matrix3f& target_from_optical_rotation,
  const Eigen::Vector3f& target_from_optical_translation)
{
  Eigen::MatrixXf optical(points.rows(), 3);
  optical.col(0).setZero();
  optical.col(1) = points.col(1);
  optical.col(2) = points.col(0);

  Eigen::MatrixXf target =
    optical * target_from_optical_rotation.transpose();
  target.rowwise() += target_from_optical_translation.transpose();
  return target;
}

}  // namespace sonar_slam
