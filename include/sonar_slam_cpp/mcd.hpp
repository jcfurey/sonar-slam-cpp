// Minimum Covariance Determinant robust location/scatter estimator — a compact
// FAST-MCD implementation replacing sklearn.covariance.MinCovDet for the ICP
// covariance sampling in slam.py (n ~ 30 samples, d = 3), including sklearn's
// consistency correction and reweighting steps.
#pragma once

#include <Eigen/Dense>

namespace sonar_slam {

struct McdResult
{
  bool success = false;
  Eigen::VectorXd location;
  Eigen::MatrixXd covariance;
};

// support_fraction: fraction of samples the estimate must cover (sklearn
// support_fraction; 0.8 in slam.py). Deterministic (fixed seed).
McdResult min_cov_det(const Eigen::MatrixXd& X, double support_fraction);

}  // namespace sonar_slam
