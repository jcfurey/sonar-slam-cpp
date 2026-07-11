#include "sonar_slam_cpp/mcd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace sonar_slam {

namespace {

// chi2 inverse-CDF constants for d = 3 (the only dimension used here)
constexpr double CHI2_MEDIAN_3 = 2.3659738843753377;   // chi2.ppf(0.5, 3)
constexpr double CHI2_975_3 = 9.348403604496475;       // chi2.ppf(0.975, 3)

struct Estimate
{
  Eigen::VectorXd mu;
  Eigen::MatrixXd cov;
  double det = std::numeric_limits<double>::infinity();
  std::vector<int> support;
};

Eigen::VectorXd mahalanobis_sq(const Eigen::MatrixXd& X, const Eigen::VectorXd& mu,
                               const Eigen::MatrixXd& cov_inv)
{
  Eigen::VectorXd d2(X.rows());
  for (int i = 0; i < X.rows(); ++i) {
    const Eigen::VectorXd diff = X.row(i).transpose() - mu;
    d2[i] = diff.dot(cov_inv * diff);
  }
  return d2;
}

// one C-step: given a support set, recompute (mu, cov), then take the h
// samples with smallest Mahalanobis distance as the new support
bool c_step(const Eigen::MatrixXd& X, int h, Estimate& est)
{
  const int n = X.rows(), d = X.cols();
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(d);
  for (int idx : est.support) mu += X.row(idx).transpose();
  mu /= static_cast<double>(est.support.size());

  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);
  for (int idx : est.support) {
    const Eigen::VectorXd diff = X.row(idx).transpose() - mu;
    cov += diff * diff.transpose();
  }
  cov /= static_cast<double>(est.support.size());

  Eigen::FullPivLU<Eigen::MatrixXd> lu(cov);
  if (!lu.isInvertible()) return false;

  const Eigen::VectorXd d2 = mahalanobis_sq(X, mu, lu.inverse());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::nth_element(order.begin(), order.begin() + h - 1, order.end(),
                   [&](int a, int b) { return d2[a] < d2[b]; });
  order.resize(h);
  std::sort(order.begin(), order.end());

  est.mu = mu;
  est.cov = cov;
  est.det = lu.determinant();
  est.support = order;
  return true;
}

}  // namespace

McdResult min_cov_det(const Eigen::MatrixXd& X, double support_fraction)
{
  McdResult result;
  const int n = X.rows(), d = X.cols();
  // sklearn MinCovDet floors: n_support = int(support_fraction * n_samples)
  const int h = static_cast<int>(support_fraction * n);
  if (n < d + 1 || h < d + 1 || h > n) return result;

  std::mt19937 rng(0x5eed);  // deterministic across runs
  std::uniform_int_distribution<int> pick(0, n - 1);

  Estimate best;
  const int n_trials = std::min(500, 30 * n);
  for (int trial = 0; trial < n_trials; ++trial) {
    // elemental start: random (d+1)-subset grown by C-steps
    Estimate est;
    std::vector<char> used(n, 0);
    while (static_cast<int>(est.support.size()) < d + 1) {
      const int idx = pick(rng);
      if (!used[idx]) { used[idx] = 1; est.support.push_back(idx); }
    }
    bool ok = true;
    double prev_det = std::numeric_limits<double>::infinity();
    for (int step = 0; step < 30 && ok; ++step) {
      ok = c_step(X, h, est);
      if (!ok) break;
      if (std::abs(prev_det - est.det) < 1e-12) break;  // converged
      prev_det = est.det;
    }
    if (ok && est.det < best.det) best = est;
  }

  if (!std::isfinite(best.det)) return result;

  // consistency correction (sklearn correct_covariance): scale so the
  // estimator is consistent at the normal model
  Eigen::FullPivLU<Eigen::MatrixXd> lu(best.cov);
  if (!lu.isInvertible()) return result;
  Eigen::VectorXd d2 = mahalanobis_sq(X, best.mu, lu.inverse());
  std::vector<double> d2_sorted(d2.data(), d2.data() + d2.size());
  std::sort(d2_sorted.begin(), d2_sorted.end());
  // np.median semantics: average the two middle elements for even n
  const double median_d2 =
    (n % 2 != 0) ? d2_sorted[n / 2]
                 : 0.5 * (d2_sorted[n / 2 - 1] + d2_sorted[n / 2]);
  const double correction = median_d2 / CHI2_MEDIAN_3;
  if (correction <= 0.0 || !std::isfinite(correction)) return result;
  Eigen::MatrixXd cov_corrected = best.cov * correction;

  // reweighting step (sklearn reweight_covariance): drop samples beyond the
  // chi2 0.975 cutoff and recompute the classical estimate on the rest
  Eigen::FullPivLU<Eigen::MatrixXd> lu2(cov_corrected);
  if (!lu2.isInvertible()) return result;
  d2 = mahalanobis_sq(X, best.mu, lu2.inverse());

  std::vector<int> keep;
  for (int i = 0; i < n; ++i)
    if (d2[i] <= CHI2_975_3) keep.push_back(i);
  if (static_cast<int>(keep.size()) < d + 1) {
    result.success = true;
    result.location = best.mu;
    result.covariance = cov_corrected;
    return result;
  }

  Eigen::VectorXd mu_w = Eigen::VectorXd::Zero(d);
  for (int idx : keep) mu_w += X.row(idx).transpose();
  mu_w /= static_cast<double>(keep.size());
  Eigen::MatrixXd cov_w = Eigen::MatrixXd::Zero(d, d);
  for (int idx : keep) {
    const Eigen::VectorXd diff = X.row(idx).transpose() - mu_w;
    cov_w += diff * diff.transpose();
  }
  cov_w /= static_cast<double>(keep.size());

  result.success = true;
  result.location = mu_w;
  result.covariance = cov_w;
  return result;
}

}  // namespace sonar_slam
