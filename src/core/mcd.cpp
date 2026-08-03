#include "sonar_slam_cpp/mcd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace sonar_slam {

namespace {

// Per-worker scratch reused across C-steps. The estimator runs up to 500
// trials x 30 C-steps per call, so anything allocated inside c_step is
// allocated ~15000 times per scan match.
struct Scratch
{
  std::vector<double> d2;
  std::vector<int> order;
  Eigen::VectorXd diff;
  Eigen::VectorXd whitened;
};

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

// Squared Mahalanobis distance of every row of X, into `out`. The per-row
// difference and the whitened product are written into caller-owned buffers
// that are already the right size, so the expressions evaluate exactly as
// before (same operands, same Eigen kernels) without a heap allocation per
// sample.
void mahalanobis_sq(const Eigen::MatrixXd& X, const Eigen::VectorXd& mu,
                    const Eigen::MatrixXd& cov_inv, Scratch& s,
                    std::vector<double>& out)
{
  const int n = static_cast<int>(X.rows()), d = static_cast<int>(X.cols());
  out.resize(static_cast<std::size_t>(n));
  if (s.diff.size() != d) s.diff.resize(d);
  if (s.whitened.size() != d) s.whitened.resize(d);
  for (int i = 0; i < n; ++i) {
    s.diff = X.row(i).transpose() - mu;
    s.whitened.noalias() = cov_inv * s.diff;
    out[static_cast<std::size_t>(i)] = s.diff.dot(s.whitened);
  }
}

// one C-step: given a support set, recompute (mu, cov), then take the h
// samples with smallest Mahalanobis distance as the new support
bool c_step(const Eigen::MatrixXd& X, int h, Estimate& est, Scratch& s)
{
  const int n = X.rows(), d = X.cols();
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(d);
  for (int idx : est.support) mu += X.row(idx).transpose();
  mu /= static_cast<double>(est.support.size());

  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d, d);
  if (s.diff.size() != d) s.diff.resize(d);
  for (int idx : est.support) {
    s.diff = X.row(idx).transpose() - mu;
    cov += s.diff * s.diff.transpose();
  }
  cov /= static_cast<double>(est.support.size());

  Eigen::FullPivLU<Eigen::MatrixXd> lu(cov);
  if (!lu.isInvertible()) return false;

  mahalanobis_sq(X, mu, lu.inverse(), s, s.d2);
  const std::vector<double>& d2 = s.d2;
  std::vector<int>& order = s.order;
  order.resize(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::nth_element(order.begin(), order.begin() + h - 1, order.end(),
                   [&](int a, int b) { return d2[a] < d2[b]; });
  order.resize(static_cast<std::size_t>(h));
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
  // CHI2_MEDIAN_3 / CHI2_975_3 are inverse-CDF constants for d = 3. Both
  // callers pass (x, y, theta) samples, but a different width would silently
  // mis-scale the consistency correction AND mis-set the reweighting cutoff —
  // a wrong covariance rather than a failure, which now feeds the NSSM
  // degeneracy gate. Refuse instead: return the default-constructed result,
  // i.e. success = false, exactly as the size guards above do. That is this
  // function's failure contract, not a silent success — the callers branch on
  // it (slam_core.cpp reports "Failed to calculate covariance" and abandons
  // the estimate). Deliberately NOT a throw: every other rejection here is a
  // return, and this runs inside the SLAM callback where a config-shaped
  // error must not unwind the graph update. `d` is fixed at 3 by both call
  // sites today, so this guards future misuse rather than a live path.
  if (d != 3) return result;

  std::mt19937 rng(0x5eed);  // deterministic across runs
  std::uniform_int_distribution<int> pick(0, n - 1);

  const int n_trials = std::min(500, 30 * n);

  // Draw every trial's elemental start FIRST, off the single deterministic RNG
  // stream. The trials themselves never touch the RNG or each other, so once
  // the starts are fixed they are independent — which is what lets the C-step
  // loop below run concurrently without changing a single result.
  std::vector<std::vector<int>> starts(static_cast<std::size_t>(n_trials));
  {
    std::vector<char> used(n, 0);
    for (int trial = 0; trial < n_trials; ++trial) {
      std::fill(used.begin(), used.end(), 0);
      std::vector<int>& s = starts[static_cast<std::size_t>(trial)];
      s.reserve(static_cast<std::size_t>(d) + 1);
      while (static_cast<int>(s.size()) < d + 1) {
        const int idx = pick(rng);
        if (!used[idx]) { used[idx] = 1; s.push_back(idx); }
      }
    }
  }

  // 500 independent C-step chains, each O(30 * n * d^2) — the dominant cost of
  // the whole sampled-covariance path (~7 ms per scan match at the deployed
  // cov_samples 30, on one core). They are pure functions of their start, so
  // spreading them across cores is free of numerical consequence.
  std::vector<Estimate> attempts(static_cast<std::size_t>(n_trials));
  std::vector<char> attempt_ok(static_cast<std::size_t>(n_trials), 0);
#pragma omp parallel
  {
    Scratch scratch;
#pragma omp for schedule(static)
    for (int trial = 0; trial < n_trials; ++trial) {
      Estimate est;
      est.support = starts[static_cast<std::size_t>(trial)];
      bool ok = true;
      double prev_det = std::numeric_limits<double>::infinity();
      for (int step = 0; step < 30 && ok; ++step) {
        ok = c_step(X, h, est, scratch);
        if (!ok) break;
        // RELATIVE tolerance: det(cov) of the deployed input (30 ICP
        // (x, y, theta) samples at sigma ~1e-2 m / 5e-3 rad) is itself
        // ~1e-12, so an absolute 1e-12 test declared convergence after the
        // first C-step on essentially every chain — the "converged FAST-MCD
        // chains" were near-elemental estimates.
        if (std::abs(prev_det - est.det) <
            1e-6 * std::max(std::abs(prev_det), std::abs(est.det)))
          break;  // converged
        prev_det = est.det;
      }
      if (ok) {
        attempt_ok[static_cast<std::size_t>(trial)] = 1;
        attempts[static_cast<std::size_t>(trial)] = std::move(est);
      }
    }
  }

  // Reduce IN TRIAL ORDER with the sequential rule (strict <, first winner
  // keeps ties), so the surviving estimate is exactly the one the serial loop
  // would have kept regardless of how the workers interleaved.
  Estimate best;
  for (int trial = 0; trial < n_trials; ++trial) {
    const std::size_t t = static_cast<std::size_t>(trial);
    // det > 0: the scatter is PSD by construction, so a NEGATIVE determinant
    // is numerical noise from a singular support set — and being tiny it
    // would otherwise beat every legitimate positive determinant.
    if (attempt_ok[t] && attempts[t].det > 0.0 && attempts[t].det < best.det)
      best = attempts[t];
  }

  if (!std::isfinite(best.det)) return result;

  Scratch scratch;
  std::vector<double> d2;

  // consistency correction (sklearn correct_covariance): scale so the
  // estimator is consistent at the normal model
  Eigen::FullPivLU<Eigen::MatrixXd> lu(best.cov);
  if (!lu.isInvertible()) return result;
  mahalanobis_sq(X, best.mu, lu.inverse(), scratch, d2);
  std::vector<double> d2_sorted(d2);
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
  mahalanobis_sq(X, best.mu, lu2.inverse(), scratch, d2);

  std::vector<int> keep;
  for (int i = 0; i < n; ++i)
    if (d2[static_cast<std::size_t>(i)] <= CHI2_975_3) keep.push_back(i);
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
