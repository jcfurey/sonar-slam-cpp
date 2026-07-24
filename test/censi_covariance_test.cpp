// Standalone validation of the retained point-to-point Censi (2007)
// covariance math. The runtime cov_method is deliberately disabled because
// the shipped ICP is point-to-plane; this test protects the isolated helper
// while a matching point-to-plane covariance implementation is pending. The
// estimator tested here is
//     cov = 2 * sensor_var * (Sum_i J_i^T J_i)^-1
// for 2D point-to-point registration. This test Monte-Carlo checks it: it
// draws known-correspondence rigid fits under per-point Gaussian noise and
// confirms the predicted covariance matches the empirical spread.
//
// Eigen-only, no ROS/GTSAM/CUDA — build and run with:
//   g++ -std=c++17 -O2 -I include -I /usr/include/eigen3
//       test/censi_covariance_test.cpp -o /tmp/censi_test && /tmp/censi_test
#include "sonar_slam_cpp/icp_covariance_math.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <random>

using sonar_slam::Matrix;

int main()
{
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> U(0.0, 8.0);
  const int N = 250;
  const double tx = 0.4, ty = -0.3, th = 0.15, sigma = 0.05;
  const double c = std::cos(th), s = std::sin(th);
  Eigen::Matrix2d R;
  R << c, -s, s, c;
  const Eigen::Vector2d t(tx, ty);

  // target cloud q and the nominal source p with R p + t = q exactly
  Eigen::MatrixXd q(N, 2), p(N, 2);
  for (int i = 0; i < N; ++i) {
    q(i, 0) = U(rng);
    q(i, 1) = U(rng);
    const Eigen::Vector2d pi = R.transpose() * (q.row(i).transpose() - t);
    p(i, 0) = pi[0];
    p(i, 1) = pi[1];
  }

  // predicted covariance at the nominal geometry
  const Matrix src_nom = p.cast<float>();
  Eigen::Matrix3d pred;
  if (!sonar_slam::censi_covariance(src_nom, th, sigma * sigma, pred)) {
    std::printf("censi_covariance reported a singular information matrix\n");
    return 1;
  }

  // Monte Carlo: noisy p, q -> known-correspondence rigid fit -> (tx, ty, theta)
  std::normal_distribution<double> G(0.0, sigma);
  const int T = 8000;
  Eigen::MatrixXd est(T, 3);
  for (int trial = 0; trial < T; ++trial) {
    Eigen::MatrixXd pn(N, 2), qn(N, 2);
    for (int i = 0; i < N; ++i) {
      pn(i, 0) = p(i, 0) + G(rng);
      pn(i, 1) = p(i, 1) + G(rng);
      qn(i, 0) = q(i, 0) + G(rng);
      qn(i, 1) = q(i, 1) + G(rng);
    }
    const Eigen::RowVector2d pbar = pn.colwise().mean(), qbar = qn.colwise().mean();
    const Eigen::Matrix2d H =
      (pn.rowwise() - pbar).transpose() * (qn.rowwise() - qbar);
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d D = Eigen::Matrix2d::Identity();
    D(1, 1) = (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0 ? -1 : 1;
    const Eigen::Matrix2d Rhat = svd.matrixV() * D * svd.matrixU().transpose();
    const Eigen::Vector2d that = qbar.transpose() - Rhat * pbar.transpose();
    est(trial, 0) = that[0];
    est(trial, 1) = that[1];
    est(trial, 2) = std::atan2(Rhat(1, 0), Rhat(0, 0));
  }
  const Eigen::RowVector3d mean = est.colwise().mean();
  const Eigen::Matrix3d emp =
    (est.rowwise() - mean).transpose() * (est.rowwise() - mean) / (T - 1);

  std::printf("            predicted      empirical     ratio\n");
  const char* nm[3] = {"var(x)   ", "var(y)   ", "var(theta)"};
  int ok = 1;
  for (int i = 0; i < 3; ++i) {
    const double r = pred(i, i) / emp(i, i);
    std::printf("%s  %.6e  %.6e   %.3f\n", nm[i], pred(i, i), emp(i, i), r);
    if (r < 0.85 || r > 1.15) ok = 0;
  }
  std::printf("cov(x,y)     %+.3e     %+.3e\n", pred(0, 1), emp(0, 1));
  std::printf("%s\n", ok ? "PASS (diagonals within 15%)" : "FAIL");
  return ok ? 0 : 1;
}
