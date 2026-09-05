#include "sonar_slam_cpp/slam_core.hpp"

#include <cmath>
#include <cstdio>
#include <random>

int main()
{
  using sonar_slam::Keyframe;
  using sonar_slam::Matrix;
  using sonar_slam::Slam;
  int failures = 0;
  const auto check = [&](bool ok, const char* label) {
    if (!ok) { std::printf("FAIL: %s\n", label); ++failures; }
  };
  // A half-turn of the target chart must not change the local covariance or
  // turn a compact cluster near pi into a mean heading near zero.
  std::mt19937 rng(19);
  std::uniform_real_distribution<float> point(-2.0f, 2.0f);
  Matrix source(80, 3);
  for (int i = 0; i < source.rows(); ++i)
    for (int c = 0; c < 3; ++c) source(i, c) = point(rng);
  std::vector<gtsam::Pose2> guesses;
  std::uniform_real_distribution<double> shift(-0.15, 0.15);
  for (int i = 0; i < 40; ++i) {
    const double x = shift(rng), y = shift(rng), yaw = shift(rng);
    guesses.emplace_back(x, y, yaw);
  }
  const gtsam::Pose2 half_turn(0.0, 0.0, M_PI);
  const Matrix target = Keyframe::transform_points(source, half_turn);
  std::vector<gtsam::Pose2> turned_guesses;
  for (const auto& g : guesses) turned_guesses.push_back(half_turn.compose(g));
  for (bool parallel : {false, true}) {
    Slam slam;
    slam.parallel_cov_samples = parallel;
    // Stop after one update to retain a measurable spread of ICP solutions.
    slam.constrained_icp_params.max_iterations = 1;
    const auto regular = slam.compute_icp_with_cov(source, source, guesses);
    const auto turned = slam.compute_icp_with_cov(source, target, turned_guesses);
    check(regular.message == "success" && turned.message == "success",
          "covariance fixture must solve in both charts");
    if (regular.message != "success" || turned.message != "success") continue;
    const auto error = half_turn.compose(regular.odom).between(turned.odom);
    std::printf("parallel=%d: half-turn error %.6f rad, yaw variances %.6g / %.6g\n",
                parallel, error.theta(), regular.cov_raw(2, 2), turned.cov_raw(2, 2));
    check(std::abs(error.theta()) < 1e-4 && error.translation().norm() < 1e-4,
          "sampled mean must respect yaw wrapping");
    check((regular.cov_raw - turned.cov_raw).norm() < 1e-6,
          "local covariance must be invariant to a half-turn of the target");
  }

  const auto rejected = [](Slam& s) {
    try { s.configure(); } catch (const std::invalid_argument&) { return true; }
    return false;
  };
  Slam s;
  s.ssm_params.cov_samples = 30;
  s.ssm_params.initialization = false;
  check(rejected(s), "required covariance without initialization must be rejected");
  s.ssm_params.initialization = true;
  s.ssm_params.cov_samples = 4;
  check(rejected(s), "sampled covariance needs at least five registrations");
  s.ssm_params.cov_samples = 30;
  check(!rejected(s), "valid sampled covariance configuration");
  s.odom_sigmas[0] = -0.1;
  check(rejected(s), "negative noise sigma must be rejected");
  return failures ? 1 : 0;
}
