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
    slam.constrained_icp_params.allow_iteration_limit = true;
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
    for (bool xyz : {false, true}) {
      slam.constrained_3d = xyz;
      const Matrix cloud = xyz ? source : Matrix(source.leftCols(2));
      const auto expired = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
      const auto timeout = slam.compute_icp_with_cov(cloud, cloud, guesses, expired);
      check(!timeout.complete && timeout.transforms.empty() && timeout.message != "success",
            "expired sampling cannot return a confident partial population");
      check(timeout.message.find("deadline") != std::string::npos,
            "sampling timeout must be reported explicitly");
      const auto retry = slam.compute_icp_with_cov(cloud, cloud, guesses);
      check(retry.complete, "sampling must recover after an expired deadline");
    }
  }

  // Enough work to interrupt a nonempty population, without asserting an
  // exact elapsed time or completion count on shared CI machines.
  {
    Slam bounded;
    bounded.parallel_cov_samples = false;
    bounded.constrained_icp_params.max_iterations = 1;
    bounded.constrained_icp_params.allow_iteration_limit = true;
    std::vector<gtsam::Pose2> many;
    many.reserve(100000);
    for (int i = 0; i < 100000; ++i) many.push_back(guesses[i % guesses.size()]);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    const auto partial = bounded.compute_icp_with_cov(source, source, many, deadline);
    check(!partial.complete && partial.transforms.size() < many.size() && partial.message != "success",
          "interrupted sampling cannot publish covariance from a partial population");
    check(partial.cov_raw.isZero(), "interrupted sampling must leave measurement covariance unavailable");
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
  s.odom_sigmas[0] = 0.2;
  s.nssm_params.cov_samples = 0;
  check(rejected(s), "required loop covariance must not silently use fixed noise");
  s.nssm_params.cov_samples = 30;
  s.nssm_params.initialization = false;
  check(rejected(s), "required loop covariance must have initialization");
  s.nssm_params.initialization = true;
  check(!rejected(s), "restored loop covariance configuration");
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 100;
  s.keyframes.push_back(std::make_shared<Keyframe>(true, stamp, gtsam::Pose3()));
  auto earlier = stamp;
  earlier.sec = 99;
  earlier.nanosec = 900000000;
  Keyframe out_of_order(true, earlier,
    gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(10, 0, 0)));
  check(!s.is_keyframe(out_of_order), "older input cannot become a graph keyframe");
  out_of_order.time = stamp;
  check(!s.is_keyframe(out_of_order), "duplicate input cannot become a graph keyframe");
  return failures ? 1 : 0;
}
