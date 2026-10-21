#include "sonar_slam_cpp/keyframe.hpp"
#include <cstdio>
#include <limits>

int main()
{
  using namespace sonar_slam;
  const auto reject = [](const Eigen::Matrix3d& covariance) {
    const auto out = publication_covariance(covariance, true);
    return !valid_pose_covariance(covariance) &&
      out.isApprox(Eigen::Matrix3d::Identity() * 1e6);
  };
  Eigen::Matrix3d good = Eigen::Vector3d(0.01, 0.04, 0.0001).asDiagonal();
  auto bad = good;
  bad(0, 1) = bad(1, 0) = 1.0;
  if (!reject(bad) || !reject(Eigen::Matrix3d::Zero())) return 1;
  bad = good;
  bad(1, 1) = std::numeric_limits<double>::quiet_NaN();
  if (!reject(bad)) return 1;
  bad = good;
  bad(0, 1) = 0.001;
  if (!reject(bad)) return 1;
  if (!publication_covariance(good, false).isApprox(Eigen::Matrix3d::Identity() * 1e6)) return 1;
  for (double scale : {1e-200, 1e200}) {
    Eigen::Matrix3d indefinite = Eigen::Matrix3d::Identity() * scale;
    indefinite(0, 1) = indefinite(1, 0) = 2 * scale;
    Eigen::Matrix3d asymmetric = Eigen::Matrix3d::Identity() * scale;
    asymmetric(0, 1) = scale;
    if (!reject(indefinite) || !reject(asymmetric)) {
      std::puts("FAIL: covariance validation changed with finite numeric scale");
      return 1;
    }
  }
  const Eigen::Matrix3d large = Eigen::Matrix3d::Identity() * 1e308;
  const auto large_published = publication_covariance(large, true);
  if (!large_published.allFinite() || !valid_pose_covariance(large_published) ||
      (large_published / 1e308 - Eigen::Matrix3d::Identity()).norm() > 1e-12) {
    std::puts("FAIL: valid large covariance overflowed during publication");
    return 1;
  }
  const auto published = publication_covariance(good, true);
  if (!valid_pose_covariance(published) || published(2, 2) < 1e-3 ||
      std::abs(published(0, 0) - good(0, 0)) > 1e-12) return 1;

  builtin_interfaces::msg::Time time;
  time.sec = 100;
  Keyframe keyframe(true, time, gtsam::Pose3());
  keyframe.set_cov(good);
  if (!keyframe.has_cov) return 1;
  keyframe.set_cov(bad);
  if (keyframe.has_cov || !keyframe.transf_cov.isZero()) return 1;
  keyframe.update(gtsam::Pose2(), good);
  if (!keyframe.has_cov) return 1;
  keyframe.update(gtsam::Pose3(), bad);
  if (keyframe.has_cov) return 1;

  const Eigen::Matrix3d heading_covariance = Eigen::Vector3d(.01, .01, .04).asDiagonal();
  const auto extrapolated = propagate_planar_covariance(heading_covariance, Eigen::Vector2d(5, 0),
    0, Eigen::Matrix3d::Zero());
  if (std::abs(extrapolated(1, 1) - 1.01) > 1e-12 ||
      std::abs(extrapolated(1, 2) - .2) > 1e-12) {
    std::puts("FAIL: heading uncertainty did not propagate through the extrapolation lever arm");
    return 1;
  }

  // Differentiate pose composition independently, with fixed-map parent
  // errors and right-tangent odometry errors, including correlated axes.
  const gtsam::Pose2 parent(1, -2, .6), motion(1.3, -.7, -.4);
  const auto nominal = parent.compose(motion);
  Eigen::Matrix3d factor;
  factor << .2, .03, .04, 0, .4, -.01, .02, .01, .1;
  const Eigen::Matrix3d parent_covariance = factor * factor.transpose();
  factor << .03, .01, .02, .01, .05, 0, -.005, .01, .02;
  const Eigen::Matrix3d motion_covariance = factor * factor.transpose();
  Eigen::Matrix3d parent_jacobian, motion_jacobian;
  const auto residual = [&](const gtsam::Pose2& value) -> Eigen::Vector3d {
    return {value.x() - nominal.x(), value.y() - nominal.y(),
      std::remainder(value.theta() - nominal.theta(), 2 * M_PI)};
  };
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero(); delta[axis] = 1e-6;
    parent_jacobian.col(axis) = residual(gtsam::Pose2(parent.x() + delta[0], parent.y() + delta[1],
      parent.theta() + delta[2]).compose(motion)) / 1e-6;
    motion_jacobian.col(axis) = residual(parent.compose(motion.compose(gtsam::Pose2::Expmap(delta)))) / 1e-6;
  }
  const Eigen::Matrix3d numerical = parent_jacobian * parent_covariance * parent_jacobian.transpose() +
    motion_jacobian * motion_covariance * motion_jacobian.transpose();
  const auto propagated = propagate_planar_covariance(parent_covariance,
    nominal.translation() - parent.translation(), nominal.theta(), motion_covariance);
  if (!valid_pose_covariance(propagated) || !propagated.isApprox(numerical, 1e-5)) {
    std::puts("FAIL: planar covariance propagation disagrees with pose-composition derivatives");
    return 1;
  }
  std::puts("PASS: covariance failure behavior and planar extrapolation transport");
}
