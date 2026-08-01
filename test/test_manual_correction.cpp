// Empirical check of the operator hand-correction path (Slam::
// add_manual_correction / undo_manual_correction): build a straight DR-only
// chain of keyframes, apply an operator fix displaced from the drifted
// estimate, and verify
//  1. the newest keyframe converges to the fix,
//  2. the correction is distributed backwards through the elastic chain
//     while the anchor prior holds the root,
//  3. undo removes the prior and the trajectory relaxes back,
//  4. the undo stack behaves (second undo refuses),
//  5. removal stays index-stable when later keyframe rounds were committed
//     after the prior (the null-hole property of committed_graph_),
//  6. ordinary keyframe rounds keep succeeding after each step.
#include <cmath>
#include <cstdio>

#include "sonar_slam_cpp/slam_core.hpp"

using sonar_slam::Keyframe;
using sonar_slam::KeyframePtr;
using sonar_slam::Slam;

#define CHECK(cond, ...)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL(%d): ", __LINE__);                                   \
      std::printf(__VA_ARGS__);                                              \
      std::printf("\n");                                                     \
      return 1;                                                              \
    }                                                                        \
  } while (0)

namespace {

KeyframePtr make_frame(int k, double x)
{
  builtin_interfaces::msg::Time t;
  t.sec = k;
  const gtsam::Pose3 dr(gtsam::Rot3::Yaw(0.0), gtsam::Point3(x, 0.0, 1.0));
  auto f = std::make_shared<Keyframe>(true, t, dr);
  f->points = sonar_slam::Matrix(0, 3);
  return f;
}

bool add_keyframe(Slam& slam, int k, double x)
{
  auto f = make_frame(k, x);
  if (slam.keyframes.empty()) {
    slam.add_prior(f);
  } else {
    f->update(slam.current_keyframe()->pose.compose(
      slam.current_keyframe()->dr_pose.between(f->dr_pose)));
    slam.add_odometry(f);
  }
  return slam.update_factor_graph(f);
}

}  // namespace

int main()
{
  Slam slam;
  slam.prior_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);
  slam.odom_sigmas = Eigen::Vector3d(0.2, 0.2, 0.02);
  slam.icp_odom_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);
  // this test drives odometry + manual priors only; say so, or configure()
  // (correctly) rejects the default require_covariance/cov_samples=0 pair
  slam.ssm_params.enable = false;
  slam.configure();

  // keyframes at x = 0..9, 1 m apart, straight line
  for (int k = 0; k < 10; ++k)
    CHECK(add_keyframe(slam, k, k), "keyframe %d update failed: %s", k,
          slam.last_error().c_str());
  const gtsam::Pose2 before = slam.current_keyframe()->pose;
  std::printf("[1] chain built: kf9 at (%.3f, %.3f, %.3f)\n", before.x(),
              before.y(), before.theta());

  // ---- apply: operator says the vehicle is 1.5 m +y and 0.5 m back in x
  const gtsam::Pose2 fix(8.5, 1.5, 0.0);
  CHECK(slam.add_manual_correction(fix), "add_manual_correction failed: %s",
        slam.last_error().c_str());
  const gtsam::Pose2 after = slam.current_keyframe()->pose;
  const double err = std::hypot(after.x() - fix.x(), after.y() - fix.y());
  std::printf("[2] correction: kf9 (%.3f, %.3f) -> (%.3f, %.3f), residual "
              "%.3f m\n",
              before.x(), before.y(), after.x(), after.y(), err);
  // the prior competes with the anchored elastic chain, so the equilibrium is
  // near but not exactly on the fix
  CHECK(err < 0.30, "newest keyframe did not move to the fix (%.3f m)", err);
  CHECK(slam.keyframes[5]->pose.y() > 0.2,
        "correction was not distributed backwards (kf5 y=%.3f)",
        slam.keyframes[5]->pose.y());
  CHECK(std::abs(slam.keyframes[0]->pose.y()) < 0.2,
        "anchor keyframe moved too much (y=%.3f)", slam.keyframes[0]->pose.y());
  CHECK(slam.manual_corrections_pending() == 1, "undo stack depth != 1");

  // ---- undo: the trajectory must relax back to the straight DR optimum
  CHECK(slam.undo_manual_correction(), "undo failed: %s",
        slam.last_error().c_str());
  const gtsam::Pose2 undone = slam.current_keyframe()->pose;
  std::printf("[3] undo: kf9 back at (%.3f, %.3f)\n", undone.x(), undone.y());
  CHECK(std::hypot(undone.x() - before.x(), undone.y() - before.y()) < 0.10,
        "kf9 did not relax back (at %.3f, %.3f)", undone.x(), undone.y());
  CHECK(std::abs(slam.keyframes[5]->pose.y()) < 0.10,
        "mid-chain did not relax back (kf5 y=%.3f)", slam.keyframes[5]->pose.y());
  CHECK(slam.manual_corrections_pending() == 0, "undo stack depth != 0");

  // ---- empty stack refuses
  CHECK(!slam.undo_manual_correction(), "undo on empty stack returned true");
  std::printf("[4] second undo refused: %s\n", slam.last_error().c_str());

  // ---- index stability: apply again, commit a LATER keyframe round on top,
  // then undo — the recorded factor index must still remove the right prior
  CHECK(slam.add_manual_correction(fix), "re-apply failed: %s",
        slam.last_error().c_str());
  CHECK(add_keyframe(slam, 10, 10.0), "post-correction keyframe failed: %s",
        slam.last_error().c_str());
  CHECK(slam.current_keyframe()->pose.y() > 1.0,
        "kf10 should chain off the corrected kf9 (y=%.3f)",
        slam.current_keyframe()->pose.y());
  CHECK(slam.undo_manual_correction(), "undo under later rounds failed: %s",
        slam.last_error().c_str());
  std::printf("[5] undo with a later committed round: kf10 at (%.3f, %.3f)\n",
              slam.current_keyframe()->pose.x(),
              slam.current_keyframe()->pose.y());
  CHECK(std::abs(slam.current_keyframe()->pose.y()) < 0.10,
        "kf10 did not relax back (y=%.3f)", slam.current_keyframe()->pose.y());

  // ---- life goes on after the undo's rebuild
  CHECK(add_keyframe(slam, 11, 11.0), "post-undo keyframe failed: %s",
        slam.last_error().c_str());
  std::printf("[6] post-undo keyframe OK: kf11 at (%.3f, %.3f)\n",
              slam.current_keyframe()->pose.x(),
              slam.current_keyframe()->pose.y());

  std::printf("PASS\n");
  return 0;
}
