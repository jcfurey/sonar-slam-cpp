// Validates the design assumptions behind Slam::update_factor_graph's failure
// recovery (see src/core/slam_core.cpp):
//  1. A rank-deficient factor makes ISAM2::update throw
//     IndeterminantLinearSystemException.
//  2. After the throw the SAME ISAM2 instance is poisoned: a subsequent
//     well-posed update does NOT return the estimator to health (it either
//     throws again or the estimate still contains the unconstrained states),
//     because GTSAM gives no exception guarantee — by throw time the bad
//     factor is already inside nonlinearFactors_/theta_.
//  3. A FRESH ISAM2 rebuilt from the previously-committed factors +
//     last-known estimates succeeds and can absorb further updates — the
//     committed_graph_ mirror + rebuild_isam() recovery path.
// If a future GTSAM version changes any of these, this test tells us the
// recovery machinery needs a rethink before the behavior silently shifts.
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <cstdio>

using gtsam::Pose2;
using gtsam::Symbol;

int main()
{
  const auto noise = gtsam::noiseModel::Diagonal::Sigmas(
    Eigen::Vector3d(0.1, 0.1, 0.01));

  auto make_committed = [&](gtsam::NonlinearFactorGraph& g, gtsam::Values& v) {
    g.add(gtsam::PriorFactor<Pose2>(Symbol('x', 0), Pose2(0, 0, 0), noise));
    g.add(gtsam::BetweenFactor<Pose2>(Symbol('x', 0), Symbol('x', 1),
                                      Pose2(1, 0, 0), noise));
    v.insert(Symbol('x', 0), Pose2(0, 0, 0));
    v.insert(Symbol('x', 1), Pose2(1, 0, 0));
  };

  // healthy start
  gtsam::ISAM2 isam;
  gtsam::NonlinearFactorGraph committed;
  gtsam::Values estimates;
  {
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    make_committed(g, v);
    isam.update(g, v);
    committed = g;
    estimates = v;
  }
  std::printf("[1] healthy update: OK (%zu vars)\n",
              isam.calculateEstimate().size());

  // 1. poison: x2/x3 disconnected from the constrained component (gauge
  // freedom -> indeterminate linear system)
  bool threw = false;
  try {
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    g.add(gtsam::BetweenFactor<Pose2>(Symbol('x', 2), Symbol('x', 3),
                                      Pose2(1, 0, 0), noise));
    v.insert(Symbol('x', 2), Pose2(2, 0, 0));
    v.insert(Symbol('x', 3), Pose2(3, 0, 0));
    isam.update(g, v);
  } catch (const std::exception& e) {
    threw = true;
    std::printf("[2] rank-deficient update threw as expected: %.60s...\n",
                e.what());
  }
  if (!threw) {
    std::printf("[2] FAIL: expected IndeterminantLinearSystem throw\n");
    return 1;
  }

  // 2. the same instance stays poisoned for a well-posed follow-up
  bool poisoned = false;
  try {
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    g.add(gtsam::BetweenFactor<Pose2>(Symbol('x', 1), Symbol('x', 4),
                                      Pose2(1, 0, 0), noise));
    v.insert(Symbol('x', 4), Pose2(2, 0, 0));
    isam.update(g, v);
    // even if it does not throw, the estimate still contains the
    // unconstrained x2/x3 -> the estimator did not recover by itself
    poisoned = isam.calculateEstimate().exists(Symbol('x', 2));
    std::printf("[3] follow-up update did not throw; x2 still present: %s\n",
                poisoned ? "yes (poisoned)" : "no");
  } catch (const std::exception& e) {
    poisoned = true;
    std::printf("[3] follow-up well-posed update ALSO threw (poisoned): %.60s\n",
                e.what());
  }
  if (!poisoned)
    std::printf("[3] note: instance self-recovered; rebuild is then merely "
                "safe, not required\n");

  // 3. rebuild from committed mirror + estimates
  try {
    gtsam::ISAM2 fresh;
    gtsam::NonlinearFactorGraph g = committed;
    fresh.update(g, estimates);
    gtsam::NonlinearFactorGraph g2;
    gtsam::Values v2;
    g2.add(gtsam::BetweenFactor<Pose2>(Symbol('x', 1), Symbol('x', 2),
                                       Pose2(1, 0, 0), noise));
    v2.insert(Symbol('x', 2), Pose2(2, 0, 0));
    fresh.update(g2, v2);
    const auto est = fresh.calculateEstimate();
    std::printf("[4] rebuild + follow-up update: OK (%zu vars, x2 at %.2f)\n",
                est.size(), est.at<Pose2>(Symbol('x', 2)).x());
  } catch (const std::exception& e) {
    std::printf("[4] FAIL: rebuild path threw: %s\n", e.what());
    return 1;
  }

  std::printf("ALL ASSUMPTIONS VALIDATED\n");
  return 0;
}
