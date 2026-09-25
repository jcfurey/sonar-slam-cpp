// Loop-closure harness: the scan-match acceptance checks against scenes with
// known ground truth, under the deployed profile (see loop_closure_harness.hpp).
// Each case states the behaviour a correct check must have; the printed
// measurements show how close the current implementation comes.
//
// KNOWN DEFECTS -- these cases are DISABLED_ until their fix lands; run them
// with --gtest_also_run_disabled_tests to see the current measurements.
//  * DegeneracyGate.CornerViewsPassTheGate: anisotropy divides by a sampled
//    sigma_min floored only at 1e-6 m, so registrations good to 1-2.5 cm are
//    rejected on convergence noise (65% pass). The corridor it must reject
//    has sigma_max 8.6-13 cm.
//  * SequentialMatch.CorrectsSmallDeadReckoningError: the same noise, plus
//    ssm/max_translation measured from the global-init seed rather than dead
//    reckoning, rejects 9 of 12 corner registrations with a 0.1 m DR error.
//  * PcmConsistency.{Translation,Yaw}AliasIsExcluded: PCM adds each source
//    keyframe's ABSOLUTE marginal to the consistency covariance. Under the
//    deployed odometry sigmas those reach ~1.3 m / 0.13 rad within 40
//    keyframes, so a 3 m or 0.35 rad alias scores md 1-3 against 11.34 and is
//    never excluded, even on a short chain.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <deque>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "loop_closure_harness.hpp"
#include "sonar_slam_cpp/icp_covariance_math.hpp"

namespace {

using namespace sonar_slam;
using namespace sonar_slam::harness;

double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// ------------------------------------------------------------------ profile

TEST(DeployedProfile, ConfiguresTheCoreFromTheSettingsFile)
{
  const ProfilePaths paths = locate_profile();
  std::printf("[profile] slam.yaml: %s\n[profile] icp.yaml:  %s\n",
              paths.slam_yaml.c_str(), paths.icp_yaml.c_str());
  RecordProperty("slam_yaml", paths.slam_yaml);
  Slam slam;
  ASSERT_NO_THROW(deployed_profile(slam));
  EXPECT_TRUE(slam.constrained_3d);
  EXPECT_GT(slam.ssm_params.cov_samples, 0);
  EXPECT_GT(slam.nssm_params.cov_samples, 0);
}

// ---------------------------------------------------------- degeneracy gate
//
// Both SSM and NSSM gate a registration on its sampled covariance: too large
// (sigma_max) or too elongated (anisotropy) means the geometry did not pin
// the translation. A correct gate passes views with corners and posts and
// rejects a featureless corridor, whatever the ICP stopping tolerance.

struct GateOutcome
{
  int trials = 0;
  int passed = 0;
  std::vector<double> sigma_max;
  std::vector<double> anisotropy;
};

GateOutcome run_gate(const char* label, const SyntheticWorld& world,
                     const std::vector<gtsam::Pose2>& views,
                     const gtsam::Pose2& step, int seeds_per_view)
{
  GateOutcome out;
  const SonarModel sonar;
  for (std::size_t v = 0; v < views.size(); ++v) {
    const std::size_t first = out.sigma_max.size();
    const int passed_before = out.passed;
    for (int seed = 0; seed < seeds_per_view; ++seed) {
      Slam slam;
      deployed_profile(slam);
      std::mt19937 rng(static_cast<unsigned>(1000 * v + seed));
      const gtsam::Pose2 target_truth = views[v];
      const gtsam::Pose2 source_truth = target_truth.compose(step);
      const Matrix target = downsample(
        scan(world, target_truth, sonar, rng),
        static_cast<float>(slam.point_resolution));
      const Matrix source = downsample(
        scan(world, source_truth, sonar, rng),
        static_cast<float>(slam.point_resolution));
      // The pipeline seeds its covariance ICPs from the best global-init
      // samples, which cluster in the basin around the true alignment.
      const gtsam::Pose2 relative = target_truth.between(source_truth);
      std::uniform_real_distribution<double> dxy(-0.2, 0.2);
      std::uniform_real_distribution<double> dyaw(-0.03, 0.03);
      std::vector<gtsam::Pose2> guesses;
      for (int g = 0; g < slam.ssm_params.cov_samples; ++g)
        guesses.push_back(relative.compose(
          gtsam::Pose2(dxy(rng), dxy(rng), dyaw(rng))));
      const auto result = slam.compute_icp_with_cov(source, target, guesses);
      ++out.trials;
      if (result.n_samples == 0) continue;  // registration failed outright
      const IcpObservability obs = assess_icp_observability(
        slam.ssm_degeneracy_prefloor ? result.cov_raw : result.cov,
        slam.ssm_max_sigma, slam.ssm_max_anisotropy);
      out.sigma_max.push_back(obs.sigma_max);
      out.anisotropy.push_back(obs.anisotropy);
      if (obs.observable) ++out.passed;
    }
    const std::vector<double> sigmas(out.sigma_max.begin() + first,
                                     out.sigma_max.end());
    const std::vector<double> ratios(out.anisotropy.begin() + first,
                                     out.anisotropy.end());
    std::printf("  [%s] view (%.1f, %.1f, %.2f): passed %d/%d, sigma_max "
                "%.4f..%.4f m, anisotropy %.1f..%.1f\n",
                label, views[v].x(), views[v].y(), views[v].theta(),
                out.passed - passed_before, seeds_per_view,
                sigmas.empty() ? 0.0 : *std::min_element(sigmas.begin(), sigmas.end()),
                sigmas.empty() ? 0.0 : *std::max_element(sigmas.begin(), sigmas.end()),
                ratios.empty() ? 0.0 : *std::min_element(ratios.begin(), ratios.end()),
                ratios.empty() ? 0.0 : *std::max_element(ratios.begin(), ratios.end()));
  }
  return out;
}

// Views facing a pool corner from 3-5 m: two walls meeting at a right angle
// (plus nearby posts) pin both translation axes. There is no geometric reason
// for any of these to fail the gate.
std::vector<gtsam::Pose2> corner_views()
{
  return {{4.0, 3.0, std::atan2(-3.0, -4.0)},
          {16.0, 3.0, std::atan2(-3.0, 4.0)},
          {16.0, 7.0, std::atan2(3.0, 4.0)},
          {4.0, 7.0, std::atan2(3.0, -4.0)}};
}

TEST(DegeneracyGate, DISABLED_CornerViewsPassTheGate)
{
  const SyntheticWorld world = SyntheticWorld::pool(20.0, 10.0);
  const GateOutcome out = run_gate(
    "corner", world, corner_views(), gtsam::Pose2(0.75, 0.0, 0.05), 5);
  const double rate = static_cast<double>(out.passed) / out.trials;
  std::printf("[gate/corner] passed %d/%d (%.0f%%), median sigma_max "
              "%.4f m, median anisotropy %.1f\n",
              out.passed, out.trials, 100.0 * rate, median(out.sigma_max),
              median(out.anisotropy));
  RecordProperty("pass_rate", std::to_string(rate));
  EXPECT_GE(rate, 0.9)
    << "corner registrations fail the degeneracy gate";
}

TEST(DegeneracyGate, FeaturelessCorridorIsRejected)
{
  const SyntheticWorld world = corridor(60.0, 3.0);
  const std::vector<gtsam::Pose2> views = {
    {-6.0, 1.5, 0.0}, {0.0, 1.5, 0.05}, {6.0, 1.5, -0.05},
    {-3.0, 1.2, M_PI}, {3.0, 1.8, M_PI - 0.05}, {9.0, 1.5, 0.1}};
  const GateOutcome out =
    run_gate("corridor", world, views, gtsam::Pose2(0.75, 0.0, 0.0), 5);
  const double rate = static_cast<double>(out.trials - out.passed) / out.trials;
  std::printf("[gate/corridor] rejected %d/%d (%.0f%%), median sigma_max "
              "%.4f m, median anisotropy %.1f\n",
              out.trials - out.passed, out.trials, 100.0 * rate,
              median(out.sigma_max), median(out.anisotropy));
  RecordProperty("reject_rate", std::to_string(rate));
  EXPECT_GE(rate, 0.9)
    << "sliding along a featureless corridor passes the degeneracy gate";
}

// ------------------------------------------------ sequential (SSM) bound
//
// ssm/max_translation is documented as the largest deviation from odometry a
// sequential registration may impose between keyframes. Over one keyframe
// step a DVL chain is far better than that, so a registration further from
// dead reckoning is more likely an alias than a correction.

struct SequentialOutcome
{
  double deviation_from_dr = 0.0;   // m, graph relative vs DR relative
  double deviation_from_truth = 0.0;
  int accepted = 0;
  std::string status;
};

SequentialOutcome run_sequential(const gtsam::Pose2& truth0,
                                 const gtsam::Pose2& dr_error, unsigned seed)
{
  Slam slam;
  deployed_profile(slam);
  const SyntheticWorld world = SyntheticWorld::pool(20.0, 10.0);
  const SonarModel sonar;
  std::mt19937 rng(seed);
  const gtsam::Pose2 truth1 = truth0.compose(gtsam::Pose2(0.75, 0.0, 0.05));
  const gtsam::Pose2 dr1 = truth1.compose(dr_error);
  const float resolution = static_cast<float>(slam.point_resolution);

  auto first = make_keyframe(
    0, truth0, downsample(scan(world, truth0, sonar, rng), resolution));
  slam.add_prior(first);
  if (!slam.update_factor_graph(first)) throw std::runtime_error(slam.last_error());
  auto second = make_keyframe(
    1, dr1, downsample(scan(world, truth1, sonar, rng), resolution));
  predict_from_dr(slam, second);
  slam.add_sequential_scan_matching(second);
  if (!slam.update_factor_graph(second)) throw std::runtime_error(slam.last_error());

  const gtsam::Pose2 graph_relative =
    slam.keyframes[0]->pose.between(slam.keyframes[1]->pose);
  SequentialOutcome out;
  out.deviation_from_dr =
    truth0.between(dr1).between(graph_relative).translation().norm();
  out.deviation_from_truth =
    truth0.between(truth1).between(graph_relative).translation().norm();
  out.accepted = slam.ssm_accepted;
  out.status = slam.last_ssm_status;
  return out;
}

TEST(SequentialMatch, DISABLED_CorrectsSmallDeadReckoningError)
{
  int trials = 0, accepted = 0, accurate = 0;
  for (const gtsam::Pose2& view : corner_views()) {
    for (unsigned seed = 0; seed < 3; ++seed) {
      const auto out =
        run_sequential(view, gtsam::Pose2(0.1, -0.05, 0.0), 11 + seed);
      ++trials;
      if (out.accepted) ++accepted;
      if (out.accepted && out.deviation_from_truth < 0.05) ++accurate;
      if (!out.accepted)
        std::printf("  [ssm/small] view (%.1f, %.1f) seed %u rejected: %s\n",
                    view.x(), view.y(), seed, out.status.c_str());
    }
  }
  std::printf("[ssm/small DR error] accepted %d/%d, within 5 cm of truth "
              "%d/%d\n", accepted, trials, accurate, trials);
  EXPECT_GE(accepted, trials - 1)
    << "well-constrained sequential registrations were rejected";
  EXPECT_EQ(accurate, accepted);
}

TEST(SequentialMatch, StaysWithinMaxTranslationOfDeadReckoning)
{
  Slam reference;
  deployed_profile(reference);
  const double bound = reference.ssm_params.max_translation;
  // DR claims the vehicle moved 0.9 m further than it did: well inside the
  // global-init window (5 sigma of odometry), far outside ssm/max_translation.
  int trials = 0, accepted = 0, violations = 0;
  double worst = 0.0;
  for (const gtsam::Pose2& view : corner_views()) {
    for (unsigned seed = 0; seed < 3; ++seed) {
      const auto out =
        run_sequential(view, gtsam::Pose2(0.9, 0.0, 0.0), 11 + seed);
      ++trials;
      if (!out.accepted) continue;
      ++accepted;
      worst = std::max(worst, out.deviation_from_dr);
      if (out.deviation_from_dr > bound + 0.05) ++violations;
    }
  }
  std::printf("[ssm/large DR error] accepted %d/%d; %d moved the graph beyond "
              "the %.2f m bound (worst %.3f m from DR)\n",
              accepted, trials, violations, bound, worst);
  EXPECT_EQ(violations, 0)
    << "accepted sequential links moved the graph further from dead "
       "reckoning than ssm/max_translation";
}

// --------------------------------------------------------------------- PCM
//
// Consecutive closures from one revisit must agree with each other once the
// graph's relative pose between their keyframes is accounted for. A correct
// consistency test accepts honest closures at any revisit heading and chain
// length, and excludes an aliased closure whose error exceeds the relative
// uncertainty between the keyframes involved -- which stays small for
// neighbouring keyframes however long the mission has run.

struct Revisit
{
  std::vector<gtsam::Pose2> truth;
  std::vector<int> first_pass;
  std::vector<int> revisit;
};

// First pass east along y = 0, an excursion of `excursion_steps`, then a
// revisit crossing the first pass at (4.5, 0) with heading `heading`.
Revisit make_revisit(double heading, int excursion_steps, double step)
{
  Revisit r;
  r.truth.emplace_back(0.0, 0.0, 0.0);
  r.first_pass.push_back(0);
  extend_leg(r.truth, gtsam::Point2(9.0, 0.0), step);
  for (int k = 1; k < static_cast<int>(r.truth.size()); ++k)
    r.first_pass.push_back(k);
  const gtsam::Point2 crossing(4.5, 0.0);
  const gtsam::Point2 direction(std::cos(heading), std::sin(heading));
  const gtsam::Point2 start = crossing - 3.0 * direction;
  const gtsam::Point2 end = crossing + 3.0 * direction;
  const double reach = std::max(2.0, excursion_steps * step / 2.0);
  extend_leg(r.truth, gtsam::Point2(9.0 + 0.5 * reach, reach), step);
  extend_leg(r.truth, start, step);
  const int before = static_cast<int>(r.truth.size());
  extend_leg(r.truth, end, step);
  for (int k = before; k < static_cast<int>(r.truth.size()); ++k)
    r.revisit.push_back(k);
  return r;
}

int nearest(const Revisit& r, const std::vector<int>& candidates, int key)
{
  int best = candidates.front();
  double best_distance = 1e9;
  for (const int c : candidates) {
    const double d =
      (r.truth[static_cast<std::size_t>(c)].translation() -
       r.truth[static_cast<std::size_t>(key)].translation()).norm();
    if (d < best_distance) {
      best_distance = d;
      best = c;
    }
  }
  return best;
}

struct PcmOutcome
{
  int trials = 0;
  int honest_accepted = 0;
  int alias_excluded = 0;
  std::vector<double> alias_min_md;
};

// Builds a fresh odometry chain per trial, queues three honest closures from
// consecutive revisit keyframes, and optionally one aliased closure from the
// next revisit keyframe.
PcmOutcome run_pcm(double heading, int excursion_steps,
                   const gtsam::Pose2* alias, int trials)
{
  PcmOutcome out;
  for (int trial = 0; trial < trials; ++trial) {
    Slam slam;
    deployed_profile(slam);
    std::mt19937 rng(static_cast<unsigned>(77 + trial));
    const Revisit r =
      make_revisit(heading, excursion_steps, slam.keyframe_translation);
    DriftModel drift;
    drift.yaw_bias_per_step = 0.0015;
    build_odometry_chain(slam, dead_reckon(r.truth, drift, rng));

    const Eigen::Matrix3d icp_cov =
      slam.icp_odom_sigmas.array().square().matrix().asDiagonal();
    const std::size_t middle = r.revisit.size() / 2;
    std::deque<ICPResult> queue;
    for (std::size_t i = middle - 1; i <= middle + 1; ++i) {
      const int source = r.revisit[i];
      const int target = nearest(r, r.first_pass, source);
      const gtsam::Pose2 measured = perturb(
        r.truth[static_cast<std::size_t>(target)].between(
          r.truth[static_cast<std::size_t>(source)]),
        0.05, 0.005, rng);
      queue.push_back(make_closure(slam, target, source, measured, icp_cov));
    }
    if (alias) {
      const int source = r.revisit[middle + 2];
      const int target = nearest(r, r.first_pass, source);
      const gtsam::Pose2 measured =
        r.truth[static_cast<std::size_t>(target)]
          .between(r.truth[static_cast<std::size_t>(source)])
          .compose(*alias);
      queue.push_back(make_closure(slam, target, source, measured, icp_cov));
    }
    const std::vector<int> clique = slam.verify_pcm(queue, slam.min_pcm);
    ++out.trials;
    const auto contains = [&](int index) {
      return std::find(clique.begin(), clique.end(), index) != clique.end();
    };
    if (contains(0) && contains(1) && contains(2)) ++out.honest_accepted;
    if (alias) {
      if (!contains(3)) ++out.alias_excluded;
      // smallest distance between the alias and any honest closure
      std::deque<ICPResult> pair{queue[0], queue[3]};
      slam.verify_pcm(pair, 2);
      out.alias_min_md.push_back(slam.last_pcm_min_md);
    }
  }
  return out;
}

struct PcmCase
{
  const char* name;
  double heading;
  int excursion_steps;
};

void PrintTo(const PcmCase& c, std::ostream* os)
{
  *os << c.name;
}

class PcmConsistency : public ::testing::TestWithParam<PcmCase>
{
};

TEST_P(PcmConsistency, HonestClosuresFormOneClique)
{
  const PcmCase c = GetParam();
  const PcmOutcome out = run_pcm(c.heading, c.excursion_steps, nullptr, 10);
  std::printf("[pcm/%s honest] accepted %d/%d\n", c.name, out.honest_accepted,
              out.trials);
  EXPECT_GE(out.honest_accepted, out.trials - 1);
}

TEST_P(PcmConsistency, DISABLED_TranslationAliasIsExcluded)
{
  const PcmCase c = GetParam();
  const gtsam::Pose2 alias(3.0, 0.0, 0.0);  // one structure spacing along-track
  const PcmOutcome out = run_pcm(c.heading, c.excursion_steps, &alias, 10);
  std::printf("[pcm/%s 3 m alias] excluded %d/%d, median md to an honest "
              "closure %.2f (threshold 11.34)\n",
              c.name, out.alias_excluded, out.trials, median(out.alias_min_md));
  EXPECT_GE(out.alias_excluded, out.trials - 1);
}

TEST_P(PcmConsistency, DISABLED_YawAliasIsExcluded)
{
  const PcmCase c = GetParam();
  const gtsam::Pose2 alias(0.0, 0.0, 0.35);
  const PcmOutcome out = run_pcm(c.heading, c.excursion_steps, &alias, 10);
  std::printf("[pcm/%s 0.35 rad alias] excluded %d/%d, median md to an "
              "honest closure %.2f (threshold 11.34)\n",
              c.name, out.alias_excluded, out.trials, median(out.alias_min_md));
  EXPECT_GE(out.alias_excluded, out.trials - 1);
}

INSTANTIATE_TEST_SUITE_P(
  Revisits, PcmConsistency,
  ::testing::Values(PcmCase{"same_heading_short", 0.0, 20},
                    PcmCase{"perpendicular_short", M_PI / 2.0, 20},
                    PcmCase{"opposite_short", M_PI, 20},
                    PcmCase{"same_heading_long", 0.0, 300},
                    PcmCase{"perpendicular_long", M_PI / 2.0, 300}),
  [](const ::testing::TestParamInfo<PcmCase>& info) {
    return std::string(info.param.name);
  });

// ------------------------------------------------------------- end to end
//
// The whole front and back end under the deployed profile: a drifted loop in
// a pool with posts must close and pull the trajectory back toward truth.

TEST(EndToEnd, DriftedPoolLoopClosesUnderTheDeployedProfile)
{
  Slam slam;
  deployed_profile(slam);
  const SyntheticWorld world = SyntheticWorld::pool(20.0, 10.0);
  const SonarModel sonar;
  std::mt19937 rng(3);
  std::vector<gtsam::Pose2> truth{gtsam::Pose2(2.0, 2.0, 0.0)};
  for (const gtsam::Point2& waypoint :
       {gtsam::Point2(18.0, 2.0), gtsam::Point2(18.0, 8.0),
        gtsam::Point2(2.0, 8.0), gtsam::Point2(2.0, 2.0),
        gtsam::Point2(10.0, 2.0)})
    extend_leg(truth, waypoint, slam.keyframe_translation);
  DriftModel drift;
  drift.yaw_bias_per_step = 0.004;
  const std::vector<gtsam::Pose2> dr = dead_reckon(truth, drift, rng);
  const float resolution = static_cast<float>(slam.point_resolution);

  for (std::size_t k = 0; k < truth.size(); ++k) {
    auto frame = make_keyframe(
      static_cast<int>(k), dr[k],
      downsample(scan(world, truth[k], sonar, rng), resolution));
    predict_from_dr(slam, frame);
    if (slam.keyframes.empty())
      slam.add_prior(frame);
    else
      slam.add_sequential_scan_matching(frame);
    ASSERT_TRUE(slam.update_factor_graph(frame)) << slam.last_error();
    slam.current_frame = frame;
    if (slam.add_nonsequential_scan_matching()) {
      ASSERT_TRUE(slam.update_factor_graph()) << slam.last_error();
    }
  }

  const double dr_error =
    (dr.back().translation() - truth.back().translation()).norm();
  const double slam_error =
    (slam.current_keyframe()->pose.translation() -
     truth.back().translation()).norm();
  std::printf("[e2e] %zu keyframes, SSM accepted %d (degenerate %d), NSSM "
              "attempts %d queued %d accepted %d reverted %d\n"
              "[e2e] final error: DR %.2f m, SLAM %.2f m; last NSSM: %s\n"
              "[e2e] NSSM rejections: %s\n",
              truth.size(), slam.ssm_accepted, slam.ssm_degenerate_rejected,
              slam.nssm_attempts, slam.nssm_queued, slam.nssm_accepted,
              slam.nssm_reverted, dr_error, slam_error,
              slam.last_nssm_status.c_str(),
              slam.nssm_reject_summary().c_str());
  EXPECT_GE(slam.nssm_accepted, 1) << "no loop closure accepted";
  EXPECT_LT(slam_error, 0.5 * dr_error)
    << "SLAM did not pull the final pose back toward truth";
}

}  // namespace
