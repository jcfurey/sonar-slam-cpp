#include "physical_scene.hpp"
#include "sonar_slam_cpp/slam_core.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <random>

using namespace sonar_slam;

namespace {
void configure(Slam& slam)
{
  slam.input_model = "measured_xyz";
  slam.registration_model = RegistrationModel::SE3;
  slam.point_resolution = .2;
  slam.keyframe_duration = .1; slam.keyframe_translation = .1; slam.keyframe_rotation = .05;
  slam.ssm_params.min_points = 50; slam.ssm_params.min_overlap_ratio = .4;
  slam.ssm_params.max_translation = 2; slam.ssm_params.max_rotation = .5;
  slam.nssm_params.min_points = 50; slam.nssm_params.min_overlap_ratio = .4;
  slam.nssm_params.min_st_sep = 4; slam.nssm_params.min_revisit_sep = 4; slam.nssm_params.source_frames = 1;
  slam.nssm_params.max_translation = 3; slam.nssm_params.max_rotation = .5;
  slam.pcm_queue_size = 4; slam.min_pcm = 2;
  slam.map_calibration_id = "synthetic_mount_v1"; slam.map_depth_datum = "synthetic_depth";
  slam.configure();
}
struct TemporaryMap
{
  std::filesystem::path path = std::filesystem::temp_directory_path() /
    ("sonar_physical_" + std::to_string(std::random_device{}()) + ".ssm");
  ~TemporaryMap() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};
}

TEST(Se3Backend, SonarCorrectsDepthRollPitchWithSoftOdometry)
{
  Slam slam; configure(slam);
  const Matrix scene = test::physical_scene();
  auto first = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), scene);
  slam.add_prior(first); ASSERT_TRUE(slam.update_factor_graph(first));
  const gtsam::Pose3 truth(gtsam::Rot3::RzRyRx(.08, -.06, .1), gtsam::Point3(.3, -.2, .4));
  const gtsam::Pose3 dr(gtsam::Rot3::RzRyRx(.18, -.15, .12), gtsam::Point3(.4, -.3, .9));
  auto next = test::physical_frame(101, truth, dr, scene);
  slam.add_sequential_scan_matching(next);
  ASSERT_TRUE(slam.update_factor_graph(next)) << slam.last_error();
  ASSERT_EQ(slam.ssm_accepted, 1) << slam.last_ssm_status;
  EXPECT_NEAR(next->pose3.z(), truth.z(), .1);
  EXPECT_NEAR(next->pose3.rotation().roll(), truth.rotation().roll(), .025);
  EXPECT_NEAR(next->pose3.rotation().pitch(), truth.rotation().pitch(), .025);
  EXPECT_TRUE(next->has_covariance6);
  EXPECT_TRUE(valid_covariance6(next->covariance6));
  EXPECT_GT((next->pose3.translation() - dr.translation()).norm(), .2);
  const gtsam::Pose3 corrected(gtsam::Rot3::RzRyRx(.25, -.2, .3), gtsam::Point3(.8, .2, 1.4));
  ASSERT_TRUE(slam.add_manual_correction(corrected, Covariance6::Identity() * 1e-5));
  EXPECT_LT(gtsam::Pose3::Logmap(corrected.between(next->pose3)).norm(), .02);
  ASSERT_TRUE(slam.undo_manual_correction());
  EXPECT_NEAR(next->pose3.z(), truth.z(), .1);
}

TEST(Se3Backend, PureAttitudeCreatesPhysicalKeyframesAndWeakGeometryFallsBack)
{
  Slam slam; configure(slam);
  Matrix floor(31 * 31, 3);
  const Matrix scene = test::physical_scene();
  for (int i = 0; i < floor.rows(); ++i) floor.row(i) = scene.row(i * 3 + 2);
  auto first = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), floor);
  slam.add_prior(first); ASSERT_TRUE(slam.update_factor_graph(first));
  const gtsam::Pose3 rotation(gtsam::Rot3::Rx(.2), gtsam::Point3::Zero());
  auto next = test::physical_frame(101, rotation, rotation, floor);
  EXPECT_TRUE(slam.is_keyframe(*next));
  slam.add_sequential_scan_matching(next);
  ASSERT_TRUE(slam.update_factor_graph(next));
  EXPECT_EQ(slam.ssm_accepted, 0);
  EXPECT_EQ(slam.ssm_degenerate_rejected, 1);
  EXPECT_TRUE(next->pose3.equals(rotation, 1e-6));
  Slam imaging;
  imaging.registration_model = RegistrationModel::SE3;
  EXPECT_THROW(imaging.configure(), std::invalid_argument);
}

TEST(Se3Backend, SixAxisPcmClosesAThreeDimensionalDriftingTrajectory)
{
  Slam slam; configure(slam);
  slam.nssm_params.enable = true;
  const Matrix scene = test::physical_scene();
  gtsam::Pose3 last_truth;
  for (int k = 0; k < 14; ++k) {
    const double phase = 2 * M_PI * k / 13;
    last_truth = gtsam::Pose3(gtsam::Rot3::RzRyRx(.08 * std::sin(phase), .06 * std::cos(phase), .1 * std::sin(phase)),
      gtsam::Point3(.8 * (1 - std::cos(phase)), .8 * std::sin(phase), .3 * std::sin(phase)));
    const auto drift = gtsam::Pose3(gtsam::Rot3::RzRyRx(.006 * k, -.004 * k, .004 * k), gtsam::Point3(.03 * k, -.02 * k, .035 * k));
    auto frame = test::physical_frame(100 + k, last_truth, last_truth.compose(drift), scene);
    if (!k) slam.add_prior(frame); else slam.add_odometry(frame);
    ASSERT_TRUE(slam.update_factor_graph(frame)) << slam.last_error();
    if (slam.add_nonsequential_scan_matching()) {
      ASSERT_TRUE(slam.update_factor_graph()) << slam.last_error();
    }
  }
  EXPECT_GE(slam.nssm_accepted, 2) << slam.last_nssm_status;
  EXPECT_GE(slam.nssm_best_clique, 2);
  EXPECT_LT(gtsam::Pose3::Logmap(last_truth.between(slam.keyframes.back()->pose3)).norm(), .2);
}

TEST(Se3Backend, PersistenceRetainsFactorsAndRefusesIncompatibleModels)
{
  const Matrix scene = test::physical_scene();
  Slam original; configure(original);
  auto first = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), scene);
  original.add_prior(first); ASSERT_TRUE(original.update_factor_graph(first));
  auto second = test::physical_frame(101, gtsam::Pose3(),
    gtsam::Pose3(gtsam::Rot3::Rx(.05), gtsam::Point3(0, 0, .2)), scene);
  original.add_sequential_scan_matching(second); ASSERT_TRUE(original.update_factor_graph(second));
  ASSERT_TRUE(original.add_manual_correction(second->pose3, Covariance6::Identity() * .02));
  ASSERT_TRUE(original.add_position_prior(1, .05, -.03, .1));
  TemporaryMap map;
  ASSERT_TRUE(original.save_map(map.path)) << original.last_error();
  Slam restored; configure(restored);
  ASSERT_TRUE(restored.load_map(map.path)) << restored.last_error();
  EXPECT_EQ(restored.loaded_keyframes(), 2);
  EXPECT_EQ(restored.manual_corrections_pending(), 1);
  EXPECT_TRUE(restored.keyframes.back()->covariance6.isApprox(original.keyframes.back()->covariance6, .02));
  EXPECT_TRUE(restored.keyframes.back()->pose3.equals(original.keyframes.back()->pose3, 1e-4));
  Slam wrong; configure(wrong); wrong.map_calibration_id = "other_mount";
  EXPECT_FALSE(wrong.load_map(map.path)); EXPECT_TRUE(wrong.keyframes.empty());
  Slam constrained; constrained.ssm_params.enable = false; constrained.configure();
  EXPECT_FALSE(constrained.load_map(map.path));
  { std::ofstream corrupt(map.path, std::ios::binary | std::ios::app); corrupt.put('x'); }
  Slam corrupt; configure(corrupt);
  EXPECT_FALSE(corrupt.load_map(map.path)); EXPECT_TRUE(corrupt.keyframes.empty());
}

TEST(Se3Backend, RelocalizationRefinesAllAxesAndRequiresNewerConfirmation)
{
  const Matrix scene = test::physical_scene();
  Slam map; configure(map);
  auto reference = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), scene);
  map.add_prior(reference); ASSERT_TRUE(map.update_factor_graph(reference));
  TemporaryMap path;
  ASSERT_TRUE(map.save_map(path.path));
  Slam resumed; configure(resumed);
  ASSERT_TRUE(resumed.load_map(path.path)) << resumed.last_error();
  const gtsam::Pose3 truth(gtsam::Rot3::RzRyRx(.08, -.04, .1), gtsam::Point3(.7, .3, .2));
  const gtsam::Pose3 odometry(gtsam::Rot3::RzRyRx(.13, -.10, .1), gtsam::Point3(0, 0, .65));
  auto observation = test::physical_frame(1000, truth, odometry, scene);
  EXPECT_FALSE(resumed.relocalize(observation));
  EXPECT_EQ(resumed.current_key(), 1);
  EXPECT_FALSE(resumed.relocalize(observation));
  observation->time.nanosec = 100000000;
  ASSERT_TRUE(resumed.relocalize(observation)) << resumed.last_error();
  EXPECT_FALSE(resumed.awaiting_relocalization());
  EXPECT_LT(gtsam::Pose3::Logmap(truth.between(observation->pose3)).norm(), .08);
  RigidRegistration reference_match(scene, resumed.se3_options.icp);
  const auto relative = reference_match.compute(observation->points, truth);
  ASSERT_TRUE(relative.converged);
  const auto expected = compose_covariance(relative.transform, resumed.keyframes.front()->covariance6,
                                           relative.covariance);
  EXPECT_TRUE(observation->covariance6.isApprox(expected, .02));
}

TEST(Se3Backend, RejectedSixAxisLoopRoundRestoresTheCommittedGraph)
{
  Slam slam; configure(slam);
  slam.nssm_params.enable = true;
  slam.post_loop_max_link_error_sigma = 1e-8;
  const Matrix scene = test::physical_scene();
  bool reverted = false;
  for (int k = 0; k < 10 && !reverted; ++k) {
    const gtsam::Pose3 truth(gtsam::Rot3::Rx(.01 * k), gtsam::Point3(.03 * k, 0, .02 * k));
    const auto dr = truth.compose(gtsam::Pose3(gtsam::Rot3::Ry(.008 * k), gtsam::Point3(.02 * k, 0, .03 * k)));
    auto frame = test::physical_frame(100 + k, truth, dr, scene);
    if (!k) slam.add_prior(frame); else slam.add_odometry(frame);
    ASSERT_TRUE(slam.update_factor_graph(frame));
    if (slam.add_nonsequential_scan_matching()) {
      const int before = slam.current_key();
      const auto pose = slam.keyframes.back()->pose3;
      EXPECT_FALSE(slam.update_factor_graph());
      EXPECT_EQ(slam.current_key(), before);
      EXPECT_TRUE(slam.keyframes.back()->pose3.equals(pose, 1e-8));
      EXPECT_EQ(slam.nssm_accepted, 0);
      EXPECT_GT(slam.nssm_reverted, 0);
      reverted = true;
    }
  }
  ASSERT_TRUE(reverted);
  auto next = test::physical_frame(200, gtsam::Pose3(), gtsam::Pose3(), scene);
  slam.add_odometry(next);
  EXPECT_TRUE(slam.update_factor_graph(next)) << slam.last_error();
  EXPECT_TRUE(next->has_covariance6);
}

TEST(Se3Backend, RepeatedPhysicalRoomsCannotConfirmRelocalization)
{
  Slam map; configure(map);
  const Matrix scene = test::physical_scene();
  for (int k = 0; k < 2; ++k) {
    const gtsam::Pose3 pose(gtsam::Rot3(), gtsam::Point3(20 * k, 0, 0));
    auto frame = test::physical_frame(100 + k, gtsam::Pose3(), pose, scene);
    if (!k) map.add_prior(frame); else map.add_odometry(frame);
    ASSERT_TRUE(map.update_factor_graph(frame));
  }
  TemporaryMap path;
  ASSERT_TRUE(map.save_map(path.path));
  Slam resumed; configure(resumed);
  ASSERT_TRUE(resumed.load_map(path.path));
  for (int k = 0; k < 2; ++k) {
    auto observation = test::physical_frame(1000 + k, gtsam::Pose3(), gtsam::Pose3(), scene);
    EXPECT_FALSE(resumed.relocalize(observation));
    EXPECT_EQ(resumed.current_key(), 2);
    EXPECT_NE(resumed.last_error().find("ambiguous"), std::string::npos) << resumed.last_error();
  }
}

TEST(Se3Backend, EmptyOdometryAnchorsCannotHideAnInformativeReference)
{
  Slam map; configure(map);
  const Matrix scene = test::physical_scene();
  const gtsam::Pose3 reference_pose(gtsam::Rot3(), gtsam::Point3(10, 0, 0));
  auto reference = test::physical_frame(100, reference_pose, reference_pose, scene);
  map.add_prior(reference); ASSERT_TRUE(map.update_factor_graph(reference));
  // These valid graph anchors represent travel through open water. They are
  // closer to every good global proposal, but cannot supply a sonar fit.
  for (int k = 0; k < 4; ++k) {
    auto empty = test::physical_frame(101 + k, gtsam::Pose3(), gtsam::Pose3(), Matrix(0, 3));
    map.add_odometry(empty); ASSERT_TRUE(map.update_factor_graph(empty));
  }
  TemporaryMap path;
  ASSERT_TRUE(map.save_map(path.path));
  Slam resumed; configure(resumed);
  ASSERT_TRUE(resumed.load_map(path.path));
  for (int k = 0; k < 2; ++k) {
    auto observation = test::physical_frame(1000 + k, gtsam::Pose3(), gtsam::Pose3(), scene);
    if (!k) {
      EXPECT_FALSE(resumed.relocalize(observation));
      EXPECT_NE(resumed.last_error().find("confirmation"), std::string::npos) << resumed.last_error();
    } else {
      ASSERT_TRUE(resumed.relocalize(observation)) << resumed.last_error();
      EXPECT_LT(gtsam::Pose3::Logmap(observation->pose3).norm(), .08);
    }
  }
}

TEST(Relocalization, BothModelsRetryAfterAnExpiredAttemptWithoutChangingTheGraph)
{
  const Matrix scene = test::physical_scene();
  for (const auto model : {RegistrationModel::ConstrainedXYYaw, RegistrationModel::SE3}) {
    const auto select = [model](Slam& slam) {
      configure(slam);
      slam.registration_model = model;
      slam.ssm_params.enable = false; slam.nssm_params.enable = false;
      slam.configure();
    };
    Slam map; select(map);
    auto reference = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), scene);
    if (model == RegistrationModel::ConstrainedXYYaw) reference->chart = CloudChart::GravityHorizontal;
    map.add_prior(reference); ASSERT_TRUE(map.update_factor_graph(reference));
    TemporaryMap path;
    ASSERT_TRUE(map.save_map(path.path));
    Slam resumed; select(resumed);
    ASSERT_TRUE(resumed.load_map(path.path));
    const auto saved_pose = resumed.keyframes.front()->pose3;
    auto observation = test::physical_frame(1000, gtsam::Pose3(), gtsam::Pose3(), scene);
    if (model == RegistrationModel::ConstrainedXYYaw) observation->chart = CloudChart::GravityHorizontal;
    const auto expired = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    EXPECT_FALSE(resumed.relocalize(observation, expired));
    EXPECT_NE(resumed.last_error().find("deadline"), std::string::npos);
    EXPECT_TRUE(resumed.awaiting_relocalization());
    EXPECT_EQ(resumed.current_key(), 1);
    EXPECT_TRUE(resumed.keyframes.front()->pose3.equals(saved_pose, 1e-12));
    EXPECT_FALSE(resumed.relocalize(observation));
    EXPECT_NE(resumed.last_error().find("confirmation"), std::string::npos) << resumed.last_error();
    observation->time.sec++;
    ASSERT_TRUE(resumed.relocalize(observation)) << resumed.last_error();
    EXPECT_EQ(resumed.current_key(), 2);
    EXPECT_FALSE(resumed.awaiting_relocalization());
    EXPECT_LT(gtsam::Pose3::Logmap(observation->pose3).norm(), .08);
  }
}

TEST(Se3Backend, PhysicalChartAndResourceLimitsAreExplicit)
{
  Slam slam; configure(slam);
  auto wrong = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), test::physical_scene());
  wrong->chart = CloudChart::GravityHorizontal;
  EXPECT_THROW(slam.add_prior(wrong), std::invalid_argument);
  auto frame = test::physical_frame(100, gtsam::Pose3(), gtsam::Pose3(), test::physical_scene());
  const Matrix original = frame->points;
  frame->points.resize(original.rows() * 4, 3);
  for (int i = 0; i < 4; ++i) frame->points.middleRows(i * original.rows(), original.rows()) = original;
  slam.add_prior(frame); ASSERT_TRUE(slam.update_factor_graph(frame));
  EXPECT_LE(frame->points.rows(), 5000);
  TemporaryMap path;
  ASSERT_TRUE(slam.save_map(path.path));
  Slam resumed; configure(resumed);
  resumed.se3_options.max_source_points = 1000;
  ASSERT_TRUE(resumed.load_map(path.path));
  EXPECT_LE(resumed.keyframes.front()->points.rows(), 1000);
  Slam bounded_seed; configure(bounded_seed);
  bounded_seed.se3_options.max_seed_points = 100;
  ASSERT_TRUE(bounded_seed.load_map(path.path));
  auto observation = test::physical_frame(1000, gtsam::Pose3(), gtsam::Pose3(), test::physical_scene());
  EXPECT_FALSE(bounded_seed.relocalize(observation));
  EXPECT_NE(bounded_seed.last_error().find("max_seed_points"), std::string::npos) << bounded_seed.last_error();
  EXPECT_EQ(bounded_seed.current_key(), 1);
}
