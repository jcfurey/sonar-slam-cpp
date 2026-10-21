#include "physical_scene.hpp"
#include "sonar_slam_cpp/slam_core.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <random>

using namespace sonar_slam;

namespace {
class GraphTransactions : public ::testing::TestWithParam<RegistrationModel>
{
protected:
  void configure(Slam& slam, bool acoustic = false)
  {
    slam.registration_model = GetParam();
    slam.input_model = "measured_xyz";
    slam.ssm_params.enable = acoustic; slam.nssm_params.enable = false;
    slam.ssm_params.initialization = false; slam.ssm_require_covariance = false;
    slam.map_calibration_id = "transaction_test_mount";
    slam.map_depth_datum = "transaction_test_depth";
    slam.configure();
  }
  KeyframePtr frame(int second, double x = 0)
  {
    auto value = test::physical_frame(second, gtsam::Pose3(),
      gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(x, 0, 0)), Matrix(0, 3));
    if (GetParam() == RegistrationModel::ConstrainedXYYaw) value->chart = CloudChart::GravityHorizontal;
    return value;
  }
  void initialize(Slam& slam)
  {
    configure(slam);
    auto first = frame(100);
    slam.add_prior(first); ASSERT_TRUE(slam.update_factor_graph(first));
  }
};
struct TemporaryMap
{
  std::filesystem::path path = std::filesystem::temp_directory_path() /
    ("sonar_transaction_" + std::to_string(std::random_device{}()) + ".ssm");
  ~TemporaryMap() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};
}

TEST_P(GraphTransactions, UnstagedFrameCannotBecomeAnOrphan)
{
  Slam slam; initialize(slam);
  const auto unstaged = frame(101, 1);
  EXPECT_FALSE(slam.update_factor_graph(unstaged));
  ASSERT_EQ(slam.current_key(), 1);
  auto next = frame(102, 2);
  slam.add_odometry(next);
  ASSERT_TRUE(slam.update_factor_graph(next)) << slam.last_error();
  EXPECT_NEAR(next->pose3.x(), 2, 1e-8);
}

TEST_P(GraphTransactions, MismatchedCommitPreservesThePreparedFrame)
{
  Slam slam; initialize(slam);
  auto prepared = frame(101, 1);
  slam.add_odometry(prepared);
  EXPECT_FALSE(slam.update_factor_graph(frame(102, 10)));
  ASSERT_EQ(slam.current_key(), 1);
  ASSERT_TRUE(slam.update_factor_graph(prepared)) << slam.last_error();
  EXPECT_EQ(slam.keyframes.back(), prepared);
  EXPECT_NEAR(prepared->pose3.x(), 1, 1e-8);
}

TEST_P(GraphTransactions, RejectedSecondStagingCannotAddAnotherMeasurement)
{
  Slam slam; initialize(slam);
  auto prepared = frame(101, 1);
  slam.add_odometry(prepared);
  EXPECT_THROW(slam.add_odometry(frame(102, 3)), std::logic_error);
  ASSERT_TRUE(slam.update_factor_graph(prepared)) << slam.last_error();
  EXPECT_NEAR(prepared->pose3.x(), 1, 1e-8);
}

TEST_P(GraphTransactions, CorrectionCannotConsumeAPreparedFrame)
{
  Slam slam; initialize(slam);
  auto prepared = frame(101, 1);
  slam.add_odometry(prepared);
  EXPECT_FALSE(slam.add_manual_correction(gtsam::Pose2(10, 0, 0)));
  ASSERT_EQ(slam.current_key(), 1);
  ASSERT_TRUE(slam.update_factor_graph(prepared)) << slam.last_error();
  EXPECT_EQ(slam.manual_corrections_pending(), 0);
  EXPECT_NEAR(prepared->pose3.x(), 1, 1e-8);
  auto next = frame(102, 2);
  slam.add_odometry(next);
  ASSERT_TRUE(slam.update_factor_graph(next)) << slam.last_error();
}

TEST_P(GraphTransactions, MapLoadCannotReplaceAPreparedFirstFrame)
{
  Slam saved; initialize(saved);
  TemporaryMap path;
  ASSERT_TRUE(saved.save_map(path.path)) << saved.last_error();
  Slam fresh; configure(fresh);
  auto prepared = frame(101, 1);
  fresh.add_prior(prepared);
  EXPECT_FALSE(fresh.load_map(path.path));
  EXPECT_FALSE(fresh.awaiting_relocalization());
  ASSERT_EQ(fresh.current_key(), 0);
  ASSERT_TRUE(fresh.update_factor_graph(prepared)) << fresh.last_error();
  EXPECT_NEAR(prepared->pose3.x(), 1, 1e-8);
}

TEST_P(GraphTransactions, MapSaveRejectsAnUncommittedFrame)
{
  Slam slam; initialize(slam);
  auto prepared = frame(101, 1);
  slam.add_odometry(prepared);
  TemporaryMap path;
  EXPECT_FALSE(slam.save_map(path.path));
  EXPECT_FALSE(std::filesystem::exists(path.path));
  ASSERT_TRUE(slam.update_factor_graph(prepared)) << slam.last_error();
  EXPECT_TRUE(slam.save_map(path.path)) << slam.last_error();
}

TEST_P(GraphTransactions, EmptyUpdatesAreHarmless)
{
  Slam slam; configure(slam);
  EXPECT_TRUE(slam.update_factor_graph());
  EXPECT_EQ(slam.current_key(), 0);
  EXPECT_FALSE(slam.has_pending_update());
  auto first = frame(100);
  slam.add_prior(first);
  EXPECT_FALSE(slam.update_factor_graph());
  EXPECT_TRUE(slam.has_pending_update());
  ASSERT_TRUE(slam.update_factor_graph(first));
  EXPECT_TRUE(slam.update_factor_graph());
  EXPECT_EQ(slam.current_key(), 1);
}

TEST_P(GraphTransactions, AdmissionEnforcesThePriorAndAcquisitionOrder)
{
  Slam slam; configure(slam);
  EXPECT_THROW(slam.add_odometry(frame(100)), std::logic_error);
  EXPECT_FALSE(slam.has_pending_update());
  auto first = frame(100);
  slam.add_prior(first); ASSERT_TRUE(slam.update_factor_graph(first));
  EXPECT_THROW(slam.add_prior(frame(101)), std::logic_error);
  EXPECT_THROW(slam.add_odometry(frame(99)), std::invalid_argument);
  EXPECT_THROW(slam.add_odometry(frame(100)), std::invalid_argument);
  auto invalid = frame(101);
  invalid->time.nanosec = 1000000000U;
  EXPECT_THROW(slam.add_odometry(invalid), std::invalid_argument);
  invalid = frame(101);
  invalid->chart = GetParam() == RegistrationModel::SE3 ? CloudChart::GravityHorizontal : CloudChart::Body;
  EXPECT_THROW(slam.add_odometry(invalid), std::invalid_argument);
  EXPECT_FALSE(slam.has_pending_update());
  auto next = frame(101, 1);
  slam.add_odometry(next); EXPECT_TRUE(slam.update_factor_graph(next));
}

TEST_P(GraphTransactions, DiscardAllowsAnotherFrameWithoutChangingCommittedState)
{
  Slam slam; initialize(slam);
  const auto before = slam.keyframes.front()->pose3;
  slam.add_odometry(frame(101, 1));
  EXPECT_TRUE(slam.has_pending_update());
  slam.discard_pending_update();
  slam.discard_pending_update();
  EXPECT_FALSE(slam.has_pending_update());
  EXPECT_EQ(slam.current_key(), 1);
  EXPECT_TRUE(slam.keyframes.front()->pose3.equals(before, 1e-12));
  auto next = frame(102, 2);
  slam.add_odometry(next);
  ASSERT_TRUE(slam.update_factor_graph(next)) << slam.last_error();
  EXPECT_NEAR(next->pose3.x(), 2, 1e-8);
}

TEST_P(GraphTransactions, OtherGraphOperationsPreservePendingWork)
{
  Slam slam; initialize(slam);
  ASSERT_TRUE(slam.add_manual_correction(gtsam::Pose2()));
  auto prepared = frame(101, 1);
  slam.add_odometry(prepared);
  EXPECT_FALSE(slam.undo_manual_correction());
  EXPECT_FALSE(slam.add_position_prior(0, 0, 0, .1));
  EXPECT_FALSE(slam.add_manual_correction(gtsam::Pose3(), Covariance6::Identity()));
  EXPECT_FALSE(slam.add_nonsequential_scan_matching());
  EXPECT_THROW(slam.configure(), std::logic_error);
  EXPECT_EQ(slam.manual_corrections_pending(), 1);
  EXPECT_TRUE(slam.has_pending_update());
  ASSERT_TRUE(slam.update_factor_graph(prepared)) << slam.last_error();
  EXPECT_TRUE(slam.undo_manual_correction()) << slam.last_error();
  EXPECT_FALSE(slam.has_pending_update());
  EXPECT_THROW(slam.configure(), std::logic_error);
}

TEST_P(GraphTransactions, SequentialAcceptanceCountsOnlyCommittedMeasurements)
{
  Slam slam; configure(slam, true);
  const Matrix scene = test::physical_scene();
  auto first = frame(100); first->points = scene;
  slam.add_prior(first); ASSERT_TRUE(slam.update_factor_graph(first));
  const auto observation = [&](int second) {
    auto next = frame(second, .2);
    next->points = transform_body_points(scene, next->dr_pose3.inverse());
    return next;
  };
  auto discarded = observation(101);
  slam.add_sequential_scan_matching(discarded);
  ASSERT_NE(slam.last_ssm_status.find("accepted"), std::string::npos) << slam.last_ssm_status;
  EXPECT_EQ(slam.ssm_accepted, 0);
  slam.discard_pending_update();
  EXPECT_EQ(slam.ssm_accepted, 0);
  auto committed = observation(102);
  slam.add_sequential_scan_matching(committed);
  ASSERT_TRUE(slam.update_factor_graph(committed)) << slam.last_error();
  EXPECT_EQ(slam.ssm_accepted, 1) << slam.last_ssm_status;
  EXPECT_FALSE(slam.has_pending_update());
}

INSTANTIATE_TEST_SUITE_P(BothModels, GraphTransactions, ::testing::Values(
  RegistrationModel::ConstrainedXYYaw, RegistrationModel::SE3));
