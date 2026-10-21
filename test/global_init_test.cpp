#include "sonar_slam_cpp/global_init.hpp"
#include "sonar_slam_cpp/mcd.hpp"
#include "sonar_slam_cpp/slam_core.hpp"
#include <gtest/gtest.h>
#include <limits>

using namespace sonar_slam;

namespace {
Eigen::Matrix<double, 3, 2> fixed_bounds()
{
  return Eigen::Matrix<double, 3, 2>::Zero();
}
void select_model(Slam& slam, RegistrationModel model)
{
  slam.registration_model = model; slam.input_model = "measured_xyz";
  slam.ssm_params.enable = false; slam.nssm_params.enable = false;
}
}

TEST(GlobalInitialization, LargeAndThinFootprintsHaveBoundedProposalStorage)
{
  // At the old noise/10 resolution these need terabytes and an overflowing
  // integer dimension respectively. A sparse map does not imply a small grid.
  for (const Eigen::Vector2f& extent : {Eigen::Vector2f(100000, 100000),
                                      Eigen::Vector2f(100000000, 0)}) {
    Matrix points(3, 3);
    points << 0, 0, 0, extent.x() * .5f, extent.y() * .5f, 0, extent.x(), extent.y(), 0;
    const auto fit = global_scan_match_init(points, {}, points, {}, .001, fixed_bounds(), 1, 1, .01);
    ASSERT_TRUE(fit.success);
    EXPECT_LE(fit.grid_cols, GLOBAL_INIT_MAX_GRID_AXIS);
    EXPECT_LE(fit.grid_rows, GLOBAL_INIT_MAX_GRID_AXIS);
    EXPECT_LE(static_cast<std::size_t>(fit.grid_rows) * fit.grid_cols, GLOBAL_INIT_MAX_GRID_CELLS);
    EXPECT_GT(fit.grid_resolution, .001 / 10);
    EXPECT_EQ(fit.best_cost, -points.rows());
  }
  Matrix local(2, 2); local << 0, 0, 2, 1;
  const auto fit = global_scan_match_init(local, {}, local, {}, .2, fixed_bounds(), 1, 1, .01);
  EXPECT_TRUE(fit.success);
  EXPECT_DOUBLE_EQ(fit.grid_resolution, .2 / 10);
}

TEST(GlobalInitialization, RemoteFiniteCandidatesDoNotOverflowIntegerCells)
{
  Matrix points(2, 2); points << 0, 0, 1, 1;
  const std::vector<Eigen::Vector3d> deltas{{0, 0, 0}, {1e30, -1e30, 0}};
  const auto costs = global_match_costs(points, {}, points, {}, .1, deltas);
  ASSERT_EQ(costs.size(), 2);
  EXPECT_FLOAT_EQ(costs[0], -2);
  EXPECT_FLOAT_EQ(costs[1], 0);
}

TEST(GlobalInitialization, RefinementRespectsTheRequestedBounds)
{
  // Unequal spacing prevents a one-metre shift from aliasing adjacent points.
  // The unconstrained optimum lies just beyond the permitted translation.
  Matrix source(3, 2); source << 0, 0, 4, 0, 11, 0;
  Matrix target(3, 2); target << 1.005f, 0, 5.02f, 0, 12.035f, 0;
  auto bounds = fixed_bounds(); bounds(0, 1) = 1;
  const auto fit = global_scan_match_init(source, {}, target, {}, .025, bounds, 63, 1, .01);
  ASSERT_TRUE(fit.success);
  EXPECT_GE(fit.delta.x(), 0);
  EXPECT_LE(fit.delta.x(), 1);
  EXPECT_DOUBLE_EQ(fit.delta.y(), 0);
  EXPECT_DOUBLE_EQ(fit.delta.theta(), 0);
  target.array() += 100;
  EXPECT_FALSE(global_scan_match_init(source, {}, target, {}, .025, bounds, 63, 1, .01).success);
}

TEST(GlobalInitialization, InvalidGeometryAndSamplingFailBeforeAllocation)
{
  Matrix points = Matrix::Zero(2, 3);
  const auto bounds = fixed_bounds();
  const auto init = [&](const Matrix& source, double noise, int n = 1, int iters = 1) {
    return global_scan_match_init(source, {}, points, {}, noise, bounds, n, iters, .01);
  };
  EXPECT_THROW(init(Matrix::Zero(2, 1), .1), std::invalid_argument);
  Matrix invalid = points; invalid(0, 0) = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(init(invalid, .1), std::invalid_argument);
  for (double noise : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(init(points, noise), std::invalid_argument);
  EXPECT_THROW(init(points, .1, 0), std::invalid_argument);
  EXPECT_THROW(init(points, .1, 1, 0), std::invalid_argument);
  EXPECT_THROW(init(points, .1, std::numeric_limits<int>::max(), 2), std::invalid_argument);
  auto inverted = bounds; inverted(0, 0) = 1;
  EXPECT_THROW(global_scan_match_init(points, {}, points, {}, .1, inverted, 1, 1, .01), std::invalid_argument);
  EXPECT_THROW(global_match_costs(points, {}, points, {}, .1,
    {Eigen::Vector3d(0, std::numeric_limits<double>::quiet_NaN(), 0)}), std::invalid_argument);
}

TEST(GlobalInitialization, BothGraphModelsValidateSamplingAtStartup)
{
  for (const auto model : {RegistrationModel::ConstrainedXYYaw, RegistrationModel::SE3}) {
    Slam valid; select_model(valid, model);
    ASSERT_NO_THROW(valid.configure());
    for (int invalid : {0, -1, std::numeric_limits<int>::max()}) {
      Slam slam;
      select_model(slam, model);
      slam.nssm_params.init_n = invalid;
      EXPECT_THROW(slam.configure(), std::invalid_argument);
    }
    Slam slam;
    select_model(slam, model);
    slam.ssm_params.init_ftol = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(slam.configure(), std::invalid_argument);
    for (int invalid : {0, -1, 60001}) {
      Slam budget;
      select_model(budget, model); budget.relocalize_max_ms = invalid;
      EXPECT_THROW(budget.configure(), std::invalid_argument);
    }
  }
}

TEST(GlobalInitialization, ExpiredDeadlineCannotReturnAPartialProposal)
{
  Matrix points(2, 3); points << 0, 0, 0, 1, 1, 0;
  const auto expired = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  const auto fit = global_scan_match_init(points, {}, points, {}, .1, fixed_bounds(), 500, 4, .01, expired);
  EXPECT_TRUE(fit.timed_out);
  EXPECT_FALSE(fit.success);
  EXPECT_TRUE(fit.pose_samples.empty());
  EXPECT_EQ(fit.grid_rows, 0);
  const auto retry = global_scan_match_init(points, {}, points, {}, .1, fixed_bounds(), 1, 1, .01);
  EXPECT_TRUE(retry.success);
  EXPECT_FALSE(retry.timed_out);
}

TEST(CovarianceDeadline, RobustEstimatorDoesNotPublishIncompleteTrials)
{
  const Eigen::MatrixXd samples = Eigen::MatrixXd::Random(40, 3);
  const auto expired = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  const auto timeout = min_cov_det(samples, .8, expired);
  EXPECT_TRUE(timeout.timed_out);
  EXPECT_FALSE(timeout.success);
  EXPECT_EQ(timeout.covariance.size(), 0);
  // Interrupt actual work as well as rejecting deadlines expired on entry.
  // Avoid an elapsed-time assertion: one covariance step is atomic.
  const Eigen::MatrixXd large = samples.replicate(1000, 1);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
  const auto interrupted = min_cov_det(large, .8, deadline);
  EXPECT_TRUE(interrupted.timed_out);
  EXPECT_FALSE(interrupted.success);
  EXPECT_EQ(interrupted.covariance.size(), 0);
  const auto retry = min_cov_det(samples, .8);
  EXPECT_TRUE(retry.success);
  EXPECT_FALSE(retry.timed_out);
}
