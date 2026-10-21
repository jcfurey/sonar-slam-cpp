#include "physical_scene.hpp"
#include <gtest/gtest.h>

using namespace sonar_slam;

TEST(RigidRegistration, RecoversAllSixDirectionsAndReusesTarget)
{
  const Matrix target = test::physical_scene();
  const gtsam::Pose3 truth(gtsam::Rot3::RzRyRx(.10, -.08, .12), gtsam::Point3(.3, -.2, .4));
  const Matrix source = transform_body_points(target, truth.inverse());
  RigidIcpParams params;
  params.normal_radius = .8;
  RigidRegistration matcher(target, params);
  for (int axis = 0; axis < 6; ++axis) {
    gtsam::Vector6 perturbation = gtsam::Vector6::Zero();
    perturbation[axis] = axis < 3 ? .08 : .25;
    const auto result = matcher.compute(source, truth.compose(gtsam::Pose3::Expmap(perturbation)));
    ASSERT_TRUE(result.converged) << result.reason;
    EXPECT_EQ(result.observed_rank, 6);
    EXPECT_LT(gtsam::Pose3::Logmap(truth.between(result.transform)).norm(), .015);
    EXPECT_GT(result.overlap, .9);
    EXPECT_TRUE(valid_covariance6(result.covariance));
  }
}

TEST(RigidRegistration, RejectsWallsFloorsAndCappedFits)
{
  const Matrix scene = test::physical_scene();
  for (int plane : {0, 1, 2}) {
    Matrix target(scene.rows() / 3, 3);
    for (Eigen::Index row = 0; row < target.rows(); ++row) target.row(row) = scene.row(row * 3 + plane);
    RigidRegistration matcher(target, RigidIcpParams{});
    const auto result = matcher.compute(target, gtsam::Pose3());
    EXPECT_FALSE(result.converged);
    EXPECT_LT(result.observed_rank, 6);
  }
  RigidIcpParams params;
  params.max_iterations = 1;
  RigidRegistration capped(scene, params);
  auto fit = capped.compute(scene, gtsam::Pose3(gtsam::Rot3::Rx(.08), gtsam::Point3(.2, .1, .15)));
  EXPECT_FALSE(fit.converged);
  EXPECT_NE(fit.reason.find("iteration"), std::string::npos);
  fit = capped.compute(scene, gtsam::Pose3(), 0);
  EXPECT_FALSE(fit.converged);
  EXPECT_TRUE(fit.timed_out);
  EXPECT_FALSE(fit.has_covariance);
  const auto expired = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  EXPECT_THROW(RigidRegistration(scene, params, expired), DeadlineExceeded);
  fit = capped.compute(scene, gtsam::Pose3());
  EXPECT_TRUE(fit.converged) << fit.reason;
  EXPECT_FALSE(fit.timed_out);
}

TEST(RigidRegistration, GlobalOriginDoesNotChangeObservability)
{
  const auto shift = gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(250, -130, 60));
  const Matrix target = transform_body_points(test::physical_scene(), shift);
  RigidRegistration matcher(target, RigidIcpParams{});
  const auto fit = matcher.compute(test::physical_scene(), shift.compose(
    gtsam::Pose3(gtsam::Rot3::Ry(.05), gtsam::Point3(.1, -.1, .1))));
  ASSERT_TRUE(fit.converged) << fit.reason;
  EXPECT_LT(gtsam::Pose3::Logmap(shift.between(fit.transform)).norm(), .02);
  EXPECT_TRUE(valid_covariance6(fit.covariance));
}

TEST(RigidCovariance, RosConversionMatchesFixedAxisPerturbations)
{
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(.3, -.4, .8), gtsam::Point3(4, -2, 7));
  Covariance6 factor = Covariance6::Identity();
  factor(0, 4) = .2; factor(1, 5) = -.1;
  const Covariance6 covariance = factor * factor.transpose() * .01;
  Covariance6 jacobian;
  for (int axis = 0; axis < 6; ++axis) {
    gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[axis] = 1e-6;
    const auto perturbed = pose.compose(gtsam::Pose3::Expmap(delta));
    jacobian.col(axis).head<3>() = (perturbed.translation() - pose.translation()) / 1e-6;
    jacobian.col(axis).tail<3>() = gtsam::Rot3::Logmap(
      perturbed.rotation().compose(pose.rotation().inverse())) / 1e-6;
  }
  const auto ros = covariance6_to_ros(pose, covariance);
  EXPECT_TRUE(ros.isApprox(jacobian * covariance * jacobian.transpose(), 1e-5));
  EXPECT_TRUE(covariance6_from_ros(pose, ros).isApprox(covariance, 1e-10));
  EXPECT_TRUE(covariance6_to_ros(pose, Covariance6::Zero()).isApprox(Covariance6::Identity() * 1e6));
  const Covariance6 inverted = inverse_covariance(pose, covariance);
  EXPECT_TRUE(valid_covariance6(inverted));
  const auto combined = compose_covariance(pose, covariance, covariance);
  EXPECT_TRUE(valid_covariance6(combined));
  const gtsam::Pose3 left(gtsam::Rot3::RzRyRx(-.2, .1, -.3), gtsam::Point3(1, 2, -.4));
  const auto nominal = left.compose(pose);
  Covariance6 left_jacobian, right_jacobian, inverse_jacobian;
  for (int axis = 0; axis < 6; ++axis) {
    gtsam::Vector6 delta = gtsam::Vector6::Zero(); delta[axis] = 1e-6;
    left_jacobian.col(axis) = gtsam::Pose3::Logmap(nominal.between(
      left.compose(gtsam::Pose3::Expmap(delta)).compose(pose))) / 1e-6;
    right_jacobian.col(axis) = gtsam::Pose3::Logmap(nominal.between(
      left.compose(pose.compose(gtsam::Pose3::Expmap(delta))))) / 1e-6;
    inverse_jacobian.col(axis) = gtsam::Pose3::Logmap(pose.inverse().between(
      pose.compose(gtsam::Pose3::Expmap(delta)).inverse())) / 1e-6;
  }
  EXPECT_TRUE(compose_covariance(pose, covariance, covariance).isApprox(
    left_jacobian * covariance * left_jacobian.transpose() +
    right_jacobian * covariance * right_jacobian.transpose(), 1e-5));
  EXPECT_TRUE(inverted.isApprox(inverse_jacobian * covariance * inverse_jacobian.transpose(), 1e-5));
}

TEST(RegistrationContract, ConstrainedInformationDoesNotInventOtherDirections)
{
  ConstrainedIcpResult fit;
  fit.success = fit.converged = true;
  const Eigen::Matrix3d covariance = Eigen::Vector3d(.04, .09, .01).asDiagonal();
  const auto result = constrained_registration_result(fit, covariance);
  EXPECT_EQ(result.model, RegistrationModel::ConstrainedXYYaw);
  EXPECT_EQ(result.observed_rank, 3);
  EXPECT_TRUE(result.has_covariance);
  EXPECT_DOUBLE_EQ(result.information(3, 3), 25);
  for (int index : {0, 1, 5}) EXPECT_TRUE(result.information.row(index).isZero());
  EXPECT_FALSE(constrained_registration_result(fit).has_covariance);
  fit.converged = false;
  EXPECT_EQ(constrained_registration_result(fit, covariance).observed_rank, 0);
}

TEST(RigidCovariance, ValidationAndRosConversionAreSafeAtExtremeFiniteScales)
{
  for (double scale : {1e-200, 1e200}) {
    Covariance6 valid = Covariance6::Identity() * scale;
    EXPECT_TRUE(valid_covariance6(valid));
    auto asymmetric = valid;
    asymmetric(0, 1) = scale;
    EXPECT_FALSE(valid_covariance6(asymmetric));
    auto indefinite = valid;
    indefinite(0, 1) = indefinite(1, 0) = 2 * scale;
    EXPECT_FALSE(valid_covariance6(indefinite));
  }
  const Covariance6 large = Covariance6::Identity() * 1e308;
  const auto ros = covariance6_to_ros(gtsam::Pose3(), large);
  EXPECT_TRUE(ros.allFinite());
  EXPECT_TRUE(valid_covariance6(ros));
  EXPECT_LT((ros / 1e308 - Covariance6::Identity()).norm(), 1e-12);
  Covariance6 correlated = large;
  correlated(0, 0) = correlated(1, 1) = correlated(0, 1) = correlated(1, 0) = 9e307;
  correlated(0, 1) -= 1e307; correlated(1, 0) -= 1e307;
  EXPECT_TRUE(valid_covariance6(correlated));
  const auto converted = covariance6_to_ros(gtsam::Pose3(gtsam::Rot3::Rz(M_PI / 4), gtsam::Point3::Zero()), correlated);
  EXPECT_TRUE(converted.allFinite());
  EXPECT_TRUE(valid_covariance6(converted));
  EXPECT_GT(converted.diagonal().maxCoeff(), 1e307);
  correlated = large;
  correlated(0, 1) = correlated(1, 0) = 9e307;
  EXPECT_TRUE(valid_covariance6(correlated));
  const auto overflow = covariance6_to_ros(gtsam::Pose3(gtsam::Rot3::Rz(M_PI / 4), gtsam::Point3::Zero()), correlated);
  EXPECT_TRUE(overflow.isApprox(Covariance6::Identity() * 1e6));
}
