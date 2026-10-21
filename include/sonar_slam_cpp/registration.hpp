#pragma once

#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/deadline.hpp"
#include <gtsam/geometry/Pose3.h>
#include <memory>
#include <optional>

namespace sonar_slam {

using Covariance6 = Eigen::Matrix<double, 6, 6>;
enum class RegistrationModel { ConstrainedXYYaw, SE3 };
enum class CloudChart { GravityHorizontal, Body };

// T_target_source maps physical source points into the target chart.
// Covariance uses right perturbations T * Exp([rx,ry,rz,tx,ty,tz]).
struct RegistrationResult
{
  RegistrationModel model = RegistrationModel::SE3;
  gtsam::Pose3 transform;
  Covariance6 covariance = Covariance6::Identity() * 1e6;
  Covariance6 information = Covariance6::Zero();
  bool has_covariance = false;
  bool converged = false;
  bool timed_out = false;
  int observed_rank = 0;
  int correspondences = 0;
  int iterations = 0;
  double overlap = 0.0;
  double rmse = 0.0;
  double seconds = 0.0;
  std::string reason;
};

// Adapt a pressure-aligned constrained fit to the common result contract.
// Its information has exact zero rows/columns in unobserved roll/pitch/z.
RegistrationResult constrained_registration_result(const ConstrainedIcpResult& fit,
  const std::optional<Eigen::Matrix3d>& sampled_covariance = std::nullopt);

struct RigidIcpParams
{
  double max_correspondence = 0.75;
  double trim_ratio = 0.8;
  int max_iterations = 50;
  int min_correspondences = 30;
  double translation_epsilon = 0.005;
  double rotation_epsilon = 0.001;
  int normal_neighbors = 15;
  double normal_radius = 1.0;
  double max_curvature = 0.1;
  double sensor_noise = 0.1;
  double rank_tolerance = 1e-4;
  double rotation_sigma_floor = 0.02;
  double translation_sigma_floor = 0.05;
  void validate() const;
};

// Owns one immutable target, normal estimates and search tree. A caller may
// reuse it for multiple guesses; compute calls on one instance are serialized.
class RigidRegistration
{
public:
  RigidRegistration(const Matrix& target, const RigidIcpParams& params,
                    Deadline deadline = Deadline::max());
  ~RigidRegistration();
  RegistrationResult compute(const Matrix& source, const gtsam::Pose3& guess,
                             int max_ms = 2000);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool valid_covariance6(const Covariance6& covariance);
Covariance6 compose_covariance(const gtsam::Pose3& right,
  const Covariance6& left_covariance, const Covariance6& right_covariance);
Covariance6 inverse_covariance(const gtsam::Pose3& pose, const Covariance6& covariance);
// ROS order is [x,y,z,rotation-about-X,Y,Z], about the pose origin in map axes.
Covariance6 covariance6_to_ros(const gtsam::Pose3& pose, const Covariance6& right_covariance);
Covariance6 covariance6_from_ros(const gtsam::Pose3& pose, const Covariance6& ros_covariance);
Matrix transform_body_points(const Matrix& points, const gtsam::Pose3& pose);

}  // namespace sonar_slam
