#include "sonar_slam_cpp/registration.hpp"
#include "sonar_slam_cpp/pose_covariance.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace sonar_slam {

RegistrationResult constrained_registration_result(const ConstrainedIcpResult& fit,
  const std::optional<Eigen::Matrix3d>& covariance)
{
  RegistrationResult result;
  result.model = RegistrationModel::ConstrainedXYYaw;
  result.transform = gtsam::Pose3(gtsam::Rot3::Yaw(std::atan2(fit.T(1, 0), fit.T(0, 0))),
    gtsam::Point3(fit.T(0, 2), fit.T(1, 2), 0));
  result.converged = fit.converged;
  result.observed_rank = fit.converged ? 3 : 0;
  result.correspondences = fit.correspondences; result.iterations = fit.iterations;
  result.rmse = fit.rmse; result.reason = fit.message;
  if (fit.converged && covariance && valid_pose_covariance(*covariance)) {
    const Eigen::LDLT<Eigen::Matrix3d> decomposition(*covariance);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive() ||
        (decomposition.vectorD().array() <= 0).any()) return result;
    constexpr int indices[3] = {3, 4, 2};
    const Eigen::Matrix3d information = decomposition.solve(Eigen::Matrix3d::Identity());
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      result.covariance(indices[i], indices[j]) = (*covariance)(i, j);
      result.information(indices[i], indices[j]) = information(i, j);
    }
    result.has_covariance = true;
  }
  return result;
}

void RigidIcpParams::validate() const
{
  for (double value : {max_correspondence, translation_epsilon, rotation_epsilon,
      normal_radius, sensor_noise, rank_tolerance, rotation_sigma_floor, translation_sigma_floor})
    if (!std::isfinite(value) || value <= 0.0)
      throw std::invalid_argument("SE3 registration distances, tolerances and noise must be finite and positive");
  if (!std::isfinite(trim_ratio) || trim_ratio <= 0.0 || trim_ratio > 1.0 ||
      max_iterations < 1 || min_correspondences < 6 || normal_neighbors < 6 ||
      !std::isfinite(max_curvature) || max_curvature <= 0.0 || max_curvature >= 1.0 / 3.0 ||
      rank_tolerance >= 1.0)
    throw std::invalid_argument("invalid SE3 registration counts, trimming or observability settings");
}

bool valid_covariance6(const Covariance6& covariance)
{
  if (!covariance.allFinite() || (covariance.diagonal().array() <= 0.0).any()) return false;
  const Covariance6 scaled = covariance / covariance.cwiseAbs().maxCoeff();
  if (!scaled.isApprox(scaled.transpose(), 1e-9)) return false;
  const Eigen::SelfAdjointEigenSolver<Covariance6> eigen(scaled);
  return eigen.info() == Eigen::Success && eigen.eigenvalues().minCoeff() > 0.0;
}

Covariance6 compose_covariance(const gtsam::Pose3& right,
  const Covariance6& left_covariance, const Covariance6& right_covariance)
{
  const Covariance6 adjoint = right.inverse().AdjointMap();
  return adjoint * left_covariance * adjoint.transpose() + right_covariance;
}

Covariance6 inverse_covariance(const gtsam::Pose3& pose, const Covariance6& covariance)
{
  const Covariance6 adjoint = pose.AdjointMap();
  return adjoint * covariance * adjoint.transpose();
}

Covariance6 covariance6_to_ros(const gtsam::Pose3& pose, const Covariance6& covariance)
{
  if (!valid_covariance6(covariance)) return Covariance6::Identity() * 1e6;
  Covariance6 jacobian = Covariance6::Zero();
  jacobian.topRightCorner<3, 3>() = pose.rotation().matrix();
  jacobian.bottomLeftCorner<3, 3>() = pose.rotation().matrix();
  const Covariance6 result = jacobian * covariance * jacobian.transpose();
  if (!valid_covariance6(result)) return Covariance6::Identity() * 1e6;
  return result * 0.5 + result.transpose() * 0.5;
}

Covariance6 covariance6_from_ros(const gtsam::Pose3& pose, const Covariance6& covariance)
{
  Covariance6 jacobian = Covariance6::Zero();
  jacobian.topRightCorner<3, 3>() = pose.rotation().matrix().transpose();
  jacobian.bottomLeftCorner<3, 3>() = pose.rotation().matrix().transpose();
  return jacobian * covariance * jacobian.transpose();
}

Matrix transform_body_points(const Matrix& points, const gtsam::Pose3& pose)
{
  if (points.cols() != 3) throw std::invalid_argument("physical clouds require Nx3 points");
  Matrix result = points * pose.rotation().matrix().cast<float>().transpose();
  result.rowwise() += pose.translation().cast<float>().transpose();
  return result;
}

struct RigidRegistration::Impl
{
  using Point = pcl::PointXYZ;
  RigidIcpParams params;
  pcl::PointCloud<Point>::Ptr cloud{new pcl::PointCloud<Point>};
  pcl::KdTreeFLANN<Point> tree;
  Matrix normals;
  Eigen::Vector3d center;

  Impl(const Matrix& target, const RigidIcpParams& p, Deadline deadline) : params(p)
  {
    if (deadline_expired(deadline)) throw DeadlineExceeded();
    params.validate();
    if (target.cols() != 3 || !target.allFinite() || target.rows() < params.normal_neighbors)
      throw std::invalid_argument("SE3 target needs finite XYZ and enough normal neighbors");
    cloud->reserve(target.rows());
    center = target.colwise().mean().cast<double>();
    for (Eigen::Index i = 0; i < target.rows(); ++i)
      cloud->emplace_back(target(i, 0), target(i, 1), target(i, 2));
    tree.setInputCloud(cloud);
    normals = Matrix::Zero(target.rows(), 3);
    std::vector<int> indices(params.normal_neighbors);
    std::vector<float> distances(params.normal_neighbors);
    for (Eigen::Index i = 0; i < target.rows(); ++i) {
      if (i % 64 == 0 && deadline_expired(deadline)) throw DeadlineExceeded();
      const int count = tree.nearestKSearch((*cloud)[i], params.normal_neighbors, indices, distances);
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      int used = 0;
      for (int j = 0; j < count; ++j) {
        if (distances[j] > params.normal_radius * params.normal_radius) continue;
        mean += target.row(indices[j]).cast<double>().transpose();
        ++used;
      }
      if (used < 6) continue;
      mean /= used;
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
      for (int j = 0; j < count; ++j) {
        if (distances[j] > params.normal_radius * params.normal_radius) continue;
        const Eigen::Vector3d delta = target.row(indices[j]).cast<double>().transpose() - mean;
        covariance.noalias() += delta * delta.transpose();
      }
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
      if (eigen.info() != Eigen::Success || eigen.eigenvalues()[1] < 1e-8 ||
          eigen.eigenvalues()[0] / eigen.eigenvalues().sum() > params.max_curvature) continue;
      normals.row(i) = eigen.eigenvectors().col(0).cast<float>().transpose();
    }
    if (deadline_expired(deadline)) throw DeadlineExceeded();
  }

  struct Linearization
  {
    Covariance6 information = Covariance6::Zero();
    gtsam::Vector6 rhs = gtsam::Vector6::Zero();
    int overlap = 0, kept = 0, rank = 0;
    double squared_error = 0.0;
  };
  Linearization linearize(const Matrix& source, const gtsam::Pose3& pose)
  {
    struct Pair { Eigen::Vector3d point, normal; double residual, distance; };
    std::vector<Pair> pairs;
    pairs.reserve(source.rows());
    const Matrix transformed = transform_body_points(source, pose);
    std::vector<int> index(1);
    std::vector<float> distance(1);
    for (Eigen::Index i = 0; i < transformed.rows(); ++i) {
      const Point query(transformed(i, 0), transformed(i, 1), transformed(i, 2));
      if (!tree.nearestKSearch(query, 1, index, distance) ||
          distance[0] > params.max_correspondence * params.max_correspondence ||
          normals.row(index[0]).squaredNorm() < 0.5) continue;
      const Eigen::Vector3d point = transformed.row(i).cast<double>().transpose();
      const auto& target = (*cloud)[index[0]];
      const Eigen::Vector3d normal = normals.row(index[0]).cast<double>().transpose();
      pairs.push_back({point, normal,
        normal.dot(point - Eigen::Vector3d(target.x, target.y, target.z)), distance[0]});
    }
    Linearization out;
    out.overlap = pairs.size();
    if (out.overlap < params.min_correspondences) return out;
    out.kept = std::max(params.min_correspondences,
      static_cast<int>(std::floor(params.trim_ratio * pairs.size())));
    std::nth_element(pairs.begin(), pairs.begin() + out.kept - 1, pairs.end(),
      [](const Pair& a, const Pair& b) { return a.distance < b.distance; });
    double scale_squared = 0.0;
    for (int i = 0; i < out.kept; ++i) {
      const auto& pair = pairs[i];
      gtsam::Vector6 jacobian;
      jacobian << (pair.point - center).cross(pair.normal), pair.normal;
      const double weight = std::min(1.0,
        2.0 * params.sensor_noise / std::max(1e-12, std::abs(pair.residual)));
      out.information.noalias() += weight * jacobian * jacobian.transpose();
      out.rhs.noalias() -= weight * jacobian * pair.residual;
      out.squared_error += pair.residual * pair.residual;
      scale_squared += (pair.point - center).squaredNorm();
    }
    Eigen::Matrix<double, 6, 1> scale = Eigen::Matrix<double, 6, 1>::Ones();
    scale.head<3>().setConstant(1.0 / std::max(1.0, std::sqrt(scale_squared / out.kept)));
    const Covariance6 scaled = scale.asDiagonal() * out.information * scale.asDiagonal();
    const Eigen::SelfAdjointEigenSolver<Covariance6> eigen(scaled);
    if (eigen.info() == Eigen::Success && eigen.eigenvalues().maxCoeff() > 0.0)
      out.rank = (eigen.eigenvalues().array() >
        eigen.eigenvalues().maxCoeff() * params.rank_tolerance).count();
    return out;
  }
};

RigidRegistration::RigidRegistration(const Matrix& target, const RigidIcpParams& params, Deadline deadline)
  : impl_(std::make_unique<Impl>(target, params, deadline)) {}
RigidRegistration::~RigidRegistration() = default;

RegistrationResult RigidRegistration::compute(const Matrix& source, const gtsam::Pose3& guess, int max_ms)
{
  RegistrationResult result;
  result.transform = guess;
  const auto start = std::chrono::steady_clock::now();
  const auto finish = [&]() {
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
  };
  if (source.cols() != 3 || !source.allFinite() ||
      source.rows() < impl_->params.min_correspondences || !guess.matrix().allFinite()) {
    result.reason = "SE3 registration requires finite source points and pose";
    return finish();
  }
  const auto& p = impl_->params;
  for (int iteration = 0; iteration < p.max_iterations; ++iteration) {
    if (std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count() >= max_ms) {
      result.reason = "SE3 registration deadline exceeded";
      result.timed_out = true;
      return finish();
    }
    const auto linear = impl_->linearize(source, result.transform);
    result.correspondences = linear.kept;
    result.observed_rank = linear.rank;
    result.overlap = static_cast<double>(linear.overlap) / source.rows();
    result.iterations = iteration + 1;
    if (linear.kept < p.min_correspondences || linear.rank < 6) {
      result.reason = "SE3 geometry does not observe all six pose directions";
      return finish();
    }
    const Eigen::LDLT<Covariance6> decomposition(linear.information);
    if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
      result.reason = "SE3 normal matrix is not positive definite";
      return finish();
    }
    const gtsam::Vector6 delta = decomposition.solve(linear.rhs);
    if (!delta.allFinite() || delta.head<3>().norm() > 0.5 || delta.tail<3>().norm() > 2.0) {
      result.reason = "SE3 optimizer step is invalid or outside its local basin";
      return finish();
    }
    const gtsam::Pose3 pivot(gtsam::Rot3(), impl_->center);
    result.transform = pivot.compose(gtsam::Pose3::Expmap(delta)).compose(
      pivot.inverse()).compose(result.transform);
    if (delta.head<3>().norm() <= p.rotation_epsilon && delta.tail<3>().norm() <= p.translation_epsilon) {
      const auto final = impl_->linearize(source, result.transform);
      result.observed_rank = final.rank;
      if (final.rank < 6 || final.kept < p.min_correspondences) break;
      result.correspondences = final.kept;
      result.overlap = static_cast<double>(final.overlap) / source.rows();
      result.rmse = std::sqrt(final.squared_error / final.kept);
      const Eigen::LDLT<Covariance6> information(final.information);
      const double variance = std::max(p.sensor_noise * p.sensor_noise,
        final.squared_error / std::max(1, final.kept - 6));
      const Covariance6 left_covariance = information.solve(Covariance6::Identity()) * variance;
      const Covariance6 adjoint = result.transform.inverse().compose(pivot).AdjointMap();
      result.covariance = adjoint * left_covariance * adjoint.transpose();
      result.covariance.diagonal().head<3>().array() += p.rotation_sigma_floor * p.rotation_sigma_floor;
      result.covariance.diagonal().tail<3>().array() += p.translation_sigma_floor * p.translation_sigma_floor;
      result.covariance = (result.covariance * 0.5 + result.covariance.transpose() * 0.5).eval();
      result.timed_out = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count() >= max_ms;
      result.converged = valid_covariance6(result.covariance) && !result.timed_out;
      result.has_covariance = result.converged;
      if (result.has_covariance) result.information = result.covariance.ldlt().solve(Covariance6::Identity());
      result.reason = result.converged ? "success" : "SE3 covariance/deadline check failed";
      return finish();
    }
  }
  result.reason = "SE3 registration iteration limit";
  return finish();
}

}  // namespace sonar_slam
