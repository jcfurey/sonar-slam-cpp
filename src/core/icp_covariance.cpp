#include "sonar_slam_cpp/icp_covariance.hpp"

#include "sonar_slam_cpp/keyframe.hpp"  // Keyframe::transform_points

#include <vector>

namespace sonar_slam {

CensiCovResult censi_icp_covariance(const Matrix& source, const Matrix& target,
                                    const gtsam::Pose2& transform,
                                    double max_dist, double sensor_var,
                                    int min_correspondences)
{
  CensiCovResult result;
  if (source.rows() == 0 || target.rows() == 0) return result;

  // associate like ICP: nearest target point to each transformed source point
  const Matrix src_t = Keyframe::transform_points(source, transform);
  auto [ids, dists] = match(target.leftCols(2), src_t.leftCols(2), 1,
                            static_cast<float>(max_dist));

  std::vector<int> matched;
  matched.reserve(ids.cols());
  for (int j = 0; j < ids.cols(); ++j)
    if (ids(0, j) != -1) matched.push_back(j);
  if (static_cast<int>(matched.size()) < min_correspondences) return result;

  // the Jacobian uses the raw (untransformed) source points
  Matrix src_matched(static_cast<long>(matched.size()), 2);
  for (std::size_t k = 0; k < matched.size(); ++k)
    src_matched.row(static_cast<long>(k)) = source.row(matched[k]).leftCols(2);

  Eigen::Matrix3d cov;
  if (!censi_covariance(src_matched, transform.theta(), sensor_var, cov))
    return result;

  result.success = true;
  result.cov = cov;
  result.n_correspondences = static_cast<int>(matched.size());
  return result;
}

}  // namespace sonar_slam
