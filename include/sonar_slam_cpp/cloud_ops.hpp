// Point-cloud operations: port of bruce_slam cpp/pcl.cpp (libpointmatcher +
// PCL) without the pybind layer. Clouds are Eigen row-per-point matrices with
// 2 or 3 columns, matching the numpy layout the Python stack used.
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sonar_slam {

using Matrix = Eigen::MatrixXf;

// Voxel (octree-grid) downsampling, samplingMethod=3 (one point per cell).
Matrix downsample(const Matrix& mat_in, float resolution);

// Same, carrying per-point descriptors (e.g. keyframe keys) through.
std::pair<Matrix, Matrix> downsample(const Matrix& mat_in, const Matrix& desc_in,
                                     float resolution);

// PCL radius outlier removal.
Matrix remove_outlier(const Matrix& mat_in, double radius, int min_points);

// KNN match of mat_in against mat_ref within max_dist; ids are -1 when
// unmatched. Returns (ids, squared dists), each knn x N like libpointmatcher.
std::pair<Eigen::MatrixXi, Matrix> match(const Matrix& mat_ref,
                                         const Matrix& mat_in, int knn,
                                         float max_dist);

// libpointmatcher ICP wrapper with the same YAML config as the Python stack.
class ICP
{
public:
  ICP();
  ~ICP();
  ICP(ICP&&) noexcept;
  ICP& operator=(ICP&&) noexcept;

  // falls back to the default libpointmatcher chain when the file is missing
  void load_from_yaml(const std::string& filename);

  // returns ("success", T) or (convergence error message, guess)
  std::pair<std::string, Eigen::Matrix3f> compute(const Matrix& source,
                                                  const Matrix& target,
                                                  const Eigen::Matrix3f& guess);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonar_slam
