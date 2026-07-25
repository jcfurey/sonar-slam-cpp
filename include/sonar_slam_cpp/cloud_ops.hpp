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

  // Class name of the CONFIGURED error minimizer, e.g.
  // "PointToPointErrorMinimizer". Read from the cached config text, so it
  // reflects the chain actually in force rather than an assumption. An empty
  // config (setDefault) reports libpointmatcher's default, which is
  // point-to-point. Used by Slam::configure to decide whether the Censi
  // covariance helper's point-to-point Hessian matches the running objective.
  std::string error_minimizer_name() const;

  // returns ("success", T) or (convergence error message, guess)
  std::pair<std::string, Eigen::Matrix3f> compute(const Matrix& source,
                                                  const Matrix& target,
                                                  const Eigen::Matrix3f& guess);

  // One registration per guess for the sampled-covariance path, run in
  // parallel across a persistent per-thread engine pool (the configured chain
  // is deterministic, so each guess's result is identical to compute()'s).
  // Guesses whose slot opens after max_ms elapsed are skipped — the parallel
  // analog of the sequential loop's time cap. Results keep guess order;
  // entries that failed to converge or were skipped have success == false.
  // Without OpenMP this degrades to the sequential loop.
  struct BatchResult
  {
    bool success = false;
    Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
  };
  std::vector<BatchResult> compute_batch(const Matrix& source,
                                         const Matrix& target,
                                         const std::vector<Eigen::Matrix3f>& guesses,
                                         int max_ms);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonar_slam
