#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/gpu.hpp"

#include <pointmatcher/PointMatcher.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>

namespace sonar_slam {

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;
typedef std::shared_ptr<PM::DataPoints> DPPtr;

namespace {

DPPtr from_eigen(const Matrix& mat)
{
  DP::Labels labels;
  labels.push_back(DP::Label("x", 1));
  labels.push_back(DP::Label("y", 1));
  if (mat.cols() == 3) labels.push_back(DP::Label("z", 1));
  labels.push_back(DP::Label("pad", 1));

  PM::Matrix padded_mat = PM::Matrix::Ones(labels.size(), mat.rows());
  padded_mat.block(0, 0, labels.size() - 1, mat.rows()) = mat.transpose();

  return DPPtr(new DP(padded_mat, labels));
}

DPPtr from_eigen(const Matrix& mat, const Matrix& desc)
{
  DP::Labels labels;
  labels.push_back(DP::Label("x", 1));
  labels.push_back(DP::Label("y", 1));
  if (mat.cols() == 3) labels.push_back(DP::Label("z", 1));
  labels.push_back(DP::Label("pad", 1));

  PM::Matrix padded_mat = PM::Matrix::Ones(labels.size(), mat.rows());
  padded_mat.block(0, 0, labels.size() - 1, mat.rows()) = mat.transpose();

  DP::Labels desc_labels;
  for (int col = 0; col < desc.cols(); ++col)
    desc_labels.push_back(DP::Label("desc" + std::to_string(col), 1));

  return DPPtr(new DP(padded_mat, labels, desc.transpose(), desc_labels));
}

}  // namespace

Matrix downsample(const Matrix& mat_in, float resolution)
{
  if (mat_in.rows() == 0) return mat_in;

  PointMatcherSupport::Parametrizable::Parameters params;
  params["maxSizeByNode"] = std::to_string(resolution);
  params["samplingMethod"] = "3";
  std::shared_ptr<PM::DataPointsFilter> filter =
    PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", params);
  DPPtr cloud_in = from_eigen(mat_in);
  filter->inPlaceFilter(*cloud_in);
  return cloud_in->features.topRows(mat_in.cols()).transpose();
}

std::pair<Matrix, Matrix> downsample(const Matrix& mat_in, const Matrix& desc_in,
                                     float resolution)
{
  if (mat_in.rows() == 0) return std::make_pair(mat_in, desc_in);

  PointMatcherSupport::Parametrizable::Parameters params;
  params["maxSizeByNode"] = std::to_string(resolution);
  params["samplingMethod"] = "3";
  std::shared_ptr<PM::DataPointsFilter> filter =
    PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", params);
  DPPtr cloud_in = from_eigen(mat_in, desc_in);
  filter->inPlaceFilter(*cloud_in);

  Matrix mat_out = cloud_in->features.topRows(mat_in.cols()).transpose();
  Matrix desc_out = cloud_in->descriptors.transpose();
  return std::make_pair(mat_out, desc_out);
}

Matrix remove_outlier(const Matrix& mat_in, double radius, int min_points)
{
  if (mat_in.rows() == 0) return mat_in;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(
    new pcl::PointCloud<pcl::PointXYZ>(mat_in.rows(), 1));
  for (int row = 0; row < mat_in.rows(); ++row) {
    cloud_in->at(row).x = mat_in(row, 0);
    cloud_in->at(row).y = mat_in(row, 1);
    cloud_in->at(row).z = mat_in.cols() == 3 ? mat_in(row, 2) : 0.f;
  }

  pcl::RadiusOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_in);
  sor.setRadiusSearch(radius);
  sor.setMinNeighborsInRadius(min_points);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
  sor.filter(*cloud_out);

  Matrix mat_out(cloud_out->size(), mat_in.cols());
  for (std::size_t row = 0; row < cloud_out->size(); ++row) {
    mat_out(row, 0) = cloud_out->at(row).x;
    mat_out(row, 1) = cloud_out->at(row).y;
    if (mat_in.cols() == 3) mat_out(row, 2) = cloud_out->at(row).z;
  }
  return mat_out;
}

std::pair<Eigen::MatrixXi, Matrix> match(const Matrix& mat_ref,
                                         const Matrix& mat_in, int knn,
                                         float max_dist)
{
#ifdef SONAR_SLAM_WITH_CUDA
  // GPU brute-force twin for the 1-NN case (every caller: the overlap
  // estimates and the Censi correspondences). Exact — scans all reference
  // points, so it can only differ from the KDTree by ULP-level distance
  // rounding on a point sitting exactly at max_dist. Below the size gate the
  // KDTree wins on transfer/launch overhead; on any device failure the
  // wrapper returns false and the CPU path below runs.
  if (knn == 1 && mat_ref.cols() == 2 && mat_in.cols() == 2 &&
      gpu::available() &&
      static_cast<std::size_t>(mat_ref.rows()) *
          static_cast<std::size_t>(mat_in.rows()) >=
        static_cast<std::size_t>(1) << 16) {
    // row-major x,y buffers (Matrix is row-per-point)
    std::vector<float> ref(2 * mat_ref.rows()), query(2 * mat_in.rows());
    for (int i = 0; i < mat_ref.rows(); ++i) {
      ref[2 * i] = mat_ref(i, 0);
      ref[2 * i + 1] = mat_ref(i, 1);
    }
    for (int i = 0; i < mat_in.rows(); ++i) {
      query[2 * i] = mat_in(i, 0);
      query[2 * i + 1] = mat_in(i, 1);
    }
    std::vector<int> ids(mat_in.rows());
    std::vector<float> dists2(mat_in.rows());
    if (gpu::nn1_cuda(ref.data(), static_cast<int>(mat_ref.rows()),
                      query.data(), static_cast<int>(mat_in.rows()), max_dist,
                      ids.data(), dists2.data())) {
      Eigen::MatrixXi ids_out(1, mat_in.rows());
      Matrix dists_out(1, mat_in.rows());
      for (int i = 0; i < mat_in.rows(); ++i) {
        ids_out(0, i) = ids[i];
        dists_out(0, i) = dists2[i];
      }
      return std::make_pair(ids_out, dists_out);
    }
  }
#endif

  PointMatcherSupport::Parametrizable::Parameters params;
  params["knn"] = std::to_string(knn);
  params["maxDist"] = std::to_string(max_dist);
  std::shared_ptr<PM::Matcher> matcher =
    PM::get().MatcherRegistrar.create("KDTreeMatcher", params);

  DPPtr cloud_ref = from_eigen(mat_ref);
  DPPtr cloud_in = from_eigen(mat_in);

  matcher->init(*cloud_ref);
  PM::Matches matches = matcher->findClosests(*cloud_in);
  return std::make_pair(matches.ids, matches.dists);
}

// ------------------------------------------------------------------------ ICP
struct ICP::Impl
{
  PM::ICP icp;
  // config source for the per-thread engine pool (compute_batch); empty ->
  // libpointmatcher defaults, mirroring load_from_yaml's fallback
  std::string yaml_filename;
  // lazily built per-OpenMP-thread engines. PM::ICP is not safe for
  // concurrent compute on one object, so each worker gets its own instance
  // configured identically; the pool persists across calls.
  std::vector<std::unique_ptr<PM::ICP>> pool;
  std::mutex pool_mutex;

  void configure(PM::ICP& engine) const
  {
    std::ifstream ifs(yaml_filename);
    if (yaml_filename.empty() || !ifs.is_open())
      engine.setDefault();
    else
      engine.loadFromYaml(ifs);
  }
};

ICP::ICP() : impl_(new Impl) { impl_->icp.setDefault(); }
ICP::~ICP() = default;
ICP::ICP(ICP&&) noexcept = default;
ICP& ICP::operator=(ICP&&) noexcept = default;

void ICP::load_from_yaml(const std::string& filename)
{
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cout << "Failed to load " << filename << ". Use default configuration."
              << std::endl;
    impl_->icp.setDefault();
    impl_->yaml_filename.clear();
  } else {
    impl_->icp.loadFromYaml(ifs);
    impl_->yaml_filename = filename;
  }
  // the pool engines mirror the main config — rebuild them on the next batch
  std::lock_guard<std::mutex> lock(impl_->pool_mutex);
  impl_->pool.clear();
}

std::pair<std::string, Eigen::Matrix3f> ICP::compute(const Matrix& source,
                                                     const Matrix& target,
                                                     const Eigen::Matrix3f& guess)
{
  DPPtr pc_source = from_eigen(source);
  DPPtr pc_target = from_eigen(target);
  PM::TransformationParameters T = guess;
  try {
    T = impl_->icp(*pc_source, *pc_target, T);
  } catch (const PM::ConvergenceError& e) {
    return std::make_pair(std::string(e.what()), Eigen::Matrix3f(guess));
  }
  return std::make_pair(std::string("success"), Eigen::Matrix3f(T));
}

std::vector<ICP::BatchResult> ICP::compute_batch(
  const Matrix& source, const Matrix& target,
  const std::vector<Eigen::Matrix3f>& guesses, int max_ms)
{
  std::vector<BatchResult> results(guesses.size());

#ifdef _OPENMP
  {
    // one engine per potential worker, configured like the main engine.
    // Registrations are deterministic given the config (the chain has no
    // sampling filters), so per-guess results are identical to running them
    // through the shared engine sequentially.
    std::lock_guard<std::mutex> lock(impl_->pool_mutex);
    const int n_threads = omp_get_max_threads();
    while (static_cast<int>(impl_->pool.size()) < n_threads) {
      auto engine = std::make_unique<PM::ICP>();
      impl_->configure(*engine);
      impl_->pool.push_back(std::move(engine));
    }
  }

  // the clouds are read-only across workers; each engine copies internally
  DPPtr pc_source = from_eigen(source);
  DPPtr pc_target = from_eigen(target);

  const auto start = std::chrono::steady_clock::now();
  const auto expired = [&]() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start)
             .count() >= max_ms;
  };

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(guesses.size()); ++i) {
    // time cap: guesses whose slot opens after the cap are skipped, the
    // parallel analog of the sequential loop's post-sample break
    if (i > 0 && expired()) continue;
    PM::ICP& engine = *impl_->pool[omp_get_thread_num()];
    PM::TransformationParameters T = guesses[i];
    try {
      T = engine(*pc_source, *pc_target, T);
    } catch (...) {
      // convergence errors mark the sample failed like the sequential loop;
      // anything else must ALSO be swallowed here — an exception escaping an
      // OpenMP region is std::terminate, not a throw to the caller
      continue;
    }
    results[i].success = true;
    results[i].T = T;
  }
#else
  // no OpenMP: identical to the historical sequential sampling loop
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < guesses.size(); ++i) {
    auto [message, T] = compute(source, target, guesses[i]);
    if (message == "success") {
      results[i].success = true;
      results[i].T = T;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
    if (elapsed.count() >= max_ms) break;
  }
#endif

  return results;
}

}  // namespace sonar_slam
