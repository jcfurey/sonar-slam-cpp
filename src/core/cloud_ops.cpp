#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/gpu.hpp"

#include <pointmatcher/PointMatcher.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

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

// ------------------------------------------------------- constrained XYZ ICP
namespace {

Eigen::Matrix3f pose_matrix(double x, double y, double yaw)
{
  const float c = static_cast<float>(std::cos(yaw));
  const float s = static_cast<float>(std::sin(yaw));
  Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
  T(0, 0) = c;
  T(0, 1) = -s;
  T(1, 0) = s;
  T(1, 1) = c;
  T(0, 2) = static_cast<float>(x);
  T(1, 2) = static_cast<float>(y);
  return T;
}

Matrix transform_xyz_planar(const Matrix& source, double x, double y,
                            double yaw)
{
  const float c = static_cast<float>(std::cos(yaw));
  const float s = static_cast<float>(std::sin(yaw));
  Matrix out = source;
  out.col(0) = c * source.col(0) - s * source.col(1);
  out.col(1) = s * source.col(0) + c * source.col(1);
  out.col(0).array() += static_cast<float>(x);
  out.col(1).array() += static_cast<float>(y);
  // source z is already expressed in the target pressure-depth chart.
  return out;
}

}  // namespace

ConstrainedIcpResult constrained_icp_xyz(
  const Matrix& source, const Matrix& target, const Eigen::Matrix3f& guess,
  const ConstrainedIcpParams& params)
{
  ConstrainedIcpResult result;
  result.T = guess;
  if (source.cols() != 3 || target.cols() != 3 || source.rows() == 0 ||
      target.rows() == 0) {
    result.message = "constrained XYZ ICP requires non-empty Nx3 clouds";
    return result;
  }
  if (!(params.max_correspondence > 0.0f) ||
      !(params.trim_ratio > 0.0 && params.trim_ratio <= 1.0) ||
      params.max_iterations < 1 || params.min_correspondences < 3) {
    result.message = "invalid constrained XYZ ICP parameters";
    return result;
  }

  double x = guess(0, 2);
  double y = guess(1, 2);
  double yaw = std::atan2(static_cast<double>(guess(1, 0)),
                          static_cast<double>(guess(0, 0)));

  for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
    const Matrix transformed = transform_xyz_planar(source, x, y, yaw);
    const auto [ids, dists] = match(
      target, transformed, 1, params.max_correspondence);

    std::vector<std::pair<float, int>> valid;
    valid.reserve(static_cast<std::size_t>(source.rows()));
    for (int i = 0; i < ids.cols(); ++i) {
      if (ids(0, i) >= 0 && std::isfinite(dists(0, i)))
        valid.emplace_back(dists(0, i), i);
    }
    const int keep = std::min<int>(
      static_cast<int>(valid.size()),
      std::max(params.min_correspondences,
               static_cast<int>(std::floor(
                 params.trim_ratio * static_cast<double>(valid.size())))));
    if (keep < params.min_correspondences) {
      result.message = "too few 3D correspondences (" +
        std::to_string(valid.size()) + ")";
      result.correspondences = static_cast<int>(valid.size());
      return result;
    }
    std::nth_element(valid.begin(), valid.begin() + keep - 1, valid.end(),
                     [](const auto& a, const auto& b) {
                       return a.first < b.first;
                     });

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    const double c = std::cos(yaw), s = std::sin(yaw);
    for (int n = 0; n < keep; ++n) {
      const int i = valid[static_cast<std::size_t>(n)].second;
      const int j = ids(0, i);
      const double px = source(i, 0), py = source(i, 1);
      const double dxdtheta = -s * px - c * py;
      const double dydtheta = c * px - s * py;
      Eigen::Matrix<double, 2, 3> J;
      J << 1.0, 0.0, dxdtheta,
           0.0, 1.0, dydtheta;
      const Eigen::Vector2d residual(
        static_cast<double>(target(j, 0) - transformed(i, 0)),
        static_cast<double>(target(j, 1) - transformed(i, 1)));
      H.noalias() += J.transpose() * J;
      b.noalias() += J.transpose() * residual;
    }

    const Eigen::LDLT<Eigen::Matrix3d> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
      result.message = "singular constrained XYZ ICP normal matrix";
      return result;
    }
    const Eigen::Vector3d delta = ldlt.solve(b);
    if (ldlt.info() != Eigen::Success || !delta.allFinite()) {
      result.message = "non-finite constrained XYZ ICP update";
      return result;
    }
    x += delta[0];
    y += delta[1];
    yaw = std::remainder(yaw + delta[2], 2.0 * M_PI);
    result.correspondences = keep;
    result.iterations = iteration + 1;
    if (delta.head<2>().norm() <= params.translation_epsilon &&
        std::abs(delta[2]) <= params.rotation_epsilon) {
      result.success = true;
      result.message = "success";
      result.T = pose_matrix(x, y, yaw);
      return result;
    }
  }

  // A bounded, finite solution at the iteration cap is still a valid ICP
  // result; the downstream overlap/covariance/maximum-correction gates decide
  // whether it is safe to become a graph factor.
  result.success = true;
  result.message = "success";
  result.T = pose_matrix(x, y, yaw);
  return result;
}

std::vector<ConstrainedIcpResult> constrained_icp_xyz_batch(
  const Matrix& source, const Matrix& target,
  const std::vector<Eigen::Matrix3f>& guesses,
  const ConstrainedIcpParams& params, int max_ms)
{
  std::vector<ConstrainedIcpResult> results(guesses.size());
  const auto start = std::chrono::steady_clock::now();
  std::vector<char> expired(guesses.size(), 0);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(guesses.size()); ++i) {
    if (i > 0 && std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start).count() >= max_ms) {
      expired[static_cast<std::size_t>(i)] = 1;
      continue;
    }
    results[static_cast<std::size_t>(i)] = constrained_icp_xyz(
      source, target, guesses[static_cast<std::size_t>(i)], params);
  }
  // Keep prefix semantics like ICP::compute_batch: a later dynamic worker may
  // have finished after an earlier slot observed the cap.
  auto first = std::find(expired.begin(), expired.end(), 1);
  if (first != expired.end())
    results.resize(static_cast<std::size_t>(first - expired.begin()));
  return results;
}

// ------------------------------------------------------------------------ ICP
struct ICP::Impl
{
  PM::ICP icp;
  // Config source for the per-thread engine pool (compute_batch): the YAML
  // TEXT cached at load_from_yaml time, never re-read from disk. A lazy disk
  // re-read could fail after startup and silently hand pool engines
  // setDefault() — whose chain includes a RandomSamplingDataPointsFilter,
  // i.e. a nondeterministic config diverging from the main engine. Empty
  // text <=> the main engine is also on setDefault(), so both stay in sync.
  std::string yaml_filename;
  std::string yaml_text;
  // lazily built per-OpenMP-thread engines. PM::ICP is not safe for
  // concurrent compute on one object, so each worker gets its own instance
  // configured identically; the pool persists across calls.
  std::vector<std::unique_ptr<PM::ICP>> pool;
  std::mutex pool_mutex;

  void configure(PM::ICP& engine) const
  {
    if (yaml_text.empty()) {
      engine.setDefault();
    } else {
      std::istringstream iss(yaml_text);
      engine.loadFromYaml(iss);
    }
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
    impl_->yaml_text.clear();
  } else {
    // cache the text once; the main engine and every pool engine configure
    // from this same string (see Impl::configure)
    std::ostringstream oss;
    oss << ifs.rdbuf();
    impl_->yaml_text = oss.str();
    std::istringstream iss(impl_->yaml_text);
    impl_->icp.loadFromYaml(iss);
    impl_->yaml_filename = filename;
  }
  // the pool engines mirror the main config — rebuild them on the next batch
  std::lock_guard<std::mutex> lock(impl_->pool_mutex);
  impl_->pool.clear();
}

std::string ICP::error_minimizer_name() const
{
  // libpointmatcher's setDefault() chain uses point-to-point
  if (impl_->yaml_text.empty()) return "PointToPointErrorMinimizer";

  // Accepts every form the shipped configs use:
  //   errorMinimizer: PointToPointErrorMinimizer
  //   errorMinimizer:
  //     PointToPointErrorMinimizer
  //   errorMinimizer:
  //     PointToPlaneErrorMinimizer:
  //       force2D: 1
  //     # PointToPointErrorMinimizer      <- comment, must NOT match
  const auto first_token = [](std::string s) -> std::string {
    const std::size_t b = s.find_first_not_of(" \t\r-");
    if (b == std::string::npos) return {};
    s = s.substr(b);
    const std::size_t e = s.find_first_of(" \t\r:");
    if (e != std::string::npos) s = s.substr(0, e);
    return s;
  };

  std::istringstream iss(impl_->yaml_text);
  std::string line;
  bool in_section = false;
  while (std::getline(iss, line)) {
    // strip comments before anything else so a commented-out alternative
    // never wins
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;

    if (!in_section) {
      const std::size_t key = line.find("errorMinimizer");
      if (key == std::string::npos) continue;
      // same-line value?
      const std::size_t colon = line.find(':', key);
      if (colon != std::string::npos) {
        const std::string tok = first_token(line.substr(colon + 1));
        if (tok.find("ErrorMinimizer") != std::string::npos) return tok;
      }
      in_section = true;
      continue;
    }
    // first non-comment token inside the section
    const std::string tok = first_token(line);
    if (tok.find("ErrorMinimizer") != std::string::npos) return tok;
    // a non-matching token means the section ended (next top-level key)
    if (!tok.empty()) break;
  }
  return "unknown";
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

  // cap-skips are recorded per index so the truncation below can reproduce
  // the sequential loop's PREFIX semantics (schedule(dynamic) otherwise keeps
  // an arbitrary subset — whichever slots opened before expiry)
  std::vector<char> cap_skipped(guesses.size(), 0);
  std::exception_ptr fatal;

  // Bound the region to the engine pool: the pool was sized from
  // omp_get_max_threads() at YAML-load time, and a later change to the OMP
  // thread budget (or a nested region) must not index past it — one engine
  // per thread is the concurrency contract, an ICP object is not safe for
  // concurrent use.
  const int batch_threads = std::max(
    1, std::min<int>(omp_get_max_threads(),
                     static_cast<int>(impl_->pool.size())));
#pragma omp parallel for schedule(dynamic) num_threads(batch_threads)
  for (int i = 0; i < static_cast<int>(guesses.size()); ++i) {
    // time cap: guesses whose slot opens after the cap are skipped, the
    // parallel analog of the sequential loop's post-sample break
    if (i > 0 && expired()) {
      cap_skipped[i] = 1;
      continue;
    }
    PM::ICP& engine = *impl_->pool[omp_get_thread_num()];
    PM::TransformationParameters T = guesses[i];
    try {
      T = engine(*pc_source, *pc_target, T);
    } catch (const PM::ConvergenceError&) {
      // a failed sample, exactly like the sequential loop
      continue;
    } catch (...) {
      // Sequential semantics: any non-convergence exception is frame-fatal
      // (it propagates out of compute_icp_with_cov to the frame-level catch).
      // It cannot escape an OpenMP region (std::terminate), so capture the
      // first one and rethrow after the loop instead of silently degrading
      // to "Too few samples".
#pragma omp critical(sonar_slam_icp_batch_fatal)
      if (!fatal) fatal = std::current_exception();
      continue;
    }
    results[i].success = true;
    results[i].T = T;
  }
  if (fatal) std::rethrow_exception(fatal);

  // Prefix truncation: drop everything at/after the FIRST cap-skipped guess.
  // The sequential loop's break keeps the best-cost prefix of the (cost-
  // sorted) guesses; without this, a fired cap would keep an arbitrary,
  // schedule-dependent subset and the MCD location/covariance would ride on
  // scheduling noise.
  for (std::size_t e = 0; e < cap_skipped.size(); ++e) {
    if (cap_skipped[e]) {
      for (std::size_t j = e; j < results.size(); ++j) results[j].success = false;
      break;
    }
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
