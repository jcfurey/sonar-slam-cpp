#include "sonar_slam_cpp/global_init.hpp"
#include "sonar_slam_cpp/gpu.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sonar_slam {

namespace {

// ---------------------------------------------------------------------------
// Sobol sequence for d <= 3 (Joe & Kuo direction numbers). Deterministic, so
// runs are reproducible — matching shgo's sampling_method="sobol" in spirit.
// ---------------------------------------------------------------------------
class Sobol3
{
public:
  Sobol3()
  {
    // dim 0: van der Corput (all m = 1)
    for (int b = 0; b < BITS; ++b) v_[0][b] = 1u << (31 - b);
    // dim 1: s=1, a=0, m=[1]
    init_dim(1, 1, 0, {1});
    // dim 2: s=2, a=1, m=[1, 3]
    init_dim(2, 2, 1, {1, 3});
  }

  std::array<double, 3> next()
  {
    // Gray-code increment
    unsigned c = 0, value = index_++;
    while (value & 1) { value >>= 1; ++c; }
    std::array<double, 3> out;
    for (int d = 0; d < 3; ++d) {
      x_[d] ^= v_[d][c];
      out[d] = x_[d] * (1.0 / 4294967296.0);
    }
    return out;
  }

private:
  static constexpr int BITS = 32;

  void init_dim(int d, int s, unsigned a, std::initializer_list<unsigned> m_init)
  {
    std::vector<unsigned> m(m_init);
    for (int b = 0; b < BITS; ++b) {
      if (b < s) {
        v_[d][b] = m[b] << (31 - b);
      } else {
        unsigned value = v_[d][b - s] ^ (v_[d][b - s] >> s);
        for (int k = 1; k < s; ++k)
          if ((a >> (s - 1 - k)) & 1) value ^= v_[d][b - k];
        v_[d][b] = value;
      }
    }
  }

  unsigned v_[3][BITS] = {};
  unsigned x_[3] = {0, 0, 0};
  unsigned index_ = 0;
};

struct CostGrid
{
  cv::Mat grid;  // CV_8UC1, 0/255 after dilation
  double xmin = 0.0, ymin = 0.0, resolution = 1.0;
};

// port of slam.py lines 503-526: rasterize target points, dilate by the
// point-noise radius
CostGrid build_grid(const Matrix& target_points, double point_noise)
{
  CostGrid g;
  const double xmin = target_points.col(0).minCoeff() - 2.0 * point_noise;
  const double ymin = target_points.col(1).minCoeff() - 2.0 * point_noise;
  const double xmax = target_points.col(0).maxCoeff() + 2.0 * point_noise;
  const double ymax = target_points.col(1).maxCoeff() + 2.0 * point_noise;
  g.resolution = point_noise / 10.0;
  g.xmin = xmin;
  g.ymin = ymin;
  const int cols = std::max(1, static_cast<int>(std::ceil((xmax - xmin) / g.resolution)));
  const int rows = std::max(1, static_cast<int>(std::ceil((ymax - ymin) / g.resolution)));
  g.grid = cv::Mat::zeros(rows, cols, CV_8UC1);

  for (int i = 0; i < target_points.rows(); ++i) {
    int r = static_cast<int>(std::lround((target_points(i, 1) - ymin) / g.resolution));
    int c = static_cast<int>(std::lround((target_points(i, 0) - xmin) / g.resolution));
    r = std::clamp(r, 0, rows - 1);
    c = std::clamp(c, 0, cols - 1);
    g.grid.at<std::uint8_t>(r, c) = 255;
  }

  const int dilate_hs = static_cast<int>(std::ceil(point_noise / g.resolution));
  const int dilate_size = 2 * dilate_hs + 1;
  cv::Mat kernel = cv::getStructuringElement(
    cv::MORPH_ELLIPSE, cv::Size(dilate_size, dilate_size),
    cv::Point(dilate_hs, dilate_hs));
  cv::dilate(g.grid, g.grid, kernel);
  return g;
}

// sample_transform = target_pose.between(source_pose.compose(delta)) packed as
// [r00 r01 r10 r11 tx ty] for the kernels
std::array<float, 6> pack_transform(const gtsam::Pose2& source_pose,
                                    const gtsam::Pose2& target_pose,
                                    const Eigen::Vector3d& delta)
{
  const gtsam::Pose2 sample_source =
    source_pose.compose(gtsam::Pose2(delta[0], delta[1], delta[2]));
  const gtsam::Pose2 T = target_pose.between(sample_source);
  const double c = std::cos(T.theta()), s = std::sin(T.theta());
  return {static_cast<float>(c),        static_cast<float>(-s),
          static_cast<float>(s),        static_cast<float>(c),
          static_cast<float>(T.x()),    static_cast<float>(T.y())};
}

void eval_costs_cpu(const Matrix& points, const std::vector<std::array<float, 6>>& Ts,
                    const CostGrid& g, std::vector<float>& costs)
{
  const int rows = g.grid.rows, cols = g.grid.cols;
  const float xmin = static_cast<float>(g.xmin), ymin = static_cast<float>(g.ymin);
  const float inv_res = 1.0f / static_cast<float>(g.resolution);
  costs.resize(Ts.size());

#pragma omp parallel for schedule(static)
  for (int t = 0; t < static_cast<int>(Ts.size()); ++t) {
    const auto& T = Ts[t];
    int hits = 0;
    for (int i = 0; i < points.rows(); ++i) {
      const float px = points(i, 0), py = points(i, 1);
      const float x = T[0] * px + T[1] * py + T[4];
      const float y = T[2] * px + T[3] * py + T[5];
      const int r = static_cast<int>(std::lround((y - ymin) * inv_res));
      const int c = static_cast<int>(std::lround((x - xmin) * inv_res));
      if (r >= 0 && r < rows && c >= 0 && c < cols &&
          g.grid.at<std::uint8_t>(r, c) > 0)
        ++hits;
    }
    costs[t] = -static_cast<float>(hits);
  }
}

void eval_costs(const Matrix& points, const std::vector<std::array<float, 6>>& Ts,
                const CostGrid& g, std::vector<float>& costs)
{
  costs.resize(Ts.size());
  if (Ts.empty() || points.rows() == 0) {
    std::fill(costs.begin(), costs.end(), 0.f);
    return;
  }
#ifdef SONAR_SLAM_WITH_CUDA
  if (gpu::available() && Ts.size() >= 8) {
    // row-major point buffer
    std::vector<float> pts(points.rows() * 2);
    for (int i = 0; i < points.rows(); ++i) {
      pts[2 * i] = points(i, 0);
      pts[2 * i + 1] = points(i, 1);
    }
    gpu::grid_cost_cuda(pts.data(), points.rows(), Ts[0].data(),
                        static_cast<int>(Ts.size()), g.grid.ptr<std::uint8_t>(),
                        g.grid.rows, g.grid.cols, static_cast<float>(g.xmin),
                        static_cast<float>(g.ymin),
                        static_cast<float>(g.resolution), costs.data());
    return;
  }
#endif
  eval_costs_cpu(points, Ts, g, costs);
}

}  // namespace

std::vector<float> global_match_costs(
  const Matrix& source_points, const gtsam::Pose2& source_pose,
  const Matrix& target_points, const gtsam::Pose2& target_pose,
  double point_noise, const std::vector<Eigen::Vector3d>& deltas)
{
  const CostGrid grid = build_grid(target_points, point_noise);
  std::vector<std::array<float, 6>> transforms;
  transforms.reserve(deltas.size());
  for (const auto& d : deltas)
    transforms.push_back(pack_transform(source_pose, target_pose, d));
  std::vector<float> costs;
  eval_costs(source_points, transforms, grid, costs);
  return costs;
}

GlobalInitResult global_scan_match_init(
  const Matrix& source_points, const gtsam::Pose2& source_pose,
  const Matrix& target_points, const gtsam::Pose2& target_pose,
  double point_noise, const Eigen::Matrix<double, 3, 2>& bounds, int n,
  int iters, double ftol)
{
  GlobalInitResult result;
  if (source_points.rows() == 0 || target_points.rows() == 0) return result;

  const CostGrid grid = build_grid(target_points, point_noise);

  auto record = [&](const Eigen::Vector3d& delta, double cost) {
    const gtsam::Pose2 sample_source =
      source_pose.compose(gtsam::Pose2(delta[0], delta[1], delta[2]));
    result.pose_samples.emplace_back(sample_source.x(), sample_source.y(),
                                     sample_source.theta(), cost);
  };

  // ------------------------------------------------- sobol sampling phase
  Sobol3 sobol;
  const int total = std::max(1, n) * std::max(1, iters);
  std::vector<Eigen::Vector3d> deltas;
  std::vector<std::array<float, 6>> transforms;
  deltas.reserve(total);
  transforms.reserve(total);
  for (int i = 0; i < total; ++i) {
    const auto u = sobol.next();
    Eigen::Vector3d d;
    for (int k = 0; k < 3; ++k)
      d[k] = bounds(k, 0) + u[k] * (bounds(k, 1) - bounds(k, 0));
    deltas.push_back(d);
    transforms.push_back(pack_transform(source_pose, target_pose, d));
  }

  std::vector<float> costs;
  eval_costs(source_points, transforms, grid, costs);

  int best = 0;
  for (int i = 0; i < total; ++i) {
    record(deltas[i], costs[i]);
    if (costs[i] < costs[best]) best = i;
  }

  // ------------------------------------------- local Nelder-Mead refinement
  // (replaces shgo's local minimization; the cost is piecewise constant so a
  // derivative-free simplex with an ftol stop criterion is appropriate)
  auto cost_of = [&](const Eigen::Vector3d& d) {
    std::vector<std::array<float, 6>> one{pack_transform(source_pose, target_pose, d)};
    std::vector<float> c;
    eval_costs_cpu(source_points, one, grid, c);
    record(d, c[0]);
    return static_cast<double>(c[0]);
  };

  std::array<Eigen::Vector3d, 4> simplex;
  std::array<double, 4> fvals;
  simplex[0] = deltas[best];
  fvals[0] = costs[best];
  for (int k = 0; k < 3; ++k) {
    simplex[k + 1] = deltas[best];
    simplex[k + 1][k] += 0.05 * (bounds(k, 1) - bounds(k, 0));
    fvals[k + 1] = cost_of(simplex[k + 1]);
  }

  const int max_nm_iters = 120;
  for (int it = 0; it < max_nm_iters; ++it) {
    // order simplex
    std::array<int, 4> order = {0, 1, 2, 3};
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return fvals[a] < fvals[b]; });
    std::array<Eigen::Vector3d, 4> sx;
    std::array<double, 4> sf;
    for (int k = 0; k < 4; ++k) { sx[k] = simplex[order[k]]; sf[k] = fvals[order[k]]; }
    simplex = sx;
    fvals = sf;

    if (std::abs(fvals[3] - fvals[0]) <= ftol) break;

    const Eigen::Vector3d centroid = (simplex[0] + simplex[1] + simplex[2]) / 3.0;
    const Eigen::Vector3d refl = centroid + (centroid - simplex[3]);
    const double f_refl = cost_of(refl);
    if (f_refl < fvals[0]) {
      const Eigen::Vector3d exp_pt = centroid + 2.0 * (centroid - simplex[3]);
      const double f_exp = cost_of(exp_pt);
      if (f_exp < f_refl) { simplex[3] = exp_pt; fvals[3] = f_exp; }
      else { simplex[3] = refl; fvals[3] = f_refl; }
    } else if (f_refl < fvals[2]) {
      simplex[3] = refl;
      fvals[3] = f_refl;
    } else {
      const Eigen::Vector3d contr = centroid + 0.5 * (simplex[3] - centroid);
      const double f_contr = cost_of(contr);
      if (f_contr < fvals[3]) { simplex[3] = contr; fvals[3] = f_contr; }
      else {
        for (int k = 1; k < 4; ++k) {
          simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0]);
          fvals[k] = cost_of(simplex[k]);
        }
      }
    }
  }

  int best_v = 0;
  for (int k = 1; k < 4; ++k)
    if (fvals[k] < fvals[best_v]) best_v = k;

  result.success = true;
  result.delta = gtsam::Pose2(simplex[best_v][0], simplex[best_v][1], simplex[best_v][2]);
  result.best_cost = fvals[best_v];
  return result;
}

}  // namespace sonar_slam
