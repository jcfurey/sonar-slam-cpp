// Map-quality metrics for the MAP_DOUBLING_FIX_PLAN §5 replay protocol:
// turns "did the doubling improve?" from an eyeball comparison into numbers.
//
// Subscribes to the aggregated SLAM map cloud (latched, so it also works
// against a recorded bag of the topic) and, on every update, rasterizes the
// x/y projection, skeletonizes the walls, and reports:
//
//   wall_len    total skeleton length (m)
//   thick_med / thick_p95
//               local wall thickness (m): 2x distance-to-free at each
//               skeleton cell (the medial-axis definition). The published
//               cloud is octree-downsampled per keyframe, but cell centroids
//               of a thin wall stay on the wall plane, so the projected
//               thickness resolves well below the downsample cell size.
//   doubled     fraction of skeleton length that has a second, parallel
//               (within 20 deg) wall in the probe band along its normal —
//               the doubled-wall signature. Band default 0.3–3.0 m, the
//               scale the CHL_Pool doubling was observed at.
//
// Usage (defaults in brackets):
//   ros2 run sonar_slam_cpp map_metrics [--topic /bruce/slam/slam/cloud]
//        [--grid 0.05] [--band 0.3 3.0] [--once]
//
// Replay a recorded run:   ros2 bag play <bag>   (cloud topic included)
// and read the last line printed — each line is one cloud update.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"

namespace sonar_slam {

namespace {

// Zhang–Suen thinning of a 0/1 uint8 image to a 1-px skeleton.
void thin(cv::Mat& img)
{
  CV_Assert(img.type() == CV_8UC1);
  cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
  bool changed = true;
  auto at = [&](int r, int c) -> int { return img.at<uint8_t>(r, c); };
  while (changed) {
    changed = false;
    for (int step = 0; step < 2; ++step) {
      marker.setTo(0);
      for (int r = 1; r < img.rows - 1; ++r) {
        for (int c = 1; c < img.cols - 1; ++c) {
          if (!at(r, c)) continue;
          const int p2 = at(r - 1, c), p3 = at(r - 1, c + 1), p4 = at(r, c + 1),
                    p5 = at(r + 1, c + 1), p6 = at(r + 1, c),
                    p7 = at(r + 1, c - 1), p8 = at(r, c - 1),
                    p9 = at(r - 1, c - 1);
          const int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (b < 2 || b > 6) continue;
          const int a = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
          if (a != 1) continue;
          if (step == 0) {
            if (p2 * p4 * p6 != 0 || p4 * p6 * p8 != 0) continue;
          } else {
            if (p2 * p4 * p8 != 0 || p2 * p6 * p8 != 0) continue;
          }
          marker.at<uint8_t>(r, c) = 1;
          changed = true;
        }
      }
      img -= marker;
    }
  }
}

// exact Euclidean distance-to-free (cell units) via two-pass sweep over
// squared distances (Felzenszwalb-style row/column decomposition is overkill
// at these sizes; a chamfer 3-4 approximation would bias the thickness stat,
// so do the exact per-row 1D transform + column combine)
cv::Mat distance_to_free(const cv::Mat& occ)
{
  const int rows = occ.rows, cols = occ.cols;
  const float INF = 1e20f;
  // 1D squared distance along columns first: nearest free in the same column
  cv::Mat g(rows, cols, CV_32F);
  for (int c = 0; c < cols; ++c) {
    float d = INF;
    for (int r = 0; r < rows; ++r) {
      d = occ.at<uint8_t>(r, c) ? d + 1.0f : 0.0f;
      g.at<float>(r, c) = d;
    }
    d = INF;
    for (int r = rows - 1; r >= 0; --r) {
      d = occ.at<uint8_t>(r, c) ? d + 1.0f : 0.0f;
      g.at<float>(r, c) = std::min(g.at<float>(r, c), d);
    }
  }
  // combine across columns with the 1D lower-envelope transform per row
  cv::Mat dist(rows, cols, CV_32F);
  std::vector<int> v(cols);
  std::vector<float> z(cols + 1), f(cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const float gc = g.at<float>(r, c);
      f[c] = gc >= INF ? INF : gc * gc;
    }
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (int q = 1; q < cols; ++q) {
      if (f[q] >= INF && f[v[k]] >= INF) {
        // both parabolas at infinity — keep the envelope stable
        continue;
      }
      float s;
      while (true) {
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * (q - v[k]));
        if (s <= z[k] && k > 0) {
          --k;
        } else {
          break;
        }
      }
      ++k;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    k = 0;
    for (int q = 0; q < cols; ++q) {
      while (z[k + 1] < q) ++k;
      const float dq = static_cast<float>(q - v[k]);
      const float val = f[v[k]] >= INF ? INF : dq * dq + f[v[k]];
      dist.at<float>(r, q) = val >= INF ? INF : std::sqrt(val);
    }
  }
  return dist;
}

struct Metrics
{
  long points = 0;
  long occupied_cells = 0;
  double wall_len = 0.0;
  double thick_med = 0.0;
  double thick_p95 = 0.0;
  double doubled_frac = 0.0;
  long skel_cells = 0;
};

Metrics analyze(const Matrix& xyz, double grid, double band_min, double band_max)
{
  Metrics m;
  m.points = xyz.rows();
  if (m.points == 0) return m;

  // bounds of the finite x/y projection
  float min_x = 1e30f, max_x = -1e30f, min_y = 1e30f, max_y = -1e30f;
  for (long i = 0; i < xyz.rows(); ++i) {
    const float x = xyz(i, 0), y = xyz(i, 1);
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }
  if (min_x > max_x) return m;

  // auto-coarsen so pathological extents cannot allocate an absurd raster
  constexpr int kMaxDim = 4096;
  double g = grid;
  while ((max_x - min_x) / g + 3 > kMaxDim || (max_y - min_y) / g + 3 > kMaxDim)
    g *= 2.0;
  if (g != grid)
    std::fprintf(stderr,
                 "[map_metrics] extent %.0fx%.0f m: grid coarsened %.3f -> "
                 "%.3f m\n",
                 max_x - min_x, max_y - min_y, grid, g);

  const int cols = static_cast<int>((max_x - min_x) / g) + 3;
  const int rows = static_cast<int>((max_y - min_y) / g) + 3;
  cv::Mat occ = cv::Mat::zeros(rows, cols, CV_8UC1);
  for (long i = 0; i < xyz.rows(); ++i) {
    const float x = xyz(i, 0), y = xyz(i, 1);
    if (!std::isfinite(x) || !std::isfinite(y)) continue;
    const int c = static_cast<int>((x - min_x) / g) + 1;
    const int r = static_cast<int>((y - min_y) / g) + 1;
    occ.at<uint8_t>(r, c) = 1;
  }
  m.occupied_cells = cv::countNonZero(occ);

  const cv::Mat dist = distance_to_free(occ);
  cv::Mat skel = occ.clone();
  thin(skel);

  // skeleton cells + local tangent orientation (PCA over a small window)
  struct SkelPx
  {
    int r, c;
    float theta;   // wall tangent, [-pi/2, pi/2)
    float thick;   // meters
  };
  std::vector<SkelPx> px;
  std::vector<cv::Point> skel_pts;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      if (skel.at<uint8_t>(r, c)) skel_pts.emplace_back(c, r);
  if (skel_pts.empty()) return m;

  const int w = std::max(2, static_cast<int>(std::lround(0.3 / g)));
  px.reserve(skel_pts.size());
  for (const auto& p : skel_pts) {
    double sx = 0, sy = 0, n = 0;
    for (int dr = -w; dr <= w; ++dr)
      for (int dc = -w; dc <= w; ++dc) {
        const int r = p.y + dr, c = p.x + dc;
        if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
        if (!skel.at<uint8_t>(r, c)) continue;
        sx += dc;
        sy += dr;
        n += 1;
      }
    // second moments about the window centroid
    double sxx = 0, syy = 0, sxy = 0;
    const double mx = sx / n, my = sy / n;
    for (int dr = -w; dr <= w; ++dr)
      for (int dc = -w; dc <= w; ++dc) {
        const int r = p.y + dr, c = p.x + dc;
        if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
        if (!skel.at<uint8_t>(r, c)) continue;
        sxx += (dc - mx) * (dc - mx);
        syy += (dr - my) * (dr - my);
        sxy += (dc - mx) * (dr - my);
      }
    const float theta =
      0.5f * std::atan2(2.0 * sxy, sxx - syy);  // principal direction
    const float d = dist.at<float>(p.y, p.x);
    px.push_back({p.y, p.x, theta,
                  static_cast<float>(2.0 * std::max(0.0f, d - 0.5f) * g)});
  }

  // thickness stats
  std::vector<float> th;
  th.reserve(px.size());
  for (const auto& s : px) th.push_back(s.thick);
  std::sort(th.begin(), th.end());
  m.thick_med = th[th.size() / 2];
  m.thick_p95 = th[static_cast<std::size_t>(th.size() * 0.95)];
  // sum actual 8-connected step lengths — counting every pixel as g reads a
  // 45-degree wall 29% short
  //
  // Each edge's length is also split evenly between its two endpoint cells,
  // giving a per-cell length weight that sums back to `len`. The doubling
  // probe below needs it: reporting doubled/px.size() would be a CELL-COUNT
  // fraction while the metric is documented (and named) as a fraction of
  // skeleton LENGTH, and a diagonal cell carries sqrt(2) times the length of
  // an axis-aligned one. Since a graph fold rotates walls, the orientation
  // mix differs between the runs this tool exists to compare -- exactly when
  // the two definitions diverge.
  cv::Mat idx_map(rows, cols, CV_32S, cv::Scalar(-1));
  for (std::size_t i = 0; i < px.size(); ++i)
    idx_map.at<int>(px[i].r, px[i].c) = static_cast<int>(i);

  std::vector<double> cell_len(px.size(), 0.0);
  double len = 0.0;
  for (std::size_t i = 0; i < px.size(); ++i) {
    const auto& s = px[i];
    auto sk = [&](int r, int c) {
      return r >= 0 && r < rows && c >= 0 && c < cols &&
             skel.at<std::uint8_t>(r, c) != 0;
    };
    auto edge = [&](int r, int c, double L) {
      len += L;
      cell_len[i] += 0.5 * L;
      const int j = idx_map.at<int>(r, c);
      if (j >= 0) cell_len[static_cast<std::size_t>(j)] += 0.5 * L;
    };
    if (sk(s.r, s.c + 1)) edge(s.r, s.c + 1, g);
    if (sk(s.r + 1, s.c)) edge(s.r + 1, s.c, g);
    if (sk(s.r + 1, s.c + 1) && !sk(s.r, s.c + 1) && !sk(s.r + 1, s.c))
      edge(s.r + 1, s.c + 1, g * M_SQRT2);
    if (sk(s.r + 1, s.c - 1) && !sk(s.r, s.c - 1) && !sk(s.r + 1, s.c))
      edge(s.r + 1, s.c - 1, g * M_SQRT2);
  }
  m.wall_len = len;
  m.skel_cells = static_cast<long>(px.size());
  // Thinning can COLLAPSE a wall instead of thinning it: two parallel 45-deg
  // walls 1 m apart (the exact doubled-wall geometry this tool exists to
  // measure) reduced 566 occupied cells to 2 skeleton cells and reported
  // wall_len 0.0 m for 20 m of wall. A single 45-deg or axis-aligned wall
  // measures correctly, so the trigger is a rasterisation/thinning
  // interaction, not orientation alone. Surface the ratio rather than let
  // every metric silently read ~0.
  if (m.occupied_cells > 100 && px.size() * 50 < static_cast<std::size_t>(m.occupied_cells))
    std::fprintf(stderr,
                 "[map_metrics] WARNING: skeleton collapsed (%zu cells from %ld "
                 "occupied). Thinning erased the walls — wall_len, thickness "
                 "and doubled are all unreliable for this cloud. Try a finer "
                 "--grid.\n",
                 px.size(), m.occupied_cells);

  // doubling probe: march the wall normal in [band_min, band_max]; doubled if
  // a near-parallel skeleton cell sits in the band on EITHER side
  cv::Mat theta_map(rows, cols, CV_32F, cv::Scalar(1e9f));
  for (const auto& s : px) theta_map.at<float>(s.r, s.c) = s.theta;
  const float ang_tol = static_cast<float>(20.0 * M_PI / 180.0);
  double doubled_len = 0.0;
  // the probe dilates its target by one cell, so the inner band edge must
  // clear the source's own 3x3 neighborhood or a coarsened grid (or small
  // --band min) makes every wall "doubled" by matching itself
  const int t0_req = static_cast<int>(std::lround(band_min / g));
  const int t0 = std::max(3, t0_req);
  if (t0 > t0_req)
    std::fprintf(stderr,
                 "[map_metrics] grid %g m coarser than the probe band: inner "
                 "edge raised %g -> %g m; doubling closer than that is not "
                 "counted\n",
                 g, band_min, t0 * g);
  const int t1 = static_cast<int>(std::lround(band_max / g));
  if (t1 < t0) {
    std::fprintf(stderr,
                 "[map_metrics] probe band [%g, %g] m collapses at grid %g m "
                 "— doubled fraction not meaningful\n",
                 band_min, band_max, g);
    return m;
  }
  for (std::size_t i = 0; i < px.size(); ++i) {
    const auto& s = px[i];
    const float nx = -std::sin(s.theta), ny = std::cos(s.theta);
    bool hit = false;
    for (int sign = -1; sign <= 1 && !hit; sign += 2) {
      for (int t = t0; t <= t1 && !hit; ++t) {
        const int c = s.c + static_cast<int>(std::lround(sign * t * nx));
        const int r = s.r + static_cast<int>(std::lround(sign * t * ny));
        if (r < 1 || r >= rows - 1 || c < 1 || c >= cols - 1) break;
        for (int dr = -1; dr <= 1 && !hit; ++dr)
          for (int dc = -1; dc <= 1 && !hit; ++dc) {
            // never let the source wall match itself through the dilation
            if (std::abs(r + dr - s.r) <= 1 && std::abs(c + dc - s.c) <= 1)
              continue;
            const float th2 = theta_map.at<float>(r + dr, c + dc);
            if (th2 > 1e8f) continue;
            // orientation distance on the pi-periodic line direction
            float dth = std::fabs(s.theta - th2);
            dth = std::min(dth, static_cast<float>(M_PI) - dth);
            if (dth < ang_tol) hit = true;
          }
      }
    }
    if (hit) doubled_len += cell_len[i];
  }
  // length fraction, matching wall_len's units (isolated specks carry no
  // edge length and so weigh on neither side)
  m.doubled_frac = len > 0.0 ? doubled_len / len : 0.0;
  return m;
}

}  // namespace

class MapMetricsNode : public rclcpp::Node
{
public:
  MapMetricsNode(const std::string& topic, double grid, double band_min,
                 double band_max, bool once)
  : Node("map_metrics"), grid_(grid), band_min_(band_min), band_max_(band_max),
    once_(once)
  {
    // latched publisher -> transient_local subscription so a bag that played
    // before we started (or the live latched sample) is still delivered
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic, rclcpp::QoS(1).reliable().transient_local(),
      [this](const sensor_msgs::msg::PointCloud2& msg) { on_cloud(msg); });
    std::fprintf(stderr,
                 "[map_metrics] waiting for clouds on %s (grid %.2f m, band "
                 "%.1f-%.1f m)...\n",
                 topic.c_str(), grid_, band_min_, band_max_);
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2& msg)
  {
    const Matrix xyz = cloud_to_xyz(msg);
    const Metrics m = analyze(xyz, grid_, band_min_, band_max_);
    ++updates_;
    std::printf(
      "[map_metrics] update=%d pts=%ld cells=%ld wall_len=%.1fm "
      "thick_med=%.2fm thick_p95=%.2fm doubled=%.1f%% skel=%ld\n",
      updates_, m.points, m.occupied_cells, m.wall_len, m.thick_med,
      m.thick_p95, 100.0 * m.doubled_frac, m.skel_cells);
    std::fflush(stdout);
    if (once_) rclcpp::shutdown();
  }

  double grid_, band_min_, band_max_;
  bool once_;
  int updates_ = 0;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  std::string topic = "/bruce/slam/slam/cloud";
  double grid = 0.05, band_min = 0.3, band_max = 3.0;
  bool once = false;
  try {
    const std::vector<std::string> args =
      rclcpp::remove_ros_arguments(argc, argv);
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& a = args[i];
      auto next = [&](double& out) {
        if (i + 1 >= args.size())
          throw std::runtime_error(a + " needs a value");
        out = std::stod(args[++i]);
      };
      if (a == "--topic" && i + 1 < args.size()) {
        topic = args[++i];
      } else if (a == "--grid") {
        next(grid);
      } else if (a == "--band") {
        next(band_min);
        next(band_max);
      } else if (a == "--once") {
        once = true;
      } else {
        throw std::runtime_error("unknown argument " + a);
      }
    }
    if (grid <= 0.0 || band_min <= 0.0 || band_max <= band_min)
      throw std::runtime_error("need grid > 0 and band max > band min > 0");
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "%s\nusage: map_metrics [--topic T] [--grid m] "
                 "[--band min max] [--once]\n",
                 e.what());
    return 2;
  }
  rclcpp::spin(std::make_shared<sonar_slam::MapMetricsNode>(
    topic, grid, band_min, band_max, once));
  rclcpp::shutdown();
  return 0;
}
