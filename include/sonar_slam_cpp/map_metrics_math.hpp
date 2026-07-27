// Map-quality math for the map_metrics tool, kept free of ROS so the numbers
// it reports can be unit-tested against synthetic walls of known length (see
// test/map_metrics_test.cpp). Same split, and same reason, as
// icp_covariance_math.hpp.
//
// Rasterise a cloud's x/y projection, thin the walls to a 1-px skeleton, and
// report total wall length, local wall thickness, and the doubled-wall
// fraction — the docs/MAP_DOUBLING_FIX_PLAN.md Â§5 replay numbers.
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "sonar_slam_cpp/cloud_ops.hpp"  // Matrix

namespace sonar_slam {


// Guo–Hall thinning of a 0/1 uint8 image to a 1-px skeleton.
//
// This replaced Zhang–Suen, which has a well-known failure on DIAGONAL LINES
// TWO PIXELS WIDE: no interior pixel of such a band satisfies A(P1) == 1, so
// the algorithm cannot thin it at all and instead nibbles one pixel off each
// end per sub-iteration until the whole wall is gone. That is not an exotic
// input — a straight wall rasterises two cells wide whenever the x and y
// quantisation phases differ, which is the generic case for any wall that is
// not axis-aligned or exactly on the grid diagonal. Measured on the
// docs/MAP_DOUBLING_FIX_PLAN §5 fixtures at --grid 0.05: two parallel 45°
// walls 1 m apart (566 occupied cells, 20 m of wall) thinned to 2 skeleton
// cells and reported wall_len 0.00 m — a PERFECT doubling score for a
// perfectly doubled map. The same defect left oblique walls unthinned and
// over-read their length by up to +30% (a 30° wall measured 13.15 m).
//
// Guo–Hall thins the same band to a proper 1-px centreline: the parallel-wall
// fixture now yields 285 skeleton cells and 19.97 m of 20.
inline void thin(cv::Mat& img)
{
  CV_Assert(img.type() == CV_8UC1);
  cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
  bool changed = true;
  auto at = [&](int r, int c) -> int { return img.at<uint8_t>(r, c); };
  // Erosion is monotone so this always terminates, but a thinning pass that
  // somehow oscillated would hang a field tool: bound it at far more sweeps
  // than the widest plausible wall needs (each sweep peels one pixel).
  int sweeps = 0;
  while (changed && ++sweeps <= 4096) {
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
          // C = number of distinct 8-connected components in the neighbourhood
          // (1 => removing this pixel cannot disconnect the skeleton)
          const int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
                        ((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
          const int n1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
          const int n2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
          const int n = std::min(n1, n2);  // 2..3 => not an end point
          // the two sub-iterations peel opposite sides so the centreline does
          // not drift
          const int m = (step == 0) ? ((p2 | p3 | (!p5)) & p4)
                                    : ((p6 | p7 | (!p9)) & p8);
          if (C == 1 && n >= 2 && n <= 3 && m == 0) {
            marker.at<uint8_t>(r, c) = 1;
            changed = true;
          }
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
inline cv::Mat distance_to_free(const cv::Mat& occ)
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

inline Metrics analyze(const Matrix& xyz, double grid, double band_min, double band_max)
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
  // ------------------------------------------------------------- wall length
  // Sum 8-connected step lengths (axis step = g, diagonal = g*sqrt(2)), then
  // apply a CORNER correction.
  //
  // Summing the steps alone is the classic digital-length over-estimate: the
  // staircase a straight line rasterises into is genuinely longer than the
  // line, by up to +7.9% (worst near 23 deg) even on a perfectly thinned
  // 8-connected skeleton. The correction subtracts a fixed cost per direction
  // change, which is what distinguishes a staircase from a straight run:
  //
  //     L = ( n_axis + sqrt(2) * n_diag - 0.108 * n_corner ) * g
  //
  // The first two coefficients are exact by construction, so an axis-aligned
  // or exactly-45 deg wall (n_corner == 0) still measures exactly. 0.108 is
  // the minimax fit over a 10 m wall swept 0-90 deg at 1 deg steps: it holds
  // the worst error to 1.82% at ANY orientation, against +7.9% for the raw
  // step sum. This matters precisely because the runs this tool compares
  // differ by a graph FOLD, which rotates walls — an orientation-dependent
  // length bias would show up as a map-quality change that never happened.
  // (Same family as the Dorst-Smeulders estimator, refit here to keep the
  // cardinal orientations exact rather than 2% short.)
  //
  // Each edge's length is also split evenly between its two endpoint cells,
  // giving a per-cell length weight that sums back to the total. The doubling
  // probe below needs it: reporting doubled/px.size() would be a CELL-COUNT
  // fraction while the metric is documented (and named) as a fraction of
  // skeleton LENGTH, and a diagonal cell carries sqrt(2) times the length of
  // an axis-aligned one. Since a graph fold rotates walls, the orientation
  // mix differs between the runs this tool exists to compare -- exactly when
  // the two definitions diverge.
  cv::Mat idx_map(rows, cols, CV_32S, cv::Scalar(-1));
  for (std::size_t i = 0; i < px.size(); ++i)
    idx_map.at<int>(px[i].r, px[i].c) = static_cast<int>(i);

  auto sk = [&](int r, int c) {
    return r >= 0 && r < rows && c >= 0 && c < cols &&
           skel.at<std::uint8_t>(r, c) != 0;
  };

  std::vector<double> cell_len(px.size(), 0.0);
  double step_len = 0.0;
  for (std::size_t i = 0; i < px.size(); ++i) {
    const auto& s = px[i];
    auto edge = [&](int r, int c, double L) {
      step_len += L;
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

  // A corner is a chain cell (exactly two 8-neighbours) whose two neighbours
  // are not diametrically opposite, i.e. the curve turns here. End points (one
  // neighbour) and junctions (three or more) are not counted: they are not
  // staircase artefacts and there are O(1) of them per wall.
  long corners = 0;
  {
    static const int dr[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    static const int dc[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    for (const auto& s : px) {
      int found = 0, d0 = -1, d1 = -1;
      for (int k = 0; k < 8 && found <= 2; ++k)
        if (sk(s.r + dr[k], s.c + dc[k])) {
          if (found == 0) d0 = k; else if (found == 1) d1 = k;
          ++found;
        }
      if (found != 2) continue;
      const int sep = std::abs(d0 - d1);
      if (std::min(sep, 8 - sep) != 4) ++corners;
    }
  }
  const double len = std::max(0.0, step_len - 0.108 * corners * g);
  // keep the per-cell weights summing to the reported length, so the doubled
  // fraction below stays a true fraction of it
  if (step_len > 0.0) {
    const double scale = len / step_len;
    for (double& w : cell_len) w *= scale;
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

}  // namespace sonar_slam
