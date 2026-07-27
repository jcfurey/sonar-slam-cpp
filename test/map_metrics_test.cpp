// map_metrics accuracy against synthetic walls of KNOWN length and known
// doubling, at the tool's documented operating point (--grid 0.05, band
// 0.3-3.0 m). Exit code is the verdict.
//
// These fixtures are the docs/MAP_DOUBLING_FIX_PLAN.md §5 cases. Two of them
// are regression guards for defects the tool shipped with:
//
//   [3] a 30 deg wall read +30% long, because Zhang-Suen could not thin the
//       2-cell-wide band an oblique wall rasterises into and every staircase
//       step was then counted at full cell length;
//   [4] two parallel 45 deg walls 1 m apart — the exact doubled-wall geometry
//       this tool exists to measure — thinned to TWO skeleton cells and
//       reported wall_len 0.00 m and doubled 0%. A perfect map-quality score
//       for a perfectly doubled map, which is the worst possible failure for
//       a metric used to accept or reject a tuning change.
//
// The bounds below are deliberately tight enough to fail if either returns.
#include "sonar_slam_cpp/map_metrics_math.hpp"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using sonar_slam::Matrix;
using sonar_slam::Metrics;

namespace {

int failures = 0;

// dense point sampling along a straight wall (1 cm spacing, finer than the
// 5 cm raster, like a real accumulated cloud)
void wall(std::vector<std::pair<double, double>>& pts, double x0, double y0,
          double len, double deg, double step = 0.01)
{
  const double th = deg * M_PI / 180.0;
  for (int i = 0; i <= static_cast<int>(len / step); ++i)
    pts.emplace_back(x0 + i * step * std::cos(th), y0 + i * step * std::sin(th));
}

Matrix cloud(const std::vector<std::pair<double, double>>& pts)
{
  Matrix m(static_cast<long>(pts.size()), 3);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    m(static_cast<long>(i), 0) = static_cast<float>(pts[i].first);
    m(static_cast<long>(i), 1) = static_cast<float>(pts[i].second);
    m(static_cast<long>(i), 2) = 0.f;
  }
  return m;
}

Metrics run(const std::vector<std::pair<double, double>>& pts)
{
  return sonar_slam::analyze(cloud(pts), 0.05, 0.3, 3.0);
}

// wall_len within `tol_pct` of the truth, and the skeleton not collapsed
void check_wall(const char* name, const Metrics& m, double truth,
                double tol_pct, bool expect_doubled)
{
  const double err = 100.0 * (m.wall_len - truth) / truth;
  bool ok = std::fabs(err) <= tol_pct;
  // a collapsed skeleton makes every metric read ~0, including `doubled`
  if (m.skel_cells * 50 < m.occupied_cells) ok = false;
  const bool doubled = m.doubled_frac > 0.9;
  if (doubled != expect_doubled) ok = false;
  std::printf("  %-38s len %6.2f / %5.2f (%+5.1f%%, tol %.1f%%) skel %5ld/%-5ld "
              "doubled %3.0f%% (want %s)  %s\n",
              name, m.wall_len, truth, err, tol_pct, m.skel_cells,
              m.occupied_cells, 100.0 * m.doubled_frac,
              expect_doubled ? "yes" : "no", ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

}  // namespace

int main()
{
  std::printf("[1] straight walls measure their true length at any orientation\n");
  {
    std::vector<std::pair<double, double>> p;
    wall(p, 0, 0, 10, 0);
    check_wall("horizontal 10 m", run(p), 10.0, 1.0, false);
  }
  {
    std::vector<std::pair<double, double>> p;
    wall(p, 0, 0, 10, 90);
    check_wall("vertical 10 m", run(p), 10.0, 1.0, false);
  }
  {
    std::vector<std::pair<double, double>> p;
    wall(p, 0, 0, 10, 45);
    check_wall("45 deg 10 m", run(p), 10.0, 1.0, false);
  }
  {  // REGRESSION: read 13.15 m (+32%) before the thinning fix
    std::vector<std::pair<double, double>> p;
    wall(p, 0, 0, 10, 30);
    check_wall("30 deg 10 m (was +32%)", run(p), 10.0, 2.5, false);
  }

  std::printf("[2] worst-case orientation error over a 0-90 deg sweep\n");
  {
    double worst = 0.0, worst_deg = 0.0;
    int collapsed = 0, false_doubling = 0;
    for (double deg = 0.0; deg <= 90.0001; deg += 1.0) {
      std::vector<std::pair<double, double>> p;
      wall(p, 0, 0, 10, deg);
      const Metrics m = run(p);
      const double e = std::fabs(100.0 * (m.wall_len - 10.0) / 10.0);
      if (e > worst) { worst = e; worst_deg = deg; }
      if (m.skel_cells * 50 < m.occupied_cells) ++collapsed;
      if (m.doubled_frac > 0.05) ++false_doubling;  // a single wall is not doubled
    }
    // the raw step sum peaks at +7.9%; the corner correction holds it under 2%
    const bool ok = worst <= 2.5 && collapsed == 0 && false_doubling == 0;
    std::printf("  worst %+.2f%% at %.0f deg, %d collapsed, %d false doubling  %s\n",
                worst, worst_deg, collapsed, false_doubling, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }

  std::printf("[3] a DOUBLED map is reported as doubled, at any orientation\n");
  {
    int missed = 0;
    double worst = 0.0;
    for (double deg = 0.0; deg <= 90.0001; deg += 5.0) {
      const double th = deg * M_PI / 180.0;
      std::vector<std::pair<double, double>> p;
      wall(p, 0, 0, 10, deg);
      wall(p, -std::sin(th), std::cos(th), 10, deg);  // 1 m along the normal
      const Metrics m = run(p);
      if (m.doubled_frac < 0.9) ++missed;
      worst = std::max(worst, std::fabs(100.0 * (m.wall_len - 20.0) / 20.0));
    }
    // REGRESSION: at 45 deg this used to collapse to 2 skeleton cells and
    // report 0.00 m of wall and 0% doubled
    const bool ok = missed == 0 && worst <= 2.5;
    std::printf("  %d of 19 orientations missed the doubling, worst length "
                "error %+.2f%%  %s\n", missed, worst, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }

  std::printf("[4] corners and closed loops stay intact\n");
  {
    std::vector<std::pair<double, double>> p;
    wall(p, 0, 0, 10, 0);
    wall(p, 10, 0, 8, 90);
    check_wall("L corner 10 + 8 m", run(p), 18.0, 1.5, false);
  }
  {
    std::vector<std::pair<double, double>> p;
    wall(p, 0, 0, 20, 0);
    wall(p, 20, 0, 12, 90);
    wall(p, 20, 12, 20, 180);
    wall(p, 0, 12, 12, 270);
    check_wall("closed 20x12 m pool", run(p), 64.0, 1.5, false);
  }

  std::printf("[5] wall thickness tracks the true wall width\n");
  {
    // a 0.4 m thick horizontal slab: thickness should read ~0.4 m
    std::vector<std::pair<double, double>> p;
    for (double y = 0.0; y <= 0.4001; y += 0.01) wall(p, 0, y, 10, 0);
    const Metrics m = run(p);
    const bool ok = m.thick_med > 0.3 && m.thick_med < 0.5;
    std::printf("  0.40 m thick slab -> thick_med %.2f m  %s\n", m.thick_med,
                ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }
  {  // an empty cloud must not crash or report anything
    const Metrics m = sonar_slam::analyze(Matrix(0, 3), 0.05, 0.3, 3.0);
    const bool ok = m.points == 0 && m.wall_len == 0.0 && m.skel_cells == 0;
    std::printf("  empty cloud -> zeros, no crash  %s\n", ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }

  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
