// Mapping core: the invariants the keyframe-anchored map rests on.
//
// mapping.cpp had no test at all, which is uncomfortable for the module that
// produces the deliverable. The properties below are the ones the design
// documents as load-bearing:
//
//  [1] a tile lands where the sonar fan actually is, in world coordinates;
//  [2] the loop-closure correction (dec -> refit -> inc) is REVERSIBLE — the
//      stated reason the occupancy rule is additive logodds rather than
//      clamped (mapping.hpp: "a clamped policy could not be undone");
//  [3] grid growth preserves already-deposited evidence and keeps the world
//      origin consistent, for excursions in every direction;
//  [4] free_tile_min_points deposits a NEUTRAL tile rather than an all-free
//      wedge when the cloud is too sparse to trust;
//  [5] the intensity counter never underflows (it is unsigned, so a dec
//      without a matching inc would read as ~4e9 rather than as a bug).
//
// Verified by mutation: dropping the dec_grid subtraction, dropping either
// prepend index shift, disabling the per-cell dedup, and freeing a whole
// bearing column instead of the cells before the first hit are all caught.
//
// NOT covered: a SMALL under-report of the edited bounding box (a few cells
// cropped off the published window's edge). Catching it needs a probe placed
// within a cell or two of the fan's outer edge, which would make the test
// brittle against the fan's range subsampling for a minor failure mode.
// Gross cropping is caught by the far-edge probes in [3].
//
// Exit code is the verdict.
#include "sonar_slam_cpp/mapping.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using sonar_slam::Mapping;
using sonar_slam::Matrix;
using sonar_slam::OccGrid;
using sonar_slam::SonarPing;

namespace {

int failures = 0;

void check(bool ok, const char* what, const char* detail = "")
{
  std::printf("  %-56s %s%s%s\n", what, ok ? "ok" : "FAIL",
              *detail ? " — " : "", detail);
  if (!ok) ++failures;
}

SonarPing make_ping(int num_ranges = 240, int num_beams = 96,
                    double res = 0.1, double range_min = 0.0,
                    double aperture = 2.2689)
{
  SonarPing p;
  p.num_ranges = num_ranges;
  p.range_resolution = res;
  p.range_min = range_min;
  p.bearings.resize(static_cast<std::size_t>(num_beams));
  for (int i = 0; i < num_beams; ++i)
    p.bearings[static_cast<std::size_t>(i)] = static_cast<float>(
      -aperture / 2 + aperture * i / (num_beams - 1.0));
  p.image.create(num_ranges, num_beams, CV_8UC1);
  p.image.setTo(120);
  return p;
}

// a compact cloud of returns straight ahead at `range` metres
Matrix wall_cloud(double range, int n = 60)
{
  Matrix m(n, 3);
  for (int i = 0; i < n; ++i) {
    m(i, 0) = static_cast<float>(range);
    m(i, 1) = static_cast<float>(-1.5 + 3.0 * i / (n - 1.0));
    m(i, 2) = 0.f;
  }
  return m;
}

void configure(Mapping& m, double x0 = -25, double y0 = -25, double w = 50,
               double h = 50, double inc = 20, double res = 0.2,
               int free_min = 0)
{
  m.x0 = x0; m.y0 = y0; m.width = w; m.height = h; m.inc = inc;
  m.resolution = res;
  m.pub_occupancy1 = true;
  m.pub_intensity = true;
  m.hit_prob = 0.8; m.miss_prob = 0.3;
  m.inflation_angle = 0.05; m.inflation_range = 0.5;
  m.outlier_filter_radius = 5.0;
  m.outlier_filter_min_points = 0;  // keep the synthetic cloud intact
  m.free_tile_min_points = free_min;
  m.min_translation = 0.5; m.min_rotation = 0.05;
  m.frame_y_sign = 1.0;
  m.configure();
}

// world (x, y) -> cell value in an OccGrid, or -2 when outside
int grid_at(const OccGrid& g, double x, double y)
{
  const int c = static_cast<int>(std::lround((x - g.origin_x) / g.resolution));
  const int r = static_cast<int>(std::lround((y - g.origin_y) / g.resolution));
  if (r < 0 || r >= g.height || c < 0 || c >= g.width) return -2;
  return g.data[static_cast<std::size_t>(r) * g.width + c];
}

bool same_grid(const OccGrid& a, const OccGrid& b)
{
  return a.width == b.width && a.height == b.height &&
         a.origin_x == b.origin_x && a.origin_y == b.origin_y &&
         a.resolution == b.resolution && a.data == b.data;
}

// Cells that differ between two grids at the same WORLD coordinates, over
// their overlap. Grids are NOT compared whole: the edited bounding box is
// documented never to shrink (mapping.cpp dec_grid), so a correction that
// swings keyframes out and back legitimately leaves a LARGER published
// window with the extra margin unknown. What must not change is the evidence
// at a given point on the seabed.
long cells_differing(const OccGrid& a, const OccGrid& b, long& overlap,
                     int& max_delta)
{
  long diff = 0;
  overlap = 0;
  max_delta = 0;
  for (int r = 0; r < a.height; ++r)
    for (int c = 0; c < a.width; ++c) {
      const double x = a.origin_x + c * a.resolution;
      const double y = a.origin_y + r * a.resolution;
      const int cb = static_cast<int>(std::lround((x - b.origin_x) / b.resolution));
      const int rb = static_cast<int>(std::lround((y - b.origin_y) / b.resolution));
      if (rb < 0 || rb >= b.height || cb < 0 || cb >= b.width) continue;
      ++overlap;
      const int va = a.data[static_cast<std::size_t>(r) * a.width + c];
      const int vb = b.data[static_cast<std::size_t>(rb) * b.width + cb];
      if (va != vb) {
        ++diff;
        max_delta = std::max(max_delta, std::abs(va - vb));
      }
    }
  return diff;
}

}  // namespace

int main()
{
  const SonarPing ping = make_ping();

  // ---- [1] a tile lands at the fan's true world position ------------------
  std::printf("[1] tile placement\n");
  {
    Mapping m;
    configure(m);
    // vehicle at (5, 5) facing +x, wall returns 8 m ahead
    m.add_keyframe(0, gtsam::Pose2(5.0, 5.0, 0.0), ping, wall_cloud(8.0));
    const OccGrid occ = m.get_occupancy_grid();
    check(!occ.empty(), "occupancy grid is non-empty");
    // occupied evidence should sit near (13, 5); free space between us and it
    const int at_wall = grid_at(occ, 13.0, 5.0);
    const int at_free = grid_at(occ, 9.0, 5.0);
    char buf[128];
    std::snprintf(buf, sizeof buf, "wall %d, free %d (want wall>50>free)",
                  at_wall, at_free);
    check(at_wall > 50 && at_free >= 0 && at_free < 50,
          "occupied at the wall, free in between", buf);
    // behind the vehicle is outside the fan -> untouched
    check(grid_at(occ, 1.0, 5.0) == 50 || grid_at(occ, 1.0, 5.0) == -2,
          "no evidence behind the vehicle");
  }

  // ---- [2] the loop-closure correction is reversible -----------------------
  std::printf("[2] dec -> refit -> inc reversibility (the additive-logodds "
              "premise)\n");
  {
    Mapping m;
    configure(m);
    for (int k = 0; k < 6; ++k)
      m.add_keyframe(k, gtsam::Pose2(k * 1.0, 0.0, 0.0), ping, wall_cloud(7.0));
    const OccGrid before_occ = m.get_occupancy_grid();
    const OccGrid before_int = m.get_intensity_grid();

    // shove every keyframe well past min_translation, then put it back
    for (int k = 0; k < 6; ++k)
      m.update_pose(k, gtsam::Pose2(k * 1.0 + 3.0, 2.0, 0.3));
    const OccGrid moved = m.get_occupancy_grid();
    check(!same_grid(before_occ, moved), "moving the keyframes changed the map");

    for (int k = 0; k < 6; ++k)
      m.update_pose(k, gtsam::Pose2(k * 1.0, 0.0, 0.0));
    const OccGrid after_occ = m.get_occupancy_grid();
    const OccGrid after_int = m.get_intensity_grid();

    // A round trip must leave the evidence at every world point EXACTLY where
    // it started — not approximately. The grid accumulates in float, and
    // x + a - a is not identically x in IEEE 754, so this is a real property
    // and not a tautology; it is also the whole justification for the
    // additive-logodds rule over a clamped one.
    char buf[128];
    long overlap = 0;
    int max_delta = 0;
    long diff = cells_differing(before_occ, after_occ, overlap, max_delta);
    std::snprintf(buf, sizeof buf, "%ld/%ld cells differ, max delta %d", diff,
                  overlap, max_delta);
    check(diff == 0 && overlap > 1000, "occupancy round-trips exactly", buf);

    diff = cells_differing(before_int, after_int, overlap, max_delta);
    std::snprintf(buf, sizeof buf, "%ld/%ld cells differ, max delta %d", diff,
                  overlap, max_delta);
    check(diff == 0 && overlap > 1000, "intensity round-trips exactly", buf);
  }

  // ---- [3] growth in every direction preserves evidence -------------------
  std::printf("[3] grid growth\n");
  {
    // start with a box the trajectory leaves on all four sides, and an `inc`
    // small enough that several growth steps are needed per excursion
    Mapping m;
    configure(m, -6, -6, 12, 12, 3.0, 0.2);
    m.add_keyframe(0, gtsam::Pose2(0.0, 0.0, 0.0), ping, wall_cloud(5.0));
    const int anchor_before = grid_at(m.get_occupancy_grid(), 5.0, 0.0);

    const double far = 40.0;
    const gtsam::Pose2 excursions[4] = {
      gtsam::Pose2(far, 0.0, 0.0), gtsam::Pose2(-far, 0.0, M_PI),
      gtsam::Pose2(0.0, far, M_PI / 2), gtsam::Pose2(0.0, -far, -M_PI / 2)};
    for (int k = 0; k < 4; ++k)
      m.add_keyframe(k + 1, excursions[k], ping, wall_cloud(5.0));

    const OccGrid occ = m.get_occupancy_grid();
    // The published window is the edited bounding box, so it must reach every
    // cell a fan actually wrote — a box that under-reports silently CROPS the
    // map at its edges. Each fan spans range_min + num_ranges*range_resolution
    // = 24 m, so the far side of each excursion's wedge must be inside.
    check(grid_at(occ, far + 20.0, 0.0) != -2, "+x fan far edge is published");
    check(grid_at(occ, -far - 20.0, 0.0) != -2, "-x fan far edge is published");
    check(grid_at(occ, 0.0, far + 20.0) != -2, "+y fan far edge is published");
    check(grid_at(occ, 0.0, -far - 20.0) != -2, "-y fan far edge is published");
    const int anchor_after = grid_at(occ, 5.0, 0.0);
    char buf[96];
    std::snprintf(buf, sizeof buf, "%d before, %d after growth", anchor_before,
                  anchor_after);
    check(anchor_before == anchor_after,
          "evidence at a fixed world point survives growth", buf);
    // and each excursion's own wall is present at its true world position
    check(grid_at(occ, far + 5.0, 0.0) > 50, "+x excursion wall present");
    check(grid_at(occ, -far - 5.0, 0.0) > 50, "-x excursion wall present");
    check(grid_at(occ, 0.0, far + 5.0) > 50, "+y excursion wall present");
    check(grid_at(occ, 0.0, -far - 5.0) > 50, "-y excursion wall present");

    // Growth SHIFTS every cached tile's stored cell indices. A shift that is
    // dropped or mis-sized leaves the grid contents correct until the next
    // correction, and only then subtracts a tile from the wrong cells — so
    // the excursions above cannot see it. Correct a keyframe that predates
    // the growth, and put it back: the map must be unchanged.
    for (int k = 0; k < 5; ++k)
      m.update_pose(k, gtsam::Pose2(excursions[k % 4].x() * 0.5 + 1.0,
                                    excursions[k % 4].y() * 0.5 - 1.0, 0.2));
    m.update_pose(0, gtsam::Pose2(0.0, 0.0, 0.0));
    for (int k = 1; k < 5; ++k) m.update_pose(k, excursions[k - 1]);

    const OccGrid healed = m.get_occupancy_grid();
    long overlap = 0;
    int max_delta = 0;
    const long diff = cells_differing(occ, healed, overlap, max_delta);
    std::snprintf(buf, sizeof buf, "%ld/%ld cells differ, max delta %d", diff,
                  overlap, max_delta);
    check(diff == 0 && overlap > 1000,
          "post-growth corrections round-trip exactly", buf);
  }

  // ---- [4] a sparse cloud deposits a NEUTRAL tile, not an all-free wedge ---
  std::printf("[4] free_tile_min_points policy\n");
  {
    Mapping sparse_ok, sparse_gated;
    configure(sparse_ok, -25, -25, 50, 50, 20, 0.2, /*free_min=*/0);
    configure(sparse_gated, -25, -25, 50, 50, 20, 0.2, /*free_min=*/20);

    // a real wall first, then a near-empty frame looking the same way
    for (Mapping* m : {&sparse_ok, &sparse_gated}) {
      m->add_keyframe(0, gtsam::Pose2(0, 0, 0), ping, wall_cloud(8.0));
      m->add_keyframe(1, gtsam::Pose2(0, 0, 0), ping, wall_cloud(8.0, 3));
    }
    const int ok_at_wall = grid_at(sparse_ok.get_occupancy_grid(), 8.0, 0.0);
    const int gated_at_wall = grid_at(sparse_gated.get_occupancy_grid(), 8.0, 0.0);
    char buf[128];
    std::snprintf(buf, sizeof buf, "ungated %d vs gated %d", ok_at_wall,
                  gated_at_wall);
    // the gated map must not have had free-space stamped over the wall
    check(gated_at_wall >= ok_at_wall,
          "gating a sparse frame does not erase the wall", buf);
    check(gated_at_wall > 50, "wall still reads occupied under the gate");
  }

  // ---- [5] the unsigned intensity counter never underflows -----------------
  std::printf("[5] intensity counter integrity\n");
  {
    Mapping m;
    configure(m);
    for (int k = 0; k < 4; ++k)
      m.add_keyframe(k, gtsam::Pose2(k * 2.0, 0.0, 0.0), ping, wall_cloud(6.0));
    // many corrections, including ones that force growth
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-30.0, 30.0);
    for (int round = 0; round < 12; ++round)
      for (int k = 0; k < 4; ++k)
        m.update_pose(k, gtsam::Pose2(u(rng), u(rng), u(rng) * 0.05));
    const OccGrid inten = m.get_intensity_grid();
    bool in_range = true;
    for (std::int8_t v : inten.data)
      if (v < -1 || v > 100) in_range = false;
    // an underflowed uint32 counter divides into a mean of ~0 or explodes the
    // rounding; either way the published value leaves 0..100
    check(in_range, "every published intensity cell stays in [-1, 100]");
    check(!inten.empty(), "intensity grid still non-empty after corrections");
  }

  // ---- [6] skipped keyframes advance the index without depositing ----------
  std::printf("[6] skipped keyframes\n");
  {
    Mapping m;
    configure(m);
    m.add_keyframe(0, gtsam::Pose2(0, 0, 0), ping, wall_cloud(6.0));
    m.add_skipped(1, gtsam::Pose2(3, 0, 0));
    m.add_keyframe(2, gtsam::Pose2(6, 0, 0), ping, wall_cloud(6.0));
    check(m.num_keyframes() == 3, "index advances past the skipped slot");
    // snapshot AFTER every real keyframe is in, so the only thing that could
    // move the map below is the skipped slot itself
    const OccGrid before = m.get_occupancy_grid();
    m.update_pose(1, gtsam::Pose2(9, 9, 1.0));  // must be a no-op
    const OccGrid after = m.get_occupancy_grid();
    long overlap = 0;
    int max_delta = 0;
    const long diff = cells_differing(before, after, overlap, max_delta);
    char buf[96];
    std::snprintf(buf, sizeof buf, "%ld/%ld cells moved", diff, overlap);
    check(diff == 0 && overlap > 100,
          "correcting a skipped keyframe deposits nothing", buf);
  }

  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
