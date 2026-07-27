// Keyframe-anchored, loop-closure-correctable sonar mapping: port of
// bruce_slam mapping.py (Mapping + Submap). Each keyframe deposits a "tile" —
// its sonar fan projected to the map plane — carrying a logodds occupancy
// contribution (from the feature cloud) and a raw-intensity/backscatter
// contribution (from the ping image). The global grids are the SUM of the
// tiles, so when a loop closure moves a keyframe's pose the map is corrected
// incrementally: dec_grid removes the tile at its old pose, fit_grid re-places
// it at the optimized pose, inc_grid re-adds it (update_pose). This is why the
// occupancy update rule is additive logodds rather than clamped — a clamped
// policy could not be undone (mapping.py:76-79).
//
// The core owns no ROS message types; get_occupancy_grid / get_intensity_grid
// hand back a plain OccGrid the node wraps into nav_msgs/OccupancyGrid.
#pragma once

#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "sonar_slam_cpp/cloud_ops.hpp"       // Matrix
#include "sonar_slam_cpp/interp.hpp"          // Interp1d (bearing -> column)
#include "sonar_slam_cpp/sonar_geometry.hpp"  // OculusProperty, SonarPing

namespace sonar_slam {

// Plain occupancy-grid payload (no ROS dependency in the core library).
// Row-major, data[row * width + col]; row 0 / col 0 is the (origin_x, origin_y)
// corner and indices increase with +x (col) and +y (row) — nav_msgs layout.
struct OccGrid
{
  double origin_x = 0.0;
  double origin_y = 0.0;
  double resolution = 0.0;
  int width = 0;   // columns (x)
  int height = 0;  // rows (y)
  std::vector<int8_t> data;  // -1 unknown, else 0..100
  bool empty() const { return width <= 0 || height <= 0 || data.empty(); }
};

// One keyframe's contribution to the map. Port of mapping.py Submap.
struct Submap
{
  int k = 0;
  gtsam::Pose2 pose;
  bool valid = false;  // false marks a skipped/missing keyframe slot

  // The sonar fan pixel coordinates in the SENSOR frame (Nx2, x-forward /
  // y-left), shared across keyframes that share the sonar geometry (rebuilt
  // only when the range/aperture changes — mapping.py reuses the previous).
  std::shared_ptr<const Matrix> sonar_xy;
  // Per-pixel deposits (aligned with sonar_xy rows). Intensity is a raw
  // 0..255 pixel per cell — uint8 (the uint32 SUMS live only in the global
  // grids); at ~1 tile/keyframe this is a 4x memory saving on the dominant
  // per-keyframe allocation of a long mission.
  std::vector<float> logodds;      // occupancy logodds
  std::vector<uint8_t> intensity;  // raw backscatter 0..255

  // Current placement into the global grid (recomputed by fit_grid). r/c are
  // cell indices; l/i are the deduplicated per-cell deposits actually summed
  // into the grids by inc_grid (and subtracted by dec_grid).
  std::vector<int> r, c;
  std::vector<float> l;
  std::vector<uint8_t> i;
};

class Mapping
{
public:
  // ------------------------------------------------------------- parameters
  // (mapping.py __init__ defaults; overridden from the node config)
  double x0 = -50.0;      // world x of grid column 0
  double y0 = -50.0;      // world y of grid row 0
  double width = 100.0;   // initial extent (m)
  double height = 100.0;
  double inc = 50.0;      // grid growth increment (m)
  double resolution = 0.2;

  bool pub_occupancy1 = true;  // additive-logodds occupancy (correctable)
  bool pub_intensity = true;   // intensity/backscatter mosaic (correctable)

  double hit_prob = 0.8;
  double miss_prob = 0.3;
  double inflation_angle = 0.05;  // rad, occupied-cell bearing inflation
  double inflation_range = 0.5;   // m,   occupied-cell range inflation

  double outlier_filter_radius = 5.0;
  int outlier_filter_min_points = 20;

  // Keyframes with fewer feature points than this deposit a NEUTRAL occupancy
  // tile (logodds 0) instead of the upstream all-free wedge — a CFAR whiff on
  // low-contrast structure must not stamp miss-logodds to max range over real
  // walls. 0 keeps the upstream policy (any point count claims free space).
  int free_tile_min_points = 0;

  // only re-place a keyframe when it moved at least this much (skip micro
  // jitter — the O(cells) dec/inc is wasted below the grid resolution)
  double min_translation = 0.5;
  double min_rotation = 0.05;

  // Sign applied to the sensor-frame y before placing a tile, so the fan
  // matches the trajectory frame's chirality: +1 when slam_node runs
  // enu_world (standard ENU trajectory), -1 for its z-down/roll-pi graph
  // frame. Must match the trajectory the poses come from.
  double frame_y_sign = 1.0;

  // ------------------------------------------------------------- pipeline
  void configure();

  // Add keyframe `key` (its optimized pose, the source ping for geometry +
  // backscatter, and its feature cloud in the sensor frame, Nx>=2). Missing
  // intermediate slots are padded so indices stay aligned with the graph.
  void add_keyframe(int key, const gtsam::Pose2& pose, const SonarPing& ping,
                    const Matrix& points);

  // Advance past keyframe `key` with an empty, invalid tile — for keyframes
  // whose ping/feature data is conclusively unrecoverable (dropped message,
  // or a restart against a latched trajectory carrying unseen history). The
  // in-order builder must not wedge behind them.
  void add_skipped(int key, const gtsam::Pose2& pose);

  // Re-place keyframe `key` at `new_pose` if it moved enough — the loop
  // closure correction (dec old tile -> refit -> inc new tile).
  void update_pose(int key, const gtsam::Pose2& new_pose);

  // Publishable products (bounded to the edited box). Occupancy is expit of
  // the summed logodds; intensity is the per-cell mean backscatter, 0..100.
  OccGrid get_occupancy_grid() const;
  OccGrid get_intensity_grid() const;

  int num_keyframes() const { return static_cast<int>(keyframes_.size()); }

private:
  bool pose_changed(const gtsam::Pose2& a, const gtsam::Pose2& b) const;
  void fit_grid(Submap& kf);
  void inc_grid(const Submap& kf);
  void dec_grid(const Submap& kf);
  // Grow the grids (and shift every cached tile) until r/c are in range;
  // mutates r/c to the (possibly shifted) valid indices.
  void adjust_bounds(std::vector<int>& r, std::vector<int>& c);

  using GridF = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using GridU = Eigen::Array<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // grid grow primitives (prepend/append zeroed rows/cols)
  static void grow_rows(GridF& g, int top, int bottom);
  static void grow_rows(GridU& g, int top, int bottom);
  static void grow_cols(GridF& g, int left, int right);
  static void grow_cols(GridU& g, int left, int right);

  // fit_grid scratch, reused across keyframes (a loop-closure correction
  // re-fits every keyframe that moved, so these would otherwise be one
  // allocation each per keyframe per round). seen_ is a dense "already
  // deposited" byte map over the current tile's bounding box, which the sonar
  // range bounds — it does not scale with the map.
  std::vector<int> fit_r_, fit_c_;
  std::vector<unsigned char> seen_;

  GridF logodds_grid_;
  GridU intensity_grid_;
  GridU counter_grid_;
  int rows_ = 0, cols_ = 0;
  int rmin_ = 0, rmax_ = 0, cmin_ = 0, cmax_ = 0;
  int inc_r_ = 0, inc_c_ = 0;

  // sonar geometry + the fan cache shared by same-geometry keyframes
  OculusProperty oculus_;
  std::shared_ptr<const Matrix> sonar_xy_;
  Interp1d b2c_;       // bearing (rad) -> beam column, linear (sonar.py)
  int r_skip_ = 1, c_skip_ = 1;
  int sub_rows_ = 0, sub_cols_ = 0;  // downsampled fan image dims
  double fan_range_min_ = 0.0;       // range of the first polar row's origin
  // image beam column whose echo belongs at fan column ci: the feature (and
  // therefore trajectory) frame's lateral is the NEGATED native bearing, so
  // the mosaic reads the mirrored beam (bearings[mirror_col_[ci]] == -bearings[ci])
  std::vector<int> mirror_col_;

  std::vector<Submap> keyframes_;
};

}  // namespace sonar_slam
