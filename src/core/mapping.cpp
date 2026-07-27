// Port of bruce_slam mapping.py (Mapping + Submap). See mapping.hpp for the
// design; comments here point at the upstream lines being reproduced.
#include "sonar_slam_cpp/mapping.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace sonar_slam {

namespace {

inline float logit(float p) { return std::log(p / (1.0f - p)); }
inline float expit(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// grow a row-major Eigen array by `top` zeroed rows above and `bottom` below,
// preserving existing contents (numpy np.r_[zeros, g] / np.r_[g, zeros]).
template <class G>
void grow_rows_impl(G& g, int top, int bottom)
{
  if (top <= 0 && bottom <= 0) return;
  const int rows = static_cast<int>(g.rows()), cols = static_cast<int>(g.cols());
  G out = G::Zero(rows + top + bottom, cols);
  if (rows > 0) out.block(top, 0, rows, cols) = g;
  g.swap(out);
}

// grow by `left`/`right` zeroed columns (np.c_[zeros, g] / np.c_[g, zeros]).
template <class G>
void grow_cols_impl(G& g, int left, int right)
{
  if (left <= 0 && right <= 0) return;
  const int rows = static_cast<int>(g.rows()), cols = static_cast<int>(g.cols());
  G out = G::Zero(rows, cols + left + right);
  if (cols > 0) out.block(0, left, rows, cols) = g;
  g.swap(out);
}

}  // namespace

void Mapping::grow_rows(GridF& g, int top, int bottom) { grow_rows_impl(g, top, bottom); }
void Mapping::grow_rows(GridU& g, int top, int bottom) { grow_rows_impl(g, top, bottom); }
void Mapping::grow_cols(GridF& g, int left, int right) { grow_cols_impl(g, left, right); }
void Mapping::grow_cols(GridU& g, int left, int right) { grow_cols_impl(g, left, right); }

void Mapping::configure()
{
  // xs = arange(0, width, res); ys = arange(0, height, res)  (mapping.py:116)
  cols_ = static_cast<int>(std::ceil(width / resolution - 1e-9));
  rows_ = static_cast<int>(std::ceil(height / resolution - 1e-9));
  cols_ = std::max(cols_, 1);
  rows_ = std::max(rows_, 1);

  if (pub_occupancy1) logodds_grid_ = GridF::Zero(rows_, cols_);
  if (pub_intensity) {
    intensity_grid_ = GridU::Zero(rows_, cols_);
    counter_grid_ = GridU::Zero(rows_, cols_);
  }

  rmax_ = cmax_ = 0;
  rmin_ = rows_ - 1;
  cmin_ = cols_ - 1;
  inc_r_ = std::max(1, static_cast<int>(inc / resolution));
  inc_c_ = std::max(1, static_cast<int>(inc / resolution));
}

bool Mapping::pose_changed(const gtsam::Pose2& a, const gtsam::Pose2& b) const
{
  const gtsam::Pose2 dp = a.between(b);
  const double dt = dp.translation().norm();
  const double dr = std::abs(dp.theta());
  return dt > min_translation || dr > min_rotation;
}

void Mapping::add_keyframe(int key, const gtsam::Pose2& pose,
                           const SonarPing& ping, const Matrix& points)
{
  const bool changed = oculus_.configure(ping);

  Submap kf;
  // the slot this keyframe will occupy — keyframes_.size() disagrees with it
  // when the padding loop below fills missed slots first
  kf.k = key;
  kf.pose = pose;
  kf.valid = true;

  const int num_ranges = oculus_.num_ranges;
  const int num_bearings = oculus_.num_bearings;

  // (Re)build the sonar-fan pixel geometry only when the range/aperture
  // changed (mapping.py:152-168 — keyframes otherwise reuse the previous fan).
  if ((changed || !sonar_xy_ || sub_rows_ == 0 ||
       ping.range_min != fan_range_min_) &&
      oculus_.range_resolution > 0.0 && oculus_.angular_resolution > 0.0 &&
      num_ranges > 0 && num_bearings > 1) {
    r_skip_ = std::max(1, static_cast<int>(std::floor(resolution / oculus_.range_resolution)));
    const double bearing_arc_res = oculus_.angular_resolution * oculus_.max_range;
    c_skip_ = std::max(1, static_cast<int>(std::floor(resolution / bearing_arc_res)));

    sub_rows_ = (num_ranges + r_skip_ - 1) / r_skip_;
    sub_cols_ = (num_bearings + c_skip_ - 1) / c_skip_;
    // polar row ri sits at range_min + (1+ri)*res, matching the feature
    // node's row<->range convention. An Oculus has range_min == 0 and this
    // reduces to the original ranges[ri]; a payload with a real minimum range
    // no longer has every tile compressed toward the sensor by range_min.
    fan_range_min_ = ping.range_min;

    Matrix xy(sub_rows_ * sub_cols_, 2);
    int idx = 0;
    for (int ri = 0; ri < num_ranges; ri += r_skip_) {
      const double R = fan_range_min_ + oculus_.range_resolution * (1 + ri);  // ranges[ri]
      for (int ci = 0; ci < num_bearings; ci += c_skip_) {
        const double B = oculus_.bearings[ci];
        xy(idx, 0) = static_cast<float>(std::cos(B) * R);
        xy(idx, 1) = static_cast<float>(std::sin(B) * R);
        ++idx;
      }
    }
    sonar_xy_ = std::make_shared<const Matrix>(std::move(xy));

    // bearing (rad) -> beam column, linear like sonar.py's interp1d
    std::vector<double> bx(oculus_.bearings.begin(), oculus_.bearings.end());
    std::vector<double> by(num_bearings);
    for (int i = 0; i < num_bearings; ++i) by[i] = i;
    b2c_ = Interp1d(bx, by, Interp1d::LINEAR, 0.0);

    // Mirrored beam lookup for the intensity mosaic: the feature cloud's
    // lateral coordinate is the NEGATED native bearing (feature_extraction
    // publishes x = -sin(B)*R), and the trajectory/occupancy products live in
    // that chirality. The fan grid's placement uses native-signed bearings,
    // so backscatter for fan column ci must be read from the beam at
    // -bearings[ci] — otherwise the mosaic mirrors port<->starboard against
    // the occupancy grid and the (bag-validated) trajectory.
    mirror_col_.assign(num_bearings, 0);
    for (int i = 0; i < num_bearings; ++i) {
      // clamp the query into the aperture: for an asymmetric fan the negated
      // bearing can fall outside [front, back], where the interp's fill value
      // (0.0) would silently alias every such beam to column 0
      const double q = std::min(
        std::max(-static_cast<double>(oculus_.bearings[i]),
                 static_cast<double>(oculus_.bearings.front())),
        static_cast<double>(oculus_.bearings.back()));
      const int m = static_cast<int>(std::lround(b2c_(q)));
      mirror_col_[i] = std::min(std::max(m, 0), num_bearings - 1);
    }
  }
  kf.sonar_xy = sonar_xy_;
  // no valid fan geometry yet (e.g. a malformed first ping): store an empty
  // tile so the keyframe index still advances but nothing is deposited.
  const bool have_geom = sonar_xy_ && sub_rows_ > 0 && sub_cols_ > 0;

  // --------- occupancy logodds tile (mapping.py:170-228) ---------
  // Filter FIRST: the free-space policy must judge the FILTERED cloud. A
  // cloud the outlier filter empties carries no more evidence than an empty
  // one, and letting it through to the all-free wedge would stamp
  // miss-logodds over real structure — the exact failure free_tile_min_points
  // exists to prevent.
  Matrix xy2(0, 2);
  if (have_geom && pub_occupancy1 && points.rows() > 0) {
    xy2 = points.leftCols(2);
    if (outlier_filter_min_points > 1 && xy2.rows() > 0)
      xy2 = remove_outlier(xy2, outlier_filter_radius, outlier_filter_min_points);
  }
  if (have_geom && pub_occupancy1 &&
      static_cast<int>(xy2.rows()) < free_tile_min_points) {
    // too few surviving returns to trust a free-space claim: deposit a
    // neutral tile — logodds 0 adds nothing and stays exactly reversible
    kf.logodds.assign(static_cast<size_t>(sub_rows_) * sub_cols_, 0.0f);
  } else if (have_geom && pub_occupancy1) {
    cv::Mat mask = cv::Mat::zeros(sub_rows_, sub_cols_, CV_32F);

    if (xy2.rows() > 0 && num_bearings > 0 && num_ranges > 0) {
      const double b_lo = oculus_.bearings.front();
      const double b_hi = oculus_.bearings.back();
      for (int p = 0; p < xy2.rows(); ++p) {
        const double px = xy2(p, 0), py = xy2(p, 1);
        if (!std::isfinite(px) || !std::isfinite(py)) continue;
        // Bin the hit into the fan column that fit_grid will place AT the
        // feature's coordinates: the fan's local Y is frame_y_sign*sin(B)*R,
        // so the matching column satisfies sin(B_col) = py/(frame_y_sign*R),
        // i.e. atan2(frame_y_sign*py, px). For enu_world (+1) this is the
        // historical atan2(py, px); for frame_y_sign = -1 the old form put
        // every hit in the mirrored column, flipping both map products
        // port<->starboard against the trajectory.
        double bearing = std::atan2(frame_y_sign * py, px);
        bearing = std::min(std::max(bearing, b_lo), b_hi);
        int col = static_cast<int>(std::lround(b2c_(bearing)));
        col = std::min(std::max(col, 0), num_bearings - 1);
        const double range = std::hypot(px, py);
        int row = static_cast<int>(std::lround(
          (range - fan_range_min_) / oculus_.range_resolution - 1.0));
        row = std::min(std::max(row, 0), num_ranges - 1);
        mask.at<float>(row / r_skip_, col / c_skip_) = 1.0f;
      }

      // inflate each hit across a Gaussian bearing/range footprint, normalized
      // so a lone hit peaks at hit_prob (mapping.py:189-216)
      const int hc = static_cast<int>(std::lround(
        inflation_angle / oculus_.angular_resolution / c_skip_));
      const int hr = static_cast<int>(std::lround(
        inflation_range / oculus_.range_resolution / r_skip_));
      cv::Mat kr = cv::getGaussianKernel(2 * hr + 1, -1, CV_32F);
      cv::Mat kc = cv::getGaussianKernel(2 * hc + 1, -1, CV_32F);
      cv::Mat kernel = kr * kc.t();
      cv::filter2D(mask, mask, CV_32F, kernel, cv::Point(-1, -1), 0.0,
                   cv::BORDER_CONSTANT);
      const float center = kernel.at<float>(hr, hc);
      if (center > 0.0f)
        mask.convertTo(mask, CV_32F, static_cast<double>(hit_prob) / center);
      cv::max(mask, 0.5, mask);
      cv::min(mask, static_cast<double>(hit_prob), mask);

      // Mark cells before the first hit (per bearing column) as free.
      //
      // DIVERGENCE from bruce_slam (mapping.py:218-223), 2026-07-25. Upstream
      // finds the first hit with argmax, which returns 0 both for "hit in row
      // 0" and for "no hit anywhere" (all-equal array), then treats 0 as the
      // latter and frees the WHOLE column. So a single row-0 return erased
      // every hit in that bearing column and stamped miss-logodds out to max
      // range — carving a radial free-space stripe straight through real
      // structure. This port had reproduced the conflation deliberately.
      //
      // Row 0 is reachable: the row binding below clamps, so any feature at
      // or inside fan_range_min + 2*range_resolution lands there, and the
      // Gaussian inflation spreads a hit up to inflation_range into it. Pool
      // ringdown / near-field noise passing CFAR is enough.
      //
      // The explicit loop can tell the two cases apart, unlike argmax:
      // `first` stays sub_rows_ only when no cell exceeded the threshold.
      //
      // Done as two ROW-major sweeps rather than a per-column walk: the mask
      // is row-major, so scanning a column touched one useful float per cache
      // line. Columns never interact (each only reads and writes its own), so
      // hoisting the per-column state into a vector is the same computation.
      first_hit_.assign(static_cast<size_t>(sub_cols_), sub_rows_);
      for (int row = 0; row < sub_rows_; ++row) {
        const float* m = mask.ptr<float>(row);
        for (int col = 0; col < sub_cols_; ++col)
          if (first_hit_[col] == sub_rows_ && m[col] > 0.5f)
            first_hit_[col] = row;  // free only what lies BEFORE the hit
      }
      for (int row = 0; row < sub_rows_; ++row) {
        float* m = mask.ptr<float>(row);
        for (int col = 0; col < sub_cols_; ++col)
          if (row < first_hit_[col]) m[col] = static_cast<float>(miss_prob);
      }
    } else {
      mask.setTo(static_cast<float>(miss_prob));  // no returns -> all free
    }

    kf.logodds.resize(static_cast<size_t>(sub_rows_) * sub_cols_);
    size_t idx = 0;
    for (int row = 0; row < sub_rows_; ++row) {
      const float* m = mask.ptr<float>(row);
      for (int col = 0; col < sub_cols_; ++col) kf.logodds[idx++] = logit(m[col]);
    }
  }

  // --------- intensity/backscatter tile (mapping.py:244-246) ---------
  if (have_geom && pub_intensity) {
    kf.intensity.resize(static_cast<size_t>(sub_rows_) * sub_cols_, 0);
    size_t idx = 0;
    for (int ri = 0; ri < num_ranges; ri += r_skip_) {
      for (int ci = 0; ci < num_bearings; ci += c_skip_) {
        uint32_t v = 0;
        // Beam read matching the occupancy evidence's chirality: with
        // frame_y_sign = +1 the evidence in fan column ci comes from the
        // NEGATED native bearing (mirrored read, see mirror_col_); with
        // frame_y_sign = -1 the corrected hit binning above puts evidence at
        // the native bearing, so read the native column.
        const int mc =
          (frame_y_sign > 0 && ci < static_cast<int>(mirror_col_.size()))
            ? mirror_col_[ci]
            : ci;
        if (ri < ping.image.rows && mc < ping.image.cols)
          v = ping.image.at<uint8_t>(ri, mc);
        if (idx < kf.intensity.size()) kf.intensity[idx] = v;
        ++idx;
      }
    }
  }

  fit_grid(kf);
  inc_grid(kf);

  // pad any missed keyframe slots, then store at index `key` (mapping.py:252-255)
  while (static_cast<int>(keyframes_.size()) < key) keyframes_.push_back(Submap{});
  keyframes_.push_back(std::move(kf));
}

void Mapping::add_skipped(int key, const gtsam::Pose2& pose)
{
  // keyframe with no recoverable ping/feature data: store an empty, invalid
  // tile so the in-order builder advances past it instead of wedging every
  // later keyframe behind it. Invisible to the grids (no sonar_xy -> fit_grid
  // no-ops) and to update_pose (valid == false).
  while (static_cast<int>(keyframes_.size()) < key) keyframes_.push_back(Submap{});
  Submap kf;
  kf.k = static_cast<int>(keyframes_.size());
  kf.pose = pose;
  kf.valid = false;
  keyframes_.push_back(std::move(kf));
}

void Mapping::update_pose(int key, const gtsam::Pose2& new_pose)
{
  if (key < 0 || key >= static_cast<int>(keyframes_.size())) return;
  Submap& kf = keyframes_[key];
  if (!kf.valid) return;
  if (!pose_changed(kf.pose, new_pose)) return;

  kf.pose = new_pose;
  dec_grid(kf);   // remove the tile at its old pose
  fit_grid(kf);   // re-place at the optimized pose
  inc_grid(kf);   // re-add
}

void Mapping::fit_grid(Submap& kf)
{
  kf.r.clear();
  kf.c.clear();
  kf.l.clear();
  kf.i.clear();
  if (!kf.sonar_xy || kf.sonar_xy->rows() == 0) return;  // empty tile, nothing to place

  const double yaw = kf.pose.theta();
  const double cs = std::cos(yaw), sn = std::sin(yaw);
  const double tx = kf.pose.x(), ty = kf.pose.y();
  const Matrix& xy = *kf.sonar_xy;
  const int N = static_cast<int>(xy.rows());

  // scratch reused across keyframes: a loop closure re-fits every moved
  // keyframe, so a per-call pair of N-element allocations is one allocation
  // per keyframe per correction round
  std::vector<int>& r = fit_r_;
  std::vector<int>& c = fit_c_;
  r.resize(static_cast<size_t>(N));
  c.resize(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    const double X = xy(i, 0);
    const double Y = frame_y_sign * xy(i, 1);
    const double wx = cs * X - sn * Y + tx;
    const double wy = sn * X + cs * Y + ty;
    r[i] = static_cast<int>(std::lround((wy - y0) / resolution));
    c[i] = static_cast<int>(std::lround((wx - x0) / resolution));
  }

  adjust_bounds(r, c);  // may grow the grids and shift every cached tile + r/c

  // Collapse pixels landing in the same cell to one deposit
  // (mapping.py:491-502), keeping the FIRST occurrence in fan order.
  //
  // Membership is tested against a dense byte map over the TILE's own
  // bounding box rather than a hash set over global cell ids. The fan is
  // bounded by the sonar's max range, so that box is O(N) — it never scales
  // with the map — and this turns ~N hash inserts per keyframe into ~N array
  // probes. On a loop closure that re-places hundreds of keyframes, the hash
  // was a measurable share of the correction.
  int rlo = r[0], rhi = r[0], clo = c[0], chi = c[0];
  for (int i = 1; i < N; ++i) {
    rlo = std::min(rlo, r[i]);
    rhi = std::max(rhi, r[i]);
    clo = std::min(clo, c[i]);
    chi = std::max(chi, c[i]);
  }
  const size_t box_w = static_cast<size_t>(chi - clo) + 1;
  const size_t box_h = static_cast<size_t>(rhi - rlo) + 1;
  seen_.assign(box_w * box_h, 0);

  kf.r.reserve(static_cast<size_t>(N));
  kf.c.reserve(static_cast<size_t>(N));
  if (pub_occupancy1) kf.l.reserve(static_cast<size_t>(N));
  if (pub_intensity) kf.i.reserve(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    const size_t id =
      static_cast<size_t>(r[i] - rlo) * box_w + static_cast<size_t>(c[i] - clo);
    if (seen_[id]) continue;
    seen_[id] = 1;
    kf.r.push_back(r[i]);
    kf.c.push_back(c[i]);
    if (pub_occupancy1) kf.l.push_back(kf.logodds[i]);
    if (pub_intensity) kf.i.push_back(kf.intensity[i]);
  }
}

void Mapping::inc_grid(const Submap& kf)
{
  if (kf.r.empty()) return;
  // the edited bounding box is folded into the same pass as the deposits
  // instead of four extra full scans of the tile
  int rlo = kf.r[0], rhi = kf.r[0], clo = kf.c[0], chi = kf.c[0];
  for (size_t j = 0; j < kf.r.size(); ++j) {
    const int rr = kf.r[j], cc = kf.c[j];
    if (pub_occupancy1) logodds_grid_(rr, cc) += kf.l[j];
    if (pub_intensity) {
      intensity_grid_(rr, cc) += kf.i[j];
      counter_grid_(rr, cc) += 1;
    }
    rlo = std::min(rlo, rr);
    rhi = std::max(rhi, rr);
    clo = std::min(clo, cc);
    chi = std::max(chi, cc);
  }
  rmin_ = std::min(rmin_, rlo);
  rmax_ = std::max(rmax_, rhi);
  cmin_ = std::min(cmin_, clo);
  cmax_ = std::max(cmax_, chi);
}

void Mapping::dec_grid(const Submap& kf)
{
  for (size_t j = 0; j < kf.r.size(); ++j) {
    const int rr = kf.r[j], cc = kf.c[j];
    if (pub_occupancy1) logodds_grid_(rr, cc) -= kf.l[j];
    if (pub_intensity) {
      intensity_grid_(rr, cc) -= kf.i[j];
      counter_grid_(rr, cc) -= 1;
    }
  }
  // the edited bounding box never shrinks (mapping.py:467)
}

void Mapping::adjust_bounds(std::vector<int>& r, std::vector<int>& c)
{
  if (r.empty()) return;

  // Growth is applied in ONE step per side instead of one `inc` at a time.
  //
  // The per-`inc` loops this replaces re-scanned r/c, reallocated and copied
  // the whole grid, and shifted EVERY cached tile of EVERY keyframe on each
  // iteration — so a keyframe landing k increments outside cost k full grid
  // copies and k passes over the entire map history. A vehicle leaving the
  // initial 100 m box, or any `inc` small relative to the excursion, paid that
  // quadratic directly. The counts below are exactly the iteration counts of
  // those loops, so the resulting geometry is identical; the scalar
  // bookkeeping is still accumulated step-by-step so the floating-point
  // origin/extent match the incremental form bit for bit.
  const auto steps_below = [](long long lo, int inc) -> long long {
    return lo < 0 ? (-lo + inc - 1) / inc : 0;  // smallest n with lo + n*inc >= 0
  };
  const auto steps_above = [](long long hi, long long limit, int inc) -> long long {
    return hi >= limit ? (hi - limit) / inc + 1 : 0;  // smallest n with hi < limit + n*inc
  };

  long long rlo = r[0], rhi = r[0], clo = c[0], chi = c[0];
  for (size_t i = 1; i < r.size(); ++i) {
    rlo = std::min<long long>(rlo, r[i]);
    rhi = std::max<long long>(rhi, r[i]);
    clo = std::min<long long>(clo, c[i]);
    chi = std::max<long long>(chi, c[i]);
  }

  // --------------------------------------------------------------- rows
  const long long n_top = steps_below(rlo, inc_r_);
  // the append loop runs AFTER the prepend shift, so it sees shifted indices
  // against the already-grown row count
  const long long n_bot =
    steps_above(rhi + n_top * inc_r_, static_cast<long long>(rows_) + n_top * inc_r_,
                inc_r_);
  if (n_top > 0 || n_bot > 0) {
    const int top = static_cast<int>(n_top * inc_r_);
    const int bottom = static_cast<int>(n_bot * inc_r_);
    if (pub_occupancy1) grow_rows(logodds_grid_, top, bottom);
    if (pub_intensity) {
      grow_rows(intensity_grid_, top, bottom);
      grow_rows(counter_grid_, top, bottom);
    }
    for (long long i = 0; i < n_top; ++i) {  // prepend bookkeeping (mapping.py:504-527)
      rows_ += inc_r_;
      rmin_ += inc_r_;
      rmax_ += inc_r_;
      y0 -= inc_r_ * resolution;
      height += inc_r_ * resolution;
    }
    for (long long i = 0; i < n_bot; ++i) {  // append bookkeeping (mapping.py:528-544)
      rows_ += inc_r_;
      height += inc_r_ * resolution;
    }
    if (top > 0) {
      for (auto& kf : keyframes_)
        for (int& rr : kf.r) rr += top;
      for (int& rr : r) rr += top;
    }
  }

  // ------------------------------------------------------------ columns
  // NOTE: fixes an upstream bug — the Python intensity/counter branches used
  // np.r_ (rows) with inc_r here; the correct growth for a column shift is
  // np.c_ with inc_c (pub_intensity was off upstream, so it was never hit).
  const long long n_left = steps_below(clo, inc_c_);
  const long long n_right =
    steps_above(chi + n_left * inc_c_, static_cast<long long>(cols_) + n_left * inc_c_,
                inc_c_);
  if (n_left > 0 || n_right > 0) {
    const int left = static_cast<int>(n_left * inc_c_);
    const int right = static_cast<int>(n_right * inc_c_);
    if (pub_occupancy1) grow_cols(logodds_grid_, left, right);
    if (pub_intensity) {
      grow_cols(intensity_grid_, left, right);
      grow_cols(counter_grid_, left, right);
    }
    for (long long i = 0; i < n_left; ++i) {  // mapping.py:545-567
      cols_ += inc_c_;
      cmin_ += inc_c_;
      cmax_ += inc_c_;
      x0 -= inc_c_ * resolution;
      width += inc_c_ * resolution;
    }
    for (long long i = 0; i < n_right; ++i) {  // mapping.py:568-585
      cols_ += inc_c_;
      width += inc_c_ * resolution;
    }
    if (left > 0) {
      for (auto& kf : keyframes_)
        for (int& cc : kf.c) cc += left;
      for (int& cc : c) cc += left;
    }
  }
}

OccGrid Mapping::get_occupancy_grid() const
{
  OccGrid g;
  if (!pub_occupancy1 || rmax_ < rmin_ || cmax_ < cmin_) return g;
  const int H = rmax_ - rmin_ + 1, W = cmax_ - cmin_ + 1;
  g.width = W;
  g.height = H;
  g.resolution = resolution;
  g.origin_x = x0 + (cmin_ - 0.5) * resolution;
  g.origin_y = y0 + (rmin_ - 0.5) * resolution;
  g.data.resize(static_cast<size_t>(H) * W);
  for (int rr = rmin_; rr <= rmax_; ++rr) {
    for (int cc = cmin_; cc <= cmax_; ++cc) {
      const float p = expit(logodds_grid_(rr, cc));
      int occ = static_cast<int>(std::lround(100.0f * p));
      occ = std::min(std::max(occ, 0), 100);
      g.data[static_cast<size_t>(rr - rmin_) * W + (cc - cmin_)] =
        static_cast<int8_t>(occ);
    }
  }
  return g;
}

OccGrid Mapping::get_intensity_grid() const
{
  OccGrid g;
  if (!pub_intensity || rmax_ < rmin_ || cmax_ < cmin_) return g;
  const int H = rmax_ - rmin_ + 1, W = cmax_ - cmin_ + 1;
  g.width = W;
  g.height = H;
  g.resolution = resolution;
  g.origin_x = x0 + (cmin_ - 0.5) * resolution;
  g.origin_y = y0 + (rmin_ - 0.5) * resolution;
  g.data.resize(static_cast<size_t>(H) * W);
  for (int rr = rmin_; rr <= rmax_; ++rr) {
    for (int cc = cmin_; cc <= cmax_; ++cc) {
      const uint32_t cnt = counter_grid_(rr, cc);
      int8_t occ = -1;
      if (cnt > 0) {
        const double mean = static_cast<double>(intensity_grid_(rr, cc)) / cnt;
        occ = static_cast<int8_t>(std::lround(mean / 255.0 * 100.0));
      }
      g.data[static_cast<size_t>(rr - rmin_) * W + (cc - cmin_)] = occ;
    }
  }
  return g;
}

}  // namespace sonar_slam
