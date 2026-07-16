// Port of bruce_slam mapping.py (Mapping + Submap). See mapping.hpp for the
// design; comments here point at the upstream lines being reproduced.
#include "sonar_slam_cpp/mapping.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_set>

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
  hit_logodds_ = logit(static_cast<float>(hit_prob));
  miss_logodds_ = logit(static_cast<float>(miss_prob));

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
  kf.k = static_cast<int>(keyframes_.size());
  kf.pose = pose;
  kf.valid = true;

  const int num_ranges = oculus_.num_ranges;
  const int num_bearings = oculus_.num_bearings;

  // (Re)build the sonar-fan pixel geometry only when the range/aperture
  // changed (mapping.py:152-168 — keyframes otherwise reuse the previous fan).
  if ((changed || !sonar_xy_ || sub_rows_ == 0) &&
      oculus_.range_resolution > 0.0 && num_ranges > 0 && num_bearings > 0) {
    r_skip_ = std::max(1, static_cast<int>(std::floor(resolution / oculus_.range_resolution)));
    const double bearing_arc_res = oculus_.angular_resolution * oculus_.max_range;
    c_skip_ = std::max(1, static_cast<int>(std::floor(resolution / bearing_arc_res)));

    sub_rows_ = (num_ranges + r_skip_ - 1) / r_skip_;
    sub_cols_ = (num_bearings + c_skip_ - 1) / c_skip_;

    Matrix xy(sub_rows_ * sub_cols_, 2);
    int idx = 0;
    for (int ri = 0; ri < num_ranges; ri += r_skip_) {
      const double R = oculus_.range_resolution * (1 + ri);  // ranges[ri]
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
  }
  kf.sonar_xy = sonar_xy_;
  // no valid fan geometry yet (e.g. a malformed first ping): store an empty
  // tile so the keyframe index still advances but nothing is deposited.
  const bool have_geom = sonar_xy_ && sub_rows_ > 0 && sub_cols_ > 0;

  // --------- occupancy logodds tile (mapping.py:170-228) ---------
  if (have_geom && pub_occupancy1) {
    cv::Mat mask = cv::Mat::zeros(sub_rows_, sub_cols_, CV_32F);

    Matrix xy2(0, 2);
    if (points.rows() > 0) {
      xy2 = points.leftCols(2);
      if (outlier_filter_min_points > 1 && xy2.rows() > 0)
        xy2 = remove_outlier(xy2, outlier_filter_radius, outlier_filter_min_points);
    }

    if (xy2.rows() > 0 && num_bearings > 0 && num_ranges > 0) {
      const double b_lo = oculus_.bearings.front();
      const double b_hi = oculus_.bearings.back();
      for (int p = 0; p < xy2.rows(); ++p) {
        const double px = xy2(p, 0), py = xy2(p, 1);
        if (!std::isfinite(px) || !std::isfinite(py)) continue;
        double bearing = std::atan2(py, px);
        bearing = std::min(std::max(bearing, b_lo), b_hi);
        int col = static_cast<int>(std::lround(b2c_(bearing)));
        col = std::min(std::max(col, 0), num_bearings - 1);
        const double range = std::hypot(px, py);
        int row = static_cast<int>(std::lround(range / oculus_.range_resolution - 1.0));
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

      // mark cells before the first hit (per bearing column) as free. Upstream
      // quirk preserved: a hit in row 0 makes argmax==0, which the reset below
      // treats as "no hit" and frees the whole column (mapping.py:218-223).
      for (int col = 0; col < sub_cols_; ++col) {
        int first = sub_rows_;
        for (int row = 0; row < sub_rows_; ++row) {
          if (mask.at<float>(row, col) > 0.5f) {
            first = (row == 0) ? sub_rows_ : row;
            break;
          }
        }
        for (int row = 0; row < first; ++row)
          mask.at<float>(row, col) = static_cast<float>(miss_prob);
      }
    } else {
      mask.setTo(static_cast<float>(miss_prob));  // no returns -> all free
    }

    kf.logodds.resize(static_cast<size_t>(sub_rows_) * sub_cols_);
    size_t idx = 0;
    for (int row = 0; row < sub_rows_; ++row)
      for (int col = 0; col < sub_cols_; ++col)
        kf.logodds[idx++] = logit(mask.at<float>(row, col));
  }

  // --------- intensity/backscatter tile (mapping.py:244-246) ---------
  if (have_geom && pub_intensity) {
    kf.intensity.resize(static_cast<size_t>(sub_rows_) * sub_cols_, 0);
    size_t idx = 0;
    for (int ri = 0; ri < num_ranges; ri += r_skip_) {
      for (int ci = 0; ci < num_bearings; ci += c_skip_) {
        uint32_t v = 0;
        if (ri < ping.image.rows && ci < ping.image.cols)
          v = ping.image.at<uint8_t>(ri, ci);
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

  std::vector<int> r(N), c(N);
  for (int i = 0; i < N; ++i) {
    const double X = xy(i, 0);
    const double Y = frame_y_sign * xy(i, 1);
    const double wx = cs * X - sn * Y + tx;
    const double wy = sn * X + cs * Y + ty;
    r[i] = static_cast<int>(std::lround((wy - y0) / resolution));
    c[i] = static_cast<int>(std::lround((wx - x0) / resolution));
  }

  adjust_bounds(r, c);  // may grow the grids and shift every cached tile + r/c

  // collapse pixels landing in the same cell to one deposit (mapping.py:491-502)
  std::unordered_set<long long> seen;
  seen.reserve(static_cast<size_t>(N));
  kf.r.clear();
  kf.c.clear();
  kf.l.clear();
  kf.i.clear();
  for (int i = 0; i < N; ++i) {
    const long long id = static_cast<long long>(r[i]) * cols_ + c[i];
    if (seen.insert(id).second) {
      kf.r.push_back(r[i]);
      kf.c.push_back(c[i]);
      if (pub_occupancy1) kf.l.push_back(kf.logodds[i]);
      if (pub_intensity) kf.i.push_back(kf.intensity[i]);
    }
  }
}

void Mapping::inc_grid(const Submap& kf)
{
  for (size_t j = 0; j < kf.r.size(); ++j) {
    const int rr = kf.r[j], cc = kf.c[j];
    if (pub_occupancy1) logodds_grid_(rr, cc) += kf.l[j];
    if (pub_intensity) {
      intensity_grid_(rr, cc) += kf.i[j];
      counter_grid_(rr, cc) += 1;
    }
  }
  if (!kf.r.empty()) {
    rmin_ = std::min(rmin_, *std::min_element(kf.r.begin(), kf.r.end()));
    rmax_ = std::max(rmax_, *std::max_element(kf.r.begin(), kf.r.end()));
    cmin_ = std::min(cmin_, *std::min_element(kf.c.begin(), kf.c.end()));
    cmax_ = std::max(cmax_, *std::max_element(kf.c.begin(), kf.c.end()));
  }
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
  auto any_lt = [](const std::vector<int>& v, int bound) {
    for (int x : v) if (x < bound) return true;
    return false;
  };
  auto any_ge = [](const std::vector<int>& v, int bound) {
    for (int x : v) if (x >= bound) return true;
    return false;
  };

  // prepend rows (mapping.py:504-527)
  while (any_lt(r, 0)) {
    if (pub_occupancy1) grow_rows(logodds_grid_, inc_r_, 0);
    if (pub_intensity) {
      grow_rows(intensity_grid_, inc_r_, 0);
      grow_rows(counter_grid_, inc_r_, 0);
    }
    rows_ += inc_r_;
    rmin_ += inc_r_;
    rmax_ += inc_r_;
    y0 -= inc_r_ * resolution;
    height += inc_r_ * resolution;
    for (auto& kf : keyframes_)
      for (int& rr : kf.r) rr += inc_r_;
    for (int& rr : r) rr += inc_r_;
  }
  // append rows (mapping.py:528-544)
  while (any_ge(r, rows_)) {
    if (pub_occupancy1) grow_rows(logodds_grid_, 0, inc_r_);
    if (pub_intensity) {
      grow_rows(intensity_grid_, 0, inc_r_);
      grow_rows(counter_grid_, 0, inc_r_);
    }
    rows_ += inc_r_;
    height += inc_r_ * resolution;
  }
  // prepend columns (mapping.py:545-567). NOTE: fixes an upstream bug — the
  // Python intensity/counter branches used np.r_ (rows) with inc_r here; the
  // correct growth for a column shift is np.c_ with inc_c (pub_intensity was
  // off upstream, so the bug was never hit).
  while (any_lt(c, 0)) {
    if (pub_occupancy1) grow_cols(logodds_grid_, inc_c_, 0);
    if (pub_intensity) {
      grow_cols(intensity_grid_, inc_c_, 0);
      grow_cols(counter_grid_, inc_c_, 0);
    }
    cols_ += inc_c_;
    cmin_ += inc_c_;
    cmax_ += inc_c_;
    x0 -= inc_c_ * resolution;
    width += inc_c_ * resolution;
    for (auto& kf : keyframes_)
      for (int& cc : kf.c) cc += inc_c_;
    for (int& cc : c) cc += inc_c_;
  }
  // append columns (mapping.py:568-585)
  while (any_ge(c, cols_)) {
    if (pub_occupancy1) grow_cols(logodds_grid_, 0, inc_c_);
    if (pub_intensity) {
      grow_cols(intensity_grid_, 0, inc_c_);
      grow_cols(counter_grid_, 0, inc_c_);
    }
    cols_ += inc_c_;
    width += inc_c_ * resolution;
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
  g.origin_x = x0 + cmin_ * resolution;
  g.origin_y = y0 + rmin_ * resolution;
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
  g.origin_x = x0 + cmin_ * resolution;
  g.origin_y = y0 + rmin_ * resolution;
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
