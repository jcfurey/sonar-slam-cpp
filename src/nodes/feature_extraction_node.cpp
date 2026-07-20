// Feature extraction node: port of bruce_slam feature_extraction.py.
// CFAR target detection on the polar sonar image (GPU with CPU fallback),
// polar -> Cartesian remap, voxel downsampling and radius outlier removal,
// publishing a PointCloud2 stamped with the source ping time.
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "sonar_slam_cpp/cfar.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/gpu.hpp"
#include "sonar_slam_cpp/interp.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"

namespace sonar_slam {

namespace {

// Map-stream cloud: 6 float fields packed in the PointXYZI 32-byte stride
// (x@0 y@4 z@8 intensity@16 range@20 incidence@24). MUST stay byte-identical
// to sonar_proc's map_points layout — the Accumulator's field-validating
// concatenate is the enforcement point.
// Features carry no incidence estimate -> -1 (unknown).
sensor_msgs::msg::PointCloud2 make_map_cloud(const Matrix& xyz,
                                             const Eigen::VectorXf& intens,
                                             const Eigen::VectorXf& ranges)
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(xyz.rows());
  msg.is_bigendian = false;
  msg.is_dense = true;
  static const char* names[6] = {"x", "y", "z", "intensity", "range", "incidence"};
  static const uint32_t offsets[6] = {0, 4, 8, 16, 20, 24};
  for (int i = 0; i < 6; ++i) {
    sensor_msgs::msg::PointField f;
    f.name = names[i];
    f.offset = offsets[i];
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    msg.fields.push_back(f);
  }
  msg.point_step = 32;
  msg.row_step = msg.point_step * msg.width;
  msg.data.assign(msg.row_step, 0);
  for (long r = 0; r < xyz.rows(); ++r) {
    float* p = reinterpret_cast<float*>(msg.data.data() + r * msg.point_step);
    p[0] = xyz(r, 0);
    p[1] = xyz(r, 1);
    p[2] = xyz(r, 2);
    p[4] = r < intens.size() ? intens(r) : 1.0f;
    p[5] = r < ranges.size() ? ranges(r) : 0.0f;
    p[6] = -1.0f;
  }
  return msg;
}

}  // namespace

class FeatureExtractionNode : public SlamNodeBase
{
public:
  FeatureExtractionNode() : SlamNodeBase("feature_extraction_node")
  {
    // CFAR parameters
    ntc_ = get_int("CFAR/Ntc");
    ngc_ = get_int("CFAR/Ngc");
    pfa_ = get_double("CFAR/Pfa");
    rank_ = get_int("CFAR/rank");
    alg_ = CFAR::alg_from_string(get_string("CFAR/alg", "SOCA"));
    threshold_ = get_int("filter/threshold");

    // Live tuning at sea: the CFAR/filter knobs are dynamic —
    //   ros2 param set /bruce/slam/feature_extraction CFAR.Pfa 0.005
    // rebuilds the detector for the next ping, no relaunch. Safe under the
    // default single-threaded executor (the swap and detect() serialize).
    param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter>& params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        auto as_num = [](const rclcpp::Parameter& p) {
          return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
                   ? static_cast<double>(p.as_int())
                   : p.as_double();
        };
        // stage into locals and commit ONLY after the CFAR constructor
        // accepts them — a rejected set must not poison the cached values
        // (rclcpp keeps the old parameter, so the members must match it)
        int ntc = ntc_, ngc = ngc_, rank = rank_, threshold = threshold_;
        double pfa = pfa_;
        CFAR::Alg alg = alg_;
        // stage the filter.* knobs too — same rule: a rejected set (any
        // parameter in the same request) must not leave a member mutated
        // while rclcpp keeps the old value
        double resolution = resolution_, radius = outlier_filter_radius_;
        int min_points = outlier_filter_min_points_, skip = skip_;
        bool rebuild = false;
        try {
          for (const auto& p : params) {
            const std::string& n = p.get_name();
            if (n == "CFAR.Ntc") {
              ntc = static_cast<int>(as_num(p));
              rebuild = true;
            } else if (n == "CFAR.Ngc") {
              ngc = static_cast<int>(as_num(p));
              rebuild = true;
            } else if (n == "CFAR.Pfa") {
              pfa = as_num(p);
              rebuild = true;
            } else if (n == "CFAR.rank") {
              rank = static_cast<int>(as_num(p));
              rebuild = true;
            } else if (n == "CFAR.alg") {
              alg = CFAR::alg_from_string(p.as_string());
            } else if (n == "filter.threshold") {
              threshold = static_cast<int>(as_num(p));
            } else if (n == "filter.resolution") {
              resolution = as_num(p);
            } else if (n == "filter.radius") {
              radius = as_num(p);
            } else if (n == "filter.min_points") {
              min_points = static_cast<int>(as_num(p));
            } else if (n == "filter.skip") {
              skip = static_cast<int>(as_num(p));
            }
          }
          if (rebuild)
            detector_ = std::make_unique<CFAR>(ntc, ngc, pfa, rank);
          ntc_ = ntc;
          ngc_ = ngc;
          pfa_ = pfa;
          rank_ = rank;
          alg_ = alg;
          threshold_ = threshold;
          resolution_ = resolution;
          outlier_filter_radius_ = radius;
          outlier_filter_min_points_ = min_points;
          skip_ = skip;
          if (rebuild)
            RCLCPP_INFO(get_logger(),
                        "CFAR rebuilt: Ntc %d, Ngc %d, Pfa %g, rank %d",
                        ntc_, ngc_, pfa_, rank_);
        } catch (const std::exception& e) {
          result.successful = false;
          result.reason = e.what();
        }
        return result;
      });

    // point cloud filtering parameters
    resolution_ = get_double("filter/resolution");
    outlier_filter_radius_ = get_double("filter/radius");
    outlier_filter_min_points_ = get_int("filter/min_points");
    skip_ = get_int("filter/skip");

    compressed_images_ = get_bool("compressed_images");

    detector_ = std::make_unique<CFAR>(ntc_, ngc_, pfa_, rank_);

    // sonar driver (pluggable adapter + configurable topic); blank driver
    // auto-selects the Oculus adapter matching compressed_images
    std::string driver = get_string("sonar/driver", "");
    if (driver.empty())
      driver = compressed_images_ ? "oculus_compressed" : "oculus_uncompressed";
    std::string topic = get_string("sonar/topic", "");
    if (topic.empty()) {
      topic = default_sonar_topic(driver);
      if (topic.empty())
        throw std::runtime_error(
          "sonar/topic must be set explicitly for sonar driver '" + driver + "'");
    }

    sonar_sub_ = subscribe_sonar(
      this, driver, topic, rclcpp::SensorDataQoS(),
      [this](const SonarPing& ping) { callback(ping); });

    feature_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SONAR_FEATURE_TOPIC, 10);
    feature_img_pub_ = create_publisher<sensor_msgs::msg::Image>(
      SONAR_FEATURE_IMG_TOPIC, 10);

    // Optional union map stream: republish the (non-sentinel) feature clouds
    // as padded XYZI onto a shared topic with sonar_proc's candidate stream,
    // so the map accumulator/vdb see the CFAR detector's structure too —
    // CFAR holds a constant false-alarm rate against the local background
    // and finds low-contrast structure the candidate edge detector misses.
    const std::string map_topic = get_string("map_points_topic", "");
    if (!map_topic.empty())
      map_points_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, 10);
    // residual TVG for the map stream (identity at 0) — keep these equal to
    // sonar_proc's values so the union's two intensity sources stay on one
    // radiometric scale
    tvg_spread_db_ = get_double("tvg_spread_db", 0.0);
    tvg_absorption_db_per_m_ = get_double("tvg_absorption_db_per_m", 0.0);

    // Sonar head pitch. The Oculus rides the Deep Trekker's pivoting head
    // (cameraHead.tilt), which sweeps up to +/-54 deg; treating it as a level
    // scanner mis-places every feature. system_interface republishes the head
    // angle on /base/sensor_tilt_angle (DEGREES). We fold it into the
    // published fan as a pitch about the lateral axis so the cloud sits at the
    // true sonar attitude in base_link. slam_node consumes only (x, y), so it
    // automatically receives the horizontal (de-tilted) projection; z carries
    // the pitch for viz / 3D consumers. If the topic never arrives the angle
    // stays 0 and behaviour is byte-for-byte the old level assumption.
    // Head-pitch FRAME GATE (2026-07-16, from CHL_Pool ping forensics):
    // bruce-slam's planar registration assumes a near-level fan. During the
    // ±54° head sweep the frame is DOMINATED by the floor bowl — a huge
    // bright curved band whose planar (x,y) projection registers against
    // real walls from other frames and pulls the graph (the observed "bad
    // SLAM corrections"), and whose features stamp curved bands into the
    // mapping tiles. Skip feature frames when |head pitch| exceeds this
    // bound (rad; 0 disables the gate). The NaN sentinel keeps the SLAM
    // synchronizer advancing; dead-reckoning odometry factors carry the
    // graph between level frames, and the sweep frames still feed the map
    // through sonar_proc's candidates/dense streams (which place them with
    // the floor/wall projections).
    max_head_pitch_ = get_double("max_head_pitch", 0.0);

    apply_head_tilt_ = get_bool("apply_head_tilt", true);
    if (apply_head_tilt_) {
      const std::string tilt_topic =
        get_string("head_tilt_topic", "/base/sensor_tilt_angle");
      tilt_sub_ = create_subscription<std_msgs::msg::Float32>(
        tilt_topic, rclcpp::SensorDataQoS(),
        [this](const std_msgs::msg::Float32& m) {
          // Float32 carries no stamp: pair each sample with its arrival time
          // so per-ping lookups interpolate at the ping stamp instead of
          // trusting the latest value — the head sweeps ~0.5 rad/s, and a
          // lagging latch both admits floor-bowl frames through the pitch
          // gate with a stale "level" reading and mis-projects z on the
          // frames it passes
          const int64_t t = this->now().nanoseconds();
          const float rad = m.data * static_cast<float>(M_PI) / 180.0f;
          std::lock_guard<std::mutex> lock(tilt_mutex_);
          if (!tilt_buf_.empty() && t < tilt_buf_.back().first)
            tilt_buf_.clear();  // time jumped backwards (bag loop): restart
          tilt_buf_.emplace_back(t, rad);
          while (tilt_buf_.size() > kTiltBufMax) tilt_buf_.pop_front();
        });
    }

    RCLCPP_INFO(get_logger(), "Feature extraction node initialized (GPU: %s)",
                gpu::available() ? "on" : "off, CPU fallback");
  }

private:
  // head pitch (rad) at time t: linear interpolation between the buffered
  // samples bracketing t, clamped to the nearest sample beyond the buffer
  // ends. An empty buffer (topic absent) returns 0 — the level assumption.
  float tilt_at(int64_t t_ns)
  {
    std::lock_guard<std::mutex> lock(tilt_mutex_);
    if (tilt_buf_.empty()) return 0.0f;
    // Clock-domain guard: samples are stamped with the node clock at ARRIVAL
    // (Float32 carries no stamp) while pings carry the driver clock. If the
    // two domains disagree (bag replay without use_sim_time, a driver on
    // device time), the ping stamp sits far outside the buffer and the edge
    // clamp below would pin every lookup to the OLDEST buffered sample —
    // strictly worse than the latest-value latch this interpolation replaced.
    // Fall back to the newest sample (the latch behavior) and say so once in
    // a while. The same fallback also covers a tilt stream that stalled.
    constexpr int64_t kDomainSlackNs = 5000000000LL;  // 5 s
    if (t_ns + kDomainSlackNs < tilt_buf_.front().first ||
        t_ns > tilt_buf_.back().first + kDomainSlackNs) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 30000,
        "head-tilt stamps and ping stamps are >5 s apart (different clock "
        "domains, or a stalled tilt stream); using the latest tilt sample "
        "instead of interpolating");
      return tilt_buf_.back().second;
    }
    if (t_ns <= tilt_buf_.front().first) return tilt_buf_.front().second;
    if (t_ns >= tilt_buf_.back().first) return tilt_buf_.back().second;
    const auto hi = std::lower_bound(
      tilt_buf_.begin(), tilt_buf_.end(), t_ns,
      [](const std::pair<int64_t, float>& s, int64_t t) { return s.first < t; });
    const auto lo = std::prev(hi);
    const double span = static_cast<double>(hi->first - lo->first);
    const double a = span > 0.0 ? static_cast<double>(t_ns - lo->first) / span : 0.0;
    return static_cast<float>((1.0 - a) * lo->second + a * hi->second);
  }

  // build (or refresh) the polar -> Cartesian maps when geometry changes
  void generate_map_xy(const SonarPing& ping)
  {
    const double res = ping.range_resolution;
    const double range_min = ping.range_min;
    // Polar row 0 is at range_min, so the Cartesian fan spans
    // [0, range_min + num_ranges*res]. Add ceil(range_min/res) near-field output
    // rows (kept at cell size `res`) so a nonzero-min-range multibeam is placed
    // at its true range. For an Oculus (range_min == 0) extra_rows is 0 and this
    // reduces byte-for-byte to the original grid.
    const int extra_rows = static_cast<int>(std::ceil(range_min / res));
    const int rows = ping.num_ranges + extra_rows;
    const double height = rows * res;
    const double width =
      std::sin((ping.bearings.back() - ping.bearings.front()) / 2.0) * height * 2.0;
    const int cols = static_cast<int>(std::ceil(width / res));

    // The cache key must cover EVERY input of the maps: res and range_min
    // (different offsets can produce identical derived dims), the grid dims,
    // and the full bearing vector (a beam-count or pattern change can leave
    // width/cols untouched while invalidating the bearing->column interp).
    if (res == res_ && range_min == range_min_ && rows == rows_ &&
        cols == cols_ && ping.bearings == bearings_)
      return;

    // bearing -> beam column, linear like feature_extraction.py
    std::vector<double> bx(ping.bearings.begin(), ping.bearings.end());
    std::vector<double> by(ping.bearings.size());
    for (std::size_t i = 0; i < by.size(); ++i) by[i] = static_cast<double>(i);
    const Interp1d f_bearings(bx, by, Interp1d::LINEAR, -1.0);

    // build into locals and commit the cache fields only after the maps exist:
    // a throwing allocation (oversized geometry) must not poison the cache, or
    // every subsequent identical ping would reuse empty/stale maps
    cv::Mat map_x(rows, cols, CV_32FC1);
    cv::Mat map_y(rows, cols, CV_32FC1);
    for (int r = 0; r < rows; ++r) {
      float* px = map_x.ptr<float>(r);
      float* py = map_y.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const double x = res * (rows - r);
        const double y = res * (-cols / 2.0 + c + 0.5);
        const double b = std::atan2(y, x);
        const double range = std::sqrt(x * x + y * y);
        // polar row = (range - range_min)/res; near-field cells (range <
        // range_min) map to a negative row and are sampled as border.
        py[c] = static_cast<float>((range - range_min) / res);
        px[c] = static_cast<float>(f_bearings(b));
      }
    }

    map_x_ = std::move(map_x);
    map_y_ = std::move(map_y);
    res_ = res;
    range_min_ = range_min;
    height_ = height;
    rows_ = rows;
    width_ = width;
    cols_ = cols;
    bearings_ = ping.bearings;
    // new maps -> new version so the GPU remap re-uploads its cached copy
    ++map_version_;
  }

  void publish_features(const builtin_interfaces::msg::Time& stamp,
                        const Matrix& points, float phi)
  {
    // Publish honest planar base_link coordinates: [x, y, 0]. The python
    // original packed the lateral coordinate into z ([x, 0, y] — a leftover
    // of bruce's roll-pi rviz static, which this stack does not use); that
    // drew the fan rolled 90 deg in an ENU viewer. The SLAM node's
    // consumption is adjusted in lockstep (slam_node.cpp slam_callback),
    // so the graph math is unchanged.
    // Rotate the sonar-plane fan (col0 = forward along the bore, col1 =
    // lateral) by the head pitch about the lateral axis: forward tips into
    // -z as the head looks down (tilt < 0). Lateral is the pitch axis and is
    // unchanged. phi == 0 reduces this to the old [x, y, 0]; the caller
    // interpolates phi at the ping stamp (tilt_at).
    const float cphi = std::cos(phi);
    const float sphi = std::sin(phi);
    Matrix xyz(points.rows(), 3);
    xyz.col(0) = points.col(0) * cphi;
    xyz.col(1) = points.col(1);
    xyz.col(2) = points.col(0) * sphi;

    sensor_msgs::msg::PointCloud2 msg = make_cloud_xyz(xyz);
    // the stamp of the source sonar image is CRITICAL to downstream sync
    msg.header.stamp = stamp;
    msg.header.frame_id = "base_link";
    feature_pub_->publish(msg);

    // union map stream (see constructor): skip the NaN skip-frame sentinel
    // (a sync signal for the slam node, poison for map consumers) and empty
    // frames (they only flood the chain/assembler with no-data warnings)
    if (map_points_pub_ && xyz.rows() > 0 && !std::isnan(xyz(0, 0))) {
      // real echo intensity (0..1): invert the extraction pixel->meters
      // mapping back into this ping's remapped grayscale. points col0 =
      // forward (rows axis), col1 = lateral (cols axis), pre-tilt — their
      // planar norm is the slant range.
      Eigen::VectorXf intens = Eigen::VectorXf::Ones(points.rows());
      Eigen::VectorXf ranges = Eigen::VectorXf::Zero(points.rows());
      for (long r = 0; r < points.rows(); ++r)
        ranges(r) = std::hypot(points(r, 0), points(r, 1));
      if (!cart_gray_.empty() && rows_ > 0 && height_ > 0 && width_ > 0) {
        for (long r = 0; r < points.rows(); ++r) {
          const int row = static_cast<int>(
            std::lround((1.0 - points(r, 0) / height_) * rows_));
          const int col = static_cast<int>(
            std::lround(cols_ / 2.0 - points(r, 1) * cols_ / width_));
          if (row >= 0 && row < cart_gray_.rows && col >= 0 &&
              col < cart_gray_.cols)
            intens(r) = cart_gray_.at<std::uint8_t>(row, col) / 255.0f;
          // residual TVG (dB; identity when both coefficients are 0) — must
          // match sonar_proc's mapIntensity so the union stays on one scale
          if (ranges(r) > 1e-3f &&
              (tvg_spread_db_ != 0.0 || tvg_absorption_db_per_m_ != 0.0)) {
            const double db =
              tvg_spread_db_ * std::log10(static_cast<double>(ranges(r))) +
              tvg_absorption_db_per_m_ * 2.0 * (static_cast<double>(ranges(r)) - 1.0);
            intens(r) = static_cast<float>(std::clamp(
              static_cast<double>(intens(r)) * std::pow(10.0, db / 20.0), 0.0, 1.0));
          }
        }
      }
      sensor_msgs::msg::PointCloud2 map_msg = make_map_cloud(xyz, intens, ranges);
      map_msg.header = msg.header;
      map_points_pub_->publish(map_msg);
    }
  }

  // interp: 0 = nearest, 1 = bilinear. Use nearest for a binary detection
  // mask (correct for a 0/1 image and bit-exact between the CPU and GPU remap
  // paths); use bilinear for the grayscale visualization image.
  cv::Mat remap_u8(const cv::Mat& src, int interp) const
  {
    const cv::Mat cont = src.isContinuous() ? src : src.clone();
#ifdef SONAR_SLAM_WITH_CUDA
    // maps are device-resident across calls (re-uploaded only when
    // map_version_ changes); a failing GPU falls through to cv::remap
    if (gpu::available()) {
      cv::Mat dst(map_x_.rows, map_x_.cols, CV_8UC1);
      if (gpu::remap_u8_cuda(cont.ptr<std::uint8_t>(), cont.rows, cont.cols,
                             map_x_.ptr<float>(), map_y_.ptr<float>(),
                             map_x_.rows, map_x_.cols, interp, map_version_,
                             dst.ptr<std::uint8_t>()))
        return dst;
    }
#endif
    cv::Mat dst;
    cv::remap(cont, dst, map_x_, map_y_,
              interp == 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
    return dst;
  }

  void callback(const SonarPing& ping)
  {
    // cheap skip test; skipped frames still publish an empty (NaN) cloud so
    // the SLAM node's time synchronizer keeps advancing
    ++frame_count_;
    const int ping_id = ping.ping_id != 0 ? ping.ping_id : frame_count_;
    // head pitch AT THE PING STAMP (interpolated), used by both the frame
    // gate and the published fan's z-fold
    const float phi = apply_head_tilt_
      ? tilt_at(rclcpp::Time(ping.stamp).nanoseconds()) : 0.0f;
    if (skip_ > 0 && ping_id % skip_ != 0) {
      Matrix nan(1, 2);
      nan.setConstant(std::numeric_limits<float>::quiet_NaN());
      publish_features(ping.stamp, nan, phi);
      return;
    }

    // head-pitch frame gate (see constructor): swept frames are floor- or
    // surface-dominated — poison for the planar registration and the tiles
    if (max_head_pitch_ > 0.0 &&
        std::fabs(phi) > static_cast<float>(max_head_pitch_)) {
      Matrix nan(1, 2);
      nan.setConstant(std::numeric_limits<float>::quiet_NaN());
      publish_features(ping.stamp, nan, phi);
      return;
    }

    // a malformed adapter frame (empty / zero-range image) must not throw out
    // of the callback and terminate the node
    if (ping.image.empty() || ping.num_ranges <= 0) {
      RCLCPP_WARN(get_logger(),
                  "Dropping sonar ping: empty or zero-range image");
      return;
    }
    // Geometry choke point for EVERY sonar adapter: the map builder divides by
    // range_resolution and dereferences bearings.front()/back() (UB on an
    // empty vector — not an exception, so the try/catch below cannot contain
    // it), and a bearing count that disagrees with the beam columns would
    // silently distort every feature.
    if (!(ping.range_resolution > 0.0) ||
        !std::isfinite(ping.range_resolution) ||
        !std::isfinite(ping.range_min) || ping.range_min < 0.0 ||
        static_cast<int>(ping.bearings.size()) != ping.image.cols) {
      RCLCPP_WARN(get_logger(),
                  "Dropping sonar ping: inconsistent geometry (res %.6g m, "
                  "range_min %.6g m, %zu bearings for %d beam columns)",
                  ping.range_resolution, ping.range_min, ping.bearings.size(),
                  ping.image.cols);
      return;
    }

    // isolate the OpenCV / detection pipeline so a bad frame drops rather than
    // crashing the node
    try {
      const cv::Mat& img = ping.image;
      generate_map_xy(ping);

      // CFAR detection with the intensity gate folded into the detector, so
      // the CPU/GPU twins apply it identically and no separate host-side
      // compare + bitwise-and pass over the whole image is needed
      cv::Mat peaks = detector_->detect(img, alg_, static_cast<float>(threshold_));

      // visualization image (Cartesian, JET colormap like cv2.applyColorMap);
      // same GPU remap path as the mask, skipped when nobody is subscribed
      if (feature_img_pub_->get_subscription_count() > 0) {
        cv::Mat vis = remap_u8(img, /*interp=*/1);  // bilinear for grayscale
        cv::applyColorMap(vis, vis, cv::COLORMAP_JET);
        std_msgs::msg::Header header;
        header.stamp = ping.stamp;
        header.frame_id = "base_link";
        feature_img_pub_->publish(
          *cv_bridge::CvImage(header, "bgr8", vis).toImageMsg());
      }

      // to Cartesian — nearest-neighbour for the binary mask
      const cv::Mat cart_peaks = remap_u8(peaks, /*interp=*/0);

      // grayscale echo image for the map stream's per-point intensity
      // (looked up at publish time, after downsample/outlier filtering)
      if (map_points_pub_)
        cart_gray_ = remap_u8(img, /*interp=*/1);

      std::vector<cv::Point> locs;
      cv::findNonZero(cart_peaks, locs);

      // image coordinates -> meters (feature_extraction.py lines 254-258)
      Matrix points(static_cast<long>(locs.size()), 2);
      for (std::size_t i = 0; i < locs.size(); ++i) {
        double x = locs[i].x - cols_ / 2.0;
        x = -1.0 * ((x / (cols_ / 2.0)) * (width_ / 2.0));
        const double y =
          -1.0 * (locs[i].y / static_cast<double>(rows_)) * height_ + height_;
        points(static_cast<long>(i), 0) = static_cast<float>(y);
        points(static_cast<long>(i), 1) = static_cast<float>(x);
      }

      if (points.rows() > 0 && resolution_ > 0)
        points = downsample(points, static_cast<float>(resolution_));

      if (outlier_filter_min_points_ > 1 && points.rows() > 0)
        points = remove_outlier(points, outlier_filter_radius_,
                                outlier_filter_min_points_);

      publish_features(ping.stamp, points, phi);
    } catch (const cv::Exception& e) {
      RCLCPP_WARN(get_logger(), "Dropping sonar ping: OpenCV error: %s", e.what());
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Dropping sonar ping: %s", e.what());
    }
  }

  std::unique_ptr<CFAR> detector_;
  CFAR::Alg alg_ = CFAR::SOCA;
  int threshold_ = 0;
  int ntc_ = 40, ngc_ = 10, rank_ = -1;
  double pfa_ = 0.1;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  double resolution_ = 0.5;
  double outlier_filter_radius_ = 1.0;
  int outlier_filter_min_points_ = 5;
  int skip_ = 5;
  bool compressed_images_ = true;
  int frame_count_ = 0;

  // remap state
  double res_ = 0.0, range_min_ = -1.0, height_ = 0.0, width_ = 0.0;
  int rows_ = 0, cols_ = 0;
  std::vector<float> bearings_;
  int map_version_ = 0;
  cv::Mat map_x_, map_y_;
  // per-ping remapped grayscale for map-stream intensity lookup
  cv::Mat cart_gray_;

  rclcpp::SubscriptionBase::SharedPtr sonar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr feature_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub_;
  double tvg_spread_db_ = 0.0;
  double tvg_absorption_db_per_m_ = 0.0;

  // sonar head pitch samples (arrival ns, rad) from /base/sensor_tilt_angle,
  // interpolated at each ping's stamp by tilt_at(); empty = level (0)
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr tilt_sub_;
  std::mutex tilt_mutex_;
  std::deque<std::pair<int64_t, float>> tilt_buf_;
  static constexpr std::size_t kTiltBufMax = 1024;
  bool apply_head_tilt_ = true;
  // skip feature frames when |head pitch| exceeds this (rad); 0 = no gate
  double max_head_pitch_ = 0.0;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::FeatureExtractionNode>());
  rclcpp::shutdown();
  return 0;
}
