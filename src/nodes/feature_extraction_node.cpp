// Feature extraction node: port of bruce_slam feature_extraction.py.
// CFAR target detection on the polar sonar image (GPU with CPU fallback),
// polar -> Cartesian remap, voxel downsampling and radius outlier removal,
// publishing a PointCloud2 stamped with the source ping time.
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <functional>
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
#include "sonar_slam_cpp/sonar_projection.hpp"

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
    // (both launch files put this node at /bruce/slam/feature_extraction;
    // run it bare with `ros2 run` and it is just /feature_extraction_node)
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
        double min_range = feature_min_range_, max_range = feature_max_range_;
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
            } else if (n == "filter.min_range") {
              min_range = as_num(p);
            } else if (n == "filter.max_range") {
              max_range = as_num(p);
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
          feature_min_range_ = min_range;
          feature_max_range_ = max_range;
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
    extract_polar_ = get_bool("filter/extract_polar", true);
    feature_min_range_ = get_double("filter/min_range", 0.0);
    feature_max_range_ = get_double("filter/max_range", 0.0);

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
    const int input_queue_depth =
      std::max(1, get_int("input_queue_depth", 20));
    const int output_queue_depth =
      std::max(1, get_int("output_queue_depth", 10));
    const auto input_qos = rclcpp::SensorDataQoS(
      rclcpp::KeepLast(static_cast<std::size_t>(input_queue_depth)));
    const auto output_qos = rclcpp::QoS(
      rclcpp::KeepLast(static_cast<std::size_t>(output_queue_depth)));

    // SonarPing's planar coordinates follow the optical convention regardless
    // of the transport header: optical +z is the boresight and +y is lateral.
    // Legacy ProjectedSonarImage bags have an empty header, while the current
    // driver may identify sensor_frame even though the samples still use
    // optical axes, so the source frame must be explicit.
    sonar_frame_id_ = get_string("sonar/frame_id", "sonar0/optical_frame");
    base_frame_id_ = get_string("sonar/base_frame_id", "base_link");
    projection_tf_timeout_ =
      get_double("sonar/projection_tf_timeout", 0.05);
    projection_tf_max_delta_ =
      get_double("sonar/projection_tf_max_delta", 0.02);
    if (sonar_frame_id_.empty() || base_frame_id_.empty())
      throw std::runtime_error(
        "sonar/frame_id and sonar/base_frame_id must not be empty");
    if (projection_tf_timeout_ < 0.0 || projection_tf_max_delta_ < 0.0)
      throw std::runtime_error(
        "sonar projection TF timing parameters must be non-negative");
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ =
      std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this);
    RCLCPP_INFO(
      get_logger(),
      "Sonar projection: %s <- %s at ping stamp (wait %.0f ms, latest-TF "
      "fallback <= %.0f ms)",
      base_frame_id_.c_str(), sonar_frame_id_.c_str(),
      projection_tf_timeout_ * 1000.0, projection_tf_max_delta_ * 1000.0);

    sonar_sub_ = subscribe_sonar(
      this, driver, topic, input_qos,
      [this](const SonarPing& ping) { callback(ping); });

    feature_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SONAR_FEATURE_TOPIC, output_qos);
    slam_feature_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SONAR_SLAM_FEATURE_TOPIC, output_qos);
    feature_img_pub_ = create_publisher<sensor_msgs::msg::Image>(
      SONAR_FEATURE_IMG_TOPIC, output_qos);

    // Optional union map stream: republish the (non-sentinel) feature clouds
    // as padded XYZI onto a shared topic with sonar_proc's candidate stream,
    // so the map accumulator/vdb see the CFAR detector's structure too —
    // CFAR holds a constant false-alarm rate against the local background
    // and finds low-contrast structure the candidate edge detector misses.
    const std::string map_topic = get_string("map_points_topic", "");
    if (!map_topic.empty())
      map_points_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, output_qos);
    // residual TVG for the map stream (identity at 0) — keep these equal to
    // sonar_proc's values so the union's two intensity sources stay on one
    // radiometric scale
    tvg_spread_db_ = get_double("tvg_spread_db", 0.0);
    tvg_absorption_db_per_m_ = get_double("tvg_absorption_db_per_m", 0.0);

    // Sonar head pitch. The Oculus rides the Deep Trekker's pivoting head
    // (cameraHead.tilt), which sweeps up to +/-54 deg. The full timestamped TF
    // above now owns the geometric projection (rotation AND lever arm); this
    // angle stream remains the admission signal for planar SLAM because it is
    // the platform's authoritative head-angle telemetry.
    // Head-pitch FRAME GATE (2026-07-16, from CHL_Pool ping forensics):
    // bruce-slam's planar registration assumes a near-level fan. During the
    // ±54° head sweep the frame is DOMINATED by the floor bowl — a huge
    // bright curved band whose planar (x,y) projection registers against
    // real walls from other frames and pulls the graph (the observed "bad
    // SLAM corrections"), and whose features stamp curved bands into the
    // mapping tiles. Evaluate the sonar head relative to base_link: vehicle
    // attitude is already represented by the robot pose and must not turn a
    // normal pitched operating pose into a rejected scan. The configured
    // bound must sit above the platform's normal head tilt but below its
    // extreme sweep. The real 3D cloud/image always publish for operators;
    // only SONAR_SLAM_FEATURE_TOPIC and the CFAR map contribution are gated.
    // The NaN sentinel keeps SLAM/mapping synchronizers advancing.
    max_head_pitch_ = get_double("max_head_pitch", 0.0);

    apply_head_tilt_ = get_bool("apply_head_tilt", true);
    if (apply_head_tilt_) {
      const std::string tilt_topic =
        get_string("head_tilt_topic", "/base/sensor_tilt_angle");
      tilt_sub_ = create_subscription<std_msgs::msg::Float32>(
        tilt_topic, input_qos,
        [this](const std_msgs::msg::Float32& m) {
          // Compatibility fallback for platforms that only publish the
          // legacy unstamped degrees topic. Once a source-stamped joint sample
          // arrives, these callback-arrival timestamps are ignored.
          const int64_t t = this->now().nanoseconds();
          const float rad = m.data * static_cast<float>(M_PI) / 180.0f;
          record_tilt(t, rad, false);
        });
      const std::string stamped_topic =
        get_string("head_tilt_stamped_topic", "");
      if (!stamped_topic.empty()) {
        tilt_stamped_sub_ = create_subscription<sensor_msgs::msg::JointState>(
          stamped_topic, input_qos,
          [this](const sensor_msgs::msg::JointState& m) {
            const auto it = std::find(
              m.name.begin(), m.name.end(), "pivot_head_joint");
            if (it == m.name.end()) return;
            const auto idx = static_cast<std::size_t>(
              std::distance(m.name.begin(), it));
            if (idx >= m.position.size() || !std::isfinite(m.position[idx]))
              return;
            record_tilt(
              rclcpp::Time(m.header.stamp).nanoseconds(),
              static_cast<float>(m.position[idx]), true);
          });
      }
    }

    RCLCPP_INFO(get_logger(), "Feature extraction node initialized (GPU: %s)",
                gpu::available() ? "on" : "off, CPU fallback");
  }

private:
  void record_tilt(const int64_t t_ns, const float rad, const bool source_stamped)
  {
    if (t_ns <= 0 || !std::isfinite(rad)) return;
    std::lock_guard<std::mutex> lock(tilt_mutex_);
    if (!source_stamped && have_stamped_tilt_) return;
    if (source_stamped && !have_stamped_tilt_) {
      // Remove arrival-stamped compatibility samples so interpolation never
      // mixes two time bases.
      tilt_buf_.clear();
      have_stamped_tilt_ = true;
    }
    if (!tilt_buf_.empty() && t_ns < tilt_buf_.back().first)
      tilt_buf_.clear();  // time jumped backwards (bag seek/loop)
    tilt_buf_.emplace_back(t_ns, rad);
    while (tilt_buf_.size() > kTiltBufMax) tilt_buf_.pop_front();
  }

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
  // Nearest beam column to `bearing`. bearings_ is strictly monotonic — the
  // adapters verify that on the first ping — so a binary search is valid;
  // pick whichever neighbour is closer. Returns 0 for an empty table.
  int beam_index(double bearing) const
  {
    if (bearings_.empty()) return 0;
    const bool ascending = bearings_.back() >= bearings_.front();
    std::size_t lo;
    if (ascending) {
      lo = static_cast<std::size_t>(
        std::lower_bound(bearings_.begin(), bearings_.end(),
                         static_cast<float>(bearing)) - bearings_.begin());
    } else {
      lo = static_cast<std::size_t>(
        std::lower_bound(bearings_.begin(), bearings_.end(),
                         static_cast<float>(bearing),
                         std::greater<float>()) - bearings_.begin());
    }
    if (lo == 0) return 0;
    if (lo >= bearings_.size()) return static_cast<int>(bearings_.size()) - 1;
    const double d_hi = std::abs(bearings_[lo] - bearing);
    const double d_lo = std::abs(bearings_[lo - 1] - bearing);
    return static_cast<int>(d_lo <= d_hi ? lo - 1 : lo);
  }

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

    // Per-beam cos/sin, tabulated once per geometry instead of once per
    // detection. The polar extractor below runs cos+sin for every CFAR hit —
    // thousands per ping at a loose operating point — and the bearing only
    // ever takes these num_beams values. Same double arguments, so the
    // published coordinates are unchanged.
    std::vector<double> cos_b(ping.bearings.size()), sin_b(ping.bearings.size());
    for (std::size_t i = 0; i < ping.bearings.size(); ++i) {
      cos_b[i] = std::cos(static_cast<double>(ping.bearings[i]));
      sin_b[i] = std::sin(static_cast<double>(ping.bearings[i]));
    }

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
    cos_bearings_ = std::move(cos_b);
    sin_bearings_ = std::move(sin_b);
    res_ = res;
    range_min_ = range_min;
    height_ = height;
    rows_ = rows;
    width_ = width;
    cols_ = cols;
    bearings_ = ping.bearings;
    // new maps -> new version so the GPU remap re-uploads its cached copy
    ++map_version_;

    // Report the blanking the CFAR window imposes for THIS geometry. It is a
    // side effect of the sliding window (detect_cpu skips the first and last
    // border rows), not a deliberate exclusion, and it moves whenever Ntc/Ngc
    // change — so an operator setting filter/min_range against wake or
    // filter/max_range against multipath needs to see where it already sits,
    // and that a configured max_range above the reported usable range does
    // nothing.
    const int border_bins = ntc_ / 2 + ngc_ / 2;
    const double blank = border_bins * res;
    const double effective_min = effectiveCfarMinRange(
      feature_min_range_, range_min, rows, res, border_bins);
    // bind the temporary: a `(... + " m").c_str()` inline in a logging macro
    // hands the sink a pointer into a string that may already be gone
    const std::string max_txt = feature_max_range_ > 0.0
                                  ? std::to_string(feature_max_range_) + " m"
                                  : std::string("off");
    RCLCPP_INFO(get_logger(),
                "sonar geometry: %d bins x %zu beams @ %.3f m; CFAR window "
                "blanks the inner %.2f m and the outer %.2f m (usable %.2f-"
                "%.2f m). filter/min_range %.2f m (effective %.2f m), "
                "max_range %s",
                ping.num_ranges, ping.bearings.size(), res, blank, blank,
                range_min + blank, range_min + rows * res - blank,
                feature_min_range_, effective_min, max_txt.c_str());
  }

  void publish_features(const builtin_interfaces::msg::Time& stamp,
                        const Matrix& points, bool slam_eligible)
  {
    // A NaN row is the intentional skip-frame synchronization sentinel. It
    // has no geometry to transform. Empty feature sets likewise need no TF.
    const bool input_sentinel =
      points.rows() == 1 && !std::isfinite(points(0, 0));
    Matrix xyz;
    if (input_sentinel) {
      xyz.resize(1, 3);
      xyz.setConstant(std::numeric_limits<float>::quiet_NaN());
      slam_eligible = false;
    } else if (points.rows() == 0) {
      xyz.resize(0, 3);
    } else {
      try {
        // Exact ping-time projection, including the sonar mount/head lever
        // arm. `points` is [forward, lateral], or optical [0, lateral,
        // forward]; projectSonarPlane applies target <- optical.
        geometry_msgs::msg::TransformStamped tf;
        try {
          tf = tf_buffer_->lookupTransform(
            base_frame_id_, sonar_frame_id_, rclcpp::Time(stamp),
            rclcpp::Duration::from_seconds(projection_tf_timeout_));
        } catch (const std::exception& exact_error) {
          // During bag replay, /clock and the sonar callback can lead
          // robot_state_publisher's joint TF by a few milliseconds. Waiting
          // on simulated time does not reliably close that scheduling gap.
          // Use the newest transform only when its stamp proves that the
          // approximation is tightly bounded; stale startup TF still fails.
          auto latest = tf_buffer_->lookupTransform(
            base_frame_id_, sonar_frame_id_, tf2::TimePointZero);
          const rclcpp::Time latest_stamp(latest.header.stamp);
          const bool timeless =
            latest_stamp.nanoseconds() == 0;  // an entirely static TF chain
          const double delta =
            std::fabs((rclcpp::Time(stamp) - latest_stamp).seconds());
          if (!timeless && delta > projection_tf_max_delta_) {
            throw std::runtime_error(
              std::string(exact_error.what()) +
              "; latest TF differs by " + std::to_string(delta * 1000.0) +
              " ms (limit " +
              std::to_string(projection_tf_max_delta_ * 1000.0) + " ms)");
          }
          tf = std::move(latest);
          RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 10000,
            "Using latest %s <- %s TF for sonar projection (stamp skew %.3f "
            "ms)",
            base_frame_id_.c_str(), sonar_frame_id_.c_str(), delta * 1000.0);
        }
        const auto& r = tf.transform.rotation;
        Eigen::Quaternionf q(
          static_cast<float>(r.w), static_cast<float>(r.x),
          static_cast<float>(r.y), static_cast<float>(r.z));
        const float q_norm = q.norm();
        const auto& t = tf.transform.translation;
        const Eigen::Vector3f translation(
          static_cast<float>(t.x), static_cast<float>(t.y),
          static_cast<float>(t.z));
        if (!std::isfinite(q_norm) ||
            q_norm <= std::numeric_limits<float>::epsilon() ||
            !translation.allFinite())
          throw std::runtime_error("TF contains a non-finite transform");
        xyz = projectSonarPlane(
          points, q.normalized().toRotationMatrix(), translation);
      } catch (const std::exception& e) {
        // Never label optical/sonar-local coordinates as base_link. A
        // sentinel keeps downstream exact-time synchronizers advancing while
        // making the missing geometry explicit to every cloud consumer.
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Cannot project sonar features at %.9f s (%s <- %s: %s); "
          "publishing a NaN sentinel instead of mislabeled geometry",
          to_sec(stamp), base_frame_id_.c_str(), sonar_frame_id_.c_str(),
          e.what());
        xyz.resize(1, 3);
        xyz.setConstant(std::numeric_limits<float>::quiet_NaN());
        slam_eligible = false;
      }
    }

    sensor_msgs::msg::PointCloud2 msg = make_cloud_xyz(xyz);
    // the stamp of the source sonar image is CRITICAL to downstream sync
    msg.header.stamp = stamp;
    msg.header.frame_id = base_frame_id_;
    feature_pub_->publish(msg);

    if (slam_eligible) {
      slam_feature_pub_->publish(msg);
    } else {
      Matrix sentinel_xyz(1, 3);
      sentinel_xyz.setConstant(std::numeric_limits<float>::quiet_NaN());
      auto sentinel = make_cloud_xyz(sentinel_xyz);
      sentinel.header = msg.header;
      slam_feature_pub_->publish(sentinel);
    }

    // union map stream (see constructor): skip the NaN skip-frame sentinel
    // (a sync signal for the slam node, poison for map consumers), gated
    // floor-dominated frames, and empty frames (they only flood the
    // chain/assembler with no-data warnings)
    if (map_points_pub_ && slam_eligible && xyz.rows() > 0 &&
        !std::isnan(xyz(0, 0))) {
      // real echo intensity (0..1): invert the extraction pixel->meters
      // mapping back into this ping's remapped grayscale. points col0 =
      // forward (rows axis), col1 = lateral (cols axis), pre-tilt — their
      // planar norm is the slant range.
      Eigen::VectorXf intens = Eigen::VectorXf::Ones(points.rows());
      Eigen::VectorXf ranges = Eigen::VectorXf::Zero(points.rows());
      for (long r = 0; r < points.rows(); ++r)
        ranges(r) = std::hypot(points(r, 0), points(r, 1));
      // In polar-extraction mode invert straight back to (range, bearing) and
      // read the ORIGINAL polar echo — no resampling on either leg. The
      // Cartesian branch keeps the legacy grayscale-remap lookup.
      const bool polar_lookup = !polar_gray_.empty() && res_ > 0.0 &&
                                !bearings_.empty();
      if (polar_lookup || (!cart_gray_.empty() && rows_ > 0 && height_ > 0 &&
                           width_ > 0)) {
        for (long r = 0; r < points.rows(); ++r) {
          int row, col;
          if (polar_lookup) {
            // lateral axis is negated on the way out (see the extractor)
            const double bearing = std::atan2(-points(r, 1), points(r, 0));
            row = static_cast<int>(std::lround((ranges(r) - range_min_) / res_));
            col = beam_index(bearing);
          } else {
            row = static_cast<int>(
              std::lround((1.0 - points(r, 0) / height_) * rows_));
            col = static_cast<int>(
              std::lround(cols_ / 2.0 - points(r, 1) * cols_ / width_));
          }
          const cv::Mat& gray = polar_lookup ? polar_gray_ : cart_gray_;
          if (row >= 0 && row < gray.rows && col >= 0 && col < gray.cols)
            intens(r) = gray.at<std::uint8_t>(row, col) / 255.0f;
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
    // Head pitch AT THE PING STAMP (interpolated) determines admission to the
    // planar consumers. The full timestamped TF performs the real geometric
    // projection below.
    const int64_t ping_stamp_ns = rclcpp::Time(ping.stamp).nanoseconds();
    const float phi = apply_head_tilt_
      ? tilt_at(ping_stamp_ns) : 0.0f;
    if (skip_ > 0 && ping_id % skip_ != 0) {
      Matrix nan(1, 2);
      nan.setConstant(std::numeric_limits<float>::quiet_NaN());
      publish_features(ping.stamp, nan, false);
      return;
    }

    const bool slam_eligible =
      max_head_pitch_ <= 0.0 ||
      std::fabs(phi) <= static_cast<float>(max_head_pitch_);
    if (!slam_eligible && !gate_active_) {
      RCLCPP_INFO(
        get_logger(),
        "Planar-SLAM feature gate active: head pitch %.1f deg exceeds "
        "%.1f deg (3D feature visualization remains live)",
        phi * 180.0 / M_PI, max_head_pitch_ * 180.0 / M_PI);
    } else if (slam_eligible && gate_active_) {
      RCLCPP_INFO(
        get_logger(),
        "Planar-SLAM feature gate cleared: head pitch %.1f deg is within "
        "%.1f deg",
        phi * 180.0 / M_PI, max_head_pitch_ * 180.0 / M_PI);
    }
    gate_active_ = !slam_eligible;

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

      Matrix points;
      if (extract_polar_ && !bearings_.empty() && res_ > 0.0) {
        // Extract in POLAR and convert exactly (docs/SONAR_FRONTEND_REVIEW.md
        // §5). The legacy path below remaps the binary mask to Cartesian FIRST
        // and reads pixel indices out of it, which makes a detection survive
        // only if some destination pixel happens to sample its polar cell.
        // Inside the range where beam arc spacing is finer than the Cartesian
        // cell — 6.8 m for an Oculus at 0.03 m, 9.4 m for the Revolution
        // preset — cells compete for pixels and lose: 68.7% and 73.9% of
        // near-field cells respectively are unreachable, and the survivors
        // come out snapped to the Cartesian grid. Reading the polar mask
        // directly has neither problem, and is CHEAPER (no mask remap at all;
        // the Cartesian remap survives only for the feature_img
        // visualization, which is already subscription-gated).
        //
        // Convention matches generate_map_xy exactly so the cloud keeps its
        // orientation: that map places Cartesian (r,c) at forward
        // res*(rows-r), lateral res*(-cols/2+c+0.5), and reads polar row
        // (range-range_min)/res at bearing atan2(lateral, forward). Inverting
        // it, polar cell (row, beam) is at range = range_min + row*res on
        // bearing bearings_[beam] — and the legacy conversion negates the
        // lateral axis, which is preserved here.
        std::vector<cv::Point> locs;  // x = beam column, y = range bin
        cv::findNonZero(peaks, locs);
        points.resize(static_cast<long>(locs.size()), 2);
        for (std::size_t i = 0; i < locs.size(); ++i) {
          const std::size_t beam = static_cast<std::size_t>(locs[i].x);
          const double range = range_min_ + locs[i].y * res_;
          points(static_cast<long>(i), 0) =
            static_cast<float>(range * cos_bearings_[beam]);
          points(static_cast<long>(i), 1) =
            static_cast<float>(-range * sin_bearings_[beam]);
        }
        // Polar echo image for the map stream's per-point intensity; the
        // lookup inverts to (range, bearing) rather than to a Cartesian pixel.
        // A cv::Mat assignment shares the pixel buffer and bumps its refcount
        // instead of copying it — every sonar adapter hands us a freshly
        // decoded image per ping, so nothing overwrites it underneath us, and
        // the consumer (publish_features) runs later in this same callback.
        if (map_points_pub_) polar_gray_ = img;
      } else {
        // Legacy path: remap the mask to Cartesian, then read pixel indices.
        // Kept behind filter/extract_polar for a one-line revert if a bag
        // replay disagrees with the change above.
        const cv::Mat cart_peaks = remap_u8(peaks, /*interp=*/0);

        // grayscale echo image for the map stream's per-point intensity
        // (looked up at publish time, after downsample/outlier filtering)
        if (map_points_pub_) cart_gray_ = remap_u8(img, /*interp=*/1);

        std::vector<cv::Point> locs;
        cv::findNonZero(cart_peaks, locs);

        // image coordinates -> meters (feature_extraction.py lines 254-258)
        points.resize(static_cast<long>(locs.size()), 2);
        for (std::size_t i = 0; i < locs.size(); ++i) {
          double x = locs[i].x - cols_ / 2.0;
          x = -1.0 * ((x / (cols_ / 2.0)) * (width_ / 2.0));
          const double y =
            -1.0 * (locs[i].y / static_cast<double>(rows_)) * height_ + height_;
          points(static_cast<long>(i), 0) = static_cast<float>(y);
          points(static_cast<long>(i), 1) = static_cast<float>(x);
        }
      }

      // Explicit range gate, in METRES, applied to whichever extractor ran.
      //
      // The CFAR window already blanks the innermost and outermost `Ntc/2 +
      // Ngc/2` range bins (detect_cpu iterates [border, rows-border)) — 2.08 m
      // for the Revolution preset, 0.75 m for an Oculus at 0.03 m — but that
      // is an ACCIDENT of window size, not a statement about the platform, and
      // it moves whenever Ntc/Ngc are tuned. These knobs make the exclusion
      // deliberate and independent:
      //   min_range — thruster wake and bubble clouds, transducer ringdown and
      //     near-field saturation, and the vehicle's own frame/tether in the
      //     beam. All produce strong, geometrically real-looking returns that
      //     are body-fixed rather than world-fixed, so they survive CFAR (they
      //     ARE bright against their surroundings) and then drag ICP toward
      //     the vehicle's own motion.
      //   max_range — in confined water this is a multipath filter with a hard
      //     physical justification: a surface or bottom bounce arrives at a
      //     LONGER path length than the direct return, so in a pool of known
      //     size any echo beyond the maximum direct-path dimension cannot be
      //     structure and is necessarily a ghost.
      // Both default to 0 (off), so the shipped behaviour is unchanged and
      // this stays an explicit operator decision made against real imagery.
      const double effective_min_range = effectiveCfarMinRange(
        feature_min_range_, range_min_, rows_, res_,
        ntc_ / 2 + ngc_ / 2);
      if (effective_min_range < feature_min_range_) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 30000,
          "filter/min_range %.2f m lies beyond this sonar's entire "
          "CFAR-usable interval; using its inner usable boundary %.2f m for "
          "this geometry instead of publishing an always-empty cloud",
          feature_min_range_, effective_min_range);
      }
      if (points.rows() > 0 &&
          (effective_min_range > 0.0 || feature_max_range_ > 0.0)) {
        long keep = 0;
        for (long i = 0; i < points.rows(); ++i) {
          const double rr = std::hypot(points(i, 0), points(i, 1));
          if (rr < effective_min_range) continue;
          if (feature_max_range_ > 0.0 && rr > feature_max_range_) continue;
          points.row(keep++) = points.row(i);
        }
        points.conservativeResize(keep, Eigen::NoChange);
      }

      if (points.rows() > 0 && resolution_ > 0)
        points = downsample(points, static_cast<float>(resolution_));

      if (outlier_filter_min_points_ > 1 && points.rows() > 0)
        points = remove_outlier(points, outlier_filter_radius_,
                                outlier_filter_min_points_);

      publish_features(ping.stamp, points, slam_eligible);
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

  // Exact point projection: base_frame_id_ <- sonar_frame_id_ at each ping
  // stamp. The source is explicit because legacy bags have blank headers and
  // some drivers describe optical-axis samples with a sensor-frame header.
  std::string sonar_frame_id_ = "sonar0/optical_frame";
  std::string base_frame_id_ = "base_link";
  double projection_tf_timeout_ = 0.05;
  double projection_tf_max_delta_ = 0.02;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  // remap state
  double res_ = 0.0, range_min_ = -1.0, height_ = 0.0, width_ = 0.0;
  int rows_ = 0, cols_ = 0;
  std::vector<float> bearings_;
  // per-beam cos/sin of bearings_, rebuilt with the remap tables
  std::vector<double> cos_bearings_, sin_bearings_;
  int map_version_ = 0;
  cv::Mat map_x_, map_y_;
  // per-ping grayscale for map-stream intensity lookup: the polar original
  // when extracting in polar, the Cartesian remap on the legacy path
  cv::Mat cart_gray_;
  cv::Mat polar_gray_;
  // extract detections from the polar mask (exact) rather than from the
  // Cartesian remap of it (lossy in the near field) — see the extractor
  bool extract_polar_ = true;
  // explicit range gate (m; 0 = off) for wake/ringdown and multipath ghosts
  double feature_min_range_ = 0.0;
  double feature_max_range_ = 0.0;

  rclcpp::SubscriptionBase::SharedPtr sonar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr feature_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr slam_feature_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub_;
  double tvg_spread_db_ = 0.0;
  double tvg_absorption_db_per_m_ = 0.0;

  // Sonar head pitch samples (source stamp, rad) from the stamped joint-state
  // topic, with the legacy arrival-stamped Float32 as a compatibility fallback.
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr tilt_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
    tilt_stamped_sub_;
  std::mutex tilt_mutex_;
  std::deque<std::pair<int64_t, float>> tilt_buf_;
  static constexpr std::size_t kTiltBufMax = 1024;
  bool have_stamped_tilt_ = false;
  bool apply_head_tilt_ = true;
  // Skip planar-consumer frames when |head pitch relative to base_link|
  // exceeds this value (rad); 0 = no gate.
  double max_head_pitch_ = 0.0;
  bool gate_active_ = false;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  return sonar_slam::run_node<sonar_slam::FeatureExtractionNode>(argc, argv);
}
