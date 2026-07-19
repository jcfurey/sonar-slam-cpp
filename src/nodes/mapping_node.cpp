// Mapping node: port of bruce_slam mapping_node.py. Builds keyframe-anchored,
// loop-closure-correctable map products from the SLAM trajectory + the sonar
// ping + the feature cloud:
//   * an additive-logodds OCCUPANCY grid (from the feature point cloud), and
//   * an INTENSITY / backscatter mosaic (from the ping image),
// both published as nav_msgs/OccupancyGrid in the map frame.
//
// The SLAM node already publishes the whole optimized trajectory (latched,
// re-published every keyframe) on SLAM_TRAJ_TOPIC, so this node needs NO
// changes there: every callback it (re)places the newest keyframe's tile and
// corrects every keyframe whose pose the graph moved — the loop closure
// correction. Ping and feature clouds are buffered by stamp and matched to the
// trajectory's per-keyframe stamps (exact for the newest keyframe; nearest
// within a small tolerance for catch-up, since the trajectory carries the
// per-row stamp only as a float32 offset).
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <opencv2/imgcodecs.hpp>

#include <fstream>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

#include "sonar_slam_cpp/geo.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/mapping.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"

namespace sonar_slam {

namespace {

rclcpp::QoS latched_qos(int depth = 1)
{
  return rclcpp::QoS(depth).reliable().transient_local();
}

int64_t stamp_ns(const builtin_interfaces::msg::Time& t)
{
  return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
}

builtin_interfaces::msg::Time ns_stamp(int64_t ns)
{
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(ns / 1000000000LL);
  t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return t;
}

// drop non-finite (NaN skip-frame sentinel) rows from an Nx3 cloud
Matrix finite_rows(const Matrix& in)
{
  int n = 0;
  for (int i = 0; i < in.rows(); ++i)
    if (in.row(i).allFinite()) ++n;
  if (n == in.rows()) return in;  // all finite: skip the second pass + copy
  Matrix out(n, in.cols());
  int at = 0;
  for (int i = 0; i < in.rows(); ++i)
    if (in.row(i).allFinite()) out.row(at++) = in.row(i);
  return out;
}

}  // namespace

class MappingNode : public SlamNodeBase
{
public:
  MappingNode() : SlamNodeBase("mapping")
  {
    const auto origin = get_double_array("origin", {-50.0, -50.0});
    const auto size = get_double_array("size", {100.0, 100.0});
    map_.x0 = origin.size() > 0 ? origin[0] : -50.0;
    map_.y0 = origin.size() > 1 ? origin[1] : -50.0;
    map_.width = size.size() > 0 ? size[0] : 100.0;
    map_.height = size.size() > 1 ? size[1] : 100.0;
    map_.resolution = get_double("resolution", 0.2);
    map_.inc = get_double("inc", 50.0);

    map_.pub_occupancy1 = get_bool("pub_occupancy1", true);
    map_.hit_prob = get_double("hit_prob", 0.8);
    map_.miss_prob = get_double("miss_prob", 0.3);
    map_.inflation_angle = get_double("inflation_angle", 0.05);
    map_.inflation_range = get_double("inflation_range", 0.5);

    map_.pub_intensity = get_bool("pub_intensity", true);

    map_.outlier_filter_radius = get_double("outlier_filter_radius", 5.0);
    map_.outlier_filter_min_points = get_int("outlier_filter_min_points", 20);
    // below this many feature points a keyframe deposits a NEUTRAL occupancy
    // tile instead of the all-free wedge (0 = upstream policy)
    map_.free_tile_min_points = get_int("free_tile_min_points", 0);

    map_.min_translation = get_double("min_translation", 0.5);
    map_.min_rotation = get_double("min_rotation", 0.05);

    // Grace window before a keyframe's inputs are declared conclusively lost:
    // the pairing tolerance (2 ms) is far too tight for that verdict — a
    // best-effort message delivered milliseconds late would be irreversibly
    // skipped the moment any newer stamp arrived. Wait until the stream has
    // advanced this far past the keyframe's stamp before giving up on it.
    stream_grace_ns_ = static_cast<int64_t>(
      get_double("stream_grace", 1.0) * 1e9);

    // must match slam_node's enu_world so the reconstructed sonar fan lands in
    // the same frame chirality as the trajectory poses
    const bool enu = get_bool("enu_world", false);
    map_.frame_y_sign = enu ? 1.0 : -1.0;

    map_.configure();

    publish_period_ = get_double("publish_period", 0.0);

    // sonar driver for the raw ping image (same as feature_extraction)
    const std::string sonar_driver = get_string("sonar/driver", "projected_sonar");
    std::string sonar_topic = get_string("sonar/topic", "");
    if (sonar_topic.empty()) sonar_topic = default_sonar_topic(sonar_driver);
    sonar_sub_ = subscribe_sonar(
      this, sonar_driver, sonar_topic, rclcpp::SensorDataQoS(),
      [this](const SonarPing& ping) { on_ping(ping); });

    feature_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      SONAR_FEATURE_TOPIC, 20,
      [this](const sensor_msgs::msg::PointCloud2& msg) { on_feature(msg); });

    // latched, re-published every keyframe -> our correction trigger
    traj_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      SLAM_TRAJ_TOPIC, latched_qos(10),
      [this](const sensor_msgs::msg::PointCloud2& msg) { on_traj(msg); });

    // Georeferenced deliverables: with a survey datum ([lat deg, lon deg,
    // bearing deg of the map +x axis, i.e. the vehicle's initial heading]),
    // ~/export_map writes the occupancy + intensity grids as PNG with UTM
    // world files (.pgw + .prj — QGIS/ArcGIS open them as georeferenced
    // rasters) and the trajectory as a WGS84 GeoJSON LineString.
    const auto datum = get_double_array("datum", {});
    if (datum.size() == 3) {
      datum_ = GeoDatum(datum[0], datum[1], datum[2]);
      has_datum_ = true;
      RCLCPP_INFO(get_logger(),
                  "datum set: %.6f, %.6f, map +x bearing %.1f deg (UTM zone "
                  "%d%c)",
                  datum[0], datum[1], datum[2], datum_.origin.zone,
                  datum_.origin.north ? 'N' : 'S');
    } else if (!datum.empty()) {
      RCLCPP_ERROR(get_logger(),
                   "datum must be [lat_deg, lon_deg, map_x_bearing_deg] — "
                   "georeferenced export disabled");
    }
    export_prefix_ = get_string("export_prefix", "/tmp/sonar_slam_survey");
    export_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/export_map",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        export_map(*res);
      });

    occupancy_pub_ =
      create_publisher<nav_msgs::msg::OccupancyGrid>(MAPPING_OCCUPANCY_TOPIC, latched_qos());
    intensity_pub_ =
      create_publisher<nav_msgs::msg::OccupancyGrid>(MAPPING_INTENSITY_TOPIC, latched_qos());

    RCLCPP_INFO(get_logger(),
                "Mapping node initialized (occupancy: %s, intensity: %s, "
                "resolution %.3f m, frame_y_sign %+.0f)",
                map_.pub_occupancy1 ? "on" : "off",
                map_.pub_intensity ? "on" : "off", map_.resolution,
                map_.frame_y_sign);
  }

private:
  struct KFPose
  {
    double x = 0, y = 0, yaw = 0;
    int64_t stamp_ns = 0;
    // pairing tolerance for this keyframe: the trajectory carries stamps as
    // float32 SECONDS offsets from the message stamp, whose absolute error
    // grows with the offset (~1.2e-7 relative) — past ~4 h of mission it
    // exceeds the fixed 2 ms tolerance and every older keyframe would
    // spuriously miss its ping/feature and be skipped
    int64_t tol_ns = kSyncTolNs;
  };

  void on_ping(const SonarPing& ping)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ping_buf_[stamp_ns(ping.stamp)] = ping;
    while (static_cast<int>(ping_buf_.size()) > kBufMax) ping_buf_.erase(ping_buf_.begin());
    if (try_build()) maybe_publish(last_stamp_ns_);
  }

  void on_feature(const sensor_msgs::msg::PointCloud2& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    feat_buf_[stamp_ns(msg.header.stamp)] = finite_rows(cloud_to_xyz(msg));
    while (static_cast<int>(feat_buf_.size()) > kBufMax) feat_buf_.erase(feat_buf_.begin());
    if (try_build()) maybe_publish(last_stamp_ns_);
  }

  void on_traj(const sensor_msgs::msg::PointCloud2& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    const int n = static_cast<int>(msg.width * std::max<uint32_t>(msg.height, 1));
    if (n == 0) return;

    // Validate all consumed fields exist before constructing iterators — a
    // PointCloud2ConstIterator throws std::runtime_error on a missing field,
    // which would escape this subscription callback and terminate the node
    // (e.g. a replayed/foreign traj cloud missing the C++-added `t` field).
    {
      auto has_field = [&msg](const char* name) {
        for (const auto& f : msg.fields)
          if (f.name == name) return true;
        return false;
      };
      if (!has_field("x") || !has_field("y") || !has_field("yaw") ||
          !has_field("t")) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Dropping trajectory: missing x/y/yaw/t field (slam node too old?)");
        return;
      }
    }

    const int64_t msg_ns = stamp_ns(msg.header.stamp);
    last_stamp_ns_ = msg_ns;

    traj_.assign(n, KFPose{});
    sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x"), iy(msg, "y"),
      iyaw(msg, "yaw"), it(msg, "t");
    for (int k = 0; k < n; ++k, ++ix, ++iy, ++iyaw, ++it) {
      traj_[k].x = *ix;
      traj_[k].y = *iy;
      traj_[k].yaw = *iyaw;
      const double off_s = static_cast<double>(*it);
      traj_[k].stamp_ns = msg_ns + llround(off_s * 1e9);
      // float32 quantization of the offset (relative eps ~1.19e-7), with a
      // small safety factor; never below the base 2 ms tolerance
      traj_[k].tol_ns = std::max<int64_t>(
        kSyncTolNs, llround(std::abs(off_s) * 1e9 * 2.4e-7));
    }

    // add the newest keyframe's tile, then correct every prior keyframe the
    // graph moved (mapping_node.py tpf_callback: add_keyframe(len-1) then
    // update_pose(x) for x in range(len-1))
    try_build();
    const int ncorr = std::min(n - 1, map_.num_keyframes());
    for (int k = 0; k < ncorr; ++k)
      map_.update_pose(k, gtsam::Pose2(traj_[k].x, traj_[k].y, traj_[k].yaw));

    maybe_publish(msg_ns);
  }

  // Build every trajectory keyframe whose ping + feature are now available.
  // A keyframe whose inputs are conclusively gone — the buffered stream has
  // already advanced past its stamp with no match (a dropped best-effort
  // message, or a restart against the latched trajectory carrying history we
  // never received) — is skipped with an empty tile: the build is strictly
  // in-order, so waiting for it would wedge every later keyframe forever.
  bool try_build()
  {
    bool built = false;
    while (map_.num_keyframes() < static_cast<int>(traj_.size())) {
      const int k = map_.num_keyframes();
      const int64_t s = traj_[k].stamp_ns;
      const int64_t tol = traj_[k].tol_ns;
      auto pit = find_near(ping_buf_, s, tol);
      auto fit = find_near(feat_buf_, s, tol);
      if (pit == ping_buf_.end() && !stream_passed(ping_buf_, s)) {
        // a dead/misrouted ping stream stalls the whole builder forever with
        // no output at all — say so instead of publishing an empty map quietly
        if (ping_buf_.empty())
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 30000,
                               "mapping: %d keyframe(s) waiting but no sonar "
                               "pings received — check sonar/driver + topic",
                               static_cast<int>(traj_.size()) - k);
        break;
      }
      if (fit == feat_buf_.end() && !stream_passed(feat_buf_, s)) {
        if (feat_buf_.empty())
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 30000,
                               "mapping: %d keyframe(s) waiting but no feature "
                               "clouds received — check the feature topic",
                               static_cast<int>(traj_.size()) - k);
        break;
      }
      if (pit == ping_buf_.end() || fit == feat_buf_.end()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 10000,
          "keyframe %d: %s stream passed its stamp with no match — "
          "skipping (empty tile)",
          k, pit == ping_buf_.end() ? "ping" : "feature");
        map_.add_skipped(k, gtsam::Pose2(traj_[k].x, traj_[k].y, traj_[k].yaw));
        continue;
      }
      map_.add_keyframe(k, gtsam::Pose2(traj_[k].x, traj_[k].y, traj_[k].yaw),
                        pit->second, fit->second);
      built = true;
    }
    return built;
  }

  // has the (in-order) buffered stream conclusively moved past stamp `key`?
  // Judged against the GRACE window, not the 2 ms pairing tolerance: a
  // milliseconds-late best-effort delivery must not be irreversibly skipped.
  template <class M>
  bool stream_passed(const M& buf, int64_t key) const
  {
    return !buf.empty() && buf.rbegin()->first > key + stream_grace_ns_;
  }

  template <class M>
  static typename M::iterator find_near(M& buf, int64_t key,
                                        int64_t tol = kSyncTolNs)
  {
    auto best = buf.end();
    int64_t bestd = tol + 1;
    auto it = buf.lower_bound(key);
    if (it != buf.end()) {
      const int64_t d = std::llabs(it->first - key);
      if (d <= tol && d < bestd) { best = it; bestd = d; }
    }
    if (it != buf.begin()) {
      auto p = std::prev(it);
      const int64_t d = std::llabs(p->first - key);
      if (d <= tol && d < bestd) { best = p; bestd = d; }
    }
    return best;
  }

  void maybe_publish(int64_t stamp)
  {
    if (publish_period_ > 0.0) {
      const double now_s = stamp * 1e-9;
      if (last_pub_s_ > 0.0 && now_s - last_pub_s_ < publish_period_) return;
      last_pub_s_ = now_s;
    }
    const auto st = ns_stamp(stamp);
    if (map_.pub_occupancy1) {
      const OccGrid g = map_.get_occupancy_grid();
      if (!g.empty()) occupancy_pub_->publish(to_occ_msg(g, st));
    }
    if (map_.pub_intensity) {
      const OccGrid g = map_.get_intensity_grid();
      if (!g.empty()) intensity_pub_->publish(to_occ_msg(g, st));
    }
  }

  // grid -> top-down PNG (map_saver shading) + UTM world file + .prj
  bool write_grid_geo(const OccGrid& g, const std::string& base,
                      std::string& out_err)
  {
    cv::Mat img(g.height, g.width, CV_8UC1);
    for (int r = 0; r < g.height; ++r)
      for (int c = 0; c < g.width; ++c) {
        const int8_t v = g.data[static_cast<std::size_t>(
          (g.height - 1 - r) * g.width + c)];
        img.at<std::uint8_t>(r, c) =
          v < 0 ? 205 : static_cast<std::uint8_t>(255 - v * 255 / 100);
      }
    if (!cv::imwrite(base + ".png", img)) {
      out_err = "imwrite failed for " + base + ".png";
      return false;
    }
    // world file: column step = map +x, image-row step = map -y (the PNG is
    // written top-down while the grid's row 0 is ymin)
    const double res = g.resolution;
    double C, F;
    datum_.map_to_utm(g.origin_x + 0.5 * res,
                      g.origin_y + (g.height - 0.5) * res, C, F);
    std::ofstream w(base + ".pgw", std::ios::trunc);
    w.precision(8);
    w << std::fixed << res * datum_.ex << "\n" << res * datum_.nx << "\n"
      << -res * datum_.ey << "\n" << -res * datum_.ny << "\n" << C << "\n"
      << F << "\n";
    std::ofstream prj(base + ".prj", std::ios::trunc);
    prj << utm_wkt(datum_.origin.zone, datum_.origin.north);
    if (!w || !prj) {
      out_err = "failed writing world/prj sidecars for " + base;
      return false;
    }
    return true;
  }

  void export_map(std_srvs::srv::Trigger::Response& res)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_datum_) {
      res.success = false;
      res.message =
        "no datum set — provide the 'datum' parameter [lat_deg, lon_deg, "
        "map_x_bearing_deg] to georeference the export";
      return;
    }
    std::string written, err;
    const OccGrid occ = map_.get_occupancy_grid();
    if (!occ.empty()) {
      if (!write_grid_geo(occ, export_prefix_ + "_occupancy", err)) {
        res.success = false;
        res.message = err;
        return;
      }
      written += export_prefix_ + "_occupancy.png ";
    }
    const OccGrid inten = map_.get_intensity_grid();
    if (!inten.empty()) {
      if (!write_grid_geo(inten, export_prefix_ + "_intensity", err)) {
        res.success = false;
        res.message = err;
        return;
      }
      written += export_prefix_ + "_intensity.png ";
    }
    if (!traj_.empty()) {
      std::ofstream gj(export_prefix_ + "_trajectory.geojson",
                       std::ios::trunc);
      gj.precision(9);
      gj << std::fixed
         << "{\"type\":\"FeatureCollection\",\"features\":[{\"type\":"
            "\"Feature\",\"properties\":{\"name\":\"sonar_slam "
            "trajectory\",\"keyframes\":"
         << traj_.size()
         << "},\"geometry\":{\"type\":\"LineString\",\"coordinates\":[";
      for (std::size_t k = 0; k < traj_.size(); ++k) {
        double lon, lat;
        datum_.map_to_lonlat(traj_[k].x, traj_[k].y, lon, lat);
        gj << (k ? "," : "") << "[" << lon << "," << lat << "]";
      }
      gj << "]}}]}\n";
      if (gj) written += export_prefix_ + "_trajectory.geojson";
    }
    if (written.empty()) {
      res.success = false;
      res.message = "nothing to export yet (no grids, no trajectory)";
      return;
    }
    res.success = true;
    res.message = "wrote " + written + "(UTM zone " +
                  std::to_string(datum_.origin.zone) +
                  (datum_.origin.north ? "N" : "S") + " sidecars)";
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  static nav_msgs::msg::OccupancyGrid to_occ_msg(
    const OccGrid& g, const builtin_interfaces::msg::Time& stamp)
  {
    nav_msgs::msg::OccupancyGrid m;
    m.header.stamp = stamp;
    m.header.frame_id = "map";
    m.info.map_load_time = stamp;
    m.info.resolution = static_cast<float>(g.resolution);
    m.info.width = static_cast<uint32_t>(g.width);
    m.info.height = static_cast<uint32_t>(g.height);
    m.info.origin.position.x = g.origin_x;
    m.info.origin.position.y = g.origin_y;
    m.info.origin.orientation.w = 1.0;
    m.data.assign(g.data.begin(), g.data.end());
    return m;
  }

  static constexpr int64_t kSyncTolNs = 2000000;  // 2 ms (pairing tolerance)
  int64_t stream_grace_ns_ = 1000000000;  // conclusively-lost verdict window
  static constexpr int kBufMax = 400;

  Mapping map_;
  std::mutex mutex_;
  // georeferenced export (see export_map)
  GeoDatum datum_;
  bool has_datum_ = false;
  std::string export_prefix_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr export_srv_;

  std::map<int64_t, SonarPing> ping_buf_;
  std::map<int64_t, Matrix> feat_buf_;
  std::vector<KFPose> traj_;
  int64_t last_stamp_ns_ = 0;
  double publish_period_ = 0.0;
  double last_pub_s_ = 0.0;

  rclcpp::SubscriptionBase::SharedPtr sonar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr feature_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr traj_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr intensity_pub_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::MappingNode>());
  rclcpp::shutdown();
  return 0;
}
