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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

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

    map_.min_translation = get_double("min_translation", 0.5);
    map_.min_rotation = get_double("min_rotation", 0.05);

    // must match slam_node's enu_world so the reconstructed sonar fan lands in
    // the same frame chirality as the trajectory poses
    const bool enu = get_bool("enu_world", true);
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
    const int64_t msg_ns = stamp_ns(msg.header.stamp);
    last_stamp_ns_ = msg_ns;

    traj_.assign(n, KFPose{});
    sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x"), iy(msg, "y"),
      iyaw(msg, "yaw"), it(msg, "t");
    for (int k = 0; k < n; ++k, ++ix, ++iy, ++iyaw, ++it) {
      traj_[k].x = *ix;
      traj_[k].y = *iy;
      traj_[k].yaw = *iyaw;
      traj_[k].stamp_ns = msg_ns + llround(static_cast<double>(*it) * 1e9);
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
  bool try_build()
  {
    bool built = false;
    while (map_.num_keyframes() < static_cast<int>(traj_.size())) {
      const int k = map_.num_keyframes();
      const int64_t s = traj_[k].stamp_ns;
      auto pit = find_near(ping_buf_, s);
      auto fit = find_near(feat_buf_, s);
      if (pit == ping_buf_.end() || fit == feat_buf_.end()) break;  // await data
      map_.add_keyframe(k, gtsam::Pose2(traj_[k].x, traj_[k].y, traj_[k].yaw),
                        pit->second, fit->second);
      built = true;
    }
    return built;
  }

  template <class M>
  static typename M::iterator find_near(M& buf, int64_t key)
  {
    auto best = buf.end();
    int64_t bestd = kSyncTolNs + 1;
    auto it = buf.lower_bound(key);
    if (it != buf.end()) {
      const int64_t d = std::llabs(it->first - key);
      if (d <= kSyncTolNs && d < bestd) { best = it; bestd = d; }
    }
    if (it != buf.begin()) {
      auto p = std::prev(it);
      const int64_t d = std::llabs(p->first - key);
      if (d <= kSyncTolNs && d < bestd) { best = p; bestd = d; }
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

  static constexpr int64_t kSyncTolNs = 2000000;  // 2 ms
  static constexpr int kBufMax = 400;

  Mapping map_;
  std::mutex mutex_;

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
