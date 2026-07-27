// Map-quality metrics for the MAP_DOUBLING_FIX_PLAN §5 replay protocol:
// turns "did the doubling improve?" from an eyeball comparison into numbers.
//
// Subscribes to the aggregated SLAM map cloud (latched, so it also works
// against a recorded bag of the topic) and, on every update, rasterizes the
// x/y projection, skeletonizes the walls, and reports:
//
//   wall_len    total skeleton length (m)
//   thick_med / thick_p95
//               local wall thickness (m): 2x distance-to-free at each
//               skeleton cell (the medial-axis definition). The published
//               cloud is octree-downsampled per keyframe, but cell centroids
//               of a thin wall stay on the wall plane, so the projected
//               thickness resolves well below the downsample cell size.
//   doubled     fraction of skeleton length that has a second, parallel
//               (within 20 deg) wall in the probe band along its normal —
//               the doubled-wall signature. Band default 0.3–3.0 m, the
//               scale the CHL_Pool doubling was observed at.
//
// Usage (defaults in brackets):
//   ros2 run sonar_slam_cpp map_metrics [--topic /bruce/slam/slam/cloud]
//        [--grid 0.05] [--band 0.3 3.0] [--once]
//
// Replay a recorded run:   ros2 bag play <bag>   (cloud topic included)
// and read the last line printed — each line is one cloud update.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/map_metrics_math.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"

namespace sonar_slam {

class MapMetricsNode : public rclcpp::Node
{
public:
  MapMetricsNode(const std::string& topic, double grid, double band_min,
                 double band_max, bool once)
  : Node("map_metrics"), grid_(grid), band_min_(band_min), band_max_(band_max),
    once_(once)
  {
    // latched publisher -> transient_local subscription so a bag that played
    // before we started (or the live latched sample) is still delivered
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic, rclcpp::QoS(1).reliable().transient_local(),
      [this](const sensor_msgs::msg::PointCloud2& msg) { on_cloud(msg); });
    std::fprintf(stderr,
                 "[map_metrics] waiting for clouds on %s (grid %.2f m, band "
                 "%.1f-%.1f m)...\n",
                 topic.c_str(), grid_, band_min_, band_max_);
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2& msg)
  {
    const Matrix xyz = cloud_to_xyz(msg);
    const Metrics m = analyze(xyz, grid_, band_min_, band_max_);
    ++updates_;
    std::printf(
      "[map_metrics] update=%d pts=%ld cells=%ld wall_len=%.1fm "
      "thick_med=%.2fm thick_p95=%.2fm doubled=%.1f%% skel=%ld\n",
      updates_, m.points, m.occupied_cells, m.wall_len, m.thick_med,
      m.thick_p95, 100.0 * m.doubled_frac, m.skel_cells);
    std::fflush(stdout);
    if (once_) rclcpp::shutdown();
  }

  double grid_, band_min_, band_max_;
  bool once_;
  int updates_ = 0;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  std::string topic = "/bruce/slam/slam/cloud";
  double grid = 0.05, band_min = 0.3, band_max = 3.0;
  bool once = false;
  try {
    const std::vector<std::string> args =
      rclcpp::remove_ros_arguments(argc, argv);
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& a = args[i];
      auto next = [&](double& out) {
        if (i + 1 >= args.size())
          throw std::runtime_error(a + " needs a value");
        out = std::stod(args[++i]);
      };
      if (a == "--topic" && i + 1 < args.size()) {
        topic = args[++i];
      } else if (a == "--grid") {
        next(grid);
      } else if (a == "--band") {
        next(band_min);
        next(band_max);
      } else if (a == "--once") {
        once = true;
      } else {
        throw std::runtime_error("unknown argument " + a);
      }
    }
    if (grid <= 0.0 || band_min <= 0.0 || band_max <= band_min)
      throw std::runtime_error("need grid > 0 and band max > band min > 0");
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "%s\nusage: map_metrics [--topic T] [--grid m] "
                 "[--band min max] [--once]\n",
                 e.what());
    return 2;
  }
  rclcpp::spin(std::make_shared<sonar_slam::MapMetricsNode>(
    topic, grid, band_min, band_max, once));
  rclcpp::shutdown();
  return 0;
}
