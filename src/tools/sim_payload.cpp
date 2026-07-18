// Demo payload simulator: drives the WHOLE standard pipeline with no
// hardware, no bags and no vendor packages — the 5-minute demo. Publishes a
// synthetic rectangular pool (the same world the end-to-end test locks
// against) as:
//   /sim/sonar_image  sensor_msgs/Image           (sonar driver: image)
//   /sim/dvl          geometry_msgs/TwistStamped  (dvl driver: twist_stamped)
//   /sim/imu          sensor_msgs/Imu             (imu driver: enu)
//   /sim/depth        std_msgs/Float64            (depth driver: float)
// The vehicle laps the pool perimeter at constant speed; the DVL carries a
// small bias so dead reckoning drifts visibly and the loop closures (and the
// RViz "2D Pose Estimate" hand correction) have something real to fix.
//
//   ros2 launch sonar_slam_cpp demo.launch.xml
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "sonar_slam_cpp/synthetic_world.hpp"

namespace sonar_slam {

class SimPayload : public rclcpp::Node
{
public:
  SimPayload() : Node("sim_payload"), world_(SyntheticWorld::pool(20.0, 10.0)),
                 rng_(7)
  {
    speed_ = declare_parameter("speed", 0.35);            // m/s along the lap
    dvl_bias_x_ = declare_parameter("dvl_bias_x", 0.010);  // m/s -> DR drift
    dvl_bias_y_ = declare_parameter("dvl_bias_y", 0.015);

    // perimeter lap, repeated forever
    const double wp[5][3] = {{2, 2, 0.0},
                             {18, 2, M_PI / 2},
                             {18, 8, M_PI},
                             {2, 8, -M_PI / 2},
                             {2, 2, 0.0}};
    for (auto& w : wp) waypoints_.push_back({w[0], w[1], w[2]});

    img_pub_ = create_publisher<sensor_msgs::msg::Image>("/sim/sonar_image", 5);
    dvl_pub_ =
      create_publisher<geometry_msgs::msg::TwistStamped>("/sim/dvl", 10);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/sim/imu", 50);
    depth_pub_ = create_publisher<std_msgs::msg::Float64>("/sim/depth", 10);

    timer_ = create_wall_timer(std::chrono::milliseconds(50),
                               [this]() { tick(); });
    RCLCPP_INFO(get_logger(),
                "sim_payload: 20x10 m pool with posts, %.2f m/s lap, DVL "
                "bias (%.3f, %.3f) m/s",
                speed_, dvl_bias_x_, dvl_bias_y_);
  }

private:
  struct Waypoint { double x, y, heading_after; };

  void tick()
  {
    const double dt = 0.05;
    // advance along the current leg; heading snaps at corners (a lap is a
    // demo, not a dynamics model)
    const auto& a = waypoints_[leg_];
    const auto& b = waypoints_[(leg_ + 1) % waypoints_.size()];
    const double len = std::hypot(b.x - a.x, b.y - a.y);
    s_ += speed_ * dt;
    if (s_ >= len) {
      s_ = 0.0;
      leg_ = (leg_ + 1) % (waypoints_.size() - 1);
      ++laps_ticks_;
    }
    const double f = len > 0.0 ? s_ / len : 0.0;
    const double yaw = std::atan2(b.y - a.y, b.x - a.x);
    pose_ = gtsam::Pose2(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), yaw);

    const auto stamp = now();
    ++tick_count_;

    // IMU (ENU orientation) every tick (20 Hz)
    sensor_msgs::msg::Imu imu;
    imu.header.stamp = stamp;
    imu.header.frame_id = "imu_link";
    imu.orientation.z = std::sin(pose_.theta() / 2.0);
    imu.orientation.w = std::cos(pose_.theta() / 2.0);
    imu.orientation_covariance[0] = 0.01;
    imu_pub_->publish(imu);

    // DVL + depth at 5 Hz
    if (tick_count_ % 4 == 0) {
      std::normal_distribution<double> vn(0.0, 0.01);
      geometry_msgs::msg::TwistStamped dvl;
      dvl.header.stamp = stamp;
      dvl.header.frame_id = "dvl_link";
      dvl.twist.linear.x = speed_ + dvl_bias_x_ + vn(rng_);
      dvl.twist.linear.y = dvl_bias_y_ + vn(rng_);
      dvl.twist.linear.z = 0.0;
      dvl_pub_->publish(dvl);

      std_msgs::msg::Float64 depth;
      std::normal_distribution<double> dn(0.0, 0.01);
      depth.data = 1.0 + dn(rng_);
      depth_pub_->publish(depth);
    }

    // sonar at 2 Hz
    if (tick_count_ % 10 == 0) {
      constexpr int kBeams = 128, kBins = 300;
      constexpr double kMaxRange = 25.0;
      const cv::Mat img = world_.polar_image(
        pose_, 130.0 * M_PI / 180.0, kBeams, kBins, kMaxRange, rng_);
      sensor_msgs::msg::Image m;
      m.header.stamp = stamp;
      m.header.frame_id = "sonar_link";
      m.height = static_cast<std::uint32_t>(img.rows);
      m.width = static_cast<std::uint32_t>(img.cols);
      m.encoding = "mono8";
      m.step = static_cast<std::uint32_t>(img.cols);
      m.data.assign(img.datastart, img.dataend);
      img_pub_->publish(m);
    }
  }

  SyntheticWorld world_;
  std::mt19937 rng_;
  std::vector<Waypoint> waypoints_;
  std::size_t leg_ = 0;
  double s_ = 0.0;
  double speed_ = 0.35, dvl_bias_x_ = 0.01, dvl_bias_y_ = 0.015;
  gtsam::Pose2 pose_{2.0, 2.0, 0.0};
  std::uint64_t tick_count_ = 0, laps_ticks_ = 0;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr dvl_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr depth_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::SimPayload>());
  rclcpp::shutdown();
  return 0;
}
