// Dead reckoning node: port of bruce_slam dead_reckoning.py. Fuses DVL
// velocities with an orientation source (IMU, FOG, both, or the SLAM heading
// feedback) plus depth into an odometry estimate.
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <optional>

#include "sonar_slam_cpp/approx_sync.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"

namespace sonar_slam {

class DeadReckoningNode : public SlamNodeBase
{
public:
  DeadReckoningNode() : SlamNodeBase("localization")
  {
    const auto imu_pose = get_double_array("imu_pose");  // [x y z roll pitch yaw]
    imu_rot_ = gtsam::Rot3::Ypr(imu_pose[5], imu_pose[4], imu_pose[3]);
    dvl_max_velocity_ = get_double("dvl_max_velocity");
    keyframe_duration_ = get_double("keyframe_duration");
    keyframe_translation_ = get_double("keyframe_translation");
    keyframe_rotation_ = get_double("keyframe_rotation");

    const std::string dvl_driver = get_string("dvl/driver", "rti_dvl");
    const std::string depth_driver = get_string("depth/driver", "bar30");
    const std::string dvl_topic = get_string("dvl/topic", DVL_TOPIC);
    const std::string depth_topic = get_string("depth/topic", DEPTH_TOPIC);

    use_gyro_ = get_bool("use_gyro");
    use_imu_ = get_bool("use_imu", true);

    traj_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("traj_dead_reck", 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(LOCALIZATION_ODOM_TOPIC, 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // latest depth reading, held like message_filters.Cache(depth_sub, 1)
    depth_sub_ = subscribe_depth(this, depth_driver, depth_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(10)),
                                 [this](const DepthReading& r) {
                                   std::lock_guard<std::mutex> lock(mutex_);
                                   last_depth_ = r;
                                 });

    std::string imu_driver;
    if (use_imu_) {
      imu_driver = get_string("imu/driver", "vn100");
      imu_legacy_ = imu_driver_is_legacy(imu_driver);
      std::string imu_topic = get_string("imu/topic", "");
      if (imu_topic.empty()) {
        if (imu_legacy_)
          imu_topic = get_int("imu_version", 1) == 1 ? IMU_TOPIC : IMU_TOPIC_MK_II;
        else
          imu_topic = IMU_TOPIC_ENU;
      }
      imu_topic_ = imu_topic;
    }

    if (use_imu_ && use_gyro_) {
      // IMU (roll/pitch) + FOG (yaw) + DVL
      sync3_ = std::make_unique<ApproxSync3<DvlReading, ImuReading, nav_msgs::msg::Odometry>>(
        300, 0.1, [this](const DvlReading& dvl, const ImuReading& imu,
                         const nav_msgs::msg::Odometry& gyro) {
          callback_with_gyro(imu, dvl, gyro);
        });
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 sync3_->add_primary(to_sec(r.stamp), r);
                               });
      imu_sub_ = subscribe_imu(this, imu_driver, imu_topic_, rclcpp::SensorDataQoS(rclcpp::KeepLast(300)),
                               [this](const ImuReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 sync3_->add_secondary_b(to_sec(r.stamp), r);
                               });
      gyro_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        GYRO_INTEGRATION_TOPIC, 300, [this](const nav_msgs::msg::Odometry& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          sync3_->add_secondary_c(to_sec(msg.header.stamp), msg);
        });
    } else if (use_imu_) {
      // IMU (roll/pitch/yaw) + DVL
      sync2_imu_ = std::make_unique<ApproxSync2<DvlReading, ImuReading>>(
        200, 0.1, [this](const DvlReading& dvl, const ImuReading& imu) {
          callback(imu, dvl);
        });
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 sync2_imu_->add_primary(to_sec(r.stamp), r);
                               });
      imu_sub_ = subscribe_imu(this, imu_driver, imu_topic_, rclcpp::SensorDataQoS(rclcpp::KeepLast(300)),
                               [this](const ImuReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 sync2_imu_->add_secondary(to_sec(r.stamp), r);
                               });
    } else if (use_gyro_) {
      // FOG only: yaw from the gyro, roll/pitch assumed level
      sync2_gyro_ = std::make_unique<ApproxSync2<DvlReading, nav_msgs::msg::Odometry>>(
        300, 0.1, [this](const DvlReading& dvl, const nav_msgs::msg::Odometry& gyro) {
          callback_gyro_only(dvl, gyro);
        });
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 sync2_gyro_->add_primary(to_sec(r.stamp), r);
                               });
      gyro_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        GYRO_INTEGRATION_TOPIC, 300, [this](const nav_msgs::msg::Odometry& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          sync2_gyro_->add_secondary(to_sec(msg.header.stamp), msg);
        });
      RCLCPP_WARN(get_logger(),
                  "Localization running in gyro-only mode (no IMU); "
                  "roll/pitch assumed level.");
    } else {
      // DVL + depth only; heading fed back from the SLAM scan matcher
      slam_yaw_ = get_double("seed_heading", 0.0) * M_PI / 180.0;
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 callback_dvl_only(r);
                               });
      slam_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        SLAM_ODOM_TOPIC, 10, [this](const nav_msgs::msg::Odometry& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          slam_yaw_ = r2g(msg.pose.pose).rotation().yaw();
        });
      RCLCPP_WARN(get_logger(),
                  "Localization running in DVL+depth only mode (no IMU/FOG); "
                  "heading is taken from the SLAM scan matcher.");
    }

    RCLCPP_INFO(get_logger(), "Localization node is initialized");
  }

private:
  // adapter-normalized quaternion + mounting rotation (imu_rotation in Python)
  gtsam::Rot3 imu_rotation(const ImuReading& imu) const
  {
    const gtsam::Rot3 rot = gtsam::Rot3::Quaternion(
      imu.orientation.w(), imu.orientation.x(), imu.orientation.y(),
      imu.orientation.z());
    return rot.compose(imu_rot_.inverse());
  }

  void callback(const ImuReading& imu, const DvlReading& dvl)
  {
    if (!last_depth_) return;

    gtsam::Rot3 rot = imu_rotation(imu);
    if (!imu_yaw0_) imu_yaw0_ = rot.yaw();

    // the fixed 90-degree roll offset is part of the historic VN100 frame
    // handling and only applies to the legacy driver
    double roll = rot.roll();
    if (imu_legacy_) roll += M_PI / 2.0;
    rot = gtsam::Rot3::Ypr(rot.yaw() - *imu_yaw0_, rot.pitch(), roll);

    send_odometry(dvl.velocity, rot, dvl.stamp, last_depth_->depth);
  }

  void callback_with_gyro(const ImuReading& imu, const DvlReading& dvl,
                          const nav_msgs::msg::Odometry& gyro_msg)
  {
    if (!last_depth_) return;

    const double gyro_yaw = r2g(gyro_msg.pose.pose).rotation().yaw();
    const gtsam::Rot3 imu_rot = imu_rotation(imu);
    const gtsam::Rot3 rot = gtsam::Rot3::Ypr(gyro_yaw, imu_rot.pitch(), imu_rot.roll());
    send_odometry(dvl.velocity, rot, dvl.stamp, last_depth_->depth);
  }

  void callback_gyro_only(const DvlReading& dvl, const nav_msgs::msg::Odometry& gyro_msg)
  {
    if (!last_depth_) return;
    const double gyro_yaw = r2g(gyro_msg.pose.pose).rotation().yaw();
    send_odometry(dvl.velocity, gtsam::Rot3::Yaw(gyro_yaw), dvl.stamp,
                  last_depth_->depth);
  }

  void callback_dvl_only(const DvlReading& dvl)
  {
    if (!last_depth_) return;
    send_odometry(dvl.velocity, gtsam::Rot3::Yaw(slam_yaw_), dvl.stamp,
                  last_depth_->depth);
  }

  void send_odometry(Eigen::Vector3d vel, const gtsam::Rot3& rot,
                     const builtin_interfaces::msg::Time& dvl_time, double depth)
  {
    // DVL velocity spike handling (dead_reckoning.py send_odometry)
    if (vel.cwiseAbs().maxCoeff() > dvl_max_velocity_) {
      if (pose_) {
        dvl_error_timer_ += to_sec(dvl_time) - to_sec(*prev_time_);
        if (dvl_error_timer_ > 5.0) {
          RCLCPP_WARN(get_logger(),
                      "DVL velocity (%.1f, %.1f, %.1f) exceeds max velocity "
                      "%.1f for %.1f secs.",
                      vel[0], vel[1], vel[2], dvl_max_velocity_, dvl_error_timer_);
        }
        vel = prev_vel_;
      } else {
        return;
      }
    } else {
      dvl_error_timer_ = 0.0;
    }

    if (pose_) {
      const double dt = to_sec(dvl_time) - to_sec(*prev_time_);
      const Eigen::Vector3d dv = (vel + prev_vel_) * 0.5;
      const Eigen::Vector3d trans = dv * dt;

      const gtsam::Pose2 pose2(pose_->x(), pose_->y(), pose_->rotation().yaw());
      const gtsam::Point2 point =
        pose2.transformFrom(gtsam::Point2(trans[0], trans[1]));
      pose_ = gtsam::Pose3(rot, gtsam::Point3(point.x(), point.y(), depth));
    } else {
      pose_ = gtsam::Pose3(rot, gtsam::Point3(0, 0, depth));
    }

    prev_time_ = dvl_time;
    prev_vel_ = vel;

    bool new_keyframe = false;
    if (keyframes_.empty()) {
      new_keyframe = true;
    } else {
      const double duration = to_sec(*prev_time_) - keyframes_.back().first;
      if (duration > keyframe_duration_) {
        const gtsam::Pose3 odom = keyframes_.back().second.between(*pose_);
        const double translation = odom.translation().norm();
        const double rotation = std::abs(odom.rotation().yaw());
        if (translation > keyframe_translation_ || rotation > keyframe_rotation_)
          new_keyframe = true;
      }
    }

    if (new_keyframe) keyframes_.emplace_back(to_sec(*prev_time_), *pose_);
    publish_pose(new_keyframe);
  }

  void publish_pose(bool publish_traj)
  {
    if (!pose_) return;

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = *prev_time_;
    odom_msg.header.frame_id = "odom";
    odom_msg.pose.pose = g2r(*pose_);
    odom_msg.child_frame_id = "base_link";
    odom_pub_->publish(odom_msg);

    const auto& p = odom_msg.pose.pose.position;
    const auto& q = odom_msg.pose.pose.orientation;
    tf_->sendTransform(make_transform(Eigen::Vector3d(p.x, p.y, p.z),
                                      Eigen::Quaterniond(q.w, q.x, q.y, q.z),
                                      odom_msg.header.stamp, "odom", "base_link"));

    if (publish_traj) {
      Matrix traj(static_cast<long>(keyframes_.size()), 7);
      for (std::size_t i = 0; i < keyframes_.size(); ++i) {
        const auto& pose = keyframes_[i].second;
        traj(i, 0) = static_cast<float>(pose.x());
        traj(i, 1) = static_cast<float>(pose.y());
        traj(i, 2) = static_cast<float>(pose.z());
        traj(i, 3) = static_cast<float>(pose.rotation().roll());
        traj(i, 4) = static_cast<float>(pose.rotation().pitch());
        traj(i, 5) = static_cast<float>(pose.rotation().yaw());
        traj(i, 6) = static_cast<float>(i);
      }
      sensor_msgs::msg::PointCloud2 msg =
        make_cloud({"x", "y", "z", "roll", "pitch", "yaw", "i"}, traj);
      msg.header = odom_msg.header;
      traj_pub_->publish(msg);
    }
  }

  gtsam::Rot3 imu_rot_;
  bool imu_legacy_ = true;
  double dvl_max_velocity_ = 0.3;
  double keyframe_duration_ = 1.0;
  double keyframe_translation_ = 4.0;
  double keyframe_rotation_ = 0.5;
  bool use_gyro_ = false, use_imu_ = true;
  std::string imu_topic_;

  std::mutex mutex_;
  std::optional<gtsam::Pose3> pose_;
  std::optional<builtin_interfaces::msg::Time> prev_time_;
  Eigen::Vector3d prev_vel_ = Eigen::Vector3d::Zero();
  std::optional<double> imu_yaw0_;
  double dvl_error_timer_ = 0.0;
  double slam_yaw_ = 0.0;
  std::optional<DepthReading> last_depth_;
  std::vector<std::pair<double, gtsam::Pose3>> keyframes_;

  rclcpp::SubscriptionBase::SharedPtr dvl_sub_, depth_sub_, imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gyro_sub_, slam_sub_;
  std::unique_ptr<ApproxSync3<DvlReading, ImuReading, nav_msgs::msg::Odometry>> sync3_;
  std::unique_ptr<ApproxSync2<DvlReading, ImuReading>> sync2_imu_;
  std::unique_ptr<ApproxSync2<DvlReading, nav_msgs::msg::Odometry>> sync2_gyro_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr traj_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::DeadReckoningNode>());
  rclcpp::shutdown();
  return 0;
}
