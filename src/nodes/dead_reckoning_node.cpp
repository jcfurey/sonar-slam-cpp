// Dead reckoning node: port of bruce_slam dead_reckoning.py. Fuses DVL
// velocities with an orientation source (IMU, FOG, both, or the SLAM heading
// feedback) plus depth into an odometry estimate.
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

#include <chrono>
#include <cmath>
#include <cstdint>
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
    // Max DVL time gap to integrate across; a larger gap (e.g. the synchronizer
    // dropped primaries during a secondary-stream stall) holds position instead
    // of integrating stale velocity into a jump. Must exceed the DVL's
    // synchronized cadence — the default accommodates any realistic DVL ping
    // rate while still catching multi-ten-second outages.
    dvl_max_gap_ = get_double("dvl_max_gap", 10.0);
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
      sync3_->set_overflow_callback([this](std::size_t n) { warn_dropped(n); });
      sync3_->set_nomatch_callback([this](std::size_t n) { warn_nomatch(n); });
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 dvl_seen_ = true;
                                 sync3_->add_primary(to_sec(r.stamp), r);
                               });
      imu_sub_ = subscribe_imu(this, imu_driver, imu_topic_, rclcpp::SensorDataQoS(rclcpp::KeepLast(300)),
                               [this](const ImuReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 imu_seen_ = true;
                                 last_imu_ = r;
                                 last_att_stamp_ = r.stamp;
                                 sync3_->add_secondary_b(to_sec(r.stamp), r);
                               });
      gyro_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        GYRO_INTEGRATION_TOPIC, 300, [this](const nav_msgs::msg::Odometry& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          gyro_seen_ = true;
          last_gyro_yaw_ = r2g(msg.pose.pose).rotation().yaw();
          last_att_stamp_ = msg.header.stamp;
          sync3_->add_secondary_c(to_sec(msg.header.stamp), msg);
        });
    } else if (use_imu_) {
      // IMU (roll/pitch/yaw) + DVL
      sync2_imu_ = std::make_unique<ApproxSync2<DvlReading, ImuReading>>(
        200, 0.1, [this](const DvlReading& dvl, const ImuReading& imu) {
          callback(imu, dvl);
        });
      sync2_imu_->set_overflow_callback([this](std::size_t n) { warn_dropped(n); });
      sync2_imu_->set_nomatch_callback([this](std::size_t n) { warn_nomatch(n); });
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 dvl_seen_ = true;
                                 sync2_imu_->add_primary(to_sec(r.stamp), r);
                               });
      imu_sub_ = subscribe_imu(this, imu_driver, imu_topic_, rclcpp::SensorDataQoS(rclcpp::KeepLast(300)),
                               [this](const ImuReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 imu_seen_ = true;
                                 last_imu_ = r;
                                 last_att_stamp_ = r.stamp;
                                 sync2_imu_->add_secondary(to_sec(r.stamp), r);
                               });
    } else if (use_gyro_) {
      // FOG only: yaw from the gyro, roll/pitch assumed level
      sync2_gyro_ = std::make_unique<ApproxSync2<DvlReading, nav_msgs::msg::Odometry>>(
        300, 0.1, [this](const DvlReading& dvl, const nav_msgs::msg::Odometry& gyro) {
          callback_gyro_only(dvl, gyro);
        });
      sync2_gyro_->set_overflow_callback([this](std::size_t n) { warn_dropped(n); });
      sync2_gyro_->set_nomatch_callback([this](std::size_t n) { warn_nomatch(n); });
      dvl_sub_ = subscribe_dvl(this, dvl_driver, dvl_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(50)),
                               [this](const DvlReading& r) {
                                 std::lock_guard<std::mutex> lock(mutex_);
                                 dvl_seen_ = true;
                                 sync2_gyro_->add_primary(to_sec(r.stamp), r);
                               });
      gyro_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        GYRO_INTEGRATION_TOPIC, 300, [this](const nav_msgs::msg::Odometry& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          gyro_seen_ = true;
          last_gyro_yaw_ = r2g(msg.pose.pose).rotation().yaw();
          last_att_stamp_ = msg.header.stamp;
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
                                 dvl_seen_ = true;
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

    // Startup watchdog: if a required stream never arrives, name it. The
    // synchronizer cannot emit without every stream, so a missing one (gyro
    // node omitted via enable_gyro:=false while use_gyro is true, a dead
    // driver, a wrong topic) otherwise means localization silently publishes
    // nothing for the whole mission.
    watchdog_ = create_wall_timer(std::chrono::seconds(10), [this]() {
      std::lock_guard<std::mutex> lock(mutex_);
      bool all_seen = true;
      if (!dvl_seen_) {
        all_seen = false;
        RCLCPP_ERROR(get_logger(),
                     "No DVL readings received yet — check dvl/driver and "
                     "dvl/topic; localization cannot start without the DVL");
      }
      if (use_imu_ && !imu_seen_) {
        all_seen = false;
        RCLCPP_ERROR(get_logger(),
                     "No IMU readings received yet — check imu/driver and "
                     "imu/topic (%s)", imu_topic_.c_str());
      }
      if (use_gyro_ && !gyro_seen_) {
        all_seen = false;
        RCLCPP_ERROR(get_logger(),
                     "No gyro integration received on %s — is gyro_node "
                     "running? (enable_gyro:=false with use_gyro: true starves "
                     "localization)", GYRO_INTEGRATION_TOPIC);
      }
      if (!last_depth_) {
        all_seen = false;
        RCLCPP_ERROR(get_logger(),
                     "No depth readings received yet — check depth/driver and "
                     "depth/topic");
      }
      if (all_seen) watchdog_->cancel();
    });

    // DVL-outage coast: when the synchronized DVL stream goes quiet
    // (bottom-lock loss, require_valid drops, secondary stall) the estimate
    // otherwise FREEZES while the vehicle keeps moving. Coast dead-reckons on
    // the last good body velocity rotated through the LIVE attitude stream
    // for up to dvl_coast seconds, then holds. 0 disables (historic
    // behavior). Covariance consumers see the same odometry topic; the SLAM
    // keyframe gate keeps coasted stretches from becoming keyframes by
    // itself only if features pair — treat long coasts as degraded DR.
    dvl_coast_ = get_double("dvl_coast", 0.0);
    if (dvl_coast_ > 0.0) {
      coast_timer_ = create_wall_timer(std::chrono::milliseconds(200),
                                       [this]() { coast_step(); });
      RCLCPP_INFO(get_logger(), "DVL coast enabled: up to %.1f s", dvl_coast_);
    }

    // Field diagnostics: the sync-health and gap counters otherwise exist
    // only as throttled log lines — publish them on /diagnostics so pairing
    // starvation is visible in rqt during a deployment (rqt_runtime_monitor /
    // diagnostics viewer), not just in a scrolled-away terminal.
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", 10);
    diag_timer_ = create_wall_timer(std::chrono::seconds(1),
                                    [this]() { publish_diagnostics(); });

    RCLCPP_INFO(get_logger(), "Localization node is initialized");
  }

private:
  // surface silently-dropped DVL primaries (synchronizer overflow while a
  // secondary stream is stalled) so the outage is observable
  void warn_dropped(std::size_t n)
  {
    dropped_total_ += n;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Localization dropped %zu DVL reading(s): an IMU/gyro stream is stalling; "
      "dead reckoning will re-anchor on recovery.",
      n);
  }

  // DVL primaries passed by the secondaries with NOTHING inside the 0.1 s
  // slop: the cross-device stamp-offset signature (DVL vs IMU/gyro driver
  // clocks disagree by more than slop). Distinct from an outage — data flows
  // on every stream, yet zero pairs ever emit and localization stays mute.
  void warn_nomatch(std::size_t n)
  {
    nomatch_total_ += n;
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Localization DVL readings are being passed by the IMU/gyro streams "
      "with no sample within the 0.1 s sync slop — the sensor driver clocks "
      "likely disagree (cross-device stamp offset). Measure the offset and "
      "set the <sensor>.stamp_offset parameter(s).");
  }

  // adapter-normalized quaternion + mounting rotation (imu_rotation in Python)
  gtsam::Rot3 imu_rotation(const ImuReading& imu) const
  {
    const gtsam::Rot3 rot = gtsam::Rot3::Quaternion(
      imu.orientation.w(), imu.orientation.x(), imu.orientation.y(),
      imu.orientation.z());
    return rot.compose(imu_rot_.inverse());
  }

  // full IMU-mode attitude: mounting rotation + yaw zeroing + the fixed
  // 90-degree roll offset of the historic VN100 frame handling (legacy
  // driver only). Shared by the paired callback and the coast path so the
  // two can never diverge.
  gtsam::Rot3 composed_imu_rot(const ImuReading& imu) const
  {
    const gtsam::Rot3 rot = imu_rotation(imu);
    double roll = rot.roll();
    if (imu_legacy_) roll += M_PI / 2.0;
    return gtsam::Rot3::Ypr(rot.yaw() - imu_yaw0_.value_or(rot.yaw()),
                            rot.pitch(), roll);
  }

  void callback(const ImuReading& imu, const DvlReading& dvl)
  {
    if (!last_depth_) return;
    if (!imu_yaw0_) imu_yaw0_ = imu_rotation(imu).yaw();
    send_odometry(dvl.velocity, composed_imu_rot(imu), dvl.stamp,
                  last_depth_->depth);
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
    ++pairs_total_;
    // coast bookkeeping: a real pair re-anchors everything
    last_pair_arrival_ = now();
    if (prev_time_) {
      const double dt0 = to_sec(dvl_time) - to_sec(*prev_time_);
      if (dt0 > 0.0 && dt0 < 5.0)
        dvl_period_ema_ = 0.9 * dvl_period_ema_ + 0.1 * dt0;
      // ABSORB any pair at/behind the integrated-to time (small negative dt):
      // the coast runs ahead on the attitude stream and sync delay can
      // deliver SEVERAL such pairs after an outage; rewinding prev_time_
      // would double-integrate the window, count fake gap events, and
      // publish backwards stamps. Refresh velocity/attitude/depth and keep
      // the clock. The reacquisition ping is exactly the one the spike gate
      // exists for, so never absorb a garbage velocity. Large negative
      // jumps (a bag rewind) still take the gap-gate reset below.
      if (dt0 <= 0.0 && dt0 > -5.0) {
        if (vel.cwiseAbs().maxCoeff() <= dvl_max_velocity_) prev_vel_ = vel;
        if (pose_)
          pose_ =
            gtsam::Pose3(rot, gtsam::Point3(pose_->x(), pose_->y(), depth));
        coast_elapsed_ = 0.0;
        coasting_ = false;
        return;
      }
    }
    coast_elapsed_ = 0.0;
    coasting_ = false;
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
      // A large (or negative/out-of-order) gap between DVL primaries — e.g. a
      // secondary stream stalled and the synchronizer dropped the intervening
      // primaries — must not integrate stale velocity into a single large
      // position jump. Hold the position, but keep tracking attitude and
      // depth (they don't depend on dt), so even a persistently-triggering
      // gate degrades gracefully instead of freezing the whole pose.
      if (dt < 0.0 || dt > dvl_max_gap_) {
        ++gap_events_;
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "DVL dt %.2fs outside (0, %.2f]s (dvl_max_gap); holding position, "
          "refreshing attitude/depth",
          dt, dvl_max_gap_);
        pose_ = gtsam::Pose3(rot, gtsam::Point3(pose_->x(), pose_->y(), depth));
        prev_time_ = dvl_time;
        prev_vel_ = vel;
        publish_pose(false);
        return;
      }
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

  // Timer body of the DVL-outage coast (dvl_coast > 0). Attitude keeps
  // flowing while the DVL is quiet, so the pose advances along the last
  // good body velocity rotated through the live attitude — bounded by
  // dvl_coast seconds, after which the position holds (the historic
  // behavior) until real pairs return. prev_time_ advances with each coast
  // step, so the returning DVL integrates only the remaining gap — the
  // coasted distance is never double-counted.
  void coast_step()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pose_ || !prev_time_) return;
    const double since_pair = (now() - last_pair_arrival_).seconds();
    // negative = node clock jumped backwards (bag loop): re-anchor
    if (since_pair < 0.0) {
      last_pair_arrival_ = now();
      return;
    }
    if (since_pair < std::max(0.3, 2.5 * dvl_period_ema_)) return;

    // data-domain dt from the attitude stream. The DVL+depth-only mode has
    // no attitude stream to stamp from — differencing now() (node clock)
    // against a sensor stamp would inject any driver clock offset into the
    // first coast step, so that mode simply does not coast.
    if (!last_att_stamp_) return;
    const builtin_interfaces::msg::Time stamp = *last_att_stamp_;
    const double dt = to_sec(stamp) - to_sec(*prev_time_);
    if (dt <= 0.0) return;
    if (coast_elapsed_ + dt > dvl_coast_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "DVL quiet for %.1f s — coast budget (dvl_coast %.1f s) exhausted; "
        "holding position",
        since_pair, dvl_coast_);
      return;
    }
    if (!coasting_) {
      ++coast_events_;
      RCLCPP_WARN(get_logger(),
                  "DVL quiet for %.1f s — coasting on last velocity "
                  "(%.2f, %.2f, %.2f) m/s + live attitude (budget %.1f s)",
                  since_pair, prev_vel_[0], prev_vel_[1], prev_vel_[2],
                  dvl_coast_);
    }
    coasting_ = true;
    coast_elapsed_ += dt;

    // live attitude, composed exactly like the paired callbacks
    gtsam::Rot3 rot = pose_->rotation();
    if (use_imu_ && last_imu_) {
      rot = composed_imu_rot(*last_imu_);
      if (use_gyro_ && last_gyro_yaw_)
        rot = gtsam::Rot3::Ypr(*last_gyro_yaw_, rot.pitch(), rot.roll());
    } else if (use_gyro_ && last_gyro_yaw_) {
      rot = gtsam::Rot3::Yaw(*last_gyro_yaw_);
    } else if (!use_imu_ && !use_gyro_) {
      rot = gtsam::Rot3::Yaw(slam_yaw_);
    }
    const double depth = last_depth_ ? last_depth_->depth : pose_->z();

    const Eigen::Vector3d trans = prev_vel_ * dt;
    const gtsam::Pose2 pose2(pose_->x(), pose_->y(), pose_->rotation().yaw());
    const gtsam::Point2 point =
      pose2.transformFrom(gtsam::Point2(trans[0], trans[1]));
    pose_ = gtsam::Pose3(rot, gtsam::Point3(point.x(), point.y(), depth));
    prev_time_ = stamp;
    publish_pose(false);
  }

  // 1 Hz /diagnostics snapshot. Level logic: pairing starvation since the
  // last tick (the stamp-offset signature) is ERROR — every downstream
  // consumer is silently frozen while it lasts; overflow drops / gap events
  // since the last tick and still-missing startup streams are WARN.
  void publish_diagnostics()
  {
    diagnostic_msgs::msg::DiagnosticStatus st;
    st.name = "sonar_slam/dead_reckoning";
    st.hardware_id = "sonar_slam_cpp";
    auto add = [&st](const char* k, const std::string& v) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = k;
      kv.value = v;
      st.values.push_back(kv);
    };

    std::uint64_t pairs, nomatch, dropped, gaps;
    bool started, missing_stream;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pairs = pairs_total_;
      nomatch = nomatch_total_;
      dropped = dropped_total_;
      gaps = gap_events_;
      started = pose_.has_value();
      missing_stream = !dvl_seen_ || (use_imu_ && !imu_seen_) ||
                       (use_gyro_ && !gyro_seen_) || !last_depth_;
      add("pairs_emitted", std::to_string(pairs));
      add("sync_nomatch_total", std::to_string(nomatch));
      add("sync_dropped_total", std::to_string(dropped));
      add("dvl_gap_events", std::to_string(gaps));
      add("coasting", coasting_ ? "true" : "false");
      add("coast_events", std::to_string(coast_events_));
      add("coast_elapsed", std::to_string(coast_elapsed_));
      add("dvl_seen", dvl_seen_ ? "true" : "false");
      add("imu_seen", imu_seen_ ? "true" : "false");
      add("gyro_seen", gyro_seen_ ? "true" : "false");
      add("depth_seen", last_depth_ ? "true" : "false");
      add("publishing", started ? "true" : "false");
    }

    if (nomatch > diag_prev_nomatch_) {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      st.message = "sync no-match starvation: streams alive but zero pairs "
                   "emitted — measure and set <sensor>.stamp_offset";
    } else if (missing_stream && !started) {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      st.message = "waiting for a required sensor stream (see *_seen)";
    } else if (dropped > diag_prev_dropped_ || gaps > diag_prev_gaps_) {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      st.message = "secondary stream stalling (drops/gap events increasing)";
    } else {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      st.message = started ? "publishing" : "waiting for first pair";
    }
    diag_prev_nomatch_ = nomatch;
    diag_prev_dropped_ = dropped;
    diag_prev_gaps_ = gaps;

    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    arr.status.push_back(std::move(st));
    diag_pub_->publish(arr);
  }

  gtsam::Rot3 imu_rot_;
  bool imu_legacy_ = true;
  double dvl_max_velocity_ = 0.3;
  double dvl_max_gap_ = 1.0;
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
  bool dvl_seen_ = false, imu_seen_ = false, gyro_seen_ = false;
  std::vector<std::pair<double, gtsam::Pose3>> keyframes_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  // DVL-outage coast state (see coast_step)
  double dvl_coast_ = 0.0;
  double dvl_period_ema_ = 0.2;
  double coast_elapsed_ = 0.0;
  bool coasting_ = false;
  std::uint64_t coast_events_ = 0;
  std::optional<ImuReading> last_imu_;
  std::optional<double> last_gyro_yaw_;
  std::optional<builtin_interfaces::msg::Time> last_att_stamp_;
  rclcpp::Time last_pair_arrival_{0, 0, RCL_ROS_TIME};
  rclcpp::TimerBase::SharedPtr coast_timer_;

  // diagnostics counters (mutated under mutex_ by the sync/odometry paths)
  std::uint64_t pairs_total_ = 0, nomatch_total_ = 0, dropped_total_ = 0,
                gap_events_ = 0;
  std::uint64_t diag_prev_nomatch_ = 0, diag_prev_dropped_ = 0,
                diag_prev_gaps_ = 0;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

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
  return sonar_slam::run_node<sonar_slam::DeadReckoningNode>(argc, argv);
}
