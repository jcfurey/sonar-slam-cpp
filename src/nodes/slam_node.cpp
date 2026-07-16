// SLAM node: port of bruce_slam slam_ros.py + slam_node.py. Synchronizes the
// sonar feature clouds with dead-reckoning odometry, runs the SSM/NSSM/ISAM2
// back-end, and publishes pose, odometry, trajectory, constraints, the
// registered map cloud and the map->odom TF.
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "sonar_slam_cpp/approx_sync.hpp"
#include "sonar_slam_cpp/gpu.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"
#include "sonar_slam_cpp/slam_core.hpp"

namespace sonar_slam {

namespace {

rclcpp::QoS latched_qos(int depth = 1)
{
  return rclcpp::QoS(depth).reliable().transient_local();
}

}  // namespace

class SlamNode : public SlamNodeBase
{
public:
  SlamNode() : SlamNodeBase("slam")
  {
    slam_.keyframe_duration = get_double("keyframe_duration");
    slam_.keyframe_translation = get_double("keyframe_translation");
    slam_.keyframe_rotation = get_double("keyframe_rotation");

    enable_slam_ = get_bool("enable_slam");
    RCLCPP_INFO(get_logger(), "SLAM STATUS: %s", enable_slam_ ? "true" : "false");

    // emit the map->odom TF in ENU (REP-105) instead of the graph's z-down
    // convention — pair with the enu_odom_relay feeding the odom input
    enu_world_ = get_bool("enu_world", false);
    // false when an external node (e.g. the map robot_localization EKF
    // fusing /bruce/slam/slam/pose) owns the map->odom transform
    publish_tf_ = get_bool("publish_tf", true);
    // min seconds between rebuilds of the O(history) viz outputs
    // (constraint markers + aggregated map cloud); <= 0 -> every keyframe
    viz_min_period_ = get_double("viz_min_period", 2.0);

    const auto prior = get_double_array("prior_sigmas");
    const auto odom = get_double_array("odom_sigmas");
    const auto icp_odom = get_double_array("icp_odom_sigmas");
    slam_.prior_sigmas = Eigen::Vector3d(prior[0], prior[1], prior[2]);
    slam_.odom_sigmas = Eigen::Vector3d(odom[0], odom[1], odom[2]);
    slam_.icp_odom_sigmas = Eigen::Vector3d(icp_odom[0], icp_odom[1], icp_odom[2]);

    slam_.point_resolution = get_double("point_resolution");
    slam_.point_noise = get_double("point_noise", 0.5);
    feature_odom_sync_max_delay_ =
      get_double("feature_odom_sync_max_delay", 0.5);

    // [sobol samples per iteration, iterations, local-refine ftol]
    const auto init_params = [this](const char* name,
                                    const std::vector<double>& def) {
      const auto v = get_double_array(name, def);
      if (v.size() != 3)
        throw std::runtime_error(std::string(name) +
                                 " must be [n, iters, ftol]");
      return v;
    };

    // ICP-covariance method when cov_samples > 0: "sampled" (many ICPs +
    // FAST-MCD, the default/parity behavior) or "censi" (one ICP + closed-form)
    const auto cov_method = [this](const char* name) {
      const std::string s = get_string(name, "sampled");
      if (s == "sampled") return SMParams::SAMPLED;
      if (s == "censi") return SMParams::CENSI;
      throw std::runtime_error(std::string(name) +
                               " must be 'sampled' or 'censi'");
    };
    slam_.censi_sensor_noise = get_double("censi_sensor_noise", 0.1);
    // parallelize the cov_samples registrations of the "sampled" method
    // across an ICP engine pool (see slam_core.hpp); false = the historical
    // one-core sequential loop
    slam_.parallel_cov_samples = get_bool("parallel_cov_samples", true);

    slam_.ssm_params.enable = get_bool("ssm/enable");
    slam_.ssm_params.min_points = get_int("ssm/min_points");
    slam_.ssm_params.max_translation = get_double("ssm/max_translation");
    slam_.ssm_params.max_rotation = get_double("ssm/max_rotation");
    slam_.ssm_params.target_frames = get_int("ssm/target_frames");
    // covariance-estimating ICP for SSM (core supported it but the config
    // never reached it — in the python original these fields were unwired,
    // which left SSM on plain point-to-point ICP with a fixed noise model;
    // that poisoned the graph on pool geometry, see SONAR_SLAM_PLAN.md)
    slam_.ssm_params.initialization = get_bool("ssm/initialization", true);
    const auto ssm_init = init_params("ssm/initialization_params", {50.0, 1.0, 0.01});
    slam_.ssm_params.init_n = static_cast<int>(ssm_init[0]);
    slam_.ssm_params.init_iters = static_cast<int>(ssm_init[1]);
    slam_.ssm_params.init_ftol = ssm_init[2];
    slam_.ssm_params.cov_samples = get_int("ssm/cov_samples", 0);
    slam_.ssm_params.cov_method = cov_method("ssm/cov_method");

    slam_.nssm_params.enable = get_bool("nssm/enable");
    slam_.nssm_params.min_st_sep = get_int("nssm/min_st_sep");
    slam_.nssm_params.min_points = get_int("nssm/min_points");
    slam_.nssm_params.max_translation = get_double("nssm/max_translation");
    slam_.nssm_params.max_rotation = get_double("nssm/max_rotation");
    slam_.nssm_params.source_frames = get_int("nssm/source_frames");
    slam_.nssm_params.initialization = get_bool("nssm/initialization", true);
    const auto nssm_init =
      init_params("nssm/initialization_params", {100.0, 5.0, 0.01});
    slam_.nssm_params.init_n = static_cast<int>(nssm_init[0]);
    slam_.nssm_params.init_iters = static_cast<int>(nssm_init[1]);
    slam_.nssm_params.init_ftol = nssm_init[2];
    slam_.nssm_params.cov_samples = get_int("nssm/cov_samples");
    slam_.nssm_params.cov_method = cov_method("nssm/cov_method");

    slam_.pcm_queue_size = get_int("pcm_queue_size");
    slam_.min_pcm = get_int("min_pcm");
    // NSSM degeneracy gate (see slam_core.hpp)
    slam_.nssm_max_sigma = get_double("nssm/max_sigma", 0.5);
    slam_.nssm_max_anisotropy = get_double("nssm/max_anisotropy", 8.0);
    slam_.nssm_max_yaw_vs_compass = get_double("nssm/max_yaw_vs_compass", 0.15);
    slam_.post_loop_max_yaw_rms = get_double("post_loop_max_yaw_rms", 0.15);

    // ICP config; falls back to the installed package share copy when unset
    std::string icp_config = get_string("icp_config", "");
    if (icp_config.empty() || !std::filesystem::is_regular_file(icp_config)) {
      const std::string fallback =
        ament_index_cpp::get_package_share_directory("sonar_slam_cpp") +
        "/config/icp.yaml";
      if (!icp_config.empty()) {
        // an EXPLICIT path that doesn't resolve (e.g. a harness loading the
        // deployed yaml without launch substitution) must be loud: the
        // package default carries open-water matcher distances and registers
        // materially differently from the deployed config
        RCLCPP_ERROR(get_logger(),
                     "icp_config '%s' is not a readable file — falling back "
                     "to package default '%s' (different matcher distances; "
                     "registration behavior will not match the deployed "
                     "config)",
                     icp_config.c_str(), fallback.c_str());
      }
      icp_config = fallback;
    }
    slam_.icp.load_from_yaml(icp_config);

    // raw sonar keeps the Oculus geometry (max range / aperture, bounding the
    // NSSM loop-closure search) current — the operator can change the sonar
    // range mid-mission, so reconfigure whenever the ping geometry changes
    const std::string sonar_driver = get_string("sonar/driver", "oculus_compressed");
    const std::string sonar_topic = get_string("sonar/topic", SONAR_TOPIC);
    sonar_sub_ = subscribe_sonar(
      this, sonar_driver, sonar_topic, rclcpp::SensorDataQoS(),
      [this](const SonarPing& ping) {
        std::lock_guard<std::mutex> lock(mutex_);
        slam_.oculus.configure(ping);
      });

    // feature cloud (primary) + dead-reckoning odometry, approx-time synced
    sync_ = std::make_unique<
      ApproxSync2<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>>(
      20, feature_odom_sync_max_delay_,
      [this](const sensor_msgs::msg::PointCloud2& feature,
             const nav_msgs::msg::Odometry& odom) { slam_callback(feature, odom); });
    // an odom outage silently drops feature frames from the sync queue —
    // surface it instead of losing keyframes with no log line
    sync_->set_overflow_callback([this](std::size_t dropped) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "feature<->odom sync dropped %zu unmatched feature frame(s) — "
        "odometry stalled or lagging beyond feature_odom_sync_max_delay",
        dropped);
    });

    feature_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      SONAR_FEATURE_TOPIC, 20, [this](const sensor_msgs::msg::PointCloud2& msg) {
        sync_->add_primary(to_sec(msg.header.stamp), msg);
      });
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      LOCALIZATION_ODOM_TOPIC, 50, [this](const nav_msgs::msg::Odometry& msg) {
        sync_->add_secondary(to_sec(msg.header.stamp), msg);
      });

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      SLAM_POSE_TOPIC, 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(SLAM_ODOM_TOPIC, 10);
    traj_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SLAM_TRAJ_TOPIC, latched_qos());
    constraint_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      SLAM_CONSTRAINT_TOPIC, latched_qos());
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SLAM_CLOUD_TOPIC, latched_qos());

    // The graph cloud doubles as the persistent map for volatile-QoS
    // consumers (nav2's STVL global costmap): they miss the transient_local
    // sample on join, and between keyframes (a hovering vehicle) nothing
    // re-marks their voxels, so STVL's voxel_decay would fade the whole
    // costmap to empty. Re-emit the cached cloud whenever no keyframe has
    // published one within cloud_republish_period seconds (<= 0 disables).
    cloud_republish_period_ = get_double("cloud_republish_period", 5.0);
    if (cloud_republish_period_ > 0.0) {
      cloud_republish_timer_ = create_wall_timer(
        std::chrono::duration<double>(cloud_republish_period_), [this]() {
          // the cached cloud is produced by the viz worker thread — its own
          // mutex, so this timer never contends with the SLAM callback
          std::lock_guard<std::mutex> lock(viz_mutex_);
          if (last_cloud_msg_.data.empty()) return;
          if ((now() - last_cloud_pub_).seconds() < cloud_republish_period_)
            return;
          last_cloud_msg_.header.stamp = now();
          cloud_pub_->publish(last_cloud_msg_);
          last_cloud_pub_ = now();
        });
    }

    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    slam_.configure();
    // observability parity with the feature node: scan-match global init
    // batches its cost evaluation on the GPU when present (override with
    // SONAR_SLAM_FORCE_CPU=1); libpointmatcher ICP itself is CPU
    RCLCPP_INFO(get_logger(), "SLAM node is initialized (GPU: %s)",
                gpu::available() ? "on" : "off");
  }

  ~SlamNode() override
  {
    // the viz worker publishes through this node's publishers — join it
    // before they are destroyed
    if (viz_thread_.joinable()) viz_thread_.join();
  }

private:
  void slam_callback(const sensor_msgs::msg::PointCloud2& feature_msg,
                     const nav_msgs::msg::Odometry& odom_msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto time = feature_msg.header.stamp;
    const gtsam::Pose3 dr_pose3 = r2g(odom_msg.pose.pose);

    auto frame = std::make_shared<Keyframe>(false, time, dr_pose3);

    // feature cloud carries true base_link 3D (head tilt folded in by
    // feature_extraction_node.cpp publish_features). Rotate by the DR
    // roll/pitch (Ry(p)*Rx(r), yaw excluded) so stored points are
    // horizon-referenced: registration then matches world-horizontal
    // projections even when the vehicle pitches, and col2 carries elevation
    // relative to the vehicle for the 3D map cloud (FULL_3D_ROADMAP.md
    // Phase 3). A level vehicle (roll=pitch=0) reduces col0/col1 to the old
    // planar values byte-for-byte.
    // dr_pose3 arrives through enu_odom_relay's roll-pi conjugation, so its
    // euler attitude is (roll, -pitch, -yaw) of the ENU vehicle attitude and
    // its z is +depth (down-positive). Undo the pitch flip to rotate the
    // ENU-frame base_link cloud; the resulting elevation column is ENU
    // up-positive relative to the vehicle.
    const Matrix xyz = cloud_to_xyz(feature_msg);
    const Eigen::Matrix3f R_h = gtsam::Rot3::Ypr(0.0,
                                                 -dr_pose3.rotation().pitch(),
                                                 dr_pose3.rotation().roll())
                                  .matrix()
                                  .cast<float>();
    const Matrix xyz_h = xyz * R_h.transpose();
    Matrix points(xyz.rows(), 3);
    points.col(0) = xyz_h.col(0);
    points.col(1) = -xyz_h.col(1);
    points.col(2) = xyz_h.col(2);

    // NaN cloud means feature extraction skipped this frame
    if (points.rows() > 0 && std::isnan(points(0, 0)))
      frame->status = false;
    else
      frame->status = slam_.is_keyframe(*frame);

    frame->twist = odom_msg.twist.twist;

    if (!slam_.keyframes.empty()) {
      const gtsam::Pose2 dr_odom =
        slam_.current_keyframe()->dr_pose.between(frame->dr_pose);
      frame->update(slam_.current_keyframe()->pose.compose(dr_odom));
    }

    if (frame->status) {
      frame->points = points;

      // The back-end runs GTSAM/ISAM2 (and libpointmatcher ICP), which can
      // throw on a degenerate/indeterminate system. Under single-threaded
      // spin an uncaught throw escapes spin() and terminates the node; catch
      // it so one bad update skips the frame instead of killing the mission.
      try {
        if (slam_.keyframes.empty())
          slam_.add_prior(frame);
        else if (enable_slam_)
          slam_.add_sequential_scan_matching(frame);
        else
          // SLAM back-end disabled: chain plain dead-reckoning odometry factors
          slam_.add_odometry(frame);

        if (!slam_.update_factor_graph(frame)) {
          // the frame's factors were dropped and the estimator rebuilt from
          // the last good state; skip loop-closure work this callback
          RCLCPP_ERROR(get_logger(),
                       "SLAM back-end update failed (%s); keyframe dropped, "
                       "estimator rebuilt from last good state",
                       slam_.last_error().c_str());
        } else if (enable_slam_ && slam_.nssm_params.enable &&
                   slam_.add_nonsequential_scan_matching()) {
          if (!slam_.update_factor_graph())
            RCLCPP_ERROR(get_logger(),
                         "SLAM loop-closure update failed (%s); loop factors "
                         "rolled back, estimator rebuilt",
                         slam_.last_error().c_str());
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(),
                     "SLAM back-end update failed, skipping frame: %s", e.what());
      }
    }

    slam_.current_frame = frame;
    publish_all();
  }

  void publish_all()
  {
    if (slam_.keyframes.empty()) return;
    publish_pose();
    if (slam_.current_frame->status) {
      publish_trajectory();
      // The constraint markers and the aggregated map cloud re-walk the
      // ENTIRE keyframe history (O(n) per keyframe, growing for the whole
      // mission) — they are pure viz, so rebuild them at most every
      // viz_min_period seconds, and on a WORKER THREAD: on a loop-closure
      // correction the O(map) rebuild + octree downsample otherwise stalls
      // this callback for the whole sweep.
      const auto now = this->now();
      if (viz_min_period_ <= 0.0 ||
          (now - last_viz_publish_).seconds() >= viz_min_period_) {
        if (schedule_viz_rebuild()) last_viz_publish_ = now;
      }
    }
  }

  // Snapshot of the per-keyframe state the viz products need — tiny (poses +
  // loop-closure links); kf->points is immutable after keyframe creation, so
  // the worker reads the clouds through the shared_ptr without the lock.
  struct VizKeyframe
  {
    KeyframePtr kf;
    gtsam::Pose2 pose;
    double p3x, p3y, p3z, dr_z;
    std::vector<std::pair<int, gtsam::Pose2>> constraints;
  };

  // called under mutex_; returns false when a previous rebuild is still
  // running (viz is periodic — the next window retries)
  bool schedule_viz_rebuild()
  {
    if (viz_busy_.exchange(true)) return false;
    if (viz_thread_.joinable()) viz_thread_.join();  // finished; reclaim

    std::vector<VizKeyframe> snap;
    snap.reserve(slam_.keyframes.size());
    for (const auto& kf : slam_.keyframes)
      snap.push_back({kf, kf->pose, kf->pose3.x(), kf->pose3.y(),
                      kf->pose3.z(), kf->dr_pose3.z(), kf->constraints});
    const auto stamp = slam_.current_keyframe()->time;

    viz_thread_ = std::thread([this, snap = std::move(snap), stamp]() {
      build_and_publish_viz(snap, stamp);
      viz_busy_ = false;
    });
    return true;
  }

  // conjugation by the roll-pi transform: flip a z-down graph-frame pose
  // back to ENU (REP-105) for external consumers
  static void flip_pose_enu(geometry_msgs::msg::Pose& p)
  {
    p.position.y = -p.position.y;
    p.position.z = -p.position.z;
    p.orientation.y = -p.orientation.y;
    p.orientation.z = -p.orientation.z;
  }

  void publish_pose()
  {
    const auto& frame = slam_.current_frame;

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = frame->time;
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose = g2r(frame->pose3);

    // 6x6 covariance in (x, y, z, roll, pitch, yaw). The planar back-end only
    // estimates x, y and yaw; z/roll/pitch are dead-reckoning pass-through, so
    // they carry a large "unobserved" variance instead of a confident value a
    // downstream EKF would mistake for a SLAM measurement of those states.
    constexpr double kUnobserved = 1e6;
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov(2, 2) = cov(3, 3) = cov(4, 4) = kUnobserved;
    const Eigen::Matrix3d& tc = slam_.current_keyframe()->transf_cov;
    const int map_idx[3] = {0, 1, 5};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) cov(map_idx[i], map_idx[j]) = tc(i, j);

    // Between-keyframe frames are dead-reckoning extrapolations past the last
    // keyframe, but transf_cov is only that keyframe's marginal. Inflate the
    // planar covariance by the DR drift accumulated since the keyframe so
    // fusion does not over-trust the extrapolated pose (the same odometry is
    // already delivered to the EKF directly).
    if (!frame->status) {
      const gtsam::Pose2 dr =
        slam_.current_keyframe()->dr_pose.between(frame->dr_pose);
      const double d = std::hypot(dr.x(), dr.y());
      const double a = std::abs(dr.theta());
      const double kt = std::max(slam_.keyframe_translation, 1e-3);
      const double kr = std::max(slam_.keyframe_rotation, 1e-3);
      const Eigen::Vector3d& os = slam_.odom_sigmas;
      cov(0, 0) += (os[0] * d / kt) * (os[0] * d / kt);
      cov(1, 1) += (os[1] * d / kt) * (os[1] * d / kt);
      cov(5, 5) += (os[2] * a / kr) * (os[2] * a / kr);
    }

    // transf_cov is zero until the keyframe's marginals are computed, and a
    // degenerate ISAM2 marginal can be NaN — either reads as infinite
    // confidence downstream. Floor the estimated DOF (NaN-safe argument order:
    // std::max(1e-3, NaN) == 1e-3) and, if anything is still non-finite, fall
    // back to a safe diagonal.
    for (int k : {0, 1, 5}) cov(k, k) = std::max(1e-3, cov(k, k));
    if (!cov.allFinite()) {
      cov.setZero();
      cov(0, 0) = cov(1, 1) = cov(5, 5) = 1e-3;
      cov(2, 2) = cov(3, 3) = cov(4, 4) = kUnobserved;
    }
    if (enu_world_) {
      flip_pose_enu(pose_msg.pose.pose);
      // conjugate the covariance by the roll-pi sign signature
      static const double S[6] = {1, -1, -1, 1, -1, -1};
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) cov(i, j) *= S[i] * S[j];
    }
    for (int i = 0; i < 36; ++i) pose_msg.pose.covariance[i] = cov(i / 6, i % 6);
    pose_pub_->publish(pose_msg);

    // map -> odom: SLAM estimate composed with inverse dead reckoning
    if (publish_tf_) {
      const gtsam::Pose3 o2m = frame->pose3.compose(frame->dr_pose3.inverse());
      const geometry_msgs::msg::Pose o2m_msg = g2r(o2m);
      double tx = o2m_msg.position.x, ty = o2m_msg.position.y, tz = o2m_msg.position.z;
      double qx = o2m_msg.orientation.x, qy = o2m_msg.orientation.y,
             qz = o2m_msg.orientation.z, qw = o2m_msg.orientation.w;
      if (enu_world_) {
        ty = -ty; tz = -tz;
        qy = -qy; qz = -qz;
      }
      tf_->sendTransform(make_transform(Eigen::Vector3d(tx, ty, tz),
                                        Eigen::Quaterniond(qw, qx, qy, qz),
                                        frame->time, "map", "odom"));
    }

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = pose_msg.header;
    // copy the FULL PoseWithCovariance, not just the pose — otherwise the
    // covariance built above never reaches this topic and fusion reads zeros
    odom_msg.pose = pose_msg.pose;
    odom_msg.child_frame_id = "base_link";
    odom_msg.twist.twist = frame->twist;
    // twist is not estimated here (the DR/Kalman upstream leave it zero), so
    // advertise a large twist covariance; the all-zero default would read as
    // zero-velocity-with-infinite-confidence to a fusing EKF.
    for (int i = 0; i < 6; ++i) odom_msg.twist.covariance[i * 6 + i] = 1e6;
    if (enu_world_) {
      // twist arrives z-down from the relay; flip it back out
      odom_msg.twist.twist.linear.y = -odom_msg.twist.twist.linear.y;
      odom_msg.twist.twist.linear.z = -odom_msg.twist.twist.linear.z;
      odom_msg.twist.twist.angular.y = -odom_msg.twist.twist.angular.y;
      odom_msg.twist.twist.angular.z = -odom_msg.twist.twist.angular.z;
    }
    odom_pub_->publish(odom_msg);

    // periodic health counters — publish_pose runs per callback (~ping
    // rate), so gate on the keyframe count CHANGING or the same line
    // repeats hundreds of times while the count sits on a multiple of 25
    if (slam_.current_key() % 25 == 0 && slam_.current_key() != last_logged_key_) {
      last_logged_key_ = slam_.current_key();
      RCLCPP_INFO(get_logger(),
                  "SLAM status: keyframes %d (last kf %ld pts), SSM factors %d, "
                  "NSSM accepted %d, last SSM: %s | NSSM attempts %d, queued %d, "
                  "best clique %d, queue depth %d, PCM min md %.1f (thresh 11.3, "
                  "%d edges), last NSSM: %s",
                  slam_.current_key(),
                  static_cast<long>(slam_.current_keyframe()->points.rows()),
                  slam_.ssm_accepted, slam_.nssm_accepted,
                  slam_.last_ssm_status.c_str(), slam_.nssm_attempts,
                  slam_.nssm_queued, slam_.nssm_best_clique,
                  slam_.nssm_queue_depth(), slam_.last_pcm_min_md,
                  slam_.last_pcm_edges, slam_.last_nssm_status.c_str());
    }
  }

  // Worker-thread body: builds and publishes the constraint markers and the
  // aggregated map cloud from the snapshot — no slam_ / mutex_ access.
  // rclcpp publishers are thread-safe; the cached-cloud fields for the
  // republish timer live under viz_mutex_.
  void build_and_publish_viz(const std::vector<VizKeyframe>& snap,
                             const builtin_interfaces::msg::Time& stamp)
  {
    // ---- constraint markers (viz follows the ENU convention when set) ----
    const double sy = enu_world_ ? -1.0 : 1.0;
    auto P = [&](double x, double y, double z) {
      return Eigen::Vector3d(x, sy * y, sy * z);
    };
    std::vector<ConstraintLink> links;
    for (std::size_t x = 1; x < snap.size(); ++x) {
      const auto& prev = snap[x - 1];
      const auto& curr = snap[x];
      const Eigen::Vector3d p1 = P(prev.p3x, prev.p3y, prev.dr_z);
      const Eigen::Vector3d p2 = P(curr.p3x, curr.p3y, curr.dr_z);
      links.push_back({{p1, p2}, "green"});

      for (const auto& [k, _] : curr.constraints) {
        if (k < 0 || k >= static_cast<int>(snap.size())) continue;
        const auto& target = snap[k];
        const Eigen::Vector3d p0 = P(target.p3x, target.p3y, target.dr_z);
        links.push_back({{p0, p2}, "red"});
      }
    }
    if (!links.empty()) {
      visualization_msgs::msg::Marker msg = ros_constraints(links);
      msg.header.stamp = stamp;
      constraint_pub_->publish(msg);
    }

    // ---- aggregated map cloud ----
    long total = 0;
    for (const auto& s : snap) total += s.kf->points.rows();
    Matrix all_points(total, 3), all_keys(total, 1);
    long at = 0;
    for (std::size_t key = 0; key < snap.size(); ++key) {
      const auto& s = snap[key];
      const long n = s.kf->points.rows();
      // x/y through the optimized SE2 pose; z in ENU world = the point's
      // up-positive elevation (col2) minus vehicle depth (pose3.z carries the
      // relay's down-positive z across updates). The graph corrects x/y/yaw;
      // z and attitude ride from dead-reckoning.
      all_points.middleRows(at, n).leftCols(2) =
        Keyframe::transform_points(s.kf->points, s.pose);
      all_points.middleRows(at, n).col(2) =
        s.kf->points.col(2).array() - static_cast<float>(s.p3z);
      all_keys.middleRows(at, n).setConstant(static_cast<float>(key));
      at += n;
    }

    // 3D octree downsample (OctreeGridDataPointsFilter is dimension-agnostic)
    auto [pts, keys] =
      downsample(all_points, all_keys, static_cast<float>(slam_.point_resolution));
    if (pts.rows() == 0) return;

    // [x, y, z, key] — col2 was built ENU (up-positive); the non-ENU (slam
    // z-down) output flips it along with y, completing the roll-pi transform
    Matrix xyzi(pts.rows(), 4);
    xyzi.col(0) = pts.col(0);
    xyzi.col(1) = enu_world_ ? Matrix(-pts.col(1)) : Matrix(pts.col(1));
    xyzi.col(2) = enu_world_ ? Matrix(pts.col(2)) : Matrix(-pts.col(2));
    xyzi.col(3) = keys.col(0);

    sensor_msgs::msg::PointCloud2 msg = make_cloud({"x", "y", "z", "i"}, xyzi);
    msg.header.stamp = stamp;
    msg.header.frame_id = "map";
    cloud_pub_->publish(msg);
    // cache for the periodic republish timer (see constructor)
    std::lock_guard<std::mutex> lock(viz_mutex_);
    last_cloud_msg_ = std::move(msg);
    last_cloud_pub_ = now();
  }

  void publish_trajectory()
  {
    // [x, y, z, roll, pitch, yaw, index]
    const float sy = enu_world_ ? -1.0f : 1.0f;
    // `t` is the keyframe stamp RELATIVE to this message's stamp (float32
    // seconds since epoch cannot hold ms precision) — consumers recover the
    // absolute stamp as msg.stamp + t. Lets the map assembler associate
    // evidence clouds (same ping stamps) to keyframes.
    const auto& msg_stamp = slam_.current_keyframe()->time;
    const double msg_stamp_s = to_sec(msg_stamp);
    Matrix traj(slam_.current_key(), 8);
    for (int k = 0; k < slam_.current_key(); ++k) {
      const auto& kf = slam_.keyframes[k];
      traj(k, 0) = static_cast<float>(kf->pose3.x());
      traj(k, 1) = sy * static_cast<float>(kf->pose3.y());
      traj(k, 2) = sy * static_cast<float>(kf->pose3.z());
      traj(k, 3) = static_cast<float>(kf->pose3.rotation().roll());
      traj(k, 4) = sy * static_cast<float>(kf->pose3.rotation().pitch());
      traj(k, 5) = sy * static_cast<float>(kf->pose3.rotation().yaw());
      traj(k, 6) = static_cast<float>(k);
      traj(k, 7) = static_cast<float>(to_sec(kf->time) - msg_stamp_s);
    }
    sensor_msgs::msg::PointCloud2 msg =
      make_cloud({"x", "y", "z", "roll", "pitch", "yaw", "i", "t"}, traj);
    msg.header.stamp = msg_stamp;
    msg.header.frame_id = "map";
    traj_pub_->publish(msg);
  }

  Slam slam_;
  std::mutex mutex_;
  // background viz rebuild (see schedule_viz_rebuild); viz_mutex_ guards the
  // cached republish cloud shared between the worker and the timer
  std::thread viz_thread_;
  std::atomic<bool> viz_busy_{false};
  std::mutex viz_mutex_;
  bool enable_slam_ = true;
  bool enu_world_ = false;
  bool publish_tf_ = true;
  double viz_min_period_ = 2.0;
  int last_logged_key_ = -1;
  rclcpp::Time last_viz_publish_{0, 0, RCL_ROS_TIME};
  double feature_odom_sync_max_delay_ = 0.5;
  double cloud_republish_period_ = 5.0;
  sensor_msgs::msg::PointCloud2 last_cloud_msg_;
  rclcpp::Time last_cloud_pub_{0, 0, RCL_ROS_TIME};
  rclcpp::TimerBase::SharedPtr cloud_republish_timer_;

  rclcpp::SubscriptionBase::SharedPtr sonar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr feature_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::unique_ptr<ApproxSync2<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>>
    sync_;

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr traj_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr constraint_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::SlamNode>());
  rclcpp::shutdown();
  return 0;
}
