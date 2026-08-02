// Sonar pose-graph node. sonar_proc supplies one flash-ping candidate cloud;
// exact-stamp TF supplies the sensor extrinsic and fused odometry pose.
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/exceptions.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "sonar_slam_cpp/gpu.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sonar_input.hpp"
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
    if (has_parameter("keyframe_min_points"))
      throw std::runtime_error(
        "keyframe_min_points is obsolete: configure admission.min_points, "
        "admission.min_occupied_cells and admission.min_azimuth_bins instead");

    enable_slam_ = get_bool("enable_slam");
    RCLCPP_INFO(get_logger(), "SLAM STATUS: %s", enable_slam_ ? "true" : "false");

    // false when an external node (e.g. the map robot_localization EKF
    // fusing /bruce/slam/slam/pose) owns the map->odom transform
    publish_tf_ = get_bool("publish_tf", true);
    // min seconds between rebuilds of the O(history) constraint markers;
    // <= 0 -> every keyframe
    viz_min_period_ = get_double("viz_min_period", 2.0);

    const auto prior = get_double_array("prior_sigmas");
    const auto odom = get_double_array("odom_sigmas");
    const auto icp_odom = get_double_array("icp_odom_sigmas");
    slam_.prior_sigmas = Eigen::Vector3d(prior[0], prior[1], prior[2]);
    slam_.odom_sigmas = Eigen::Vector3d(odom[0], odom[1], odom[2]);
    slam_.icp_odom_sigmas = Eigen::Vector3d(icp_odom[0], icp_odom[1], icp_odom[2]);

    slam_.point_resolution = get_double("point_resolution");
    slam_.point_noise = get_double("point_noise", 0.5);
    // The global-init cost grid divides its resolution and dilation radius
    // by point_noise; zero/negative is a config error, not a tunable.
    if (!(slam_.point_noise > 0.0))
      throw std::runtime_error("point_noise must be > 0");
    slam_.sonar_max_range = get_double("sonar_max_range", 30.0);
    slam_.sonar_horizontal_aperture =
      get_double("sonar_horizontal_aperture", 2.2689280275926285);
    points_topic_ = get_string("points_topic", SONAR_POINTS_TOPIC);
    odom_frame_ = get_string("odom_frame", "odom");
    base_frame_ = get_string("base_frame", "base_link");
    tf_lookup_timeout_ = get_double("tf_lookup_timeout", 0.05);
    max_head_pitch_ = get_double("max_head_pitch", 0.52);

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
    // FAST-MCD, the default) or "censi" (one ICP + the point-to-point closed
    // form). Slam::configure() validates "censi" against the LOADED ICP chain
    // and throws a naming error if the objectives do not match.
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
    slam_.ssm_params.min_overlap_ratio =
      get_double("ssm/min_overlap_ratio", 0.0);
    slam_.ssm_params.max_translation = get_double("ssm/max_translation");
    slam_.ssm_params.max_rotation = get_double("ssm/max_rotation");
    slam_.ssm_params.target_frames = get_int("ssm/target_frames");
    // covariance-estimating ICP for SSM (core supported it but the config
    // never reached it — in the python original these fields were unwired,
    // which left SSM on ICP with a fixed noise model; that poisoned the graph
    // on pool geometry)
    slam_.ssm_params.initialization = get_bool("ssm/initialization", true);
    const auto ssm_init = init_params("ssm/initialization_params", {50.0, 1.0, 0.01});
    slam_.ssm_params.init_n = static_cast<int>(ssm_init[0]);
    slam_.ssm_params.init_iters = static_cast<int>(ssm_init[1]);
    slam_.ssm_params.init_ftol = ssm_init[2];
    slam_.ssm_params.cov_samples = get_int("ssm/cov_samples", 0);
    slam_.ssm_params.cov_method = cov_method("ssm/cov_method");
    slam_.ssm_require_covariance =
      get_bool("ssm/require_covariance", true);
    slam_.ssm_max_sigma = get_double("ssm/max_sigma", 0.5);
    slam_.ssm_max_anisotropy = get_double("ssm/max_anisotropy", 8.0);
    slam_.ssm_degeneracy_prefloor =
      get_bool("ssm/degeneracy_prefloor", true);

    // Open-water admission happens before keyframe selection, including the
    // first frame. Keep the count at least as strict as SSM so a cloud too
    // sparse to make a sonar factor cannot pollute graph/NSSM history.
    admission_params_.min_points =
      get_int("admission/min_points", slam_.ssm_params.min_points);
    admission_params_.cell_size =
      get_double("admission/cell_size", 0.5);
    admission_params_.min_occupied_cells =
      get_int("admission/min_occupied_cells", 8);
    admission_params_.azimuth_bin_size =
      get_double("admission/azimuth_bin_size", 0.08726646259971647);
    admission_params_.min_azimuth_bins =
      get_int("admission/min_azimuth_bins", 3);
    if (admission_params_.min_points < slam_.ssm_params.min_points)
      throw std::runtime_error(
        "admission.min_points must be >= ssm.min_points");
    if (!(admission_params_.cell_size > 0.0) ||
        admission_params_.min_occupied_cells < 1 ||
        !(admission_params_.azimuth_bin_size > 0.0) ||
        admission_params_.min_azimuth_bins < 1)
      throw std::runtime_error(
        "admission cell/bin sizes must be positive and minimum counts >= 1");

    slam_.nssm_params.enable = get_bool("nssm/enable");
    slam_.nssm_params.min_st_sep = get_int("nssm/min_st_sep");
    slam_.nssm_params.min_revisit_sep = get_int("nssm/min_revisit_sep", 10);
    slam_.nssm_params.min_points = get_int("nssm/min_points");
    slam_.nssm_params.min_overlap_ratio =
      get_double("nssm/min_overlap_ratio", 0.0);
    slam_.nssm_params.max_translation = get_double("nssm/max_translation");
    slam_.nssm_params.max_rotation = get_double("nssm/max_rotation");
    slam_.nssm_params.source_frames = get_int("nssm/source_frames");
    slam_.nssm_params.fan_drift_trans =
      get_double("nssm/fan_drift_trans", 0.01);
    slam_.nssm_params.fan_drift_rot = get_double("nssm/fan_drift_rot", 0.0017);
    slam_.nssm_params.initialization = get_bool("nssm/initialization", true);
    const auto nssm_init =
      init_params("nssm/initialization_params", {100.0, 5.0, 0.01});
    slam_.nssm_params.init_n = static_cast<int>(nssm_init[0]);
    slam_.nssm_params.init_iters = static_cast<int>(nssm_init[1]);
    slam_.nssm_params.init_ftol = nssm_init[2];
    slam_.nssm_params.cov_samples = get_int("nssm/cov_samples");
    slam_.nssm_params.cov_method = cov_method("nssm/cov_method");

    // min_st_sep builds the NSSM candidate pool (k < current_key -
    // min_st_sep); min_revisit_sep then filters it (source_key - f >=
    // min_revisit_sep). Since source_key == current_key - 1, the pool admits
    // gaps >= min_st_sep while the revisit floor demands >= min_revisit_sep,
    // so once min_revisit_sep >= min_st_sep the exclusion zone contributes
    // NOTHING to target selection and tuning it has no effect. That is the
    // deployed state (8 vs 10) — and venue/field.yaml lists min_st_sep as a
    // tuning candidate it cannot actually influence. Warn rather than throw:
    // min_st_sep still gates when NSSM starts at all, and still bounds
    // source_frames, so the configuration is legal, just partly inert.
    if (slam_.nssm_params.min_revisit_sep >= slam_.nssm_params.min_st_sep)
      RCLCPP_WARN(
        get_logger(),
        "nssm/min_st_sep (%d) does not affect loop-closure target selection: "
        "nssm/min_revisit_sep (%d) is >= it, so the revisit floor already "
        "excludes everything the exclusion zone would have. Tuning min_st_sep "
        "will change only when NSSM starts. Raise it above min_revisit_sep to "
        "make it bind on target selection.",
        slam_.nssm_params.min_st_sep, slam_.nssm_params.min_revisit_sep);

    slam_.pcm_queue_size = get_int("pcm_queue_size");
    slam_.min_pcm = get_int("min_pcm");
    // NSSM degeneracy gate (see slam_core.hpp)
    slam_.nssm_max_sigma = get_double("nssm/max_sigma", 0.5);
    slam_.nssm_max_anisotropy = get_double("nssm/max_anisotropy", 8.0);
    slam_.nssm_degeneracy_prefloor = get_bool("nssm/degeneracy_prefloor", false);
    slam_.nssm_max_yaw_vs_compass = get_double("nssm/max_yaw_vs_compass", 0.15);
    slam_.nssm_max_translation_vs_dr =
      get_double("nssm/max_translation_vs_dr", 0.0);
    slam_.post_loop_max_yaw_rms = get_double("post_loop_max_yaw_rms", 0.15);
    slam_.post_loop_max_link_error_sigma =
      get_double("post_loop_max_link_error_sigma", 3.0);
    // The chain-tear bound changed UNITS on 2026-07-25 (metres -> sigma of
    // the link's own noise model). Silently ignoring a stale key would leave
    // a deployment running the 3.0 default while its YAML says 1.0 and its
    // operator believes the config is in force — so say so, loudly, once.
    if (has_parameter("post_loop_max_translation_err"))
      RCLCPP_ERROR(
        get_logger(),
        "'post_loop_max_translation_err' is set but NO LONGER READ: the "
        "post-loop chain-tear bound is now 'post_loop_max_link_error_sigma' "
        "and is expressed in SIGMA of each link's own noise model, not "
        "metres (rtabmap RGBD/OptimizeMaxError; its default is 3.0). "
        "Running with post_loop_max_link_error_sigma=%.2f. Remove the old "
        "key once the new bound is tuned.",
        slam_.post_loop_max_link_error_sigma);
    // loop-factor robust kernel (default OFF — see slam_core.hpp: DCS muted
    // exactly the meaningful corrections; verify+revert is the protection)
    slam_.nssm_use_dcs = get_bool("nssm/use_dcs", false);
    slam_.nssm_dcs_phi = get_double("nssm/dcs_phi", 1.0);
    slam_.loop_extra_iterations = get_int("loop_extra_iterations", 3);
    // operator hand-correction trust [x m, y m, yaw rad] — yaw soft by
    // default (see slam_core.hpp: the verify stack assumes compass yaw)
    const auto mc_sigmas =
      get_double_array("manual_correction_sigmas", {0.2, 0.2, 0.5});
    if (mc_sigmas.size() == 3)
      slam_.manual_correction_sigmas =
        Eigen::Vector3d(mc_sigmas[0], mc_sigmas[1], mc_sigmas[2]);
    else
      RCLCPP_ERROR(get_logger(),
                   "manual_correction_sigmas must be [x, y, yaw]; keeping "
                   "defaults [0.2, 0.2, 0.5]");

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

    const int input_queue_depth =
      std::max(1, get_int("input_queue_depth", 50));
    const int output_queue_depth =
      std::max(1, get_int("output_queue_depth", 10));
    // Best-effort by default: live prioritizes latency and the sonar_proc
    // publisher is best-effort there anyway. The replay overlay raises
    // input_queue_depth expecting a LOSSLESS stream, which best-effort cannot
    // promise even against a reliable publisher — so it also sets
    // input_reliable true (sonar_proc's replay overlay makes the producer
    // reliable), which is what actually makes the deep queue deliver.
    auto input_qos = rclcpp::SensorDataQoS(
      rclcpp::KeepLast(static_cast<std::size_t>(input_queue_depth)));
    if (get_bool("input_reliable", false)) input_qos.reliable();
    const auto output_qos = rclcpp::QoS(
      rclcpp::KeepLast(static_cast<std::size_t>(output_queue_depth)));
    // Oculus is a flash-imaging sonar: a cloud is one rigid ping, not a
    // spinning scan, so per-point "deskew" is fictitious. Resolve both the
    // sensor extrinsic and odom pose at the cloud's acquisition stamp instead
    // of pairing it to a nearest odometry message with a 0.5 s slop.
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ =
      std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, true);
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic_, input_qos,
      [this](const sensor_msgs::msg::PointCloud2& msg) {
        points_callback(msg);
      });

    // Operator hand-correction: RViz's "2D Pose Estimate" button publishes a
    // map-frame planar pose on /initialpose — applied as a prior on the
    // newest keyframe (see manual_correction_callback). Empty topic disables.
    const std::string mc_topic =
      get_string("manual_correction_topic", "/initialpose");
    if (!mc_topic.empty()) {
      manual_correction_sub_ =
        create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          mc_topic, 1,
          [this](const geometry_msgs::msg::PoseWithCovarianceStamped& msg) {
            manual_correction_callback(msg);
          });
      // paired undo (stack semantics — call repeatedly to peel corrections):
      //   ros2 service call /bruce/slam/slam/undo_manual_correction std_srvs/srv/Trigger
      undo_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/undo_manual_correction",
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          undo_manual_correction(*res);
        });
    }

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      SLAM_POSE_TOPIC, output_qos);
    odom_pub_ =
      create_publisher<nav_msgs::msg::Odometry>(SLAM_ODOM_TOPIC, output_qos);
    // Depth 1 deliberately, decoupled from output_queue_depth: each
    // trajectory message carries the FULL keyframe history and fully
    // supersedes its predecessor, so a late joiner needs exactly one. A
    // deeper transient-local writer only retains stale snapshots — under the
    // replay overlay's output_queue_depth 256 it dumped 256 full-trajectory
    // clouds to every late-joining assembler.
    traj_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      SLAM_TRAJ_TOPIC, latched_qos(1));
    constraint_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      SLAM_CONSTRAINT_TOPIC, latched_qos());

    // Mirror the graph funnel and exact-stamp input health on /diagnostics so
    // TF starvation and the NSSM accept/reject balance are visible in rqt.
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", 10);
    diag_timer_ = create_timer(std::chrono::seconds(1),
                               [this]() { publish_diagnostics(); });

    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // map->odom republish timer (rtabmap tf_delay / tf_tolerance). Decouples
    // the edge from the sensor callback so a sonar or input-TF dropout
    // no longer starves every map-frame consumer, and future-dates it to
    // match the odom EKF's transform_time_offset. <= 0 restores the old
    // publish-inline-from-the-callback behaviour.
    tf_publish_period_ = get_double("tf_publish_period", 0.05);  // 20 Hz
    tf_tolerance_ = get_double("tf_tolerance", 0.1);
    if (publish_tf_ && tf_publish_period_ > 0.0)
      tf_timer_ = create_timer(
        std::chrono::duration<double>(tf_publish_period_),
        [this]() { republish_map_odom(); });

    slam_.configure();

    // Map persistence: ~/save_map serializes the whole keyframe map;
    // map_load_path restores a previous session at startup and arms
    // relocalization (the first dense frame is globally scan-matched
    // against the loaded map before normal operation resumes).
    map_save_path_ = get_string("map_save_path", "/tmp/sonar_slam_map.ssm");
    save_map_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/save_map",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        save_map(*res);
      });
    slam_.relocalize_min_overlap = get_double("relocalize_min_overlap", 0.5);
    const std::string load_path = get_string("map_load_path", "");
    if (!load_path.empty()) {
      if (slam_.load_map(load_path))
        RCLCPP_INFO(get_logger(),
                    "loaded %d keyframes from '%s' — relocalizing on the "
                    "first dense candidate frame",
                    slam_.loaded_keyframes(), load_path.c_str());
      else
        RCLCPP_ERROR(get_logger(),
                     "map_load_path '%s' failed to load (%s); starting fresh",
                     load_path.c_str(), slam_.last_error().c_str());
    }

    // USBL/LBL absolute position input: map-frame position fixes become
    // gated, stamp-matched priors on the nearest keyframe (position-only —
    // acoustic positioning has no heading). Empty topic disables.
    const std::string usbl_driver = get_string("usbl/driver", "pose_cov");
    const std::string usbl_topic = get_string("usbl/topic", "");
    usbl_max_innovation_ = get_double("usbl/max_innovation", 10.0);
    usbl_min_sigma_ = get_double("usbl/min_sigma", 0.5);
    usbl_max_stamp_delta_ = get_double("usbl/max_stamp_delta", 1.0);
    usbl_frame_id_ = get_string("usbl/frame_id", "");
    if (!usbl_topic.empty()) {
      if (usbl_driver == "pose_cov") {
        usbl_sub_ = create_subscription<
          geometry_msgs::msg::PoseWithCovarianceStamped>(
          usbl_topic, rclcpp::SensorDataQoS(),
          [this](const geometry_msgs::msg::PoseWithCovarianceStamped& m) {
            usbl_callback(m.header.stamp, m.header.frame_id,
                          m.pose.pose.position.x, m.pose.pose.position.y,
                          m.pose.covariance[0], m.pose.covariance[7]);
          });
      } else if (usbl_driver == "odom") {
        usbl_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          usbl_topic, rclcpp::SensorDataQoS(),
          [this](const nav_msgs::msg::Odometry& m) {
            usbl_callback(m.header.stamp, m.header.frame_id,
                          m.pose.pose.position.x, m.pose.pose.position.y,
                          m.pose.covariance[0], m.pose.covariance[7]);
          });
      } else {
        RCLCPP_ERROR(get_logger(),
                     "usbl/driver '%s' unknown (pose_cov | odom) — USBL "
                     "input disabled",
                     usbl_driver.c_str());
      }
    }

    // Scan-match global initialization batches its cost evaluation on the GPU
    // when present (override with
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
  // Update the cached map->odom. With tf_publish_period_ > 0 the timer is the
  // sole publisher (see MapOdomTf); otherwise broadcast inline exactly as
  // before, so the timer can be disabled without changing behaviour.
  void set_map_odom(const Eigen::Vector3d& t, const Eigen::Quaterniond& q,
                    const builtin_interfaces::msg::Time& data_stamp)
  {
    if (tf_publish_period_ <= 0.0) {
      tf_->sendTransform(make_transform(t, q, data_stamp, "map", odom_frame_));
      return;
    }
    std::lock_guard<std::mutex> lock(tf_mutex_);
    map_odom_tf_ = {true, t, q, rclcpp::Time(data_stamp), now()};
  }

  // Timer body: re-broadcast the cached correction, future-dated.
  //
  // Stamping is done in the DATA clock domain, not the node's. The robot
  // stack sets use_sim_time = is_sim or is_bag (bringup_robot.launch.xml),
  // but a standalone/harness replay can run this node on wall time while
  // frames carry bag time — stamping with now() directly would then inject
  // transforms decades away from the rest of TF.
  // Advancing the last data stamp by the node-clock interval since it was
  // cached is correct in BOTH domains: under sim time the two clocks are the
  // same and this reduces to now() + tolerance (rtabmap's form), and under
  // wall time a bag playing at ~1x advances data time at ~1x real time.
  //
  // map->odom is a slowly-varying correction, not motion, so holding it
  // constant and extrapolating the STAMP forward is safe between updates —
  // which is the whole reason rtabmap can run this at a fixed 20 Hz.
  void republish_map_odom()
  {
    MapOdomTf c;
    {
      std::lock_guard<std::mutex> lock(tf_mutex_);
      if (!map_odom_tf_.valid) return;
      c = map_odom_tf_;
    }
    const double elapsed = (now() - c.cached_at).seconds();
    // negative = node clock jumped backwards (bag loop): fall back to the
    // cached stamp rather than emitting a transform in the past
    const rclcpp::Time stamp =
      c.data_stamp +
      rclcpp::Duration::from_seconds((elapsed > 0.0 ? elapsed : 0.0) +
                                     tf_tolerance_);
    tf_->sendTransform(make_transform(c.t, c.q, stamp, "map", odom_frame_));
  }

  static gtsam::Pose3 pose_from_transform(
    const geometry_msgs::msg::TransformStamped& tf)
  {
    const auto& t = tf.transform.translation;
    const auto& q = tf.transform.rotation;
    return gtsam::Pose3(gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
                        gtsam::Point3(t.x, t.y, t.z));
  }

  void points_callback(const sensor_msgs::msg::PointCloud2& msg)
  {
    if (msg.header.frame_id.empty()) {
      ++tf_lookup_failures_;
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Dropping sonar points with an empty frame_id");
      return;
    }

    geometry_msgs::msg::TransformStamped base_sensor;
    geometry_msgs::msg::TransformStamped odom_base;
    try {
      const auto stamp = rclcpp::Time(msg.header.stamp);
      // ONE deadline shared by both lookups: under single-threaded spin these
      // block the executor, and two full budgets back to back doubled the
      // worst-case stall per ping (0.4 s at the replay overlay's 0.20 s).
      // The second lookup gets whatever the first left unspent — both
      // transforms describe the same instant, so once the buffer has caught
      // up to the stamp the second lookup is typically immediate.
      const auto lookup_start = std::chrono::steady_clock::now();
      const auto timeout = rclcpp::Duration::from_seconds(tf_lookup_timeout_);
      base_sensor =
        tf_buffer_->lookupTransform(base_frame_, msg.header.frame_id, stamp, timeout);
      const double spent =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      lookup_start)
          .count();
      const auto remaining = rclcpp::Duration::from_seconds(
        std::max(0.0, tf_lookup_timeout_ - spent));
      odom_base =
        tf_buffer_->lookupTransform(odom_frame_, base_frame_, stamp, remaining);
    } catch (const tf2::TransformException& e) {
      ++tf_lookup_failures_;
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Dropping sonar ping at %.6f: exact-stamp TF unavailable (%s <- %s "
        "and %s <- %s): %s",
        to_sec(msg.header.stamp), base_frame_.c_str(),
        msg.header.frame_id.c_str(), odom_frame_.c_str(),
        base_frame_.c_str(), e.what());
      return;
    }

    const auto& q_msg = base_sensor.transform.rotation;
    Eigen::Quaternionf q_bs(
      static_cast<float>(q_msg.w), static_cast<float>(q_msg.x),
      static_cast<float>(q_msg.y), static_cast<float>(q_msg.z));
    if (!(q_bs.norm() > 1e-6f)) {
      ++tf_lookup_failures_;
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Dropping sonar ping: %s <- %s has an invalid rotation",
        base_frame_.c_str(), msg.header.frame_id.c_str());
      return;
    }
    q_bs.normalize();
    const Eigen::Matrix3f R_bs = q_bs.toRotationMatrix();

    // Optical +Z is the sonar boresight. Its elevation in base_link gives the
    // head gate without another joint-state subscription or arrival-time latch.
    const double head_pitch = optical_boresight_pitch(R_bs);
    const bool scan_usable =
      max_head_pitch_ <= 0.0 || std::fabs(head_pitch) <= max_head_pitch_;
    if (!scan_usable) {
      ++pitch_rejections_;
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Planar SLAM is skipping head-swept ping (boresight %.1f deg, limit "
        "%.1f deg); sonar_proc mapping streams remain unaffected",
        head_pitch * 180.0 / M_PI, max_head_pitch_ * 180.0 / M_PI);
    }

    Matrix points;
    if (scan_usable) {
      const Matrix xyz_sensor = cloud_to_xyz(msg);
      const auto& tr = base_sensor.transform.translation;
      const Eigen::RowVector3f t_bs(
        static_cast<float>(tr.x), static_cast<float>(tr.y),
        static_cast<float>(tr.z));
      // Store returns in the gravity-horizontal vehicle frame. This is a
      // single rigid ping transform, not per-point deskew.
      const gtsam::Pose3 dr_pose3 = pose_from_transform(odom_base);
      const Eigen::Matrix3f R_h =
        gtsam::Rot3::Ypr(0.0, dr_pose3.rotation().pitch(),
                        dr_pose3.rotation().roll())
          .matrix().cast<float>();
      points = finite_points(
        project_flash_ping(xyz_sensor, R_bs, t_bs, R_h));
      slam_callback(msg.header.stamp, dr_pose3, points, true);
      return;
    }

    points.resize(0, 3);
    slam_callback(msg.header.stamp, pose_from_transform(odom_base), points, false);
  }

  void slam_callback(const builtin_interfaces::msg::Time& time,
                     const gtsam::Pose3& dr_pose3, const Matrix& points,
                     bool scan_usable)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto frame = std::make_shared<Keyframe>(false, time, dr_pose3);
    ScanAdmission admission;
    bool scan_informative = false;
    if (scan_usable) {
      admission = assess_scan(points, admission_params_);
      scan_informative = admission.informative;
      last_admission_summary_ =
        std::string(scan_admission_reason(admission.reason)) +
        " (points " + std::to_string(admission.finite_points) +
        ", cells " + std::to_string(admission.occupied_cells) +
        ", azimuth bins " +
        std::to_string(admission.occupied_azimuth_bins) + ")";
      if (scan_informative) {
        ++admitted_scans_;
        last_input_mode_ = "sonar_eligible";
      } else {
        ++sparse_scan_rejections_;
        if (admission.reason == ScanAdmissionReason::NOT_ENOUGH_POINTS)
          ++sparse_point_rejections_;
        else if (admission.reason == ScanAdmissionReason::NOT_ENOUGH_CELLS)
          ++sparse_cell_rejections_;
        else if (admission.reason == ScanAdmissionReason::NOT_ENOUGH_AZIMUTH)
          ++sparse_azimuth_rejections_;
        last_input_mode_ = "odometry_only_sparse";
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 10000,
          "Sonar ping rejected as graph evidence: %s; publishing fused-odometry "
          "extrapolation only",
          last_admission_summary_.c_str());
      }
    } else {
      last_input_mode_ = "odometry_only_head_sweep";
      last_admission_summary_ = "head pitch outside planar SLAM gate";
    }

    // A loaded map intercepts the pipeline until relocalization lands: the
    // DR chain of this session has no relation to the map frame yet, so no
    // factor may enter and nothing may publish until the global scan match
    // places the vehicle.
    if (slam_.awaiting_relocalization()) {
      if (scan_informative &&
          points.rows() >= slam_.nssm_params.min_points &&
          points.rows() > 0) {
        frame->status = true;
        frame->points = points;
        bool ok = false;
        try {
          ok = slam_.relocalize(frame);
        } catch (const std::exception& e) {
          RCLCPP_ERROR(get_logger(), "relocalization attempt threw: %s",
                       e.what());
        }
        if (ok) {
          RCLCPP_INFO(get_logger(),
                      "relocalized against loaded map at (%.2f, %.2f, "
                      "%.1f deg) — resuming SLAM",
                      frame->pose.x(), frame->pose.y(),
                      frame->pose.theta() * 180.0 / M_PI);
          slam_.current_frame = frame;
          publish_all();
          return;
        }
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "waiting to relocalize against the loaded map (%d keyframes): %s",
        slam_.loaded_keyframes(), slam_.last_error().c_str());
      return;
    }

    frame->status = scan_informative && slam_.is_keyframe(*frame);

    // Open water or a head sweep can precede the first informative keyframe.
    // Keep map->odom connected and publish the fused-odometry pose, but do not
    // insert a graph state or retain the cloud as future NSSM evidence.
    if (!frame->status && slam_.keyframes.empty() && publish_tf_) {
      set_map_odom(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(),
                   frame->time);
      RCLCPP_INFO_ONCE(
        get_logger(),
        "Publishing identity map->odom while waiting for the first usable "
        "SLAM candidate frame.");
    }

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
          // per-closure geometry, logged before the update so it precedes any
          // post-loop revert message (diagnoses legit-fix vs parallel-wall alias)
          for (const auto& g : slam_.last_nssm_inserted_geom)
            RCLCPP_INFO(get_logger(), "NSSM %s", g.c_str());
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
    if (!slam_.current_frame) return;
    publish_pose();
    if (slam_.keyframes.empty()) return;
    if (slam_.current_frame->status) {
      publish_trajectory();
      // Constraint markers re-walk the graph history, so rate-limit and build
      // them off the sensor callback.
      const auto now = this->now();
      const double since = (now - last_viz_publish_).seconds();
      // since < 0 = clock jumped backwards (bag loop): rebuild rather than
      // suppress viz for the whole looped pass
      if (viz_min_period_ <= 0.0 || since < 0.0 || since >= viz_min_period_) {
        if (schedule_viz_rebuild()) last_viz_publish_ = now;
      }
    }
  }

  // Snapshot of the per-keyframe state the constraint visualization needs.
  struct VizKeyframe
  {
    gtsam::Pose2 pose;
    double z;
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
      snap.push_back({kf->pose, kf->pose3.z(), kf->constraints});
    const auto stamp = slam_.current_keyframe()->time;

    viz_thread_ = std::thread([this, snap = std::move(snap), stamp]() {
      build_and_publish_viz(snap, stamp);
      viz_busy_ = false;
    });
    return true;
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
    Eigen::Matrix3d tc = Eigen::Matrix3d::Zero();
    if (slam_.keyframes.empty()) {
      // No informative sonar frame yet: map == odom and this output is the
      // fused-odometry extrapolation. TF carries no covariance, so use the
      // configured DR link noise rather than claiming scan-match confidence.
      tc = slam_.odom_sigmas.array().square().matrix().asDiagonal();
    } else {
      tc = slam_.current_keyframe()->transf_cov;
    }
    const int map_idx[3] = {0, 1, 5};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) cov(map_idx[i], map_idx[j]) = tc(i, j);

    // Between-keyframe frames are dead-reckoning extrapolations past the last
    // keyframe, but transf_cov is only that keyframe's marginal. Inflate the
    // planar covariance by the DR drift accumulated since the keyframe so
    // fusion does not over-trust the extrapolated pose (the same odometry is
    // already delivered to the EKF directly).
    if (!frame->status && !slam_.keyframes.empty()) {
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
    for (int i = 0; i < 36; ++i) pose_msg.pose.covariance[i] = cov(i / 6, i % 6);
    pose_pub_->publish(pose_msg);

    // map -> odom: SLAM estimate composed with inverse dead reckoning
    if (publish_tf_) {
      const gtsam::Pose3 o2m = frame->pose3.compose(frame->dr_pose3.inverse());
      const geometry_msgs::msg::Pose o2m_msg = g2r(o2m);
      double tx = o2m_msg.position.x, ty = o2m_msg.position.y, tz = o2m_msg.position.z;
      double qx = o2m_msg.orientation.x, qy = o2m_msg.orientation.y,
             qz = o2m_msg.orientation.z, qw = o2m_msg.orientation.w;
      set_map_odom(Eigen::Vector3d(tx, ty, tz),
                   Eigen::Quaterniond(qw, qx, qy, qz), frame->time);
    }

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header = pose_msg.header;
    // copy the FULL PoseWithCovariance, not just the pose — otherwise the
    // covariance built above never reaches this topic and fusion reads zeros
    odom_msg.pose = pose_msg.pose;
    odom_msg.child_frame_id = base_frame_;
    odom_msg.twist.twist = frame->twist;
    // twist is not estimated here (the DR/Kalman upstream leave it zero), so
    // advertise a large twist covariance; the all-zero default would read as
    // zero-velocity-with-infinite-confidence to a fusing EKF.
    for (int i = 0; i < 6; ++i) odom_msg.twist.covariance[i * 6 + i] = 1e6;
    odom_pub_->publish(odom_msg);

    // periodic health counters — publish_pose runs per callback (~ping
    // rate), so gate on the keyframe count CHANGING or the same line
    // repeats hundreds of times while the count sits on a multiple of 25
    if (!slam_.keyframes.empty() && slam_.current_key() % 25 == 0 &&
        slam_.current_key() != last_logged_key_) {
      last_logged_key_ = slam_.current_key();
      RCLCPP_INFO(get_logger(),
                  "SLAM status: keyframes %d (last kf %ld pts), SSM factors %d, "
                  "NSSM accepted %d, last SSM: %s | NSSM attempts %d, queued %d, "
                  "best clique %d, queue depth %d, PCM min md %.1f (thresh 11.3, "
                  "%d edges), last NSSM: %s | NSSM rejects [%s]",
                  slam_.current_key(),
                  static_cast<long>(slam_.current_keyframe()->points.rows()),
                  slam_.ssm_accepted, slam_.nssm_accepted,
                  slam_.last_ssm_status.c_str(), slam_.nssm_attempts,
                  slam_.nssm_queued, slam_.nssm_best_clique,
                  slam_.nssm_queue_depth(), slam_.last_pcm_min_md,
                  slam_.last_pcm_edges, slam_.last_nssm_status.c_str(),
                  slam_.nssm_reject_summary().c_str());
    }
  }

  // Worker-thread body: build the graph constraint markers from a snapshot.
  void build_and_publish_viz(const std::vector<VizKeyframe>& snap,
                             const builtin_interfaces::msg::Time& stamp)
  {
    std::vector<ConstraintLink> links;
    for (std::size_t x = 1; x < snap.size(); ++x) {
      const auto& prev = snap[x - 1];
      const auto& curr = snap[x];
      const Eigen::Vector3d p1(prev.pose.x(), prev.pose.y(), prev.z);
      const Eigen::Vector3d p2(curr.pose.x(), curr.pose.y(), curr.z);
      links.push_back({{p1, p2}, "green"});

      for (const auto& [k, _] : curr.constraints) {
        if (k < 0 || k >= static_cast<int>(snap.size())) continue;
        const auto& target = snap[k];
        const Eigen::Vector3d p0(
          target.pose.x(), target.pose.y(), target.z);
        links.push_back({{p0, p2}, "red"});
      }
    }
    if (!links.empty()) {
      visualization_msgs::msg::Marker msg = ros_constraints(links);
      msg.header.stamp = stamp;
      constraint_pub_->publish(msg);
    }
  }

  void publish_trajectory()
  {
    // [x, y, z, roll, pitch, yaw, index]
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
      traj(k, 1) = static_cast<float>(kf->pose3.y());
      traj(k, 2) = static_cast<float>(kf->pose3.z());
      traj(k, 3) = static_cast<float>(kf->pose3.rotation().roll());
      traj(k, 4) = static_cast<float>(kf->pose3.rotation().pitch());
      traj(k, 5) = static_cast<float>(kf->pose3.rotation().yaw());
      traj(k, 6) = static_cast<float>(k);
      traj(k, 7) = static_cast<float>(to_sec(kf->time) - msg_stamp_s);
    }
    sensor_msgs::msg::PointCloud2 msg =
      make_cloud({"x", "y", "z", "roll", "pitch", "yaw", "i", "t"}, traj);
    msg.header.stamp = msg_stamp;
    msg.header.frame_id = "map";
    traj_pub_->publish(msg);
  }

  // Operator hand-correction: a map-frame planar pose fix (typically RViz's
  // "2D Pose Estimate") becomes a prior on the NEWEST keyframe; the elastic
  // DR chain distributes the correction backwards through the trajectory.
  // Operator input is ground truth by declaration — no post-loop
  // verification runs on this round — but it enters as a (soft-yaw) prior,
  // not a hard reset, so the graph still negotiates the exact pose.
  void manual_correction_callback(
    geometry_msgs::msg::PoseWithCovarianceStamped msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slam_.keyframes.empty()) {
      RCLCPP_WARN(get_logger(),
                  "manual correction ignored: no keyframes in the graph yet");
      return;
    }
    // with a loaded map the keyframes are the PREVIOUS session's — a fix
    // here would drag the saved map toward the click instead of placing the
    // vehicle; relocalization must land first
    if (slam_.awaiting_relocalization()) {
      RCLCPP_WARN(get_logger(),
                  "manual correction ignored while relocalizing against the "
                  "loaded map — wait for relocalization to land");
      return;
    }
    const gtsam::Pose3 p = r2g(msg.pose.pose);
    gtsam::Pose2 fix(p.x(), p.y(), p.rotation().yaw());
    // The click marks the vehicle's CURRENT pose, which can be up to a full
    // keyframe interval past the newest keyframe on a moving vehicle —
    // back-compose the DR delta so the prior targets where the KEYFRAME is,
    // not where the vehicle is now.
    if (slam_.current_frame &&
        slam_.current_frame != slam_.current_keyframe()) {
      const gtsam::Pose2 dr_delta = slam_.current_keyframe()->dr_pose.between(
        slam_.current_frame->dr_pose);
      fix = fix.compose(dr_delta.inverse());
    }
    const int key = slam_.current_key() - 1;
    const gtsam::Pose2 before = slam_.current_keyframe()->pose;

    if (!slam_.add_manual_correction(fix)) {
      RCLCPP_ERROR(get_logger(),
                   "manual correction failed (%s); estimator rebuilt from "
                   "last good state",
                   slam_.last_error().c_str());
      return;
    }

    const gtsam::Pose2 after = slam_.current_keyframe()->pose;
    RCLCPP_INFO(get_logger(),
                "manual correction: keyframe %d (%.2f, %.2f, %.1f deg) -> "
                "(%.2f, %.2f, %.1f deg); operator fix was "
                "(%.2f, %.2f, %.1f deg)",
                key, before.x(), before.y(), before.theta() * 180.0 / M_PI,
                after.x(), after.y(), after.theta() * 180.0 / M_PI,
                fix.x(), fix.y(), fix.theta() * 180.0 / M_PI);

    republish_corrected_state();
  }

  // Service twin of the manual correction: pop the most recent manual prior
  // off the graph and re-solve (the trajectory relaxes back toward the
  // uncorrected optimum). Same immediate-feedback republish as applying one.
  void undo_manual_correction(std_srvs::srv::Trigger::Response& res)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slam_.awaiting_relocalization()) {
      res.success = false;
      res.message = "relocalization pending — nothing to undo yet";
      return;
    }
    if (slam_.manual_corrections_pending() == 0) {
      res.success = false;
      res.message = "no manual correction to undo";
      return;
    }
    const gtsam::Pose2 before = slam_.current_keyframe()->pose;
    if (!slam_.undo_manual_correction()) {
      res.success = false;
      res.message = "undo failed: " + slam_.last_error();
      RCLCPP_ERROR(get_logger(), "manual-correction undo failed (%s)",
                   slam_.last_error().c_str());
      return;
    }
    const gtsam::Pose2 after = slam_.current_keyframe()->pose;
    char buf[192];
    std::snprintf(buf, sizeof buf,
                  "removed last manual prior: newest keyframe "
                  "(%.2f, %.2f, %.1f deg) -> (%.2f, %.2f, %.1f deg); "
                  "%d manual correction(s) remain",
                  before.x(), before.y(), before.theta() * 180.0 / M_PI,
                  after.x(), after.y(), after.theta() * 180.0 / M_PI,
                  slam_.manual_corrections_pending());
    res.success = true;
    res.message = buf;
    RCLCPP_INFO(get_logger(), "%s", buf);
    republish_corrected_state();
  }

  void save_map(std_srvs::srv::Trigger::Response& res)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slam_.keyframes.empty()) {
      res.success = false;
      res.message = "no keyframes to save";
      return;
    }
    if (slam_.save_map(map_save_path_)) {
      res.success = true;
      res.message = std::to_string(slam_.current_key()) + " keyframes -> " +
                    map_save_path_;
      RCLCPP_INFO(get_logger(), "map saved: %s", res.message.c_str());
    } else {
      res.success = false;
      res.message = slam_.last_error();
      RCLCPP_ERROR(get_logger(), "map save failed: %s", res.message.c_str());
    }
  }

  // USBL/LBL fix -> position prior on the stamp-nearest keyframe. Gates:
  // one fix per keyframe (acoustic rates would otherwise bloat the graph),
  // innovation bound (a multipath outlier must not yank the map), stamp
  // match within usbl_max_stamp_delta (USBL latency is real; an unmatched
  // fix belongs to no keyframe).
  // Linear interpolation of the DR chain at an arbitrary time (caller holds
  // mutex_). Keyframes are >= keyframe_duration apart (1.0 s deployed) and
  // usbl/max_stamp_delta is the same order, so a geodesic lerp between the
  // bracketing keyframes is adequate. Clamps to the ends of the chain.
  gtsam::Pose2 dr_pose_at(double t) const
  {
    const auto& kfs = slam_.keyframes;
    if (t <= to_sec(kfs.front()->time)) return kfs.front()->dr_pose;
    if (t >= to_sec(kfs.back()->time)) return kfs.back()->dr_pose;
    std::size_t lo = 0, hi = kfs.size() - 1;
    while (hi - lo > 1) {  // keyframe times are ascending
      const std::size_t mid = (lo + hi) / 2;
      if (to_sec(kfs[mid]->time) <= t) lo = mid; else hi = mid;
    }
    const double t0 = to_sec(kfs[lo]->time), t1 = to_sec(kfs[hi]->time);
    const double a = t1 > t0 ? (t - t0) / (t1 - t0) : 0.0;
    const gtsam::Pose2& p0 = kfs[lo]->dr_pose;
    const gtsam::Pose2 d = p0.between(kfs[hi]->dr_pose);
    return p0.compose(gtsam::Pose2::Expmap(a * gtsam::Pose2::Logmap(d)));
  }

  // Transponder offset in the configured base frame, from TF. Cached after the first success:
  // the mount is static, so one lookup is enough and a later TF hiccup must
  // not silently drop the correction. Returns false until it resolves.
  bool usbl_lever_arm(const std::string& frame, Eigen::Vector2d& r_bt)
  {
    if (usbl_lever_arm_valid_) { r_bt = usbl_lever_arm_; return true; }
    const std::string src = usbl_frame_id_.empty() ? frame : usbl_frame_id_;
    if (src.empty() || src == base_frame_) {
      usbl_lever_arm_ = Eigen::Vector2d::Zero();
      usbl_lever_arm_valid_ = true;
      r_bt = usbl_lever_arm_;
      return true;
    }
    try {
      const auto tf = tf_buffer_->lookupTransform(base_frame_, src,
                                                  tf2::TimePointZero);
      usbl_lever_arm_ =
        Eigen::Vector2d(tf.transform.translation.x, tf.transform.translation.y);
      usbl_lever_arm_valid_ = true;
      RCLCPP_INFO(get_logger(),
                  "USBL transponder '%s' is (%.3f, %.3f) m from %s — "
                  "lever arm will be removed from every fix",
                  src.c_str(), usbl_lever_arm_.x(), usbl_lever_arm_.y(),
                  base_frame_.c_str());
      r_bt = usbl_lever_arm_;
      return true;
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "USBL lever arm unavailable (%s <- %s: %s). Fixes are being "
        "applied AS IF the transponder were at %s — on a masted "
        "transponder that injects a heading-correlated position bias. Set "
        "usbl/frame_id or publish the mount TF.",
        base_frame_.c_str(), src.c_str(), e.what(), base_frame_.c_str());
      return false;
    }
  }

  void usbl_callback(const builtin_interfaces::msg::Time& stamp,
                     const std::string& frame_id, double x, double y,
                     double var_x, double var_y)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slam_.keyframes.empty() || slam_.awaiting_relocalization()) return;
    const double t = to_sec(stamp);
    int best = -1;
    double best_dt = usbl_max_stamp_delta_;
    for (int k = slam_.current_key() - 1; k >= 0; --k) {
      const double kt = to_sec(slam_.keyframes[k]->time);
      const double dt = std::fabs(kt - t);
      // STRICT compare: under a looped bag a previous pass can carry the
      // identical replayed stamp — scanning newest-first, the current-pass
      // keyframe must keep the match
      if (dt < best_dt) {
        best_dt = dt;
        best = k;
      }
      if (kt < t - usbl_max_stamp_delta_) break;  // keyframes are ordered
    }
    if (best < 0) {
      ++usbl_rejected_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "USBL fix has no keyframe within %.1f s (check usbl stamp domain / "
        "usbl.max_stamp_delta)",
        usbl_max_stamp_delta_);
      return;
    }
    if (usbl_applied_.count(best)) return;

    // 1. LEVER ARM. The fix locates the TRANSPONDER, not the base frame. A masted
    //    transponder's offset rotates with vehicle heading, so leaving it in
    //    is a heading-CORRELATED bias, not a constant one that calibrates
    //    out.
    // 2. MOTION. The fix is associated to the stamp-nearest keyframe within
    //    usbl/max_stamp_delta (1.0 s deployed). At 0.5 m/s that admits up to
    //    0.5 m of pure position error — the same size as usbl/min_sigma —
    //    unless the DR motion between the two instants is removed.
    const gtsam::Pose2 dr_fix = dr_pose_at(t);
    const double c = std::cos(dr_fix.theta()), sn = std::sin(dr_fix.theta());
    Eigen::Vector2d r_bt;
    if (usbl_lever_arm(frame_id, r_bt) && r_bt.squaredNorm() > 0.0) {
      x -= c * r_bt.x() - sn * r_bt.y();
      y -= sn * r_bt.x() + c * r_bt.y();
    }
    const gtsam::Pose2 motion = dr_fix.between(dr_pose_at(to_sec(slam_.keyframes[best]->time)));
    x += c * motion.x() - sn * motion.y();
    y += sn * motion.x() + c * motion.y();

    const gtsam::Pose2& kp = slam_.keyframes[best]->pose;
    const double inno = std::hypot(kp.x() - x, kp.y() - y);
    if (usbl_max_innovation_ > 0.0 && inno > usbl_max_innovation_) {
      ++usbl_rejected_;
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "USBL fix rejected: %.1f m from the estimate (> usbl.max_innovation "
        "%.1f m) — multipath outlier, or the map has genuinely drifted that "
        "far",
        inno, usbl_max_innovation_);
      return;
    }
    const double sigma =
      std::max(usbl_min_sigma_, std::sqrt(std::max(var_x, var_y)));
    if (!slam_.add_position_prior(best, x, y, sigma)) {
      RCLCPP_ERROR(get_logger(), "USBL prior failed (%s); estimator rebuilt",
                   slam_.last_error().c_str());
      return;
    }
    usbl_applied_.insert(best);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "USBL fix on keyframe %d: innovation %.2f m, sigma %.2f m (%d applied)",
      best, inno, sigma, slam_.position_priors_applied);
  }

  // 1 Hz /diagnostics snapshot of the SLAM funnel and exact-stamp input health.
  void publish_diagnostics()
  {
    diagnostic_msgs::msg::DiagnosticStatus st;
    st.name = "sonar_slam/slam";
    st.hardware_id = "sonar_slam_cpp";
    auto add = [&st](const char* k, const std::string& v) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = k;
      kv.value = v;
      st.values.push_back(kv);
    };

    const std::uint64_t tf_failures = tf_lookup_failures_;
    const std::uint64_t pitch_rejections = pitch_rejections_;
    int reverted = 0;
    std::string input_mode;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      reverted = slam_.nssm_reverted;
      input_mode = last_input_mode_;
      add("keyframes", std::to_string(slam_.current_key()));
      add("ssm_factors", std::to_string(slam_.ssm_accepted));
      add("ssm_degenerate_rejected",
          std::to_string(slam_.ssm_degenerate_rejected));
      add("nssm_accepted", std::to_string(slam_.nssm_accepted));
      add("nssm_attempts", std::to_string(slam_.nssm_attempts));
      add("nssm_queued", std::to_string(slam_.nssm_queued));
      add("nssm_queue_depth", std::to_string(slam_.nssm_queue_depth()));
      add("nssm_best_clique", std::to_string(slam_.nssm_best_clique));
      add("nssm_reverted", std::to_string(reverted));
      add("nssm_rejects", slam_.nssm_reject_summary());
      add("last_ssm", slam_.last_ssm_status);
      add("last_nssm", slam_.last_nssm_status);
      add("manual_corrections_applied",
          std::to_string(slam_.manual_corrections_applied));
      add("manual_corrections_undone",
          std::to_string(slam_.manual_corrections_undone));
      add("manual_priors_active",
          std::to_string(slam_.manual_corrections_pending()));
      add("usbl_priors_applied",
          std::to_string(slam_.position_priors_applied));
      add("usbl_rejected", std::to_string(usbl_rejected_));
      add("awaiting_relocalization",
          slam_.awaiting_relocalization() ? "true" : "false");
      add("input_mode", last_input_mode_);
      add("last_scan_admission", last_admission_summary_);
      add("admitted_scans", std::to_string(admitted_scans_));
      add("sparse_scan_rejections",
          std::to_string(sparse_scan_rejections_));
      add("sparse_rejections_points",
          std::to_string(sparse_point_rejections_));
      add("sparse_rejections_cells",
          std::to_string(sparse_cell_rejections_));
      add("sparse_rejections_azimuth",
          std::to_string(sparse_azimuth_rejections_));
    }
    add("exact_tf_failures", std::to_string(tf_failures));
    add("head_pitch_rejections", std::to_string(pitch_rejections));
    add("gpu", gpu::available() ? "on" : "off");

    if (tf_failures > diag_prev_tf_failures_) {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      st.message = "exact ping-time sensor/odometry TF unavailable";
    } else if (reverted > diag_prev_reverted_) {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      st.message = "loop-closure round reverted by post-loop verification";
    } else if (input_mode == "odometry_only_sparse") {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      st.message = "odometry-only: uninformative sonar";
    } else if (input_mode == "odometry_only_head_sweep") {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      st.message = "odometry-only: sonar head outside planar gate";
    } else {
      st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      st.message = "running";
    }
    diag_prev_tf_failures_ = tf_failures;
    diag_prev_reverted_ = reverted;

    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    arr.status.push_back(std::move(st));
    diag_pub_->publish(arr);
  }

  // Immediate feedback after a manual correction / undo, without waiting for
  // the next sonar frame (called under mutex_): re-anchor the between-keyframe
  // extrapolation on the corrected keyframe (skip when the current frame IS
  // that keyframe — the graph update already refreshed it, and re-running the
  // Pose2 overload would clobber its estimated z), republish pose + map->odom
  // TF, the latched trajectory (mapping re-renders its tiles from it), and
  // rebuild the viz products.
  void republish_corrected_state()
  {
    if (slam_.current_frame) {
      if (slam_.current_frame != slam_.current_keyframe()) {
        const gtsam::Pose2 dr_odom = slam_.current_keyframe()->dr_pose.between(
          slam_.current_frame->dr_pose);
        slam_.current_frame->update(
          slam_.current_keyframe()->pose.compose(dr_odom));
      }
      publish_pose();
    }
    publish_trajectory();
    if (schedule_viz_rebuild()) last_viz_publish_ = now();
  }

  Slam slam_;
  std::mutex mutex_;
  // Background graph-marker rebuild (see schedule_viz_rebuild).
  std::thread viz_thread_;
  std::atomic<bool> viz_busy_{false};
  bool enable_slam_ = true;
  bool publish_tf_ = true;
  double viz_min_period_ = 2.0;
  int last_logged_key_ = -1;
  rclcpp::Time last_viz_publish_{0, 0, RCL_ROS_TIME};
  std::string points_topic_;
  std::string odom_frame_ = "odom";
  std::string base_frame_ = "base_link";
  double tf_lookup_timeout_ = 0.05;
  double max_head_pitch_ = 0.52;
  ScanAdmissionParams admission_params_;
  std::string last_input_mode_ = "waiting_for_sonar";
  std::string last_admission_summary_ = "none yet";
  std::uint64_t admitted_scans_ = 0;
  std::uint64_t sparse_scan_rejections_ = 0;
  std::uint64_t sparse_point_rejections_ = 0;
  std::uint64_t sparse_cell_rejections_ = 0;
  std::uint64_t sparse_azimuth_rejections_ = 0;

  // ---- map->odom republish (rtabmap CoreWrapper's transformThread_) --------
  // The edge used to be broadcast ONLY from the sonar callback, so a sonar or
  // input-TF dropout stopped it entirely and every
  // map-frame consumer (nav2 global costmap, bt_navigator, the accumulator
  // chain, RViz) starved. It also carried no future-dating while the odom EKF
  // future-dates odom->base_link by transform_time_offset 0.1, leaving the
  // map edge systematically the older of the two.
  //
  // The callback now only UPDATES this cache and the timer is the sole
  // publisher (rtabmap's structure), so the two cannot interleave stamps.
  struct MapOdomTf
  {
    bool valid = false;
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    rclcpp::Time data_stamp{0, 0, RCL_ROS_TIME};  // stamp of the source frame
    rclcpp::Time cached_at{0, 0, RCL_ROS_TIME};   // node clock when cached
  };
  std::mutex tf_mutex_;
  MapOdomTf map_odom_tf_;
  // Exact-stamp sensor and odometry transforms; also serves the optional USBL
  // transponder lever arm.
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  double tf_publish_period_ = 0.05;  // s; <= 0 -> publish inline (old behavior)
  double tf_tolerance_ = 0.1;        // s of future-dating
  rclcpp::TimerBase::SharedPtr tf_timer_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    manual_correction_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr undo_srv_;
  // map persistence + USBL input
  std::string map_save_path_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_map_srv_;
  rclcpp::SubscriptionBase::SharedPtr usbl_sub_;
  double usbl_max_innovation_ = 10.0;
  double usbl_min_sigma_ = 0.5;
  double usbl_max_stamp_delta_ = 1.0;
  // Frame the USBL fix is reported in. Empty = trust the message's
  // header.frame_id. Resolved to a base_link offset once via TF and cached
  // (the mount is static).
  std::string usbl_frame_id_;
  Eigen::Vector2d usbl_lever_arm_ = Eigen::Vector2d::Zero();
  bool usbl_lever_arm_valid_ = false;
  std::set<int> usbl_applied_;
  std::uint64_t usbl_rejected_ = 0;
  // Subscription counters are atomic because the TF listener has its own
  // thread while diagnostics run on the executor.
  std::atomic<std::uint64_t> tf_lookup_failures_{0}, pitch_rejections_{0};
  std::uint64_t diag_prev_tf_failures_ = 0;
  int diag_prev_reverted_ = 0;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr traj_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr constraint_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  return sonar_slam::run_node<sonar_slam::SlamNode>(argc, argv);
}
