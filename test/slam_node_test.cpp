// In-process harness for the slam_node ingestion layer — the seam between
// TF/proc_points and the pose graph that only ran hand-verified until now:
// exact-stamp TF admission, the head-pitch gate, the DR yaw spike gate's
// NODE wiring, scan admission counters, the odometry-only fallback, and the
// bag-rewind session reset (fresh graph + refreshed depth-1 latches).
//
// The node class lives inside slam_node.cpp with its own main(); include it
// with main renamed so the class is testable without restructuring the
// build. The test drives the node through an executor with a fixture TF
// tree and synthetic clouds, then reads outcomes from /diagnostics.
#define main slam_node_disabled_main
#include "../src/nodes/slam_node.cpp"
#undef main

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#define CHECK(cond, ...)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("FAIL(%d): ", __LINE__);                                 \
      std::printf(__VA_ARGS__);                                            \
      std::printf("\n");                                                   \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

sensor_msgs::msg::PointCloud2 make_cloud_msg(double stamp_s, int n_points,
                                             double spread)
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp.sec = static_cast<int32_t>(stamp_s);
  msg.header.stamp.nanosec =
    static_cast<uint32_t>((stamp_s - static_cast<int32_t>(stamp_s)) * 1e9);
  msg.header.frame_id = "sonar0/projection_frame";
  msg.height = 1;
  msg.width = n_points;
  msg.is_bigendian = false;
  msg.is_dense = true;
  for (int i = 0; i < 3; ++i) {
    sensor_msgs::msg::PointField f;
    f.name = i == 0 ? "x" : (i == 1 ? "y" : "z");
    f.offset = 4 * i;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;
    msg.fields.push_back(f);
  }
  msg.point_step = 12;
  msg.row_step = 12 * n_points;
  msg.data.resize(msg.row_step);
  auto* f = reinterpret_cast<float*>(msg.data.data());
  // an L of structure in the optical fan plane (y lateral, z forward):
  // enough occupied cells and azimuth spread to pass admission when
  // n_points is generous, too little when it is tiny
  for (int i = 0; i < n_points; ++i) {
    const float t = static_cast<float>(i) / std::max(1, n_points - 1);
    f[3 * i + 0] = 0.0f;
    f[3 * i + 1] = static_cast<float>((t - 0.5) * 2.0 * spread);
    f[3 * i + 2] = 3.0f + static_cast<float>((i % 2) ? t * spread : 0.0);
  }
  return msg;
}

geometry_msgs::msg::TransformStamped make_tf(double stamp_s,
                                             const std::string& parent,
                                             const std::string& child,
                                             double x, double yaw,
                                             double pitch = 0.0)
{
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp.sec = static_cast<int32_t>(stamp_s);
  t.header.stamp.nanosec =
    static_cast<uint32_t>((stamp_s - static_cast<int32_t>(stamp_s)) * 1e9);
  t.header.frame_id = parent;
  t.child_frame_id = child;
  t.transform.translation.x = x;
  tf2::Quaternion q;
  q.setRPY(0.0, pitch, yaw);
  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  t.transform.rotation.w = q.w();
  return t;
}

}  // namespace

int main()
{
  rclcpp::init(0, nullptr);
  int rc = 1;
  {
    // The node requires the full parameter file (params are
    // throw-if-missing); load the packaged slam.yaml and override the
    // harness specifics on top (later CLI -p wins over the file).
    const char* cfg = std::getenv("SLAM_TEST_CONFIG");
    CHECK(cfg != nullptr, "SLAM_TEST_CONFIG not set");
    rclcpp::NodeOptions opts;
    opts.arguments({"--ros-args", "--params-file", cfg,
                    "-p", "points_topic:=/test/points",
                    "-p", "ssm.enable:=false",
                    "-p", "nssm.enable:=false",
                    "-p", "dr.max_yaw_rate:=1.5",
                    "-p", "max_head_pitch:=0.52",
                    "-p", "tf_lookup_timeout:=0.05",
                    "-p", "tf_buffer_duration:=120.0"});
    auto node = std::make_shared<sonar_slam::SlamNode>(opts);
    auto pub_node = rclcpp::Node::make_shared("slam_harness");
    auto points_pub = pub_node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/test/points", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    tf2_ros::StaticTransformBroadcaster static_tf(pub_node);
    tf2_ros::TransformBroadcaster dyn_tf(pub_node);

    std::map<std::string, std::string> diag;
    auto diag_sub =
      pub_node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
        "/diagnostics", rclcpp::QoS(rclcpp::KeepLast(4)),
        [&](const diagnostic_msgs::msg::DiagnosticArray& m) {
          for (const auto& st : m.status)
            for (const auto& kv : st.values)
              diag[kv.key] = kv.value;
        });

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    exec.add_node(pub_node);

    // optical mount: +90 deg about y (level boresight)
    static_tf.sendTransform(
      make_tf(0.0, "base_link", "sonar0/projection_frame", 0.0, 0.0, M_PI / 2));

    auto spin_for = [&](double seconds) {
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::duration<double>(seconds);
      while (std::chrono::steady_clock::now() < deadline)
        exec.spin_some(std::chrono::milliseconds(10));
    };
    auto odom_at = [&](double stamp, double x, double yaw) {
      dyn_tf.sendTransform(make_tf(stamp, "odom", "base_link", x, yaw));
    };
    auto diag_int = [&](const char* k) {
      return diag.count(k) ? std::atoll(diag[k].c_str()) : -1;
    };

    spin_for(0.3);  // discovery + static TF delivery

    // [1] healthy ping: exact-stamp TF present, structured cloud -> admitted
    odom_at(100.0, 0.0, 0.0);
    odom_at(100.2, 0.05, 0.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(100.1, 120, 2.0));
    spin_for(1.4);  // includes a diagnostics tick
    CHECK(diag_int("admitted_scans") == 1,
          "healthy ping not admitted (admitted_scans=%lld)",
          diag_int("admitted_scans"));
    CHECK(diag_int("keyframes") >= 1, "no keyframe from the healthy ping");
    std::printf("[1] healthy ping admitted\n");

    // [2] sparse ping: TF fine, 3 points -> rejected as evidence,
    // odometry-only fallback
    odom_at(101.0, 0.10, 0.0);
    odom_at(101.2, 0.12, 0.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(101.1, 3, 0.5));
    spin_for(1.4);
    CHECK(diag_int("sparse_scan_rejections") == 1,
          "sparse ping not rejected (sparse=%lld)",
          diag_int("sparse_scan_rejections"));
    CHECK(diag["input_mode"] == "odometry_only_sparse",
          "input_mode '%s', expected odometry_only_sparse",
          diag["input_mode"].c_str());
    std::printf("[2] sparse ping rejected as evidence\n");

    // [3] Explicit safety-gate compatibility: this fixture launches with a
    // positive max_head_pitch, so a larger sweep is skipped while odometry
    // continues. Production defaults the gate off for constrained 3-D.
    static_tf.sendTransform(make_tf(0.0, "base_link", "sonar0/projection_frame",
                                    0.0, 0.0, M_PI / 2 + 0.6));
    odom_at(102.0, 0.20, 0.0);
    odom_at(102.2, 0.22, 0.0);
    spin_for(0.3);
    points_pub->publish(make_cloud_msg(102.1, 120, 2.0));
    spin_for(1.4);
    CHECK(diag_int("head_pitch_rejections") == 1,
          "swept head not rejected (pitch=%lld)",
          diag_int("head_pitch_rejections"));
    static_tf.sendTransform(
      make_tf(0.0, "base_link", "sonar0/projection_frame", 0.0, 0.0, M_PI / 2));
    std::printf("[3] head-swept scan skipped\n");

    // [4] DR yaw spike: 40 deg step in 100 ms -> ping dropped by the yaw
    // gate before either branch consumes it
    odom_at(103.0, 0.30, 0.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(103.0, 120, 2.0));
    spin_for(0.6);
    odom_at(103.1, 0.31, 40.0 * M_PI / 180.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(103.1, 120, 2.0));
    spin_for(1.4);
    CHECK(diag_int("dr_yaw_rejections") == 1,
          "yaw spike not rejected (dr_yaw=%lld)", diag_int("dr_yaw_rejections"));
    std::printf("[4] DR yaw spike dropped\n");

    // [5] missing exact-stamp odom TF -> ping dropped, counted
    const long long tf_before = diag_int("exact_tf_failures");
    points_pub->publish(make_cloud_msg(200.0, 120, 2.0));  // no TF near 200s
    spin_for(1.6);
    CHECK(diag_int("exact_tf_failures") > tf_before,
          "missing TF not counted (%lld -> %lld)", tf_before,
          diag_int("exact_tf_failures"));
    std::printf("[5] exact-stamp TF failure counted\n");

    // [6] bag rewind: a ping stamped far behind the last processed one must
    // reset the session — fresh graph, refreshed depth-1 latches. First grow
    // the session to keyframe 2 with a > keyframe_translation DR step so the
    // reset is observable as a keyframe-count drop.
    odom_at(103.6, 4.0, 0.0);
    odom_at(103.8, 4.1, 0.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(103.7, 120, 2.0));
    spin_for(1.4);
    CHECK(diag_int("keyframes") == 2,
          "big DR step did not promote keyframe 2 (keyframes=%lld)",
          diag_int("keyframes"));
    const long long adm_before_rewind = diag_int("admitted_scans");

    // Rewound stamps stay inside even the live 10 s tf2 cache (this test
    // uses the longer replay cache). The DR yaw gate must NOT
    // swallow this ping: on a backward stamp it drops its baseline instead
    // of rejecting, so the rewound ping reaches the session-reset check.
    odom_at(94.9, 0.0, 0.0);
    odom_at(95.1, 0.05, 0.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(95.0, 120, 2.0));
    spin_for(1.4);
    CHECK(diag_int("keyframes") == 1,
          "rewound ping did not reset the session (keyframes=%lld)",
          diag_int("keyframes"));
    CHECK(diag_int("admitted_scans") == adm_before_rewind + 1,
          "rewound ping not admitted into the fresh session (%lld -> %lld)",
          adm_before_rewind, diag_int("admitted_scans"));

    // The depth-1 transient-local traj latch must now hold the NEW session
    // only: a late joiner gets one retained cloud — width 1, stamped at the
    // rewound ping — never the dead session's 2-keyframe snapshot (whose
    // stamps a looped replay would exactly reuse for wrong associations).
    sensor_msgs::msg::PointCloud2 latched;
    bool got_latched = false;
    auto traj_sub =
      pub_node->create_subscription<sensor_msgs::msg::PointCloud2>(
        sonar_slam::SLAM_TRAJ_TOPIC,
        rclcpp::QoS(rclcpp::KeepLast(4)).reliable().transient_local(),
        [&](const sensor_msgs::msg::PointCloud2& m) {
          latched = m;
          got_latched = true;
        });
    spin_for(1.0);
    CHECK(got_latched, "no retained sample on the traj latch after rewind");
    CHECK(latched.width == 1,
          "traj latch still carries the dead session (width=%u)",
          latched.width);
    CHECK(latched.header.stamp.sec == 95,
          "traj latch stamp %d != rewound session start",
          latched.header.stamp.sec);
    std::printf("[6] rewind reset: session cleared, latches refreshed\n");

    // [7] accelerated replay can put /tf twenty seconds ahead of a queued
    // cloud. The replay cache must preserve its exact odometry sample.
    const auto tf_before_backlog = diag_int("exact_tf_failures");
    const auto admitted_before_backlog = diag_int("admitted_scans");
    odom_at(104.9, 0.1, 0.0);
    odom_at(105.1, 0.1, 0.0);
    odom_at(125.0, 0.1, 0.0);
    spin_for(0.2);
    points_pub->publish(make_cloud_msg(105.0, 120, 2.0));
    spin_for(1.4);
    CHECK(diag_int("exact_tf_failures") == tf_before_backlog,
          "replay backlog lost the historical odometry transform");
    CHECK(diag_int("admitted_scans") == admitted_before_backlog + 1,
          "backlogged sonar ping was not admitted");
    std::printf("[7] replay backlog retains exact-stamp TF\n");

    rc = 0;
    std::printf("PASS\n");
    exec.remove_node(node);
    exec.remove_node(pub_node);
  }
  rclcpp::shutdown();
  return rc;
}
