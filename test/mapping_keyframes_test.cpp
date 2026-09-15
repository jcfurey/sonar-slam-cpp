// Exercise the public node boundary: mapping trajectory anchors survive weak
// acoustics, while their clouds cannot become sequential or loop evidence.
#define main slam_node_disabled_main
#include "../src/nodes/slam_node.cpp"
#undef main

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#define CHECK(condition, message) \
  do { if (!(condition)) { std::fprintf(stderr, "FAIL line %d: %s\n", \
                           __LINE__, message); return 1; } } while (0)

namespace {
builtin_interfaces::msg::Time stamp(double seconds)
{
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
}

geometry_msgs::msg::TransformStamped odom(double seconds, double x, double yaw = 0.0)
{
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp(seconds);
  tf.header.frame_id = "odom";
  tf.child_frame_id = "base_link";
  tf.transform.translation.x = x;
  tf.transform.rotation.z = std::sin(yaw / 2);
  tf.transform.rotation.w = std::cos(yaw / 2);
  return tf;
}

sensor_msgs::msg::PointCloud2 cloud(double seconds, double x, bool informative)
{
  // Three static walls, expressed in a moving optical sonar frame. No
  // randomness; successive informative frames have known exact overlap.
  const int count = informative ? 300 : 3;
  sonar_slam::Matrix points(count, 3);
  for (int i = 0; i < count; ++i) {
    const double t = (i % 100) * 0.1;
    const double forward = i < 100 ? 10.0 : 1.0 + t;
    const double lateral = i < 100 ? t - 5.0 : (i < 200 ? -5.0 : 5.0);
    points.row(i) << 0.0, lateral, forward - x;
  }
  auto msg = sonar_slam::make_cloud({"x", "y", "z"}, points);
  msg.header.stamp = stamp(seconds);
  msg.header.frame_id = "sonar0/projection_frame";
  return msg;
}
}  // namespace

int main()
{
  rclcpp::init(0, nullptr);
  const char* cfg = std::getenv("SLAM_TEST_CONFIG");
  CHECK(cfg, "SLAM_TEST_CONFIG missing");
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "--params-file", cfg,
    "-p", "points_topic:=/mapping_test/points",
    "-p", "keyframe_translation:=0.75",
    "-p", "point_resolution:=0.1",
    "-p", "tf_lookup_timeout:=0.02"});
  auto slam = std::make_shared<sonar_slam::SlamNode>(options);
  auto probe = rclcpp::Node::make_shared("mapping_keyframes_probe");
  auto pub = probe->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/mapping_test/points", rclcpp::QoS(10).reliable());
  tf2_ros::StaticTransformBroadcaster mount(probe);
  tf2_ros::TransformBroadcaster tf(probe);
  auto optical = odom(0.0, 0.0);
  optical.header.frame_id = "base_link";
  optical.child_frame_id = "sonar0/projection_frame";
  optical.transform.rotation.y = std::sin(M_PI / 4);
  optical.transform.rotation.w = std::cos(M_PI / 4);
  mount.sendTransform(optical);

  std::map<std::string, std::string> diagnostics;
  auto diag_sub = probe->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
    "/diagnostics", rclcpp::QoS(4),
    [&](const diagnostic_msgs::msg::DiagnosticArray& msg) {
      for (const auto& status : msg.status)
        if (status.name == "sonar_slam/slam")
          for (const auto& kv : status.values) diagnostics[kv.key] = kv.value;
    });
  sensor_msgs::msg::PointCloud2 trajectory;
  sensor_msgs::msg::PointCloud2 mapping_trajectory;
  auto traj_sub = probe->create_subscription<sensor_msgs::msg::PointCloud2>(
    sonar_slam::SLAM_TRAJ_TOPIC, rclcpp::QoS(4).reliable().transient_local(),
    [&](const sensor_msgs::msg::PointCloud2& msg) { trajectory = msg; });
  auto mapping_sub = probe->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/slam/mapping_traj", rclcpp::QoS(4).reliable().transient_local(),
    [&](const sensor_msgs::msg::PointCloud2& msg) { mapping_trajectory = msg; });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(slam);
  executor.add_node(probe);
  auto spin = [&](double seconds) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < until)
      executor.spin_some(std::chrono::milliseconds(10));
  };
  auto value = [&](const char* key) {
    return diagnostics.count(key) ? std::atoll(diagnostics[key].c_str()) : -1;
  };
  auto ping = [&](double seconds, double x, bool informative) {
    tf.sendTransform(odom(seconds - 0.05, x));
    tf.sendTransform(odom(seconds + 0.05, x));
    spin(0.15);
    pub->publish(cloud(seconds, x, informative));
    spin(1.3);
  };
  spin(0.4);

  ping(100.0, 0.0, false);
  CHECK(value("keyframes") == 1 && trajectory.width == 1,
        "first sparse scan failed to publish a mapping trajectory anchor");
  CHECK(mapping_trajectory.width == 0, "mapping anchor raced its same-ping evidence");
  ping(102.0, 1.0, false);
  CHECK(value("keyframes") == 2 && trajectory.width == 2,
        "moved sparse scan failed to extend the mapping trajectory");
  CHECK(mapping_trajectory.width == 1 && mapping_trajectory.header.stamp.sec == 100,
        "mapping anchor was not delayed with its acquisition stamp intact");
  CHECK(value("odometry_only_keyframes") == 2 && value("registration_points") == 0,
        "sparse acoustic returns leaked into registration history");
  CHECK(value("ssm_factors") == 0 && value("nssm_attempts") == 0 &&
        diagnostics["last_ssm"] == "none yet" && diagnostics["last_nssm"] == "none yet",
        "odometry-only anchors attempted acoustic registration");
  float last_x = 0.0f;
  std::memcpy(&last_x, trajectory.data.data() + trajectory.point_step, sizeof(float));
  CHECK(std::abs(last_x - 1.0f) < 0.01f, "sparse trajectory does not follow odometry");

  // Both frames pass the exact deployed covariance/overlap/degeneracy gates.
  // First informative frame has no acoustic target, then matching resumes.
  ping(104.0, 2.0, true);
  CHECK(value("registration_keyframes") == 1 && value("registration_points") == 300,
        "first informative scan was not retained as registration evidence");
  CHECK(value("ssm_factors") == 0, "first informative frame matched sparse history");
  ping(106.0, 3.0, true);
  CHECK(value("registration_keyframes") == 2 && value("registration_points") == 600,
        "informative evidence missing after sparse interval");
  CHECK(value("ssm_factors") == 1, "sequential matching did not resume after sparse interval");

  const auto anchors = value("keyframes");
  // Exact timestamp without a TF sample must not create any mapping state.
  pub->publish(cloud(200.0, 10.0, false));
  spin(1.3);
  CHECK(value("exact_tf_failures") == 1 && value("keyframes") == anchors,
        "missing timestamp transform created an odometry anchor");
  // Implausible pose turn is rejected before both mapping and registration.
  tf.sendTransform(odom(106.1, 3.0, M_PI / 2));
  spin(0.15);
  pub->publish(cloud(106.1, 3.0, false));
  spin(1.3);
  CHECK(value("dr_yaw_rejections") == 1 && value("keyframes") == anchors,
        "invalid yaw pose created an odometry anchor");
  CHECK(value("registration_points") == 600 && value("ssm_factors") == 1,
        "invalid inputs changed acoustic evidence or factors");

  // Zero is TF's 'latest', never an exact acquisition time. Reject it before
  // lookup rather than allowing it to rewind the graph or mapping history.
  pub->publish(cloud(0.0, 3.0, false));
  spin(1.3);
  CHECK(value("exact_tf_failures") == 2 && value("keyframes") == anchors,
        "zero acquisition stamp reached the graph");
  const auto old_mapping_count = mapping_trajectory.width;
  for (double time : {108.0, 110.0, 112.0}) ping(time, 3.0, false);
  CHECK(value("keyframes") == anchors && value("ssm_factors") == 1,
        "stationary mapping cadence changed graph or acoustic factors");
  CHECK(mapping_trajectory.width > old_mapping_count && mapping_trajectory.header.stamp.sec == 110,
        "stationary mapping history failed to advance at source cadence");
  ping(112.5, 3.0, false);
  spin(2.5);
  CHECK(mapping_trajectory.header.stamp.sec == 112 && mapping_trajectory.header.stamp.nanosec == 500000000,
        "final partial tail did not flush its true acquisition stamp");
  const auto tail_width = mapping_trajectory.width;
  auto last_x_in = [](const sensor_msgs::msg::PointCloud2& msg) {
    float x;
    std::memcpy(&x, msg.data.data() + (msg.width - 1) * msg.point_step, sizeof(x));
    return x;
  };
  const auto x_before = last_x_in(mapping_trajectory);
  auto correction_pub = probe->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 1);
  spin(0.2);
  geometry_msgs::msg::PoseWithCovarianceStamped correction;
  correction.header.frame_id = "map";
  correction.pose.pose.position.x = 13.0;
  correction.pose.pose.orientation.w = 1.0;
  correction_pub->publish(correction);
  spin(1.3);
  CHECK(mapping_trajectory.width == tail_width && last_x_in(mapping_trajectory) > x_before + 1.0f,
        "graph correction did not republish existing mapping anchors");
  CHECK(mapping_trajectory.header.stamp.nanosec == 500000000,
        "correction changed an existing mapping anchor timestamp");

  // Stay inside this node's live 10 s TF cache while crossing the rewind gate.
  ping(105.0, 0.0, false);
  CHECK(value("keyframes") == 1 && mapping_trajectory.width == 0,
        "rewind retained old graph/mapping latch");
  ping(107.0, 0.0, false);
  CHECK(mapping_trajectory.width == 1 && mapping_trajectory.header.stamp.sec == 105,
        "rewound mapping history did not restart with stable new-session indices");
  std::printf("PASS: sparse anchors, delayed stationary coverage/tail, correction, invalid input and rewind\n");
  executor.remove_node(slam);
  executor.remove_node(probe);
  rclcpp::shutdown();
  return 0;
}
