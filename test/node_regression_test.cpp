#include "sonar_slam_cpp/node.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "physical_scene.hpp"
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <gtest/gtest.h>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <thread>

namespace {
class NodeRegression : public ::testing::Test
{
protected:
  virtual bool physical() const { return false; }
  virtual std::string manual_mode() const { return physical() ? "se3" : "planar"; }
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
  void SetUp() override
  {
    static int sequence = 0;
    ns = "/sonar_regression_" + std::to_string(++sequence);
    map_frame = "regression_map_" + std::to_string(sequence);
    odom_frame = "regression_odom_" + std::to_string(sequence);
    base_frame = "regression_base_" + std::to_string(sequence);
    sensor_frame = "regression_sonar_" + std::to_string(sequence);
    mount_frame = "regression_mount_" + std::to_string(sequence);
    const char* config = std::getenv("SLAM_TEST_CONFIG");
    ASSERT_NE(config, nullptr);
    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "-r", "__ns:=" + ns, "--params-file", config,
      "-p", "points_topic:=" + ns + "/points", "-p", "input_model:=measured_xyz",
      "-p", physical() ? "registration_model:=se3" : "registration_model:=constrained_xyyaw",
      "-p", "manual_correction_mode:=" + manual_mode(),
      "-p", "point_resolution:=0.2", "-p", "registration.voxel_resolution:=0.2",
      "-p", "icp_config:=" + (std::filesystem::path(config).parent_path() / "icp.yaml").string(),
      "-p", "input_reliable:=true", "-p", "input_queue_depth:=1",
      "-p", physical() ? "ssm.enable:=true" : "ssm.enable:=false", "-p", "nssm.enable:=false",
      "-p", "map_frame:=" + map_frame, "-p", "odom_frame:=" + odom_frame,
      "-p", "base_frame:=" + base_frame, "-p", "max_head_pitch:=0.52",
      "-p", "tf_lookup_timeout:=0.01", "-p", "tf_buffer_duration:=120.0",
      "-p", "keyframe_duration:=0.01", "-p", "keyframe_translation:=0.2",
      "-p", "mapping_anchor_interval:=0.5", "-p", "manual_correction_topic:=" + ns + "/manual",
      "-p", "manual_correction_sigmas:=[0.001, 0.001, 0.001]",
      "-p", "usbl.topic:=" + ns + "/usbl", "-p", "usbl.frame_id:=" + mount_frame,
      "-p", "usbl.max_innovation:=100.0", "-p", "usbl.min_sigma:=0.001"});
    node = sonar_slam::make_slam_node(options);
    probe = rclcpp::Node::make_shared("sonar_regression_probe_" + std::to_string(sequence));
    cloud_pub = probe->create_publisher<sensor_msgs::msg::PointCloud2>(ns + "/points", 10);
    manual_pub = probe->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(ns + "/manual", 10);
    usbl_pub = probe->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(ns + "/usbl", 10);
    pose_sub = probe->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      ns + "/slam/pose", 10, [this](const geometry_msgs::msg::PoseWithCovarianceStamped& msg) { pose = msg; ++poses; });
    mapping_sub = probe->create_subscription<sensor_msgs::msg::PointCloud2>(
      ns + "/slam/mapping_traj", rclcpp::QoS(1).reliable().transient_local(),
      [this](const sensor_msgs::msg::PointCloud2& msg) { mapping = msg; ++mappings; });
    diag_sub = probe->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", 10, [this](const diagnostic_msgs::msg::DiagnosticArray& msg) {
        for (const auto& status : msg.status)
          for (const auto& value : status.values) diagnostics[value.key] = value.value;
      });
    dynamic_tf = std::make_unique<tf2_ros::TransformBroadcaster>(probe);
    static_tf = std::make_unique<tf2_ros::StaticTransformBroadcaster>(probe);
    auto sensor = tf(0.0, 0.0);
    sensor.header.frame_id = base_frame;
    sensor.child_frame_id = sensor_frame;
    auto mount = sensor;
    mount.child_frame_id = mount_frame;
    mount.transform.translation.x = 1.0;
    static_tf->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{sensor, mount});
    executor.add_node(node);
    node_added = true;
    executor.add_node(probe);
    probe_added = true;
    spin(0.3);
  }
  void TearDown() override
  {
    if (node) node->get_node_base_interface()->get_default_callback_group()->can_be_taken_from() = true;
    if (node_added) executor.remove_node(node);
    if (probe_added) executor.remove_node(probe);
  }
  void spin(double seconds)
  {
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < until) {
      executor.spin_some(std::chrono::milliseconds(5));
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  geometry_msgs::msg::TransformStamped tf(double seconds, double x)
  {
    geometry_msgs::msg::TransformStamped result;
    result.header.stamp = rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
    result.header.frame_id = odom_frame;
    result.child_frame_id = base_frame;
    result.transform.translation.x = x;
    result.transform.rotation.w = 1.0;
    return result;
  }
  void ping(double seconds, double x, int count = 100)
  {
    dynamic_tf->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{
      tf(seconds - 0.01, x), tf(seconds + 0.01, x)});
    spin(0.04);
    sonar_slam::Matrix points(count, 3);
    for (int i = 0; i < count; ++i) points.row(i) << 3.0 + (i % 10) * 0.2, (i / 10 - 5) * 0.3, (i % 3) * 0.2;
    auto cloud = sonar_slam::make_cloud_xyz(points);
    cloud.header.stamp = rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
    cloud.header.frame_id = sensor_frame;
    cloud_pub->publish(cloud);
    spin(0.08);
  }
  void physical_ping(double seconds, const gtsam::Pose3& truth, const gtsam::Pose3& dr)
  {
    auto before = tf(seconds - .01, dr.x());
    auto after = tf(seconds + .01, dr.x());
    const auto pose = sonar_slam::g2r(dr);
    for (auto* message : {&before, &after}) {
      message->transform.rotation = pose.orientation;
      message->transform.translation.y = pose.position.y;
      message->transform.translation.z = pose.position.z;
    }
    dynamic_tf->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{before, after});
    spin(.05);
    const auto points = sonar_slam::transform_body_points(sonar_slam::test::physical_scene(), truth.inverse());
    auto cloud = sonar_slam::make_cloud_xyz(points);
    cloud.header.stamp = rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
    cloud.header.frame_id = sensor_frame;
    cloud_pub->publish(cloud);
    spin(.3);
  }
  long count(const std::string& key) { return std::stol(diagnostics.at(key)); }
  std::string ns, map_frame, odom_frame, base_frame, sensor_frame, mount_frame;
  rclcpp::Node::SharedPtr node, probe;
  rclcpp::executors::SingleThreadedExecutor executor;
  std::unique_ptr<tf2_ros::TransformBroadcaster> dynamic_tf;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr manual_pub, usbl_pub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mapping_sub;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_sub;
  geometry_msgs::msg::PoseWithCovarianceStamped pose;
  sensor_msgs::msg::PointCloud2 mapping;
  std::map<std::string, std::string> diagnostics;
  int poses = 0, mappings = 0;
  bool node_added = false, probe_added = false;
};

TEST_F(NodeRegression, BodyAxesAndNamespacedStartupOnlyParameters)
{
  ping(100.0, 0.0);
  spin(1.0);
  EXPECT_EQ(count("admitted_scans"), 1);
  EXPECT_EQ(count("head_pitch_rejections"), 0);
  EXPECT_EQ(pose.header.frame_id, map_frame);
  EXPECT_GT(poses, 0);
  EXPECT_TRUE(node->describe_parameter("enable_slam").read_only);
  EXPECT_FALSE(node->set_parameter(rclcpp::Parameter("enable_slam", false)).successful);
}

TEST_F(NodeRegression, DenseMeasuredCloudIsBoundedBeforeRetention)
{
  ping(100.0, 0.0, 20000);
  spin(1.0);
  EXPECT_EQ(count("admitted_scans"), 1);
  EXPECT_GT(count("registration_points"), 0);
  EXPECT_LE(count("registration_points"), 5000);
}

TEST_F(NodeRegression, MappingContinuesWhileRegistrationCannotRun)
{
  ping(100.0, 0.0);
  ASSERT_GT(poses, 0);
  auto group = node->get_node_base_interface()->get_default_callback_group();
  group->can_be_taken_from() = false;
  for (int i = 1; i <= 9; ++i) ping(100.0 + i, 0.3 * i);
  ASSERT_GE(mapping.header.stamp.sec, 107);
  EXPECT_GE(mapping.width, 8U);
  group->can_be_taken_from() = true;
  spin(1.3);
  EXPECT_EQ(count("registration_queue_pending"), 0);
  EXPECT_EQ(count("registration_queue_dropped"), 0);
  EXPECT_EQ(count("registration_queue_errors"), 0);
  const auto xyz = sonar_slam::cloud_to_xyz(mapping);
  ASSERT_GT(xyz.rows(), 0);
  EXPECT_NEAR(xyz(xyz.rows() - 1, 0), 2.7, 1e-4);
}

TEST_F(NodeRegression, ExternalFixesUseMapAttitudeAndPublishWithoutSonar)
{
  for (int i = 0; i < 4; ++i) ping(200 + i * 2, i);
  ASSERT_GT(poses, 0);
  geometry_msgs::msg::PoseWithCovarianceStamped correction;
  correction.header.frame_id = map_frame;
  correction.pose.pose.position.x = 3.0;
  correction.pose.pose.orientation.z = std::sin(0.55);
  correction.pose.pose.orientation.w = std::cos(0.55);
  manual_pub->publish(correction);
  spin(0.4);
  const auto before = pose;
  ASSERT_GT(std::abs(sonar_slam::r2g(before.pose.pose).rotation().yaw()), 0.5);
  auto fix = before;
  const Eigen::Vector3d lever = sonar_slam::r2g(before.pose.pose).rotation().matrix() * Eigen::Vector3d(1, 0, 0);
  fix.pose.pose.position.x += lever.x();
  fix.pose.pose.position.y += lever.y();
  fix.pose.covariance.fill(0.0);
  fix.pose.covariance[0] = fix.pose.covariance[7] = 1e-6;
  const int before_poses = poses, before_mappings = mappings;
  usbl_pub->publish(fix);
  spin(1.1);
  EXPECT_EQ(count("usbl_priors_applied"), 1);
  EXPECT_GT(poses, before_poses);
  EXPECT_GT(mappings, before_mappings);
  EXPECT_NEAR(pose.pose.pose.position.x, before.pose.pose.position.x, 0.2);
  EXPECT_NEAR(pose.pose.pose.position.y, before.pose.pose.position.y, 0.2);
  correction.header.frame_id = odom_frame;
  correction.pose.pose.position.x = 1000;
  fix.header.frame_id = odom_frame;
  const int before_invalid = poses;
  manual_pub->publish(correction);
  usbl_pub->publish(fix);
  spin(1.1);
  EXPECT_EQ(poses, before_invalid);
  EXPECT_GE(count("usbl_rejected"), 1);
}

TEST_F(NodeRegression, StampedManualFixUsesAcquisitionOdometryAndRejectsUnavailableTime)
{
  for (int i = 0; i < 3; ++i) ping(100 + i, i);
  ASSERT_EQ(pose.header.stamp.sec, 102);
  dynamic_tf->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{
    tf(102.99, 3), tf(103.01, 3)});
  spin(.05);
  geometry_msgs::msg::PoseWithCovarianceStamped correction;
  correction.header.frame_id = map_frame;
  correction.header.stamp.sec = 103;
  correction.pose.pose.position.x = 10;
  correction.pose.pose.orientation.w = 1;
  const int previous = poses;
  manual_pub->publish(correction);
  spin(.4);
  ASSERT_GT(poses, previous);
  // The fix locates the vehicle one metre beyond the last sonar keyframe.
  // The retained pose at 102 s must therefore move to 9 m, not 10 m.
  EXPECT_EQ(pose.header.stamp.sec, 102);
  EXPECT_NEAR(pose.pose.pose.position.x, 9, .01);
  correction.pose.pose.position.x = 100;
  const int before_invalid = poses;
  for (const auto stamp : {std::pair<int, unsigned>(500, 0), {-1, 0}, {103, 1000000000}}) {
    correction.header.stamp.sec = stamp.first;
    correction.header.stamp.nanosec = stamp.second;
    manual_pub->publish(correction);
    spin(.15);
  }
  EXPECT_EQ(poses, before_invalid);
}

TEST_F(NodeRegression, HeadingUncertaintyPropagatesIntoExtrapolatedPosition)
{
  ping(100, 0);
  ASSERT_GT(poses, 0);
  const auto parent = pose;
  ping(101, .1);  // less than the 0.2 m graph-keyframe threshold
  ASSERT_EQ(pose.header.stamp.sec, 101);
  const double displacement = pose.pose.pose.position.x - parent.pose.pose.position.x;
  ASSERT_NEAR(displacement, .1, 1e-5);
  EXPECT_NEAR(pose.pose.covariance[11], parent.pose.covariance[11] +
    displacement * parent.pose.covariance[35], 1e-8);
  EXPECT_GT(pose.pose.covariance[11], 5e-5);
}

class PhysicalNodeRegression : public NodeRegression
{
protected:
  bool physical() const override { return true; }
};

TEST_F(PhysicalNodeRegression, PreservesBodyCloudsAndPublishesSixAxisCorrections)
{
  physical_ping(100, gtsam::Pose3(), gtsam::Pose3());
  ASSERT_GT(poses, 0);
  const gtsam::Pose3 truth(gtsam::Rot3::RzRyRx(.08, -.06, .1), gtsam::Point3(.3, -.2, .4));
  const gtsam::Pose3 dr(gtsam::Rot3::RzRyRx(.18, -.15, .12), gtsam::Point3(.4, -.3, .9));
  physical_ping(102, truth, dr);
  spin(1);
  ASSERT_GT(poses, 1);
  EXPECT_NEAR(pose.pose.pose.position.z, truth.z(), .12);
  EXPECT_NEAR(sonar_slam::r2g(pose.pose.pose).rotation().roll(), truth.rotation().roll(), .03);
  EXPECT_EQ(diagnostics.at("registration_model"), "se3");
  EXPECT_EQ(diagnostics.at("state_chart"), "physical_body");
  sonar_slam::Covariance6 covariance;
  for (int i = 0; i < 36; ++i) covariance(i / 6, i % 6) = pose.pose.covariance[i];
  EXPECT_TRUE(sonar_slam::valid_covariance6(covariance));
  EXPECT_LT(covariance.diagonal().maxCoeff(), 1.0);
  geometry_msgs::msg::PoseWithCovarianceStamped correction = pose;
  const gtsam::Pose3 desired(gtsam::Rot3::RzRyRx(.22, -.12, .3), gtsam::Point3(1, .2, 1.4));
  correction.pose.pose = sonar_slam::g2r(desired);
  correction.pose.covariance.fill(0);
  for (int i = 0; i < 6; ++i) correction.pose.covariance[i * 6 + i] = 1e-5;
  const int previous = poses;
  manual_pub->publish(correction);
  spin(.5);
  EXPECT_GT(poses, previous);
  EXPECT_NEAR(pose.pose.pose.position.z, desired.z(), .03);
  EXPECT_NEAR(sonar_slam::r2g(pose.pose.pose).rotation().pitch(), desired.rotation().pitch(), .03);
  EXPECT_EQ(mapping.header.frame_id, map_frame);
  auto acoustic = pose;
  const gtsam::Point3 transponder = desired.transformFrom(gtsam::Point3(1, 0, 0));
  acoustic.pose.pose.position.x = transponder.x();
  acoustic.pose.pose.position.y = transponder.y();
  acoustic.pose.covariance.fill(0);
  acoustic.pose.covariance[0] = acoustic.pose.covariance[7] = 1e-6;
  const int before_fix = poses;
  usbl_pub->publish(acoustic);
  spin(.5);
  EXPECT_GT(poses, before_fix);
  EXPECT_NEAR(pose.pose.pose.position.x, desired.x(), .03);
  EXPECT_NEAR(pose.pose.pose.position.y, desired.y(), .03);
}

TEST_F(PhysicalNodeRegression, StampedManualPoseUsesAllSixFixTimeDimensions)
{
  physical_ping(100, gtsam::Pose3(), gtsam::Pose3());
  ASSERT_GT(poses, 0);
  const auto before = sonar_slam::r2g(pose.pose.pose);
  const gtsam::Pose3 future(gtsam::Rot3::RzRyRx(.12, -.08, .15), gtsam::Point3(1, .4, .5));
  const auto future_message = sonar_slam::g2r(future);
  auto first = tf(102.99, future.x());
  auto last = tf(103.01, future.x());
  for (auto* transform : {&first, &last}) {
    transform->transform.rotation = future_message.orientation;
    transform->transform.translation.y = future.y();
    transform->transform.translation.z = future.z();
  }
  dynamic_tf->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{first, last});
  spin(.05);
  geometry_msgs::msg::PoseWithCovarianceStamped correction;
  correction.header.frame_id = map_frame; correction.header.stamp.sec = 103;
  correction.pose.pose = future_message;
  for (int i = 0; i < 6; ++i) correction.pose.covariance[i * 6 + i] = 1e-5;
  const int previous = poses;
  manual_pub->publish(correction);
  spin(.4);
  ASSERT_GT(poses, previous);
  EXPECT_EQ(pose.header.stamp.sec, 100);
  EXPECT_LT(gtsam::Pose3::Logmap(before.between(sonar_slam::r2g(pose.pose.pose))).norm(), 1e-5);
}

class PhysicalPlanarNodeRegression : public PhysicalNodeRegression
{
protected:
  std::string manual_mode() const override { return "planar"; }
};

TEST_F(PhysicalPlanarNodeRegression, StampedManualPoseKeepsFixTimeDepthAndAttitude)
{
  physical_ping(100, gtsam::Pose3(), gtsam::Pose3());
  ASSERT_GT(poses, 0);
  const auto before = sonar_slam::r2g(pose.pose.pose);
  const gtsam::Pose3 future(gtsam::Rot3::RzRyRx(.15, -.1, .2), gtsam::Point3(1, .5, .4));
  const auto future_message = sonar_slam::g2r(future);
  auto first = tf(102.99, future.x());
  auto last = tf(103.01, future.x());
  for (auto* transform : {&first, &last}) {
    transform->transform.rotation = future_message.orientation;
    transform->transform.translation.y = future.y();
    transform->transform.translation.z = future.z();
  }
  dynamic_tf->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{first, last});
  spin(.05);
  geometry_msgs::msg::PoseWithCovarianceStamped correction;
  correction.header.frame_id = map_frame; correction.header.stamp.sec = 103;
  correction.pose.pose = sonar_slam::g2r(gtsam::Pose3(gtsam::Rot3::Yaw(future.rotation().yaw()),
    gtsam::Point3(future.x(), future.y(), 0)));
  const int previous = poses;
  manual_pub->publish(correction);
  spin(.4);
  ASSERT_GT(poses, previous);
  EXPECT_LT(gtsam::Pose3::Logmap(before.between(sonar_slam::r2g(pose.pose.pose))).norm(), 1e-5);
}
}  // namespace
