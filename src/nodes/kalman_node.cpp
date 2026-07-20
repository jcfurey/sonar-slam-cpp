// Kalman filter node: port of bruce_slam kalman.py. 12-state filter
// (x, y, z, roll, pitch, yaw and rates) corrected by DVL, IMU, FOG and depth.
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <optional>

#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/node_base.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"
#include "sonar_slam_cpp/sensors.hpp"

namespace sonar_slam {

class KalmanNode : public SlamNodeBase
{
public:
  using Vec12 = Eigen::Matrix<double, 12, 1>;
  using Mat12 = Eigen::Matrix<double, 12, 12>;
  using Mat3x12 = Eigen::Matrix<double, 3, 12>;

  KalmanNode() : SlamNodeBase("kalman")
  {
    // matrices are stored flattened row-major in the YAML
    state_vector_ = as_vector<12>(get_double_array("state_vector"));
    cov_matrix_ = as_matrix<12, 12>(get_double_array("cov_matrix"));
    R_dvl_ = as_matrix<3, 3>(get_double_array("R_dvl"));
    dt_dvl_ = get_double("dt_dvl");
    H_dvl_ = as_matrix<3, 12>(get_double_array("H_dvl"));
    R_imu_ = as_matrix<3, 3>(get_double_array("R_imu"));
    dt_imu_ = get_double("dt_imu");
    H_imu_ = as_matrix<3, 12>(get_double_array("H_imu"));
    H_gyro_ = as_matrix<3, 12>(get_double_array("H_gyro"));
    R_gyro_ = as_matrix<3, 3>(get_double_array("R_gyro"));
    dt_gyro_ = get_double("dt_gyro");
    H_depth_ = as_matrix<3, 12>(get_double_array("H_depth"));
    R_depth_ = as_matrix<3, 3>(get_double_array("R_depth"));
    dt_depth_ = get_double("dt_depth");
    Q_ = as_matrix<12, 12>(get_double_array("Q"));
    A_imu_ = as_matrix<12, 12>(get_double_array("A_imu"));

    const double x = get_double("offset/x") * M_PI / 180.0;
    const double y = get_double("offset/y") * M_PI / 180.0;
    const double z = get_double("offset/z") * M_PI / 180.0;
    offset_matrix_ = (Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX()))
                       .toRotationMatrix();

    dvl_max_velocity_ = get_double("dvl_max_velocity");
    use_gyro_ = get_bool("use_gyro");
    imu_offset_ = get_double("imu_offset") * M_PI / 180.0;

    const std::string dvl_driver = get_string("dvl/driver", "rti_dvl");
    const std::string depth_driver = get_string("depth/driver", "bar30");
    const std::string imu_driver = get_string("imu/driver", "vn100");

    std::string imu_topic = get_string("imu/topic", "");
    if (imu_topic.empty()) {
      if (imu_driver_is_legacy(imu_driver))
        imu_topic = get_int("imu_version", 1) == 1 ? IMU_TOPIC : IMU_TOPIC_MK_II;
      else
        imu_topic = IMU_TOPIC_ENU;
    }

    imu_sub_ = subscribe_imu(this, imu_driver, imu_topic, rclcpp::SensorDataQoS(rclcpp::KeepLast(250)),
                             [this](const ImuReading& r) { imu_callback(r); });
    dvl_sub_ = subscribe_dvl(this, dvl_driver, get_string("dvl/topic", DVL_TOPIC),
                             rclcpp::SensorDataQoS(rclcpp::KeepLast(250)),
                             [this](const DvlReading& r) { dvl_callback(r); });
    depth_sub_ = subscribe_depth(
      this, depth_driver, get_string("depth/topic", DEPTH_TOPIC), rclcpp::SensorDataQoS(rclcpp::KeepLast(250)),
      [this](const DepthReading& r) { pressure_callback(r); });

    if (use_gyro_) {
      gyro_sub_ = subscribe_gyro(
        this, get_string("gyro/driver", "kvh_gyro"),
        get_string("gyro/topic", GYRO_TOPIC), rclcpp::SensorDataQoS(rclcpp::KeepLast(250)),
        [this](const GyroReading& r) { gyro_callback(r); });
    }

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(LOCALIZATION_ODOM_TOPIC, 250);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    pose_ = gtsam::Pose3(gtsam::Rot3::Ypr(0., 0., 0.), gtsam::Point3(0, 0, 0));

    RCLCPP_INFO(get_logger(), "Kalman Node is initialized");
  }

private:
  template <int N>
  static Eigen::Matrix<double, N, 1> as_vector(const std::vector<double>& v)
  {
    Eigen::Matrix<double, N, 1> out;
    for (int i = 0; i < N; ++i) out[i] = v.at(i);
    return out;
  }

  template <int R, int C>
  static Eigen::Matrix<double, R, C> as_matrix(const std::vector<double>& v)
  {
    Eigen::Matrix<double, R, C> out;
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) out(r, c) = v.at(r * C + c);
    return out;
  }

  void kalman_predict(const Vec12& previous_x, const Mat12& previous_P,
                      const Mat12& A, Vec12& predicted_x, Mat12& predicted_P) const
  {
    predicted_P = A * previous_P * A.transpose() + Q_;
    predicted_x = A * previous_x;
  }

  void kalman_correct(const Vec12& predicted_x, const Mat12& predicted_P,
                      const Eigen::Vector3d& z, const Mat3x12& H,
                      const Eigen::Matrix3d& R, Vec12& corrected_x,
                      Mat12& corrected_P) const
  {
    const Eigen::Matrix<double, 12, 3> K =
      predicted_P * H.transpose() * (H * predicted_P * H.transpose() + R).inverse();
    corrected_x = predicted_x + K * (z - H * predicted_x);
    // Joseph form keeps P symmetric and positive-definite under finite
    // precision over long high-rate (~250 Hz) runs, unlike the naive P - KHP.
    const Mat12 ImKH = Mat12::Identity() - K * H;
    corrected_P = ImKH * predicted_P * ImKH.transpose() + K * R * K.transpose();
  }

  void gyro_callback(const GyroReading& reading)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const Eigen::Vector3d arr = offset_matrix_.transpose() * reading.delta;
    const Eigen::Vector3d meas(arr[0], 0.0, 0.0);
    kalman_correct(state_vector_, cov_matrix_, meas, H_gyro_, R_gyro_,
                   state_vector_, cov_matrix_);
    yaw_gyro_ += state_vector_[11];
  }

  void dvl_callback(const DvlReading& dvl)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (dvl.velocity.cwiseAbs().maxCoeff() > dvl_max_velocity_) return;
    kalman_correct(state_vector_, cov_matrix_, dvl.velocity, H_dvl_, R_dvl_,
                   state_vector_, cov_matrix_);
  }

  void pressure_callback(const DepthReading& depth)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const Eigen::Vector3d meas(depth.depth, 0.0, 0.0);
    kalman_correct(state_vector_, cov_matrix_, meas, H_depth_, R_depth_,
                   state_vector_, cov_matrix_);
  }

  void imu_callback(const ImuReading& imu)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Calculate measured dt FIRST
    double dt = dt_imu_;
    const double t_now = to_sec(imu.stamp);
    if (last_imu_stamp_ > 0.0) {
      const double measured = t_now - last_imu_stamp_;
      if (measured >= 0.0 && measured < 10.0 * dt_imu_) dt = measured;
    }
    last_imu_stamp_ = t_now;

    // 2. Update the transition matrix dynamically
    A_imu_(0, 6) = dt; A_imu_(1, 7) = dt; A_imu_(2, 8) = dt; // Pos = Pos + Vel*dt
    A_imu_(3, 9) = dt; A_imu_(4, 10) = dt; A_imu_(5, 11) = dt; // Ang = Ang + Rate*dt

    // 3. Now run the prediction
    Vec12 predicted_x;
    Mat12 predicted_P;
    kalman_predict(state_vector_, cov_matrix_, A_imu_, predicted_x, predicted_P);

    // scipy's extrinsic 'xyz' euler decomposition (R = Rz(yaw)Ry(pitch)Rx(roll))
    // is exactly gtsam's roll/pitch/yaw convention
    const gtsam::Rot3 rot = gtsam::Rot3::Quaternion(
      imu.orientation.w(), imu.orientation.x(), imu.orientation.y(),
      imu.orientation.z());
    const double roll_x = rot.roll();
    const double pitch_y = rot.pitch();
    const double yaw_z = rot.yaw();

    Eigen::Vector3d euler_angle(imu_offset_ + roll_x, pitch_y, yaw_z);
    if (!imu_yaw0_) imu_yaw0_ = yaw_z;
    euler_angle[2] -= *imu_yaw0_;

    // Wrap the ANGLE innovations: when the vehicle's yaw (or a legacy-frame
    // roll) crosses +-pi the raw measurement jumps by 2*pi against the
    // continuous filtered state, and the unwrapped residual would slew every
    // state through the gain for the whole convergence window (inherited
    // from kalman.py; fixed by decision — correctness over parity).
    const Eigen::Vector3d predicted_euler = H_imu_ * predicted_x;
    for (int i : {0, 2})
      euler_angle[i] = predicted_euler[i] +
                       std::remainder(euler_angle[i] - predicted_euler[i],
                                      2.0 * M_PI);

    kalman_correct(predicted_x, predicted_P, euler_angle, H_imu_, R_imu_,
                   state_vector_, cov_matrix_);

    // Use the filtered velocity to update x/y over the MEASURED stamp dt.
    // The nominal dt_imu_ made travelled distance scale with message COUNT:
    // a 1% rate mismatch (or best-effort QoS drops under load) was a 1% scale
    // error on the whole trajectory. A_imu_ keeps the configured nominal dt
    // (its dt entries are config data), where the residual error is
    // second-order; the position integration here is where it accumulates
    // linearly. Clamped: a gap (dropout/bag jump) must not integrate stale
    // velocity into a jump; negative dt (bag loop) re-anchors the timebase.
    // dt was measured at the top of the callback (shared with the A_imu_
    // transition matrix) and last_imu_stamp_ already advanced there.
    const double trans_x = state_vector_[6] * dt;
    const double trans_y = state_vector_[7] * dt;
    const gtsam::Point2 local_point(trans_x, trans_y);

    gtsam::Rot3 R;
    gtsam::Pose2 pose2;
    if (use_gyro_) {
      R = gtsam::Rot3::Ypr(yaw_gyro_, state_vector_[4], state_vector_[3]);
      pose2 = gtsam::Pose2(pose_.x(), pose_.y(), yaw_gyro_);
    } else {
      R = gtsam::Rot3::Ypr(state_vector_[5], state_vector_[4], state_vector_[3]);
      pose2 = gtsam::Pose2(pose_.x(), pose_.y(), pose_.rotation().yaw());
    }

    const gtsam::Point2 point = pose2.transformFrom(local_point);
    pose_ = gtsam::Pose3(R, gtsam::Point3(point.x(), point.y(), 0));
    send_odometry(imu.stamp);
  }

  void send_odometry(const builtin_interfaces::msg::Time& t)
  {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = t;
    odom_msg.header.frame_id = "odom";
    odom_msg.pose.pose = g2r(pose_);
    odom_msg.child_frame_id = "base_link";
    odom_pub_->publish(odom_msg);

    const auto& p = odom_msg.pose.pose.position;
    const auto& q = odom_msg.pose.pose.orientation;
    tf_->sendTransform(make_transform(Eigen::Vector3d(p.x, p.y, p.z),
                                      Eigen::Quaterniond(q.w, q.x, q.y, q.z),
                                      t, "odom", "base_link"));
  }

  Vec12 state_vector_ = Vec12::Zero();
  Mat12 cov_matrix_ = Mat12::Zero();
  double last_imu_stamp_ = 0.0;
  Mat12 Q_ = Mat12::Zero(), A_imu_ = Mat12::Identity();
  Eigen::Matrix3d R_dvl_, R_imu_, R_gyro_, R_depth_;
  Mat3x12 H_dvl_, H_imu_, H_gyro_, H_depth_;
  double dt_dvl_ = 0.2, dt_imu_ = 0.005, dt_gyro_ = 0.004, dt_depth_ = 0.25;
  Eigen::Matrix3d offset_matrix_ = Eigen::Matrix3d::Identity();
  double dvl_max_velocity_ = 0.5;
  bool use_gyro_ = false;
  double imu_offset_ = 0.0;
  double yaw_gyro_ = 0.0;
  std::optional<double> imu_yaw0_;
  gtsam::Pose3 pose_;
  std::mutex mutex_;

  rclcpp::SubscriptionBase::SharedPtr imu_sub_, dvl_sub_, depth_sub_, gyro_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace sonar_slam

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sonar_slam::KalmanNode>());
  rclcpp::shutdown();
  return 0;
}
