#pragma once

#include "sonar_slam_cpp/keyframe.hpp"

namespace sonar_slam {
class Slam;

struct Se3Options
{
  RigidIcpParams icp;
  // Configured uncertainty of fused TF increments, never additional raw
  // IMU/pressure factors. Order [roll,pitch,yaw,x,y,z], in rad and metres.
  gtsam::Vector6 prior_sigmas = (gtsam::Vector6() << .1, .1, .1, .1, .1, .1).finished();
  gtsam::Vector6 odom_sigmas = (gtsam::Vector6() << .1, .1, .1, .3, .3, .15).finished();
  double max_rotation_sigma = .25;
  double relocalize_depth_range = 1.0;
  double relocalize_attitude_range = .15;
  int max_source_points = 5000;
  int max_seed_points = 250000;
};

// Physical Pose3 graph. Kept separate from the horizon graph so adding a new
// measurement model cannot silently change the established constrained path.
class Se3Backend
{
public:
  explicit Se3Backend(Slam& owner);
  ~Se3Backend();
  void reset();
  bool is_keyframe(const Keyframe& frame) const;
  void add_prior(const KeyframePtr& frame);
  void add_odometry(const KeyframePtr& frame);
  void add_sequential(const KeyframePtr& frame);
  bool add_nonsequential();
  bool update(const KeyframePtr& frame);
  bool has_pending_update() const;
  void discard_pending_update();
  bool add_manual(const gtsam::Pose2& pose);
  bool add_manual(const gtsam::Pose3& pose, const Covariance6& covariance);
  bool undo_manual();
  bool add_position(int key, double x, double y, double sigma,
                    const gtsam::Point3& point_in_key = gtsam::Point3::Zero());
  bool save(const std::string& path);
  bool load(const std::string& path);
  bool relocalize(const KeyframePtr& frame, Deadline deadline = Deadline::max());
  int queue_depth() const;
  int manual_count() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace sonar_slam
