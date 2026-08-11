// SLAM bookkeeping types: port of bruce_slam slam_objects.py (STATUS,
// Keyframe, InitializationResult, ICPResult, SMParams).
#pragma once

#include <Eigen/Dense>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

// -------------------------------------------------------------------- status
struct Status
{
  enum Code {
    NOT_ENOUGH_POINTS,
    LARGE_TRANSFORMATION,
    NOT_ENOUGH_OVERLAP,
    NOT_CONVERGED,
    INITIALIZATION_FAILURE,
    SUCCESS,
  };

  Code code = SUCCESS;
  std::string description;

  Status() = default;
  Status(Code c) : code(c) {}
  explicit operator bool() const { return code == SUCCESS; }

  const char* name() const
  {
    switch (code) {
      case NOT_ENOUGH_POINTS: return "Not enough points";
      case LARGE_TRANSFORMATION: return "Large transformation";
      case NOT_ENOUGH_OVERLAP: return "Not enough overlap";
      case NOT_CONVERGED: return "Not converged";
      case INITIALIZATION_FAILURE: return "Initialization failure";
      case SUCCESS: return "Success";
    }
    return "?";
  }
};

// ---------------------------------------------------------------- conversions
inline gtsam::Pose2 pose322(const gtsam::Pose3& pose)
{
  return gtsam::Pose2(pose.x(), pose.y(), pose.rotation().yaw());
}

inline gtsam::Pose3 pose223(const gtsam::Pose2& pose)
{
  return gtsam::Pose3(gtsam::Rot3::Yaw(pose.theta()),
                      gtsam::Point3(pose.x(), pose.y(), 0.0));
}

// ------------------------------------------------------------------- keyframe
struct Keyframe
{
  Keyframe(bool status_, const builtin_interfaces::msg::Time& time_,
           const gtsam::Pose3& dr_pose3_,
           Matrix points_ = Matrix::Zero(0, 3), double head_pitch_ = 0.0)
    : status(status_), time(time_), dr_pose3(dr_pose3_),
      dr_pose(pose322(dr_pose3_)), pose3(dr_pose3_), pose(pose322(dr_pose3_)),
      points(std::move(points_)), head_pitch(head_pitch_)
  {
  }

  // update following a SLAM optimization step (slam_objects.py Keyframe.update)
  void update(const gtsam::Pose2& new_pose)
  {
    pose = new_pose;
    pose3 = gtsam::Pose3(
      gtsam::Rot3::Ypr(new_pose.theta(), dr_pose3.rotation().pitch(),
                       dr_pose3.rotation().roll()),
      gtsam::Point3(new_pose.x(), new_pose.y(), dr_pose3.z()));
    transf_points = transform_points(points, horizon_pose3());
    update_transf_cov();
  }

  void update(const gtsam::Pose2& new_pose, const Eigen::Matrix3d& new_cov)
  {
    cov = new_cov;
    has_cov = true;
    update(new_pose);
  }

  // SE3 graph update: the graph estimates
  // HORIZON poses — yaw-only rotation with z a live state — because point
  // clouds are attitude-rotated at ingestion (roll/pitch in a pose would
  // double-apply them). z comes from the estimate; roll/pitch ride from DR.
  void update(const gtsam::Pose3& est_h)
  {
    pose = gtsam::Pose2(est_h.x(), est_h.y(), est_h.rotation().yaw());
    pose3 = gtsam::Pose3(
      gtsam::Rot3::Ypr(pose.theta(), dr_pose3.rotation().pitch(),
                       dr_pose3.rotation().roll()),
      gtsam::Point3(est_h.x(), est_h.y(), est_h.z()));
    transf_points = transform_points(points, horizon_pose3());
    update_transf_cov();
  }

  void update(const gtsam::Pose3& est_h, const Eigen::Matrix3d& new_cov)
  {
    cov = new_cov;
    has_cov = true;
    update(est_h);
  }

  // Covariance-only refresh: update cov and the global-frame transf_cov WITHOUT
  // re-transforming the point cloud. transf_points depends only on pose+points,
  // so when only the marginal covariance changed (pose unchanged) this avoids a
  // redundant O(cloud) transform (used on the loop-closure marginal refresh).
  void set_cov(const Eigen::Matrix3d& new_cov)
  {
    cov = new_cov;
    has_cov = true;
    update_transf_cov();
  }

  // the keyframe's current estimate as a horizon pose (graph state chart)
  gtsam::Pose3 horizon_pose3() const
  {
    return gtsam::Pose3(gtsam::Rot3::Yaw(pose.theta()),
                        gtsam::Point3(pose.x(), pose.y(), pose3.z()));
  }

  // the DR measurement as a horizon pose (planar dr_pose + measured depth)
  gtsam::Pose3 horizon_dr_pose3() const
  {
    return gtsam::Pose3(gtsam::Rot3::Yaw(dr_pose.theta()),
                        gtsam::Point3(dr_pose.x(), dr_pose.y(), dr_pose3.z()));
  }

  static Matrix transform_points(const Matrix& pts, const gtsam::Pose2& pose)
  {
    if (pts.rows() == 0) return Matrix::Zero(0, pts.cols());
    const float c = static_cast<float>(std::cos(pose.theta()));
    const float s = static_cast<float>(std::sin(pose.theta()));
    Eigen::Matrix2f R;
    R << c, -s, s, c;
    Matrix out = pts;
    out.leftCols(2) = pts.leftCols(2) * R.transpose();
    out.col(0).array() += static_cast<float>(pose.x());
    out.col(1).array() += static_cast<float>(pose.y());
    return out;
  }

  // Horizon-frame transform used by constrained 3-D registration.  The graph
  // pose contains yaw + xyz; roll/pitch have already been removed once at
  // ping ingestion.  Applying only horizon yaw here avoids double attitude
  // rotation while preserving pressure-depth separation between scans.
  static Matrix transform_points(const Matrix& pts, const gtsam::Pose3& pose)
  {
    Matrix out = transform_points(pts, pose322(pose));
    if (out.cols() >= 3)
      out.col(2).array() += static_cast<float>(pose.z());
    return out;
  }

  bool status = false;
  builtin_interfaces::msg::Time time;

  gtsam::Pose3 dr_pose3;   // dead reckoning 3d pose
  gtsam::Pose2 dr_pose;    // dead reckoning 2d pose
  gtsam::Pose3 pose3;      // estimated 3d pose
  gtsam::Pose2 pose;       // estimated 2d pose

  bool has_cov = false;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();        // local frame
  Eigen::Matrix3d transf_cov = Eigen::Matrix3d::Zero(); // global frame

  Matrix points;         // local horizon-referenced frame, Nx3 (x, y, elev)
  Matrix transf_points;  // global horizon frame, Nx3 (x/y + pressure depth)

  // Sonar optical-boresight elevation at the ping stamp.  It participates in
  // keyframe selection so a stationary mechanical head sweep still builds a
  // multi-angle local submap instead of being discarded as "no motion".
  double head_pitch = 0.0;

  // non-sequential constraints aka loop closures: (target key, transform)
  std::vector<std::pair<int, gtsam::Pose2>> constraints;

  geometry_msgs::msg::Twist twist;

private:
  void update_transf_cov()
  {
    if (!has_cov) return;
    const double c = std::cos(pose.theta()), s = std::sin(pose.theta());
    Eigen::Matrix2d R;
    R << c, -s, s, c;
    transf_cov = cov;
    transf_cov.topLeftCorner<2, 2>() = R * cov.topLeftCorner<2, 2>() * R.transpose();
    transf_cov.topRightCorner<2, 1>() = R * cov.topRightCorner<2, 1>();
    transf_cov.bottomLeftCorner<1, 2>() = cov.bottomLeftCorner<1, 2>() * R.transpose();
  }
};

using KeyframePtr = std::shared_ptr<Keyframe>;

// -------------------------------------------------- scan-matching containers
struct InitializationResult
{
  Matrix source_points = Matrix::Zero(0, 2);
  Matrix target_points = Matrix::Zero(0, 2);
  int source_key = -1;
  int target_key = -1;
  gtsam::Pose2 source_pose;
  gtsam::Pose2 target_pose;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  Status status;
  bool has_estimated_source_pose = false;
  gtsam::Pose2 estimated_source_pose;
  // [x, y, theta, cost] samples from global init
  std::vector<Eigen::Vector4d> source_pose_samples;
};

struct ICPResult
{
  ICPResult(const InitializationResult& init, bool use_samples,
            double sample_eps = 0.01)
    : source_points(init.source_points), target_points(init.target_points),
      source_key(init.source_key), target_key(init.target_key),
      source_pose(init.source_pose), target_pose(init.target_pose),
      status(init.status)
  {
    if (init.has_estimated_source_pose)
      initial_transform = target_pose.between(init.estimated_source_pose);
    else
      initial_transform = target_pose.between(source_pose);

    if (use_samples && !init.source_pose_samples.empty()) {
      // sort sampled source poses by cost, convert to transforms in the
      // target frame, filter near-duplicates (slam_objects.py ICPResult)
      std::vector<int> idx(init.source_pose_samples.size());
      for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
      std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return init.source_pose_samples[a][3] < init.source_pose_samples[b][3];
      });
      std::vector<gtsam::Pose2> transforms;
      transforms.reserve(idx.size());
      for (int i : idx) {
        const auto& s = init.source_pose_samples[i];
        transforms.push_back(
          target_pose.between(gtsam::Pose2(s[0], s[1], s[2])));
      }
      initial_transforms.push_back(transforms[0]);
      for (std::size_t i = 1; i < transforms.size(); ++i) {
        const gtsam::Pose2 d = initial_transforms.back().between(transforms[i]);
        const double dist = std::sqrt(d.x() * d.x() + d.y() * d.y() +
                                      d.theta() * d.theta());
        if (dist >= sample_eps) initial_transforms.push_back(transforms[i]);
      }
    }
  }

  Matrix source_points;
  Matrix target_points;
  int source_key;
  int target_key;
  gtsam::Pose2 source_pose;
  gtsam::Pose2 target_pose;
  Status status;

  bool has_estimated_transform = false;
  gtsam::Pose2 estimated_transform;
  gtsam::Pose2 initial_transform;
  std::vector<gtsam::Pose2> initial_transforms;

  bool has_cov = false;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d cov_raw = Eigen::Matrix3d::Zero();  // pre-floor cov (degeneracy gate)
  bool inserted = false;
  // quarantined by a post-loop verification revert: this closure (as part of
  // its clique) demonstrably bent the graph, so it must not be re-inserted or
  // vote in PCM cliques again — without this the same clique re-forms on the
  // next candidate, re-fails verify, and triggers an O(map) estimator rebuild
  // every NSSM round until the queue window slides past it.
  bool rejected = false;
  int n_sample_transforms = 0;
};

// ------------------------------------------------------ scan-matching params
struct SMParams
{
  // how the ICP factor covariance is estimated when cov_samples > 0:
  //   SAMPLED - run cov_samples ICPs from the best init guesses and take a
  //             robust (FAST-MCD) covariance of the results (slam.py default)
  //   CENSI   - one ICP plus the Censi (2007) closed form. Its Hessian is
  //             point-to-POINT, so Slam::configure accepts it only when the
  //             LOADED ICP chain minimises the same objective (it reads
  //             ICP::error_minimizer_name(); the package default icp.yaml is
  //             point-to-plane, so censi is rejected there)
  enum CovMethod { SAMPLED, CENSI };

  bool enable = true;
  bool initialization = true;
  // (sobol samples per iteration, iterations, local-refine ftol)
  int init_n = 50;
  int init_iters = 1;
  double init_ftol = 0.01;

  int min_points = 50;
  // Minimum fraction of the SOURCE cloud that must find a correspondence in
  // the target after registration (rtabmap Icp/CorrespondenceRatio, default
  // 0.1). 0 disables.
  //
  // Complements min_points, which is an ABSOLUTE count and therefore
  // venue-coupled: pool keyframes carry 80-160 points and want 60, field
  // keyframes carry 19-63 and 60 kills every closure there (venue/*.yaml
  // exists solely to carry that difference). A ratio asks the
  // scale-independent question — "did most of what I can see actually
  // match?" — so it should port across venues where the count cannot.
  double min_overlap_ratio = 0.0;
  double max_translation = 2.0;
  double max_rotation = M_PI / 6.0;
  int min_st_sep = 1;
  // Safety floor on the source->target keyframe index gap for a loop closure:
  // frames closer in index than this can never become the target. NOT the main
  // trail/revisit separator — that is the contiguity clustering in
  // initialize_nonsequential_scan_matching (the in-fan trail spans
  // ~max_range/keyframe_translation keyframes, far more than any usable floor).
  // This just guards the degenerate case where a revisit run abuts the trail.
  int min_revisit_sep = 10;
  int source_frames = 5;
  int target_frames = 3;
  // Bounded-drift rates used to inflate the anchor keyframe's fresh covariance
  // when padding the candidate fan for older source frames (whose stored
  // covariances are stale — only the newest keyframe's is refreshed by
  // update_factor_graph). Added as (rate * age_seconds)^2 to the diagonal.
  double fan_drift_trans = 0.01;    // m / s
  double fan_drift_rot = 0.0017;    // rad / s (~0.1 deg/s)
  int cov_samples = 0;
  CovMethod cov_method = SAMPLED;
};

}  // namespace sonar_slam
