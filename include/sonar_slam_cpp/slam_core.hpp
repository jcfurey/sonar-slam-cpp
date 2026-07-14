// The SLAM back-end: port of bruce_slam slam.py. Graph-based pose SLAM with
// ISAM2, sequential scan matching (SSM), non-sequential scan matching (NSSM,
// loop closures) verified by pairwise consistency maximization (PCM).
#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <deque>
#include <optional>

#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/keyframe.hpp"
#include "sonar_slam_cpp/sonar_geometry.hpp"

namespace sonar_slam {

class Slam
{
public:
  Slam();

  // sanity checks + noise model construction (slam.py configure())
  void configure();

  // ------------------------------------------------------------- parameters
  double keyframe_duration = 1.0;
  double keyframe_translation = 3.0;
  double keyframe_rotation = 0.5;

  Eigen::Vector3d prior_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);
  Eigen::Vector3d odom_sigmas = Eigen::Vector3d(0.2, 0.2, 0.02);
  Eigen::Vector3d icp_odom_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);

  double point_resolution = 0.5;  // downsampling for ICP / publishing
  double point_noise = 0.5;       // noise radius in overlap estimation
  // per-point sonar coordinate noise std (m) for the Censi covariance path
  double censi_sensor_noise = 0.1;

  SMParams ssm_params;
  SMParams nssm_params;

  int pcm_queue_size = 5;
  int min_pcm = 3;

  OculusProperty oculus;

  // ----------------------------------------------------------------- state
  std::vector<KeyframePtr> keyframes;
  KeyframePtr current_frame;

  ICP icp;

  int ssm_accepted = 0;
  int nssm_accepted = 0;

  const KeyframePtr& current_keyframe() const { return keyframes.back(); }
  int current_key() const { return static_cast<int>(keyframes.size()); }

  // ------------------------------------------------------------- pipeline
  bool is_keyframe(const Keyframe& frame) const;
  void add_prior(const KeyframePtr& keyframe);
  void add_odometry(const KeyframePtr& keyframe);
  void add_sequential_scan_matching(const KeyframePtr& keyframe);
  // returns true when a loop closure was accepted into the graph
  bool add_nonsequential_scan_matching();
  // Incorporate the buffered factors/values into ISAM2. Returns false when the
  // solve failed: the offending frame's factors (and any loop closures marked
  // this round) are rolled back, the estimator is rebuilt from the previously
  // committed factors, and last_error() describes the failure. The caller
  // should log it and skip any further graph work this callback.
  bool update_factor_graph(const KeyframePtr& keyframe = nullptr);
  const std::string& last_error() const { return last_error_; }

  // ------------------------------------------------------------- utilities
  // accumulate points of `frames` in the frame of `ref_key` (or global when
  // ref_key < 0), voxel-downsampled
  Matrix get_points(const std::vector<int>& frames, int ref_key) const;
  // same, also returning the keyframe key of every point
  std::pair<Matrix, std::vector<int>> get_points_with_keys(
    const std::vector<int>& frames) const;

  std::pair<std::string, gtsam::Pose2> compute_icp(
    const Matrix& source_points, const Matrix& target_points,
    const gtsam::Pose2& guess);

  struct IcpCovResult
  {
    std::string message;
    gtsam::Pose2 odom;
    Eigen::Matrix3d cov;
    int n_samples = 0;
  };
  IcpCovResult compute_icp_with_cov(const Matrix& source_points,
                                    const Matrix& target_points,
                                    const std::vector<gtsam::Pose2>& guesses);

  int get_overlap(const Matrix& source_points, const Matrix& target_points,
                  const gtsam::Pose2* source_pose = nullptr,
                  std::vector<int>* indices_out = nullptr) const;

  // PCM over the loop-closure queue; returns indices into the queue
  std::vector<int> verify_pcm(const std::deque<ICPResult>& queue,
                              int min_pcm_value) const;

private:
  InitializationResult initialize_sequential_scan_matching(
    const KeyframePtr& keyframe);
  InitializationResult initialize_nonsequential_scan_matching();

  // run ICP and (when cov_samples > 0) estimate its covariance by the
  // configured method, filling ret2.estimated_transform / cov / status.
  // Shared by the SSM and NSSM pipelines.
  void run_scan_match_icp(ICPResult& ret2, const SMParams& params);

  gtsam::SharedNoiseModel create_noise_model(const Eigen::Vector3d& sigmas) const;
  gtsam::SharedNoiseModel create_full_noise_model(const Eigen::Matrix3d& cov) const;

  // failure recovery for update_factor_graph: undo loop-closure bookkeeping
  // written before a failed solve, and reconstruct ISAM2 from the committed
  // factor mirror (a thrown ISAM2::update leaves the estimator in an
  // undefined, partially-mutated state — GTSAM gives no exception guarantee)
  void rollback_pending_loops();
  void rebuild_isam();

  gtsam::ISAM2 isam_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  // mirror of every factor that has been through a SUCCESSFUL isam_.update —
  // the rebuild source when an update throws
  gtsam::NonlinearFactorGraph committed_graph_;
  // indices into nssm_queue_ of loops marked inserted since the last
  // successful update (rolled back if that update fails)
  std::vector<int> pending_loops_;
  std::string last_error_;

  gtsam::SharedNoiseModel prior_model_, odom_model_, icp_odom_model_;

  std::deque<ICPResult> nssm_queue_;
};

}  // namespace sonar_slam
