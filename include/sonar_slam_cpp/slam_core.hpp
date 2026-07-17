// The SLAM back-end: port of bruce_slam slam.py. Graph-based pose SLAM with
// ISAM2, sequential scan matching (SSM), non-sequential scan matching (NSSM,
// loop closures) verified by pairwise consistency maximization (PCM).
#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <deque>
#include <map>
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
  // Minimum feature points for a frame to become a keyframe (0 = off,
  // upstream parity). Genuinely empty clouds (~8% of pings on the pool bags)
  // otherwise become 0-point keyframes that dilute the graph and deposit
  // empty map tiles — MAP_DOUBLING_FIX_PLAN.md fix 3b.
  int keyframe_min_points = 0;

  Eigen::Vector3d prior_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);
  Eigen::Vector3d odom_sigmas = Eigen::Vector3d(0.2, 0.2, 0.02);
  Eigen::Vector3d icp_odom_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);

  double point_resolution = 0.5;  // downsampling for ICP / publishing
  double point_noise = 0.5;       // noise radius in overlap estimation
  // per-point sonar coordinate noise std (m) for the Censi covariance path
  double censi_sensor_noise = 0.1;
  // run the sampled-covariance registrations (cov_samples per scan match)
  // across an OpenMP per-thread ICP engine pool instead of one core
  // sequentially. Per-guess results are unchanged (deterministic chain); the
  // only divergence is that the 2 s cap rarely fires, so MCD sees the full
  // sample set. See docs/DIVERGENCES.md.
  bool parallel_cov_samples = true;

  SMParams ssm_params;
  SMParams nssm_params;

  int pcm_queue_size = 5;
  int min_pcm = 3;

  // NSSM degeneracy gate (wall-aliasing closures are self-consistent, so PCM
  // alone cannot reject them): reject a closure whose sampled/Censi
  // translation covariance is too large or too elongated (sliding along a
  // featureless wall). Pair with a tight nssm/max_rotation — yaw corrections
  // beyond compass noise are bogus by construction in this anchored frame.
  double nssm_max_sigma = 0.5;       // max sqrt(largest translation eigenvalue), m
  double nssm_max_anisotropy = 8.0;  // max sigma_max / sigma_min
  // ABSOLUTE yaw gate against the compass (kills discrete rotational
  // aliasing — e.g. a near-square pool aliases at 90°, which ICP matches
  // confidently and symmetric pairs pass PCM): the closure's relative yaw
  // must agree with the compass-anchored DR relative yaw within this bound.
  // Neither max_rotation (ICP-vs-Sobol refinement only) nor the covariance
  // gate (catches sliding, not confident-but-wrong locks) covers this mode.
  double nssm_max_yaw_vs_compass = 0.15;  // rad (~8.6°, > compass noise)
  // optimize-then-verify (rtabmap RGBD/OptimizeMaxError analog): after a
  // loop-closure insert, the whole graph's optimized-vs-DR yaw RMS must stay
  // under this bound or the loops are rolled back (see update_factor_graph)
  double post_loop_max_yaw_rms = 0.15;  // rad
  // post-loop CHAIN-TEAR check: max allowed |optimized - DR| consecutive
  // keyframe separation (m). Parallel-wall translational aliases pass every
  // per-closure gate (compass agrees between parallel walls, wall-to-wall
  // locks are compact, mutually-consistent aliases satisfy PCM) — but to
  // win they must stretch weak sequential links by many meters. DR is
  // drift-free over one ~0.75 m keyframe interval, so a large tear is
  // unambiguous. 0 disables.
  double post_loop_max_translation_err = 1.0;  // m

  OculusProperty oculus;

  // ----------------------------------------------------------------- state
  std::vector<KeyframePtr> keyframes;
  KeyframePtr current_frame;

  ICP icp;

  int ssm_accepted = 0;
  int nssm_accepted = 0;
  // most recent SSM outcome ("accepted" or "<Status name>: <description>") —
  // surfaced in the periodic status log so a 0-factor run explains itself
  std::string last_ssm_status = "none yet";
  // same for NSSM, plus pipeline counters: attempts that got past the
  // st_sep/current_frame gate, candidates that survived every per-closure
  // gate into the PCM queue, and the largest consistent clique seen
  std::string last_nssm_status = "none yet";
  int nssm_attempts = 0;
  int nssm_queued = 0;
  int nssm_best_clique = 0;
  int nssm_queue_depth() const { return static_cast<int>(nssm_queue_.size()); }
  // cumulative histogram of terminal NSSM rejection reasons over the whole run,
  // keyed by Status category (LARGE_TRANSFORMATION is split into "(compass)" vs
  // geometric so the tunable rot/trans bar stays distinct from the untouchable
  // aliasing guard). Turns the sampled "last NSSM" reason into a full
  // distribution so tuning targets the biggest bar, not whatever fired last.
  std::map<std::string, int> nssm_reject_hist;
  std::string nssm_reject_summary() const;
  // PCM introspection: the smallest pairwise Mahalanobis distance seen in
  // the last verify (how close the queue is to forming an edge; threshold
  // 11.34 = chi2.ppf(0.99, 3)) and how many consistency edges formed
  double last_pcm_min_md = -1.0;
  int last_pcm_edges = 0;

  const KeyframePtr& current_keyframe() const { return keyframes.back(); }
  int current_key() const { return static_cast<int>(keyframes.size()); }

  // ------------------------------------------------------------- pipeline
  bool is_keyframe(const Keyframe& frame) const;
  void add_prior(const KeyframePtr& keyframe);
  void add_odometry(const KeyframePtr& keyframe);
  // absolute depth + zero-attitude prior on X(key) (Phase 4 horizon chart)
  void add_state_unary(int key, const KeyframePtr& keyframe);
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

  // PCM over the loop-closure queue; returns indices into the queue.
  // Non-const: records last_pcm_min_md / last_pcm_edges introspection.
  std::vector<int> verify_pcm(const std::deque<ICPResult>& queue,
                              int min_pcm_value);

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
  // quarantine=true additionally marks the rolled-back loops rejected (a
  // verification revert: the clique is demonstrably bad); false leaves them
  // eligible for retry (a solver failure says nothing about the loops)
  void rollback_pending_loops(bool quarantine);
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

  // 6D models over the Pose3 tangent (rx, ry, rz, tx, ty, tz); see configure()
  gtsam::SharedNoiseModel prior_model_, odom_model_, icp_odom_model_;
  // per-keyframe absolute depth + zero-attitude prior (wide x/y/yaw)
  gtsam::SharedNoiseModel unary_model_;

  std::deque<ICPResult> nssm_queue_;
};

}  // namespace sonar_slam
