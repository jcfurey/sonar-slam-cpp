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
  // Reserved by the disabled point-to-point Censi reference implementation.
  double censi_sensor_noise = 0.1;
  // run the sampled-covariance registrations (cov_samples per scan match)
  // across an OpenMP per-thread ICP engine pool instead of one core
  // sequentially. Per-guess results are unchanged (deterministic chain); the
  // only divergence is that the 2 s cap rarely fires, so MCD sees the full
  // sample set.
  bool parallel_cov_samples = true;

  SMParams ssm_params;
  SMParams nssm_params;

  // SSM must fail closed when sampled/Censi covariance is unavailable or says
  // the registration has a large/weak translation axis. Sparse/open-water
  // periods then use only the fused-odometry link.
  bool ssm_require_covariance = true;
  double ssm_max_sigma = 0.5;
  double ssm_max_anisotropy = 8.0;
  bool ssm_degeneracy_prefloor = true;

  int pcm_queue_size = 5;
  int min_pcm = 3;

  // NSSM degeneracy gate (wall-aliasing closures are self-consistent, so PCM
  // alone cannot reject them): reject a closure whose sampled/Censi
  // translation covariance is too large or too elongated (sliding along a
  // featureless wall). Pair with a tight nssm/max_rotation — yaw corrections
  // beyond compass noise are bogus by construction in this anchored frame.
  double nssm_max_sigma = 0.5;       // max sqrt(largest translation eigenvalue), m
  double nssm_max_anisotropy = 8.0;  // max sigma_max / sigma_min
  // Evaluate the degeneracy gate on the PRE-floor covariance (opt-in). The
  // per-eigenvalue floor in compute_icp_with_cov raises the collapsed axis up to
  // the ICP-model sigma before this gate reads it, capping observable anisotropy
  // and letting elongated wall-slide covariances pass. True => gate on the raw
  // (pre-floor) covariance so the elongation the gate exists to catch survives.
  bool nssm_degeneracy_prefloor = false;
  // Robust kernel on loop factors. Default OFF: DCS scales a factor's weight
  // by 2*phi/(phi+chi2), and a GENUINE closure correcting drift D with ICP
  // sigma s starts at chi2=(D/s)^2 — for D=1 m, s=5 cm that is 400, i.e. 0.5%
  // weight: exactly the corrections SLAM exists to make are the ones muted.
  // Outlier protection is the optimize-then-verify revert + quarantine, which
  // judges the ACTUAL optimized graph instead of pre-shrinking every
  // correction. Set true (with nssm_dcs_phi) to restore the old behavior.
  bool nssm_use_dcs = false;
  double nssm_dcs_phi = 1.0;
  // extra ISAM2 iterations after inserting loop factors, BEFORE the post-loop
  // verification: one Gauss-Newton pass under-converges a large correction,
  // and judging that transient state reverts (and quarantines) genuine loops
  int loop_extra_iterations = 3;
  // Operator hand-correction trust (x m, y m, yaw rad). Yaw is deliberately
  // SOFT by default: DR yaw is compass-anchored and the whole post-loop
  // verification stack assumes it — a hard manual yaw contradicting the
  // compass would make every later loop round's yaw-RMS check trip
  // spuriously. Position is where the operator's knowledge is; tighten yaw
  // only if the compass itself is what is being corrected.
  Eigen::Vector3d manual_correction_sigmas = Eigen::Vector3d(0.2, 0.2, 0.5);
  // ABSOLUTE yaw gate against the compass (kills discrete rotational
  // aliasing — e.g. a near-square pool aliases at 90°, which ICP matches
  // confidently and symmetric pairs pass PCM): the closure's relative yaw
  // must agree with the compass-anchored DR relative yaw within this bound.
  // Neither max_rotation (ICP-vs-Sobol refinement only) nor the covariance
  // gate (catches sliding, not confident-but-wrong locks) covers this mode.
  double nssm_max_yaw_vs_compass = 0.15;  // rad (~8.6°, > compass noise)
  // OPTIONAL absolute TRANSLATION gate — the x/y analog of the yaw gate
  // above, for parallel-wall translational aliasing on symmetric venues. When
  // >0, it clamps the Sobol init search and rejects a final ICP relative pose
  // outside the same radius from DR. The final check matters because ICP can
  // otherwise walk from a bounded seed into a distant alias basin. Set just
  // above the real revisit-drift scale. 0 disables both checks.
  double nssm_max_translation_vs_dr = 0.0;  // m; 0 disables
  // optimize-then-verify (rtabmap RGBD/OptimizeMaxError analog): after a
  // loop-closure insert, the whole graph's optimized-vs-DR yaw RMS must stay
  // under this bound or the loops are rolled back (see update_factor_graph)
  double post_loop_max_yaw_rms = 0.15;  // rad
  // Post-loop CHAIN-TEAR check, in SIGMA of the link's own noise model
  // (rtabmap RGBD/OptimizeMaxError, Graph.cpp computeMaxGraphErrors; its
  // default is 3.0). Parallel-wall translational aliases pass every
  // per-closure gate — compass agrees between parallel walls, wall-to-wall
  // locks are compact, mutually-consistent aliases satisfy PCM — but to win
  // they must stretch consecutive links well beyond what those links'
  // factors claimed.
  //
  // Replaced an ABSOLUTE metre bound (post_loop_max_translation_err) on
  // 2026-07-25. The absolute form judged every link against one number
  // regardless of how far it spanned, so a link bridging a long DR-only gap
  // (the whole head-pitch-gated head sweep, during which no keyframe is
  // created) read as a tear purely for being long. Normalising by the link's
  // own sigma — which now scales with span, see add_odometry — removes that.
  // 0 disables.
  double post_loop_max_link_error_sigma = 3.0;  // sigma

  // Physical fan bounds used only to preselect loop-closure target points.
  // sonar_proc owns the live ping geometry; conservative static bounds keep
  // this back-end independent of raw sonar messages.
  double sonar_max_range = 30.0;
  double sonar_horizontal_aperture = 2.2689280275926285;  // radians(130)

  // ----------------------------------------------------------------- state
  std::vector<KeyframePtr> keyframes;
  KeyframePtr current_frame;

  ICP icp;

  int ssm_accepted = 0;
  int ssm_degenerate_rejected = 0;
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
  // loop rounds reverted by the post-loop verification (diagnostics)
  int nssm_reverted = 0;
  // per-callback diagnostic: for each loop-closure factor inserted this round,
  // its ICP-vs-DR correction geometry (keys + trans m / yaw rad). Distinguishes
  // a legit drift fix (small trans, ~0 yaw) from a parallel-wall alias (large
  // trans and/or non-zero yaw) when the round is later reverted. Cleared at the
  // top of every add_nonsequential_scan_matching call; the node logs it.
  std::vector<std::string> last_nssm_inserted_geom;
  // operator hand-corrections applied / undone over the run (diagnostics)
  int manual_corrections_applied = 0;
  int manual_corrections_undone = 0;

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
  // Operator-supplied absolute planar fix (graph/map chart) applied as a
  // prior on the NEWEST keyframe and fully converged immediately. The elastic
  // DR chain distributes the correction backwards through the trajectory
  // (and the mapping node re-renders its tiles from the republished
  // trajectory). Operator input is ground truth by declaration: no post-loop
  // verification runs. Returns false when there is no keyframe yet or the
  // solve failed (last_error()).
  bool add_manual_correction(const gtsam::Pose2& map_pose);
  // ------------------------------------------------- persistence + fixes
  // Serialize/restore the keyframe map (binary; poses + clouds). load_map
  // must run on an EMPTY session: it restores the keyframes, rebuilds an
  // anchored chain of tight between factors holding the saved (already
  // optimized) shape, and arms relocalization — the next dense frame is
  // globally scan-matched against the loaded map (relocalize()) before
  // normal operation resumes. Loop factors are not serialized; their effect
  // is baked into the saved poses.
  bool save_map(const std::string& path);
  bool load_map(const std::string& path);
  bool relocalize(const KeyframePtr& frame);
  bool awaiting_relocalization() const { return awaiting_relocalization_; }
  int loaded_keyframes() const { return loaded_key_count_; }
  // min fraction of a frame's points that must land on the loaded map for a
  // relocalization to be accepted
  double relocalize_min_overlap = 0.5;
  // Absolute position fix (USBL/LBL) on keyframe `key`: x/y at sigma_xy,
  // heading and everything else wide. Runs a normal (non-hard) update —
  // acoustic fixes are frequent and incremental, unlike operator input.
  bool add_position_prior(int key, double x, double y, double sigma_xy);
  int position_priors_applied = 0;
  // Remove the MOST RECENT manual-correction prior from the committed graph
  // and re-solve, relaxing the trajectory back. Stack semantics: call again
  // to peel earlier corrections. Returns false (last_error()) when there is
  // nothing to undo or the re-solve failed.
  bool undo_manual_correction();
  // manual priors currently in the graph (undo stack depth)
  int manual_corrections_pending() const
  {
    return static_cast<int>(manual_prior_indices_.size());
  }
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
    Eigen::Matrix3d cov_raw = Eigen::Matrix3d::Zero();  // pre-floor (degeneracy gate)
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

  // Odometry noise model scaled to the link's ACTUAL span. odom_sigmas
  // describes one nominal keyframe step (keyframe_translation /
  // keyframe_rotation); a link bridging a long DR-only gap accumulates
  // proportionally more drift and must not be weighted as a nominal step.
  // Mirrors the inflation publish_pose already applies to the PUBLISHED
  // covariance. Scale is floored at 1 — never MORE confident than configured.
  gtsam::SharedNoiseModel scaled_odom_model(const gtsam::Pose2& delta,
                                            double& trans_sigma,
                                            double& rot_sigma) const;

  void record_consecutive_link(int key, const gtsam::Pose2& measured,
                               double trans_sigma, double rot_sigma);

  // True when both keyframes' dr_pose values were produced by the SAME
  // dead-reckoning epoch, i.e. they are differenceable. A loaded map's
  // keyframes carry the dr_pose they were serialized with in a PREVIOUS
  // mission; this session's keyframes start from this mission's DR origin.
  // Within either group `between` is meaningful; across the boundary it is
  // an arbitrary offset. (loaded_key_count_ == 0 => one epoch, always true.)
  bool same_dr_epoch(int key_a, int key_b) const;

  // Predicted target->source relative pose used by the loop-closure
  // consistency gates.
  //
  // Dead reckoning is the preferred predictor because it is INDEPENDENT of
  // the pose graph: a graph already bent by an earlier bad closure cannot
  // then excuse the next one. Across a map-load boundary DR cannot supply it
  // (see same_dr_epoch), so the graph estimate does — after relocalization
  // that IS the common frame tying the loaded chain to this session, and it
  // is the same reference the Sobol init bounds are already centred on.
  // dr_backed reports which was used, for the gate's diagnostic text.
  gtsam::Pose2 predicted_relative_pose(int target_key, int source_key,
                                       bool* dr_backed = nullptr) const;

  // failure recovery for update_factor_graph: undo loop-closure bookkeeping
  // written before a failed solve, and reconstruct ISAM2 from the committed
  // factor mirror (a thrown ISAM2::update leaves the estimator in an
  // undefined, partially-mutated state — GTSAM gives no exception guarantee)
  // quarantine=true additionally marks rolled-back loops rejected (a
  // verification revert); false leaves them eligible for retry (a solver
  // failure says nothing about the loops).
  // culprit_link >= 0 narrows the quarantine to the closures SPANNING that
  // consecutive link — the chain-tear check names one, and only closures
  // crossing it can have stretched it, so the rest of the round stays
  // eligible. -1 convicts the whole round (the yaw check is global).
  void rollback_pending_loops(bool quarantine, long culprit_link = -1);
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
  // next update carries a manual correction: converge hard (like a loop
  // round) even though pending_loops_ is empty
  bool force_converge_ = false;
  // committed_graph_ indices of the applied manual-correction priors (an
  // undo stack). Truncations never invalidate these: they only drop factors
  // of FAILED rounds, which are never recorded here, and undo removal
  // null-holes the graph in place instead of shifting indices.
  std::vector<std::size_t> manual_prior_indices_;
  // per-link tear RATIO recorded when the link first exceeded the chain-tear
  // bound after an accepted correction round (-1 = under bound); the growth
  // gate judges against this persistent baseline (see update_factor_graph).
  // Still required after the switch to a sigma ratio: normalising fixes
  // long-link false positives, but NOT latching — a link legitimately
  // stretched past the bound by an accepted USBL/manual fix would otherwise
  // veto every later round forever, and repeated sub-gate aliases could
  // ratchet the graph a fraction of the bound at a time.
  std::vector<double> tear_baseline_;

  // What the factor between keyframes k-1 and k actually MEASURED, indexed
  // by k (index 0 unused; invalid entries are skipped). The post-loop check
  // judges the optimized relative pose against this — rtabmap
  // Graph.cpp:computeMaxGraphErrors compares against the link transform —
  // rather than against dead reckoning. DR is only the measurement for
  // add_odometry links; an SSM-accepted link measures ICP, so a DR
  // comparison charges a healthy graph a standing offset for agreeing with
  // the very factor it was given (bounded by ssm/max_translation).
  struct ConsecutiveLink
  {
    bool valid = false;
    gtsam::Pose2 measured;      // (k-1) -> k, as the factor measured it
    double trans_sigma = 1.0;   // m
    double rot_sigma = 1.0;     // rad
  };
  std::vector<ConsecutiveLink> consecutive_links_;
  // map persistence / relocalization state (see load_map)
  int loaded_key_count_ = 0;
  bool awaiting_relocalization_ = false;
  std::string last_error_;

  // 6D models over the Pose3 tangent (rx, ry, rz, tx, ty, tz); see configure()
  gtsam::SharedNoiseModel prior_model_, odom_model_, icp_odom_model_;
  // per-keyframe absolute depth + zero-attitude prior (wide x/y/yaw)
  gtsam::SharedNoiseModel unary_model_;

  std::deque<ICPResult> nssm_queue_;
};

}  // namespace sonar_slam
