#include "sonar_slam_cpp/slam_core.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/global_init.hpp"
#include "sonar_slam_cpp/icp_covariance.hpp"
#include "sonar_slam_cpp/icp_covariance_math.hpp"
#include "sonar_slam_cpp/mcd.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace sonar_slam {

namespace {

inline gtsam::Key X(int i) { return gtsam::Symbol('x', i); }

// ISAM2 tuned for a pose graph that must absorb loop-closure corrections.
// The GTSAM defaults (relinearizeThreshold 0.1, relinearizeSkip 10) are batch
// throughput settings: they let a correction sit un-relinearized for up to 10
// updates and freeze any variable whose delta stays under 0.1 — a 1 m / few-
// degree correction then "staircases" across many keyframes, and the post-
// loop verification (which runs right after the update) judges that
// under-converged transient. A ~100-1000 keyframe planar graph relinearizes
// cheaply; favor correctness.
gtsam::ISAM2Params make_isam2_params()
{
  gtsam::ISAM2Params p;
  p.relinearizeThreshold = 0.01;
  p.relinearizeSkip = 1;
  return p;
}

// SE3 graph chart: states are HORIZON poses
// (yaw-only rotation, live z). Sonar ICP observes (x, y, yaw); depth is an
// absolute measurement; roll/pitch are 0 BY CONSTRUCTION in this chart (the
// clouds are attitude-rotated at ingestion). Sigma constants for the
// dimensions each factor does not observe:
constexpr double kSigmaWide = 10.0;      // effectively unconstrained
constexpr double kSigmaTightRP = 1e-3;   // pins chart roll/pitch at 0
constexpr double kSigmaDepthAbs = 0.02;  // absolute depth (bridge), m
constexpr double kSigmaDepthDelta = 0.05;  // DR depth delta over a keyframe, m

// lift planar sigmas (x, y, theta) into the Pose3 tangent (rx,ry,rz,tx,ty,tz)
gtsam::SharedNoiseModel lift_sigmas(const Eigen::Vector3d& s, double s_rp,
                                    double s_z)
{
  gtsam::Vector6 v;
  v << s_rp, s_rp, s[2], s[0], s[1], s_z;
  return gtsam::noiseModel::Diagonal::Sigmas(v);
}

// embed a planar (x, y, theta) covariance into the Pose3 tangent, leaving the
// unobserved dimensions (roll, pitch, z) at kSigmaWide^2
Eigen::Matrix<double, 6, 6> embed_planar_cov(const Eigen::Matrix3d& c)
{
  Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Zero();
  C(0, 0) = C(1, 1) = C(5, 5) = kSigmaWide * kSigmaWide;
  constexpr int map[3] = {3, 4, 2};  // x->tx, y->ty, theta->rz
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) C(map[i], map[j]) = c(i, j);
  return C;
}

// lift a planar ICP transform into the horizon chart (z unobserved -> 0)
gtsam::Pose3 lift_planar(const gtsam::Pose2& p)
{
  return gtsam::Pose3(gtsam::Rot3::Yaw(p.theta()),
                      gtsam::Point3(p.x(), p.y(), 0.0));
}

}  // namespace

Slam::Slam() : isam_(make_isam2_params())
{
  // SSM defaults (slam.py __init__)
  ssm_params.initialization = true;
  ssm_params.init_n = 50;
  ssm_params.init_iters = 1;
  ssm_params.init_ftol = 0.01;
  ssm_params.min_st_sep = 1;
  ssm_params.min_points = 50;
  ssm_params.max_translation = 2.0;
  ssm_params.max_rotation = M_PI / 6.0;
  ssm_params.target_frames = 3;
  ssm_params.cov_samples = 0;

  // NSSM defaults
  nssm_params.initialization = true;
  nssm_params.init_n = 100;
  nssm_params.init_iters = 5;
  nssm_params.init_ftol = 0.01;
  nssm_params.min_st_sep = 10;
  nssm_params.min_points = 100;
  nssm_params.max_translation = 6.0;
  nssm_params.max_rotation = M_PI / 2.0;
  nssm_params.source_frames = 5;
  nssm_params.cov_samples = 30;
  nssm_params.fan_drift_trans = 0.01;
  nssm_params.fan_drift_rot = 0.0017;
}

void Slam::configure()
{
  // config sanity (slam.py used asserts; these must also fire in NDEBUG
  // builds — the default build type disables assert())
  // The Censi helper builds a point-to-point J'J Hessian and does not consume
  // the surface normals a point-to-plane objective minimises, so it may only
  // be used with a point-to-point ICP chain. Test the chain ACTUALLY LOADED
  // rather than asserting a constant: the package default
  // (config/icp.yaml) is PointToPlaneErrorMinimizer, but the deployed
  // settings_erdc override is PointToPointErrorMinimizer — the old hard
  // rejection claimed "the runtime ICP is point-to-plane", which was false
  // for the deployed configuration and foreclosed a valid option on a wrong
  // premise. (icp.load_from_yaml runs before configure(), so this is known.)
  if (constrained_3d &&
      (ssm_params.cov_method == SMParams::CENSI ||
       nssm_params.cov_method == SMParams::CENSI))
    throw std::invalid_argument(
      "cov_method 'censi' is planar-only; constrained_3d uses XYZ "
      "correspondences and requires cov_method 'sampled'");
  if (!constrained_3d &&
      (ssm_params.cov_method == SMParams::CENSI ||
       nssm_params.cov_method == SMParams::CENSI)) {
    const std::string em = icp.error_minimizer_name();
    if (em.find("PointToPoint") == std::string::npos)
      throw std::invalid_argument(
        "cov_method 'censi' requires a point-to-point ICP chain: the Censi "
        "covariance Hessian is point-to-point and does not consume the "
        "surface normals a point-to-plane objective minimises. The loaded "
        "icp_config uses '" + em +
        "'. Use cov_method 'sampled', or configure "
        "PointToPointErrorMinimizer in icp_config.");
  }
  if (constrained_3d &&
      (!(constrained_icp_params.max_correspondence > 0.0f) ||
       !(constrained_icp_params.trim_ratio > 0.0 &&
         constrained_icp_params.trim_ratio <= 1.0) ||
       constrained_icp_params.max_iterations < 1 ||
       constrained_icp_params.min_correspondences < 3 ||
       !(constrained_icp_params.translation_epsilon >= 0.0) ||
       !(constrained_icp_params.rotation_epsilon >= 0.0)))
    throw std::invalid_argument("invalid constrained_3d ICP parameters");
  if (nssm_params.cov_samples > 0 &&
      nssm_params.cov_samples >= nssm_params.init_n * nssm_params.init_iters)
    throw std::invalid_argument(
      "nssm/cov_samples must be < initialization n * iters");
  if (ssm_params.cov_samples > 0 &&
      ssm_params.cov_samples >= ssm_params.init_n * ssm_params.init_iters)
    throw std::invalid_argument(
      "ssm/cov_samples must be < initialization n * iters");
  if (nssm_params.source_frames >= nssm_params.min_st_sep)
    throw std::invalid_argument(
      "nssm/source_frames must be < nssm/min_st_sep");
  // require_covariance is only satisfiable when a covariance is actually
  // produced, and run_scan_match_icp produces one ONLY when cov_samples > 0
  // (the sampled AND censi paths are both behind that gate). The contradictory
  // pair would silently reject every SSM registration as "degenerate:
  // covariance unavailable" — an all-odometry graph that still burns the full
  // global-init + ICP compute per keyframe, distinguishable from a healthy
  // run only by reading the reject counters. Refuse to start instead.
  if (ssm_params.enable && ssm_require_covariance && ssm_params.cov_samples <= 0)
    throw std::invalid_argument(
      "ssm/require_covariance needs ssm/cov_samples > 0: no covariance is "
      "estimated at cov_samples 0, so every sequential scan match would be "
      "rejected as 'covariance unavailable' and the graph would silently "
      "degrade to odometry-only. Set ssm/cov_samples (e.g. 30, matching "
      "nssm), or set ssm/require_covariance: false.");
  // 6D lifts of the configured planar sigmas (see the chart note above):
  // prior/odometry constrain chart roll/pitch (0) and depth (absolute for the
  // prior, DR delta for odometry); ICP factors are wide in everything the
  // sonar cannot observe. The per-keyframe unary carries absolute depth +
  // zero-attitude and is wide in the planar dimensions the graph owns.
  prior_model_ = lift_sigmas(prior_sigmas, kSigmaTightRP, kSigmaDepthAbs);
  odom_model_ = lift_sigmas(odom_sigmas, kSigmaTightRP, kSigmaDepthDelta);
  icp_odom_model_ = lift_sigmas(icp_odom_sigmas, kSigmaWide, kSigmaWide);
  gtsam::Vector6 u;
  u << kSigmaTightRP, kSigmaTightRP, kSigmaWide, kSigmaWide, kSigmaWide,
    kSigmaDepthAbs;
  unary_model_ = gtsam::noiseModel::Diagonal::Sigmas(u);
}

gtsam::SharedNoiseModel Slam::scaled_odom_model(const gtsam::Pose2& delta,
                                                double& trans_sigma,
                                                double& rot_sigma) const
{
  const double d = std::hypot(delta.x(), delta.y());
  const double a = std::abs(delta.theta());
  const double kt = std::max(keyframe_translation, 1e-3);
  const double kr = std::max(keyframe_rotation, 1e-3);
  // Floor at 1: a SHORT link is not more trustworthy than the configured
  // nominal step (odom_sigmas already includes a constant per-fix error
  // floor), but a LONG one is genuinely worse. Same form as the published
  // covariance inflation in publish_pose.
  const double st = std::max(1.0, d / kt);
  const double sr = std::max(1.0, a / kr);
  trans_sigma = std::max(odom_sigmas[0], odom_sigmas[1]) * st;
  rot_sigma = odom_sigmas[2] * sr;
  const Eigen::Vector3d s(odom_sigmas[0] * st, odom_sigmas[1] * st,
                          odom_sigmas[2] * sr);
  return lift_sigmas(s, kSigmaTightRP, kSigmaDepthDelta * st);
}

void Slam::record_consecutive_link(int key, const gtsam::Pose2& measured,
                                   double trans_sigma, double rot_sigma)
{
  if (key < 0) return;
  if (static_cast<int>(consecutive_links_.size()) <= key)
    consecutive_links_.resize(key + 1);
  consecutive_links_[key] = {true, measured, std::max(trans_sigma, 1e-6),
                             std::max(rot_sigma, 1e-6)};
}

bool Slam::same_dr_epoch(int key_a, int key_b) const
{
  if (loaded_key_count_ <= 0) return true;
  return (key_a < loaded_key_count_) == (key_b < loaded_key_count_);
}

gtsam::Pose2 Slam::predicted_relative_pose(int target_key, int source_key,
                                           bool* dr_backed) const
{
  const bool dr = same_dr_epoch(target_key, source_key);
  if (dr_backed) *dr_backed = dr;
  return dr ? keyframes[target_key]->dr_pose.between(
                keyframes[source_key]->dr_pose)
            : keyframes[target_key]->pose.between(keyframes[source_key]->pose);
}

// ---------------------------------------------------------------------------
// point accumulation
// ---------------------------------------------------------------------------
Matrix Slam::get_points(const std::vector<int>& frames, int ref_key) const
{
  const bool use_ref = ref_key >= 0;
  gtsam::Pose2 ref_pose;
  gtsam::Pose3 ref_pose3;
  if (use_ref) {
    ref_pose = keyframes[ref_key]->pose;
    ref_pose3 = keyframes[ref_key]->horizon_pose3();
  }

  long total = 0;
  for (int key : frames) total += keyframes[key]->points.rows();
  const int dimensions = constrained_3d ? 3 : 2;
  Matrix all_points(total, dimensions);

  long at = 0;
  for (int key : frames) {
    if (use_ref) {
      const Matrix transf = constrained_3d
        ? Keyframe::transform_points(
            keyframes[key]->points,
            ref_pose3.between(keyframes[key]->horizon_pose3()))
        : Keyframe::transform_points(
            keyframes[key]->points.leftCols(2),
            ref_pose.between(keyframes[key]->pose));
      all_points.middleRows(at, transf.rows()) = transf;
      at += transf.rows();
    } else {
      const Matrix& transf = keyframes[key]->transf_points;  // reference, no copy
      all_points.middleRows(at, transf.rows()) =
        constrained_3d ? transf : transf.leftCols(2);
      at += transf.rows();
    }
  }

  return downsample(all_points, static_cast<float>(point_resolution));
}

std::pair<Matrix, std::vector<int>> Slam::get_points_with_keys(
  const std::vector<int>& frames) const
{
  long total = 0;
  for (int key : frames) total += keyframes[key]->points.rows();
  const int dimensions = constrained_3d ? 3 : 2;
  Matrix all_points(total, dimensions);
  Matrix all_keys(total, 1);

  long at = 0;
  for (int key : frames) {
    const Matrix& transf = keyframes[key]->transf_points;
    all_points.middleRows(at, transf.rows()) =
      constrained_3d ? transf : transf.leftCols(2);
    all_keys.middleRows(at, transf.rows()).setConstant(static_cast<float>(key));
    at += transf.rows();
  }

  auto [pts, keys] = downsample(all_points, all_keys,
                                static_cast<float>(point_resolution));
  std::vector<int> key_vec(pts.rows());
  for (int i = 0; i < pts.rows(); ++i)
    key_vec[i] = static_cast<int>(keys(i, 0));
  return {pts, key_vec};
}

// ---------------------------------------------------------------------------
// ICP
// ---------------------------------------------------------------------------
std::pair<std::string, gtsam::Pose2> Slam::compute_icp(
  const Matrix& source_points, const Matrix& target_points,
  const gtsam::Pose2& guess)
{
  Eigen::Matrix3f g = guess.matrix().cast<float>();
  if (constrained_3d) {
    const ConstrainedIcpResult result = constrained_icp_xyz(
      source_points, target_points, g, constrained_icp_params);
    const double x = result.T(0, 2), y = result.T(1, 2);
    const double theta = std::atan2(result.T(1, 0), result.T(0, 0));
    return {result.message, gtsam::Pose2(x, y, theta)};
  }
  auto [message, T] = icp.compute(source_points, target_points, g);
  const double x = T(0, 2), y = T(1, 2);
  const double theta = std::atan2(T(1, 0), T(0, 0));
  return {message, gtsam::Pose2(x, y, theta)};
}

Slam::IcpCovResult Slam::compute_icp_with_cov(
  const Matrix& source_points, const Matrix& target_points,
  const std::vector<gtsam::Pose2>& guesses)
{
  IcpCovResult result;

  std::vector<Eigen::Vector3d> sample_transforms;
  if (parallel_cov_samples) {
    // per-thread engine pool; each guess's registration is identical to the
    // sequential loop's (deterministic chain), collected in guess order. The
    // 2 s cap (slam.py) still applies, but in parallel it rarely fires — so
    // MCD typically sees all cov_samples instead of however many one core
    // got through in 2 s.
    std::vector<Eigen::Matrix3f> gm(guesses.size());
    for (std::size_t i = 0; i < guesses.size(); ++i)
      gm[i] = guesses[i].matrix().cast<float>();
    if (constrained_3d) {
      const auto batch = constrained_icp_xyz_batch(
        source_points, target_points, gm, constrained_icp_params, 2000);
      for (const auto& r : batch) {
        if (!r.success) continue;
        sample_transforms.emplace_back(r.T(0, 2), r.T(1, 2),
                                       std::atan2(r.T(1, 0), r.T(0, 0)));
      }
    } else {
      const auto batch = icp.compute_batch(source_points, target_points, gm, 2000);
      for (const auto& r : batch) {
        if (!r.success) continue;
        sample_transforms.emplace_back(r.T(0, 2), r.T(1, 2),
                                       std::atan2(r.T(1, 0), r.T(0, 0)));
      }
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (const auto& g : guesses) {
      const auto [message, odom] =
        compute_icp(source_points, target_points, g);
      if (message == "success") {
        sample_transforms.emplace_back(odom.x(), odom.y(), odom.theta());
      }
      // enforce a max run time for this loop (slam.py: 2 s)
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
      if (elapsed.count() >= 2000) break;
    }
  }

  if (sample_transforms.size() < 5) {
    result.message = "Too few samples for covariance computation";
    return result;
  }

  Eigen::MatrixXd samples(sample_transforms.size(), 3);
  for (std::size_t i = 0; i < sample_transforms.size(); ++i)
    samples.row(i) = sample_transforms[i];

  // robust covariance — outliers are frequent (slam.py uses MinCovDet)
  const McdResult mcd = min_cov_det(samples, 0.8);
  if (!mcd.success) {
    result.message = "Failed to calculate covariance";
    return result;
  }

  const gtsam::Pose2 m(mcd.location[0], mcd.location[1], mcd.location[2]);
  Eigen::Matrix3d cov = mcd.covariance;

  // unrotate to local frame
  const Eigen::Matrix2d R = m.rotation().matrix();
  cov.topRows<2>() = R.transpose() * cov.topRows<2>();
  cov.leftCols<2>() = cov.leftCols<2>() * R;

  // Never report a covariance smaller than the fixed ICP model — but floor
  // PER AXIS, not by determinant. The historical whole-matrix swap on
  // det(cov) < det(default) replaced exactly the elongated matrices (one
  // collapsed axis crushes the determinant) with the confident isotropic
  // default, blinding the NSSM degeneracy gate in the wall-sliding regime it
  // was built for. Clamping the translation-block eigenvalues and the yaw
  // variance up to the default's floors keeps the collapsed axis honest while
  // preserving the elongation (sigma_max survives for the gates). Raising a
  // diagonal block's eigenvalues adds a PSD term, so the matrix stays PSD.
  // Capture the pre-floor covariance for the degeneracy gate (it needs the
  // elongation the floor below erases — see nssm_degeneracy_prefloor).
  const Eigen::Matrix3d cov_raw = cov;
  const Eigen::Vector3d floor_var = icp_odom_sigmas.array().square().matrix();
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov.topLeftCorner<2, 2>());
    // A failed eigensolve leaves eigenvalues/vectors unspecified; flooring
    // with garbage would fabricate a confident covariance. Report failure —
    // the caller falls back to the fixed model (SSM) or rejects (NSSM).
    if (es.info() != Eigen::Success) {
      result.message = "Failed to calculate covariance";
      return result;
    }
    const double vmin = std::min(floor_var[0], floor_var[1]);
    const Eigen::Vector2d ev = es.eigenvalues().cwiseMax(vmin);
    cov.topLeftCorner<2, 2>() =
      es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    cov(2, 2) = std::max(cov(2, 2), floor_var[2]);
  }

  result.message = "success";
  result.odom = m;
  result.cov = cov;
  result.cov_raw = cov_raw;
  result.n_samples = static_cast<int>(sample_transforms.size());
  return result;
}

int Slam::get_overlap(const Matrix& source_points, const Matrix& target_points,
                      const gtsam::Pose2* source_pose,
                      std::vector<int>* indices_out) const
{
  // Avoid copying the full source cloud when no pose transform is needed
  // (the loop-refine overlap call passes source_pose == nullptr).
  Matrix src_storage;
  const Matrix* src_ptr = &source_points;
  if (source_pose) {
    src_storage = Keyframe::transform_points(source_points, *source_pose);
    src_ptr = &src_storage;
  }
  const Matrix& src = *src_ptr;

  const Matrix target_match = constrained_3d
    ? target_points : Matrix(target_points.leftCols(2));
  const Matrix source_match = constrained_3d
    ? src : Matrix(src.leftCols(2));
  auto [ids, dists] = match(target_match, source_match, 1,
                            static_cast<float>(point_noise));
  int count = 0;
  if (indices_out) indices_out->resize(ids.cols());
  for (int i = 0; i < ids.cols(); ++i) {
    if (indices_out) (*indices_out)[i] = ids(0, i);
    if (ids(0, i) != -1) ++count;
  }
  return count;
}

// ---------------------------------------------------------------------------
// factors
// ---------------------------------------------------------------------------
bool Slam::is_keyframe(const Keyframe& frame) const
{
  if (keyframes.empty()) return true;

  const double duration = to_sec(frame.time) - to_sec(current_keyframe()->time);
  // negative = time jumped backwards (bag loop / restart): without this the
  // gate returns false against the last pre-jump keyframe until time catches
  // up — zero keyframes for the whole looped pass
  if (duration < 0.0) return true;
  if (duration < keyframe_duration) return false;

  const gtsam::Pose2 dr_odom = keyframes.back()->dr_pose.between(frame.dr_pose);
  const gtsam::Pose3 dr_odom3 =
    keyframes.back()->horizon_dr_pose3().between(frame.horizon_dr_pose3());
  const double translation = constrained_3d
    ? std::sqrt(dr_odom3.x() * dr_odom3.x() +
                dr_odom3.y() * dr_odom3.y() +
                dr_odom3.z() * dr_odom3.z())
    : std::hypot(dr_odom.translation().x(), dr_odom.translation().y());
  const double rotation = std::abs(dr_odom.theta());
  const double head_rotation =
    std::abs(frame.head_pitch - keyframes.back()->head_pitch);
  return translation > keyframe_translation || rotation > keyframe_rotation ||
         (constrained_3d && keyframe_head_rotation > 0.0 &&
          head_rotation > keyframe_head_rotation);
}

void Slam::add_prior(const KeyframePtr& keyframe)
{
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), keyframe->horizon_pose3(),
                                              prior_model_));
  values_.insert(X(0), keyframe->horizon_pose3());
}

void Slam::add_odometry(const KeyframePtr& keyframe)
{
  // horizon-chart DR delta: planar odometry + the measured depth change
  const gtsam::Pose3 dr_odom =
    keyframes.back()->horizon_pose3().between(keyframe->horizon_pose3());
  // Noise scaled to what this link actually spans, not to the nominal
  // keyframe step. Input admission and the head-pitch gate can force
  // status=false for a whole +/-54 deg sweep, during which NO keyframe is
  // created — so the link that closes the sweep can span many metres and
  // must not carry the same 0.2 m sigma as a 0.75 m step.
  const gtsam::Pose2 dr_planar =
    keyframes.back()->dr_pose.between(keyframe->dr_pose);
  double ts = 0.0, rs = 0.0;
  const gtsam::SharedNoiseModel model = scaled_odom_model(dr_planar, ts, rs);
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
    X(current_key() - 1), X(current_key()), dr_odom, model));
  values_.insert(X(current_key()), keyframe->horizon_pose3());
  add_state_unary(current_key(), keyframe);
  record_consecutive_link(current_key(), dr_planar, ts, rs);
}

void Slam::add_state_unary(int key, const KeyframePtr& keyframe)
{
  // absolute depth + zero-attitude anchor for the dimensions no scan-match
  // factor observes (wide in x/y/yaw, which the graph owns)
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
    X(key), keyframe->horizon_dr_pose3(), unary_model_));
}

// ---------------------------------------------------------------------------
// sequential scan matching
// ---------------------------------------------------------------------------
InitializationResult Slam::initialize_sequential_scan_matching(
  const KeyframePtr& keyframe)
{
  InitializationResult ret;
  ret.status = Status(Status::SUCCESS);

  ret.source_key = current_key();
  ret.target_key = current_key() - 1;
  ret.source_pose = keyframe->pose;
  ret.target_pose = current_keyframe()->pose;

  ret.source_points = constrained_3d
    ? keyframe->points : Matrix(keyframe->points.leftCols(2));
  if (constrained_3d && ret.source_points.cols() >= 3) {
    // Express the source elevations in the target keyframe's pressure-depth
    // chart.  ICP then solves x/y/yaw only; z is already known.
    ret.source_points.col(2).array() += static_cast<float>(
      keyframe->pose3.z() - current_keyframe()->pose3.z());
  }
  std::vector<int> target_frames;
  for (int k = std::max(0, current_key() - ssm_params.target_frames);
       k < current_key(); ++k)
    target_frames.push_back(k);
  ret.target_points = get_points(target_frames, ret.target_key);
  // odom_sigmas are standard deviations, so a covariance is their squares.
  // slam.py wrote diag(odom_sigmas) here (sigma-as-variance); square it for
  // correctness. Currently vestigial — the C++ global-init cost dropped the
  // pose-prior term that consumed it — but kept correct in case it is wired up.
  ret.cov = odom_sigmas.array().square().matrix().asDiagonal();

  if (!ssm_params.enable) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "source points " + std::to_string(ret.source_points.rows());
    return ret;
  }

  if (ret.source_points.rows() < ssm_params.min_points) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "source points " + std::to_string(ret.source_points.rows());
    return ret;
  }
  if (ret.target_points.rows() < ssm_params.min_points) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "target points " + std::to_string(ret.target_points.rows());
    return ret;
  }

  if (!ssm_params.initialization) return ret;

  // search space: +-5 sigma of the odometry noise
  Eigen::Matrix<double, 3, 2> bounds;
  for (int k = 0; k < 3; ++k) {
    bounds(k, 0) = -5.0 * odom_sigmas[k];
    bounds(k, 1) = 5.0 * odom_sigmas[k];
  }

  const GlobalInitResult init = global_scan_match_init(
    ret.source_points, ret.source_pose, ret.target_points, ret.target_pose,
    point_noise, bounds, ssm_params.init_n, ssm_params.init_iters,
    ssm_params.init_ftol);

  if (init.success) {
    ret.source_pose_samples = init.pose_samples;
    ret.estimated_source_pose = ret.source_pose.compose(init.delta);
    ret.has_estimated_source_pose = true;
    ret.status.description = "matching cost " + std::to_string(init.best_cost);
  } else {
    ret.status = Status(Status::INITIALIZATION_FAILURE);
  }
  return ret;
}

void Slam::run_scan_match_icp(ICPResult& ret2, const SMParams& params)
{
  const bool want_cov = params.initialization && params.cov_samples > 0;

  if (want_cov && params.cov_method == SMParams::SAMPLED) {
    // run the best init guesses through ICP and take a robust covariance
    std::vector<gtsam::Pose2> guesses(
      ret2.initial_transforms.begin(),
      ret2.initial_transforms.begin() +
        std::min<std::size_t>(params.cov_samples,
                              ret2.initial_transforms.size()));
    const IcpCovResult icp_ret =
      compute_icp_with_cov(ret2.source_points, ret2.target_points, guesses);
    if (icp_ret.message != "success") {
      ret2.status = Status(Status::NOT_CONVERGED);
      ret2.status.description = icp_ret.message;
    } else {
      ret2.estimated_transform = icp_ret.odom;
      ret2.has_estimated_transform = true;
      ret2.cov = icp_ret.cov;
      ret2.cov_raw = icp_ret.cov_raw;
      ret2.has_cov = true;
      ret2.n_sample_transforms = icp_ret.n_samples;
      ret2.status.description = std::to_string(icp_ret.n_samples) + " samples";
    }
    return;
  }

  // single ICP; the Censi path additionally attaches a closed-form covariance
  auto [message, odom] =
    compute_icp(ret2.source_points, ret2.target_points, ret2.initial_transform);
  if (message != "success") {
    ret2.status = Status(Status::NOT_CONVERGED);
    ret2.status.description = message;
    return;
  }
  ret2.estimated_transform = odom;
  ret2.has_estimated_transform = true;

  if (want_cov && params.cov_method == SMParams::CENSI) {
    const CensiCovResult c = censi_icp_covariance(
      ret2.source_points, ret2.target_points, odom, point_noise,
      censi_sensor_noise * censi_sensor_noise);
    if (c.success) {
      // Censi produces the covariance in the global-translation chart (residual
      // r = R(theta) p + t - q, so dr/dt = I in the WORLD frame). Un-rotate the
      // translation block into the Pose2 body/tangent frame that GTSAM's
      // Gaussian::Covariance factor expects — the identical conversion the
      // sampled path applies (compute_icp_with_cov, a port of slam.py:377-380),
      // so both covariance methods feed factors in a consistent frame. The
      // determinant (used by the floor below) is rotation-invariant, but the
      // off-diagonal / axis orientation is not.
      Eigen::Matrix3d cov = c.cov;
      const Eigen::Matrix2d R = odom.rotation().matrix();
      cov.topRows<2>() = R.transpose() * cov.topRows<2>();
      cov.leftCols<2>() = cov.leftCols<2>() * R;
      // never report a covariance smaller than the fixed ICP model
      const Eigen::Matrix3d default_cov =
        icp_odom_sigmas.array().square().matrix().asDiagonal();
      if (cov.determinant() < default_cov.determinant()) cov = default_cov;
      ret2.cov = cov;
      // Censi path keeps the whole-matrix det swap (separate known issue), so
      // there is no distinct pre-floor cov here — mirror cov (the prefloor gate
      // is a no-op on the Censi path).
      ret2.cov_raw = cov;
      ret2.has_cov = true;
      ret2.n_sample_transforms = c.n_correspondences;
      ret2.status.description = "censi " + std::to_string(c.n_correspondences);
    } else {
      // too few correspondences for a covariance — keep the fixed model
      ret2.status.description = "censi (fixed model)";
    }
  }
}

void Slam::add_sequential_scan_matching(const KeyframePtr& keyframe)
{
  InitializationResult ret = initialize_sequential_scan_matching(keyframe);

  if (!ret.status) {
    last_ssm_status =
      std::string(ret.status.name()) + " (" + ret.status.description + ")";
    add_odometry(keyframe);
    return;
  }

  ICPResult ret2(ret, ssm_params.cov_samples > 0 &&
                        ssm_params.cov_method == SMParams::SAMPLED);
  run_scan_match_icp(ret2, ssm_params);

  // the transformation compared to dead reckoning can't be too large
  double delta_translation = std::numeric_limits<double>::quiet_NaN();
  double delta_rotation = std::numeric_limits<double>::quiet_NaN();
  if (ret2.status) {
    const gtsam::Pose2 delta =
      ret2.initial_transform.between(ret2.estimated_transform);
    delta_translation =
      std::hypot(delta.translation().x(), delta.translation().y());
    delta_rotation = std::abs(delta.theta());
    // inverted comparisons so a NaN delta (degenerate transform) fails closed
    if (!(delta_translation <= ssm_params.max_translation) ||
        !(delta_rotation <= ssm_params.max_rotation)) {
      ret2.status = Status(Status::LARGE_TRANSFORMATION);
      ret2.status.description = "trans " + std::to_string(delta_translation) +
                                " rot " + std::to_string(delta_rotation);
    }
  }

  // Per-link max_rotation alone does not bound the whole sequential chain:
  // several same-sign corrections just below that threshold can ratchet the
  // map tens of degrees away from DR. Evaluate the proposed
  // source pose before inserting its factor. If the target is already outside
  // the bound (manual/absolute correction), allow a link that holds or heals
  // the discrepancy but never one that worsens it.
  if (ret2.status && ssm_max_yaw_vs_dr > 0.0) {
    const gtsam::Pose2 proposed =
      ret2.target_pose.compose(ret2.estimated_transform);
    const double target_yaw_error = std::abs(std::remainder(
      ret2.target_pose.theta() - keyframes[ret2.target_key]->dr_pose.theta(),
      2.0 * M_PI));
    const double source_yaw_error = std::abs(std::remainder(
      proposed.theta() - keyframe->dr_pose.theta(), 2.0 * M_PI));
    if (!ssm_yaw_consistent(
          ret2.target_pose.theta(),
          keyframes[ret2.target_key]->dr_pose.theta(), proposed.theta(),
          keyframe->dr_pose.theta(), ssm_max_yaw_vs_dr)) {
      ret2.status = Status(Status::LARGE_TRANSFORMATION);
      ret2.status.description =
        "compass-inconsistent cumulative yaw " +
        std::to_string(source_yaw_error) + " rad (target was " +
        std::to_string(target_yaw_error) + ")";
    }
  }

  // there must be enough overlap between the two point clouds
  if (ret2.status) {
    const int overlap = get_overlap(ret2.source_points, ret2.target_points,
                                    &ret2.estimated_transform);
    // Absolute count AND fraction-of-source (rtabmap Icp/CorrespondenceRatio).
    // The count is venue-coupled; the ratio is not — see SMParams.
    const long src_n = ret2.source_points.rows();
    const double ratio = src_n > 0 ? static_cast<double>(overlap) / static_cast<double>(src_n) : 0.0;
    if (overlap < ssm_params.min_points ||
        (ssm_params.min_overlap_ratio > 0.0 &&
         ratio < ssm_params.min_overlap_ratio))
      ret2.status = Status(Status::NOT_ENOUGH_OVERLAP);
    // append — don't clobber the success path's "N samples" diagnostic
    if (!ret2.status.description.empty()) ret2.status.description += ", ";
    ret2.status.description += "overlap " + std::to_string(overlap) +
                              " (" + std::to_string(ratio) + " of source)";
  }

  // A numerically successful ICP is not automatically an observable sonar
  // measurement. Sparse/scattered water-column returns and line-like geometry
  // can converge while leaving a large or weak translation axis. Fail closed
  // and use the fused-odometry factor instead.
  if (ret2.status) {
    if (ssm_require_covariance && !ret2.has_cov) {
      ret2.status = Status(Status::NOT_CONVERGED);
      ret2.status.description = "degenerate: covariance unavailable";
      ++ssm_degenerate_rejected;
    } else if (ret2.has_cov) {
      const Eigen::Matrix3d& gate_cov =
        ssm_degeneracy_prefloor ? ret2.cov_raw : ret2.cov;
      const IcpObservability obs = assess_icp_observability(
        gate_cov, ssm_max_sigma, ssm_max_anisotropy);
      if (!obs.observable) {
        ret2.status = Status(Status::NOT_CONVERGED);
        ret2.status.description =
          "degenerate: sigma " + std::to_string(obs.sigma_max) +
          " aniso " + std::to_string(obs.anisotropy);
        ++ssm_degenerate_rejected;
      }
    }
  }

  if (ret2.status) {
    if (!ret2.status.description.empty()) ret2.status.description += ", ";
    ret2.status.description +=
      "corr trans " + std::to_string(delta_translation) +
      " rot " + std::to_string(delta_rotation);
    const gtsam::SharedNoiseModel model =
      ret2.has_cov
        ? gtsam::noiseModel::Gaussian::Covariance(embed_planar_cov(ret2.cov))
        : icp_odom_model_;
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      X(ret2.target_key), X(ret2.source_key),
      lift_planar(ret2.estimated_transform), model));
    // initial value: planar compose; z from DR (the factor is z-wide, the
    // unary below anchors it)
    const gtsam::Pose2 init2 = ret2.target_pose.compose(ret2.estimated_transform);
    values_.insert(X(ret2.source_key),
                   gtsam::Pose3(gtsam::Rot3::Yaw(init2.theta()),
                                gtsam::Point3(init2.x(), init2.y(),
                                              keyframe->dr_pose3.z())));
    add_state_unary(ret2.source_key, keyframe);
    // This consecutive link measures ICP, NOT dead reckoning — record what
    // the factor actually claimed so the post-loop check judges the graph
    // against it (see ConsecutiveLink).
    {
      double ts = std::max(icp_odom_sigmas[0], icp_odom_sigmas[1]);
      double rs = icp_odom_sigmas[2];
      if (ret2.has_cov) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(
          ret2.cov.topLeftCorner<2, 2>());
        const double ev = es.eigenvalues().maxCoeff();
        // NaN-safe: std::max launders NaN to its other argument
        if (std::isfinite(ev) && ev > 0.0) ts = std::sqrt(ev);
        if (std::isfinite(ret2.cov(2, 2)) && ret2.cov(2, 2) > 0.0)
          rs = std::sqrt(ret2.cov(2, 2));
      }
      record_consecutive_link(ret2.source_key, ret2.estimated_transform, ts, rs);
    }
    ret2.inserted = true;
    ++ssm_accepted;
    last_ssm_status = "accepted (" + ret2.status.description + ")";
  } else {
    last_ssm_status =
      std::string(ret2.status.name()) + " (" + ret2.status.description + ")";
    add_odometry(keyframe);
  }
}

// ---------------------------------------------------------------------------
// non-sequential scan matching (loop closures)
// ---------------------------------------------------------------------------
InitializationResult Slam::initialize_nonsequential_scan_matching()
{
  InitializationResult ret;
  ret.status = Status(Status::SUCCESS);

  ret.source_key = current_key() - 1;
  // Anchor the search on the just-added keyframe (keyframes[source_key]) whose
  // frame the source_points live in. slam.py used self.current_frame.pose here,
  // but both stacks assign current_frame only AFTER this runs, so it points at
  // the previous callback's frame — offsetting the global-init search from its
  // own geometry by up to a keyframe step. Intentional divergence from
  // bruce_slam that fixes that latent bug.
  ret.source_pose = keyframes[ret.source_key]->pose;
  ret.estimated_source_pose = ret.source_pose;
  ret.has_estimated_source_pose = false;

  std::vector<int> source_frames;
  for (int k = ret.source_key;
       k > ret.source_key - nssm_params.source_frames && k >= 0; --k)
    source_frames.push_back(k);
  ret.source_points = get_points(source_frames, ret.source_key);

  if (ret.source_points.rows() < nssm_params.min_points) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "source points " + std::to_string(ret.source_points.rows());
    return ret;
  }

  // all keyframes except the exclusion zone around the current one
  std::vector<int> target_frames;
  for (int k = 0; k < current_key() - nssm_params.min_st_sep; ++k)
    target_frames.push_back(k);

  auto [target_points_all, target_keys_all] = get_points_with_keys(target_frames);

  // keep only points that could be inside the sonar fan of a source frame,
  // padded by the pose uncertainty.
  // NOTE: update_factor_graph only refreshes the newest keyframe's covariance
  // (carried from slam.py), so the older source
  // frames' stored covariances are stale — and typically over-confident, which
  // would UNDER-pad the fan and drop overlapping target points (missed
  // closures). Instead of the stale per-frame cov we take the freshly
  // refreshed anchor covariance and inflate it by the source frame's age at
  // bounded drift rates (fan_drift_trans/rot). This only sets the candidate
  // pre-selection PADDING, and inflation can only widen the fan, so it errs
  // toward over-selection (a bit more ICP work, never a missed overlap).
  // Intentional divergence from bruce_slam.
  const long n_target_all = static_cast<long>(target_points_all.rows());
  std::vector<char> sel(n_target_all, 0);
  long n_sel = 0;
  Eigen::Matrix3d cov;
  const Eigen::Matrix3d fresh_cov = keyframes[ret.source_key]->cov;
  const double anchor_time = to_sec(keyframes[ret.source_key]->time);
  for (int source_frame : source_frames) {
    // every target point is already in some source frame's padded fan — the
    // remaining frames can only re-select them
    if (n_sel == n_target_all) break;
    const gtsam::Pose2& pose = keyframes[source_frame]->pose;

    // age >= 0: source_frames descend from the anchor (newest) keyframe.
    const double age_s = anchor_time - to_sec(keyframes[source_frame]->time);
    cov = fresh_cov;
    cov(0, 0) += std::pow(nssm_params.fan_drift_trans * age_s, 2);
    cov(1, 1) += std::pow(nssm_params.fan_drift_trans * age_s, 2);
    cov(2, 2) += std::pow(nssm_params.fan_drift_rot * age_s, 2);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov.topLeftCorner<2, 2>());
    const double translation_std = std::sqrt(es.eigenvalues().maxCoeff());
    const double rotation_std = std::sqrt(cov(2, 2));
    const double range_bound = translation_std * 5.0 + sonar_max_range;
    const double bearing_bound =
      rotation_std * 5.0 + sonar_horizontal_aperture * 0.5;

    const Matrix local_points = constrained_3d
      ? Keyframe::transform_points(
          target_points_all,
          keyframes[source_frame]->horizon_pose3().inverse())
      : Keyframe::transform_points(target_points_all, pose.inverse());
    // This runs over the WHOLE accumulated map cloud once per source frame,
    // so the two transcendentals per point dominate the gate. Both are
    // skipped for points a previous source frame already selected, and atan2
    // is skipped for anything outside the range bound — the same conjunction,
    // just short-circuited, so the selection is unchanged.
    for (int i = 0; i < local_points.rows(); ++i) {
      if (sel[i]) continue;
      const double range = std::hypot(local_points(i, 0), local_points(i, 1));
      if (!(range < range_bound)) continue;
      const double bearing = std::atan2(local_points(i, 1), local_points(i, 0));
      if (std::abs(bearing) < bearing_bound) {
        sel[i] = 1;
        ++n_sel;
      }
    }
  }

  Matrix target_points(n_sel, constrained_3d ? 3 : 2);
  std::vector<int> target_keys(n_sel);
  long at = 0;
  for (int i = 0; i < target_points_all.rows(); ++i) {
    if (!sel[i]) continue;
    target_points.row(at) = target_points_all.row(i);
    target_keys[at] = target_keys_all[i];
    ++at;
  }

  // find candidate frames with enough nearby points
  std::map<int, int> counts;
  for (int k : target_keys) counts[k]++;
  // the keyframes that survive the fan/covariance gate (>10 nearby points):
  // this is the LOCAL overlapping submap, not the whole map. Used below to
  // build the final target cloud, matching slam.py's reassignment of
  // target_frames = target_frames[counts > 10] (std::map keeps ascending order,
  // matching np.unique). Without this the loop-closure ICP/overlap/covariance
  // would register against the entire explored trajectory.
  std::vector<int> candidate_frames;  // ascending (std::map order)
  for (const auto& [frame, count] : counts)
    if (count > 10) candidate_frames.push_back(frame);

  if (candidate_frames.empty() ||
      target_points.rows() < nssm_params.min_points) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "target points " + std::to_string(target_points.rows());
    return ret;
  }

  // Revisit selection (intentional divergence from bruce_slam, which took the
  // raw argmax-overlap frame as the target). The
  // in-fan candidates include the current pass's own recent trail (the
  // keyframes just outside the min_st_sep exclusion), which carry the most
  // overlap. argmax therefore anchors the loop factor to a near-sequential
  // trailing frame ("same area, same time") and never to a genuine revisit, so
  // out-and-back passes are never tied together and walls render doubled.
  //
  // Two-stage fix, floor-primary so it is robust when the sonar range covers
  // the whole environment (candidates then form one unbroken index run with no
  // gap to cluster on — e.g. a small pool):
  //   (1) FLOOR: a revisit target must be at least min_revisit_sep keyframes
  //       older than the source. This alone excludes the recent trail.
  //   (2) CLUSTER (refinement): if the survivors still split into multiple
  //       contiguous index runs, the highest-index run is the far tail of the
  //       current pass's trail; drop it so the target is a distinct episode.
  //       A single run means the floor already isolated the revisit — keep it
  //       (never over-drop to empty).
  std::vector<int> revisit_frames;  // stays ascending (candidate_frames is)
  for (int f : candidate_frames)
    if (ret.source_key - f >= nssm_params.min_revisit_sep)
      revisit_frames.push_back(f);

  if (revisit_frames.empty()) {
    ret.status = Status(Status::NOT_ENOUGH_OVERLAP);
    ret.status.description =
      "no-revisit (candidates " + std::to_string(candidate_frames.size()) +
      ", none older than min_revisit_sep " +
      std::to_string(nssm_params.min_revisit_sep) + ")";
    return ret;
  }

  // Segment the surviving candidates into contiguous index RUNS — one run per
  // distinct visiting episode — and keep exactly ONE.
  //
  // The earlier form dropped only the TRAILING run, so with three or more runs
  // (two genuine earlier visits plus the trail tail) two distinct episodes
  // survived together and were aggregated into a single target cloud by
  // get_points() below — which transforms each keyframe's points by the
  // CURRENT graph estimate. If those episodes are mis-aligned in the graph,
  // which is exactly the situation the closure exists to fix, the target is
  // smeared by that misalignment and ICP registers against a doubled cloud.
  // That is the wall-doubling failure this whole revisit path was built to
  // prevent, reintroduced one step later at target construction.
  //
  // rtabmap has no such exposure: getPaths() segments candidates into
  // neighbour-link-connected runs and attempts ONE detection per path
  // (Rtabmap.cpp — "segment poses by paths, only one detection per path"),
  // comparing up to RGBD/ProximityMaxPaths of them SEPARATELY, never merging
  // clouds across paths.
  const int kRevisitGap = 3;  // index gap that splits one episode from the next
  std::vector<std::pair<std::size_t, std::size_t>> runs;  // [begin, end)
  {
    std::size_t begin = 0;
    for (std::size_t i = 1; i <= revisit_frames.size(); ++i)
      if (i == revisit_frames.size() ||
          revisit_frames[i] - revisit_frames[i - 1] > kRevisitGap) {
        runs.emplace_back(begin, i);
        begin = i;
      }
  }
  // >1 run: the highest-index run is the current pass's own trailing tail
  // (the same assumption as before). Never drop to empty.
  if (runs.size() > 1) runs.pop_back();
  // Among the remaining episodes take the one carrying the most in-fan
  // points — the visit that actually saw this structure best.
  std::size_t best_run = 0;
  long best_run_points = -1;
  for (std::size_t r = 0; r < runs.size(); ++r) {
    long n = 0;
    for (std::size_t i = runs[r].first; i < runs[r].second; ++i)
      n += counts[revisit_frames[i]];
    if (n > best_run_points) { best_run_points = n; best_run = r; }
  }
  candidate_frames.assign(revisit_frames.begin() + runs[best_run].first,
                          revisit_frames.begin() + runs[best_run].second);

  // Restrict the target cloud to the revisit frames so the Sobol init, the
  // post-init refine (which recomputes counts from target_keys), and the final
  // submap all register against the OLD pass, not the trail.
  const std::set<int> revisit_set(candidate_frames.begin(),
                                  candidate_frames.end());
  {
    long n_keep = 0;
    for (int k : target_keys) n_keep += revisit_set.count(k);
    Matrix kept_points(n_keep, constrained_3d ? 3 : 2);
    std::vector<int> kept_keys(n_keep);
    long at = 0;
    for (int i = 0; i < target_points.rows(); ++i) {
      if (!revisit_set.count(target_keys[i])) continue;
      kept_points.row(at) = target_points.row(i);
      kept_keys[at] = target_keys[i];
      ++at;
    }
    target_points = std::move(kept_points);
    target_keys = std::move(kept_keys);
  }

  if (target_points.rows() < nssm_params.min_points) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "target points " + std::to_string(target_points.rows());
    return ret;
  }

  // best target = max overlap among revisit frames only
  int best_frame = candidate_frames.front(), best_count = -1;
  for (int frame : candidate_frames) {
    const int count = counts[frame];
    if (count > best_count) { best_count = count; best_frame = frame; }
  }

  ret.target_key = best_frame;
  ret.target_pose = keyframes[ret.target_key]->pose;
  ret.target_points = constrained_3d
    ? Keyframe::transform_points(
        target_points, keyframes[ret.target_key]->horizon_pose3().inverse())
    : Keyframe::transform_points(target_points, ret.target_pose.inverse());
  ret.cov = keyframes[ret.source_key]->cov;

  if (!nssm_params.initialization) {
    if (constrained_3d)
      ret.source_points.col(2).array() += static_cast<float>(
        keyframes[ret.source_key]->pose3.z() -
        keyframes[ret.target_key]->pose3.z());
    return ret;
  }

  // Bounds from the source-pose uncertainty. slam.py reused `cov` leaked from
  // the fan-selection loop above — the OLDEST source frame's covariance, a
  // loop-variable-leak bug. Use the anchor keyframe's covariance instead
  // (ret.cov == keyframes[source_key], the newest source frame), so the search
  // bounds match ret.source_pose; its marginal was just refreshed by
  // update_factor_graph, so it is current. (Intentional divergence from
  // bruce_slam.)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(ret.cov.topLeftCorner<2, 2>());
  const double translation_std = std::sqrt(es.eigenvalues().maxCoeff());
  const double rotation_std = std::sqrt(ret.cov(2, 2));
  // Compass-clamped init yaw (MAP_DOUBLING_FIX_PLAN.md fix 3a): the compass
  // gate downstream rejects any closure whose relative yaw disagrees with DR
  // by more than nssm_max_yaw_vs_compass, so yaw candidates beyond that band
  // can only produce closures the gate will kill — and on near-square
  // geometry a loose ±5σ window spans the ~90° alias basin, where the Sobol
  // search then lands (56% of CHL_Pool rejects were compass rejects).
  // Translation keeps the full ±5σ freedom; only yaw is clamped to the band
  // the gate already enforces.
  const double yaw_bound =
    nssm_max_yaw_vs_compass > 0.0
      ? std::min(5.0 * rotation_std, nssm_max_yaw_vs_compass)
      : 5.0 * rotation_std;
  // Optional TRANSLATION clamp (parallel-wall aliasing lever, default off): cap
  // the Sobol init x/y search so it cannot reach a parallel-wall basin many
  // metres off the DR-predicted overlap. 0 keeps the full ±5σ freedom.
  const double trans_bound =
    nssm_max_translation_vs_dr > 0.0
      ? std::min(5.0 * translation_std, nssm_max_translation_vs_dr)
      : 5.0 * translation_std;
  Eigen::Matrix<double, 3, 2> bounds;
  bounds << -trans_bound, trans_bound,
            -trans_bound, trans_bound,
            -yaw_bound, yaw_bound;

  const GlobalInitResult init = global_scan_match_init(
    ret.source_points, ret.source_pose, ret.target_points, ret.target_pose,
    point_noise, bounds, nssm_params.init_n, nssm_params.init_iters,
    nssm_params.init_ftol);

  if (!init.success) {
    ret.status = Status(Status::INITIALIZATION_FAILURE);
    return ret;
  }

  ret.estimated_source_pose = ret.source_pose.compose(init.delta);
  ret.has_estimated_source_pose = true;
  ret.source_pose_samples = init.pose_samples;
  ret.status.description = "matching cost " + std::to_string(init.best_cost);

  // refine the target key: the frame with maximum overlap against the
  // initialized source points
  const Matrix estimated_source_points = constrained_3d
    ? Keyframe::transform_points(
        ret.source_points,
        gtsam::Pose3(
          gtsam::Rot3::Yaw(ret.estimated_source_pose.theta()),
          gtsam::Point3(ret.estimated_source_pose.x(),
                        ret.estimated_source_pose.y(),
                        keyframes[ret.source_key]->pose3.z())))
    : Keyframe::transform_points(ret.source_points,
                                 ret.estimated_source_pose);
  std::vector<int> indices;
  get_overlap(estimated_source_points, target_points, nullptr, &indices);

  std::map<int, int> counts1;
  for (std::size_t i = 0; i < indices.size(); ++i)
    if (indices[i] != -1) counts1[target_keys[indices[i]]]++;
  if (counts1.empty()) {
    ret.status = Status(Status::NOT_ENOUGH_OVERLAP);
    ret.status.description = "0";
    return ret;
  }

  int refined_frame = -1, refined_count = -1;
  for (const auto& [frame, count] : counts1)
    if (count > refined_count) { refined_count = count; refined_frame = frame; }

  ret.target_key = refined_frame;
  ret.target_pose = keyframes[ret.target_key]->pose;
  // aggregate only the fan-selected candidate submap (not the whole map) into
  // the refined target frame — see candidate_frames above.
  ret.target_points = get_points(candidate_frames, ret.target_key);
  if (constrained_3d)
    ret.source_points.col(2).array() += static_cast<float>(
      keyframes[ret.source_key]->pose3.z() -
      keyframes[ret.target_key]->pose3.z());

  return ret;
}

namespace {
// Histogram bucket for a terminal NSSM rejection. Status::name() already gives
// a stable category, but LARGE_TRANSFORMATION covers both the geometric
// trans/rot bound (tunable via max_translation/max_rotation) AND the
// compass-consistency gate (the aliasing guard — do NOT loosen), so split them
// by the description prefix set at their respective gate sites.
std::string nssm_hist_key(const Status& s)
{
  std::string key = s.name();
  const std::string& d = s.description;
  if (d.rfind("compass-inconsistent", 0) == 0) key += " (compass)";
  else if (d.rfind("DR-translation-inconsistent", 0) == 0)
    key += " (DR translation)";
  else if (d.rfind("degenerate", 0) == 0) key += " (degenerate)";
  else if (d.rfind("no-revisit", 0) == 0) key += " (no-revisit)";
  return key;
}
}  // namespace

bool Slam::add_nonsequential_scan_matching()
{
  last_nssm_inserted_geom.clear();
  // current_frame is assigned at the END of the SLAM callback, so it is null
  // until the second callback
  if (!current_frame || current_key() < nssm_params.min_st_sep) return false;

  ++nssm_attempts;
  InitializationResult ret = initialize_nonsequential_scan_matching();
  if (!ret.status) {
    last_nssm_status =
      std::string(ret.status.name()) + " (" + ret.status.description + ")";
    ++nssm_reject_hist[nssm_hist_key(ret.status)];
    return false;
  }

  ICPResult ret2(ret, nssm_params.cov_samples > 0 &&
                        nssm_params.cov_method == SMParams::SAMPLED);
  run_scan_match_icp(ret2, nssm_params);

  if (ret2.status) {
    const gtsam::Pose2 delta =
      ret2.initial_transform.between(ret2.estimated_transform);
    const double delta_translation =
      std::hypot(delta.translation().x(), delta.translation().y());
    const double delta_rotation = std::abs(delta.theta());
    // inverted comparisons so a NaN delta (degenerate transform) fails closed
    if (!(delta_translation <= nssm_params.max_translation) ||
        !(delta_rotation <= nssm_params.max_rotation)) {
      ret2.status = Status(Status::LARGE_TRANSFORMATION);
      ret2.status.description = "trans " + std::to_string(delta_translation) +
                                " rot " + std::to_string(delta_rotation);
    }
  }

  if (ret2.status) {
    const int overlap = get_overlap(ret2.source_points, ret2.target_points,
                                    &ret2.estimated_transform);
    const long src_n = ret2.source_points.rows();
    const double ratio = src_n > 0 ? static_cast<double>(overlap) / static_cast<double>(src_n) : 0.0;
    if (overlap < nssm_params.min_points ||
        (nssm_params.min_overlap_ratio > 0.0 &&
         ratio < nssm_params.min_overlap_ratio))
      ret2.status = Status(Status::NOT_ENOUGH_OVERLAP);
    // append — don't clobber the success path's "N samples" diagnostic
    if (!ret2.status.description.empty()) ret2.status.description += ", ";
    ret2.status.description += "overlap " + std::to_string(overlap) +
                              " (" + std::to_string(ratio) + " of source)";
  }

  // Absolute translation-consistency gate, paired with the Sobol bound in
  // initialize_nonsequential_scan_matching(). The initialization clamp alone
  // is not a final bound: ICP may refine as far as nssm/max_translation from
  // that seed. On CHL_Pool, max_translation_vs_dr=1.0 still admitted
  // parallel-wall aliases demanding 2-5 m corrections because ICP walked
  // from the edge of the 1 m initialization window into the alias basin.
  // Judge the final target->source transform against the predicted one,
  // matching the absolute compass check below. 0 disables.
  //
  // CROSS-SESSION (2026-07-26): the predictor comes from
  // predicted_relative_pose, NOT from dr_pose directly. A closure onto a
  // LOADED map spans two dead-reckoning epochs, whose dr_pose difference is
  // an arbitrary offset — typically the whole distance between the two
  // missions' DR origins, so every such closure failed this gate and map
  // persistence silently produced no loop closures at all (measured: 0
  // accepted post-load with the gate at 1.0 m, 10 with it disabled). Across
  // the boundary the graph estimate is the common frame and becomes the
  // reference; within one epoch DR still is, unchanged. The chain-tear check
  // skips the same boundary for the same reason (loaded_key_count_ below).
  if (ret2.status && nssm_max_translation_vs_dr > 0.0) {
    bool dr_backed = true;
    const gtsam::Pose2 rel =
      predicted_relative_pose(ret2.target_key, ret2.source_key, &dr_backed);
    const gtsam::Pose2 corr = rel.between(ret2.estimated_transform);
    const double trans_err = std::hypot(corr.x(), corr.y());
    // Inverted comparison makes a non-finite transform fail closed.
    if (!(trans_err <= nssm_max_translation_vs_dr)) {
      ret2.status = Status(Status::LARGE_TRANSFORMATION);
      ret2.status.description =
        std::string("DR-translation-inconsistent (closure translation vs ") +
        (dr_backed ? "DR" : "graph estimate, cross-session") +
        " differs by " + std::to_string(trans_err) + " m)";
    }
  }

  // Optional DR-yaw consistency gate. This was added after discrete rotational
  // aliasing in a near-square pool produced confident wrong locks. Field work
  // around steel subsequently showed that the compass-backed DR yaw can jump,
  // so zero now explicitly disables this gate. Geometric covariance, overlap,
  // PCM, and chain-tear checks remain active when it is disabled.
  //
  // Deliberately reads dr_pose directly rather than going through
  // predicted_relative_pose because this optional check compares against the
  // DR yaw source itself, not the graph. It is disabled in the deployed
  // steel-structure configuration.
  if (ret2.status && nssm_max_yaw_vs_compass > 0.0) {
    const double dr_rel_yaw = keyframes[ret2.target_key]
                                ->dr_pose.between(keyframes[ret2.source_key]->dr_pose)
                                .theta();
    const double yaw_err = std::abs(std::remainder(
      ret2.estimated_transform.theta() - dr_rel_yaw, 2.0 * M_PI));
    if (yaw_err > nssm_max_yaw_vs_compass) {
      ret2.status = Status(Status::LARGE_TRANSFORMATION);
      ret2.status.description =
        "compass-inconsistent (closure yaw vs DR yaw differs by " +
        std::to_string(yaw_err) + " rad)";
    }
  }

  // Degeneracy gate:
  // pool/wall aliasing produced self-consistent-but-wrong closures that PCM
  // could not reject (2026-07 CHL_Pool: 9 accepted closures demanding 21 deg
  // RMS yaw correction — why NSSM was turned off). Sliding along featureless
  // structure shows up as an elongated translation covariance in the
  // sampled/Censi estimate, so reject closures whose covariance is too large
  // or too anisotropic. nssm/max_rotation separately bounds how far ICP may
  // refine away from its initialization.
  if (ret2.status && ret2.has_cov) {
    // Gate on the pre-floor covariance when enabled: the per-eigenvalue floor in
    // compute_icp_with_cov otherwise caps the observable anisotropy and lets
    // elongated wall-slide covariances pass this gate.
    const Eigen::Matrix3d& gate_cov =
      nssm_degeneracy_prefloor ? ret2.cov_raw : ret2.cov;
    const IcpObservability obs = assess_icp_observability(
      gate_cov, nssm_max_sigma, nssm_max_anisotropy);
    if (!obs.observable) {
      ret2.status = Status(Status::NOT_CONVERGED);
      ret2.status.description =
        "degenerate: sigma " + std::to_string(obs.sigma_max) +
        " aniso " + std::to_string(obs.anisotropy);
    }
  }

  bool accepted = false;
  if (ret2.status) {
    ++nssm_queued;
    // slide the PCM queue window
    while (!nssm_queue_.empty() &&
           ret2.source_key - nssm_queue_.front().source_key > pcm_queue_size)
      nssm_queue_.pop_front();

    nssm_queue_.push_back(ret2);
    const std::vector<int> pcm = verify_pcm(nssm_queue_, min_pcm);
    nssm_best_clique = std::max(nssm_best_clique, static_cast<int>(pcm.size()));
    last_nssm_status = pcm.empty()
      ? "queued (queue " + std::to_string(nssm_queue_.size()) +
          ", no consistent clique >= " + std::to_string(min_pcm) + ")"
      : "PCM clique " + std::to_string(pcm.size());

    for (int m : pcm) {
      ICPResult& loop = nssm_queue_[m];
      // rejected: quarantined by a post-loop verification revert — never
      // re-insert a closure that already demonstrably bent the graph
      if (loop.inserted || loop.rejected) continue;

      // Loop-factor noise model. DCS (Agarwal et al.) is OPT-IN
      // (nssm_use_dcs): it scales weight by 2*phi/(phi+chi2), and a genuine
      // closure correcting drift D starts at chi2=(D/sigma)^2 — large BY
      // DESIGN — so with cm-scale ICP sigmas it muted exactly the meaningful
      // corrections (0.5% weight at D=1 m, sigma=5 cm) while corrections
      // small enough to pass were invisible. Outlier damage-limitation is the
      // optimize-then-verify revert + quarantine below, which judges the real
      // optimized graph instead of pre-shrinking every correction.
      const gtsam::SharedNoiseModel base =
        loop.has_cov
          ? gtsam::noiseModel::Gaussian::Covariance(embed_planar_cov(loop.cov))
          : icp_odom_model_;
      const gtsam::SharedNoiseModel model =
        nssm_use_dcs
          ? gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::DCS::Create(nssm_dcs_phi), base)
          : base;
      graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        X(loop.target_key), X(loop.source_key),
        lift_planar(loop.estimated_transform), model));
      keyframes[loop.source_key]->constraints.emplace_back(
        loop.target_key, loop.estimated_transform);
      loop.inserted = true;
      // Diagnostic: the correction this closure demands, ICP relative pose vs
      // the predicted relative pose (same target->source convention as the
      // factor above). Legit ~1 m drift fix => trans ~drift, yaw ~0; a
      // parallel-wall alias => large trans and/or non-zero yaw. Logged by the
      // node so it lands right before any post-loop revert message.
      // Uses the same predictor as the translation gate, so a cross-session
      // closure reports against the graph estimate instead of printing the
      // meaningless difference of two DR epochs (labelled in the text).
      {
        bool dr_backed = true;
        const gtsam::Pose2 dr_rel = predicted_relative_pose(
          loop.target_key, loop.source_key, &dr_backed);
        const gtsam::Pose2 corr = dr_rel.between(loop.estimated_transform);
        last_nssm_inserted_geom.push_back(
          "closure src=" + std::to_string(loop.source_key) + " tgt=" +
          std::to_string(loop.target_key) + " gap=" +
          std::to_string(loop.source_key - loop.target_key) +
          (dr_backed ? "" : " [cross-session: ref=graph]") +
          " | ICP-vs-DR corr trans=" +
          std::to_string(std::hypot(corr.x(), corr.y())) + "m yaw=" +
          std::to_string(corr.theta()) + "rad | DR rel=(" +
          std::to_string(dr_rel.x()) + "," + std::to_string(dr_rel.y()) + "," +
          std::to_string(dr_rel.theta()) + ") ICP rel=(" +
          std::to_string(loop.estimated_transform.x()) + "," +
          std::to_string(loop.estimated_transform.y()) + "," +
          std::to_string(loop.estimated_transform.theta()) + ") | " +
          loop.status.description);
      }
      // tentative until the next update_factor_graph succeeds; rolled back
      // (inserted flag, constraints entry, counter) if the solve throws
      pending_loops_.push_back(m);
      ++nssm_accepted;
      accepted = true;
      last_nssm_status = "accepted (clique " + std::to_string(pcm.size()) + ")";
    }
  } else {
    last_nssm_status =
      std::string(ret2.status.name()) + " (" + ret2.status.description + ")";
    ++nssm_reject_hist[nssm_hist_key(ret2.status)];
  }
  return accepted;
}

std::string Slam::nssm_reject_summary() const
{
  if (nssm_reject_hist.empty()) return "none";
  std::string out;
  for (const auto& [reason, count] : nssm_reject_hist) {
    if (!out.empty()) out += ", ";
    out += reason + ":" + std::to_string(count);
  }
  return out;
}

// ---------------------------------------------------------------------------
// manual correction
// ---------------------------------------------------------------------------
bool Slam::add_manual_correction(const gtsam::Pose2& map_pose)
{
  if (keyframes.empty()) return false;
  const int key = current_key() - 1;
  // planar prior lifted into the horizon chart: roll/pitch/z stay wide (the
  // per-keyframe unary owns them; z target = current z so the wide prior is
  // centered), x/y/yaw at the operator-trust sigmas
  const gtsam::Pose3 target(
    gtsam::Rot3::Yaw(map_pose.theta()),
    gtsam::Point3(map_pose.x(), map_pose.y(),
                  keyframes[key]->horizon_pose3().z()));
  // where this prior will land inside committed_graph_ once
  // update_factor_graph mirrors the buffer (push_back preserves append
  // order) — recorded so undo_manual_correction can remove exactly this
  // factor later
  const std::size_t committed_idx = committed_graph_.size() + graph_.size();
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
    X(key), target,
    lift_sigmas(manual_correction_sigmas, kSigmaWide, kSigmaWide)));
  force_converge_ = true;
  if (!update_factor_graph()) return false;
  manual_prior_indices_.push_back(committed_idx);
  ++manual_corrections_applied;
  return true;
}

bool Slam::undo_manual_correction()
{
  if (manual_prior_indices_.empty()) {
    last_error_ = "no manual correction to undo";
    return false;
  }
  const std::size_t idx = manual_prior_indices_.back();
  // Null-hole removal (FactorGraph::remove) keeps every other recorded index
  // valid; rebuild_isam skips the holes when feeding the fresh estimator.
  // The index is always live: verification/failure truncations only drop
  // factors of rounds that were never recorded here.
  gtsam::NonlinearFactor::shared_ptr removed;
  if (idx < committed_graph_.size() && committed_graph_[idx]) {
    removed = committed_graph_[idx];
    committed_graph_.remove(idx);
  }
  rebuild_isam();
  // The rebuild relinearizes AT the corrected poses, which are no longer the
  // optimum without the prior — relax the trajectory back and sweep the new
  // estimate into the keyframes through the ordinary update path (the factor
  // buffer is empty; converge hard, mirroring the correction round this
  // undoes; pending_loops_ is empty so no verification runs).
  force_converge_ = true;
  if (!update_factor_graph()) {
    // the undo did NOT apply — restore the factor and the estimator, or a
    // retried "undo failed" would peel the PREVIOUS correction instead
    if (removed) {
      committed_graph_.replace(idx, removed);
      rebuild_isam();
    }
    return false;
  }
  manual_prior_indices_.pop_back();
  ++manual_corrections_undone;
  return true;
}

// ---------------------------------------------------------------------------
// map persistence, relocalization, absolute position fixes
// ---------------------------------------------------------------------------
namespace {

constexpr char kMapMagic[9] = "SSLMMAP2";
constexpr std::uint32_t kMapVersion = 1;

void put_pose3(std::ofstream& f, const gtsam::Pose3& p)
{
  const gtsam::Quaternion q = p.rotation().toQuaternion();
  const double d[7] = {q.w(), q.x(), q.y(), q.z(), p.x(), p.y(), p.z()};
  f.write(reinterpret_cast<const char*>(d), sizeof d);
}

gtsam::Pose3 get_pose3(std::ifstream& f)
{
  double d[7];
  f.read(reinterpret_cast<char*>(d), sizeof d);
  return gtsam::Pose3(gtsam::Rot3::Quaternion(d[0], d[1], d[2], d[3]),
                      gtsam::Point3(d[4], d[5], d[6]));
}

}  // namespace

bool Slam::save_map(const std::string& path)
{
  // Write to a sibling temp file and rename into place.
  //
  // Opening `path` directly with trunc destroyed the existing map the moment
  // the save STARTED, so a full disk, a crash, or a kill mid-write left
  // neither the old map nor a valid new one — and this file can represent
  // hours of survey. The short-write check below already existed, but by the
  // time it fires the previous map is long gone.
  //
  // rename(2) within a directory is atomic, so any reader (including a later
  // load_map) sees either the untouched old map or the complete new one.
  const std::string tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
  if (!f) {
    last_error_ = "cannot open '" + tmp + "' for writing";
    return false;
  }
  f.write(kMapMagic, 8);
  const std::uint32_t version = kMapVersion;
  const std::uint32_t n = static_cast<std::uint32_t>(keyframes.size());
  f.write(reinterpret_cast<const char*>(&version), 4);
  f.write(reinterpret_cast<const char*>(&n), 4);
  for (const auto& kf : keyframes) {
    const std::int64_t stamp_ns =
      static_cast<std::int64_t>(kf->time.sec) * 1000000000ll + kf->time.nanosec;
    f.write(reinterpret_cast<const char*>(&stamp_ns), 8);
    put_pose3(f, kf->dr_pose3);
    put_pose3(f, kf->pose3);
    const double p2[3] = {kf->pose.x(), kf->pose.y(), kf->pose.theta()};
    f.write(reinterpret_cast<const char*>(p2), sizeof p2);
    const std::int64_t rows = kf->points.rows();
    f.write(reinterpret_cast<const char*>(&rows), 8);
    if (rows > 0) {
      // row-major float triplets, independent of Eigen's storage order
      std::vector<float> buf(static_cast<std::size_t>(rows) * 3);
      for (long r = 0; r < rows; ++r)
        for (int c = 0; c < 3; ++c)
          buf[static_cast<std::size_t>(r) * 3 + c] = kf->points(r, c);
      f.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(buf.size() * sizeof(float)));
    }
  }
  // the buffered stream only surfaces ENOSPC at flush — close BEFORE the
  // verdict or a truncated save reports success
  f.close();
  std::error_code ec;
  if (f.fail()) {
    last_error_ = "short write to '" + tmp + "'";
    std::filesystem::remove(tmp, ec);  // leave no partial file behind
    return false;
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    last_error_ = "cannot move '" + tmp + "' into place: " + ec.message();
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

bool Slam::load_map(const std::string& path)
{
  if (!keyframes.empty()) {
    last_error_ = "load_map requires an empty session (call before any frame)";
    return false;
  }
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    last_error_ = "cannot open '" + path + "'";
    return false;
  }
  char magic[8];
  std::uint32_t version = 0, n = 0;
  f.read(magic, 8);
  f.read(reinterpret_cast<char*>(&version), 4);
  f.read(reinterpret_cast<char*>(&n), 4);
  if (!f || std::memcmp(magic, kMapMagic, 8) != 0 || version != kMapVersion) {
    last_error_ = "'" + path + "' is not a version-" +
                  std::to_string(kMapVersion) + " sonar_slam map";
    return false;
  }
  constexpr std::uint32_t kMaxKeyframes = 1000000;
  if (n > kMaxKeyframes) {
    last_error_ = "map header claims an implausible keyframe count";
    return false;
  }

  gtsam::NonlinearFactorGraph g;
  gtsam::Values v;
  for (std::uint32_t k = 0; k < n; ++k) {
    std::int64_t stamp_ns = 0, rows = 0;
    f.read(reinterpret_cast<char*>(&stamp_ns), 8);
    const gtsam::Pose3 dr = get_pose3(f);
    const gtsam::Pose3 p3 = get_pose3(f);
    double p2[3];
    f.read(reinterpret_cast<char*>(p2), sizeof p2);
    f.read(reinterpret_cast<char*>(&rows), 8);
    if (!f || rows < 0 || rows > 50000000) {
      last_error_ = "truncated/corrupt map file at keyframe " +
                    std::to_string(k);
      keyframes.clear();
      return false;
    }
    builtin_interfaces::msg::Time t;
    t.sec = static_cast<std::int32_t>(stamp_ns / 1000000000ll);
    t.nanosec = static_cast<std::uint32_t>(stamp_ns % 1000000000ll);
    auto kf = std::make_shared<Keyframe>(true, t, dr);
    kf->points = Matrix(rows, 3);
    if (rows > 0) {
      std::vector<float> buf(static_cast<std::size_t>(rows) * 3);
      f.read(reinterpret_cast<char*>(buf.data()),
             static_cast<std::streamsize>(buf.size() * sizeof(float)));
      // a truncation INSIDE the last keyframe's payload would otherwise load
      // "successfully" with the missing tail as (0,0,0) phantom points
      if (!f) {
        last_error_ = "truncated point payload at keyframe " + std::to_string(k);
        keyframes.clear();
        return false;
      }
      for (long r = 0; r < rows; ++r)
        for (int c = 0; c < 3; ++c)
          kf->points(r, c) = buf[static_cast<std::size_t>(r) * 3 + c];
    }
    // restore the optimized estimate through the horizon chart (yaw-only
    // rotation, saved z); roll/pitch ride from the saved DR pose
    kf->update(gtsam::Pose3(gtsam::Rot3::Yaw(p2[2]),
                            gtsam::Point3(p2[0], p2[1], p3.z())));
    keyframes.push_back(kf);

    // Rebuild an equivalent-strength graph instead of serializing factors:
    // the saved poses are a solved optimum, so an anchored chain of tight
    // between factors holds the shape exactly, without the fragility of
    // serializing arbitrary factor types. Loop closures' effect is baked
    // into the saved poses.
    const int key = static_cast<int>(k);
    if (k == 0) {
      g.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), kf->horizon_pose3(),
                                             prior_model_));
    } else {
      const gtsam::Pose3 rel =
        keyframes[k - 1]->horizon_pose3().between(kf->horizon_pose3());
      g.add(gtsam::BetweenFactor<gtsam::Pose3>(
        X(key - 1), X(key), rel,
        lift_sigmas(icp_odom_sigmas, kSigmaTightRP, kSigmaDepthDelta)));
    }
    g.add(gtsam::PriorFactor<gtsam::Pose3>(X(key), kf->horizon_dr_pose3(),
                                           unary_model_));
    v.insert(X(key), kf->horizon_pose3());
  }

  if (f.peek() != std::ifstream::traits_type::eof()) {
    last_error_ = "trailing data after the last keyframe — corrupt map file";
    keyframes.clear();
    return false;
  }

  try {
    isam_.update(g, v);
  } catch (const std::exception& e) {
    last_error_ = std::string("loaded map failed to solve: ") + e.what();
    keyframes.clear();
    isam_ = gtsam::ISAM2(make_isam2_params());
    return false;
  }
  committed_graph_ = g;
  loaded_key_count_ = static_cast<int>(n);
  // A restored map carries poses and clouds but not the per-link factor
  // measurements, so every loaded link is marked invalid and skipped by the
  // post-loop chain-tear check. Sizing the vector here keeps it aligned with
  // `keyframes` so links added this session land at the right index.
  consecutive_links_.assign(keyframes.size(), ConsecutiveLink{});
  awaiting_relocalization_ = true;
  return true;
}

bool Slam::relocalize(const KeyframePtr& frame)
{
  if (!awaiting_relocalization_) {
    last_error_ = "no loaded map awaiting relocalization";
    return false;
  }
  if (frame->points.rows() < nssm_params.min_points) {
    last_error_ = "frame too sparse (" + std::to_string(frame->points.rows()) +
                  " < " + std::to_string(nssm_params.min_points) + " points)";
    return false;
  }
  std::vector<int> keys(static_cast<std::size_t>(loaded_key_count_));
  for (int k = 0; k < loaded_key_count_; ++k) keys[static_cast<std::size_t>(k)] = k;
  const Matrix target = get_points(keys, -1);
  if (target.rows() == 0) {
    last_error_ = "loaded map has no points";
    return false;
  }

  // Search the whole map footprint. Yaw stays near DR as a finite search
  // prior; the deployed EKF now propagates that yaw primarily from A50 gyro
  // deltas and gives magnetic heading only weak, innovation-gated authority.
  float min_x = 1e30f, max_x = -1e30f, min_y = 1e30f, max_y = -1e30f;
  for (long r = 0; r < target.rows(); ++r) {
    min_x = std::min(min_x, target(r, 0));
    max_x = std::max(max_x, target(r, 0));
    min_y = std::min(min_y, target(r, 1));
    max_y = std::max(max_y, target(r, 1));
  }
  // yaw-0 center: global_scan_match_init composes delta in the CENTER's
  // body frame, so a heading-carrying center would rotate the translation
  // search box with the vehicle and miss parts of an elongated map. With
  // theta = 0 the delta x/y live in world axes and the delta yaw IS the
  // absolute yaw, bounded around the DR heading prior.
  const gtsam::Pose2 center(0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.0);
  const double dr_yaw = frame->dr_pose.theta();
  Eigen::Matrix<double, 3, 2> bounds;
  bounds << -(0.5 * (max_x - min_x) + 5.0), 0.5 * (max_x - min_x) + 5.0,
            -(0.5 * (max_y - min_y) + 5.0), 0.5 * (max_y - min_y) + 5.0,
            dr_yaw - 0.35, dr_yaw + 0.35;

  const GlobalInitResult init = global_scan_match_init(
    frame->points, center, target, gtsam::Pose2(), point_noise, bounds,
    500, 4, 0.01);
  const double hits = -init.best_cost;
  const double need = relocalize_min_overlap * static_cast<double>(frame->points.rows());
  if (!init.success || hits < need) {
    last_error_ = "relocalization overlap too low (" +
                  std::to_string(static_cast<int>(hits)) + " hits < " +
                  std::to_string(static_cast<int>(need)) + " needed)";
    return false;
  }
  gtsam::Pose2 matched = center.compose(
    gtsam::Pose2(init.delta.x(), init.delta.y(), init.delta.theta()));

  if (constrained_3d) {
    Matrix source_global_depth = frame->points;
    source_global_depth.col(2).array() +=
      static_cast<float>(frame->dr_pose3.z());
    const auto [message, refined] = compute_icp(
      source_global_depth, target, matched);
    if (message != "success") {
      last_error_ = "3-D relocalization refinement failed: " + message;
      return false;
    }
    matched = refined;
    const int overlap = get_overlap(
      source_global_depth, target, &matched);
    if (overlap < need) {
      last_error_ = "3-D relocalization overlap too low (" +
                    std::to_string(overlap) + " hits < " +
                    std::to_string(static_cast<int>(need)) + " needed)";
      return false;
    }
  }

  // enter the map at the matched pose: moderate prior (the global init is
  // grid-coarse; SSM/NSSM tighten from here), plus the usual state unary
  frame->update(gtsam::Pose3(gtsam::Rot3::Yaw(matched.theta()),
                             gtsam::Point3(matched.x(), matched.y(),
                                           frame->dr_pose3.z())));
  const int key = current_key();
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
    X(key), frame->horizon_pose3(),
    lift_sigmas(Eigen::Vector3d(0.5, 0.5, 0.1), kSigmaTightRP, kSigmaDepthAbs)));
  values_.insert(X(key), frame->horizon_pose3());
  add_state_unary(key, frame);
  force_converge_ = true;
  // stay armed if the solve fails — update_factor_graph rolled the frame
  // back, so the next dense frame retries cleanly
  if (!update_factor_graph(frame)) return false;
  awaiting_relocalization_ = false;
  return true;
}

bool Slam::add_position_prior(int key, double x, double y, double sigma_xy)
{
  if (key < 0 || key >= current_key()) {
    last_error_ = "position prior key out of range";
    return false;
  }
  // position-only fix: yaw/attitude stay wide (USBL measures position, not
  // heading); z target = current estimate so the wide z prior is centered
  const gtsam::Pose3 h = keyframes[static_cast<std::size_t>(key)]->horizon_pose3();
  const gtsam::Pose3 target(h.rotation(), gtsam::Point3(x, y, h.z()));
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
    X(key), target,
    lift_sigmas(Eigen::Vector3d(sigma_xy, sigma_xy, kSigmaWide), kSigmaWide,
                kSigmaWide)));
  if (!update_factor_graph()) return false;
  ++position_priors_applied;
  return true;
}

// ---------------------------------------------------------------------------
// graph update
// ---------------------------------------------------------------------------
bool Slam::update_factor_graph(const KeyframePtr& keyframe)
{
  if (keyframe) keyframes.push_back(keyframe);

  // Optimized-vs-MEASURED discrepancy for one consecutive keyframe link, in
  // SIGMA of that link's own noise model. Port of rtabmap
  // Graph.cpp:computeMaxGraphErrors (linearErrorRatio): the optimizer having
  // torn the chain to satisfy a loop closure shows up as the graph walking
  // away from what the link's factor actually measured.
  //
  // Two changes from the earlier form (2026-07-25):
  //   * compares against the LINK, not dead reckoning. DR is the measurement
  //     only for add_odometry links; an SSM-accepted link measures ICP, so
  //     the old DR comparison charged a healthy graph a standing offset (up
  //     to ssm/max_translation = 0.3 m) for agreeing with its own factor.
  //   * normalised by the link's sigma, which now scales with span
  //     (add_odometry) — so a link bridging a long DR-only gap through a
  //     head-pitch-gated sweep is no longer convicted for being long.
  // Componentwise max, matching rtabmap, and direction-sensitive: it reads
  // large for the classic fold-back that REVERSES a link while preserving
  // its length. Links with no recorded measurement (a loaded map's
  // keyframes) return 0 and are skipped.
  const auto link_error_ratio = [this](std::size_t k) -> double {
    const std::size_t li = k + 1;  // links are indexed by their newer endpoint
    if (li >= consecutive_links_.size() || !consecutive_links_[li].valid)
      return 0.0;
    const ConsecutiveLink& L = consecutive_links_[li];
    const gtsam::Pose2 opt =
      keyframes[k]->pose.between(keyframes[k + 1]->pose);
    const double lin = std::max(std::abs(L.measured.x() - opt.x()),
                                std::abs(L.measured.y() - opt.y()));
    return lin / L.trans_sigma;
  };

  // Pre-round snapshot, loop rounds only (add_nonsequential_scan_matching
  // fills pending_loops_ before calling us; a plain keyframe round leaves it
  // empty and skips the post-loop verification below). It serves two jobs
  // down there: the delta semantics of the chain-tear check, and an exact
  // revert.
  std::vector<gtsam::Pose3> pre_round_poses;
  std::vector<double> pre_tear;
  if (!pending_loops_.empty()) {
    pre_round_poses.reserve(keyframes.size());
    for (const auto& kf : keyframes)
      pre_round_poses.push_back(kf->horizon_pose3());
    if (post_loop_max_link_error_sigma > 0.0 && keyframes.size() > 1) {
      pre_tear.resize(keyframes.size() - 1);
      for (std::size_t k = 0; k + 1 < keyframes.size(); ++k)
        pre_tear[k] = link_error_ratio(k);
    }
  }

  // isam_.update can throw (e.g. IndeterminantLinearSystem on a rank-deficient
  // scan-match factor), and GTSAM gives NO exception guarantee: by the time
  // elimination throws, the new factor is already in nonlinearFactors_, X(n)
  // is in theta_, and the Bayes tree may be partially recalculated. Clearing
  // the local buffers cannot un-poison the estimator, so recovery is: drop the
  // offending frame (pop the keyframe, roll back loop bookkeeping) and REBUILD
  // ISAM2 from the mirror of everything that previously solved.
  try {
    isam_.update(graph_, values_);
  } catch (const std::exception& e) {
    last_error_ = e.what();
    graph_.resize(0);
    values_.clear();
    // a discarded manual-correction prior must not make the NEXT round
    // converge hard on its behalf
    force_converge_ = false;
    if (keyframe) { keyframes.pop_back(); consecutive_links_.resize(keyframes.size()); }
    // solver failure says nothing about the loops themselves — no quarantine
    rollback_pending_loops(/*quarantine=*/false);
    rebuild_isam();
    return false;
  }
  // The factors are inside the estimator the moment update() returns — mirror
  // them NOW so committed_graph_ can never desync from isam_ if anything below
  // throws. A verification revert truncates back to committed_before, which
  // restores the "committed graph excludes the reverted round" property the
  // rebuild depends on.
  const std::size_t committed_before = committed_graph_.size();
  committed_graph_.push_back(graph_);
  graph_.resize(0);
  values_.clear();

  try {
  // Converge BEFORE judging: a loop-closure (or manual-correction) round
  // needs more than the single Gauss-Newton pass isam_.update() performs —
  // verifying (yaw-RMS/chain-tear) against the half-relaxed transient would
  // revert and QUARANTINE genuine closures. A few extra iterations are cheap
  // and only run on correction rounds.
  const bool converge_hard = !pending_loops_.empty() || force_converge_;
  force_converge_ = false;
  if (converge_hard)
    for (int i = 0; i < loop_extra_iterations; ++i) isam_.update();
  const gtsam::Values values = isam_.calculateEstimate();
  // A healthy estimator always carries at least the anchor state; an empty
  // estimate means a preceding rebuild failed and the sweep below would
  // write identity into the newest keyframe — unwind through the post-solve
  // catch instead.
  if (values.empty() && !keyframes.empty())
    throw std::runtime_error("solver returned an empty estimate");
  // Extract the optimized poses sequentially (gtsam map lookups; throws
  // loudly on a missing key, and a throw must not originate inside an OpenMP
  // region), then re-transform every keyframe's cloud in PARALLEL — on a
  // loop-closure correction every keyframe moves, and this transform sweep
  // over the whole map history was the sequential half of the CPU spike
  // (the other half, ISAM2 relinearization, is TBB-parallel inside GTSAM).
  if (values.size() > keyframes.size())
    throw std::out_of_range(
      "solver returned more states than keyframes");  // was .at()'s job
  std::vector<gtsam::Pose3> new_poses(values.size());
  for (std::size_t x = 0; x < values.size(); ++x)
    new_poses[x] = values.at<gtsam::Pose3>(X(static_cast<int>(x)));
#pragma omp parallel for schedule(dynamic)
  for (std::size_t x = 0; x < new_poses.size(); ++x)
    keyframes[x]->update(new_poses[x]);
  // the newest optimized pose, for the marginal refresh below
  gtsam::Pose3 pose;
  if (!new_poses.empty()) pose = new_poses.back();

  // Optional rtabmap-style optimize-then-verify yaw check. It is retained for
  // deployments with an independently trustworthy DR yaw reference, but zero
  // disables it around steel. The geometric chain-tear check below remains
  // active independently.
  if (!pending_loops_.empty()) {
    // WINDOWED max-RMS: a fold bends a LOCAL run of keyframes, and a global
    // RMS dilutes as ~1/sqrt(N) with mission length until any localized fold
    // slips under a fixed threshold (a 5-frame 90° fold passes a 0.15 rad
    // global RMS once N ≳ 550). The worst sliding-window RMS keeps the
    // detector's sensitivity constant over the whole mission while still
    // averaging out per-frame compass noise; W spans several times the
    // observed fold extents.
    constexpr int kYawRmsWindow = 20;
    const std::size_t n_kf = keyframes.size();
    std::vector<double> prefix(n_kf + 1, 0.0);
    for (std::size_t i = 0; i < n_kf; ++i) {
      const double e = std::remainder(
        keyframes[i]->pose.theta() - keyframes[i]->dr_pose.theta(), 2.0 * M_PI);
      prefix[i + 1] = prefix[i] + e * e;
    }
    const std::size_t win =
      std::min<std::size_t>(n_kf, static_cast<std::size_t>(kYawRmsWindow));
    double worst_ms = 0.0;
    for (std::size_t i = 0; win > 0 && i + win <= n_kf; ++i)
      worst_ms =
        std::max(worst_ms, (prefix[i + win] - prefix[i]) / static_cast<double>(win));
    const double rms = std::sqrt(worst_ms);
    // Chain-tear check (2026-07-16 CHL_Pool fold): parallel-wall
    // TRANSLATIONAL aliases beat every per-closure gate — compass agrees
    // between parallel walls seen at the same heading, wall-to-wall locks
    // are compact (degeneracy gate blind), and mutually-consistent aliases
    // give PCM a perfect clique (md 0.0 observed) — but the optimizer can
    // only satisfy them by stretching weak (DR-only) consecutive links by
    // many meters (8-18 m edges observed in the folded graph).
    //
    // Judged PER LINK and as a DELTA against the pre-round snapshot: a link
    // is this round's fault only if it both breaks the bound AND is worse
    // than it already was. The original form — global max against the
    // absolute bound — latched: one link left stretched ~1.02 m read above
    // the 1.0 m bound on every subsequent round, vetoing closures on evidence
    // that had nothing to do with them. Per link rather than max-vs-max
    // because one old stretched link would otherwise mask a fresh tear
    // elsewhere in the chain — exactly the fold this check exists to catch.
    double worst_tear = 0.0;
    bool tear_bad = false;
    // Index of the consecutive link this round tore worst — the handle for
    // the surgical quarantine below (rtabmap keeps the equivalent as
    // MaxGraphErrors::linearLink so repairGraph can drop exactly that edge).
    long worst_tear_link = -1;
    if (post_loop_max_link_error_sigma > 0.0) {
      for (std::size_t k = 0; k + 1 < keyframes.size(); ++k) {
        // The link joining a LOADED map to the new session has no recorded
        // measurement (link_error_ratio returns 0 for it), but skip it
        // explicitly too: the two sides come from different DR epochs, so
        // nothing about it is meaningful to this check.
        if (loaded_key_count_ > 0 &&
            k + 1 == static_cast<std::size_t>(loaded_key_count_))
          continue;
        const double tear = link_error_ratio(k);
        const double before = k < pre_tear.size() ? pre_tear[k] : 0.0;
        // Re-solve jitter on a link this round never touched, in SIGMA.
        // A numerical guard, not a tuning lever, so it stays out of the
        // config. (Was 0.01 m; 0.05 sigma is the same order against the
        // 0.1-0.2 m sigmas these links carry.)
        constexpr double tear_jitter_eps = 0.05;  // sigma
        // A link legitimately stretched past the bound by an ACCEPTED
        // absolute fix (USBL / manual) absorbs every later loop correction
        // too — it moves well beyond the numeric epsilon on genuine rounds.
        // Judged against the epsilon it would coin-flip vetoes forever, so
        // an already-over-bound link must grow by a material fraction of
        // the bound to convict a new round.
        // Judge growth against the PERSISTENT baseline recorded when the
        // link first exceeded the bound (an accepted absolute fix), not the
        // previous round's snapshot — per-round references let repeated
        // sub-gate aliases ratchet the graph a fraction of the bound at a
        // time. NOTE this delta/baseline machinery survives the switch to a
        // sigma ratio: normalising fixes long-link false positives, it does
        // NOT fix latching, which is a separate failure mode.
        double ref = before;
        if (k < tear_baseline_.size() && tear_baseline_[k] >= 0.0)
          ref = tear_baseline_[k];
        const double growth_gate = ref > post_loop_max_link_error_sigma
                                     ? 0.25 * post_loop_max_link_error_sigma
                                     : tear_jitter_eps;
        if (tear > post_loop_max_link_error_sigma && tear > ref + growth_gate) {
          tear_bad = true;
          if (tear > worst_tear) {
            worst_tear = tear;
            worst_tear_link = static_cast<long>(k);
          }
        }
      }
    }
    const bool yaw_bad = !post_loop_yaw_consistent(rms, post_loop_max_yaw_rms);
    if (yaw_bad || tear_bad) {
      last_error_ = yaw_bad
        ? ("post-loop DR-yaw check failed: optimized-vs-DR windowed yaw RMS " +
           std::to_string(rms) + " rad > " +
           std::to_string(post_loop_max_yaw_rms))
        : ("post-loop chain-tear check failed: this round pulled a consecutive "
           "keyframe link " + std::to_string(worst_tear) +
           " sigma away from what its own factor measured > " +
           std::to_string(post_loop_max_link_error_sigma));
      last_nssm_status = "REVERTED (" + last_error_ + ")";
      ++nssm_reverted;
      // un-mirror this round's factors so the rebuild excludes them, and
      // QUARANTINE the loops: the clique demonstrably bent the graph, so it
      // must not re-form and re-fail (accept/revert/rebuild churn) on the
      // next candidate.
      committed_graph_.resize(committed_before);
      // Surgical quarantine: the yaw check is GLOBAL so it convicts the whole
      // round (-1), but a chain tear names one link, and only the closures
      // that actually span it can have stretched it. The rest are un-inserted
      // (their factors were just truncated away) but stay eligible — a clique
      // of 3 used to be blacklisted wholesale because one member was an
      // alias, and with min_pcm 2 losing the honest ones makes every later
      // clique harder to form.
      rollback_pending_loops(/*quarantine=*/true,
                             yaw_bad ? -1 : worst_tear_link);
      // Restore the pre-round poses EXACTLY, then rebuild from them. They are
      // the converged optimum of committed_graph_ (truncated back above, so
      // nothing it constrains has changed) — which means the rebuilt
      // estimator relinearizes AT the optimum and holds them. Seeding the
      // rebuild from the bent poses instead (what the sweep above wrote into
      // the keyframes) left a residual bend that a single isam_.update could
      // not walk back, and that residual is itself a chain tear: the delta
      // check above would read it as pre-existing and excuse it from then on.
      const std::size_t n = std::min(pre_round_poses.size(), keyframes.size());
#pragma omp parallel for schedule(dynamic)
      for (std::size_t x = 0; x < n; ++x)
        keyframes[x]->update(pre_round_poses[x]);
      rebuild_isam();
      return false;
    }
  }

  // verified (or no loops this round): the mirrored factors stand, and this
  // round's loop-closure marks become permanent
  pending_loops_.clear();

  // Maintain the per-link over-bound baseline the chain-tear growth gate
  // compares against: record the tear when a link FIRST ends a correction
  // round beyond the bound (typically stretched by an accepted USBL/manual
  // fix), clear it when the link heals back under the bound.
  if (converge_hard && post_loop_max_link_error_sigma > 0.0 &&
      keyframes.size() > 1) {
    tear_baseline_.resize(keyframes.size() - 1, -1.0);
    for (std::size_t k = 0; k + 1 < keyframes.size(); ++k) {
      if (loaded_key_count_ > 0 &&
          k + 1 == static_cast<std::size_t>(loaded_key_count_))
        continue;
      const double t = link_error_ratio(k);
      if (t > post_loop_max_link_error_sigma) {
        if (tear_baseline_[k] < 0.0) tear_baseline_[k] = t;
      } else {
        tear_baseline_[k] = -1.0;
      }
    }
  }

  // Only the latest keyframe's covariance is refreshed — a known limitation
  // carried from slam.py (its own "TODO propagate cov from previous keyframe"),
  // so older keyframes keep stale marginals. Recomputing every keyframe's
  // marginal each step is O(n^2) and would starve the callback.
  // marginalCovariance can throw on a weakly-constrained
  // variable (IndeterminantLinearSystem); keep the optimized pose and skip the
  // covariance update rather than propagating the failure out of the callback.
  try {
    const Eigen::MatrixXd cov6 =
      isam_.marginalCovariance(X(static_cast<int>(values.size()) - 1));
    // extract the planar (x, y, theta) block from the Pose3 tangent marginal
    // for the downstream 3x3 transf_cov machinery
    constexpr int map[3] = {3, 4, 2};
    Eigen::Matrix3d cov;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) cov(i, j) = cov6(map[i], map[j]);
    keyframes.back()->set_cov(cov);
  } catch (const std::exception&) {
    keyframes.back()->update(pose);
  }

  // PCM marginal refresh (fixes the DIVERGENCES.md known limitation: only
  // the newest keyframe's marginal was ever refreshed, so PCM tested
  // consistency against stale covariances and honest cliques never formed):
  // refresh the marginals of every keyframe involved in a queued closure —
  // a handful of O(1) calls per update, not the O(n^2) full-history sweep.
  for (const ICPResult& q : nssm_queue_) {
    for (const int k : {q.source_key, q.target_key}) {
      if (k < 0 || k >= static_cast<int>(keyframes.size())) continue;
      try {
        const Eigen::MatrixXd c6 = isam_.marginalCovariance(X(k));
        constexpr int map[3] = {3, 4, 2};
        Eigen::Matrix3d c;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) c(i, j) = c6(map[i], map[j]);
        keyframes[k]->set_cov(c);
      } catch (const std::exception&) {
        // weakly constrained variable: keep the stale marginal
      }
    }
  }

  // refresh the poses in pending loop closures for PCM
  for (ICPResult& ret : nssm_queue_) {
    ret.source_pose = keyframes[ret.source_key]->pose;
    ret.target_pose = keyframes[ret.target_key]->pose;
    if (ret.inserted)
      ret.estimated_transform = ret.target_pose.between(ret.source_pose);
  }
  return true;
  } catch (const std::exception& e) {
    // Any throw between a successful solve and the end of this round (the
    // estimate sweep's own consistency throw, an allocation failure, ...)
    // would otherwise strand a half-applied round: mirrored factors for a
    // keyframe we are about to drop, loop marks with no surviving factors,
    // half-swept poses. Unwind exactly like a verification revert — minus the
    // quarantine, since the loops themselves were never judged.
    last_error_ = std::string("post-solve failure: ") + e.what();
    committed_graph_.resize(committed_before);
    if (keyframe) { keyframes.pop_back(); consecutive_links_.resize(keyframes.size()); }
    rollback_pending_loops(/*quarantine=*/false);
    const std::size_t n = std::min(pre_round_poses.size(), keyframes.size());
    for (std::size_t x = 0; x < n; ++x) keyframes[x]->update(pre_round_poses[x]);
    rebuild_isam();
    return false;
  }
}

void Slam::rollback_pending_loops(bool quarantine, long culprit_link)
{
  // Loops are marked inserted (and appended to their keyframe's constraints)
  // in add_nonsequential_scan_matching, BEFORE the update that actually
  // incorporates their factors. If that update failed, the factors were
  // discarded — un-mark them; with quarantine=true (a verification revert)
  // they are additionally flagged rejected so the demonstrably-bad closure
  // can neither re-insert nor vote in future PCM cliques.
  //
  // culprit_link >= 0 narrows the quarantine to the closures that SPAN that
  // consecutive link (target_key <= k and k+1 <= source_key) — only those can
  // have stretched it. Everything else in the round is rolled back but stays
  // eligible. -1 quarantines the whole round, which is right for the yaw
  // check because that one is global and cannot attribute blame.
  //
  // Never quarantines nothing: if no pending loop spans the named link the
  // attribution failed, so fall back to convicting the round.
  bool any_spans = false;
  if (quarantine && culprit_link >= 0)
    for (int m : pending_loops_) {
      if (m < 0 || m >= static_cast<int>(nssm_queue_.size())) continue;
      const ICPResult& l = nssm_queue_[m];
      if (l.target_key <= culprit_link && culprit_link + 1 <= l.source_key) {
        any_spans = true;
        break;
      }
    }

  for (int m : pending_loops_) {
    if (m < 0 || m >= static_cast<int>(nssm_queue_.size())) continue;
    ICPResult& loop = nssm_queue_[m];
    if (!loop.inserted) continue;
    loop.inserted = false;
    const bool spans =
      !any_spans || (loop.target_key <= culprit_link &&
                     culprit_link + 1 <= loop.source_key);
    if (quarantine && spans) loop.rejected = true;
    if (loop.source_key >= 0 &&
        loop.source_key < static_cast<int>(keyframes.size())) {
      auto& constraints = keyframes[loop.source_key]->constraints;
      if (!constraints.empty()) constraints.pop_back();
    }
    --nssm_accepted;
  }
  pending_loops_.clear();
}

void Slam::resetSession()
{
  keyframes.clear();
  current_frame.reset();
  isam_ = gtsam::ISAM2(make_isam2_params());
  graph_.resize(0);
  values_.clear();
  committed_graph_ = gtsam::NonlinearFactorGraph();
  pending_loops_.clear();
  nssm_queue_.clear();
  consecutive_links_.clear();
  tear_baseline_.clear();
  manual_prior_indices_.clear();
  force_converge_ = false;
  loaded_key_count_ = 0;
  awaiting_relocalization_ = false;
  ssm_accepted = 0;
  ssm_degenerate_rejected = 0;
  nssm_accepted = 0;
  nssm_attempts = 0;
  nssm_queued = 0;
  nssm_best_clique = 0;
  nssm_reverted = 0;
  nssm_reject_hist.clear();
  last_nssm_inserted_geom.clear();
  last_ssm_status = "none yet";
  last_nssm_status = "none yet";
  last_pcm_min_md = -1.0;
  last_pcm_edges = 0;
  manual_corrections_applied = 0;
  manual_corrections_undone = 0;
  position_priors_applied = 0;
  last_error_.clear();
}

void Slam::rebuild_isam()
{
  // Reconstruct the estimator from the committed factor mirror, linearizing at
  // the last optimized keyframe poses. These factors solved together before,
  // so this update is expected to succeed; if it somehow does not, keep the
  // fresh estimator and report — the next frame retries against it.
  isam_ = gtsam::ISAM2(make_isam2_params());
  gtsam::Values estimates;
  for (std::size_t i = 0; i < keyframes.size(); ++i)
    estimates.insert(X(static_cast<int>(i)), keyframes[i]->horizon_pose3());
  try {
    // skip null holes (undo_manual_correction removes factors in place to
    // keep recorded indices stable)
    gtsam::NonlinearFactorGraph graph;
    graph.reserve(committed_graph_.size());
    for (const auto& f : committed_graph_)
      if (f) graph.push_back(f);
    isam_.update(graph, estimates);
  } catch (const std::exception& e) {
    last_error_ += std::string("; estimator rebuild also failed: ") + e.what();
  }
}

// ---------------------------------------------------------------------------
// pairwise consistency maximization
// ---------------------------------------------------------------------------
namespace {

// Bron-Kerbosch with pivoting (port of slam.py find_cliques)
void find_cliques(const std::map<int, std::set<int>>& adj,
                  std::vector<int>& current, std::set<int> subg,
                  std::set<int> cand, std::vector<std::vector<int>>& cliques)
{
  if (subg.empty()) {
    cliques.push_back(current);
    return;
  }
  // pivot: vertex maximizing |cand ∩ adj(u)|
  int pivot = *subg.begin();
  std::size_t best = 0;
  for (int u : subg) {
    std::size_t n = 0;
    for (int v : cand)
      if (adj.at(u).count(v)) ++n;
    if (n >= best) { best = n; pivot = u; }
  }
  std::vector<int> ext;
  for (int v : cand)
    if (!adj.at(pivot).count(v)) ext.push_back(v);

  for (int q : ext) {
    cand.erase(q);
    current.push_back(q);
    std::set<int> subg_q, cand_q;
    for (int v : subg)
      if (adj.at(q).count(v)) subg_q.insert(v);
    for (int v : cand)
      if (adj.at(q).count(v)) cand_q.insert(v);
    find_cliques(adj, current, subg_q, cand_q, cliques);
    current.pop_back();
  }
}

}  // namespace

std::vector<int> Slam::verify_pcm(const std::deque<ICPResult>& queue,
                                  int min_pcm_value)
{
  last_pcm_min_md = -1.0;
  last_pcm_edges = 0;
  if (static_cast<int>(queue.size()) < min_pcm_value) return {};

  // build the consistency graph; quarantined (rejected) closures neither
  // insert nor vote — a reverted alias clique must not lend consistency to
  // the next alias that joins the queue
  std::map<int, std::set<int>> adj;
  for (std::size_t a = 0; a < queue.size(); ++a) {
    if (queue[a].rejected) continue;
    for (std::size_t b = a + 1; b < queue.size(); ++b) {
      if (queue[b].rejected) continue;
      const ICPResult& ret_il = queue[a];
      const ICPResult& ret_jk = queue[b];

      const gtsam::Pose2& pi = ret_il.target_pose;
      const gtsam::Pose2& pj = ret_jk.target_pose;
      const gtsam::Pose2& pil = ret_il.estimated_transform;
      const gtsam::Pose2 plk = ret_il.source_pose.between(ret_jk.source_pose);
      const gtsam::Pose2& pjk1 = ret_jk.estimated_transform;
      const gtsam::Pose2 pjk2 = pj.between(pi.compose(pil).compose(plk));

      const Eigen::Vector3d error = gtsam::Pose2::Logmap(pjk1.between(pjk2));
      Eigen::Matrix3d cov = ret_jk.cov;
      if (!ret_jk.has_cov)
        cov = icp_odom_sigmas.array().square().matrix().asDiagonal();
      // The consistency error routes through the GRAPH's relative pose
      // between the two closures' source keyframes (plk) — its uncertainty
      // (the drift PCM exists to tolerate) must appear in the chi2
      // denominator or honest pairs on a drifted segment always fail.
      // GTSAM marginals are BODY-frame tangent covariances (the tangent chart
      // sits at the pose, Exp on the right), so rotate each keyframe's
      // marginal into the world frame FIRST, sum there, then take the sum
      // into pj's frame — the earlier code treated the body marginals as
      // world and applied a spurious unrotation. Ignoring the (helpful)
      // cross-correlation still makes this an over-estimate: permissive, so
      // PCM keeps relying on the per-closure gates for outright junk.
      const auto body_to_world = [](Eigen::Matrix3d c, const gtsam::Pose2& p) {
        const Eigen::Matrix2d Rb = p.rotation().matrix();
        c.topRows<2>() = Rb * c.topRows<2>();
        c.leftCols<2>() = c.leftCols<2>() * Rb.transpose();
        return c;
      };
      Eigen::Matrix3d rel =
        body_to_world(keyframes[ret_il.source_key]->cov, ret_il.source_pose) +
        body_to_world(keyframes[ret_jk.source_key]->cov, ret_jk.source_pose);
      const Eigen::Matrix2d R = pj.rotation().matrix();
      rel.topRows<2>() = R.transpose() * rel.topRows<2>();
      rel.leftCols<2>() = rel.leftCols<2>() * R;
      cov += rel;
      const double md = error.dot(cov.inverse() * error);
      if (last_pcm_min_md < 0.0 || md < last_pcm_min_md) last_pcm_min_md = md;
      // chi2.ppf(0.99, 3) = 11.34
      if (md < 11.34) {
        ++last_pcm_edges;
        adj[static_cast<int>(a)].insert(static_cast<int>(b));
        adj[static_cast<int>(b)].insert(static_cast<int>(a));
      }
    }
  }

  if (adj.empty()) return {};

  std::set<int> vertices;
  for (const auto& [v, _] : adj) vertices.insert(v);
  std::vector<std::vector<int>> cliques;
  std::vector<int> current;
  find_cliques(adj, current, vertices, vertices, cliques);

  if (cliques.empty()) return {};

  const auto& maximum = *std::max_element(
    cliques.begin(), cliques.end(),
    [](const auto& x, const auto& y) { return x.size() < y.size(); });
  if (static_cast<int>(maximum.size()) < min_pcm_value) return {};
  return maximum;
}

}  // namespace sonar_slam
