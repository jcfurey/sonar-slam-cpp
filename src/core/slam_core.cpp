#include "sonar_slam_cpp/slam_core.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/global_init.hpp"
#include "sonar_slam_cpp/icp_covariance.hpp"
#include "sonar_slam_cpp/mcd.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace sonar_slam {

namespace {

inline gtsam::Key X(int i) { return gtsam::Symbol('x', i); }

// SE3 graph chart (FULL_3D_ROADMAP.md Phase 4): states are HORIZON poses
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

Slam::Slam()
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
}

void Slam::configure()
{
  // config sanity (slam.py used asserts; these must also fire in NDEBUG
  // builds — the default build type disables assert())
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

gtsam::SharedNoiseModel Slam::create_noise_model(const Eigen::Vector3d& sigmas) const
{
  return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
}

gtsam::SharedNoiseModel Slam::create_full_noise_model(const Eigen::Matrix3d& cov) const
{
  return gtsam::noiseModel::Gaussian::Covariance(cov);
}

// ---------------------------------------------------------------------------
// point accumulation
// ---------------------------------------------------------------------------
Matrix Slam::get_points(const std::vector<int>& frames, int ref_key) const
{
  const bool use_ref = ref_key >= 0;
  gtsam::Pose2 ref_pose;
  if (use_ref) ref_pose = keyframes[ref_key]->pose;

  long total = 0;
  for (int key : frames) total += keyframes[key]->points.rows();
  Matrix all_points(total, 2);

  long at = 0;
  for (int key : frames) {
    Matrix transf;
    if (use_ref) {
      const gtsam::Pose2 transf_pose = ref_pose.between(keyframes[key]->pose);
      transf = Keyframe::transform_points(keyframes[key]->points, transf_pose);
    } else {
      transf = keyframes[key]->transf_points;
    }
    all_points.middleRows(at, transf.rows()) = transf;
    at += transf.rows();
  }

  return downsample(all_points, static_cast<float>(point_resolution));
}

std::pair<Matrix, std::vector<int>> Slam::get_points_with_keys(
  const std::vector<int>& frames) const
{
  long total = 0;
  for (int key : frames) total += keyframes[key]->points.rows();
  Matrix all_points(total, 2);
  Matrix all_keys(total, 1);

  long at = 0;
  for (int key : frames) {
    const Matrix& transf = keyframes[key]->transf_points;
    all_points.middleRows(at, transf.rows()) = transf;
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
    const auto batch = icp.compute_batch(source_points, target_points, gm, 2000);
    for (const auto& r : batch) {
      if (!r.success) continue;
      sample_transforms.emplace_back(r.T(0, 2), r.T(1, 2),
                                     std::atan2(r.T(1, 0), r.T(0, 0)));
    }
  } else {
    const auto start = std::chrono::steady_clock::now();
    for (const auto& g : guesses) {
      Eigen::Matrix3f gm = g.matrix().cast<float>();
      auto [message, T] = icp.compute(source_points, target_points, gm);
      if (message == "success") {
        sample_transforms.emplace_back(T(0, 2), T(1, 2),
                                       std::atan2(T(1, 0), T(0, 0)));
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
  const Eigen::Vector3d floor_var = icp_odom_sigmas.array().square().matrix();
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov.topLeftCorner<2, 2>());
    const double vmin = std::min(floor_var[0], floor_var[1]);
    const Eigen::Vector2d ev = es.eigenvalues().cwiseMax(vmin);
    cov.topLeftCorner<2, 2>() =
      es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    cov(2, 2) = std::max(cov(2, 2), floor_var[2]);
  }

  result.message = "success";
  result.odom = m;
  result.cov = cov;
  result.n_samples = static_cast<int>(sample_transforms.size());
  return result;
}

int Slam::get_overlap(const Matrix& source_points, const Matrix& target_points,
                      const gtsam::Pose2* source_pose,
                      std::vector<int>* indices_out) const
{
  Matrix src = source_points;
  if (source_pose) src = Keyframe::transform_points(src, *source_pose);

  auto [ids, dists] = match(target_points.leftCols(2), src.leftCols(2), 1,
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
  if (duration < keyframe_duration) return false;

  const gtsam::Pose2 dr_odom = keyframes.back()->dr_pose.between(frame.dr_pose);
  const double translation =
    std::hypot(dr_odom.translation().x(), dr_odom.translation().y());
  const double rotation = std::abs(dr_odom.theta());
  return translation > keyframe_translation || rotation > keyframe_rotation;
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
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
    X(current_key() - 1), X(current_key()), dr_odom, odom_model_));
  values_.insert(X(current_key()), keyframe->horizon_pose3());
  add_state_unary(current_key(), keyframe);
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

  // registration is planar: keep the x/y projection, drop the elevation col
  ret.source_points = keyframe->points.leftCols(2);
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
  if (ret2.status) {
    const gtsam::Pose2 delta =
      ret2.initial_transform.between(ret2.estimated_transform);
    const double delta_translation =
      std::hypot(delta.translation().x(), delta.translation().y());
    const double delta_rotation = std::abs(delta.theta());
    // inverted comparisons so a NaN delta (degenerate transform) fails closed
    if (!(delta_translation <= ssm_params.max_translation) ||
        !(delta_rotation <= ssm_params.max_rotation)) {
      ret2.status = Status(Status::LARGE_TRANSFORMATION);
      ret2.status.description = "trans " + std::to_string(delta_translation) +
                                " rot " + std::to_string(delta_rotation);
    }
  }

  // there must be enough overlap between the two point clouds
  if (ret2.status) {
    const int overlap = get_overlap(ret2.source_points, ret2.target_points,
                                    &ret2.estimated_transform);
    if (overlap < ssm_params.min_points)
      ret2.status = Status(Status::NOT_ENOUGH_OVERLAP);
    // append — don't clobber the success path's "N samples" diagnostic
    if (!ret2.status.description.empty()) ret2.status.description += ", ";
    ret2.status.description += "overlap " + std::to_string(overlap);
  }

  if (ret2.status) {
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
  // bruce_slam that fixes that latent bug (see docs/DIVERGENCES.md).
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
  // (carried from slam.py; see docs/DIVERGENCES.md), so the older source
  // frames' covariances here are stale. This only widens/narrows the fan
  // PADDING for candidate pre-selection, a second-order effect, so it is left
  // as-is; the high-impact consumer (the global-init search bounds below) now
  // uses the freshly refreshed anchor-keyframe covariance.
  std::vector<char> sel(target_points_all.rows(), 0);
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (int source_frame : source_frames) {
    const gtsam::Pose2& pose = keyframes[source_frame]->pose;
    cov = keyframes[source_frame]->cov;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov.topLeftCorner<2, 2>());
    const double translation_std = std::sqrt(es.eigenvalues().maxCoeff());
    const double rotation_std = std::sqrt(cov(2, 2));
    const double range_bound = translation_std * 5.0 + oculus.max_range;
    const double bearing_bound =
      rotation_std * 5.0 + oculus.horizontal_aperture * 0.5;

    const Matrix local_points =
      Keyframe::transform_points(target_points_all, pose.inverse());
    for (int i = 0; i < local_points.rows(); ++i) {
      const double range = std::hypot(local_points(i, 0), local_points(i, 1));
      const double bearing = std::atan2(local_points(i, 1), local_points(i, 0));
      if (range < range_bound && std::abs(bearing) < bearing_bound) sel[i] = 1;
    }
  }

  long n_sel = 0;
  for (char s : sel) n_sel += s;
  Matrix target_points(n_sel, 2);
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
  int best_frame = -1, best_count = 0;
  bool any = false;
  // the keyframes that survive the fan/covariance gate (>10 nearby points):
  // this is the LOCAL overlapping submap, not the whole map. Used below to
  // build the final target cloud, matching slam.py's reassignment of
  // target_frames = target_frames[counts > 10] (std::map keeps ascending order,
  // matching np.unique). Without this the loop-closure ICP/overlap/covariance
  // would register against the entire explored trajectory.
  std::vector<int> candidate_frames;
  for (const auto& [frame, count] : counts) {
    if (count > 10) {
      any = true;
      candidate_frames.push_back(frame);
      if (count > best_count) { best_count = count; best_frame = frame; }
    }
  }

  if (!any || target_points.rows() < nssm_params.min_points) {
    ret.status = Status(Status::NOT_ENOUGH_POINTS);
    ret.status.description =
      "target points " + std::to_string(target_points.rows());
    return ret;
  }

  ret.target_key = best_frame;
  ret.target_pose = keyframes[ret.target_key]->pose;
  ret.target_points =
    Keyframe::transform_points(target_points, ret.target_pose.inverse());
  ret.cov = keyframes[ret.source_key]->cov;

  if (!nssm_params.initialization) return ret;

  // Bounds from the source-pose uncertainty. slam.py reused `cov` leaked from
  // the fan-selection loop above — the OLDEST source frame's covariance, a
  // loop-variable-leak bug. Use the anchor keyframe's covariance instead
  // (ret.cov == keyframes[source_key], the newest source frame), so the search
  // bounds match ret.source_pose; its marginal was just refreshed by
  // update_factor_graph, so it is current. (Intentional divergence from
  // bruce_slam — see docs/DIVERGENCES.md.)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(ret.cov.topLeftCorner<2, 2>());
  const double translation_std = std::sqrt(es.eigenvalues().maxCoeff());
  const double rotation_std = std::sqrt(ret.cov(2, 2));
  Eigen::Matrix<double, 3, 2> bounds;
  bounds << -5.0 * translation_std, 5.0 * translation_std,
            -5.0 * translation_std, 5.0 * translation_std,
            -5.0 * rotation_std, 5.0 * rotation_std;

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
  const Matrix estimated_source_points =
    Keyframe::transform_points(ret.source_points, ret.estimated_source_pose);
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

  return ret;
}

bool Slam::add_nonsequential_scan_matching()
{
  // current_frame is assigned at the END of the SLAM callback, so it is null
  // until the second callback
  if (!current_frame || current_key() < nssm_params.min_st_sep) return false;

  ++nssm_attempts;
  InitializationResult ret = initialize_nonsequential_scan_matching();
  if (!ret.status) {
    last_nssm_status =
      std::string(ret.status.name()) + " (" + ret.status.description + ")";
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
    if (overlap < nssm_params.min_points)
      ret2.status = Status(Status::NOT_ENOUGH_OVERLAP);
    // append — don't clobber the success path's "N samples" diagnostic
    if (!ret2.status.description.empty()) ret2.status.description += ", ";
    ret2.status.description += "overlap " + std::to_string(overlap);
  }

  // Compass-consistency gate (ABSOLUTE, added 2026-07-15 night after two
  // accepted closures rotated the map 90°+): discrete rotational aliasing —
  // a near-square pool aliases at ~90° — produces CONFIDENT wrong locks
  // (tight isotropic covariance, tiny ICP refinement) and symmetric pairs
  // that agree with each other, so every relative gate below passes. The
  // compass cannot be aliased: both keyframes' DR yaws are compass-anchored
  // and drift-free, so the closure's measured relative yaw must match the
  // DR relative yaw within compass noise.
  if (ret2.status) {
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

  // Degeneracy gate (FULL_3D_ROADMAP.md "Path to survey-grade" item 1):
  // pool/wall aliasing produced self-consistent-but-wrong closures that PCM
  // could not reject (2026-07 CHL_Pool: 9 accepted closures demanding 21 deg
  // RMS yaw correction — why NSSM was turned off). Sliding along featureless
  // structure shows up as an elongated translation covariance in the
  // sampled/Censi estimate, so reject closures whose covariance is too large
  // or too anisotropic. The yaw half of the gate is nssm/max_rotation,
  // tightened in slam.yaml to the compass-anchored bound.
  if (ret2.status && ret2.has_cov) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(
      ret2.cov.topLeftCorner<2, 2>());
    const double ev_lo = es.eigenvalues()[0];
    const double ev_hi = es.eigenvalues()[1];
    const double sig_max = std::sqrt(std::max(0.0, ev_hi));
    const double sig_min = std::sqrt(std::max(1e-12, ev_lo));
    // check the RAW eigenvalues for NaN — std::max launders NaN to its other
    // argument, so a NaN covariance would otherwise sail through as sigma 0
    if (!std::isfinite(ev_lo) || !std::isfinite(ev_hi) ||
        sig_max > nssm_max_sigma || sig_max / sig_min > nssm_max_anisotropy) {
      ret2.status = Status(Status::NOT_CONVERGED);
      ret2.status.description = "degenerate: sigma " + std::to_string(sig_max) +
                                " aniso " + std::to_string(sig_max / sig_min);
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
      if (loop.inserted) continue;

      // Robust kernel on loop factors (rtabmap/Vertigo-style damage
      // limitation): DCS (Agarwal et al., dynamic covariance scaling)
      // down-weights any closure whose residual grows during optimization —
      // a bad closure that fools every gate loses the argument with the
      // odometry chain instead of winning it.
      const gtsam::SharedNoiseModel base =
        loop.has_cov
          ? gtsam::noiseModel::Gaussian::Covariance(embed_planar_cov(loop.cov))
          : icp_odom_model_;
      const gtsam::SharedNoiseModel model = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::DCS::Create(1.0), base);
      graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        X(loop.target_key), X(loop.source_key),
        lift_planar(loop.estimated_transform), model));
      keyframes[loop.source_key]->constraints.emplace_back(
        loop.target_key, loop.estimated_transform);
      loop.inserted = true;
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
  }
  return accepted;
}

// ---------------------------------------------------------------------------
// graph update
// ---------------------------------------------------------------------------
bool Slam::update_factor_graph(const KeyframePtr& keyframe)
{
  if (keyframe) keyframes.push_back(keyframe);

  // |optimized - DR| separation for one consecutive keyframe pair. DR is
  // drift-free over a single ~0.75 m keyframe interval, so a deviation here is
  // the optimizer having torn the chain to satisfy a loop closure.
  const auto link_tear = [this](std::size_t k) {
    const double d_opt = (keyframes[k + 1]->pose.translation() -
                          keyframes[k]->pose.translation())
                           .norm();
    const double d_dr = (keyframes[k + 1]->dr_pose.translation() -
                         keyframes[k]->dr_pose.translation())
                          .norm();
    return std::fabs(d_opt - d_dr);
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
    if (post_loop_max_translation_err > 0.0 && keyframes.size() > 1) {
      pre_tear.resize(keyframes.size() - 1);
      for (std::size_t k = 0; k + 1 < keyframes.size(); ++k)
        pre_tear[k] = link_tear(k);
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
    if (keyframe) keyframes.pop_back();
    rollback_pending_loops();
    rebuild_isam();
    return false;
  }
  const gtsam::Values values = isam_.calculateEstimate();
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

  // rtabmap-style optimize-then-verify (RGBD/OptimizeMaxError analog, using
  // the reference rtabmap doesn't have — an absolute compass): when this
  // round inserted loop closures, check the WHOLE graph's optimized yaws
  // against the compass-anchored DR yaws. A closure that fooled every
  // per-closure gate (confident rotational alias) still has to bend the
  // trajectory away from the compass to win — the historical "21 deg RMS
  // yaw correction" diagnostic, automated. On failure the loops are rolled
  // back and the estimator rebuilt without them.
  if (!pending_loops_.empty()) {
    double ss = 0.0;
    for (const auto& kf : keyframes) {
      const double e =
        std::remainder(kf->pose.theta() - kf->dr_pose.theta(), 2.0 * M_PI);
      ss += e * e;
    }
    const double rms = std::sqrt(ss / static_cast<double>(keyframes.size()));
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
    if (post_loop_max_translation_err > 0.0) {
      for (std::size_t k = 0; k + 1 < keyframes.size(); ++k) {
        const double tear = link_tear(k);
        const double before = k < pre_tear.size() ? pre_tear[k] : 0.0;
        // covers re-solve jitter on a link this round never touched (~5 mm
        // observed), and sits orders below any real fold — a numerical guard,
        // not a tuning lever, so it stays out of the config
        constexpr double tear_jitter_eps = 0.01;  // m
        if (tear > post_loop_max_translation_err &&
            tear > before + tear_jitter_eps) {
          tear_bad = true;
          worst_tear = std::max(worst_tear, tear);
        }
      }
    }
    const bool yaw_bad = rms > post_loop_max_yaw_rms;
    if (yaw_bad || tear_bad) {
      last_error_ = yaw_bad
        ? ("post-loop compass check failed: optimized-vs-DR yaw RMS " +
           std::to_string(rms) + " rad > " +
           std::to_string(post_loop_max_yaw_rms))
        : ("post-loop chain-tear check failed: this round tore a consecutive "
           "keyframe link to " + std::to_string(worst_tear) +
           " m from DR > " + std::to_string(post_loop_max_translation_err));
      last_nssm_status = "REVERTED (" + last_error_ + ")";
      graph_.resize(0);
      values_.clear();
      rollback_pending_loops();
      // Restore the pre-round poses EXACTLY, then rebuild from them. They are
      // the converged optimum of committed_graph_ — this round appends to it
      // only on success (below), so nothing it constrains has changed — which
      // means the rebuilt estimator relinearizes AT the optimum and holds
      // them. Seeding the rebuild from the bent poses instead (what the sweep
      // above wrote into the keyframes) left a residual bend that a single
      // isam_.update could not walk back, and that residual is itself a chain
      // tear: the delta check above would read it as pre-existing and excuse
      // it from then on.
      const std::size_t n = std::min(pre_round_poses.size(), keyframes.size());
#pragma omp parallel for schedule(dynamic)
      for (std::size_t x = 0; x < n; ++x)
        keyframes[x]->update(pre_round_poses[x]);
      rebuild_isam();
      return false;
    }
  }

  // verified (or no loops this round): the factors become permanent; mirror
  // them for failure recovery, and let this round's loop-closure marks stand
  committed_graph_.push_back(graph_);
  graph_.resize(0);
  values_.clear();
  pending_loops_.clear();

  // Only the latest keyframe's covariance is refreshed — a known limitation
  // carried from slam.py (its own "TODO propagate cov from previous keyframe"),
  // so older keyframes keep stale marginals. Recomputing every keyframe's
  // marginal each step is O(n^2) and would starve the callback; deferred (see
  // docs/DIVERGENCES.md). marginalCovariance can throw on a weakly-constrained
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
    keyframes.back()->update(pose, cov);
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
        keyframes[k]->update(keyframes[k]->horizon_pose3(), c);
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
}

void Slam::rollback_pending_loops()
{
  // Loops are marked inserted (and appended to their keyframe's constraints)
  // in add_nonsequential_scan_matching, BEFORE the update that actually
  // incorporates their factors. If that update failed, the factors were
  // discarded — un-mark them so PCM can re-insert on a later round instead of
  // carrying phantom constraints forever.
  for (int m : pending_loops_) {
    if (m < 0 || m >= static_cast<int>(nssm_queue_.size())) continue;
    ICPResult& loop = nssm_queue_[m];
    if (!loop.inserted) continue;
    loop.inserted = false;
    if (loop.source_key >= 0 &&
        loop.source_key < static_cast<int>(keyframes.size())) {
      auto& constraints = keyframes[loop.source_key]->constraints;
      if (!constraints.empty()) constraints.pop_back();
    }
    --nssm_accepted;
  }
  pending_loops_.clear();
}

void Slam::rebuild_isam()
{
  // Reconstruct the estimator from the committed factor mirror, linearizing at
  // the last optimized keyframe poses. These factors solved together before,
  // so this update is expected to succeed; if it somehow does not, keep the
  // fresh estimator and report — the next frame retries against it.
  isam_ = gtsam::ISAM2();
  gtsam::Values estimates;
  for (std::size_t i = 0; i < keyframes.size(); ++i)
    estimates.insert(X(static_cast<int>(i)), keyframes[i]->horizon_pose3());
  try {
    gtsam::NonlinearFactorGraph graph = committed_graph_;
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

  // build the consistency graph
  std::map<int, std::set<int>> adj;
  for (std::size_t a = 0; a < queue.size(); ++a)
    for (std::size_t b = a + 1; b < queue.size(); ++b) {
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
      // Approximate the relative covariance as the sum of the (freshly
      // refreshed) world marginals, unrotated into pj's frame — the same
      // world->local pattern as compute_icp_with_cov. Ignoring the (helpful)
      // cross-correlation makes this an over-estimate: permissive, so PCM
      // still relies on the per-closure gates for outright junk.
      Eigen::Matrix3d rel = keyframes[ret_il.source_key]->cov +
                            keyframes[ret_jk.source_key]->cov;
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
