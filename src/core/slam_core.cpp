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

  prior_model_ = create_noise_model(prior_sigmas);
  odom_model_ = create_noise_model(odom_sigmas);
  icp_odom_model_ = create_noise_model(icp_odom_sigmas);
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

  // never report a covariance smaller than the fixed ICP model
  const Eigen::Matrix3d default_cov =
    icp_odom_sigmas.array().square().matrix().asDiagonal();
  if (cov.determinant() < default_cov.determinant()) cov = default_cov;

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
  graph_.add(gtsam::PriorFactor<gtsam::Pose2>(X(0), keyframe->pose, prior_model_));
  values_.insert(X(0), keyframe->pose);
}

void Slam::add_odometry(const KeyframePtr& keyframe)
{
  const gtsam::Pose2 dr_odom = keyframes.back()->pose.between(keyframe->pose);
  graph_.add(gtsam::BetweenFactor<gtsam::Pose2>(
    X(current_key() - 1), X(current_key()), dr_odom, odom_model_));
  values_.insert(X(current_key()), keyframe->pose);
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

  ret.source_points = keyframe->points;
  std::vector<int> target_frames;
  for (int k = std::max(0, current_key() - ssm_params.target_frames);
       k < current_key(); ++k)
    target_frames.push_back(k);
  ret.target_points = get_points(target_frames, ret.target_key);
  ret.cov = odom_sigmas.asDiagonal();

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
      // never report a covariance smaller than the fixed ICP model
      Eigen::Matrix3d cov = c.cov;
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
    if (delta_translation > ssm_params.max_translation ||
        delta_rotation > ssm_params.max_rotation) {
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
    ret2.status.description = "overlap " + std::to_string(overlap);
  }

  if (ret2.status) {
    const gtsam::SharedNoiseModel model =
      ret2.has_cov ? create_full_noise_model(ret2.cov) : icp_odom_model_;
    graph_.add(gtsam::BetweenFactor<gtsam::Pose2>(
      X(ret2.target_key), X(ret2.source_key), ret2.estimated_transform, model));
    values_.insert(X(ret2.source_key),
                   ret2.target_pose.compose(ret2.estimated_transform));
    ret2.inserted = true;
    ++ssm_accepted;
  } else {
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
  // anchor the search on the just-added keyframe (keyframes[source_key]) whose
  // frame the source_points live in — NOT current_frame, which the node only
  // assigns after this runs, so it still points at the previous callback's
  // frame and would offset the global-init search by up to a keyframe step
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
  // padded by the pose uncertainty
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
  for (const auto& [frame, count] : counts) {
    if (count > 10) {
      any = true;
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

  // bounds from the source pose uncertainty (uses the last source frame's cov,
  // like slam.py's loop variable leak)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov.topLeftCorner<2, 2>());
  const double translation_std = std::sqrt(es.eigenvalues().maxCoeff());
  const double rotation_std = std::sqrt(cov(2, 2));
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
  ret.target_points = get_points(target_frames, ret.target_key);

  return ret;
}

bool Slam::add_nonsequential_scan_matching()
{
  // current_frame is assigned at the END of the SLAM callback, so it is null
  // until the second callback
  if (!current_frame || current_key() < nssm_params.min_st_sep) return false;

  InitializationResult ret = initialize_nonsequential_scan_matching();
  if (!ret.status) return false;

  ICPResult ret2(ret, nssm_params.cov_samples > 0 &&
                        nssm_params.cov_method == SMParams::SAMPLED);
  run_scan_match_icp(ret2, nssm_params);

  if (ret2.status) {
    const gtsam::Pose2 delta =
      ret2.initial_transform.between(ret2.estimated_transform);
    const double delta_translation =
      std::hypot(delta.translation().x(), delta.translation().y());
    const double delta_rotation = std::abs(delta.theta());
    if (delta_translation > nssm_params.max_translation ||
        delta_rotation > nssm_params.max_rotation) {
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
    ret2.status.description = std::to_string(overlap);
  }

  bool accepted = false;
  if (ret2.status) {
    // slide the PCM queue window
    while (!nssm_queue_.empty() &&
           ret2.source_key - nssm_queue_.front().source_key > pcm_queue_size)
      nssm_queue_.pop_front();

    nssm_queue_.push_back(ret2);
    const std::vector<int> pcm = verify_pcm(nssm_queue_, min_pcm);

    for (int m : pcm) {
      ICPResult& loop = nssm_queue_[m];
      if (loop.inserted) continue;

      const gtsam::SharedNoiseModel model =
        loop.has_cov ? create_full_noise_model(loop.cov) : icp_odom_model_;
      graph_.add(gtsam::BetweenFactor<gtsam::Pose2>(
        X(loop.target_key), X(loop.source_key), loop.estimated_transform, model));
      keyframes[loop.source_key]->constraints.emplace_back(
        loop.target_key, loop.estimated_transform);
      loop.inserted = true;
      ++nssm_accepted;
      accepted = true;
    }
  }
  return accepted;
}

// ---------------------------------------------------------------------------
// graph update
// ---------------------------------------------------------------------------
void Slam::update_factor_graph(const KeyframePtr& keyframe)
{
  if (keyframe) keyframes.push_back(keyframe);

  isam_.update(graph_, values_);
  graph_.resize(0);
  values_.clear();

  const gtsam::Values values = isam_.calculateEstimate();
  gtsam::Pose2 pose;
  for (std::size_t x = 0; x < values.size(); ++x) {
    pose = values.at<gtsam::Pose2>(X(static_cast<int>(x)));
    keyframes.at(x)->update(pose);  // .at() fails loudly if the counts diverge
  }

  // only the latest covariance is updated (like slam.py). marginalCovariance
  // can throw on a weakly-constrained variable (IndeterminantLinearSystem);
  // keep the optimized pose and skip the covariance update rather than
  // propagating the failure out of the callback.
  try {
    const Eigen::Matrix3d cov =
      isam_.marginalCovariance(X(static_cast<int>(values.size()) - 1));
    keyframes.back()->update(pose, cov);
  } catch (const std::exception&) {
    keyframes.back()->update(pose);
  }

  // refresh the poses in pending loop closures for PCM
  for (ICPResult& ret : nssm_queue_) {
    ret.source_pose = keyframes[ret.source_key]->pose;
    ret.target_pose = keyframes[ret.target_key]->pose;
    if (ret.inserted)
      ret.estimated_transform = ret.target_pose.between(ret.source_pose);
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
                                  int min_pcm_value) const
{
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
      const double md = error.dot(cov.inverse() * error);
      // chi2.ppf(0.99, 3) = 11.34
      if (md < 11.34) {
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
