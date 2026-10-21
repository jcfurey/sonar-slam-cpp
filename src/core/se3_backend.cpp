#include "sonar_slam_cpp/se3_backend.hpp"
#include "sonar_slam_cpp/slam_core.hpp"
#include "sonar_slam_cpp/common.hpp"
#include "sonar_slam_cpp/global_init.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/config.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <cstring>
#include <unordered_set>
#include <set>

namespace sonar_slam {
namespace {
#if GTSAM_VERSION_NUMERIC >= 40300
using FactorJacobian = gtsam::OptionalMatrixType;
#else
using FactorJacobian = boost::optional<gtsam::Matrix&>;
#endif
gtsam::Key key(int index) { return gtsam::Symbol('x', index); }
double angle(const gtsam::Pose3& pose) { return gtsam::Rot3::Logmap(pose.rotation()).norm(); }
Covariance6 diagonal(const gtsam::Vector6& sigmas) { return sigmas.array().square().matrix().asDiagonal(); }
bool distinct_hypotheses(const RegistrationResult& a, const RegistrationResult& b)
{
  const auto difference = a.transform.between(b.transform);
  Covariance6 jacobian;
  const auto residual = gtsam::Pose3::Logmap(difference, jacobian);
  const Covariance6 covariance = jacobian * compose_covariance(difference, a.covariance, b.covariance) * jacobian.transpose();
  const Eigen::LDLT<Covariance6> inverse(covariance);
  return inverse.info() != Eigen::Success || !inverse.isPositive() ||
    residual.dot(inverse.solve(residual)) > 16.811893829770927;
}
void cap_points(Matrix& points, int maximum)
{
  if (points.rows() <= maximum) return;
  Matrix retained(maximum, 3);
  for (int i = 0; i < maximum; ++i)
    retained.row(i) = points.row(static_cast<Eigen::Index>(i) * points.rows() / maximum);
  points = std::move(retained);
}

class AbsolutePlanarFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  AbsolutePlanarFactor(gtsam::Key k, const Eigen::Vector3d& measurement,
                      const gtsam::SharedNoiseModel& noise, bool position_only,
                      const gtsam::Point3& point = gtsam::Point3::Zero())
    : NoiseModelFactor1(noise, k), measurement_(measurement), point_(point), position_only_(position_only) {}
  gtsam::Vector evaluateError(const gtsam::Pose3& pose,
      FactorJacobian H = {}) const override
  {
    const std::function<gtsam::Vector(const gtsam::Pose3&)> error = [this](const auto& value) {
      gtsam::Vector residual(position_only_ ? 2 : 3);
      const gtsam::Point3 position = position_only_ ? value.transformFrom(point_) : value.translation();
      residual[0] = position.x() - measurement_.x();
      residual[1] = position.y() - measurement_.y();
      if (!position_only_) residual[2] = std::remainder(value.rotation().yaw() - measurement_.z(), 2.0 * M_PI);
      return residual;
    };
    if (H) *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(error, pose);
    return error(pose);
  }
private:
  Eigen::Vector3d measurement_;
  gtsam::Point3 point_;
  bool position_only_;
};

gtsam::ISAM2Params engine_params()
{
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = .01;
  params.relinearizeSkip = 1;
  return params;
}
}  // namespace

struct Se3Backend::Impl
{
  enum class Kind : std::uint8_t { Prior, Between, Planar, Position };
  struct Record
  {
    Kind kind = Kind::Prior;
    int a = 0, b = 0;
    gtsam::Pose3 pose;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Covariance6 covariance = Covariance6::Identity();
    bool manual = false, loop = false, active = true;
    double dcs_phi = 0.0;
  };
  struct Link
  {
    bool valid = false;
    gtsam::Pose3 measured;
    Covariance6 covariance = Covariance6::Identity();
    Covariance6 odometry_covariance = Covariance6::Identity();
    double accepted_absolute_tear = 0.0;
  };
  struct Loop
  {
    int target = 0, source = 0;
    gtsam::Pose3 transform;
    Covariance6 covariance = Covariance6::Identity();
    bool inserted = false;
  };
  Slam& s;
  gtsam::ISAM2 engine{engine_params()};
  std::vector<Record> records, staged;
  gtsam::Values staged_values;
  KeyframePtr staged_frame;
  Link staged_link;
  bool staged_acoustic = false;
  std::vector<Link> links;
  std::deque<Loop> loops;
  std::set<std::pair<int, int>> quarantined;
  std::vector<std::size_t> manual_stack;
  std::optional<gtsam::Pose3> confirmation;
  double confirmation_stamp = 0.0;
  int confirmations = 0;

  explicit Impl(Slam& owner) : s(owner)
  {
    s.se3_options.icp.validate();
    for (const auto& sigmas : {s.se3_options.prior_sigmas, s.se3_options.odom_sigmas})
      if (!sigmas.allFinite() || (sigmas.array() <= 0).any())
        throw std::invalid_argument("SE3 prior/odometry sigmas need six finite positive values");
    if (!std::isfinite(s.se3_options.max_rotation_sigma) || s.se3_options.max_rotation_sigma <= 0 ||
        !std::isfinite(s.se3_options.relocalize_depth_range) || s.se3_options.relocalize_depth_range < 0 ||
        !std::isfinite(s.se3_options.relocalize_attitude_range) || s.se3_options.relocalize_attitude_range < 0 ||
        s.relocalize_min_confirmations < 2 || s.relocalize_min_overlap <= 0 ||
        s.relocalize_min_overlap > 1 || !std::isfinite(s.relocalize_min_overlap) ||
        s.pcm_queue_size < 2 || s.pcm_queue_size > 16 || s.min_pcm < 2 || s.min_pcm > s.pcm_queue_size)
      throw std::invalid_argument("invalid SE3 covariance, relocalization or PCM settings");
    for (double value : {s.keyframe_duration, s.keyframe_translation, s.keyframe_rotation,
        s.keyframe_head_rotation, s.sonar_max_range, s.post_loop_max_link_error_sigma})
      if (!std::isfinite(value) || value < 0) throw std::invalid_argument("SE3 keyframe/range/verification settings must be finite and nonnegative");
    for (const auto& params : {s.ssm_params, s.nssm_params})
      if (params.min_points < 6 || params.source_frames < 1 || params.target_frames < 1 ||
          params.min_revisit_sep < 1 || !std::isfinite(params.max_translation) || params.max_translation <= 0 ||
          !std::isfinite(params.max_rotation) || params.max_rotation <= 0 ||
          !std::isfinite(params.min_overlap_ratio) || params.min_overlap_ratio < 0 || params.min_overlap_ratio > 1)
        throw std::invalid_argument("invalid SE3 scan matching limits");
    if (s.se3_options.max_source_points < std::max({s.ssm_params.min_points, s.nssm_params.min_points,
          s.se3_options.icp.min_correspondences, s.se3_options.icp.normal_neighbors}) ||
        s.se3_options.max_source_points > 1000000 || s.se3_options.max_seed_points < 100 ||
        s.se3_options.max_seed_points > 10000000 || s.nssm_params.source_frames > 16 || s.ssm_params.target_frames > 16 ||
        s.nssm_params.source_frames >= s.nssm_params.min_st_sep)
      throw std::invalid_argument("SE3 source/seed limits must cover admission; source/target windows are at most 16 and source_frames < min_st_sep");
  }

  gtsam::NonlinearFactor::shared_ptr factor(const Record& record) const
  {
    const int dimensions = record.kind == Kind::Position ? 2 : record.kind == Kind::Planar ? 3 : 6;
    gtsam::SharedNoiseModel noise = gtsam::noiseModel::Gaussian::Covariance(
      record.covariance.topLeftCorner(dimensions, dimensions));
    if (record.dcs_phi > 0)
      noise = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::DCS::Create(record.dcs_phi), noise);
    if (record.kind == Kind::Prior)
      return gtsam::NonlinearFactor::shared_ptr(new gtsam::PriorFactor<gtsam::Pose3>(key(record.a), record.pose, noise));
    if (record.kind == Kind::Between)
      return gtsam::NonlinearFactor::shared_ptr(new gtsam::BetweenFactor<gtsam::Pose3>(key(record.a), key(record.b), record.pose, noise));
    return gtsam::NonlinearFactor::shared_ptr(new AbsolutePlanarFactor(key(record.a), record.position, noise,
      record.kind == Kind::Position, record.pose.translation()));
  }

  gtsam::Values committed_values() const
  {
    gtsam::Values values;
    for (int k = 0; k < s.current_key(); ++k) values.insert(key(k), s.keyframes[k]->pose3);
    return values;
  }
  void rebuild()
  {
    engine = gtsam::ISAM2(engine_params());
    gtsam::NonlinearFactorGraph graph;
    for (const auto& record : records) if (record.active) graph.push_back(factor(record));
    if (graph.empty()) return;
    engine.update(graph, committed_values());
    for (int i = 0; i < 5; ++i) engine.update();
  }
  void clear_staged()
  {
    staged.clear(); staged_values.clear(); staged_frame.reset();
    staged_link = Link{}; staged_acoustic = false;
  }
  void apply(const gtsam::Values& estimate)
  {
    for (int k = 0; k < s.current_key(); ++k) {
      const auto& pose = estimate.at<gtsam::Pose3>(key(k));
      if (!pose.matrix().allFinite()) throw std::runtime_error("SE3 graph produced a non-finite pose");
      if (!s.keyframes[k]->pose3.equals(pose, 1e-12) || s.keyframes[k]->transf_points.rows() != s.keyframes[k]->points.rows())
        s.keyframes[k]->update(pose);
    }
    if (!s.keyframes.empty()) {
      try { s.keyframes.back()->set_covariance6(engine.marginalCovariance(key(s.current_key() - 1))); }
      catch (const std::exception&) { s.keyframes.back()->invalidate_covariance(); }
    }
  }

  Covariance6 odom_covariance(const gtsam::Pose3& increment) const
  {
    gtsam::Vector6 sigmas = s.se3_options.odom_sigmas;
    sigmas.head<3>() *= std::max(1.0, angle(increment) / std::max(.001, s.keyframe_rotation));
    sigmas.tail<3>() *= std::max(1.0, increment.translation().norm() / std::max(.001, s.keyframe_translation));
    return diagonal(sigmas);
  }
  void prepare(const KeyframePtr& frame, bool prior)
  {
    if (s.has_pending_update()) throw std::logic_error("SE3 update must commit or be discarded before staging another frame");
    s.validate_new_frame(frame, prior);
    staged_frame = frame;
    cap_points(frame->points, s.se3_options.max_source_points);
    const int k = s.current_key();
    Record record;
    record.a = prior ? k : k - 1;
    record.b = k;
    if (prior) {
      record.kind = Kind::Prior; record.pose = frame->dr_pose3;
      record.covariance = diagonal(s.se3_options.prior_sigmas);
      frame->update(frame->dr_pose3);
    } else {
      record.kind = Kind::Between;
      record.pose = s.keyframes.back()->dr_pose3.between(frame->dr_pose3);
      record.covariance = odom_covariance(record.pose);
      frame->update(s.keyframes.back()->pose3.compose(record.pose));
      staged_link = {true, record.pose, record.covariance, record.covariance};
    }
    staged.push_back(record);
    staged_values.insert(key(k), frame->pose3);
    staged_frame = frame;
  }
  bool acceptable(const RegistrationResult& fit, const SMParams& params, double max_sigma) const
  {
    if (!fit.converged || fit.observed_rank != 6 || !valid_covariance6(fit.covariance) ||
        fit.correspondences < params.min_points || fit.overlap < params.min_overlap_ratio) return false;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> translation(fit.covariance.bottomRightCorner<3, 3>());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rotation(fit.covariance.topLeftCorner<3, 3>());
    return translation.info() == Eigen::Success && rotation.info() == Eigen::Success &&
      (max_sigma <= 0 || translation.eigenvalues().maxCoeff() <= max_sigma * max_sigma) &&
      rotation.eigenvalues().maxCoeff() <= std::pow(s.se3_options.max_rotation_sigma, 2);
  }

  RegistrationResult register_cloud(const Matrix& source, const Matrix& target,
      const gtsam::Pose3& predicted, const SMParams& params, bool global)
  {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    const auto timeout = [&]() {
      RegistrationResult result;
      result.transform = predicted; result.timed_out = true;
      result.reason = "SE3 registration deadline exceeded; incomplete hypothesis search rejected";
      return result;
    };
    if (target.rows() < s.se3_options.icp.normal_neighbors) {
      RegistrationResult result; result.reason = "SE3 target too sparse for surface normals"; return result;
    }
    std::unique_ptr<RigidRegistration> matcher;
    try { matcher = std::make_unique<RigidRegistration>(target, s.se3_options.icp, deadline); }
    catch (const DeadlineExceeded&) { return timeout(); }
    std::vector<gtsam::Pose3> guesses{predicted};
    if (global && params.initialization) {
      const gtsam::Rot3 rp = gtsam::Rot3::Ypr(0, predicted.rotation().pitch(), predicted.rotation().roll());
      const Matrix leveled = transform_body_points(source, gtsam::Pose3(rp, gtsam::Point3(0, 0, 0)));
      Eigen::Matrix<double, 3, 2> bounds;
      bounds << -params.max_translation, params.max_translation,
                -params.max_translation, params.max_translation, -params.max_rotation, params.max_rotation;
      const auto seed = global_scan_match_init(leveled, pose322(predicted), target, gtsam::Pose2(),
        s.point_noise, bounds, params.init_n, params.init_iters, params.init_ftol, deadline);
      if (seed.timed_out) return timeout();
      if (seed.success) {
        const auto xy = pose322(predicted).compose(seed.delta);
        guesses.emplace_back(gtsam::Rot3::Ypr(xy.theta(), predicted.rotation().pitch(), predicted.rotation().roll()),
          gtsam::Point3(xy.x(), xy.y(), predicted.z()));
      }
    }
    // Refine depth and attitude; they are optimization variables, not
    // pressure prealignment or fixed IMU rotations of the source cloud.
    const auto center = guesses.back();
    for (int axis : {0, 1, 5}) for (double sign : {-1.0, 1.0}) {
      gtsam::Vector6 delta = gtsam::Vector6::Zero();
      delta[axis] = sign * (axis == 5 ? .25 : .05);
      guesses.push_back(center.compose(gtsam::Pose3::Expmap(delta)));
    }
    RegistrationResult best;
    std::vector<RegistrationResult> hypotheses;
    std::vector<double> costs;
    double best_score = std::numeric_limits<double>::infinity();
    for (const auto& guess : guesses) {
      const int budget = remaining_ms(deadline, 2000);
      if (budget <= 0) return timeout();
      auto fit = matcher->compute(source, guess, budget);
      if (fit.timed_out || deadline_expired(deadline)) return timeout();
      if (!fit.converged) { if (!best.converged) best = fit; continue; }
      const double score = fit.rmse * fit.rmse +
        (1.0 - fit.overlap) * std::pow(s.se3_options.icp.max_correspondence, 2);
      hypotheses.push_back(fit); costs.push_back(score);
      if (score < best_score) { best = std::move(fit); best_score = score; }
    }
    if (deadline_expired(deadline)) return timeout();
    for (std::size_t i = 0; i < hypotheses.size(); ++i)
      if (costs[i] <= best_score * 1.2 + 1e-4 && distinct_hypotheses(best, hypotheses[i])) {
        best.converged = best.has_covariance = false; best.information.setZero();
        best.reason = "SE3 registration ambiguous between distinct hypotheses";
        return best;
      }
    if (best.converged) {
      const auto correction = predicted.between(best.transform);
      if (correction.translation().norm() > params.max_translation || angle(correction) > params.max_rotation) {
        best.converged = false;
        best.reason = "SE3 registration exceeds fused-odometry correction limits";
      }
    }
    return best;
  }

  Covariance6 path_covariance(int from, int to) const
  {
    if (from == to) return Covariance6::Identity() * 1e-12;
    if (from > to) {
      const auto forward = s.keyframes[to]->pose3.between(s.keyframes[from]->pose3);
      return inverse_covariance(forward, path_covariance(to, from));
    }
    Covariance6 covariance = Covariance6::Zero();
    for (int k = from + 1; k <= to; ++k) {
      const auto relative = s.keyframes[k - 1]->pose3.between(s.keyframes[k]->pose3);
      const Covariance6 edge = k < static_cast<int>(links.size())
        ? links[k].odometry_covariance : Covariance6::Identity() * 1e6;
      covariance = compose_covariance(relative, covariance, edge);
    }
    return covariance;
  }
  double consistency(const Loop& a, const Loop& b) const
  {
    const auto source_motion = s.keyframes[a.source]->pose3.between(s.keyframes[b.source]->pose3);
    const auto target_motion = s.keyframes[b.target]->pose3.between(s.keyframes[a.target]->pose3);
    gtsam::Pose3 cycle = a.transform;
    Covariance6 covariance = a.covariance;
    covariance = compose_covariance(source_motion, covariance, path_covariance(a.source, b.source));
    cycle = cycle.compose(source_motion);
    covariance = compose_covariance(b.transform.inverse(), covariance, inverse_covariance(b.transform, b.covariance));
    cycle = cycle.compose(b.transform.inverse());
    covariance = compose_covariance(target_motion, covariance, path_covariance(b.target, a.target));
    cycle = cycle.compose(target_motion);
    Covariance6 logarithm;
    const auto residual = gtsam::Pose3::Logmap(cycle, logarithm);
    covariance = logarithm * covariance * logarithm.transpose();
    const Eigen::LDLT<Covariance6> inverse(covariance);
    return inverse.info() == Eigen::Success && inverse.isPositive()
      ? residual.dot(inverse.solve(residual)) : std::numeric_limits<double>::infinity();
  }
  double tear_ratio(const gtsam::Values& estimate, int k) const
  {
    const auto actual = estimate.at<gtsam::Pose3>(key(k - 1)).between(estimate.at<gtsam::Pose3>(key(k)));
    const auto residual = gtsam::Pose3::Logmap(links[k].measured.between(actual));
    const Eigen::LLT<Covariance6> noise(links[k].covariance);
    return noise.info() == Eigen::Success ? noise.matrixL().solve(residual).cwiseAbs().maxCoeff() :
      std::numeric_limits<double>::infinity();
  }
  bool commit(const KeyframePtr& frame)
  {
    if (frame != staged_frame) {
      s.last_error_ = "SE3 committed frame does not match the prepared update";
      return false;
    }
    if (!s.has_pending_update()) { s.last_error_.clear(); return true; }
    const bool loop_round = std::any_of(staged.begin(), staged.end(), [](const auto& r) { return r.loop; });
    const auto records_before = records.size(), frames_before = s.keyframes.size();
    const auto links_before = links.size(), manual_before = manual_stack.size();
    const int accepted_before = s.nssm_accepted, ssm_before = s.ssm_accepted;
    std::vector<gtsam::Pose3> poses_before;
    std::vector<std::size_t> constraints_before;
    for (const auto& previous : s.keyframes) {
      poses_before.push_back(previous->pose3); constraints_before.push_back(previous->constraints.size());
    }
    const auto loops_before = loops;
    try {
      gtsam::NonlinearFactorGraph graph;
      for (const auto& record : staged) graph.push_back(factor(record));
      engine.update(graph, staged_values);
      const bool manual_round = std::any_of(staged.begin(), staged.end(), [](const auto& r) { return r.manual; });
      for (int i = 0; i < (manual_round ? 5 : loop_round ? std::max(3, s.loop_extra_iterations) : 1); ++i) engine.update();
      const auto estimate = engine.calculateEstimate();
      // Validate every estimate before mutating the published state.
      if (estimate.size() != s.keyframes.size() + (frame ? 1 : 0))
        throw std::runtime_error("SE3 solver state count does not match the committed keyframes");
      for (const auto& value : estimate)
        if (!estimate.at<gtsam::Pose3>(value.key).matrix().allFinite())
          throw std::runtime_error("SE3 solve returned non-finite state");
      if (loop_round && s.post_loop_max_link_error_sigma > 0) {
        for (int k = 1; k < s.current_key(); ++k) {
          if (k >= static_cast<int>(links.size()) || !links[k].valid) continue;
          if (tear_ratio(estimate, k) > std::max(s.post_loop_max_link_error_sigma,
                links[k].accepted_absolute_tear) + 1e-3)
            throw std::runtime_error("SE3 post-loop verification found a six-axis chain tear");
        }
      }
      const auto old_size = records.size();
      records.insert(records.end(), staged.begin(), staged.end());
      for (std::size_t i = old_size; i < records.size(); ++i)
        if (records[i].manual) manual_stack.push_back(i);
      if (frame) {
        s.keyframes.push_back(frame);
        links.push_back(staged_link);
        if (staged_acoustic) ++s.ssm_accepted;
      }
      for (const auto& record : staged) if (record.loop) {
        ++s.nssm_accepted;
        s.keyframes[record.b]->constraints.emplace_back(record.a, pose322(record.pose));
        for (auto& loop : loops)
          if (loop.target == record.a && loop.source == record.b) loop.inserted = true;
      }
      apply(estimate);
      const bool absolute_round = std::any_of(staged.begin(), staged.end(), [](const auto& record) {
        return record.manual || record.kind == Kind::Position;
      });
      for (int k = 1; k < static_cast<int>(links.size()); ++k) if (links[k].valid) {
        if (absolute_round) links[k].accepted_absolute_tear = tear_ratio(estimate, k);
        else if (loop_round && links[k].accepted_absolute_tear > 0)
          links[k].accepted_absolute_tear = std::min(links[k].accepted_absolute_tear, tear_ratio(estimate, k));
      }
      clear_staged();
      s.last_error_.clear();
      return true;
    } catch (const std::exception& error) {
      s.last_error_ = error.what();
      records.resize(records_before); links.resize(links_before); manual_stack.resize(manual_before);
      s.keyframes.resize(frames_before);
      s.nssm_accepted = accepted_before; s.ssm_accepted = ssm_before;
      for (std::size_t k = 0; k < frames_before; ++k) {
        s.keyframes[k]->update(poses_before[k]); s.keyframes[k]->constraints.resize(constraints_before[k]);
      }
      loops = loops_before;
      if (loop_round) {
        ++s.nssm_reverted;
        for (const auto& record : staged) if (record.loop) quarantined.emplace(record.a, record.b);
        loops.erase(std::remove_if(loops.begin(), loops.end(), [&](const auto& loop) {
          return quarantined.count({loop.target, loop.source}); }), loops.end());
      }
      clear_staged();
      rebuild();
      return false;
    }
  }
};

Se3Backend::Se3Backend(Slam& owner) : impl_(std::make_unique<Impl>(owner)) {}
Se3Backend::~Se3Backend() = default;
void Se3Backend::reset() { impl_ = std::make_unique<Impl>(impl_->s); }
int Se3Backend::queue_depth() const { return impl_->loops.size(); }
int Se3Backend::manual_count() const { return impl_->manual_stack.size(); }

bool Se3Backend::is_keyframe(const Keyframe& frame) const
{
  const auto& s = impl_->s;
  if (s.keyframes.empty()) return true;
  const auto& previous = s.keyframes.back();
  const double duration = to_sec(frame.time) - to_sec(previous->time);
  if (duration <= 0 || duration < s.keyframe_duration) return false;
  const auto increment = previous->dr_pose3.between(frame.dr_pose3);
  return increment.translation().norm() > s.keyframe_translation || angle(increment) > s.keyframe_rotation ||
    (s.keyframe_head_rotation > 0 && std::abs(frame.head_pitch - previous->head_pitch) > s.keyframe_head_rotation);
}
void Se3Backend::add_prior(const KeyframePtr& frame) { impl_->prepare(frame, true); }
void Se3Backend::add_odometry(const KeyframePtr& frame) { impl_->prepare(frame, false); }
bool Se3Backend::update(const KeyframePtr& frame) { return impl_->commit(frame); }
bool Se3Backend::has_pending_update() const
{
  return impl_->staged_frame || !impl_->staged.empty() || !impl_->staged_values.empty();
}
void Se3Backend::discard_pending_update() { impl_->clear_staged(); }

void Se3Backend::add_sequential(const KeyframePtr& frame)
{
  auto& p = *impl_;
  auto& s = p.s;
  p.prepare(frame, false);
  if (!s.ssm_params.enable || frame->points.rows() < s.ssm_params.min_points) {
    s.last_ssm_status = "SE3 odometry-only: no eligible acoustic source";
    return;
  }
  const int target_key = s.current_key() - 1;
  std::vector<int> keys;
  const int first = std::max(s.loaded_key_count_, target_key - s.ssm_params.target_frames + 1);
  for (int k = first; k <= target_key; ++k) if (s.keyframes[k]->points.rows()) keys.push_back(k);
  Matrix target = s.get_points(keys, target_key);
  cap_points(target, s.se3_options.max_source_points * 4);
  if (target.rows() < s.ssm_params.min_points) { s.last_ssm_status = "SE3 target too sparse"; return; }
  const auto predicted = s.keyframes.back()->dr_pose3.between(frame->dr_pose3);
  RegistrationResult fit;
  try { fit = p.register_cloud(frame->points, target, predicted, s.ssm_params, false); }
  catch (const std::exception& error) { fit.reason = error.what(); }
  if (!p.acceptable(fit, s.ssm_params, s.ssm_max_sigma)) {
    ++s.ssm_degenerate_rejected;
    s.last_ssm_status = "SE3 odometry-only: " + fit.reason;
    return;
  }
  Impl::Record record;
  record.kind = Impl::Kind::Between;
  record.a = target_key; record.b = target_key + 1;
  record.pose = fit.transform; record.covariance = fit.covariance;
  p.staged.push_back(record);
  p.staged_values.update(key(record.b), s.keyframes.back()->pose3.compose(fit.transform));
  p.staged_link.measured = fit.transform;
  p.staged_link.covariance = fit.covariance;
  p.staged_acoustic = true;
  s.last_ssm_status = "SE3 accepted: rank 6, overlap " + std::to_string(fit.overlap) +
    ", rmse " + std::to_string(fit.rmse);
}

bool Se3Backend::add_nonsequential()
{
  auto& p = *impl_;
  auto& s = p.s;
  if (!s.require_idle_update("loop closure")) return false;
  s.last_nssm_inserted_geom.clear();
  if (s.awaiting_relocalization_ || !s.nssm_params.enable || s.current_key() < s.nssm_params.min_st_sep + 1) return false;
  ++s.nssm_attempts;
  const auto reject = [&](const std::string& reason) {
    s.last_nssm_status = reason; ++s.nssm_reject_hist[reason]; return false;
  };
  const int source_key = s.current_key() - 1;
  const auto& source_frame = s.keyframes.back();
  if (!source_frame->has_covariance6) return reject("SE3 source marginal unavailable");
  std::vector<int> sources;
  for (int k = std::max(s.loaded_key_count_, source_key - s.nssm_params.source_frames + 1); k <= source_key; ++k)
    if (s.keyframes[k]->points.rows()) sources.push_back(k);
  Matrix source = s.get_points(sources, source_key);
  cap_points(source, s.se3_options.max_source_points);
  if (source.rows() < s.nssm_params.min_points) return reject("SE3 source too sparse");
  int target_key = -1;
  double nearest = std::numeric_limits<double>::infinity();
  for (int k = 0; k <= source_key - std::max(s.nssm_params.min_st_sep, s.nssm_params.min_revisit_sep); ++k) {
    if (!s.keyframes[k]->points.rows() || p.quarantined.count({k, source_key})) continue;
    const double distance = source_frame->pose3.between(s.keyframes[k]->pose3).translation().norm();
    if (distance < nearest && distance <= s.sonar_max_range + s.nssm_params.max_translation) {
      nearest = distance; target_key = k;
    }
  }
  if (target_key < 0) return reject("SE3 no separated local target");
  std::vector<int> targets;
  for (int k = std::max(0, target_key - 1); k <= std::min(target_key + 1, source_key - s.nssm_params.min_revisit_sep); ++k)
    if (s.keyframes[k]->points.rows()) targets.push_back(k);
  Matrix target = s.get_points(targets, target_key);
  cap_points(target, s.se3_options.max_source_points * 4);
  if (target.rows() < s.nssm_params.min_points) return reject("SE3 target too sparse");
  const auto predicted = s.same_dr_epoch(target_key, source_key)
    ? s.keyframes[target_key]->dr_pose3.between(source_frame->dr_pose3)
    : s.keyframes[target_key]->pose3.between(source_frame->pose3);
  const auto fit = p.register_cloud(source, target, predicted, s.nssm_params, true);
  if (!p.acceptable(fit, s.nssm_params, s.nssm_max_sigma)) return reject("SE3 unobservable or rejected registration");
  if (std::any_of(p.loops.begin(), p.loops.end(), [&](const auto& loop) {
      return loop.target == target_key && loop.source == source_key; })) return false;
  p.loops.push_back({target_key, source_key, fit.transform, fit.covariance, false});
  while (static_cast<int>(p.loops.size()) > s.pcm_queue_size) p.loops.pop_front();
  ++s.nssm_queued;
  const int count = p.loops.size();
  std::vector<std::vector<bool>> edges(count, std::vector<bool>(count, true));
  s.last_pcm_edges = 0; s.last_pcm_min_md = -1;
  for (int a = 0; a < count; ++a) for (int b = a + 1; b < count; ++b) {
    const double distance = p.consistency(p.loops[a], p.loops[b]);
    if (std::isfinite(distance) && (s.last_pcm_min_md < 0 || distance < s.last_pcm_min_md)) s.last_pcm_min_md = distance;
    edges[a][b] = edges[b][a] = distance <= 16.811893829770927;  // chi-square(6), 99%
    s.last_pcm_edges += edges[a][b];
  }
  std::vector<int> best;
  // Queue capacity is validated below 20 to keep exhaustive PCM bounded.
  for (unsigned mask = 1; mask < (1U << count); ++mask) {
    std::vector<int> clique;
    bool valid = true;
    for (int a = 0; a < count && valid; ++a) if (mask & (1U << a)) {
      for (int b : clique) if (!edges[a][b]) { valid = false; break; }
      clique.push_back(a);
    }
    if (valid && clique.size() > best.size()) best = std::move(clique);
  }
  s.nssm_best_clique = std::max(s.nssm_best_clique, static_cast<int>(best.size()));
  if (static_cast<int>(best.size()) < s.min_pcm) return reject("SE3 awaiting six-axis PCM support");
  for (int index : best) {
    const auto& loop = p.loops[index];
    if (loop.inserted) continue;
    Impl::Record record;
    record.kind = Impl::Kind::Between; record.a = loop.target; record.b = loop.source;
    record.pose = loop.transform; record.covariance = loop.covariance; record.loop = true;
    record.dcs_phi = s.nssm_use_dcs ? s.nssm_dcs_phi : 0;
    p.staged.push_back(record);
    s.last_nssm_inserted_geom.push_back("SE3 " + std::to_string(record.a) + " -> " + std::to_string(record.b));
  }
  s.last_nssm_status = "SE3 consistent six-axis loop constraints staged";
  return !p.staged.empty();
}

bool Se3Backend::add_manual(const gtsam::Pose2& pose)
{
  auto& p = *impl_;
  if (!p.s.require_idle_update("manual correction")) return false;
  if (p.s.keyframes.empty() || p.s.awaiting_relocalization_) return false;
  if (std::abs(std::cos(p.s.keyframes.back()->pose3.rotation().pitch())) < 1e-3) {
    p.s.last_error_ = "planar heading is undefined near vertical attitude; use a full pose correction";
    return false;
  }
  Impl::Record record;
  record.kind = Impl::Kind::Planar; record.a = p.s.current_key() - 1;
  record.position = Eigen::Vector3d(pose.x(), pose.y(), pose.theta());
  record.covariance.topLeftCorner<3, 3>() = p.s.manual_correction_sigmas.array().square().matrix().asDiagonal();
  record.manual = true;
  if (!record.position.allFinite()) return false;
  p.staged.push_back(record);
  if (!p.commit(nullptr)) return false;
  ++p.s.manual_corrections_applied;
  return true;
}

bool Se3Backend::add_manual(const gtsam::Pose3& pose, const Covariance6& covariance)
{
  auto& p = *impl_;
  if (!p.s.require_idle_update("manual correction")) return false;
  if (p.s.keyframes.empty() || p.s.awaiting_relocalization_ ||
      !pose.matrix().allFinite() || !valid_covariance6(covariance)) {
    p.s.last_error_ = "full pose correction needs a localized graph and finite positive definite covariance";
    return false;
  }
  Impl::Record record;
  record.kind = Impl::Kind::Prior; record.a = p.s.current_key() - 1;
  record.pose = pose; record.covariance = covariance; record.manual = true;
  p.staged.push_back(record);
  if (!p.commit(nullptr)) return false;
  ++p.s.manual_corrections_applied;
  return true;
}

bool Se3Backend::undo_manual()
{
  auto& p = *impl_;
  if (!p.s.require_idle_update("undo correction")) return false;
  if (p.s.awaiting_relocalization_ || p.manual_stack.empty()) { p.s.last_error_ = "no localized SE3 manual correction to undo"; return false; }
  const auto index = p.manual_stack.back();
  p.records[index].active = false;
  try {
    p.rebuild();
    const auto estimate = p.engine.calculateEstimate();
    p.apply(estimate);
    for (int k = 1; k < static_cast<int>(p.links.size()); ++k)
      if (p.links[k].valid) p.links[k].accepted_absolute_tear = p.tear_ratio(estimate, k);
    p.manual_stack.pop_back();
    ++p.s.manual_corrections_undone;
    return true;
  } catch (const std::exception& error) {
    p.records[index].active = true;
    p.s.last_error_ = error.what();
    p.rebuild();
    return false;
  }
}

bool Se3Backend::add_position(int index, double x, double y, double sigma, const gtsam::Point3& point)
{
  auto& p = *impl_;
  if (!p.s.require_idle_update("position correction")) return false;
  if (index < 0 || index >= p.s.current_key() || !std::isfinite(x) || !std::isfinite(y) ||
      !std::isfinite(sigma) || sigma <= 0 || !point.allFinite() || p.s.awaiting_relocalization_) {
    p.s.last_error_ = "invalid or unlocalized SE3 position prior"; return false;
  }
  Impl::Record record;
  record.kind = Impl::Kind::Position; record.a = index; record.position << x, y, 0;
  record.pose = gtsam::Pose3(gtsam::Rot3(), point);
  record.covariance.topLeftCorner<2, 2>() = Eigen::Matrix2d::Identity() * sigma * sigma;
  p.staged.push_back(record);
  if (!p.commit(nullptr)) return false;
  ++p.s.position_priors_applied;
  return true;
}

namespace {
// Native binary payload with explicit version/model metadata, bounded lengths
// and a streaming integrity checksum. This is a map format, not a wire format.
struct MapWriter
{
  std::ofstream file;
  std::uint64_t hash = 14695981039346656037ULL;
  explicit MapWriter(const std::string& path) : file(path, std::ios::binary | std::ios::trunc) {}
  void bytes(const void* pointer, std::size_t count)
  {
    file.write(static_cast<const char*>(pointer), count);
    if (!file) throw std::runtime_error("SE3 map write failed");
    const auto* data = static_cast<const unsigned char*>(pointer);
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ data[i]) * 1099511628211ULL;
  }
  template<class T> void scalar(const T& value) { bytes(&value, sizeof value); }
  void text(const std::string& value)
  {
    if (value.size() > 4096) throw std::runtime_error("map metadata exceeds 4096 bytes");
    scalar(static_cast<std::uint32_t>(value.size())); bytes(value.data(), value.size());
  }
  void pose(const gtsam::Pose3& value)
  {
    const auto q = value.rotation().toQuaternion();
    const double values[7] = {q.w(), q.x(), q.y(), q.z(), value.x(), value.y(), value.z()};
    bytes(values, sizeof values);
  }
  void finish()
  {
    file.write(reinterpret_cast<const char*>(&hash), sizeof hash);
    file.close();
    if (!file) throw std::runtime_error("SE3 map could not be completed");
  }
};
struct MapReader
{
  std::ifstream file;
  std::uint64_t hash = 14695981039346656037ULL, remaining = 0;
  explicit MapReader(const std::string& path) : file(path, std::ios::binary), remaining(std::filesystem::file_size(path)) {}
  void bytes(void* pointer, std::size_t count)
  {
    if (count > remaining || !file.read(static_cast<char*>(pointer), count)) throw std::runtime_error("SE3 map is truncated");
    remaining -= count;
    const auto* data = static_cast<const unsigned char*>(pointer);
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ data[i]) * 1099511628211ULL;
  }
  template<class T> T scalar() { T value{}; bytes(&value, sizeof value); return value; }
  std::string text()
  {
    const auto count = scalar<std::uint32_t>();
    if (count > 4096) throw std::runtime_error("invalid SE3 metadata length");
    std::string result(count, '\0'); bytes(result.data(), count); return result;
  }
  gtsam::Pose3 pose()
  {
    double values[7]; bytes(values, sizeof values);
    for (double value : values) if (!std::isfinite(value)) throw std::runtime_error("non-finite SE3 map pose");
    Eigen::Quaterniond q(values[0], values[1], values[2], values[3]);
    if (std::abs(q.norm() - 1.0) > 1e-3) throw std::runtime_error("invalid SE3 map quaternion");
    q.normalize();
    return gtsam::Pose3(gtsam::Rot3::Quaternion(q.w(), q.x(), q.y(), q.z()),
      gtsam::Point3(values[4], values[5], values[6]));
  }
  Covariance6 covariance()
  {
    Covariance6 result; bytes(result.data(), sizeof(double) * 36);
    if (!valid_covariance6(result)) throw std::runtime_error("invalid SE3 map covariance");
    return result;
  }
  void finish()
  {
    std::uint64_t expected = 0;
    if (remaining != sizeof expected || !file.read(reinterpret_cast<char*>(&expected), sizeof expected) || expected != hash)
      throw std::runtime_error("SE3 map checksum or trailing data is invalid");
  }
};
}  // namespace

bool Se3Backend::save(const std::string& path)
{
  auto& p = *impl_;
  if (!p.s.require_idle_update("map save")) return false;
  if (p.s.keyframes.empty() || !p.staged.empty() || p.s.map_calibration_id.empty() || p.s.map_depth_datum.empty() ||
      std::any_of(p.s.keyframes.begin(), p.s.keyframes.end(), [](const auto& frame) {
        return frame->chart != CloudChart::Body || frame->points.cols() != 3 || !frame->points.allFinite() ||
          !frame->pose3.matrix().allFinite();
      })) {
    p.s.last_error_ = "SE3 map save needs committed states, map_calibration_id and map_depth_datum";
    return false;
  }
  const std::string temporary = path + ".tmp." + std::to_string(std::random_device{}());
  try {
    MapWriter out(temporary);
    out.bytes("SSLMSE3\0", 8); out.scalar(std::uint32_t(1));
    out.text("measured_xyz"); out.text("se3/body");
    out.text(p.s.map_calibration_id); out.text(p.s.map_depth_datum); out.text("fused_tf");
    out.scalar(static_cast<std::uint32_t>(p.s.current_key()));
    const auto active = std::count_if(p.records.begin(), p.records.end(), [](const auto& r) { return r.active; });
    out.scalar(static_cast<std::uint32_t>(active));
    for (int k = 0; k < p.s.current_key(); ++k) {
      const auto& frame = p.s.keyframes[k];
      const std::int64_t stamp = static_cast<std::int64_t>(frame->time.sec) * 1000000000LL + frame->time.nanosec;
      out.scalar(stamp); out.pose(frame->dr_pose3); out.pose(frame->pose3);
      out.scalar(static_cast<std::uint8_t>(frame->has_covariance6));
      if (frame->has_covariance6) out.bytes(frame->covariance6.data(), 36 * sizeof(double));
      out.scalar(static_cast<std::uint64_t>(frame->points.rows()));
      out.bytes(frame->points.data(), frame->points.size() * sizeof(float));
      const auto& link = p.links.at(k);
      out.scalar(static_cast<std::uint8_t>(link.valid)); out.pose(link.measured);
      out.bytes(link.covariance.data(), 36 * sizeof(double));
      out.bytes(link.odometry_covariance.data(), 36 * sizeof(double));
      out.scalar(link.accepted_absolute_tear);
    }
    for (const auto& record : p.records) if (record.active) {
      out.scalar(static_cast<std::uint8_t>(record.kind));
      out.scalar(static_cast<std::uint32_t>(record.a)); out.scalar(static_cast<std::uint32_t>(record.b));
      out.pose(record.pose); out.bytes(record.position.data(), 3 * sizeof(double));
      out.bytes(record.covariance.data(), 36 * sizeof(double));
      out.scalar(static_cast<std::uint8_t>(record.manual)); out.scalar(static_cast<std::uint8_t>(record.loop));
      out.scalar(record.dcs_phi);
    }
    out.finish();
    std::filesystem::rename(temporary, path);
    p.s.last_error_.clear();
    return true;
  } catch (const std::exception& error) {
    std::error_code ignored; std::filesystem::remove(temporary, ignored);
    p.s.last_error_ = error.what(); return false;
  }
}

bool Se3Backend::load(const std::string& path)
{
  auto& p = *impl_;
  auto& s = p.s;
  if (!s.require_idle_update("map load")) return false;
  if (!s.keyframes.empty()) { s.last_error_ = "SE3 map load requires an empty session"; return false; }
  try {
    MapReader in(path);
    char magic[8]; in.bytes(magic, sizeof magic);
    if (std::memcmp(magic, "SSLMSE3\0", 8) || in.scalar<std::uint32_t>() != 1)
      throw std::runtime_error("map is not a supported physical SE3 map; constrained charts cannot be imported implicitly");
    if (in.text() != "measured_xyz" || in.text() != "se3/body") throw std::runtime_error("SE3 map measurement/state model mismatch");
    const auto calibration = in.text(), datum = in.text(), source = in.text();
    if (s.map_calibration_id.empty() || calibration != s.map_calibration_id ||
        datum != s.map_depth_datum || source != "fused_tf")
      throw std::runtime_error("SE3 map calibration, depth datum or odometry model mismatch");
    const auto count = in.scalar<std::uint32_t>(), factors = in.scalar<std::uint32_t>();
    if (!count || count > 1000000 || count > in.remaining / 700 || factors > in.remaining / 380)
      throw std::runtime_error("invalid SE3 map state/factor counts");
    std::vector<KeyframePtr> frames;
    std::vector<Impl::Link> links;
    std::vector<Impl::Record> records;
    for (std::uint32_t k = 0; k < count; ++k) {
      const auto nanos = in.scalar<std::int64_t>();
      if (nanos <= 0 || nanos / 1000000000LL > std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("invalid SE3 map timestamp");
      builtin_interfaces::msg::Time stamp;
      stamp.sec = nanos / 1000000000LL; stamp.nanosec = nanos % 1000000000LL;
      const auto dr = in.pose(), pose = in.pose();
      auto frame = std::make_shared<Keyframe>(true, stamp, dr);
      frame->chart = CloudChart::Body;
      const auto has_covariance = in.scalar<std::uint8_t>();
      if (has_covariance > 1) throw std::runtime_error("invalid SE3 covariance flag");
      if (has_covariance) frame->set_covariance6(in.covariance());
      const auto rows = in.scalar<std::uint64_t>();
      if (rows > 10000000 || rows > in.remaining / (3 * sizeof(float))) throw std::runtime_error("invalid SE3 cloud count");
      frame->points.resize(rows, 3); in.bytes(frame->points.data(), rows * 3 * sizeof(float));
      if (!frame->points.allFinite()) throw std::runtime_error("non-finite SE3 map points");
      cap_points(frame->points, s.se3_options.max_source_points);
      frame->update(pose);
      frames.push_back(frame);
      Impl::Link link;
      const auto valid = in.scalar<std::uint8_t>();
      if (valid > 1) throw std::runtime_error("invalid SE3 link flag");
      link.valid = valid; link.measured = in.pose();
      link.covariance = in.covariance(); link.odometry_covariance = in.covariance();
      link.accepted_absolute_tear = in.scalar<double>();
      if (!std::isfinite(link.accepted_absolute_tear) || link.accepted_absolute_tear < 0)
        throw std::runtime_error("invalid SE3 absolute-correction baseline");
      links.push_back(link);
    }
    for (std::uint32_t i = 0; i < factors; ++i) {
      Impl::Record record;
      const auto kind = in.scalar<std::uint8_t>();
      if (kind > static_cast<unsigned>(Impl::Kind::Position)) throw std::runtime_error("invalid SE3 factor type");
      record.kind = static_cast<Impl::Kind>(kind);
      const auto a = in.scalar<std::uint32_t>(), b = in.scalar<std::uint32_t>();
      if (a >= count || b >= count) throw std::runtime_error("invalid SE3 factor keys");
      record.a = a; record.b = b; record.pose = in.pose();
      in.bytes(record.position.data(), 3 * sizeof(double)); record.covariance = in.covariance();
      const auto manual = in.scalar<std::uint8_t>(), loop = in.scalar<std::uint8_t>();
      record.dcs_phi = in.scalar<double>();
      if (manual > 1 || loop > 1 || !record.position.allFinite() || !std::isfinite(record.dcs_phi) || record.dcs_phi < 0)
        throw std::runtime_error("invalid SE3 factor metadata");
      record.manual = manual; record.loop = loop;
      records.push_back(record);
    }
    in.finish();
    gtsam::ISAM2 candidate(engine_params());
    gtsam::NonlinearFactorGraph graph;
    for (const auto& record : records) graph.push_back(p.factor(record));
    gtsam::Values values;
    for (std::uint32_t k = 0; k < count; ++k) values.insert(key(k), frames[k]->pose3);
    candidate.update(graph, values);
    for (int i = 0; i < 5; ++i) candidate.update();
    const auto estimate = candidate.calculateEstimate();
    for (std::uint32_t k = 0; k < count; ++k)
      if (!estimate.at<gtsam::Pose3>(key(k)).matrix().allFinite()) throw std::runtime_error("loaded SE3 graph is not finite");
    p.records = std::move(records); p.links = std::move(links); p.engine = std::move(candidate);
    s.keyframes = std::move(frames);
    for (std::size_t i = 0; i < p.records.size(); ++i) {
      if (p.records[i].manual) p.manual_stack.push_back(i);
      if (p.records[i].loop) s.keyframes[p.records[i].b]->constraints.emplace_back(p.records[i].a, pose322(p.records[i].pose));
    }
    p.apply(estimate);
    s.loaded_key_count_ = count; s.awaiting_relocalization_ = true;
    s.last_error_.clear();
    return true;
  } catch (const std::exception& error) {
    s.last_error_ = error.what();
    return false;
  }
}

bool Se3Backend::relocalize(const KeyframePtr& frame, Deadline deadline)
{
  auto& p = *impl_;
  auto& s = p.s;
  if (!s.require_idle_update("relocalization")) return false;
  deadline = std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(s.relocalize_max_ms));
  const auto timeout = [&]() {
    s.last_error_ = "SE3 relocalization deadline exceeded; incomplete search rejected";
    return false;
  };
  if (deadline_expired(deadline)) return timeout();
  if (!s.awaiting_relocalization_ || frame->chart != CloudChart::Body ||
      frame->points.cols() != 3 || !frame->points.allFinite() || frame->points.rows() < s.nssm_params.min_points) {
    s.last_error_ = "SE3 relocalization needs a loaded map and informative source"; return false;
  }
  // Empty/sparse odometry anchors are valid graph states, but cannot occupy
  // the nearest-reference shortlist ahead of informative sonar keyframes.
  std::vector<int> keys;
  for (int k = 0; k < s.loaded_key_count_; ++k) {
    if (deadline_expired(deadline)) return timeout();
    if (s.keyframes[k]->points.rows() >= s.se3_options.icp.normal_neighbors) keys.push_back(k);
  }
  cap_points(frame->points, s.se3_options.max_source_points);
  // Stream an XY proposal index without copying the dense historical map or
  // building whole-map normals. All actual SE3 fits use physical references.
  const double resolution = std::max(.25, s.point_noise);
  std::unordered_set<std::uint64_t> occupied;
  std::vector<Eigen::Vector3f> seed_points;
  for (int k : keys) for (Eigen::Index i = 0; i < s.keyframes[k]->transf_points.rows(); ++i) {
    if (i % 1024 == 0 && deadline_expired(deadline)) return timeout();
    const auto& points = s.keyframes[k]->transf_points;
    const double x = std::floor(points(i, 0) / resolution), y = std::floor(points(i, 1) / resolution);
    if (std::abs(x) > std::numeric_limits<std::int32_t>::max() || std::abs(y) > std::numeric_limits<std::int32_t>::max()) {
      s.last_error_ = "SE3 map coordinates exceed the proposal index range"; return false;
    }
    const std::uint64_t cell = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::int32_t>(x))) << 32) |
      static_cast<std::uint32_t>(static_cast<std::int32_t>(y));
    if (occupied.insert(cell).second) {
      if (seed_points.size() >= static_cast<std::size_t>(s.se3_options.max_seed_points)) {
        s.last_error_ = "SE3 global proposal exceeds se3.max_seed_points; select a smaller reference map"; return false;
      }
      seed_points.emplace_back(points(i, 0), points(i, 1), 0);
    }
  }
  Matrix target(seed_points.size(), 3);
  for (std::size_t i = 0; i < seed_points.size(); ++i) target.row(i) = seed_points[i].transpose();
  if (target.rows() < s.nssm_params.min_points) { s.last_error_ = "SE3 map too sparse"; return false; }
  const auto minimum = target.leftCols(2).colwise().minCoeff().eval();
  const auto maximum = target.leftCols(2).colwise().maxCoeff().eval();
  const gtsam::Pose2 center((minimum.x() + maximum.x()) * .5, (minimum.y() + maximum.y()) * .5, 0);
  const auto rp = gtsam::Rot3::Ypr(0, frame->dr_pose3.rotation().pitch(), frame->dr_pose3.rotation().roll());
  const Matrix proposal = transform_body_points(frame->points, gtsam::Pose3(rp, gtsam::Point3::Zero()));
  Eigen::Matrix<double, 3, 2> bounds;
  bounds << -(maximum.x() - minimum.x()) * .5 - 2, (maximum.x() - minimum.x()) * .5 + 2,
            -(maximum.y() - minimum.y()) * .5 - 2, (maximum.y() - minimum.y()) * .5 + 2,
            frame->dr_pose3.rotation().yaw() - .35, frame->dr_pose3.rotation().yaw() + .35;
  const auto init = global_scan_match_init(proposal, center, target, gtsam::Pose2(), s.point_noise, bounds, 500, 4, .01, deadline);
  if (init.timed_out || deadline_expired(deadline)) return timeout();
  if (!init.success) { s.last_error_ = "SE3 global proposal failed"; return false; }
  std::vector<Eigen::Vector4d> samples = init.pose_samples;
  std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) { return a[3] < b[3]; });
  std::vector<gtsam::Pose3> guesses;
  const auto seed = center.compose(init.delta);
  const auto make_pose = [&](double x, double y, double yaw) {
    return gtsam::Pose3(gtsam::Rot3::Ypr(yaw, frame->dr_pose3.rotation().pitch(), frame->dr_pose3.rotation().roll()),
      gtsam::Point3(x, y, frame->dr_pose3.z()));
  };
  guesses.push_back(make_pose(seed.x(), seed.y(), seed.theta()));
  for (std::size_t i = 0; i < std::min<std::size_t>(24, samples.size()); ++i)
    guesses.push_back(make_pose(samples[i][0], samples[i][1], samples[i][2]));
  const auto primary = guesses.front();
  for (int axis : {0, 1, 5}) for (double sign : {-1.0, 1.0}) {
    gtsam::Vector6 delta = gtsam::Vector6::Zero();
    delta[axis] = sign * (axis == 5 ? s.se3_options.relocalize_depth_range : s.se3_options.relocalize_attitude_range);
    guesses.push_back(primary.compose(gtsam::Pose3::Expmap(delta)));
  }
  std::vector<RegistrationResult> fits;
  std::vector<int> fit_references;
  std::map<int, std::unique_ptr<RigidRegistration>> matchers;
  std::vector<double> costs;
  std::size_t best = 0;
  for (const auto& guess : guesses) {
    if (deadline_expired(deadline)) return timeout();
    auto references = keys;
    const auto count = std::min<std::size_t>(3, references.size());
    std::partial_sort(references.begin(), references.begin() + count, references.end(), [&](int a, int b) {
      return (s.keyframes[a]->pose3.translation() - guess.translation()).squaredNorm() <
        (s.keyframes[b]->pose3.translation() - guess.translation()).squaredNorm();
    });
    for (std::size_t i = 0; i < count; ++i) {
      const int reference = references[i];
      auto& matcher = matchers[reference];
      try {
        if (!matcher) matcher = std::make_unique<RigidRegistration>(s.keyframes[reference]->points, s.se3_options.icp, deadline);
      } catch (const DeadlineExceeded&) { return timeout(); }
      const int budget = remaining_ms(deadline, s.relocalize_max_ms);
      if (budget <= 0) return timeout();
      auto fit = matcher->compute(frame->points, s.keyframes[reference]->pose3.between(guess), budget);
      if (fit.timed_out || deadline_expired(deadline)) return timeout();
      if (!p.acceptable(fit, s.nssm_params, s.nssm_max_sigma) || fit.overlap < s.relocalize_min_overlap) continue;
      fit.transform = s.keyframes[reference]->pose3.compose(fit.transform);
      const double score = fit.rmse * fit.rmse + (1.0 - fit.overlap) * std::pow(s.se3_options.icp.max_correspondence, 2);
      fits.push_back(std::move(fit)); costs.push_back(score); fit_references.push_back(reference);
      if (costs.back() < costs[best]) best = costs.size() - 1;
    }
  }
  if (fits.empty()) { s.last_error_ = "SE3 relocalization lacks an observable six-axis fit"; return false; }
  const auto& fit = fits[best];
  for (std::size_t i = 0; i < fits.size(); ++i) {
    if (costs[i] <= costs[best] * 1.2 + 1e-4 && distinct_hypotheses(fit, fits[i])) {
      p.confirmation.reset(); p.confirmations = 0;
      s.last_error_ = "SE3 relocalization ambiguous between separated physical poses"; return false;
    }
  }
  // Tie the resumed session to an observed reference pose. An independent
  // world prior here would count the saved map's gauge information again and
  // lose its correlation with the new state. Use a physical between factor.
  const int reference = fit_references[best];
  const auto matched = fit.transform;
  auto relative = fit;
  relative.transform = s.keyframes[reference]->pose3.between(matched);
  const auto offset = matched.compose(frame->dr_pose3.inverse());
  const Covariance6 boundary = compose_covariance(relative.transform,
    p.path_covariance(s.current_key() - 1, reference), relative.covariance);
  if (deadline_expired(deadline)) return timeout();
  const double stamp = to_sec(frame->time);
  if (p.confirmation) {
    if (stamp <= p.confirmation_stamp) { s.last_error_ = "SE3 confirmation needs a newer observation"; return false; }
    const auto difference = p.confirmation->between(offset);
    if (stamp - p.confirmation_stamp > 5 || difference.translation().norm() > .5 || angle(difference) > .17453292519943295)
      p.confirmations = 0;
  }
  p.confirmation = offset; p.confirmation_stamp = stamp;
  if (++p.confirmations < s.relocalize_min_confirmations) {
    s.last_error_ = "SE3 relocalization awaiting consistent physical-pose confirmation"; return false;
  }
  frame->chart = CloudChart::Body; frame->update(matched);
  Impl::Record connection;
  connection.kind = Impl::Kind::Between;
  connection.a = reference; connection.b = s.current_key();
  connection.pose = relative.transform; connection.covariance = relative.covariance;
  p.staged_link = {false, s.keyframes.back()->pose3.between(matched), boundary, boundary, 0};
  frame->constraints.emplace_back(reference, pose322(relative.transform));
  p.staged.push_back(connection); p.staged_frame = frame;
  p.staged_values.insert(key(s.current_key()), matched);
  if (!p.commit(frame)) return false;
  s.awaiting_relocalization_ = false;
  return true;
}

}  // namespace sonar_slam
