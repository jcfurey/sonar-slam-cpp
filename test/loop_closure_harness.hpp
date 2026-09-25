// Loop-closure harness: synthetic scenes with known ground truth for the
// checks that decide whether a scan match enters the graph -- the sampled
// covariance degeneracy gate, the sequential (SSM) correction bound and the
// PCM consistency test -- run with the DEPLOYED parameters and no bags.
//
// The profile is read from the settings package's slam.yaml (and icp.yaml
// beside it) when the build found it, so the harness follows deployment
// without copying numbers; otherwise it falls back to this package's config.
#pragma once

#include <gtsam/geometry/Pose2.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sonar_slam_cpp/slam_core.hpp"
#include "sonar_slam_cpp/synthetic_world.hpp"

namespace sonar_slam::harness {

// ------------------------------------------------------------ profile

struct ProfilePaths
{
  std::string slam_yaml;
  std::string icp_yaml;
};

inline ProfilePaths locate_profile()
{
  ProfilePaths paths;
  if (const char* configured = std::getenv("SLAM_HARNESS_CONFIG"))
    paths.slam_yaml = configured;
  if (paths.slam_yaml.empty() ||
      !std::filesystem::is_regular_file(paths.slam_yaml))
    paths.slam_yaml = std::string(TEST_SOURCE_DIR) + "/config/slam.yaml";
  const auto sibling =
    std::filesystem::path(paths.slam_yaml).parent_path() / "icp.yaml";
  paths.icp_yaml = std::filesystem::is_regular_file(sibling)
                     ? sibling.string()
                     : std::string(TEST_SOURCE_DIR) + "/config/icp.yaml";
  return paths;
}

// Reads the node's ros__parameters by slash-separated key, the same names
// slam_node.cpp declares.
class ParamReader
{
public:
  explicit ParamReader(const std::string& path)
  {
    const YAML::Node root = YAML::LoadFile(path);
    for (const auto& entry : root) {
      if (entry.second["ros__parameters"]) {
        params_ = YAML::Clone(entry.second["ros__parameters"]);
        break;
      }
    }
    if (!params_ || !params_.IsMap())
      throw std::runtime_error(path + ": no ros__parameters map");
  }

  template <class T>
  T get(const std::string& key, const T& fallback) const
  {
    const auto node = lookup(key);
    return node ? node->as<T>() : fallback;
  }

  template <class T>
  T require(const std::string& key) const
  {
    const auto node = lookup(key);
    if (!node) throw std::runtime_error("missing parameter " + key);
    return node->as<T>();
  }

private:
  std::optional<YAML::Node> lookup(const std::string& key) const
  {
    YAML::Node current;
    current.reset(params_);
    std::size_t start = 0;
    while (true) {
      const std::size_t slash = key.find('/', start);
      const std::string part = key.substr(start, slash - start);
      if (!current.IsMap()) return std::nullopt;
      const YAML::Node& view = current;
      const YAML::Node next = view[part];
      if (!next.IsDefined()) return std::nullopt;
      current.reset(next);
      if (slash == std::string::npos) return current;
      start = slash + 1;
    }
  }

  YAML::Node params_;
};

inline Eigen::Vector3d vector3(const std::vector<double>& values,
                               const std::string& key)
{
  if (values.size() != 3) throw std::runtime_error(key + " needs 3 values");
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

inline SMParams::CovMethod cov_method(const std::string& name)
{
  if (name == "sampled") return SMParams::SAMPLED;
  if (name == "censi") return SMParams::CENSI;
  throw std::runtime_error("unknown cov_method " + name);
}

// Mirrors slam_node.cpp's parameter -> core mapping for every setting the
// scan-matching and loop-closure checks read.
inline void apply_profile(Slam& slam, const ParamReader& p,
                          const std::string& icp_yaml)
{
  slam.keyframe_duration = p.require<double>("keyframe_duration");
  slam.keyframe_translation = p.require<double>("keyframe_translation");
  slam.keyframe_rotation = p.require<double>("keyframe_rotation");
  slam.keyframe_head_rotation =
    p.get<double>("keyframe_head_rotation", 0.17453292519943295);
  slam.prior_sigmas = vector3(
    p.require<std::vector<double>>("prior_sigmas"), "prior_sigmas");
  slam.odom_sigmas = vector3(
    p.require<std::vector<double>>("odom_sigmas"), "odom_sigmas");
  slam.icp_odom_sigmas = vector3(
    p.require<std::vector<double>>("icp_odom_sigmas"), "icp_odom_sigmas");
  slam.point_resolution = p.require<double>("point_resolution");
  slam.point_noise = p.get<double>("point_noise", 0.5);
  slam.constrained_3d = p.get<bool>("constrained_3d/enabled", true);
  auto& icp = slam.constrained_icp_params;
  icp.max_correspondence = static_cast<float>(
    p.get<double>("constrained_3d/max_correspondence", 0.75));
  icp.trim_ratio = p.get<double>("constrained_3d/trim_ratio", 0.8);
  icp.max_iterations = p.get<int>("constrained_3d/max_iterations", 40);
  icp.translation_epsilon =
    p.get<double>("constrained_3d/translation_epsilon", 0.005);
  icp.rotation_epsilon =
    p.get<double>("constrained_3d/rotation_epsilon", 0.001);
  icp.min_correspondences =
    p.get<int>("constrained_3d/min_correspondences", 6);
  slam.sonar_max_range = p.get<double>("sonar_max_range", 30.0);
  slam.sonar_horizontal_aperture =
    p.get<double>("sonar_horizontal_aperture", 2.2689280275926285);
  slam.censi_sensor_noise = p.get<double>("censi_sensor_noise", 0.1);
  slam.parallel_cov_samples = p.get<bool>("parallel_cov_samples", true);

  auto& ssm = slam.ssm_params;
  ssm.enable = p.require<bool>("ssm/enable");
  ssm.min_points = p.require<int>("ssm/min_points");
  ssm.min_overlap_ratio = p.get<double>("ssm/min_overlap_ratio", 0.0);
  ssm.max_translation = p.require<double>("ssm/max_translation");
  ssm.max_rotation = p.require<double>("ssm/max_rotation");
  ssm.target_frames = p.require<int>("ssm/target_frames");
  ssm.initialization = p.get<bool>("ssm/initialization", true);
  const auto ssm_init = p.get<std::vector<double>>(
    "ssm/initialization_params", {50.0, 1.0, 0.01});
  ssm.init_n = static_cast<int>(ssm_init.at(0));
  ssm.init_iters = static_cast<int>(ssm_init.at(1));
  ssm.init_ftol = ssm_init.at(2);
  ssm.cov_samples = p.get<int>("ssm/cov_samples", 0);
  ssm.cov_method = cov_method(p.get<std::string>("ssm/cov_method", "sampled"));
  slam.ssm_require_covariance = p.get<bool>("ssm/require_covariance", true);
  slam.ssm_max_sigma = p.get<double>("ssm/max_sigma", 0.5);
  slam.ssm_max_anisotropy = p.get<double>("ssm/max_anisotropy", 8.0);
  slam.ssm_degeneracy_prefloor = p.get<bool>("ssm/degeneracy_prefloor", true);
  slam.ssm_max_yaw_vs_dr = p.get<double>("ssm/max_yaw_vs_dr", 0.0);

  auto& nssm = slam.nssm_params;
  nssm.enable = p.require<bool>("nssm/enable");
  nssm.min_st_sep = p.require<int>("nssm/min_st_sep");
  nssm.min_revisit_sep = p.get<int>("nssm/min_revisit_sep", 10);
  nssm.min_points = p.require<int>("nssm/min_points");
  nssm.min_overlap_ratio = p.get<double>("nssm/min_overlap_ratio", 0.0);
  nssm.max_translation = p.require<double>("nssm/max_translation");
  nssm.max_rotation = p.require<double>("nssm/max_rotation");
  nssm.source_frames = p.require<int>("nssm/source_frames");
  nssm.fan_drift_trans = p.get<double>("nssm/fan_drift_trans", 0.01);
  nssm.fan_drift_rot = p.get<double>("nssm/fan_drift_rot", 0.0017);
  nssm.initialization = p.get<bool>("nssm/initialization", true);
  const auto nssm_init = p.get<std::vector<double>>(
    "nssm/initialization_params", {100.0, 5.0, 0.01});
  nssm.init_n = static_cast<int>(nssm_init.at(0));
  nssm.init_iters = static_cast<int>(nssm_init.at(1));
  nssm.init_ftol = nssm_init.at(2);
  nssm.cov_samples = p.require<int>("nssm/cov_samples");
  nssm.cov_method =
    cov_method(p.get<std::string>("nssm/cov_method", "sampled"));
  slam.pcm_queue_size = p.require<int>("pcm_queue_size");
  slam.min_pcm = p.require<int>("min_pcm");
  slam.nssm_max_sigma = p.get<double>("nssm/max_sigma", 0.5);
  slam.nssm_max_anisotropy = p.get<double>("nssm/max_anisotropy", 8.0);
  slam.nssm_degeneracy_prefloor =
    p.get<bool>("nssm/degeneracy_prefloor", false);
  slam.nssm_max_yaw_vs_compass =
    p.get<double>("nssm/max_yaw_vs_compass", 0.0);
  slam.nssm_max_translation_vs_dr =
    p.get<double>("nssm/max_translation_vs_dr", 0.0);
  slam.post_loop_max_yaw_rms = p.get<double>("post_loop_max_yaw_rms", 0.0);
  slam.post_loop_max_link_error_sigma =
    p.get<double>("post_loop_max_link_error_sigma", 3.0);
  slam.nssm_use_dcs = p.get<bool>("nssm/use_dcs", false);
  slam.nssm_dcs_phi = p.get<double>("nssm/dcs_phi", 1.0);
  slam.loop_extra_iterations = p.get<int>("loop_extra_iterations", 3);

  slam.icp.load_from_yaml(icp_yaml);
  slam.configure();
}

inline void deployed_profile(Slam& slam)
{
  const ProfilePaths paths = locate_profile();
  apply_profile(slam, ParamReader(paths.slam_yaml), paths.icp_yaml);
}

// ------------------------------------------------------------- scenes

// Two long parallel walls and nothing else: translation along the corridor
// is unobservable, the geometry the degeneracy gate exists to reject.
inline SyntheticWorld corridor(double length, double width)
{
  SyntheticWorld world;
  world.walls.push_back({-length / 2.0, 0.0, length / 2.0, 0.0});
  world.walls.push_back({-length / 2.0, width, length / 2.0, width});
  return world;
}

struct SonarModel
{
  double aperture = 130.0 * M_PI / 180.0;
  int beams = 256;
  double max_range = 10.0;
  double noise = 0.02;  // m, radial-sample scale at HF
};

inline Matrix scan(const SyntheticWorld& world, const gtsam::Pose2& truth,
                   const SonarModel& sonar, std::mt19937& rng)
{
  return world.scan_cloud(truth, sonar.aperture, sonar.beams,
                          sonar.max_range, sonar.noise, rng);
}

inline KeyframePtr make_keyframe(int index, const gtsam::Pose2& dr,
                                 Matrix points = Matrix::Zero(0, 3))
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 1000 + index;
  const gtsam::Pose3 dr3(gtsam::Rot3::Yaw(dr.theta()),
                         gtsam::Point3(dr.x(), dr.y(), 1.0));
  auto frame = std::make_shared<Keyframe>(true, stamp, dr3);
  frame->points = std::move(points);
  return frame;
}

// Seed a new keyframe's graph estimate the way slam_node does: the previous
// estimate advanced by the dead-reckoning increment.
inline void predict_from_dr(const Slam& slam, const KeyframePtr& frame)
{
  if (slam.keyframes.empty()) return;
  const auto& previous = slam.current_keyframe();
  frame->update(previous->pose.compose(previous->dr_pose.between(frame->dr_pose)));
}

// ------------------------------------------------ trajectories and drift

// Straight-line poses from `from` to `to` at `step` spacing (heading along
// the direction of travel), excluding `from`.
inline void extend_leg(std::vector<gtsam::Pose2>& path, const gtsam::Point2& to,
                       double step)
{
  const gtsam::Pose2 from = path.back();
  const double dx = to.x() - from.x();
  const double dy = to.y() - from.y();
  const double length = std::hypot(dx, dy);
  const int steps = std::max(1, static_cast<int>(std::lround(length / step)));
  const double heading = std::atan2(dy, dx);
  for (int i = 1; i <= steps; ++i) {
    const double f = static_cast<double>(i) / steps;
    path.emplace_back(from.x() + f * dx, from.y() + f * dy, heading);
  }
}

// Dead reckoning: truth increments with per-step noise and a heading bias,
// integrated from the true start. A DVL/AHRS chain drifts smoothly this way.
struct DriftModel
{
  double sigma_xy = 0.01;          // m per step
  double sigma_yaw = 0.002;        // rad per step
  double yaw_bias_per_step = 0.0;  // rad per step
};

inline std::vector<gtsam::Pose2> dead_reckon(
  const std::vector<gtsam::Pose2>& truth, const DriftModel& drift,
  std::mt19937& rng)
{
  std::normal_distribution<double> xy(0.0, drift.sigma_xy);
  std::normal_distribution<double> yaw(0.0, drift.sigma_yaw);
  std::vector<gtsam::Pose2> dr{truth.front()};
  for (std::size_t i = 1; i < truth.size(); ++i) {
    const gtsam::Pose2 increment = truth[i - 1].between(truth[i]);
    const gtsam::Pose2 noisy(increment.x() + xy(rng), increment.y() + xy(rng),
                             increment.theta() + yaw(rng) +
                               drift.yaw_bias_per_step);
    dr.push_back(dr.back().compose(noisy));
  }
  return dr;
}

// Odometry-only graph over a DR chain: every keyframe carries the ISAM2
// marginal it had when it was newest, which is its current marginal while
// the graph holds no loop closure.
inline void build_odometry_chain(Slam& slam,
                                 const std::vector<gtsam::Pose2>& dr)
{
  for (std::size_t k = 0; k < dr.size(); ++k) {
    auto frame = make_keyframe(static_cast<int>(k), dr[k]);
    predict_from_dr(slam, frame);
    if (slam.keyframes.empty())
      slam.add_prior(frame);
    else
      slam.add_odometry(frame);
    if (!slam.update_factor_graph(frame))
      throw std::runtime_error("odometry chain update failed: " +
                               slam.last_error());
  }
}

// A loop-closure candidate as the NSSM pipeline would queue it: graph poses
// of both keyframes, a target->source measurement and its covariance.
inline ICPResult make_closure(const Slam& slam, int target, int source,
                              const gtsam::Pose2& measured,
                              const Eigen::Matrix3d& cov)
{
  InitializationResult init;
  init.target_key = target;
  init.source_key = source;
  init.target_pose = slam.keyframes.at(static_cast<std::size_t>(target))->pose;
  init.source_pose = slam.keyframes.at(static_cast<std::size_t>(source))->pose;
  ICPResult closure(init, false);
  closure.has_estimated_transform = true;
  closure.estimated_transform = measured;
  closure.has_cov = true;
  closure.cov = cov;
  closure.cov_raw = cov;
  return closure;
}

inline gtsam::Pose2 perturb(const gtsam::Pose2& pose, double sigma_xy,
                            double sigma_yaw, std::mt19937& rng)
{
  std::normal_distribution<double> xy(0.0, sigma_xy);
  std::normal_distribution<double> yaw(0.0, sigma_yaw);
  return pose.compose(gtsam::Pose2(xy(rng), xy(rng), yaw(rng)));
}

}  // namespace sonar_slam::harness
