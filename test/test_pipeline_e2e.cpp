// End-to-end pipeline test on a synthetic rectangular pool: no bags, no DDS,
// no proprietary data — pure algorithmic coverage of the paths a field run
// exercises.
//  [1] CFAR detects the wall returns in a synthetic polar image.
//  [2] A drifted perimeter loop through the SLAM core (SSM registration,
//      NSSM loop closure through every defense gate, ISAM2) ends with a
//      loop closure accepted and the trajectory error reduced vs raw DR.
//  [3] The map survives save -> load, and a new session relocalizes into it
//      by global scan match within tolerance.
//  [4] A USBL-style absolute position prior pulls the estimate toward truth.
#include <cmath>
#include <cstdio>

#include "sonar_slam_cpp/cfar.hpp"
#include "sonar_slam_cpp/slam_core.hpp"
#include "sonar_slam_cpp/synthetic_world.hpp"

using sonar_slam::CFAR;
using sonar_slam::Keyframe;
using sonar_slam::KeyframePtr;
using sonar_slam::Matrix;
using sonar_slam::Slam;
using sonar_slam::SyntheticWorld;

#define CHECK(cond, ...)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL(%d): ", __LINE__);                                   \
      std::printf(__VA_ARGS__);                                              \
      std::printf("\n");                                                     \
      return 1;                                                              \
    }                                                                        \
  } while (0)

namespace {

constexpr double kAperture = 130.0 * M_PI / 180.0;
constexpr double kMaxRange = 25.0;
constexpr int kBeams = 256;

void configure_slam(Slam& slam)
{
  slam.prior_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);
  slam.odom_sigmas = Eigen::Vector3d(0.2, 0.2, 0.02);
  slam.icp_odom_sigmas = Eigen::Vector3d(0.1, 0.1, 0.01);
  slam.point_resolution = 0.3;
  slam.point_noise = 0.5;

  slam.ssm_params.enable = true;
  slam.ssm_params.min_points = 30;
  slam.ssm_params.max_translation = 3.0;
  slam.ssm_params.max_rotation = 0.5;
  slam.ssm_params.target_frames = 3;
  slam.ssm_params.initialization = true;
  slam.ssm_params.init_n = 50;
  slam.ssm_params.init_iters = 1;
  slam.ssm_params.init_ftol = 0.01;
  slam.ssm_params.cov_samples = 0;

  slam.nssm_params.enable = true;
  slam.nssm_params.min_st_sep = 10;
  slam.nssm_params.min_points = 30;
  slam.nssm_params.max_translation = 6.0;
  slam.nssm_params.max_rotation = 1.0;
  slam.nssm_params.source_frames = 5;
  slam.nssm_params.initialization = true;
  slam.nssm_params.init_n = 100;
  slam.nssm_params.init_iters = 3;
  slam.nssm_params.init_ftol = 0.01;
  slam.nssm_params.cov_samples = 0;

  slam.pcm_queue_size = 5;
  slam.min_pcm = 2;
  slam.icp.load_from_yaml(std::string(TEST_SOURCE_DIR) + "/config/icp.yaml");
  slam.configure();
}

KeyframePtr make_frame(int k, const gtsam::Pose2& dr)
{
  builtin_interfaces::msg::Time t;
  t.sec = k;
  const gtsam::Pose3 dr3(gtsam::Rot3::Yaw(dr.theta()),
                         gtsam::Point3(dr.x(), dr.y(), 1.0));
  return std::make_shared<Keyframe>(true, t, dr3);
}

// one slam_callback-equivalent step; returns false on a dropped frame
bool feed(Slam& slam, int k, const gtsam::Pose2& truth, const gtsam::Pose2& dr,
          const SyntheticWorld& world, std::mt19937& rng)
{
  auto frame = make_frame(k, dr);
  frame->points = world.scan_cloud(truth, kAperture, kBeams, kMaxRange, 0.03, rng);
  if (!slam.keyframes.empty())
    frame->update(slam.current_keyframe()->pose.compose(
      slam.current_keyframe()->dr_pose.between(frame->dr_pose)));
  try {
    if (slam.keyframes.empty())
      slam.add_prior(frame);
    else
      slam.add_sequential_scan_matching(frame);
    if (!slam.update_factor_graph(frame)) return false;
    slam.current_frame = frame;  // NSSM anchors its search on it
    if (slam.nssm_params.enable && slam.add_nonsequential_scan_matching())
      if (!slam.update_factor_graph()) return false;
  } catch (const std::exception& e) {
    std::printf("  (frame %d threw: %s)\n", k, e.what());
    return false;
  }
  return true;
}

}  // namespace

int main()
{
  const SyntheticWorld world = SyntheticWorld::pool(20.0, 10.0);
  std::mt19937 rng(42);

  // ---- [1] CFAR on a synthetic polar image --------------------------------
  {
    std::vector<int> wall_rows;
    const cv::Mat img = world.polar_image(gtsam::Pose2(10.0, 5.0, 0.0),
                                          kAperture, 128, 200, kMaxRange, rng,
                                          &wall_rows);
    const CFAR cfar(40, 10, 1e-2);
    const cv::Mat mask = cfar.detect(img, CFAR::CA, 60.0f);
    int cols_with_wall = 0, cols_detected = 0;
    long stray = 0;
    for (int b = 0; b < img.cols; ++b) {
      const int wr = wall_rows[static_cast<std::size_t>(b)];
      if (wr < 0) continue;
      ++cols_with_wall;
      bool hit = false;
      for (int r = 0; r < img.rows; ++r) {
        if (!mask.at<std::uint8_t>(r, b)) continue;
        if (std::abs(r - wr) <= 3)
          hit = true;
        else
          ++stray;
      }
      if (hit) ++cols_detected;
    }
    std::printf("[1] CFAR: %d/%d wall beams detected, %ld stray detections\n",
                cols_detected, cols_with_wall, stray);
    CHECK(cols_with_wall > 60, "fixture broken: only %d wall beams",
          cols_with_wall);
    CHECK(cols_detected > cols_with_wall * 7 / 10,
          "CFAR missed the walls (%d/%d)", cols_detected, cols_with_wall);
    CHECK(stray < static_cast<long>(cols_with_wall) * 2,
          "CFAR false-alarm rate implausible (%ld stray)", stray);
  }

  // ---- [2] drifted perimeter loop: SSM + NSSM close it --------------------
  Slam slam;
  configure_slam(slam);

  // perimeter waypoints, revisiting the first leg to offer loop closures
  struct Leg { double x0, y0, x1, y1, yaw; };
  const std::vector<Leg> legs = {
    {2, 2, 18, 2, 0.0},
    {18, 2, 18, 8, M_PI / 2},
    {18, 8, 2, 8, M_PI},
    {2, 8, 2, 2, -M_PI / 2},
    {2, 2, 14, 2, 0.0},
  };
  std::vector<gtsam::Pose2> truths;
  for (const auto& l : legs) {
    const double len = std::hypot(l.x1 - l.x0, l.y1 - l.y0);
    const int steps = static_cast<int>(len / 0.75);
    for (int i = 0; i < steps; ++i) {
      const double f = static_cast<double>(i) / steps;
      truths.emplace_back(l.x0 + f * (l.x1 - l.x0), l.y0 + f * (l.y1 - l.y0),
                          l.yaw);
    }
  }

  // DR: compass-anchored yaw (true), slowly drifting translation
  double drift_x = 0.0, drift_y = 0.0;
  int fed = 0, dropped = 0;
  std::vector<gtsam::Pose2> drs;
  for (std::size_t k = 0; k < truths.size(); ++k) {
    drift_x += 0.004;
    drift_y += 0.008;
    const gtsam::Pose2 dr(truths[k].x() + drift_x, truths[k].y() + drift_y,
                          truths[k].theta());
    drs.push_back(dr);
    if (feed(slam, static_cast<int>(k), truths[k], dr, world, rng))
      ++fed;
    else
      ++dropped;
  }
  CHECK(dropped * 10 < fed, "too many dropped frames (%d/%d)", dropped, fed);

  const int last = slam.current_key() - 1;
  const gtsam::Pose2 truth_last = truths[truths.size() - 1];
  const double dr_err = std::hypot(drs.back().x() - truth_last.x(),
                                   drs.back().y() - truth_last.y());
  const double slam_err =
    std::hypot(slam.keyframes[last]->pose.x() - truth_last.x(),
               slam.keyframes[last]->pose.y() - truth_last.y());
  std::printf("[2] loop: %d keyframes, ssm %d, nssm accepted %d (reverted "
              "%d); final err slam %.2f m vs dr %.2f m\n",
              slam.current_key(), slam.ssm_accepted, slam.nssm_accepted,
              slam.nssm_reverted, slam_err, dr_err);
  std::printf("    nssm rejects: [%s]\n", slam.nssm_reject_summary().c_str());
  CHECK(dr_err > 0.5, "fixture broken: DR did not drift (%.2f m)", dr_err);
  CHECK(slam.nssm_accepted >= 1, "no loop closure was ever accepted");
  CHECK(slam_err < dr_err * 0.7,
        "loop closure did not meaningfully correct (slam %.2f vs dr %.2f)",
        slam_err, dr_err);

  // ---- [3] persistence roundtrip + relocalization --------------------------
  const std::string map_path = "test_pipeline_e2e_map.ssm";
  CHECK(slam.save_map(map_path), "save_map failed: %s",
        slam.last_error().c_str());
  Slam s2;
  configure_slam(s2);
  CHECK(s2.load_map(map_path), "load_map failed: %s", s2.last_error().c_str());
  CHECK(s2.loaded_keyframes() == slam.current_key(),
        "loaded %d of %d keyframes", s2.loaded_keyframes(), slam.current_key());
  CHECK(s2.awaiting_relocalization(), "load did not arm relocalization");

  const gtsam::Pose2 reloc_truth(6.0, 2.0, 0.0);
  auto frame2 = make_frame(1000, gtsam::Pose2(0.0, 0.0, 0.0));
  frame2->points =
    world.scan_cloud(reloc_truth, kAperture, kBeams, kMaxRange, 0.03, rng);
  CHECK(s2.relocalize(frame2), "relocalize failed: %s",
        s2.last_error().c_str());
  const double reloc_err =
    std::hypot(frame2->pose.x() - reloc_truth.x(),
               frame2->pose.y() - reloc_truth.y());
  std::printf("[3] persistence: %d keyframes reloaded; relocalized %.2f m "
              "from truth\n",
              s2.loaded_keyframes(), reloc_err);
  CHECK(reloc_err < 0.8, "relocalization landed %.2f m from truth", reloc_err);
  std::remove(map_path.c_str());

  // ---- [4] USBL-style absolute position prior ------------------------------
  const double before =
    std::hypot(slam.keyframes[last]->pose.x() - truth_last.x(),
               slam.keyframes[last]->pose.y() - truth_last.y());
  CHECK(slam.add_position_prior(last, truth_last.x(), truth_last.y(), 0.2),
        "add_position_prior failed: %s", slam.last_error().c_str());
  const double after =
    std::hypot(slam.keyframes[last]->pose.x() - truth_last.x(),
               slam.keyframes[last]->pose.y() - truth_last.y());
  std::printf("[4] position prior: newest keyframe error %.3f -> %.3f m\n",
              before, after);
  CHECK(after <= before + 1e-6, "position prior made the estimate worse");
  CHECK(slam.position_priors_applied == 1, "prior counter wrong");

  std::printf("PASS\n");
  return 0;
}
