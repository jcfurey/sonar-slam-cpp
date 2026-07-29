// Synthetic rectangular-pool point-cloud fixture for the SLAM back-end test.
// Deterministic: callers own the RNG.
#pragma once

#include <gtsam/geometry/Pose2.h>

#include <cmath>
#include <random>
#include <vector>

#include "sonar_slam_cpp/cloud_ops.hpp"

namespace sonar_slam {

struct SyntheticWorld
{
  // wall segments (x1, y1, x2, y2), meters
  std::vector<Eigen::Vector4d> walls;

  static SyntheticWorld pool(double w, double h)
  {
    SyntheticWorld sw;
    sw.walls.push_back({0.0, 0.0, w, 0.0});
    sw.walls.push_back({w, 0.0, w, h});
    sw.walls.push_back({w, h, 0.0, h});
    sw.walls.push_back({0.0, h, 0.0, 0.0});
    // Interior posts: perfectly smooth straight walls are DEGENERATE for
    // registration (translation along the wall is unobservable, ICP slides)
    // — a failure mode real venues don't have. The posts give the world the
    // corner structure reality provides.
    const double px[6] = {5.0, 9.0, 13.0, 16.0, 4.0, 11.0};
    const double py[6] = {3.0, 7.0, 4.0, 6.5, 7.5, 2.5};
    for (int i = 0; i < 6; ++i) sw.add_box(px[i], py[i], 0.4);
    return sw;
  }

  void add_box(double cx, double cy, double size)
  {
    const double s = size / 2.0;
    walls.push_back({cx - s, cy - s, cx + s, cy - s});
    walls.push_back({cx + s, cy - s, cx + s, cy + s});
    walls.push_back({cx + s, cy + s, cx - s, cy + s});
    walls.push_back({cx - s, cy + s, cx - s, cy - s});
  }

  // first wall hit of the ray from (x, y) along phi; < 0 when nothing within
  // max_range
  double raycast(double x, double y, double phi, double max_range) const
  {
    const double dx = std::cos(phi), dy = std::sin(phi);
    double best = -1.0;
    for (const auto& w : walls) {
      const double sx = w[2] - w[0], sy = w[3] - w[1];
      const double den = dx * sy - dy * sx;
      if (std::fabs(den) < 1e-12) continue;
      const double qx = w[0] - x, qy = w[1] - y;
      const double t = (qx * sy - qy * sx) / den;   // along the ray
      const double u = (qx * dy - qy * dx) / den;   // along the segment
      if (t <= 1e-6 || t > max_range || u < 0.0 || u > 1.0) continue;
      if (best < 0.0 || t < best) best = t;
    }
    return best;
  }

  // Planar candidate cloud in the VEHICLE frame at a (ground-truth) pose:
  // one point per beam that hits a wall within range, with isotropic noise.
  // Layout matches Keyframe::points (N x 3, col2 = elevation = 0).
  Matrix scan_cloud(const gtsam::Pose2& pose, double aperture, int beams,
                    double max_range, double noise_std,
                    std::mt19937& rng) const
  {
    std::normal_distribution<double> noise(0.0, noise_std);
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(static_cast<std::size_t>(beams));
    for (int b = 0; b < beams; ++b) {
      const double bearing =
        -aperture / 2.0 + aperture * (b + 0.5) / static_cast<double>(beams);
      const double d =
        raycast(pose.x(), pose.y(), pose.theta() + bearing, max_range);
      if (d < 0.0) continue;
      pts.emplace_back(d * std::cos(bearing) + noise(rng),
                       d * std::sin(bearing) + noise(rng));
    }
    Matrix m(static_cast<long>(pts.size()), 3);
    for (long i = 0; i < m.rows(); ++i) {
      m(i, 0) = static_cast<float>(pts[static_cast<std::size_t>(i)].x());
      m(i, 1) = static_cast<float>(pts[static_cast<std::size_t>(i)].y());
      m(i, 2) = 0.0f;
    }
    return m;
  }

};

}  // namespace sonar_slam
