// Empirical checks of the cloud_ops primitives every scan admission and
// registration path consumes (downsample, descriptor carry-through, radius
// outlier removal, KNN match). These ran untested: a silent behavior change
// here shifts admission counts and match sets everywhere downstream.
#include <cmath>
#include <cstdio>
#include <set>

#include "sonar_slam_cpp/cloud_ops.hpp"

using sonar_slam::Matrix;

#define CHECK(cond, ...)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("FAIL(%d): ", __LINE__);                                 \
      std::printf(__VA_ARGS__);                                            \
      std::printf("\n");                                                   \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main()
{
  // ---- downsample: one representative per occupied cell, and every input
  // point stays within a cell diagonal of some representative
  {
    Matrix in(6, 3);
    in << 0.01f, 0.01f, 0.0f,   // cell A
          0.04f, 0.02f, 0.0f,   // cell A again
          1.00f, 0.00f, 0.0f,   // cell B
          1.02f, 0.03f, 0.0f,   // cell B again
          0.00f, 1.00f, 0.0f,   // cell C
          3.00f, 3.00f, 0.0f;   // cell D
    const float res = 0.1f;
    const Matrix out = sonar_slam::downsample(in, res);
    CHECK(out.rows() == 4, "expected 4 occupied cells, got %ld",
          (long)out.rows());
    const float diag = res * std::sqrt(3.0f);
    for (int i = 0; i < in.rows(); ++i) {
      float best = 1e9f;
      for (int j = 0; j < out.rows(); ++j)
        best = std::min(best,
                        (in.row(i) - out.row(j)).norm());
      CHECK(best <= diag,
            "input %d is %.3f m from every representative", i, best);
    }
    std::printf("[1] downsample: %ld cells, max gap bounded\n",
                (long)out.rows());
  }

  // ---- downsample with descriptors: the carried descriptor belongs to a
  // point of the SAME cell (here: descriptor == x rounded to its cell id)
  {
    Matrix in(4, 3);
    in << 0.01f, 0.0f, 0.0f,
          0.03f, 0.0f, 0.0f,
          2.00f, 0.0f, 0.0f,
          2.04f, 0.0f, 0.0f;
    Matrix desc(4, 1);
    desc << 0.0f, 0.0f, 2.0f, 2.0f;   // cell id along x
    const auto [pts, carried] = sonar_slam::downsample(in, desc, 0.1f);
    CHECK(pts.rows() == 2 && carried.rows() == 2,
          "expected 2 representatives, got %ld/%ld", (long)pts.rows(),
          (long)carried.rows());
    for (int j = 0; j < pts.rows(); ++j) {
      const float expect = pts(j, 0) < 1.0f ? 0.0f : 2.0f;
      CHECK(carried(j, 0) == expect,
            "descriptor crossed cells: rep x=%.2f carries %.1f", pts(j, 0),
            carried(j, 0));
    }
    std::printf("[2] descriptor carry-through OK\n");
  }

  // ---- remove_outlier: an isolated point goes, a dense cluster stays
  {
    Matrix in(5, 3);
    in << 0.0f, 0.0f, 0.0f,
          0.05f, 0.0f, 0.0f,
          0.0f, 0.05f, 0.0f,
          0.05f, 0.05f, 0.0f,
          5.0f, 5.0f, 0.0f;      // loner
    const Matrix out = sonar_slam::remove_outlier(in, 0.2, 2);
    CHECK(out.rows() == 4, "expected the loner removed, got %ld rows",
          (long)out.rows());
    for (int i = 0; i < out.rows(); ++i)
      CHECK(out(i, 0) < 1.0f, "the loner survived at row %d", i);
    std::printf("[3] radius outlier removal OK\n");
  }

  // ---- match: nearest ids within max_dist, -1 beyond it, squared dists
  {
    Matrix ref(3, 2);
    ref << 0.0f, 0.0f,
           1.0f, 0.0f,
           0.0f, 1.0f;
    Matrix in(3, 2);
    in << 0.1f, 0.0f,     // nearest ref 0 at 0.1
          0.9f, 0.0f,     // nearest ref 1 at 0.1
          10.0f, 10.0f;   // nothing within reach
    const auto [ids, d2] = sonar_slam::match(ref, in, 1, 0.5f);
    CHECK(ids.cols() == 3 && ids.rows() == 1, "unexpected match shape");
    CHECK(ids(0, 0) == 0 && ids(0, 1) == 1,
          "wrong nearest ids: %d %d", ids(0, 0), ids(0, 1));
    CHECK(ids(0, 2) == -1, "far point matched to %d", ids(0, 2));
    CHECK(std::fabs(d2(0, 0) - 0.01f) < 1e-5f &&
            std::fabs(d2(0, 1) - 0.01f) < 1e-5f,
          "squared distances wrong: %.4f %.4f", d2(0, 0), d2(0, 1));
    std::printf("[4] knn match OK\n");
  }

  // ---- constrained XYZ ICP: elevation participates in correspondence
  // selection, while the recovered motion remains exactly x/y/yaw.
  {
    Matrix target(160, 3);
    for (int i = 0; i < 80; ++i) {
      const float z = 0.025f * static_cast<float>(i);
      target.row(i) << 0.0f, 0.03f * static_cast<float>(i), z;
      target.row(80 + i) << 0.03f * static_cast<float>(i), 0.0f, z;
    }
    const double yaw = 0.08;
    const float c = static_cast<float>(std::cos(yaw));
    const float s = static_cast<float>(std::sin(yaw));
    Matrix source = target;
    source.col(0) = c * target.col(0) - s * target.col(1);
    source.col(1) = s * target.col(0) + c * target.col(1);
    source.col(0).array() += 0.25f;
    source.col(1).array() -= 0.12f;

    sonar_slam::ConstrainedIcpParams p;
    p.max_correspondence = 0.6f;
    p.translation_epsilon = 1e-5;
    p.rotation_epsilon = 1e-5;
    const auto fit = sonar_slam::constrained_icp_xyz(
      source, target, Eigen::Matrix3f::Identity(), p);
    CHECK(fit.success, "constrained XYZ ICP failed: %s", fit.message.c_str());
    // source = A*target+t, so target ~= A^-1*(source-t)
    const double expect_yaw = -yaw;
    const double expect_x = -(std::cos(expect_yaw) * 0.25 -
                              std::sin(expect_yaw) * -0.12);
    const double expect_y = -(std::sin(expect_yaw) * 0.25 +
                              std::cos(expect_yaw) * -0.12);
    const double got_yaw = std::atan2(fit.T(1, 0), fit.T(0, 0));
    CHECK(std::fabs(fit.T(0, 2) - expect_x) < 0.04 &&
            std::fabs(fit.T(1, 2) - expect_y) < 0.04 &&
            std::fabs(got_yaw - expect_yaw) < 0.04,
          "constrained fit (%.3f, %.3f, %.3f), expected (%.3f, %.3f, %.3f)",
          fit.T(0, 2), fit.T(1, 2), got_yaw,
          expect_x, expect_y, expect_yaw);

    Matrix wrong_height = source;
    wrong_height.col(2).array() += 2.0f;
    p.max_correspondence = 0.4f;
    const auto rejected = sonar_slam::constrained_icp_xyz(
      wrong_height, target, Eigen::Matrix3f::Identity(), p);
    CHECK(!rejected.success,
          "height-separated scans falsely registered through XY projection");
    std::printf("[5] constrained XYZ registration and height rejection OK\n");
  }

  std::printf("PASS\n");
  return 0;
}
