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

  std::printf("PASS\n");
  return 0;
}
