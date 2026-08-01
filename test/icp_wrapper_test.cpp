// The ICP wrapper's config introspection and identity-registration sanity.
// error_minimizer_name() feeds Slam::configure's decision on whether the
// Censi point-to-point Hessian matches the running objective — a wrong
// answer silently mispairs covariance model and minimizer.
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
  // ---- a missing YAML falls back to the libpointmatcher default chain,
  // which is point-to-point — and says so
  sonar_slam::ICP icp;
  icp.load_from_yaml("/nonexistent/definitely_missing_icp.yaml");
  const std::string name = icp.error_minimizer_name();
  CHECK(name.find("PointToPoint") != std::string::npos,
        "default chain reported '%s', expected a PointToPoint minimizer",
        name.c_str());
  std::printf("[1] default minimizer: %s\n", name.c_str());

  // ---- packaged config loads and reports ITS minimizer
  if (const char* cfg = std::getenv("ICP_TEST_CONFIG")) {
    icp.load_from_yaml(cfg);
    std::printf("[1b] packaged minimizer: %s\n",
                icp.error_minimizer_name().c_str());
    CHECK(!icp.error_minimizer_name().empty(),
          "packaged config reports no minimizer");
  }

  // ---- registering a realistic cloud against itself converges to identity
  // (an L-shaped wall, 120 points: enough structure for the chain's
  // filters; tiny clouds are legitimately degenerate for the default chain)
  Matrix cloud(120, 2);
  for (int i = 0; i < 60; ++i) {
    cloud(i, 0) = 0.05f * i;      // wall along x
    cloud(i, 1) = 0.0f;
    cloud(60 + i, 0) = 0.0f;      // wall along y
    cloud(60 + i, 1) = 0.05f * i;
  }
  const auto [status, T] =
    icp.compute(cloud, cloud, Eigen::Matrix3f::Identity());
  CHECK(status == "success", "self-registration failed: %s", status.c_str());
  CHECK((T - Eigen::Matrix3f::Identity()).norm() < 1e-3,
        "self-registration drifted (|T - I| = %.6f)",
        (T - Eigen::Matrix3f::Identity()).norm());
  std::printf("[2] self-registration is identity\n");

  // ---- a known translation is recovered, pinning the CONVENTION:
  // compute(source, target) returns T with target ~= T * source, so a
  // source displaced +0.3 in x registers with T carrying -0.3
  Matrix shifted = cloud;
  shifted.col(0).array() += 0.3f;
  Eigen::Matrix3f guess = Eigen::Matrix3f::Identity();
  const auto [status2, T2] = icp.compute(shifted, cloud, guess);
  CHECK(status2 == "success", "shift registration failed: %s",
        status2.c_str());
  CHECK(std::fabs(T2(0, 2) + 0.3f) < 0.02f && std::fabs(T2(1, 2)) < 0.02f,
        "recovered translation (%.3f, %.3f), expected (-0.3, 0) under the "
        "target = T * source convention", T2(0, 2), T2(1, 2));
  std::printf("[3] known shift recovered (source->target convention)\n");

  std::printf("PASS\n");
  return 0;
}
