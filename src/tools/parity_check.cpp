// CPU/GPU parity and performance self-check. Runs every CUDA kernel against
// its CPU twin on synthetic sonar-like data and reports max mismatch + timing.
// With no GPU (or SONAR_SLAM_FORCE_CPU=1) it degrades to a CPU smoke test.
//
// Exit codes:
//   0  every comparison passed (count reported)
//   1  a comparison FAILED
//   2  nothing was compared — CPU-only build or no visible device. NOT a pass.
//
// --smoke-ok downgrades 2 to 0 while still printing the NOT-CHECKED notice.
// It exists for the registered `test_runtime_parity` ament test, which must
// stay green in the configurations where parity is simply not observable
// (the package's own CI builds -DSONAR_SLAM_ENABLE_CUDA=OFF, and a CUDA build
// inside a GPU-less container is just as common) while still going red on a
// real mismatch. Operators running the tool by hand get the strict contract:
// a green light from a tool that compared nothing is worse than no answer.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "sonar_slam_cpp/cfar.hpp"
#include "sonar_slam_cpp/gpu.hpp"

using Clock = std::chrono::steady_clock;

namespace {

// no intensity gate in the parity harness — keep every CFAR hit
constexpr float kNoGate = -std::numeric_limits<float>::infinity();

double ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv)
{
  bool smoke_ok = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--smoke-ok") {
      smoke_ok = true;
    } else {
      std::fprintf(stderr,
                   "parity_check: unknown argument '%s'\n"
                   "usage: parity_check [--smoke-ok]\n",
                   argv[i]);
      return 1;
    }
  }

  const int rows = 716, cols = 512;  // typical Oculus polar image size
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.f, 80.f);

  cv::Mat img(rows, cols, CV_32FC1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) img.at<float>(r, c) = dist(rng);
  // sprinkle strong targets
  for (int k = 0; k < 200; ++k)
    img.at<float>(rng() % rows, rng() % cols) = 255.f;

  const bool gpu = sonar_slam::gpu::available();
  std::printf("GPU available: %s\n", gpu ? "yes" : "no (CPU-only run)");

  sonar_slam::CFAR cfar(40, 10, 1e-2, 10);
  std::printf("threshold factors: CA %.3f SOCA %.3f GOCA %.3f OS %.3f\n",
              cfar.threshold_factor(sonar_slam::CFAR::CA),
              cfar.threshold_factor(sonar_slam::CFAR::SOCA),
              cfar.threshold_factor(sonar_slam::CFAR::GOCA),
              cfar.threshold_factor(sonar_slam::CFAR::OS));

  int failures = 0;
  // Every comparison below is gated on BOTH `gpu` and SONAR_SLAM_WITH_CUDA.
  // Without them nothing runs, and reporting the failure count alone made a
  // CPU-only build (the package's own CI uses -DSONAR_SLAM_ENABLE_CUDA=OFF)
  // print "parity OK" and exit 0 having compared nothing — a green light from
  // a tool whose entire job is certifying GPU/CPU equivalence. Count what was
  // actually checked and say so.
  int compared = 0;

  // ------------------------------------------------------------------ CFAR
  for (const auto alg : {sonar_slam::CFAR::CA, sonar_slam::CFAR::SOCA,
                         sonar_slam::CFAR::GOCA, sonar_slam::CFAR::OS}) {
    cv::Mat cpu_mask(rows, cols, CV_8UC1);
    auto t0 = Clock::now();
    sonar_slam::CFAR::detect_cpu(img.ptr<float>(), rows, cols,
                                 static_cast<int>(alg), 20, 5, 10,
                                 cfar.threshold_factor(alg), kNoGate,
                                 cpu_mask.ptr<std::uint8_t>());
    const double cpu_ms = ms_since(t0);

    if (gpu) {
#ifdef SONAR_SLAM_WITH_CUDA
      cv::Mat gpu_mask(rows, cols, CV_8UC1);
      t0 = Clock::now();
      const bool ok = sonar_slam::gpu::cfar_cuda(
        img.ptr<float>(), rows, cols, static_cast<int>(alg), 20, 5, 10,
        cfar.threshold_factor(alg), kNoGate, gpu_mask.ptr<std::uint8_t>());
      const double gpu_ms = ms_since(t0);
      if (!ok) {
        std::printf("CFAR alg %d: GPU path failed\n", static_cast<int>(alg));
        ++failures;
        continue;
      }
      const int diff = cv::countNonZero(cpu_mask != gpu_mask);
      ++compared;
      std::printf("CFAR alg %d: cpu %.2f ms, gpu %.2f ms, mismatched px %d\n",
                  static_cast<int>(alg), cpu_ms, gpu_ms, diff);
      // float summation order can flip a borderline comparison on rare pixels
      if (diff > rows * cols / 10000) ++failures;
#endif
    } else {
      std::printf("CFAR alg %d: cpu %.2f ms (%d detections)\n",
                  static_cast<int>(alg), cpu_ms, cv::countNonZero(cpu_mask));
    }
  }

#ifdef SONAR_SLAM_WITH_CUDA
  // ----------------------------------------------------------------- remap
  if (gpu) {
    // grayscale, NOT binary: on a {0,1} image `diff > 1` can never fire and
    // the check was vacuous — a broken GPU remap stayed "parity OK"
    cv::Mat mask(rows, cols, CV_8UC1);
    cv::randu(mask, 0, 256);
    cv::Mat map_x(rows, cols, CV_32FC1), map_y(rows, cols, CV_32FC1);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        map_x.at<float>(r, c) = static_cast<float>(c) * 0.98f + 1.3f;
        map_y.at<float>(r, c) = static_cast<float>(r) * 1.01f - 0.7f;
      }

    cv::Mat cpu_dst;
    cv::remap(mask, cpu_dst, map_x, map_y, cv::INTER_LINEAR);
    cv::Mat gpu_dst(rows, cols, CV_8UC1);
    if (!sonar_slam::gpu::remap_u8_cuda(
          mask.ptr<std::uint8_t>(), rows, cols, map_x.ptr<float>(),
          map_y.ptr<float>(), rows, cols, 1, /*map_version=*/-1,
          gpu_dst.ptr<std::uint8_t>())) {
      std::printf("remap linear: GPU path failed\n");
      ++failures;
    } else {
      // The CUDA path reproduces OpenCV's 5-bit coordinate quantization and
      // fixed-point weights, so uint8 output must be bit-exact.
      cv::Mat diff;
      cv::absdiff(cpu_dst, gpu_dst, diff);
      const int bad = cv::countNonZero(diff);
      std::printf("remap linear: differing pixels: %d\n", bad);
      if (bad != 0) ++failures;
    }

    // the production detection-mask path uses NEAREST (bit-identical
    // contract) — exercise it against cv::INTER_NEAREST exactly
    cv::Mat cpu_nn, gpu_nn(rows, cols, CV_8UC1);
    cv::remap(mask, cpu_nn, map_x, map_y, cv::INTER_NEAREST);
    if (!sonar_slam::gpu::remap_u8_cuda(
          mask.ptr<std::uint8_t>(), rows, cols, map_x.ptr<float>(),
          map_y.ptr<float>(), rows, cols, 0, /*map_version=*/-1,
          gpu_nn.ptr<std::uint8_t>())) {
      std::printf("remap nearest: GPU path failed\n");
      ++failures;
    } else {
      cv::Mat diff_nn;
      cv::absdiff(cpu_nn, gpu_nn, diff_nn);
      const int bad_nn = cv::countNonZero(diff_nn);
      std::printf("remap nearest: differing pixels: %d\n", bad_nn);
      if (bad_nn != 0) ++failures;
    }
  }

  // ------------------------------------------------------- global-init cost
  if (gpu) {
    constexpr int grid_rows = 8, grid_cols = 9;
    constexpr float xmin = -1.f, ymin = -2.f, resolution = 0.5f;
    std::vector<std::uint8_t> grid(grid_rows * grid_cols, 0);
    for (const auto& rc : {std::pair<int, int>{2, 2}, {3, 4}, {5, 6}, {6, 1}})
      grid[rc.first * grid_cols + rc.second] = 255;

    const std::vector<float> points = {
      0.f, 0.f, 1.f, 0.5f, 2.f, 1.f, -0.5f, 1.f, 4.f, 4.f,
    };
    const std::vector<float> transforms = {
      1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
      1.f, 0.f, 0.f, 1.f, 1.f, 0.5f,
      0.f, -1.f, 1.f, 0.f, 0.f, 0.f,
    };
    const int n_points = static_cast<int>(points.size() / 2);
    const int n_transforms = static_cast<int>(transforms.size() / 6);
    std::vector<float> got(n_transforms), expected(n_transforms);
    if (!sonar_slam::gpu::grid_cost_cuda(
          points.data(), n_points, transforms.data(), n_transforms, grid.data(),
          grid_rows, grid_cols, xmin, ymin, resolution, got.data())) {
      std::printf("grid cost: GPU path failed\n");
      ++failures;
    } else {
      for (int t = 0; t < n_transforms; ++t) {
        const float* T = transforms.data() + 6 * t;
        int hits = 0;
        for (int p = 0; p < n_points; ++p) {
          const float x = T[0] * points[2 * p] + T[1] * points[2 * p + 1] + T[4];
          const float y = T[2] * points[2 * p] + T[3] * points[2 * p + 1] + T[5];
          const int r = static_cast<int>(std::nearbyint((y - ymin) / resolution));
          const int c = static_cast<int>(std::nearbyint((x - xmin) / resolution));
          if (r >= 0 && r < grid_rows && c >= 0 && c < grid_cols &&
              grid[r * grid_cols + c] != 0)
            ++hits;
        }
        expected[t] = -static_cast<float>(hits);
      }
      int bad = 0;
      for (int t = 0; t < n_transforms; ++t)
        if (got[t] != expected[t]) ++bad;
      ++compared;
      std::printf("grid cost: mismatched transforms: %d\n", bad);
      if (bad != 0) ++failures;
    }
  }

  // --------------------------------------------------------------- exact 1-NN
  if (gpu) {
    const std::vector<float> ref = {
      0.f, 0.f, 2.f, 0.f, 0.f, 3.f, 5.f, 5.f, 5.f, 5.f,
    };
    const std::vector<float> query = {
      0.25f, 0.f, 1.8f, 0.1f, 0.f, 2.9f, 5.f, 5.f, 100.f, 100.f,
    };
    const int n_ref = static_cast<int>(ref.size() / 2);
    const int n_query = static_cast<int>(query.size() / 2);
    std::vector<int> ids(n_query);
    std::vector<float> dists2(n_query);
    if (!sonar_slam::gpu::nn1_cuda(ref.data(), n_ref, query.data(), n_query,
                                    1.f, ids.data(), dists2.data())) {
      std::printf("1-NN: GPU path failed\n");
      ++failures;
    } else {
      const int expected_ids[] = {0, 1, 2, 3, -1};
      int bad = 0;
      for (int q = 0; q < n_query; ++q) {
        if (ids[q] != expected_ids[q]) {
          ++bad;
          continue;
        }
        if (ids[q] < 0) {
          if (!std::isinf(dists2[q])) ++bad;
          continue;
        }
        const float dx = query[2 * q] - ref[2 * ids[q]];
        const float dy = query[2 * q + 1] - ref[2 * ids[q] + 1];
        if (std::abs(dists2[q] - (dx * dx + dy * dy)) > 1e-6f) ++bad;
      }
      ++compared;
      std::printf("1-NN: mismatched queries: %d\n", bad);
      if (bad != 0) ++failures;
    }
  }
#endif

  if (failures) {
    std::printf("PARITY FAILURES: %d (of %d comparisons)\n", failures, compared);
    return 1;
  }
  if (compared == 0) {
    std::printf(
      "parity NOT CHECKED: 0 comparisons ran.\n"
      "  Built without CUDA (-DSONAR_SLAM_ENABLE_CUDA=OFF), or no device is\n"
      "  visible. This is NOT a pass — there was nothing to compare.\n");
    if (!smoke_ok) return 2;
    std::printf(
      "  --smoke-ok: reporting success anyway (CPU smoke run only). The CPU\n"
      "  kernels above ran without crashing; GPU parity remains UNVERIFIED.\n");
    return 0;
  }
  std::printf("parity OK (%d comparisons, 0 failures)\n", compared);
  return 0;
}
