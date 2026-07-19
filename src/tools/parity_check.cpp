// CPU/GPU parity and performance self-check. Runs every CUDA kernel against
// its CPU twin on synthetic sonar-like data and reports max mismatch + timing.
// Exits nonzero on any parity failure. With no GPU (or SONAR_SLAM_FORCE_CPU=1)
// it degrades to a CPU smoke test.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
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

int main()
{
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
      // cv::remap uses fixed-point bilinear (5-bit fractions); allow +-1 counts
      cv::Mat diff;
      cv::absdiff(cpu_dst, gpu_dst, diff);
      const int bad = cv::countNonZero(diff > 1);
      std::printf("remap linear: pixels differing by >1: %d\n", bad);
      if (bad > rows * cols / 1000) ++failures;
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
#endif

  std::printf(failures ? "PARITY FAILURES: %d\n" : "parity OK (%d)\n", failures);
  return failures ? 1 : 0;
}
