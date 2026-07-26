// Validation of the CFAR mathematics in src/core/cfar.cpp against independent
// closed forms and Monte Carlo simulation, plus bit-exactness of the
// prefix-sum fast path. Companion to docs/MATH_NOTES.md.
//
//  1. The bisection-solved threshold factors satisfy the closed-form P_FA
//     expressions (CA product form; SOCA/GOCA finite sums; OS Renyi product
//     for the (rank+1)-th smallest — the statistic the detector actually
//     thresholds).
//  2. Monte Carlo under the exponential (square-law) noise model: the
//     realized false-alarm rate of each detector statistic matches the design
//     P_FA, and the pre-fix OS tau (solved for the rank-th smallest) is shown
//     to under-shoot the target — the carried-over calibration bug.
//  3. detect() on uint8 images (prefix-sum path) is bit-identical to
//     detect_cpu (naive float path) for CA/SOCA/GOCA, with and without the
//     folded intensity gate.
//  4. Speed: prefix vs naive on a 716x512 sonar-sized image.
//
// Build (OpenCV core only, no ROS/GTSAM/CUDA):
//   g++ -std=c++17 -O2 -fopenmp -I include -I /usr/include/opencv4
//       test/cfar_math_test.cpp src/core/cfar.cpp src/core/gpu_dispatch.cpp
//       -lopencv_core -o /tmp/cfar_math_test && /tmp/cfar_math_test
#include "sonar_slam_cpp/cfar.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using sonar_slam::CFAR;

namespace {

constexpr int N = 40, NGC = 10, RANK = 10;  // config/feature.yaml values
constexpr double PFA = 0.1;
constexpr float kNoGate = -std::numeric_limits<float>::infinity();

// ---- independent closed forms (docs/MATH_NOTES.md) -------------------------
double pfa_ca(double t) { return std::pow(1.0 + t / N, -double(N)); }

double pfa_soca(double t)
{
  const double h = N / 2.0, a = t / h;
  double s = 0.0;
  for (int k = 0; k < N / 2; ++k)
    s += std::exp(std::lgamma(h + k) - std::lgamma(k + 1.0) - std::lgamma(h)) *
         std::pow(2.0 + a, -(h + k));
  return 2.0 * s;
}

double pfa_goca(double t)
{
  const double h = N / 2.0, a = t / h;
  return 2.0 * std::pow(1.0 + a, -h) - pfa_soca(t);
}

// Renyi product for P(X > t * Z_(j)), Z_(j) the j-th SMALLEST of N iid Exp(1)
double pfa_os_jth(double t, int j)
{
  double p = 1.0;
  for (int i = 0; i < j; ++i) p *= (N - i) / (t + N - i);
  return p;
}

double bisect(double (*f)(double, int), int j, double target)
{
  double lo = 1e-8, hi = 1e8;
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (f(mid, j) - target > 0.0 ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

}  // namespace

int main()
{
  int fails = 0;
  CFAR cfar(N, NGC, PFA, RANK);
  const double t_ca = cfar.threshold_factor(CFAR::CA);
  const double t_so = cfar.threshold_factor(CFAR::SOCA);
  const double t_go = cfar.threshold_factor(CFAR::GOCA);
  const double t_os = cfar.threshold_factor(CFAR::OS);

  // ---- 1. closed-form agreement -------------------------------------------
  struct { const char* n; double got; } checks[] = {
    {"CA  ", pfa_ca(t_ca)}, {"SOCA", pfa_soca(t_so)}, {"GOCA", pfa_goca(t_go)},
    {"OS  ", pfa_os_jth(t_os, RANK + 1)},  // detector uses train[RANK], the (RANK+1)-th smallest
  };
  std::printf("[1] solved tau satisfies the closed-form P_FA = %.3f:\n", PFA);
  for (const auto& c : checks) {
    const bool ok = std::abs(c.got - PFA) < 1e-9;
    std::printf("    %s tau -> P_FA = %.12f  %s\n", c.n, c.got, ok ? "OK" : "FAIL");
    if (!ok) ++fails;
  }

  // ---- 2. Monte Carlo false-alarm rates -----------------------------------
  {
    std::mt19937_64 rng(42);
    std::exponential_distribution<double> ex(1.0);
    const int T = 2'000'000;
    long hit_ca = 0, hit_so = 0, hit_os_new = 0, hit_os_old = 0;
    const double t_os_old = bisect(pfa_os_jth, RANK, PFA);  // pre-fix model
    std::vector<double> z(N);
    for (int t = 0; t < T; ++t) {
      for (auto& v : z) v = ex(rng);
      const double cell = ex(rng);
      double s1 = 0, s2 = 0;
      for (int i = 0; i < N / 2; ++i) s1 += z[i];
      for (int i = N / 2; i < N; ++i) s2 += z[i];
      if (cell > t_ca * (s1 + s2) / N) ++hit_ca;
      if (cell > t_so * std::min(s1, s2) / (N / 2)) ++hit_so;
      std::nth_element(z.begin(), z.begin() + RANK, z.end());
      if (cell > t_os * z[RANK]) ++hit_os_new;      // fixed tau
      if (cell > t_os_old * z[RANK]) ++hit_os_old;  // carried-over tau
    }
    const double eca = double(hit_ca) / T, eso = double(hit_so) / T;
    const double enew = double(hit_os_new) / T, eold = double(hit_os_old) / T;
    const double pred_old = PFA * (N - RANK) / (t_os_old + N - RANK);
    std::printf("[2] Monte Carlo (%d trials), target P_FA = %.3f:\n", T, PFA);
    std::printf("    CA   empirical = %.4f\n", eca);
    std::printf("    SOCA empirical = %.4f\n", eso);
    std::printf("    OS   empirical (fixed tau)  = %.4f\n", enew);
    std::printf("    OS   empirical (old tau)    = %.4f  (theory predicts %.4f)\n",
                eold, pred_old);
    const bool ok = std::abs(eca - PFA) < 3e-3 && std::abs(eso - PFA) < 3e-3 &&
                    std::abs(enew - PFA) < 3e-3 && std::abs(eold - pred_old) < 3e-3 &&
                    eold < PFA - 0.01;  // the old calibration measurably under-shoots
    std::printf("    %s\n", ok ? "OK (fixed tau hits target; old tau under-shoots as derived)"
                              : "FAIL");
    if (!ok) ++fails;
  }

  // ---- 3. prefix path bit-identical to the naive float path ----------------
  {
    std::mt19937 rng(7);
    int mismatches = 0, cases = 0;
    const int b = (N + NGC) / 2;  // border
    const int sizes[][2] = {{716, 512}, {64, 32}, {51, 7}, {2 * b + 1, 3}, {2 * b, 4}};
    for (const auto& sz : sizes) {
      const int rows = sz[0], cols = sz[1];
      cv::Mat img(rows, cols, CV_8UC1);
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          img.at<std::uint8_t>(r, c) = static_cast<std::uint8_t>(rng() % 256);
      cv::Mat fimg;
      img.convertTo(fimg, CV_32FC1);
      for (auto alg : {CFAR::CA, CFAR::SOCA, CFAR::GOCA}) {
        for (float gate : {kNoGate, 65.f}) {
          cv::Mat naive(rows, cols, CV_8UC1);
          CFAR::detect_cpu(fimg.ptr<float>(), rows, cols, int(alg), N / 2,
                           NGC / 2, RANK, cfar.threshold_factor(alg), gate,
                           naive.ptr<std::uint8_t>());
          const cv::Mat fast = cfar.detect(img, alg, gate);  // prefix path
          for (int i = 0; i < rows * cols; ++i)
            if (naive.data[i] != fast.data[i]) ++mismatches;
          ++cases;
        }
      }
    }
    std::printf("[3] prefix vs naive: %d cases, %d byte mismatches  %s\n", cases,
                mismatches, mismatches == 0 ? "OK (bit-identical)" : "FAIL");
    if (mismatches != 0) ++fails;
  }

  // ---- 4. speed -------------------------------------------------------------
  {
    const int rows = 716, cols = 512, iters = 30;
    std::mt19937 rng(3);
    cv::Mat img(rows, cols, CV_8UC1);
    for (int i = 0; i < rows * cols; ++i) img.data[i] = rng() % 256;
    cv::Mat fimg;
    img.convertTo(fimg, CV_32FC1);
    cv::Mat m(rows, cols, CV_8UC1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      CFAR::detect_cpu(fimg.ptr<float>(), rows, cols, int(CFAR::SOCA), N / 2,
                       NGC / 2, RANK, t_so, kNoGate, m.ptr<std::uint8_t>());
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) (void)cfar.detect(img, CFAR::SOCA);
    const auto t2 = std::chrono::steady_clock::now();
    const double ms_naive =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double ms_fast =
      std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
    std::printf("[4] 716x512 SOCA: naive %.2f ms, prefix %.2f ms  (%.1fx)\n",
                ms_naive, ms_fast, ms_naive / ms_fast);
  }

  // ---- 5. realized false-alarm rate of detect() on uint8 IMAGES -------------
  // Stage 2 above validates the threshold FORMULAS by Monte Carlo over
  // continuous doubles. That cannot see quantization: the runtime detector
  // consumes uint8 polar frames, where the reference statistic is an integer.
  // This stage drives the shipped detect() path end to end on pure exponential
  // clutter (square-law detected Rayleigh — the model the factors are derived
  // under, docs/MATH_NOTES.md §1) with NO targets, so every detection is a
  // false alarm, and asserts the realized rate against nominal.
  //
  // It is also the regression guard for CFAR/rank: on uint8 the shipped-by-
  // bruce_slam rank 10 realizes ~13% more false alarms than requested while
  // Rohling's 3N/4 tracks nominal, which is why feature.yaml now ships 30
  // (docs/SONAR_FRONTEND_REVIEW.md §3).
  {
    const int rows = 716, cols = 512;
    const int border = N / 2 + NGC / 2;
    std::mt19937 rng(11);
    std::exponential_distribution<double> ex(1.0);
    // mean 30 matches the synthetic fixture's background level, so the
    // absolute intensity gate sits in the same place relative to the clutter
    cv::Mat clutter(rows, cols, CV_8UC1);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        clutter.at<std::uint8_t>(r, c) =
          static_cast<std::uint8_t>(std::min(255.0, 30.0 * ex(rng)));

    const auto realized = [&](const CFAR& d, CFAR::Alg alg) {
      const cv::Mat m = d.detect(clutter, alg);  // no intensity gate
      long hits = 0, valid = 0;
      for (int r = border; r < rows - border; ++r)
        for (int c = 0; c < cols; ++c) {
          ++valid;
          if (m.at<std::uint8_t>(r, c)) ++hits;
        }
      return static_cast<double>(hits) / static_cast<double>(std::max(1L, valid));
    };

    std::printf("[5] realized P_FA of detect() on uint8 exponential clutter"
                " (target %.3f):\n", PFA);
    // The averaging detectors are accurate on quantized data: their statistic
    // is a SUM over Ntc cells, so the quantization step is diluted N-fold.
    struct { const char* n; CFAR::Alg a; } avg[] = {
      {"CA", CFAR::CA}, {"SOCA", CFAR::SOCA}, {"GOCA", CFAR::GOCA}};
    for (const auto& c : avg) {
      const double got = realized(cfar, c.a);
      const bool ok = std::abs(got - PFA) < 0.01;
      std::printf("    %-4s realized = %.4f  %s\n", c.n, got, ok ? "OK" : "FAIL");
      if (!ok) ++fails;
    }
    // OS uses a single order statistic, so quantization is NOT diluted.
    const CFAR os_low(N, NGC, PFA, 10);          // bruce_slam's N/4
    const CFAR os_rohling(N, NGC, PFA, 3 * N / 4);  // shipped
    const double fa_low = realized(os_low, CFAR::OS);
    const double fa_roh = realized(os_rohling, CFAR::OS);
    std::printf("    OS rank %2d realized = %.4f (tau %.3f)\n", 10, fa_low,
                os_low.threshold_factor(CFAR::OS));
    std::printf("    OS rank %2d realized = %.4f (tau %.3f)  <- shipped\n",
                3 * N / 4, fa_roh, os_rohling.threshold_factor(CFAR::OS));
    if (!(std::abs(fa_roh - PFA) < 0.01)) {
      std::printf("    FAIL: shipped OS rank does not hold nominal P_FA\n");
      ++fails;
    }
    // The ordering is the finding, and it must not silently invert: the
    // 3N/4 rank has to stay closer to nominal than N/4 on quantized input.
    if (!(std::abs(fa_roh - PFA) < std::abs(fa_low - PFA))) {
      std::printf("    FAIL: rank %d is no longer closer to nominal than %d\n",
                  3 * N / 4, 10);
      ++fails;
    }
  }

  std::printf("%s\n", fails ? "FAIL" : "ALL PASS");
  return fails ? 1 : 0;
}
