#include "sonar_slam_cpp/cfar.hpp"
#include "sonar_slam_cpp/gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace sonar_slam {

// ---------------------------------------------------------------------------
// Threshold factors (port of CFAR.py). The Pfa expressions are smooth and
// monotonically decreasing in x, so a bracketed bisection replaces scipy.root.
// ---------------------------------------------------------------------------
namespace {

double solve_root(const std::function<double(double)>& f)
{
  // find a sign change over a wide log-spaced bracket, then bisect
  double prev_x = 1e-8, prev_f = f(prev_x);
  for (int i = 1; i <= 2000; ++i) {
    const double x = 1e-8 * std::pow(10.0, 16.0 * i / 2000.0);  // up to 1e8
    const double fx = f(x);
    if (std::isfinite(fx) && std::isfinite(prev_f) && prev_f * fx <= 0.0) {
      double lo = prev_x, hi = x;
      // f(lo) only changes when lo moves, so carry it rather than
      // re-evaluating it every iteration — same bracketing, half the
      // evaluations of a function that costs O(Ntc) lgamma/exp/pow calls.
      double f_lo = f(lo);
      for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = f(mid);
        if (f_lo * f_mid <= 0.0) {
          hi = mid;
        } else {
          lo = mid;
          f_lo = f_mid;
        }
      }
      return 0.5 * (lo + hi);
    }
    prev_x = x;
    prev_f = fx;
  }
  throw std::runtime_error("CFAR threshold factor: root not found");
}

}  // namespace

CFAR::CFAR(int Ntc, int Ngc, double Pfa, int rank)
  : Ntc_(Ntc), Ngc_(Ngc), rank_(rank < 0 ? Ntc / 2 : rank), Pfa_(Pfa)
{
  if (Ntc_ % 2 != 0 || Ngc_ % 2 != 0)
    throw std::invalid_argument("CFAR: Ntc and Ngc must be even");
  if (rank_ < 0 || rank_ >= Ntc_)
    throw std::invalid_argument("CFAR: rank must be in [0, Ntc)");

  tau_ca_ = calc_threshold_factor_ca();
  tau_soca_ = solve_root([this](double x) { return pfa_soca(x); });
  tau_goca_ = solve_root([this](double x) { return pfa_goca(x); });
  tau_os_ = solve_root([this](double x) { return pfa_os(x); });
}

CFAR::Alg CFAR::alg_from_string(const std::string& name)
{
  if (name == "CA") return CA;
  if (name == "SOCA") return SOCA;
  if (name == "GOCA") return GOCA;
  if (name == "OS") return OS;
  throw std::invalid_argument("CFAR: unknown algorithm '" + name + "'");
}

double CFAR::calc_threshold_factor_ca() const
{
  return Ntc_ * (std::pow(Pfa_, -1.0 / Ntc_) - 1.0);
}

double CFAR::pfa_gosoca_core(double x) const
{
  const double half = Ntc_ / 2.0;
  double temp = 0.0;
  for (int k = 0; k < Ntc_ / 2; ++k) {
    const double l1 = std::lgamma(half + k);
    const double l2 = std::lgamma(k + 1.0);
    const double l3 = std::lgamma(half);
    temp += std::exp(l1 - l2 - l3) * std::pow(2.0 + x / half, -k);
  }
  return temp * std::pow(2.0 + x / half, -half);
}

double CFAR::pfa_soca(double x) const { return pfa_gosoca_core(x) - Pfa_ / 2.0; }

double CFAR::pfa_goca(double x) const
{
  const double half = Ntc_ / 2.0;
  const double temp = std::pow(1.0 + x / half, -half);
  return temp - pfa_gosoca_core(x) - Pfa_ / 2.0;
}

double CFAR::pfa_os(double x) const
{
  // P_FA for the detector's ACTUAL statistic train[rank_], i.e. the
  // (rank_+1)-th smallest of the Ntc training cells (0-indexed nth_element).
  // By the Renyi spacing representation of exponential order statistics,
  //   P_FA(T) = prod_{i=0}^{rank_} (N - i) / (T + N - i),
  // which in log-Gamma form is Gamma(N+1)/Gamma(N-rank) * Gamma(T+N-rank)/
  // Gamma(T+N+1). CFAR.py (and the original cfar.cpp) used one factor fewer
  // (Gamma(N-rank+1), Gamma(T+N-rank+1)) — the model for the rank_-th
  // smallest — so its solved tau over-thresholded and the realized P_FA fell
  // ~(N-rank)/(T+N-rank) below the requested one. Derivation and Monte Carlo
  // validation: docs/MATH_NOTES.md; intentional divergence, docs/DIVERGENCES.md.
  const double l1 = std::lgamma(Ntc_ + 1.0);
  const double l2 = std::lgamma(static_cast<double>(Ntc_ - rank_));
  const double l4 = std::lgamma(x + Ntc_ - rank_);
  const double l6 = std::lgamma(x + Ntc_ + 1.0);
  return std::exp(l1 - l2 + l4 - l6) - Pfa_;
}

double CFAR::threshold_factor(Alg alg) const
{
  switch (alg) {
    case CA: return tau_ca_;
    case SOCA: return tau_soca_;
    case GOCA: return tau_goca_;
    case OS: return tau_os_;
  }
  return tau_soca_;
}

// ---------------------------------------------------------------------------
// Detectors (port of cpp/cfar.cpp). The window slides along the range axis
// (rows) within each beam column. OpenMP parallelizes over columns.
// ---------------------------------------------------------------------------
void CFAR::detect_cpu(const float* img, int rows, int cols, int alg,
                      int train_hs, int guard_hs, int rank, double tau,
                      float threshold, std::uint8_t* mask_out)
{
  const int border = train_hs + guard_hs;
  std::fill(mask_out, mask_out + static_cast<std::size_t>(rows) * cols, 0);

#pragma omp parallel for schedule(static)
  for (int col = 0; col < cols; ++col) {
    std::vector<float> train;  // OS scratch
    for (int row = border; row < rows - border; ++row) {
      const float cell = img[static_cast<std::size_t>(row) * cols + col];
      bool hit = false;
      if (alg == 0) {  // CA
        float sum_train = 0.f;
        for (int i = row - border; i <= row + border; ++i)
          if (std::abs(i - row) > guard_hs)
            sum_train += img[static_cast<std::size_t>(i) * cols + col];
        hit = cell > tau * sum_train / (2.0 * train_hs);
      } else if (alg == 1 || alg == 2) {  // SOCA / GOCA
        float leading = 0.f, lagging = 0.f;
        for (int i = row - border; i <= row + border; ++i) {
          if ((i - row) > guard_hs)
            lagging += img[static_cast<std::size_t>(i) * cols + col];
          else if ((i - row) < -guard_hs)
            leading += img[static_cast<std::size_t>(i) * cols + col];
        }
        const float sum_train =
          alg == 1 ? std::min(leading, lagging) : std::max(leading, lagging);
        hit = cell > tau * sum_train / train_hs;
      } else {  // OS
        train.clear();
        for (int i = row - border; i <= row + border; ++i)
          if (std::abs(i - row) > guard_hs)
            train.push_back(img[static_cast<std::size_t>(i) * cols + col]);
        std::nth_element(train.begin(), train.begin() + rank, train.end());
        hit = cell > tau * train[rank];
      }
      // fold the intensity gate into the detector so the CPU/GPU twins do it
      // identically and the caller needs no separate compare + bitwise-and
      mask_out[static_cast<std::size_t>(row) * cols + col] =
        (hit && cell > threshold) ? 1 : 0;
    }
  }
}

namespace {

// Exact prefix-sum CA/SOCA/GOCA for integer-valued (CV_8UC1) images: per
// column, int32 prefix sums make every training-window sum O(1) instead of
// O(Ntc + Ngc). PROVABLY bit-identical to detect_cpu on uint8 input: every
// partial float sum there is an integer <= Ntc*255 < 2^24, hence exact and
// order-independent (IEEE 754 correct rounding of representable results), and
// equal to the integer prefix difference computed here; the threshold
// expressions are then evaluated identically. Full proof: docs/MATH_NOTES.md
// (Theorems 7-8). OS keeps the nth_element path (order statistic).
void detect_cpu_prefix_u8(const std::uint8_t* img, int rows, int cols, int alg,
                          int train_hs, int guard_hs, double tau,
                          float threshold, std::uint8_t* mask_out)
{
  const int border = train_hs + guard_hs;
  std::fill(mask_out, mask_out + static_cast<std::size_t>(rows) * cols, 0);

#pragma omp parallel for schedule(static)
  for (int col = 0; col < cols; ++col) {
    std::vector<std::int32_t> P(static_cast<std::size_t>(rows) + 1);
    P[0] = 0;
    for (int i = 0; i < rows; ++i)
      P[i + 1] = P[i] + img[static_cast<std::size_t>(i) * cols + col];

    for (int row = border; row < rows - border; ++row) {
      // same value the float path reads: uint8 -> float is exact
      const float cell =
        static_cast<float>(img[static_cast<std::size_t>(row) * cols + col]);
      // leading = sum over [row-border, row-guard_hs-1],
      // lagging = sum over [row+guard_hs+1, row+border]
      const std::int32_t leading = P[row - guard_hs] - P[row - border];
      const std::int32_t lagging = P[row + border + 1] - P[row + guard_hs + 1];
      bool hit;
      if (alg == 0) {  // CA
        const float sum_train = static_cast<float>(leading + lagging);
        hit = cell > tau * sum_train / (2.0 * train_hs);
      } else {  // SOCA / GOCA
        const float sum_train = static_cast<float>(
          alg == 1 ? std::min(leading, lagging) : std::max(leading, lagging));
        hit = cell > tau * sum_train / train_hs;
      }
      mask_out[static_cast<std::size_t>(row) * cols + col] =
        (hit && cell > threshold) ? 1 : 0;
    }
  }
}

cv::Mat to_float(const cv::Mat& img)
{
  if (img.type() == CV_32FC1) return img.isContinuous() ? img : img.clone();
  cv::Mat fimg;
  img.convertTo(fimg, CV_32FC1);
  return fimg;
}

}  // namespace

cv::Mat CFAR::detect(const cv::Mat& img, Alg alg, float threshold) const
{
  cv::Mat mask(img.rows, img.cols, CV_8UC1);
  const int train_hs = Ntc_ / 2, guard_hs = Ngc_ / 2;
  const double tau = threshold_factor(alg);

#ifdef SONAR_SLAM_WITH_CUDA
  // the wrapper refuses unsupported configs and reports device errors; either
  // way the CPU paths below produce the result
  if (gpu::available()) {
    // Runtime polar images are uint8 — upload them as uint8. Converting to
    // float first cost a whole-image host pass plus an allocation per ping
    // and quadrupled the host->device transfer, all so the kernel could
    // immediately re-widen the same values. uint8 -> float is exact, so the
    // mask is identical either way (parity_check still exercises the float
    // entry point against the float CPU twin).
    if (img.type() == CV_8UC1) {
      const cv::Mat u8 = img.isContinuous() ? img : img.clone();
      if (gpu::cfar_u8_cuda(u8.ptr<std::uint8_t>(), img.rows, img.cols,
                            static_cast<int>(alg), train_hs, guard_hs, rank_,
                            tau, threshold, mask.ptr<std::uint8_t>()))
        return mask;
    } else {
      const cv::Mat fimg = to_float(img);
      if (gpu::cfar_cuda(fimg.ptr<float>(), img.rows, img.cols,
                         static_cast<int>(alg), train_hs, guard_hs, rank_, tau,
                         threshold, mask.ptr<std::uint8_t>()))
        return mask;
    }
  }
#endif

  // bit-exact O(1)-per-pixel path for the runtime case (uint8 polar images)
  if (img.type() == CV_8UC1 && alg != OS) {
    const cv::Mat u8 = img.isContinuous() ? img : img.clone();
    detect_cpu_prefix_u8(u8.ptr<std::uint8_t>(), img.rows, img.cols,
                         static_cast<int>(alg), train_hs, guard_hs, tau,
                         threshold, mask.ptr<std::uint8_t>());
    return mask;
  }

  const cv::Mat fimg = to_float(img);
  detect_cpu(fimg.ptr<float>(), img.rows, img.cols, static_cast<int>(alg),
             train_hs, guard_hs, rank_, tau, threshold, mask.ptr<std::uint8_t>());
  return mask;
}

}  // namespace sonar_slam
