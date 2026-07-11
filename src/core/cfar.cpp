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
      for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (f(lo) * f(mid) <= 0.0) hi = mid; else lo = mid;
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
  const double l1 = std::lgamma(Ntc_ + 1.0);
  const double l2 = std::lgamma(Ntc_ - rank_ + 1.0);
  const double l4 = std::lgamma(x + Ntc_ - rank_ + 1.0);
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
                      std::uint8_t* mask_out)
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
      mask_out[static_cast<std::size_t>(row) * cols + col] = hit ? 1 : 0;
    }
  }
}

cv::Mat CFAR::detect(const cv::Mat& img, Alg alg) const
{
  cv::Mat fimg;
  if (img.type() == CV_32FC1)
    fimg = img.isContinuous() ? img : img.clone();
  else
    img.convertTo(fimg, CV_32FC1);

  cv::Mat mask(img.rows, img.cols, CV_8UC1);
  const int train_hs = Ntc_ / 2, guard_hs = Ngc_ / 2;
  const double tau = threshold_factor(alg);

#ifdef SONAR_SLAM_WITH_CUDA
  // the wrapper refuses unsupported configs and reports device errors; either
  // way the CPU twin below produces the result
  if (gpu::available() &&
      gpu::cfar_cuda(fimg.ptr<float>(), img.rows, img.cols,
                     static_cast<int>(alg), train_hs, guard_hs, rank_, tau,
                     mask.ptr<std::uint8_t>()))
    return mask;
#endif
  detect_cpu(fimg.ptr<float>(), img.rows, img.cols, static_cast<int>(alg),
             train_hs, guard_hs, rank_, tau, mask.ptr<std::uint8_t>());
  return mask;
}

}  // namespace sonar_slam
