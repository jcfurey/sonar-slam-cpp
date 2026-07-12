// Constant False Alarm Rate detection: port of bruce_slam CFAR.py (threshold
// factor solving) + cpp/cfar.cpp (the detectors). GPU-accelerated with a
// bit-exact CPU fallback; dispatch is automatic via gpu::available().
#pragma once

#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <string>

namespace sonar_slam {

class CFAR
{
public:
  enum Alg { CA = 0, SOCA = 1, GOCA = 2, OS = 3 };

  // rank < 0 selects the default Ntc/2 (CFAR.py: rank=None)
  CFAR(int Ntc, int Ngc, double Pfa, int rank = -1);

  static Alg alg_from_string(const std::string& name);

  // img: CV_8UC1 or CV_32FC1 polar image (rows = range bins, cols = beams).
  // Returns a CV_8UC1 mask with 1 at detections (matches cfar.cpp output).
  // A detection also requires cell intensity > threshold; the default keeps
  // every CFAR hit (no intensity gate), so pass filter/threshold to fold the
  // gate into the detector instead of a separate compare + bitwise-and.
  cv::Mat detect(const cv::Mat& img, Alg alg,
                 float threshold = -std::numeric_limits<float>::infinity()) const;

  double threshold_factor(Alg alg) const;

  int Ntc() const { return Ntc_; }
  int Ngc() const { return Ngc_; }
  int rank() const { return rank_; }

  // exposed for the parity check tool
  static void detect_cpu(const float* img, int rows, int cols, int alg,
                         int train_hs, int guard_hs, int rank, double tau,
                         float threshold, std::uint8_t* mask_out);

private:
  double calc_threshold_factor_ca() const;
  double calc_threshold_factor(int alg) const;   // SOCA/GOCA/OS via root finding
  double pfa_gosoca_core(double x) const;
  double pfa_soca(double x) const;
  double pfa_goca(double x) const;
  double pfa_os(double x) const;

  int Ntc_, Ngc_, rank_;
  double Pfa_;
  double tau_ca_, tau_soca_, tau_goca_, tau_os_;
};

}  // namespace sonar_slam
