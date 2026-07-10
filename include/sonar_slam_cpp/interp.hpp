// 1-D interpolation mirroring the scipy.interp1d configurations used by
// bruce_slam: kind='linear'/'cubic', bounds_error=False, fill_value=-1,
// assume_sorted=True. The cubic variant is a natural cubic spline.
#pragma once

#include <cmath>
#include <vector>

namespace sonar_slam {

class Interp1d
{
public:
  enum Kind { LINEAR, CUBIC };

  Interp1d() = default;

  Interp1d(std::vector<double> x, std::vector<double> y, Kind kind,
           double fill_value = -1.0)
    : x_(std::move(x)), y_(std::move(y)), kind_(kind), fill_(fill_value)
  {
    if (kind_ == CUBIC && x_.size() >= 3) build_spline();
  }

  double operator()(double xq) const
  {
    if (x_.empty()) return fill_;
    if (xq < x_.front() || xq > x_.back()) return fill_;
    // locate the interval by binary search
    std::size_t lo = 0, hi = x_.size() - 1;
    while (hi - lo > 1) {
      std::size_t mid = (lo + hi) / 2;
      if (x_[mid] <= xq) lo = mid; else hi = mid;
    }
    const double h = x_[hi] - x_[lo];
    if (h <= 0.0) return y_[lo];
    const double t = (xq - x_[lo]) / h;
    if (kind_ == LINEAR || m_.empty()) {
      return y_[lo] + t * (y_[hi] - y_[lo]);
    }
    // natural cubic spline on [lo, hi]
    const double a = y_[lo], b = y_[hi];
    const double m0 = m_[lo], m1 = m_[hi];
    const double u = 1.0 - t;
    return u * a + t * b +
           ((u * u * u - u) * m0 + (t * t * t - t) * m1) * (h * h) / 6.0;
  }

  bool valid() const { return !x_.empty(); }

private:
  void build_spline()
  {
    // solve the tridiagonal system for second derivatives (natural BCs)
    const std::size_t n = x_.size();
    m_.assign(n, 0.0);
    std::vector<double> a(n, 0.0), b(n, 0.0), c(n, 0.0), d(n, 0.0);
    b[0] = 1.0; b[n - 1] = 1.0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
      const double h0 = x_[i] - x_[i - 1];
      const double h1 = x_[i + 1] - x_[i];
      a[i] = h0 / 6.0;
      b[i] = (h0 + h1) / 3.0;
      c[i] = h1 / 6.0;
      d[i] = (y_[i + 1] - y_[i]) / h1 - (y_[i] - y_[i - 1]) / h0;
    }
    // Thomas algorithm
    for (std::size_t i = 1; i < n; ++i) {
      const double w = a[i] / b[i - 1];
      b[i] -= w * c[i - 1];
      d[i] -= w * d[i - 1];
    }
    m_[n - 1] = d[n - 1] / b[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) m_[i] = (d[i] - c[i] * m_[i + 1]) / b[i];
  }

  std::vector<double> x_, y_, m_;
  Kind kind_ = LINEAR;
  double fill_ = -1.0;
};

}  // namespace sonar_slam
