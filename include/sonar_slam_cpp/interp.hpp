// 1-D interpolation used by feature extraction (and, in the Python original,
// the sonar bearing<->column remap).
//
// LINEAR mirrors scipy.interp1d(kind='linear', bounds_error=False,
// fill_value=-1, assume_sorted=True) exactly, and is the ONLY kind exercised
// at runtime here.
//
// CUBIC is a NATURAL cubic spline (second derivative = 0 at both ends).
// CAVEAT: scipy.interp1d(kind='cubic') uses NOT-A-KNOT end conditions, so this
// does NOT reproduce scipy near the first/last interval (they agree only in
// the interior — e.g. ~0.1 divergence on a mildly-curved cubic near the ends).
// It is currently UNUSED: the cubic bearing<->column remap from sonar.py was
// not carried into this port (sonar_geometry.cpp builds no such map). If that
// remap is ever ported, replace this with a not-a-knot spline (mind the
// zero-pivot on uniform grids) and validate against scipy. See
// docs/DIVERGENCES.md.
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
    // solve the tridiagonal system for second derivatives (NATURAL BCs — not
    // scipy's not-a-knot; see the header caveat. Unused at runtime.)
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
