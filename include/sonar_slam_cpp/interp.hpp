// 1-D interpolation matching the scipy.interp1d configurations used by
// bruce_slam (bounds_error=False, fill_value=-1, assume_sorted=True):
//   LINEAR mirrors scipy.interp1d(kind='linear');
//   CUBIC  mirrors scipy.interp1d(kind='cubic') — a NOT-A-KNOT cubic spline.
// Both agree with scipy to ~1e-15, validated on uniform and non-uniform grids
// (test/interp_spline_test.cpp). x must be ascending; out-of-range queries
// return fill_value (default -1). Only LINEAR is exercised at runtime here;
// CUBIC is kept scipy-faithful for persisted back-end compatibility.
#pragma once

#include <Eigen/Dense>
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
    if (kind_ == CUBIC && x_.size() >= 4) build_spline();
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
    // Cubic spline on [lo, hi], moment (second-derivative) form. The boundary
    // condition lives entirely in how m_ was solved — build_spline() uses
    // NOT-A-KNOT (scipy's interp1d(kind='cubic') default), not natural, which
    // an earlier version of this comment claimed. The evaluation below is
    // boundary-condition agnostic.
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
    // Not-a-knot cubic spline, matching scipy.interp1d(kind='cubic'): the third
    // derivative is continuous across the first and last interior knots. Solve
    // the second-derivative (moment) system A m = r for m_i = S''(x_i). A is
    // tridiagonal except for one extra element in the first and last rows (the
    // not-a-knot conditions), so a general dense solve is used — n is small and
    // this runs only when the table is (re)built. Unlike a naive Thomas sweep,
    // the pivoted dense solve is robust on uniform grids. scipy requires >= 4
    // points for a cubic; fall back to linear (empty m_) otherwise.
    const int n = static_cast<int>(x_.size());
    m_.clear();
    if (n < 4) return;
    auto h = [&](int i) { return x_[i + 1] - x_[i]; };

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd r = Eigen::VectorXd::Zero(n);
    for (int i = 1; i < n - 1; ++i) {
      A(i, i - 1) = h(i - 1);
      A(i, i) = 2.0 * (h(i - 1) + h(i));
      A(i, i + 1) = h(i);
      r(i) = 6.0 * ((y_[i + 1] - y_[i]) / h(i) - (y_[i] - y_[i - 1]) / h(i - 1));
    }
    // not-a-knot at x[1]:   S'''(x1-) = S'''(x1+)
    A(0, 0) = -h(1);
    A(0, 1) = h(0) + h(1);
    A(0, 2) = -h(0);
    // not-a-knot at x[n-2]: S'''(x_{n-2}-) = S'''(x_{n-2}+)
    A(n - 1, n - 3) = -h(n - 2);
    A(n - 1, n - 2) = h(n - 3) + h(n - 2);
    A(n - 1, n - 1) = -h(n - 3);

    const Eigen::VectorXd m = A.fullPivLu().solve(r);
    m_.assign(m.data(), m.data() + n);
  }

  std::vector<double> x_, y_, m_;
  Kind kind_ = LINEAR;
  double fill_ = -1.0;
};

}  // namespace sonar_slam
