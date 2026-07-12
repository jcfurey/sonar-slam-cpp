// Validation that Interp1d CUBIC reproduces scipy.interp1d(kind='cubic')
// (a not-a-knot cubic spline). The reference values below were generated with
// scipy 1.x (see the generator note); the "uniform" grid is the case where a
// naive Thomas solve of the not-a-knot moment system hits a zero pivot, so it
// guards the robust dense solve in interp.hpp.
//
// Eigen-only, no ROS/GTSAM/CUDA — build and run with:
//   g++ -std=c++17 -O2 -I include -I /usr/include/eigen3
//       test/interp_spline_test.cpp -o /tmp/interp_test && /tmp/interp_test
#include "sonar_slam_cpp/interp.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using sonar_slam::Interp1d;

namespace {
struct Case
{
  std::string name;
  std::vector<double> x, y, q, scipy;  // knots, values, queries, scipy cubic(q)
};

// scipy.interp1d(kind='cubic') reference (not-a-knot); f(x)=1+2x-0.5x^2+0.3 sin(2x)
const std::vector<Case> kCases = {
  { "nonuniform", {0, 0.69999999999999996, 1.5, 2.1000000000000001, 3.3999999999999999, 4}, {1, 2.450634918996538, 2.9173360024179602, 2.7335272682759237, 2.1682340053415827, 1.2968074739870146}, {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}, {1, 2.1472476108881389, 2.7568199567301663, 2.9173360024179602, 2.7703955141561494, 2.6190275423985776, 2.4460603040077915, 2.0670609025278139, 1.2968074739870146} },
  { "uniform", {0, 1, 2, 3, 4, 5}, {1, 2.7727892280477047, 2.7729592514076216, 2.4161753505403221, 1.2968074739870146, -1.663206333266811}, {0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5}, {1, 2.4154372547137148, 2.8664946576838561, 2.8064230004311006, 2.6333668292751091, 2.3380880199767051, 1.704836661608929, 0.46351633036022843, -1.663206333266811} },
};
}  // namespace

int main()
{
  int fails = 0;
  for (const auto& c : kCases) {
    Interp1d g(c.x, c.y, Interp1d::CUBIC);
    double maxerr = 0.0;
    for (std::size_t i = 0; i < c.q.size(); ++i)
      maxerr = std::max(maxerr, std::abs(g(c.q[i]) - c.scipy[i]));
    const bool ok = maxerr < 1e-9;
    std::printf("%-11s n=%zu  max|cpp - scipy| = %.3e  %s\n", c.name.c_str(),
                c.x.size(), maxerr, ok ? "OK" : "FAIL");
    if (!ok) ++fails;
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS (matches scipy not-a-knot)");
  return fails ? 1 : 0;
}
