#include "sonar_slam_cpp/mapping_trajectory.hpp"
#include <cstdio>
#include <limits>

#define CHECK(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL:%d %s\n", __LINE__, m); return 1; } } while (0)

builtin_interfaces::msg::Time at(int sec, unsigned ns = 0)
{
  builtin_interfaces::msg::Time time; time.sec = sec; time.nanosec = ns; return time;
}

int main()
{
  using namespace sonar_slam;
  MappingTrajectory mapping(2.0);
  const gtsam::Pose3 raw(gtsam::Rot3::Ypr(0.4, -0.1, 0.2), gtsam::Point3(1, 2, -3));
  std::vector<KeyframePtr> parents{
    std::make_shared<Keyframe>(true, at(100), raw),
    std::make_shared<Keyframe>(true, at(104), raw)};
  CHECK(mapping.observe(at(100), raw, 0).accepted && mapping.anchors().empty(),
        "first anchor was not delayed for same-ping cloud delivery");
  CHECK(!mapping.observe(at(101, 900000000), raw, 0).published, "anchor published before delay");
  CHECK(mapping.observe(at(102), raw, 0).published && mapping.anchors().size() == 1,
        "source-time cadence did not publish the old anchor");
  CHECK(mapping.anchors()[0].time.sec == 100, "delayed anchor changed acquisition stamp");
  mapping.observe(at(104), raw, 1);
  mapping.observe(at(104, 400000000), raw, 1);
  CHECK(mapping.flush() && mapping.anchors().size() == 4, "partial tail was not flushed");
  CHECK(mapping.anchors().back().time.nanosec == 400000000 && !mapping.flush(),
        "tail flush minted time or duplicated an old observation");

  CHECK(mapping.pose(0, parents).equals(raw, 1e-10), "physical roll/pitch were double-applied");
  const gtsam::Pose3 correction0(gtsam::Rot3::Yaw(0.3), gtsam::Point3(10, 2, 0.5));
  const gtsam::Pose3 correction1(gtsam::Rot3::Yaw(-0.2), gtsam::Point3(20, -1, -0.2));
  // Independent graph corrections, as produced by loop optimization. Anchors
  // must read live parent states; a cached pose or latest-parent-only correction
  // would fail one of these two assertions.
  parents[0]->pose3 = correction0.compose(raw);
  parents[1]->pose3 = correction1.compose(raw);
  CHECK(mapping.pose(0, parents).equals(correction0.compose(raw), 1e-10), "old-parent correction missing");
  CHECK(mapping.pose(3, parents).equals(correction1.compose(raw), 1e-10), "new-parent correction missing");
  CHECK(mapping.anchors()[0].time.sec == 100 && mapping.anchors()[3].time.sec == 104,
        "graph correction changed immutable anchor timestamps");

  CHECK(!mapping.observe(at(0), raw, 0).accepted, "zero stamp accepted");
  CHECK(!mapping.observe(at(-1), raw, 0).accepted, "negative stamp accepted");
  CHECK(!mapping.observe(at(110, 1000000000), raw, 0).accepted, "unnormalized stamp accepted");
  CHECK(!mapping.observe(at(104, 400000000), raw, 0).accepted, "duplicate stamp accepted");
  const gtsam::Pose3 bad(gtsam::Rot3(), gtsam::Point3(std::numeric_limits<double>::quiet_NaN(), 0, 0));
  CHECK(!mapping.observe(at(110), bad, 0).accepted && !mapping.flush(), "invalid pose changed tail");

  mapping.clear();
  CHECK(mapping.anchors().empty() && mapping.pending() == 0 && !mapping.flush(), "rewind retained history/tail");
  // No sleeps: accelerated replay must be governed by acquisition time.
  for (int i = 0; i <= 60; ++i)
    mapping.observe(at(90 + i / 5, (i % 5) * 200000000U), raw, 0);
  CHECK(mapping.anchors().size() == 6 && mapping.pending() == 1, "accelerated source cadence failed");
  CHECK(mapping.flush() && mapping.anchors().size() == 7, "accelerated final tail missing");
  for (std::size_t i = 0; i < mapping.anchors().size(); ++i)
    CHECK(mapping.anchors()[i].time.sec == 90 + static_cast<int>(2 * i), "cadence/index/stamp drifted");
  std::puts("PASS: delayed cadence, exact tail, parent corrections, invalid input and rewind");
}
