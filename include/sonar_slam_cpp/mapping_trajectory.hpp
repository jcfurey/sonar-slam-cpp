#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "sonar_slam_cpp/keyframe.hpp"

namespace sonar_slam {

// Pose-only mapping history. Anchors carry no acoustic points or graph factors.
// Re-evaluate each anchor against its parent's CURRENT optimized pose so loop
// closures/manual corrections also move evidence acquired between graph states.
class MappingTrajectory
{
public:
  struct Anchor {
    builtin_interfaces::msg::Time time;
    gtsam::Pose3 dr_pose;
    std::size_t parent;
  };

  explicit MappingTrajectory(double interval = 2.0)
  {
    if (!std::isfinite(interval) || interval <= 0.0 || interval > 1e9)
      throw std::invalid_argument("mapping_anchor_interval must be finite, positive and <= 1e9 s");
    interval_ns_ = std::max<int64_t>(1, std::llround(interval * 1e9));
  }

  struct Observation { bool accepted; bool published; };

  Observation observe(const builtin_interfaces::msg::Time& time,
                      const gtsam::Pose3& dr_pose, std::size_t parent)
  {
    if (!dr_pose.matrix().allFinite() || ns(time) <= 0 || time.nanosec >= 1000000000U ||
        (latest_ && ns(time) <= ns(latest_->time))) return {false, false};
    latest_ = Anchor{time, dr_pose, parent};
    const Anchor* selected = pending_.empty()
      ? (anchors_.empty() ? nullptr : &anchors_.back()) : &pending_.back();
    if (!selected || ns(time) - ns(selected->time) >= interval_ns_)
      pending_.push_back(*latest_);
    bool published = false;
    // Give all cloud topics an acquisition interval to arrive before the
    // assembler advances its association watermark past the selected stamp.
    while (!pending_.empty() && ns(time) - ns(pending_.front().time) >= interval_ns_) {
      anchors_.push_back(pending_.front());
      pending_.pop_front();
      published = true;
    }
    return {true, published};
  }

  // A wall timer flushes the final partial interval using its ACQUISITION
  // stamp. With no newer valid observation this is a no-op, even if time runs.
  bool flush()
  {
    if (!latest_) return false;
    const Anchor* selected = pending_.empty()
      ? (anchors_.empty() ? nullptr : &anchors_.back()) : &pending_.back();
    if (!selected || ns(latest_->time) > ns(selected->time)) pending_.push_back(*latest_);
    const bool published = !pending_.empty();
    while (!pending_.empty()) {
      anchors_.push_back(pending_.front());
      pending_.pop_front();
    }
    return published;
  }

  gtsam::Pose3 pose(std::size_t index, const std::vector<KeyframePtr>& parents) const
  {
    const auto& anchor = anchors_.at(index);
    const auto& parent = parents.at(anchor.parent);
    return parent->pose3.compose(parent->dr_pose3.inverse()).compose(anchor.dr_pose);
  }

  const std::vector<Anchor>& anchors() const { return anchors_; }
  std::size_t pending() const { return pending_.size(); }
  void clear() { anchors_.clear(); pending_.clear(); latest_.reset(); }

private:
  static int64_t ns(const builtin_interfaces::msg::Time& time)
  {
    return static_cast<int64_t>(time.sec) * 1000000000LL + time.nanosec;
  }
  int64_t interval_ns_;
  std::optional<Anchor> latest_;
  std::vector<Anchor> anchors_;
  std::deque<Anchor> pending_;
};

}  // namespace sonar_slam
