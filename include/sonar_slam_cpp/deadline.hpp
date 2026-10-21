#pragma once

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace sonar_slam {
using Deadline = std::chrono::steady_clock::time_point;

inline bool deadline_expired(Deadline deadline)
{
  return std::chrono::steady_clock::now() >= deadline;
}

// Round down: never grant a nested operation more time than its caller has.
inline int remaining_ms(Deadline deadline, int limit)
{
  if (deadline == Deadline::max()) return limit;
  const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
    deadline - std::chrono::steady_clock::now()).count();
  return static_cast<int>(std::clamp<decltype(left)>(left, 0, limit));
}

class DeadlineExceeded : public std::runtime_error
{
public:
  DeadlineExceeded() : std::runtime_error("registration deadline exceeded") {}
};
}  // namespace sonar_slam
