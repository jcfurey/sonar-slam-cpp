// Slop-based approximate time synchronizer over normalized reading structs,
// mirroring rospy's ApproximateTimeSynchronizer semantics: the first queue is
// the pacing (primary) stream; each of its messages is paired with the
// nearest message within `slop` seconds from every secondary queue. A match
// is emitted once every secondary stream has advanced past the primary stamp
// (or its queue is deep enough that the nearest neighbor is final).
#pragma once

#include <cmath>
#include <deque>
#include <functional>
#include <optional>
#include <utility>

namespace sonar_slam {

namespace detail {

template <typename T>
struct Stamped
{
  double t;
  T value;
};

template <typename T>
std::optional<Stamped<T>> nearest_within(const std::deque<Stamped<T>>& q,
                                         double t, double slop)
{
  std::optional<Stamped<T>> best;
  double best_dt = slop;
  for (const auto& item : q) {
    const double dt = std::abs(item.t - t);
    if (dt <= best_dt) {
      best_dt = dt;
      best = item;
    }
  }
  return best;
}

template <typename T>
void drop_older(std::deque<Stamped<T>>& q, double t)
{
  while (!q.empty() && q.front().t <= t) q.pop_front();
}

template <typename T>
void trim(std::deque<Stamped<T>>& q, std::size_t max_size)
{
  while (q.size() > max_size) q.pop_front();
}

}  // namespace detail

// ------------------------------------------------------------------- 2-way
template <typename A, typename B>
class ApproxSync2
{
public:
  using Callback = std::function<void(const A&, const B&)>;

  ApproxSync2(std::size_t queue_size, double slop, Callback cb)
    : queue_size_(queue_size), slop_(slop), cb_(std::move(cb))
  {
  }

  void add_primary(double t, const A& a)
  {
    qa_.push_back({t, a});
    detail::trim(qa_, queue_size_);
    try_emit();
  }

  void add_secondary(double t, const B& b)
  {
    qb_.push_back({t, b});
    detail::trim(qb_, queue_size_);
    try_emit();
  }

private:
  void try_emit()
  {
    while (!qa_.empty()) {
      const auto& head = qa_.front();
      // wait until the secondary stream has passed head.t + slop, so the
      // nearest neighbor is final
      if (qb_.empty() || qb_.back().t < head.t + slop_) return;
      const auto match = detail::nearest_within(qb_, head.t, slop_);
      if (match) {
        cb_(head.value, match->value);
        detail::drop_older(qb_, match->t - slop_);
      }
      qa_.pop_front();
    }
  }

  std::size_t queue_size_;
  double slop_;
  Callback cb_;
  std::deque<detail::Stamped<A>> qa_;
  std::deque<detail::Stamped<B>> qb_;
};

// ------------------------------------------------------------------- 3-way
template <typename A, typename B, typename C>
class ApproxSync3
{
public:
  using Callback = std::function<void(const A&, const B&, const C&)>;

  ApproxSync3(std::size_t queue_size, double slop, Callback cb)
    : queue_size_(queue_size), slop_(slop), cb_(std::move(cb))
  {
  }

  void add_primary(double t, const A& a)
  {
    qa_.push_back({t, a});
    detail::trim(qa_, queue_size_);
    try_emit();
  }

  void add_secondary_b(double t, const B& b)
  {
    qb_.push_back({t, b});
    detail::trim(qb_, queue_size_);
    try_emit();
  }

  void add_secondary_c(double t, const C& c)
  {
    qc_.push_back({t, c});
    detail::trim(qc_, queue_size_);
    try_emit();
  }

private:
  void try_emit()
  {
    while (!qa_.empty()) {
      const auto& head = qa_.front();
      if (qb_.empty() || qb_.back().t < head.t + slop_) return;
      if (qc_.empty() || qc_.back().t < head.t + slop_) return;
      const auto mb = detail::nearest_within(qb_, head.t, slop_);
      const auto mc = detail::nearest_within(qc_, head.t, slop_);
      if (mb && mc) {
        cb_(head.value, mb->value, mc->value);
        detail::drop_older(qb_, mb->t - slop_);
        detail::drop_older(qc_, mc->t - slop_);
      }
      qa_.pop_front();
    }
  }

  std::size_t queue_size_;
  double slop_;
  Callback cb_;
  std::deque<detail::Stamped<A>> qa_;
  std::deque<detail::Stamped<B>> qb_;
  std::deque<detail::Stamped<C>> qc_;
};

}  // namespace sonar_slam
