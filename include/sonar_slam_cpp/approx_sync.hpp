// Slop-based approximate time synchronizer over normalized reading structs,
// mirroring rospy's ApproximateTimeSynchronizer semantics: the first queue is
// the pacing (primary) stream; each of its messages is paired with the
// nearest message within `slop` seconds from every secondary queue. A match
// is emitted once every secondary stream has advanced past the primary stamp
// (or its queue is deep enough that the nearest neighbor is final).
//
// Thread safety: add_* calls are serialized on an internal mutex, so the
// synchronizer is safe under multithreaded executors. The callback fires
// while that mutex is held — callbacks must not call back into the same
// synchronizer.
#pragma once

#include <cmath>
#include <deque>
#include <functional>
#include <mutex>
#include <utility>

namespace sonar_slam {

namespace detail {

template <typename T>
struct Stamped
{
  double t;
  T value;
};

// Returns a POINTER into the queue (null when nothing lies within slop) rather
// than a copy of the match — the callers only read it, and the queued value is
// a whole ROS message. Valid until the queue is mutated, which the callers do
// only after they are done with it.
template <typename T>
const Stamped<T>* nearest_within(const std::deque<Stamped<T>>& q, double t,
                                 double slop)
{
  const Stamped<T>* best = nullptr;
  double best_dt = slop;
  for (const auto& item : q) {
    const double dt = std::abs(item.t - t);
    if (dt <= best_dt) {
      best_dt = dt;
      best = &item;
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

// Secondary-queue trim: evict by TIME with the configured count as a floor.
// A count-only trim makes the retained window rate-dependent — above
// queue_size/(2*slop) Hz the true nearest neighbour is evicted before the
// primary can emit, and every pair silently rides the window edge. Entries
// older than `keep_after` (oldest pending primary − slop) can never match a
// pending or future in-order primary; `max_size` only bounds memory when the
// primary stream stalls.
template <typename T>
void trim_secondary(std::deque<Stamped<T>>& q, double keep_after,
                    std::size_t min_keep, std::size_t max_size)
{
  while (q.size() > max_size) q.pop_front();
  while (q.size() > min_keep && q.front().t < keep_after) q.pop_front();
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

  // Fires (while the internal mutex is held) with the count of primary messages
  // silently dropped because a secondary stream stalled and the primary queue
  // overflowed. Lets the caller surface the outage instead of losing it quietly.
  void set_overflow_callback(std::function<void(std::size_t)> cb)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    on_overflow_ = std::move(cb);
  }

  // Fires (mutex held) once per primary popped WITHOUT a match: the secondary
  // stream advanced past it but no sample lay within slop — the signature of a
  // constant cross-device stamp offset larger than slop, which otherwise
  // starves the consumer with no symptom at all.
  void set_nomatch_callback(std::function<void(std::size_t)> cb)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    on_nomatch_ = std::move(cb);
  }

  void add_primary(double t, const A& a)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // time jumped backwards (bag loop / restart): stale large stamps would
    // block try_emit's pass-gate for the whole looped pass — restart clean
    if (!qa_.empty() && t < qa_.front().t) { qa_.clear(); qb_.clear(); }
    qa_.push_back({t, a});
    if (qa_.size() > queue_size_) {
      const std::size_t dropped = qa_.size() - queue_size_;
      detail::trim(qa_, queue_size_);
      if (on_overflow_) on_overflow_(dropped);
    }
    try_emit();
  }

  void add_secondary(double t, const B& b)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!qb_.empty() && t < qb_.front().t) qb_.clear();
    qb_.push_back({t, b});
    const double keep_after = (qa_.empty() ? t : qa_.front().t) - slop_;
    detail::trim_secondary(qb_, keep_after, queue_size_, kMaxSecondary);
    try_emit();
  }

private:
  static constexpr std::size_t kMaxSecondary = 4096;  // stall memory bound

  void try_emit()
  {
    while (!qa_.empty()) {
      const auto& head = qa_.front();
      // wait until the secondary stream has passed head.t + slop, so the
      // nearest neighbor is final
      if (qb_.empty() || qb_.back().t < head.t + slop_) return;
      const auto* match = detail::nearest_within(qb_, head.t, slop_);
      if (match) {
        const double match_t = match->t;  // read before qb_ is mutated
        cb_(head.value, match->value);
        // keep the matched secondary (drop only strictly-older ones): a dense
        // secondary (e.g. odometry) may be the nearest neighbour of several
        // primaries. This deliberately diverges from message_filters ATS,
        // which consumes a matched message — here we prefer pairing every
        // primary (each a keyframe candidate) over one-shot secondary use.
        detail::drop_older(qb_, match_t - slop_);
      } else if (on_nomatch_) {
        // the secondary passed this primary with nothing inside slop: a
        // cross-device stamp offset > slop reaches steady state HERE, with
        // zero emitted pairs — surface it
        on_nomatch_(1);
      }
      qa_.pop_front();
    }
  }

  std::size_t queue_size_;
  double slop_;
  Callback cb_;
  std::function<void(std::size_t)> on_overflow_;
  std::function<void(std::size_t)> on_nomatch_;
  std::mutex mutex_;
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

  // Fires (while the internal mutex is held) with the count of primary messages
  // dropped because a secondary stalled and the primary queue overflowed.
  void set_overflow_callback(std::function<void(std::size_t)> cb)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    on_overflow_ = std::move(cb);
  }

  // Fires (mutex held) once per primary popped WITHOUT a full match — the
  // cross-device stamp-offset starvation signature (see ApproxSync2).
  void set_nomatch_callback(std::function<void(std::size_t)> cb)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    on_nomatch_ = std::move(cb);
  }

  void add_primary(double t, const A& a)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // backward time jump (bag loop / restart): restart clean, see ApproxSync2
    if (!qa_.empty() && t < qa_.front().t) {
      qa_.clear();
      qb_.clear();
      qc_.clear();
    }
    qa_.push_back({t, a});
    if (qa_.size() > queue_size_) {
      const std::size_t dropped = qa_.size() - queue_size_;
      detail::trim(qa_, queue_size_);
      if (on_overflow_) on_overflow_(dropped);
    }
    try_emit();
  }

  void add_secondary_b(double t, const B& b)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!qb_.empty() && t < qb_.front().t) qb_.clear();
    qb_.push_back({t, b});
    const double keep_after = (qa_.empty() ? t : qa_.front().t) - slop_;
    detail::trim_secondary(qb_, keep_after, queue_size_, kMaxSecondary);
    try_emit();
  }

  void add_secondary_c(double t, const C& c)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!qc_.empty() && t < qc_.front().t) qc_.clear();
    qc_.push_back({t, c});
    const double keep_after = (qa_.empty() ? t : qa_.front().t) - slop_;
    detail::trim_secondary(qc_, keep_after, queue_size_, kMaxSecondary);
    try_emit();
  }

private:
  static constexpr std::size_t kMaxSecondary = 4096;  // stall memory bound

  void try_emit()
  {
    while (!qa_.empty()) {
      const auto& head = qa_.front();
      if (qb_.empty() || qb_.back().t < head.t + slop_) return;
      if (qc_.empty() || qc_.back().t < head.t + slop_) return;
      const auto* mb = detail::nearest_within(qb_, head.t, slop_);
      const auto* mc = detail::nearest_within(qc_, head.t, slop_);
      if (mb && mc) {
        const double mb_t = mb->t, mc_t = mc->t;  // read before the queues move
        cb_(head.value, mb->value, mc->value);
        detail::drop_older(qb_, mb_t - slop_);
        detail::drop_older(qc_, mc_t - slop_);
      } else if (on_nomatch_) {
        on_nomatch_(1);
      }
      qa_.pop_front();
    }
  }

  std::size_t queue_size_;
  double slop_;
  Callback cb_;
  std::function<void(std::size_t)> on_overflow_;
  std::function<void(std::size_t)> on_nomatch_;
  std::mutex mutex_;
  std::deque<detail::Stamped<A>> qa_;
  std::deque<detail::Stamped<B>> qb_;
  std::deque<detail::Stamped<C>> qc_;
};

}  // namespace sonar_slam
