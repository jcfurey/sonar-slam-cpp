#pragma once

#include "sonar_slam_cpp/keyframe.hpp"
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>

namespace sonar_slam {

struct PendingPing
{
  builtin_interfaces::msg::Time time;
  gtsam::Pose3 dr_pose;
  Matrix points;
  bool usable = false;
  double head_pitch = 0.0;
  std::uint64_t epoch = 0;
  std::filesystem::path spill;
  Eigen::Index rows = 0;
  std::uint64_t checksum = 0;
};

// Mapping intake never waits for registration. Live queues discard the oldest
// unprocessed registration; reliable replay spills excess payloads to an owned
// temporary directory instead. Survey evidence is owned by the mapper/bag.
class RegistrationQueue
{
public:
  struct Stats { std::size_t pending, resident, spilled; std::uint64_t dropped; };

  RegistrationQueue(std::size_t resident_limit, bool reliable)
    : limit_(resident_limit), reliable_(reliable)
  {
    if (!limit_) throw std::invalid_argument("registration queue limit must be positive");
  }
  ~RegistrationQueue()
  {
    std::error_code error;
    if (!directory_.empty()) std::filesystem::remove_all(directory_, error);
  }

  void push(PendingPing ping)
  {
    if (ping.points.cols() != 3 || !ping.points.allFinite())
      throw std::invalid_argument("registration queue requires finite Nx3 points");
    std::lock_guard<std::mutex> lock(mutex_);
    if (!reliable_ && queue_.size() >= limit_) {
      queue_.pop_front();
      --resident_;
      ++dropped_;
    }
    if (reliable_ && resident_ >= limit_) {
      spill(ping);
    } else {
      ++resident_;
    }
    queue_.push_back(std::move(ping));
  }

  bool pop(PendingPing& ping)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.empty()) return false;
      ping = std::move(queue_.front());
      queue_.pop_front();
      if (ping.spill.empty()) --resident_;
    }
    if (!ping.spill.empty()) {
      std::ifstream file(ping.spill, std::ios::binary);
      ping.points.resize(ping.rows, 3);
      const auto bytes = static_cast<std::streamsize>(ping.points.size() * sizeof(float));
      if (!file.read(reinterpret_cast<char*>(ping.points.data()), bytes) ||
          file.peek() != std::ifstream::traits_type::eof() ||
          checksum(ping.points) != ping.checksum)
        throw std::runtime_error("registration spill is unreadable or corrupt: " + ping.spill.string());
      file.close();
      std::error_code error;
      std::filesystem::remove(ping.spill, error);
      ping.spill.clear();
    }
    return true;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code error;
    for (const auto& ping : queue_)
      if (!ping.spill.empty()) std::filesystem::remove(ping.spill, error);
    queue_.clear();
    resident_ = 0;
  }

  Stats stats() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return {queue_.size(), resident_, queue_.size() - resident_, dropped_};
  }

private:
  static std::uint64_t checksum(const Matrix& points)
  {
    const auto* bytes = reinterpret_cast<const unsigned char*>(points.data());
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < static_cast<std::size_t>(points.size()) * sizeof(float); ++i)
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
  }

  void spill(PendingPing& ping)
  {
    if (directory_.empty()) {
      std::random_device random;
      for (int i = 0; i < 16; ++i) {
        auto candidate = std::filesystem::temp_directory_path() /
          ("sonar_slam_pending_" + std::to_string(random()) + "_" + std::to_string(random()));
        if (std::filesystem::create_directory(candidate)) { directory_ = candidate; break; }
      }
      if (directory_.empty()) throw std::runtime_error("cannot reserve registration spill directory");
    }
    const auto path = directory_ / (std::to_string(sequence_++) + ".xyz");
    const auto temporary = path.string() + ".tmp";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(ping.points.data()),
      static_cast<std::streamsize>(ping.points.size() * sizeof(float)));
    file.close();
    if (!file) {
      std::error_code error;
      std::filesystem::remove(temporary, error);
      throw std::runtime_error("cannot retain registration payload in " + directory_.string());
    }
    std::filesystem::rename(temporary, path);
    ping.rows = ping.points.rows();
    ping.checksum = checksum(ping.points);
    ping.spill = path;
    ping.points.resize(0, 3);
  }

  const std::size_t limit_;
  const bool reliable_;
  mutable std::mutex mutex_;
  std::deque<PendingPing> queue_;
  std::size_t resident_ = 0;
  std::uint64_t dropped_ = 0, sequence_ = 0;
  std::filesystem::path directory_;
};

}  // namespace sonar_slam
