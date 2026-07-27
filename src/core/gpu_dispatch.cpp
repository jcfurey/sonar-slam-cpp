#include "sonar_slam_cpp/gpu.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace sonar_slam {
namespace gpu {

#ifdef SONAR_SLAM_WITH_CUDA
bool device_present();  // defined in cuda/gpu_runtime.cu
#endif

namespace {

// Tripped by disable(), never reset. Relaxed ordering suffices: the flag only
// selects between two implementations that produce the same result, so a
// reader racing the trip merely takes the GPU path one more time and fails
// over through the path it already handles.
std::atomic<bool> g_disabled{false};

}  // namespace

bool available()
{
  static const bool ok = [] {
#ifdef SONAR_SLAM_WITH_CUDA
    const char* force_cpu = std::getenv("SONAR_SLAM_FORCE_CPU");
    if (force_cpu && force_cpu[0] != '\0' && force_cpu[0] != '0') return false;
    return device_present();
#else
    return false;
#endif
  }();
  return ok && !g_disabled.load(std::memory_order_relaxed);
}

void disable(const char* why)
{
  if (g_disabled.exchange(true, std::memory_order_relaxed)) return;
  std::fprintf(stderr,
               "sonar_slam_cpp: GPU path disabled for the rest of this process "
               "(%s). The CPU twins produce the same result, so output is "
               "unaffected; expect higher per-ping latency. Set "
               "SONAR_SLAM_FORCE_CPU=1 to skip the GPU entirely and silence "
               "the probe.\n",
               why != nullptr ? why : "unspecified");
}

}  // namespace gpu
}  // namespace sonar_slam
