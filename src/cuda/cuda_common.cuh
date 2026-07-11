// Shared helpers for the CUDA wrappers: error checking that reports the
// failure and lets the caller fall back to the CPU twin, plus grow-only
// device buffers reused across calls (per-call cudaMalloc/cudaFree at ping
// rate is measurable overhead).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

namespace sonar_slam {
namespace gpu {
namespace detail {

inline bool check(cudaError_t err, const char* what)
{
  if (err == cudaSuccess) return true;
  std::fprintf(stderr,
               "sonar_slam_cpp CUDA error in %s: %s (falling back to CPU)\n",
               what, cudaGetErrorString(err));
  return false;
}

// Grow-only device allocation reused across calls. Instances are static in
// the wrappers and intentionally never freed: the CUDA context reclaims
// device memory at process exit, and freeing from a static destructor would
// race context teardown.
class DeviceBuffer
{
public:
  bool ensure(std::size_t bytes, const char* what)
  {
    if (bytes <= capacity_) return true;
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      capacity_ = 0;
    }
    if (!check(cudaMalloc(&ptr_, bytes), what)) {
      ptr_ = nullptr;
      return false;
    }
    capacity_ = bytes;
    return true;
  }

  template <typename T>
  T* as() const
  {
    return static_cast<T*>(ptr_);
  }

private:
  void* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace detail
}  // namespace gpu
}  // namespace sonar_slam
