// Shared helpers for the CUDA wrappers: error checking that reports the
// failure and lets the caller fall back to the CPU twin, plus grow-only
// device buffers reused across calls (per-call cudaMalloc/cudaFree at ping
// rate is measurable overhead).
#pragma once

#include <cuda_runtime.h>

#include "sonar_slam_cpp/gpu.hpp"

#include <cstddef>
#include <cstdio>
#include <initializer_list>

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

// Grow-only device allocation reused across calls. Instances are static in the
// wrappers and are never freed at teardown: the CUDA context reclaims device
// memory at process exit, and freeing from a static destructor would race
// context teardown. release() is for use while the context is live.
class DeviceBuffer
{
public:
  bool ensure(std::size_t bytes, const char* what)
  {
    if (bytes <= capacity_) return true;
    // Release before allocating, not after. The replacement is strictly
    // larger, so holding both at once doubles the peak and halves the size
    // this can service on a device that is already short of memory.
    release();
    const cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
      ptr_ = nullptr;
      // Report the device's own accounting next to the request. The size
      // alone is not diagnostic -- a 2 MB failure on an 8 GB card looks
      // absurd either way. "2 MB requested, 3 MB free of 8 GB" says the
      // device is contended and names the scale of the shortfall; "2 MB
      // requested, 7 GB free" says the ceiling is per-process (an RLIMIT_AS
      // or cgroup cap, the usual container case, since CUDA reserves tens of
      // GB of address space up front) and that freeing VRAM elsewhere cannot
      // help. Those two need opposite fixes, so print the number that
      // distinguishes them rather than making the reader go find it while
      // the failure is still live.
      std::size_t free_bytes = 0, total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        free_bytes = total_bytes = 0;
      std::fprintf(stderr,
                   "sonar_slam_cpp CUDA allocation failed for %s: requested "
                   "%zu bytes, %s; device reports %zu of %zu bytes free\n",
                   what, bytes, cudaGetErrorString(err), free_bytes,
                   total_bytes);
      // cudaMalloc's failure is also latched in the runtime's per-thread
      // error state. Consume it so a later cudaGetLastError() -- the launch
      // check in the wrappers -- does not re-report this allocation as a
      // kernel launch failure.
      cudaGetLastError();
      disable("device allocation failure");
      return false;
    }
    capacity_ = bytes;
    return true;
  }

  // Free the allocation and return to the empty state; safe on an already
  // empty buffer. Only called while the CUDA context is live (from ensure()
  // and ensure_all()), never from a destructor -- see the class comment.
  void release()
  {
    if (ptr_ == nullptr) return;
    cudaFree(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
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

struct BufferRequest
{
  DeviceBuffer* buf;
  std::size_t bytes;
  const char* what;
};

// Acquire a whole working set or none of it.
//
// Allocating piecemeal and bailing on the first failure leaves the buffers
// that DID fit holding device memory that the CPU fallback cannot use -- on a
// device that just proved it is short. Because the wrappers' buffers are
// static, that memory stays pinned for the life of the process, so the
// partial set makes the next allocation by anyone (including this process)
// strictly less likely to succeed. Roll the set back instead.
inline bool ensure_all(std::initializer_list<BufferRequest> requests)
{
  for (const BufferRequest& r : requests) {
    if (r.buf->ensure(r.bytes, r.what)) continue;
    for (const BufferRequest& s : requests) s.buf->release();
    return false;
  }
  return true;
}

}  // namespace detail
}  // namespace gpu
}  // namespace sonar_slam
