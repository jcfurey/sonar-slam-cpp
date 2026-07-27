// CUDA CFAR kernels — one thread per pixel, window along the range axis (rows)
// within each beam column. Mirrors the arithmetic of cpp/cfar.cpp exactly.
#include "cuda_common.cuh"
#include "sonar_slam_cpp/gpu.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>

namespace sonar_slam {
namespace gpu {

namespace {

// Sample fetch. The uint8 overload widens in-register, which is EXACT and
// therefore produces the same float the host-side convertTo(CV_32F) would
// have: every value 0..255 is representable, so the whole detector downstream
// is bit-identical whichever entry point ran.
__device__ inline float sample(const float* p, std::size_t i) { return p[i]; }
__device__ inline float sample(const std::uint8_t* p, std::size_t i)
{
  return static_cast<float>(p[i]);
}

template <typename T>
__global__ void cfar_kernel(const T* __restrict__ img, int rows, int cols,
                            int alg, int train_hs, int guard_hs, int rank,
                            double tau, float threshold,
                            std::uint8_t* __restrict__ mask)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= cols || row >= rows) return;

  const int border = train_hs + guard_hs;
  const std::size_t idx = static_cast<std::size_t>(row) * cols + col;
  if (row < border || row >= rows - border) {
    mask[idx] = 0;
    return;
  }

  const float cell = sample(img, idx);
  bool hit = false;

  if (alg == 0) {  // CA
    float sum_train = 0.f;
    for (int i = row - border; i <= row + border; ++i) {
      const int d = i - row;
      if (d > guard_hs || d < -guard_hs)
        sum_train += sample(img, static_cast<std::size_t>(i) * cols + col);
    }
    hit = cell > tau * sum_train / (2.0 * train_hs);
  } else if (alg == 1 || alg == 2) {  // SOCA / GOCA
    float leading = 0.f, lagging = 0.f;
    for (int i = row - border; i <= row + border; ++i) {
      const int d = i - row;
      if (d > guard_hs)
        lagging += sample(img, static_cast<std::size_t>(i) * cols + col);
      else if (d < -guard_hs)
        leading += sample(img, static_cast<std::size_t>(i) * cols + col);
    }
    const float sum_train = alg == 1 ? fminf(leading, lagging) : fmaxf(leading, lagging);
    hit = cell > tau * sum_train / train_hs;
  } else {  // OS: k-th smallest of the training cells
    float train[kMaxOsTrainCells];
    int n = 0;
    for (int i = row - border; i <= row + border && n < kMaxOsTrainCells; ++i) {
      const int d = i - row;
      if (d > guard_hs || d < -guard_hs)
        train[n++] = sample(img, static_cast<std::size_t>(i) * cols + col);
    }
    // selection by counting: value with exactly `rank` smaller elements
    // (partial selection sort — n is small, at most Ntc)
    for (int k = 0; k <= rank; ++k) {
      int min_j = k;
      for (int j = k + 1; j < n; ++j)
        if (train[j] < train[min_j]) min_j = j;
      const float tmp = train[k];
      train[k] = train[min_j];
      train[min_j] = tmp;
    }
    hit = cell > tau * train[rank];
  }

  // fold the intensity gate in so the kernel matches detect_cpu exactly
  mask[idx] = (hit && cell > threshold) ? 1 : 0;
}

// Device buffers and the serializing mutex are shared by both entry points:
// only one of them runs per image in a given process, and a template-local
// `static` would give each instantiation its own grow-only allocation.
std::mutex g_mutex;
detail::DeviceBuffer g_img_buf, g_mask_buf;

template <typename T>
bool cfar_launch(const T* img, int rows, int cols, int alg, int train_hs,
                 int guard_hs, int rank, double tau, float threshold,
                 const char* what_img, std::uint8_t* mask_out)
{
  // the OS kernel's selection buffer is fixed-size; larger training windows
  // must run on the CPU rather than silently truncate
  if (alg == 3 && 2 * train_hs > kMaxOsTrainCells) {
    // atomic so the one-shot warning is race-free without taking the buffer
    // mutex on this early-return path
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
      std::fprintf(stderr,
                   "sonar_slam_cpp: OS-CFAR Ntc %d exceeds the GPU limit %d, "
                   "using the CPU path\n",
                   2 * train_hs, kMaxOsTrainCells);
    return false;
  }

  std::lock_guard<std::mutex> lock(g_mutex);

  const std::size_t n_img = static_cast<std::size_t>(rows) * cols;
  if (!detail::ensure_all({{&g_img_buf, n_img * sizeof(T), what_img},
                           {&g_mask_buf, n_img, "cfar_cuda mask"}}))
    return false;
  if (!detail::check(cudaMemcpy(g_img_buf.as<T>(), img, n_img * sizeof(T),
                                cudaMemcpyHostToDevice),
                     "cfar_cuda upload"))
    return false;

  const dim3 block(32, 8);
  const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  cfar_kernel<T><<<grid, block>>>(g_img_buf.as<T>(), rows, cols, alg, train_hs,
                                  guard_hs, rank, tau, threshold,
                                  g_mask_buf.as<std::uint8_t>());
  if (!detail::check(cudaGetLastError(), "cfar_cuda launch")) return false;

  return detail::check(cudaMemcpy(mask_out, g_mask_buf.as<std::uint8_t>(), n_img,
                                  cudaMemcpyDeviceToHost),
                       "cfar_cuda download");
}

}  // namespace

bool cfar_cuda(const float* img, int rows, int cols, int alg, int train_hs,
               int guard_hs, int rank, double tau, float threshold,
               std::uint8_t* mask_out)
{
  return cfar_launch(img, rows, cols, alg, train_hs, guard_hs, rank, tau,
                     threshold, "cfar_cuda img (float)", mask_out);
}

bool cfar_u8_cuda(const std::uint8_t* img, int rows, int cols, int alg,
                  int train_hs, int guard_hs, int rank, double tau,
                  float threshold, std::uint8_t* mask_out)
{
  return cfar_launch(img, rows, cols, alg, train_hs, guard_hs, rank, tau,
                     threshold, "cfar_cuda img (uint8)", mask_out);
}

}  // namespace gpu
}  // namespace sonar_slam
