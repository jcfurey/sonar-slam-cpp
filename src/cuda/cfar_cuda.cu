// CUDA CFAR kernels — one thread per pixel, window along the range axis (rows)
// within each beam column. Mirrors the arithmetic of cpp/cfar.cpp exactly.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace sonar_slam {
namespace gpu {

namespace {

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "sonar_slam_cpp CUDA error %s at %s:%d\n",             \
              cudaGetErrorString(err__), __FILE__, __LINE__);                \
    }                                                                        \
  } while (0)

__global__ void cfar_kernel(const float* __restrict__ img, int rows, int cols,
                            int alg, int train_hs, int guard_hs, int rank,
                            double tau, std::uint8_t* __restrict__ mask)
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

  const float cell = img[idx];
  bool hit = false;

  if (alg == 0) {  // CA
    float sum_train = 0.f;
    for (int i = row - border; i <= row + border; ++i) {
      const int d = i - row;
      if (d > guard_hs || d < -guard_hs)
        sum_train += img[static_cast<std::size_t>(i) * cols + col];
    }
    hit = cell > tau * sum_train / (2.0 * train_hs);
  } else if (alg == 1 || alg == 2) {  // SOCA / GOCA
    float leading = 0.f, lagging = 0.f;
    for (int i = row - border; i <= row + border; ++i) {
      const int d = i - row;
      if (d > guard_hs)
        lagging += img[static_cast<std::size_t>(i) * cols + col];
      else if (d < -guard_hs)
        leading += img[static_cast<std::size_t>(i) * cols + col];
    }
    const float sum_train = alg == 1 ? fminf(leading, lagging) : fmaxf(leading, lagging);
    hit = cell > tau * sum_train / train_hs;
  } else {  // OS: k-th smallest of the training cells (rank < Ntc <= 128)
    float train[128];
    int n = 0;
    for (int i = row - border; i <= row + border && n < 128; ++i) {
      const int d = i - row;
      if (d > guard_hs || d < -guard_hs)
        train[n++] = img[static_cast<std::size_t>(i) * cols + col];
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

  mask[idx] = hit ? 1 : 0;
}

}  // namespace

void cfar_cuda(const float* img, int rows, int cols, int alg, int train_hs,
               int guard_hs, int rank, double tau, std::uint8_t* mask_out)
{
  const std::size_t n_img = static_cast<std::size_t>(rows) * cols;
  float* d_img = nullptr;
  std::uint8_t* d_mask = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img, n_img * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mask, n_img));
  CUDA_CHECK(cudaMemcpy(d_img, img, n_img * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  cfar_kernel<<<grid, block>>>(d_img, rows, cols, alg, train_hs, guard_hs, rank,
                               tau, d_mask);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(mask_out, d_mask, n_img, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_img));
  CUDA_CHECK(cudaFree(d_mask));
}

}  // namespace gpu
}  // namespace sonar_slam
