// Polar -> Cartesian remap of uint8 images with float32 coordinate maps.
// Matches cv::remap semantics with BORDER_CONSTANT(0): dst(r,c) samples
// src(map_y(r,c), map_x(r,c)); negative map values (the fill for out-of-fan
// pixels) fall outside and produce 0.
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

__device__ inline float sample_or_zero(const std::uint8_t* src, int rows,
                                       int cols, int r, int c)
{
  if (r < 0 || r >= rows || c < 0 || c >= cols) return 0.f;
  return static_cast<float>(src[static_cast<std::size_t>(r) * cols + c]);
}

__global__ void remap_kernel(const std::uint8_t* __restrict__ src, int src_rows,
                             int src_cols, const float* __restrict__ map_x,
                             const float* __restrict__ map_y, int dst_rows,
                             int dst_cols, int interp,
                             std::uint8_t* __restrict__ dst)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= dst_cols || r >= dst_rows) return;

  const std::size_t idx = static_cast<std::size_t>(r) * dst_cols + c;
  const float x = map_x[idx];
  const float y = map_y[idx];

  float value = 0.f;
  if (interp == 0) {  // nearest
    value = sample_or_zero(src, src_rows, src_cols,
                           __float2int_rn(y), __float2int_rn(x));
  } else {  // bilinear
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float fx = x - x0, fy = y - y0;
    const float v00 = sample_or_zero(src, src_rows, src_cols, y0, x0);
    const float v01 = sample_or_zero(src, src_rows, src_cols, y0, x0 + 1);
    const float v10 = sample_or_zero(src, src_rows, src_cols, y0 + 1, x0);
    const float v11 = sample_or_zero(src, src_rows, src_cols, y0 + 1, x0 + 1);
    value = (1.f - fy) * ((1.f - fx) * v00 + fx * v01) +
            fy * ((1.f - fx) * v10 + fx * v11);
  }
  dst[idx] = static_cast<std::uint8_t>(fminf(fmaxf(value + 0.5f, 0.f), 255.f));
}

}  // namespace

void remap_u8_cuda(const std::uint8_t* src, int src_rows, int src_cols,
                   const float* map_x, const float* map_y, int dst_rows,
                   int dst_cols, int interp, std::uint8_t* dst)
{
  const std::size_t n_src = static_cast<std::size_t>(src_rows) * src_cols;
  const std::size_t n_dst = static_cast<std::size_t>(dst_rows) * dst_cols;

  std::uint8_t *d_src = nullptr, *d_dst = nullptr;
  float *d_mx = nullptr, *d_my = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, n_src));
  CUDA_CHECK(cudaMalloc(&d_dst, n_dst));
  CUDA_CHECK(cudaMalloc(&d_mx, n_dst * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_my, n_dst * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, src, n_src, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mx, map_x, n_dst * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_my, map_y, n_dst * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 block(32, 8);
  const dim3 grid((dst_cols + block.x - 1) / block.x,
                  (dst_rows + block.y - 1) / block.y);
  remap_kernel<<<grid, block>>>(d_src, src_rows, src_cols, d_mx, d_my, dst_rows,
                                dst_cols, interp, d_dst);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(dst, d_dst, n_dst, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  CUDA_CHECK(cudaFree(d_mx));
  CUDA_CHECK(cudaFree(d_my));
}

}  // namespace gpu
}  // namespace sonar_slam
