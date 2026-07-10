// Batched grid-overlap cost for global scan-match initialization: one block
// per candidate transform, threads stride over the source points, block-wide
// reduction of the hit count. Mirrors the cost of slam.py's
// get_matching_cost_subroutine1.
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

__global__ void grid_cost_kernel(const float* __restrict__ points, int n_points,
                                 const float* __restrict__ transforms,
                                 const std::uint8_t* __restrict__ grid,
                                 int grid_rows, int grid_cols, float xmin,
                                 float ymin, float resolution,
                                 float* __restrict__ costs)
{
  __shared__ int block_hits;
  if (threadIdx.x == 0) block_hits = 0;
  __syncthreads();

  const float* T = transforms + blockIdx.x * 6;  // [r00 r01 r10 r11 tx ty]
  int hits = 0;
  for (int p = threadIdx.x; p < n_points; p += blockDim.x) {
    const float px = points[2 * p], py = points[2 * p + 1];
    const float x = T[0] * px + T[1] * py + T[4];
    const float y = T[2] * px + T[3] * py + T[5];
    const int r = __float2int_rn((y - ymin) / resolution);
    const int c = __float2int_rn((x - xmin) / resolution);
    if (r >= 0 && r < grid_rows && c >= 0 && c < grid_cols &&
        grid[static_cast<std::size_t>(r) * grid_cols + c] > 0)
      ++hits;
  }
  atomicAdd(&block_hits, hits);
  __syncthreads();
  if (threadIdx.x == 0) costs[blockIdx.x] = -static_cast<float>(block_hits);
}

}  // namespace

void grid_cost_cuda(const float* points, int n_points, const float* transforms,
                    int n_samples, const std::uint8_t* grid, int grid_rows,
                    int grid_cols, float xmin, float ymin, float resolution,
                    float* costs_out)
{
  const std::size_t n_grid = static_cast<std::size_t>(grid_rows) * grid_cols;

  float *d_points = nullptr, *d_transforms = nullptr, *d_costs = nullptr;
  std::uint8_t* d_grid = nullptr;
  CUDA_CHECK(cudaMalloc(&d_points, 2 * n_points * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_transforms, 6 * n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_costs, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grid, n_grid));
  CUDA_CHECK(cudaMemcpy(d_points, points, 2 * n_points * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_transforms, transforms, 6 * n_samples * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_grid, grid, n_grid, cudaMemcpyHostToDevice));

  const int threads = 128;
  grid_cost_kernel<<<n_samples, threads>>>(d_points, n_points, d_transforms,
                                           d_grid, grid_rows, grid_cols, xmin,
                                           ymin, resolution, d_costs);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(costs_out, d_costs, n_samples * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(d_transforms));
  CUDA_CHECK(cudaFree(d_costs));
  CUDA_CHECK(cudaFree(d_grid));
}

}  // namespace gpu
}  // namespace sonar_slam
