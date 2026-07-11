// Batched grid-overlap cost for global scan-match initialization: one block
// per candidate transform, threads stride over the source points, block-wide
// reduction of the hit count. Mirrors the cost of slam.py's
// get_matching_cost_subroutine1.
#include "cuda_common.cuh"
#include "sonar_slam_cpp/gpu.hpp"

#include <cstdint>
#include <mutex>

namespace sonar_slam {
namespace gpu {

namespace {

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
  // multiply by the reciprocal and round-to-nearest-even, the same arithmetic
  // as the CPU twin (and np.round), so both paths score candidates identically
  const float inv_res = 1.0f / resolution;
  int hits = 0;
  for (int p = threadIdx.x; p < n_points; p += blockDim.x) {
    const float px = points[2 * p], py = points[2 * p + 1];
    const float x = T[0] * px + T[1] * py + T[4];
    const float y = T[2] * px + T[3] * py + T[5];
    const int r = __float2int_rn((y - ymin) * inv_res);
    const int c = __float2int_rn((x - xmin) * inv_res);
    if (r >= 0 && r < grid_rows && c >= 0 && c < grid_cols &&
        grid[static_cast<std::size_t>(r) * grid_cols + c] > 0)
      ++hits;
  }
  atomicAdd(&block_hits, hits);
  __syncthreads();
  if (threadIdx.x == 0) costs[blockIdx.x] = -static_cast<float>(block_hits);
}

}  // namespace

bool grid_cost_cuda(const float* points, int n_points, const float* transforms,
                    int n_samples, const std::uint8_t* grid, int grid_rows,
                    int grid_cols, float xmin, float ymin, float resolution,
                    float* costs_out)
{
  static std::mutex mutex;
  static detail::DeviceBuffer points_buf, transforms_buf, costs_buf, grid_buf;
  std::lock_guard<std::mutex> lock(mutex);

  const std::size_t n_grid = static_cast<std::size_t>(grid_rows) * grid_cols;
  if (!points_buf.ensure(2 * n_points * sizeof(float), "grid_cost_cuda points") ||
      !transforms_buf.ensure(6 * n_samples * sizeof(float),
                             "grid_cost_cuda transforms") ||
      !costs_buf.ensure(n_samples * sizeof(float), "grid_cost_cuda costs") ||
      !grid_buf.ensure(n_grid, "grid_cost_cuda grid"))
    return false;

  if (!detail::check(cudaMemcpy(points_buf.as<float>(), points,
                                2 * n_points * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "grid_cost_cuda points upload") ||
      !detail::check(cudaMemcpy(transforms_buf.as<float>(), transforms,
                                6 * n_samples * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "grid_cost_cuda transforms upload") ||
      !detail::check(cudaMemcpy(grid_buf.as<std::uint8_t>(), grid, n_grid,
                                cudaMemcpyHostToDevice),
                     "grid_cost_cuda grid upload"))
    return false;

  const int threads = 128;
  grid_cost_kernel<<<n_samples, threads>>>(
    points_buf.as<float>(), n_points, transforms_buf.as<float>(),
    grid_buf.as<std::uint8_t>(), grid_rows, grid_cols, xmin, ymin, resolution,
    costs_buf.as<float>());
  if (!detail::check(cudaGetLastError(), "grid_cost_cuda launch")) return false;

  return detail::check(cudaMemcpy(costs_out, costs_buf.as<float>(),
                                  n_samples * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "grid_cost_cuda download");
}

}  // namespace gpu
}  // namespace sonar_slam
