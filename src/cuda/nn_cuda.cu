// Brute-force 1-nearest-neighbour for the overlap/correspondence queries
// (cloud_ops match()): one thread per query point, reference cloud streamed
// through shared memory. Exact — every reference point is scanned, so unlike
// an approximate tree there is no epsilon; ties keep the lowest reference
// index (strict < during the scan). Matches libnabo's contract: id -1 and
// dist2 inf when the nearest point is beyond max_dist.
#include "cuda_common.cuh"
#include "sonar_slam_cpp/gpu.hpp"

#include <math_constants.h>

#include <mutex>

namespace sonar_slam {
namespace gpu {

namespace {

__global__ void nn1_kernel(const float2* __restrict__ ref, int n_ref,
                           const float2* __restrict__ query, int n_query,
                           float max_dist2, int* __restrict__ ids,
                           float* __restrict__ dists2)
{
  extern __shared__ float2 tile[];
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  // out-of-range threads still walk the tile loop (uniform __syncthreads),
  // they just never write a result
  const float2 qp = q < n_query ? query[q] : make_float2(0.f, 0.f);

  float best = CUDART_INF_F;
  int best_id = -1;
  for (int base = 0; base < n_ref; base += blockDim.x) {
    const int j = base + static_cast<int>(threadIdx.x);
    if (j < n_ref) tile[threadIdx.x] = ref[j];
    __syncthreads();
    const int m = min(static_cast<int>(blockDim.x), n_ref - base);
    for (int t = 0; t < m; ++t) {
      const float dx = qp.x - tile[t].x;
      const float dy = qp.y - tile[t].y;
      const float d2 = dx * dx + dy * dy;
      if (d2 < best) {
        best = d2;
        best_id = base + t;
      }
    }
    __syncthreads();
  }

  if (q < n_query) {
    if (best_id >= 0 && best <= max_dist2) {
      ids[q] = best_id;
      dists2[q] = best;
    } else {
      ids[q] = -1;
      dists2[q] = CUDART_INF_F;
    }
  }
}

}  // namespace

bool nn1_cuda(const float* ref_xy, int n_ref, const float* query_xy,
              int n_query, float max_dist, int* ids_out, float* dists2_out)
{
  if (n_ref <= 0 || n_query <= 0) return false;

  static std::mutex mutex;
  static detail::DeviceBuffer ref_buf, query_buf, ids_buf, dists_buf;
  std::lock_guard<std::mutex> lock(mutex);

  if (!ref_buf.ensure(2 * n_ref * sizeof(float), "nn1_cuda ref") ||
      !query_buf.ensure(2 * n_query * sizeof(float), "nn1_cuda query") ||
      !ids_buf.ensure(n_query * sizeof(int), "nn1_cuda ids") ||
      !dists_buf.ensure(n_query * sizeof(float), "nn1_cuda dists"))
    return false;

  if (!detail::check(cudaMemcpy(ref_buf.as<float>(), ref_xy,
                                2 * n_ref * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "nn1_cuda ref upload") ||
      !detail::check(cudaMemcpy(query_buf.as<float>(), query_xy,
                                2 * n_query * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "nn1_cuda query upload"))
    return false;

  const int threads = 256;
  const int blocks = (n_query + threads - 1) / threads;
  const std::size_t shmem = threads * sizeof(float2);
  nn1_kernel<<<blocks, threads, shmem>>>(
    ref_buf.as<float2>(), n_ref, query_buf.as<float2>(), n_query,
    max_dist * max_dist, ids_buf.as<int>(), dists_buf.as<float>());
  if (!detail::check(cudaGetLastError(), "nn1_cuda launch")) return false;

  if (!detail::check(cudaMemcpy(ids_out, ids_buf.as<int>(),
                                n_query * sizeof(int), cudaMemcpyDeviceToHost),
                     "nn1_cuda ids download") ||
      !detail::check(cudaMemcpy(dists2_out, dists_buf.as<float>(),
                                n_query * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "nn1_cuda dists download"))
    return false;

  return true;
}

}  // namespace gpu
}  // namespace sonar_slam
