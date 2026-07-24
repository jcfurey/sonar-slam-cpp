// Polar -> Cartesian remap of uint8 images with float32 coordinate maps.
// Matches cv::remap semantics with BORDER_CONSTANT(0): dst(r,c) samples
// src(map_y(r,c), map_x(r,c)); negative map values (the fill for out-of-fan
// pixels) fall outside and produce 0.
//
// The coordinate maps only change when the sonar geometry changes, so they
// are kept device-resident across calls, keyed by the caller's map_version.
#include "cuda_common.cuh"
#include "sonar_slam_cpp/gpu.hpp"

#include <cstdint>
#include <mutex>

namespace sonar_slam {
namespace gpu {

namespace {

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

  if (interp == 0) {  // nearest
    const float value = sample_or_zero(src, src_rows, src_cols,
                                       __float2int_rn(y), __float2int_rn(x));
    dst[idx] = static_cast<std::uint8_t>(value);
  } else {  // bilinear
    // OpenCV INTER_LINEAR first quantizes each coordinate to 5 fractional
    // bits, then applies a fixed-point interpolation table. A mathematically
    // exact float bilinear sample can differ by several uint8 counts on a
    // high-contrast image, so reproduce that quantization here instead of
    // merely approximating cv::remap.
    constexpr int kBits = 5;
    constexpr int kScale = 1 << kBits;
    const int sx = __float2int_rn(x * static_cast<float>(kScale));
    const int sy = __float2int_rn(y * static_cast<float>(kScale));
    const int x0 = sx >> kBits;
    const int y0 = sy >> kBits;
    const int ax = sx & (kScale - 1);
    const int ay = sy & (kScale - 1);

    const int v00 = static_cast<int>(
      sample_or_zero(src, src_rows, src_cols, y0, x0));
    const int v01 = static_cast<int>(
      sample_or_zero(src, src_rows, src_cols, y0, x0 + 1));
    const int v10 = static_cast<int>(
      sample_or_zero(src, src_rows, src_cols, y0 + 1, x0));
    const int v11 = static_cast<int>(
      sample_or_zero(src, src_rows, src_cols, y0 + 1, x0 + 1));

    const int value =
      (kScale - ax) * (kScale - ay) * v00 +
      ax * (kScale - ay) * v01 +
      (kScale - ax) * ay * v10 +
      ax * ay * v11;
    dst[idx] = static_cast<std::uint8_t>(
      (value + (kScale * kScale) / 2) / (kScale * kScale));
  }
}

}  // namespace

bool remap_u8_cuda(const std::uint8_t* src, int src_rows, int src_cols,
                   const float* map_x, const float* map_y, int dst_rows,
                   int dst_cols, int interp, int map_version, std::uint8_t* dst)
{
  static std::mutex mutex;
  static detail::DeviceBuffer src_buf, dst_buf, mx_buf, my_buf;
  static int cached_version = -1;
  static int cached_rows = 0, cached_cols = 0;
  // caller identity: the version counter alone is ambiguous the moment two
  // consumers share this process (composed nodes); BOTH host map pointers
  // disambiguate — map_x and map_y are independent buffers, so keying on
  // map_x alone could reuse a stale map_y upload
  static const float* cached_mx = nullptr;
  static const float* cached_my = nullptr;
  std::lock_guard<std::mutex> lock(mutex);

  const std::size_t n_src = static_cast<std::size_t>(src_rows) * src_cols;
  const std::size_t n_dst = static_cast<std::size_t>(dst_rows) * dst_cols;

  if (!src_buf.ensure(n_src, "remap_u8_cuda src") ||
      !dst_buf.ensure(n_dst, "remap_u8_cuda dst") ||
      !mx_buf.ensure(n_dst * sizeof(float), "remap_u8_cuda map_x") ||
      !my_buf.ensure(n_dst * sizeof(float), "remap_u8_cuda map_y"))
    return false;

  const bool maps_cached = map_version >= 0 && map_version == cached_version &&
                           dst_rows == cached_rows && dst_cols == cached_cols &&
                           map_x == cached_mx && map_y == cached_my;
  if (!maps_cached) {
    cached_version = -1;  // stays invalid if either upload fails
    if (!detail::check(cudaMemcpy(mx_buf.as<float>(), map_x,
                                  n_dst * sizeof(float), cudaMemcpyHostToDevice),
                       "remap_u8_cuda map_x upload") ||
        !detail::check(cudaMemcpy(my_buf.as<float>(), map_y,
                                  n_dst * sizeof(float), cudaMemcpyHostToDevice),
                       "remap_u8_cuda map_y upload"))
      return false;
    cached_version = map_version;
    cached_rows = dst_rows;
    cached_cols = dst_cols;
    cached_mx = map_x;
    cached_my = map_y;
  }

  if (!detail::check(cudaMemcpy(src_buf.as<std::uint8_t>(), src, n_src,
                                cudaMemcpyHostToDevice),
                     "remap_u8_cuda src upload"))
    return false;

  const dim3 block(32, 8);
  const dim3 grid((dst_cols + block.x - 1) / block.x,
                  (dst_rows + block.y - 1) / block.y);
  remap_kernel<<<grid, block>>>(src_buf.as<std::uint8_t>(), src_rows, src_cols,
                                mx_buf.as<float>(), my_buf.as<float>(),
                                dst_rows, dst_cols, interp,
                                dst_buf.as<std::uint8_t>());
  if (!detail::check(cudaGetLastError(), "remap_u8_cuda launch")) return false;

  return detail::check(cudaMemcpy(dst, dst_buf.as<std::uint8_t>(), n_dst,
                                  cudaMemcpyDeviceToHost),
                       "remap_u8_cuda download");
}

}  // namespace gpu
}  // namespace sonar_slam
