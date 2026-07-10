// Runtime GPU dispatch. Every CUDA kernel in this package has a bit-exact (or
// numerically equivalent) CPU twin; gpu::available() decides once per process
// which path runs. Force the CPU path with SONAR_SLAM_FORCE_CPU=1.
#pragma once

#include <cstdint>

namespace sonar_slam {
namespace gpu {

// True when the package was built with CUDA, a device is present, and
// SONAR_SLAM_FORCE_CPU is not set. Cached after the first call.
bool available();

#ifdef SONAR_SLAM_WITH_CUDA
// --- raw kernel entry points (defined in src/cuda/*.cu) ----------------------
// CFAR over a polar image (row-major, rows = range bins, cols = beams).
// alg: 0=CA 1=SOCA 2=GOCA 3=OS. Output mask is 0/1 uint8.
void cfar_cuda(const float* img, int rows, int cols, int alg, int train_hs,
               int guard_hs, int rank, double tau, std::uint8_t* mask_out);

// Nearest / bilinear remap of a uint8 image with float32 maps (cv::remap
// semantics, constant 0 border). interp: 0=nearest, 1=linear.
void remap_u8_cuda(const std::uint8_t* src, int src_rows, int src_cols,
                   const float* map_x, const float* map_y, int dst_rows,
                   int dst_cols, int interp, std::uint8_t* dst);

// Batched grid-overlap cost for the global scan-match initialization.
// transforms: n_samples x 6 row-major [r00 r01 r10 r11 tx ty]; grid is a 0/255
// occupancy image. costs_out[i] = -(number of transformed points landing on a
// nonzero grid cell), matching slam.py's matching-cost subroutine.
void grid_cost_cuda(const float* points, int n_points, const float* transforms,
                    int n_samples, const std::uint8_t* grid, int grid_rows,
                    int grid_cols, float xmin, float ymin, float resolution,
                    float* costs_out);
#endif

}  // namespace gpu
}  // namespace sonar_slam
