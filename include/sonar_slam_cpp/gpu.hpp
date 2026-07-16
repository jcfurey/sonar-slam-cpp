// Runtime GPU dispatch. Every CUDA kernel in this package has a bit-exact (or
// numerically equivalent) CPU twin; gpu::available() decides once per process
// which path runs. Force the CPU path with SONAR_SLAM_FORCE_CPU=1.
//
// All CUDA wrappers return false when the GPU could not produce a result
// (allocation/copy/launch failure, or an unsupported configuration) so the
// caller can fall back to the CPU twin — a GPU error must degrade, never
// corrupt the output. Device buffers are cached and reused across calls to
// avoid per-frame cudaMalloc/cudaFree churn at ping rate.
#pragma once

#include <cstdint>

namespace sonar_slam {
namespace gpu {

// True when the package was built with CUDA, a device is present, and
// SONAR_SLAM_FORCE_CPU is not set. Cached after the first call.
bool available();

#ifdef SONAR_SLAM_WITH_CUDA

// Upper bound of the OS-CFAR training window supported by the kernel (size of
// its per-thread selection buffer). cfar_cuda returns false for larger
// windows so the CPU twin runs instead of silently truncating.
inline constexpr int kMaxOsTrainCells = 128;

// --- raw kernel entry points (defined in src/cuda/*.cu) ----------------------
// CFAR over a polar image (row-major, rows = range bins, cols = beams).
// alg: 0=CA 1=SOCA 2=GOCA 3=OS. A detection also requires cell > threshold
// (the intensity gate, folded in so the kernel matches the CPU twin). Output
// mask is 0/1 uint8.
bool cfar_cuda(const float* img, int rows, int cols, int alg, int train_hs,
               int guard_hs, int rank, double tau, float threshold,
               std::uint8_t* mask_out);

// Nearest / bilinear remap of a uint8 image with float32 maps (cv::remap
// semantics, constant 0 border). interp: 0=nearest, 1=linear.
// map_version tags the (map_x, map_y) pair so the wrapper can keep the maps
// device-resident across calls: pass the same non-negative value while the
// maps are unchanged and bump it whenever they are rebuilt. A negative
// version disables caching (maps are re-uploaded on every call).
bool remap_u8_cuda(const std::uint8_t* src, int src_rows, int src_cols,
                   const float* map_x, const float* map_y, int dst_rows,
                   int dst_cols, int interp, int map_version, std::uint8_t* dst);

// Batched grid-overlap cost for the global scan-match initialization.
// transforms: n_samples x 6 row-major [r00 r01 r10 r11 tx ty]; grid is a 0/255
// occupancy image. costs_out[i] = -(number of transformed points landing on a
// nonzero grid cell), matching slam.py's matching-cost subroutine.
bool grid_cost_cuda(const float* points, int n_points, const float* transforms,
                    int n_samples, const std::uint8_t* grid, int grid_rows,
                    int grid_cols, float xmin, float ymin, float resolution,
                    float* costs_out);

// Exact brute-force 1-NN of query_xy against ref_xy (both row-major x,y
// pairs), libnabo contract: ids_out[q] = -1 and dists2_out[q] = inf when the
// nearest reference point is farther than max_dist (dists are SQUARED, like
// KDTreeMatcher's). Backs cloud_ops match() for the overlap estimates.
bool nn1_cuda(const float* ref_xy, int n_ref, const float* query_xy,
              int n_query, float max_dist, int* ids_out, float* dists2_out);
#endif

}  // namespace gpu
}  // namespace sonar_slam
