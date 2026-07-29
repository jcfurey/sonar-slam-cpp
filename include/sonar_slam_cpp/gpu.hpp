// Runtime dispatch for scan-matching cost and nearest-neighbor kernels.
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
// SONAR_SLAM_FORCE_CPU is not set. Cached after the first call, then gated on
// disable() below.
bool available();

// Abandon the GPU path for the remainder of the process; available() returns
// false from then on. Called by the wrappers when a device allocation fails.
//
// Retrying such a failure is worse than useless: the buffers are grow-only and
// a failed ensure() leaves capacity at 0, so the next ping re-enters the
// driver, fails identically, and logs again -- at ping rate, for as long as
// the pressure lasts. Nothing about the process's own behaviour clears it,
// because the memory belongs to someone else. The CPU twin already produces
// the same result, so taking it immediately costs only latency.
//
// Reports once; later calls are silent. Thread-safe. Deliberately one-way:
// re-probing would reintroduce exactly the retry storm this exists to stop.
void disable(const char* why);

#ifdef SONAR_SLAM_WITH_CUDA

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
