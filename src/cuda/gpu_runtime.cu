// Device probing for the runtime CPU/GPU dispatch.
#include <cuda_runtime.h>

namespace sonar_slam {
namespace gpu {

bool device_present()
{
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

}  // namespace gpu
}  // namespace sonar_slam
