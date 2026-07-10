#include "sonar_slam_cpp/gpu.hpp"

#include <cstdlib>

namespace sonar_slam {
namespace gpu {

#ifdef SONAR_SLAM_WITH_CUDA
bool device_present();  // defined in cuda/gpu_runtime.cu
#endif

bool available()
{
  static const bool ok = [] {
#ifdef SONAR_SLAM_WITH_CUDA
    const char* force_cpu = std::getenv("SONAR_SLAM_FORCE_CPU");
    if (force_cpu && force_cpu[0] != '\0' && force_cpu[0] != '0') return false;
    return device_present();
#else
    return false;
#endif
  }();
  return ok;
}

}  // namespace gpu
}  // namespace sonar_slam
