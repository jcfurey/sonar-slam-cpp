cmake_minimum_required(VERSION 3.22)
project(sonar_slam_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# ---------------------------------------------------------------- CUDA (optional)
# Kernels compile only when a CUDA toolchain is present; every kernel has a CPU
# twin selected at runtime, so a CPU-only build is fully functional.
option(SONAR_SLAM_ENABLE_CUDA "Build the CUDA kernels" ON)
set(HAVE_CUDA FALSE)
if(SONAR_SLAM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 3090; override with -DCMAKE_CUDA_ARCHITECTURES
    endif()
    set(HAVE_CUDA TRUE)
    message(STATUS "sonar_slam_cpp: CUDA kernels enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "sonar_slam_cpp: no CUDA compiler found, building CPU-only")
  endif()
endif()

# ------------------------------------------------------------------ dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(marine_acoustic_msgs REQUIRED)
find_package(sonar_oculus REQUIRED)
find_package(rti_dvl REQUIRED)
find_package(bar30_depth REQUIRED)
find_package(kvh_gyro REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(GTSAM REQUIRED)
# libpointmatcher's exported target references yaml-cpp::yaml-cpp but does not
# pull it in itself (same workaround as bruce_slam)
find_package(yaml-cpp REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree)
find_package(OpenMP)

include_directories(include)

# ------------------------------------------------------------------- CUDA kernels
if(HAVE_CUDA)
  add_library(sonar_slam_cuda STATIC
    src/cuda/gpu_runtime.cu
    src/cuda/cfar_cuda.cu
    src/cuda/remap_cuda.cu
    src/cuda/cost_cuda.cu
  )
  set_target_properties(sonar_slam_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sonar_slam_cuda PRIVATE include)
  # the .cu files include gpu.hpp, whose declarations are behind this define
  target_compile_definitions(sonar_slam_cuda PRIVATE SONAR_SLAM_WITH_CUDA)
endif()

# -------------------------------------------------------------------- core library
add_library(sonar_slam_core SHARED
  src/core/cfar.cpp
  src/core/gpu_dispatch.cpp
  src/core/sonar_geometry.cpp
  src/core/cloud_ops.cpp
  src/core/global_init.cpp
  src/core/mcd.cpp
  src/core/icp_covariance.cpp
  src/core/slam_core.cpp
)
target_include_directories(sonar_slam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(sonar_slam_core
  ${OpenCV_LIBS}
  gtsam
  ${PCL_LIBRARIES}
)
# the core types carry message stamps/twists (no rclcpp dependency though)
ament_target_dependencies(sonar_slam_core builtin_interfaces geometry_msgs)
# libpointmatcher exports either an imported target or classic variables
if(TARGET libpointmatcher::libpointmatcher)
  target_link_libraries(sonar_slam_core libpointmatcher::libpointmatcher)
else()
  target_include_directories(sonar_slam_core PUBLIC ${libpointmatcher_INCLUDE_DIRS})
  target_link_libraries(sonar_slam_core ${libpointmatcher_LIBRARIES})
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(sonar_slam_core OpenMP::OpenMP_CXX)
endif()
if(HAVE_CUDA)
  target_compile_definitions(sonar_slam_core PUBLIC SONAR_SLAM_WITH_CUDA)
  target_link_libraries(sonar_slam_core sonar_slam_cuda CUDA::cudart_static)
  find_package(CUDAToolkit REQUIRED)
endif()

# --------------------------------------------------------------------- ROS helpers
add_library(sonar_slam_ros SHARED
  src/ros/sensors.cpp
  src/ros/ros_conversions.cpp
)
target_link_libraries(sonar_slam_ros sonar_slam_core)
ament_target_dependencies(sonar_slam_ros
  rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
  builtin_interfaces tf2_ros cv_bridge marine_acoustic_msgs
  sonar_oculus rti_dvl bar30_depth kvh_gyro)

# --------------------------------------------------------------------------- nodes
function(sonar_slam_node name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} sonar_slam_ros)
  ament_target_dependencies(${name}
    rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
    builtin_interfaces tf2_ros cv_bridge marine_acoustic_msgs
    sonar_oculus rti_dvl bar30_depth kvh_gyro ament_index_cpp)
  install(TARGETS ${name} DESTINATION lib/${PROJECT_NAME})
endfunction()

sonar_slam_node(feature_extraction_node src/nodes/feature_extraction_node.cpp)
sonar_slam_node(slam_node               src/nodes/slam_node.cpp)
sonar_slam_node(dead_reckoning_node     src/nodes/dead_reckoning_node.cpp)
sonar_slam_node(gyro_node               src/nodes/gyro_node.cpp)
sonar_slam_node(kalman_node             src/nodes/kalman_node.cpp)
sonar_slam_node(enu_odom_relay_node     src/nodes/enu_odom_relay_node.cpp)

# CPU/GPU parity + performance self-check (not installed as a ROS node)
add_executable(parity_check src/tools/parity_check.cpp)
target_link_libraries(parity_check sonar_slam_core)
install(TARGETS parity_check DESTINATION lib/${PROJECT_NAME})

# function-level parity harness against the Python bruce_slam originals
add_executable(parity_vs_python src/tools/parity_vs_python.cpp)
target_link_libraries(parity_vs_python sonar_slam_core)
ament_target_dependencies(parity_vs_python builtin_interfaces geometry_msgs)
install(TARGETS parity_vs_python DESTINATION lib/${PROJECT_NAME})

# ------------------------------------------------------------------------- install
install(TARGETS sonar_slam_core sonar_slam_ros
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
