cmake_minimum_required(VERSION 3.22)
project(sonar_slam_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Only the scan-matching kernels remain here. Image scanning, CFAR and polar
# projection belong to sonar_proc.
option(SONAR_SLAM_ENABLE_CUDA "Build the scan-matching CUDA kernels" ON)
set(HAVE_CUDA FALSE)
if(SONAR_SLAM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES 86-real 86-virtual CACHE STRING
          "CUDA architectures (RTX 3090 default)")
    endif()
    enable_language(CUDA)
    set(HAVE_CUDA TRUE)
    message(STATUS
      "sonar_slam_cpp: CUDA scan matching enabled "
      "(arch ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS
      "sonar_slam_cpp: no CUDA compiler found, building CPU-only")
  endif()
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

function(auto_target_dependencies target)
  if(COMMAND ament_auto_depend_on_packages)
    ament_auto_depend_on_packages(${target} PACKAGES ${ARGN})
  else()
    ament_target_dependencies(${target} ${ARGN})
  endif()
endfunction()

find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(GTSAM REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree)
find_package(OpenMP)

include_directories(include)

if(HAVE_CUDA)
  add_library(sonar_slam_cuda STATIC
    src/cuda/gpu_runtime.cu
    src/cuda/cost_cuda.cu
    src/cuda/nn_cuda.cu)
  set_target_properties(sonar_slam_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sonar_slam_cuda PRIVATE include)
  target_compile_definitions(
    sonar_slam_cuda PRIVATE SONAR_SLAM_WITH_CUDA)
endif()

add_library(sonar_slam_core SHARED
  src/core/gpu_dispatch.cpp
  src/core/cloud_ops.cpp
  src/core/global_init.cpp
  src/core/mcd.cpp
  src/core/icp_covariance.cpp
  src/core/slam_core.cpp)
target_include_directories(sonar_slam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS})
target_link_libraries(sonar_slam_core
  ${OpenCV_LIBS}
  gtsam
  ${PCL_LIBRARIES})
auto_target_dependencies(
  sonar_slam_core builtin_interfaces geometry_msgs)
if(TARGET libpointmatcher::libpointmatcher)
  target_link_libraries(
    sonar_slam_core libpointmatcher::libpointmatcher)
else()
  target_include_directories(
    sonar_slam_core PUBLIC ${libpointmatcher_INCLUDE_DIRS})
  target_link_libraries(
    sonar_slam_core ${libpointmatcher_LIBRARIES})
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(sonar_slam_core OpenMP::OpenMP_CXX)
endif()
if(HAVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(
    sonar_slam_core PUBLIC SONAR_SLAM_WITH_CUDA)
  target_link_libraries(
    sonar_slam_core sonar_slam_cuda CUDA::cudart_static)
endif()

add_library(sonar_slam_ros SHARED src/ros/ros_conversions.cpp)
target_link_libraries(sonar_slam_ros sonar_slam_core)
auto_target_dependencies(
  sonar_slam_ros
  std_msgs sensor_msgs geometry_msgs visualization_msgs builtin_interfaces)

add_executable(slam_node src/nodes/slam_node.cpp)
target_link_libraries(slam_node sonar_slam_ros)
auto_target_dependencies(
  slam_node
  rclcpp std_msgs std_srvs diagnostic_msgs sensor_msgs geometry_msgs nav_msgs
  visualization_msgs builtin_interfaces tf2 tf2_ros ament_index_cpp)
install(TARGETS slam_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_test REQUIRED)

  add_executable(test_interp_spline test/interp_spline_test.cpp)
  target_include_directories(
    test_interp_spline PRIVATE ${EIGEN3_INCLUDE_DIRS})
  ament_add_test(test_interp_spline
    COMMAND $<TARGET_FILE:test_interp_spline>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_censi_covariance test/censi_covariance_test.cpp)
  target_include_directories(
    test_censi_covariance PRIVATE ${EIGEN3_INCLUDE_DIRS})
  ament_add_test(test_censi_covariance
    COMMAND $<TARGET_FILE:test_censi_covariance>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_sonar_input test/sonar_input_test.cpp)
  target_include_directories(
    test_sonar_input PRIVATE ${EIGEN3_INCLUDE_DIRS})
  ament_add_test(test_sonar_input
    COMMAND $<TARGET_FILE:test_sonar_input>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_cloud_ops test/cloud_ops_test.cpp)
  target_link_libraries(test_cloud_ops sonar_slam_core)
  ament_add_test(test_cloud_ops
    COMMAND $<TARGET_FILE:test_cloud_ops>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_icp_wrapper test/icp_wrapper_test.cpp)
  target_link_libraries(test_icp_wrapper sonar_slam_core)
  ament_add_test(test_icp_wrapper
    COMMAND $<TARGET_FILE:test_icp_wrapper>
    ENV ICP_TEST_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/config/icp.yaml
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_isam_recovery test/test_isam_recovery.cpp)
  target_include_directories(
    test_isam_recovery PRIVATE ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_isam_recovery gtsam)
  ament_add_test(test_isam_recovery
    COMMAND $<TARGET_FILE:test_isam_recovery>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(
    test_manual_correction test/test_manual_correction.cpp)
  target_link_libraries(test_manual_correction sonar_slam_core)
  auto_target_dependencies(
    test_manual_correction builtin_interfaces geometry_msgs)
  ament_add_test(test_manual_correction
    COMMAND $<TARGET_FILE:test_manual_correction>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_pipeline_e2e test/test_pipeline_e2e.cpp)
  target_link_libraries(test_pipeline_e2e sonar_slam_core)
  auto_target_dependencies(
    test_pipeline_e2e builtin_interfaces geometry_msgs)
  target_compile_definitions(
    test_pipeline_e2e
    PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  ament_add_test(test_pipeline_e2e
    COMMAND $<TARGET_FILE:test_pipeline_e2e>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    TIMEOUT 300)
endif()

install(TARGETS sonar_slam_core sonar_slam_ros
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_auto_package()
