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
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      # RTX 3090 = sm_86. `-real` emits the cubin for this GPU; `-virtual` also
      # embeds PTX so a newer GPU can JIT. Override with -DCMAKE_CUDA_ARCHITECTURES.
      set(CMAKE_CUDA_ARCHITECTURES 86-real 86-virtual)
    endif()
    # A stale CACHED value (the observed case: 52, from an old configure) is
    # worse than it looks: under nvcc 12.x it still BUILDS (pre-75 archs are
    # merely deprecated), so the kernels ship as compute_52 PTX that the
    # sm_86 device must JIT — silently slow, no build error to notice — and
    # under nvcc 13+ (which dropped sm_5x) the target fails outright while
    # colcon continue-on-error keeps the stale install. Treat anything below
    # sm_75 (Turing) as a stale cache on any modern toolkit and self-heal to
    # sm_86 so the kernels actually rebuild for the device.
    # Only the LEGACY CMake auto-default (sm_52 and below, Maxwell) is treated
    # as a stale cache and force-reset: that is the value old configure runs
    # actually left behind, and no modern deliberate target uses it. Anything
    # in (52, 75) — P100/GTX10xx/V100/Xavier are all real nvcc-12 targets —
    # is respected with a warning, so a user's -DCMAKE_CUDA_ARCHITECTURES=70
    # is never silently replaced with cubins their device cannot load.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
      set(_cuda_arch_stale_max 52)
      set(_cuda_arch_warn_max 74)
    else()
      set(_cuda_arch_stale_max 0)
      set(_cuda_arch_warn_max 0)
    endif()
    set(_cuda_arch_stale FALSE)
    set(_cuda_arch_old FALSE)
    foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
      string(REGEX MATCH "^[0-9]+" _an "${_a}")
      if(_an AND _an LESS_EQUAL _cuda_arch_stale_max)
        set(_cuda_arch_stale TRUE)
      elseif(_an AND _an LESS_EQUAL _cuda_arch_warn_max)
        set(_cuda_arch_old TRUE)
      endif()
    endforeach()
    if(_cuda_arch_stale)
      message(WARNING "sonar_slam_cpp: cached CMAKE_CUDA_ARCHITECTURES "
        "'${CMAKE_CUDA_ARCHITECTURES}' is the legacy CMake default (<= sm_52) "
        "— treating as a stale cache and resetting to 86. Reconfigure with "
        "-DCMAKE_CUDA_ARCHITECTURES=<arch> to pick your own GPU.")
      set(CMAKE_CUDA_ARCHITECTURES 86-real 86-virtual CACHE STRING "" FORCE)
    elseif(_cuda_arch_old)
      message(WARNING "sonar_slam_cpp: CMAKE_CUDA_ARCHITECTURES "
        "'${CMAKE_CUDA_ARCHITECTURES}' targets a pre-Turing GPU; keeping it "
        "as configured (kernels will build for that device).")
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
find_package(std_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(marine_acoustic_msgs REQUIRED)

# Legacy BlueROV-payload vendor message packages. OPTIONAL: a build that only
# targets standard-message payloads (e.g. the Deep Trekker Revolution, which
# runs on marine_acoustic_msgs + sensor_msgs) does not need any of them. Each
# package found here defines HAVE_<PKG> so src/ros/sensors.cpp compiles the
# matching driver adapter; an un-found driver reports a runtime error when
# selected. dvl_msgs is the optional native Water Linked A50 message package.
find_package(sonar_oculus QUIET)
find_package(rti_dvl QUIET)
find_package(bar30_depth QUIET)
find_package(kvh_gyro QUIET)
find_package(dvl_msgs QUIET)

set(SONAR_SLAM_VENDOR_MSG_DEPS "")
set(SONAR_SLAM_VENDOR_MSG_DEFS "")
foreach(pkg_def
    sonar_oculus:HAVE_SONAR_OCULUS  rti_dvl:HAVE_RTI_DVL
    bar30_depth:HAVE_BAR30_DEPTH    kvh_gyro:HAVE_KVH_GYRO
    dvl_msgs:HAVE_DVL_MSGS)
  string(REPLACE ":" ";" pd ${pkg_def})
  list(GET pd 0 pkg)
  list(GET pd 1 def)
  if(${pkg}_FOUND)
    list(APPEND SONAR_SLAM_VENDOR_MSG_DEPS ${pkg})
    list(APPEND SONAR_SLAM_VENDOR_MSG_DEFS ${def})
  endif()
endforeach()
if(SONAR_SLAM_VENDOR_MSG_DEPS)
  message(STATUS "sonar_slam_cpp: vendor message packages found: ${SONAR_SLAM_VENDOR_MSG_DEPS}")
else()
  message(STATUS "sonar_slam_cpp: no legacy vendor message packages found — "
                 "standard-message drivers only (Revolution-style build)")
endif()

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
    src/cuda/nn_cuda.cu
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
  src/core/mapping.cpp
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
  ${SONAR_SLAM_VENDOR_MSG_DEPS})
# sensors.cpp guards each vendor adapter behind its HAVE_<PKG> define; only the
# packages actually found above are compiled in.
if(SONAR_SLAM_VENDOR_MSG_DEFS)
  target_compile_definitions(sonar_slam_ros PRIVATE ${SONAR_SLAM_VENDOR_MSG_DEFS})
endif()

# --------------------------------------------------------------------------- nodes
function(sonar_slam_node name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} sonar_slam_ros)
  # nodes select drivers by parameter string and never include a vendor header;
  # the vendor message runtime comes transitively through sonar_slam_ros
  ament_target_dependencies(${name}
    rclcpp std_msgs std_srvs diagnostic_msgs sensor_msgs geometry_msgs
    nav_msgs visualization_msgs builtin_interfaces tf2_ros cv_bridge
    marine_acoustic_msgs ${SONAR_SLAM_VENDOR_MSG_DEPS} ament_index_cpp)
  install(TARGETS ${name} DESTINATION lib/${PROJECT_NAME})
endfunction()

sonar_slam_node(feature_extraction_node src/nodes/feature_extraction_node.cpp)
sonar_slam_node(slam_node               src/nodes/slam_node.cpp)
sonar_slam_node(mapping_node            src/nodes/mapping_node.cpp)
sonar_slam_node(dead_reckoning_node     src/nodes/dead_reckoning_node.cpp)
sonar_slam_node(gyro_node               src/nodes/gyro_node.cpp)
sonar_slam_node(kalman_node             src/nodes/kalman_node.cpp)
sonar_slam_node(enu_odom_relay_node     src/nodes/enu_odom_relay_node.cpp)

# CPU/GPU parity + performance self-check (not installed as a ROS node)
add_executable(parity_check src/tools/parity_check.cpp)
target_link_libraries(parity_check sonar_slam_core)
install(TARGETS parity_check DESTINATION lib/${PROJECT_NAME})

# map-quality metrics over the aggregated SLAM cloud (wall thickness +
# doubled-wall fraction; the MAP_DOUBLING_FIX_PLAN §5 replay numbers)
sonar_slam_node(map_metrics src/tools/map_metrics.cpp)

# cross-device stamp-offset probe (measures <sensor>.stamp_offset instead of
# asking the operator to)
sonar_slam_node(stamp_probe src/tools/stamp_probe.cpp)

# synthetic-pool payload simulator for demo.launch.xml (no hardware/bags)
sonar_slam_node(sim_payload src/tools/sim_payload.cpp)

# function-level parity harness against the Python bruce_slam originals
add_executable(parity_vs_python src/tools/parity_vs_python.cpp)
target_link_libraries(parity_vs_python sonar_slam_core)
ament_target_dependencies(parity_vs_python builtin_interfaces geometry_msgs)
install(TARGETS parity_vs_python DESTINATION lib/${PROJECT_NAME})

# --------------------------------------------------------------------------- tests
# Plain-executable tests (exit code is the verdict) registered through
# ament_add_test — no gtest dependency, runs under `colcon test`. They lock in
# the empirically-validated GTSAM behaviors the back-end's recovery and
# hand-correction machinery depend on; if a GTSAM upgrade changes them, these
# fail before the field does.
if(BUILD_TESTING)
  find_package(ament_cmake_test REQUIRED)

  add_executable(test_cfar_math test/cfar_math_test.cpp)
  target_link_libraries(test_cfar_math sonar_slam_core)
  ament_add_test(test_cfar_math
    COMMAND $<TARGET_FILE:test_cfar_math>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    TIMEOUT 120)

  add_executable(test_interp_spline test/interp_spline_test.cpp)
  target_include_directories(test_interp_spline PRIVATE ${EIGEN3_INCLUDE_DIRS})
  ament_add_test(test_interp_spline
    COMMAND $<TARGET_FILE:test_interp_spline>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_censi_covariance test/censi_covariance_test.cpp)
  target_include_directories(test_censi_covariance PRIVATE ${EIGEN3_INCLUDE_DIRS})
  ament_add_test(test_censi_covariance
    COMMAND $<TARGET_FILE:test_censi_covariance>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_sonar_geometry test/sonar_geometry_test.cpp)
  target_link_libraries(test_sonar_geometry sonar_slam_core)
  ament_add_test(test_sonar_geometry
    COMMAND $<TARGET_FILE:test_sonar_geometry>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_isam_recovery test/test_isam_recovery.cpp)
  target_include_directories(test_isam_recovery PRIVATE ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_isam_recovery gtsam)
  ament_add_test(test_isam_recovery
    COMMAND $<TARGET_FILE:test_isam_recovery>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  add_executable(test_manual_correction test/test_manual_correction.cpp)
  target_link_libraries(test_manual_correction sonar_slam_core)
  ament_target_dependencies(test_manual_correction
    builtin_interfaces geometry_msgs)
  ament_add_test(test_manual_correction
    COMMAND $<TARGET_FILE:test_manual_correction>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO)

  # end-to-end pipeline on a synthetic pool: CFAR -> registration -> loop
  # closure -> persistence -> relocalization -> absolute fixes, no bags/DDS
  add_executable(test_pipeline_e2e test/test_pipeline_e2e.cpp)
  target_link_libraries(test_pipeline_e2e sonar_slam_core)
  ament_target_dependencies(test_pipeline_e2e builtin_interfaces geometry_msgs)
  target_compile_definitions(test_pipeline_e2e
    PRIVATE TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  ament_add_test(test_pipeline_e2e
    COMMAND $<TARGET_FILE:test_pipeline_e2e>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    TIMEOUT 300)

  # In CPU-only CI this is a CPU smoke test; on a CUDA build with a visible
  # device the same registered test exercises every GPU kernel against its CPU
  # twin.
  ament_add_test(test_runtime_parity
    COMMAND $<TARGET_FILE:parity_check>
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
    TIMEOUT 120)

  # test/parity_driver.py additionally needs the original bruce_slam package
  # and generated reference fixtures, so it remains a manual cross-project
  # compatibility harness rather than a hermetic package test.
endif()

# ------------------------------------------------------------------------- install
install(TARGETS sonar_slam_core sonar_slam_ros
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
