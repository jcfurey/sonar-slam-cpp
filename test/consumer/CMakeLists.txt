cmake_minimum_required(VERSION 3.22)
project(installed_sonar_consumer LANGUAGES CXX)
find_package(sonar_slam_cpp REQUIRED)
add_executable(installed_sonar_consumer main.cpp)
target_compile_features(installed_sonar_consumer PRIVATE cxx_std_17)
target_link_libraries(installed_sonar_consumer PRIVATE
  sonar_slam_cpp::sonar_slam_core sonar_slam_cpp::sonar_slam_ros
  sonar_slam_cpp::sonar_slam_component)
