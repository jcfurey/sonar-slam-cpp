#include "sonar_slam_cpp/slam_core.hpp"
#include "sonar_slam_cpp/node.hpp"
#include "sonar_slam_cpp/ros_conversions.hpp"

int main()
{
  sonar_slam::Slam slam;
  slam.ssm_params.enable = false;
  slam.configure();
  const auto cloud = sonar_slam::make_cloud_xyz(sonar_slam::Matrix::Zero(2, 3));
  return cloud.width == 2 && slam.current_key() == 0 ? 0 : 1;
}
