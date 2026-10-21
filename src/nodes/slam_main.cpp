#include "sonar_slam_cpp/node.hpp"
#include "sonar_slam_cpp/node_base.hpp"

int main(int argc, char** argv)
{
  return sonar_slam::run_node_factory(argc, argv, [] { return sonar_slam::make_slam_node(); });
}
