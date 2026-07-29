#include <Eigen/Dense>

#include <cmath>
#include <cstdio>

#include "sonar_slam_cpp/sonar_input.hpp"

#define CHECK(cond, ...)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("FAIL(%d): ", __LINE__);                                 \
      std::printf(__VA_ARGS__);                                            \
      std::printf("\n");                                                   \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main()
{
  using sonar_slam::Matrix;

  const Eigen::Matrix3f optical_level =
    Eigen::AngleAxisf(static_cast<float>(M_PI_2), Eigen::Vector3f::UnitY())
      .toRotationMatrix();
  CHECK(std::fabs(sonar_slam::optical_boresight_pitch(optical_level)) < 1e-6,
        "level optical frame was not level");

  const float head_pitch = 0.4f;
  const Eigen::Matrix3f optical_swept =
    Eigen::AngleAxisf(head_pitch, Eigen::Vector3f::UnitY()).toRotationMatrix()
    * optical_level;
  CHECK(std::fabs(
          sonar_slam::optical_boresight_pitch(optical_swept) + head_pitch)
          < 1e-6,
        "head pitch was not recovered from the stamped transform");

  Matrix sensor(2, 3);
  sensor << 0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 2.0f;
  const Eigen::RowVector3f lever(0.25f, -0.1f, 0.05f);
  const Eigen::Matrix3f level_vehicle =
    Eigen::AngleAxisf(-0.2f, Eigen::Vector3f::UnitY()).toRotationMatrix();
  const Matrix actual = sonar_slam::project_flash_ping(
    sensor, optical_level, lever, level_vehicle);

  Matrix expected = sensor * optical_level.transpose();
  expected.rowwise() += lever;
  expected = expected * level_vehicle.transpose();
  CHECK((actual - expected).cwiseAbs().maxCoeff() < 1e-6f,
        "rigid ping projection changed point geometry");

  // The same transform must apply to every row: inter-point separation can
  // rotate, but cannot acquire motion-dependent shear.
  const Eigen::Vector3f before = sensor.row(1) - sensor.row(0);
  const Eigen::Vector3f after = actual.row(1) - actual.row(0);
  CHECK(std::fabs(before.norm() - after.norm()) < 1e-6f,
        "flash ping was not transformed rigidly");

  std::printf("PASS\n");
  return 0;
}
