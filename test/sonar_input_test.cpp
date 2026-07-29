#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <limits>

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

  Matrix malformed(3, 3);
  malformed << 1.0f, 0.0f, 0.0f,
               std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f,
               2.0f, 0.0f, 0.0f;
  CHECK(sonar_slam::finite_points(malformed).rows() == 2,
        "non-finite returns were not removed");

  sonar_slam::ScanAdmissionParams gate;
  gate.min_points = 6;
  gate.cell_size = 0.5;
  gate.min_occupied_cells = 4;
  gate.azimuth_bin_size = 0.1;
  gate.min_azimuth_bins = 3;

  Matrix too_few(5, 3);
  too_few << 1.0f, 0.0f, 0.0f,
             2.0f, 0.2f, 0.0f,
             3.0f, -0.2f, 0.0f,
             4.0f, 0.4f, 0.0f,
             5.0f, -0.4f, 0.0f;
  auto admission = sonar_slam::assess_scan(too_few, gate);
  CHECK(!admission.informative &&
          admission.reason ==
            sonar_slam::ScanAdmissionReason::NOT_ENOUGH_POINTS,
        "point-count gate admitted a sparse scan");

  Matrix cluster(6, 3);
  cluster << 1.01f, 0.01f, 0.0f,
             1.02f, 0.02f, 0.0f,
             1.03f, 0.03f, 0.0f,
             1.04f, 0.04f, 0.0f,
             1.05f, 0.05f, 0.0f,
             1.06f, 0.06f, 0.0f;
  admission = sonar_slam::assess_scan(cluster, gate);
  CHECK(!admission.informative &&
          admission.reason ==
            sonar_slam::ScanAdmissionReason::NOT_ENOUGH_CELLS,
        "occupied-cell gate admitted a dense blob");

  Matrix one_ray(6, 3);
  one_ray << 1.0f, 0.0f, 0.0f,
             2.0f, 0.0f, 0.0f,
             3.0f, 0.0f, 0.0f,
             4.0f, 0.0f, 0.0f,
             5.0f, 0.0f, 0.0f,
             6.0f, 0.0f, 0.0f;
  admission = sonar_slam::assess_scan(one_ray, gate);
  CHECK(!admission.informative &&
          admission.reason ==
            sonar_slam::ScanAdmissionReason::NOT_ENOUGH_AZIMUTH,
        "azimuth gate admitted a single water-column ray");

  Matrix structured(6, 3);
  structured << 1.0f, 0.0f, 0.0f,
                2.0f, 0.0f, 0.0f,
                1.0f, 0.2f, 0.0f,
                2.0f, 0.4f, 0.0f,
                1.0f, -0.2f, 0.0f,
                2.0f, -0.4f, 0.0f;
  admission = sonar_slam::assess_scan(structured, gate);
  CHECK(admission.informative, "structured scan was rejected: %s",
        sonar_slam::scan_admission_reason(admission.reason));

  std::printf("PASS\n");
  return 0;
}
