// Function-level parity harness against the original Python bruce_slam stack.
// Reads fixture inputs written by a Python driver script (raw binary matrices:
// [i32 rows][i32 cols][row-major data]), runs the C++ ports on the identical
// inputs, and writes cpp_* outputs next to them for numeric comparison.
//
// Usage: parity_vs_python <fixture_dir>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtsam/geometry/Pose2.h>

#include "sonar_slam_cpp/cfar.hpp"
#include "sonar_slam_cpp/cloud_ops.hpp"
#include "sonar_slam_cpp/global_init.hpp"
#include "sonar_slam_cpp/gpu.hpp"
#include "sonar_slam_cpp/keyframe.hpp"
#include "sonar_slam_cpp/mcd.hpp"

using sonar_slam::Matrix;

namespace {

// no intensity gate in the parity harness — keep every CFAR hit
constexpr float kNoGate = -std::numeric_limits<float>::infinity();

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> read_mat(
  const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::int32_t rows = 0, cols = 0;
  f.read(reinterpret_cast<char*>(&rows), 4);
  f.read(reinterpret_cast<char*>(&cols), 4);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(rows, cols);
  f.read(reinterpret_cast<char*>(m.data()),
         static_cast<std::streamsize>(sizeof(T)) * rows * cols);
  if (!f) throw std::runtime_error("short read on " + path);
  return m;
}

template <typename Derived>
void write_mat(const std::string& path, const Eigen::MatrixBase<Derived>& m)
{
  using T = typename Derived::Scalar;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rm = m;
  std::ofstream f(path, std::ios::binary);
  const std::int32_t rows = static_cast<std::int32_t>(rm.rows());
  const std::int32_t cols = static_cast<std::int32_t>(rm.cols());
  f.write(reinterpret_cast<const char*>(&rows), 4);
  f.write(reinterpret_cast<const char*>(&cols), 4);
  f.write(reinterpret_cast<const char*>(rm.data()),
          static_cast<std::streamsize>(sizeof(T)) * rows * cols);
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::fprintf(stderr, "usage: parity_vs_python <fixture_dir>\n");
    return 2;
  }
  const std::string dir = std::string(argv[1]) + "/";
  const bool gpu = sonar_slam::gpu::available();
  std::printf("GPU available: %s\n", gpu ? "yes" : "no");

  // ------------------------------------------------------------------ CFAR
  {
    const auto img = read_mat<float>(dir + "img.f32");
    const auto taus = read_mat<float>(dir + "taus.f32");  // 4x1: CA SOCA GOCA OS
    const int rows = img.rows(), cols = img.cols();
    const int train_hs = 20, guard_hs = 5, rank = 10;
    const char* names[4] = {"ca", "soca", "goca", "os"};
    for (int alg = 0; alg < 4; ++alg) {
      Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        mask(rows, cols);
      sonar_slam::CFAR::detect_cpu(img.data(), rows, cols, alg, train_hs,
                                   guard_hs, rank, taus(alg, 0), kNoGate,
                                   mask.data());
      write_mat(dir + "cpp_mask_" + names[alg] + ".u8", mask);
#ifdef SONAR_SLAM_WITH_CUDA
      if (gpu) {
        Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          gmask(rows, cols);
        if (sonar_slam::gpu::cfar_cuda(img.data(), rows, cols, alg, train_hs,
                                       guard_hs, rank, taus(alg, 0), kNoGate,
                                       gmask.data()))
          write_mat(dir + "cpp_mask_gpu_" + names[alg] + ".u8", gmask);
        else
          std::printf("CFAR alg %d: GPU path failed, no gpu mask written\n", alg);
      }
#endif
    }
  }

  // ----------------------------------------------------------- cloud ops
  {
    const Matrix cloud = read_mat<float>(dir + "cloud.f32");
    const Matrix keys = read_mat<float>(dir + "keys.f32");

    write_mat(dir + "cpp_downsample.f32", sonar_slam::downsample(cloud, 0.5f));

    auto [pts, desc] = sonar_slam::downsample(cloud, keys, 0.5f);
    write_mat(dir + "cpp_downsample_pts.f32", pts);
    write_mat(dir + "cpp_downsample_keys.f32", desc);

    write_mat(dir + "cpp_outlier.f32", sonar_slam::remove_outlier(cloud, 1.0, 5));

    const Matrix match_ref = read_mat<float>(dir + "match_ref.f32");
    const Matrix match_in = read_mat<float>(dir + "match_in.f32");
    auto [ids, dists] = sonar_slam::match(match_ref, match_in, 1, 0.5f);
    write_mat(dir + "cpp_match_ids.i32", ids);
    write_mat(dir + "cpp_match_dists.f32", dists);
  }

  // ------------------------------------------------------------------ ICP
  {
    const Matrix source = read_mat<float>(dir + "icp_source.f32");
    const Matrix target = read_mat<float>(dir + "icp_target.f32");
    const Eigen::Matrix3f guess = read_mat<float>(dir + "icp_guess.f32");
    sonar_slam::ICP icp;
    icp.load_from_yaml(dir + "icp.yaml");
    auto [message, T] = icp.compute(source, target, guess);
    std::printf("ICP: %s\n", message.c_str());
    write_mat(dir + "cpp_icp_T.f32", T);
  }

  // ------------------------------------------------- transform_points
  {
    const Matrix pts = read_mat<float>(dir + "cloud.f32");
    const gtsam::Pose2 pose(1.5, -2.0, 0.7);
    write_mat(dir + "cpp_transformed.f32",
              sonar_slam::Keyframe::transform_points(pts, pose));
  }

  // ------------------------------------------------- global-init cost
  {
    const Matrix src = read_mat<float>(dir + "gi_source.f32");
    const Matrix tgt = read_mat<float>(dir + "gi_target.f32");
    const auto poses = read_mat<float>(dir + "gi_poses.f32");   // 2x3
    const auto deltas_m = read_mat<float>(dir + "gi_deltas.f32");  // Kx3
    const gtsam::Pose2 source_pose(poses(0, 0), poses(0, 1), poses(0, 2));
    const gtsam::Pose2 target_pose(poses(1, 0), poses(1, 1), poses(1, 2));
    std::vector<Eigen::Vector3d> deltas;
    for (int i = 0; i < deltas_m.rows(); ++i)
      deltas.emplace_back(deltas_m(i, 0), deltas_m(i, 1), deltas_m(i, 2));

    // dispatched path (GPU when present)
    const auto costs = sonar_slam::global_match_costs(src, source_pose, tgt,
                                                      target_pose, 0.5, deltas);
    Eigen::MatrixXf out(costs.size(), 1);
    for (std::size_t i = 0; i < costs.size(); ++i) out(i, 0) = costs[i];
    write_mat(dir + "cpp_costs.f32", out);
  }

  // ------------------------------------------------------------------ MCD
  {
    const auto samples_f = read_mat<float>(dir + "mcd_samples.f32");
    const Eigen::MatrixXd samples = samples_f.cast<double>();
    const auto mcd = sonar_slam::min_cov_det(samples, 0.8);
    if (mcd.success) {
      write_mat(dir + "cpp_mcd_loc.f32",
                Eigen::MatrixXf(mcd.location.transpose().cast<float>()));
      write_mat(dir + "cpp_mcd_cov.f32", Eigen::MatrixXf(mcd.covariance.cast<float>()));
    } else {
      std::printf("MCD failed\n");
    }
  }

  std::printf("cpp outputs written to %s\n", dir.c_str());
  return 0;
}
