# sonar_slam_cpp

All-C++/CUDA port of the `bruce_slam` sonar SLAM stack — zero Python anywhere
(nodes are `rclcpp`, the launch file is XML). Every GPU kernel has a CPU twin
selected at runtime, so the same binary runs on the Jetson/desktop GPU or on a
CPU-only machine with identical behavior.

## Drop-in compatibility

Topic names, parameter names and YAML layouts are **identical** to the Python
`bruce_slam` package — the configs under `config/` are drop-in compatible, and
any tuning done for the Python stack (including
`src/settings/params/localization/sonar_slam/*.yaml`) applies unchanged. The
two stacks are interchangeable per-node: you can run the C++ feature extractor
against the Python SLAM node or vice versa.

| executable | ports | Python original |
| --- | --- | --- |
| `feature_extraction_node` | CFAR + polar→Cartesian + cloud filtering | `feature_extraction.py` |
| `slam_node` | SSM/NSSM/PCM/ISAM2 back-end | `slam.py`, `slam_ros.py` |
| `dead_reckoning_node` | DVL+IMU/FOG dead reckoning (4 modes) | `dead_reckoning.py` |
| `gyro_node` | FOG delta-angle integration | `gyro.py` |
| `kalman_node` | 12-state Kalman filter | `kalman.py` |
| `enu_odom_relay_node` | ENU→z-down odometry relay | `scripts/enu_odom_relay.py` |
| `parity_check` | CPU/GPU parity + perf self-test | — |

Not ported: `mapping_node` (not launched by `slam.launch.py`) and the offline
bag-pump mode (use `ros2 bag play` instead).

## GPU acceleration & CPU fallback

CUDA kernels (compiled only when a CUDA toolchain exists; the build degrades
to CPU-only silently otherwise):

- **CFAR** (CA/SOCA/GOCA/OS) over the polar image — one thread per pixel.
- **Polar→Cartesian remap** of the detection mask (bilinear, `cv::remap`
  semantics).
- **Batched global scan-match cost** — all Sobol candidate poses of the
  SSM/NSSM initialization evaluated in one launch.

Runtime dispatch: `gpu::available()` = built with CUDA ∧ device present ∧
`SONAR_SLAM_FORCE_CPU` unset. CPU twins are OpenMP-parallel and are the same
arithmetic (the parity_check tool verifies mask equality per pixel). Any
runtime CUDA failure (allocation, copy, launch) makes the wrapper return
false and that call falls back to the CPU twin — a GPU error degrades, it
never corrupts the output. Device buffers are cached and reused across calls,
and the remap coordinate maps stay device-resident until the sonar geometry
changes. The CFAR kernel folds the intensity gate in (no separate host-side
compare + bitwise-and pass), and the binary detection mask is remapped with
nearest-neighbour so the GPU and CPU paths are bit-identical. On the CPU, the
CA/SOCA/GOCA detectors use an exact integer prefix-sum fast path for uint8
images — O(1) per pixel and provably bit-identical to the sliding-window
reference (11× measured; proofs in `docs/MATH_NOTES.md`).

Deferred GPU work (needs on-hardware benchmarking to land safely): fusing the
whole per-ping feature pipeline (CFAR → mask → remap) into one device
round-trip; page-locked host buffers + CUDA streams to overlap copy and
compute; and a device-resident grid for the Nelder-Mead scan-match refinement
(only worthwhile alongside batched multi-candidate evaluation — see
`global_init.cpp`).

Python→C++ library replacements (no Python deps remain):

| Python | C++ replacement |
| --- | --- |
| `scipy.optimize.shgo` (sobol) | 3-dim Sobol sequence + Nelder–Mead refine (`global_init.cpp`) |
| `sklearn.covariance.MinCovDet` | compact FAST-MCD w/ consistency correction + reweighting (`mcd.cpp`) |
| `scipy.optimize.root` (CFAR τ) | bracketed bisection (`cfar.cpp`) |
| `scipy.interpolate.interp1d` | linear / natural-cubic spline (`interp.hpp`) |
| `networkx`-style `find_cliques` | Bron–Kerbosch with pivoting (`slam_core.cpp`) |
| `message_filters` (Python ATS) | slop-based `ApproxSync2/3` (`approx_sync.hpp`) |
| gtsam (Python wheel) | libgtsam-dev 4.2 (system) |
| `bruce_slam.pcl` pybind module | direct libpointmatcher/PCL calls (`cloud_ops.cpp`) |

Beyond the straight port, one opt-in extension is available: the ICP factor
covariance can be estimated by the Censi (2007) closed form
(`ssm/cov_method` / `nssm/cov_method: censi`) — one ICP plus an analytic
covariance instead of `cov_samples` registrations + FAST-MCD. The default
(`sampled`) reproduces `bruce_slam` exactly; see `docs/RESEARCH.md`.

## Build

Inside the `nautilus-robot-gpu-1` container (ROS 2 Jazzy, CUDA 12.8):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
colcon build --packages-select sonar_slam_cpp --merge-install
```

Requires standard ROS 2 messages (`marine_acoustic_msgs`, `sensor_msgs`,
`geometry_msgs`, `nav_msgs`, `std_msgs`) plus system
`ros-jazzy-libpointmatcher`, `libgtsam-dev`, PCL and OpenCV. The legacy
BlueROV driver message stubs (`sonar_oculus`, `rti_dvl`, `bar30_depth`,
`kvh_gyro`) are **optional** — `find_package(... QUIET)` compiles their
adapters when present and the build degrades to standard-message drivers
otherwise, so a standard-payload deployment needs none of them.

## Sensor payloads

Nodes pick their sensor drivers by parameter, so different vehicles are
supported with a config overlay, not code changes. Ready presets live in
`config/payloads/`; e.g. the **Deep Trekker Revolution** (Water Linked
DVL-A50, Blueprint Oculus, ENU AHRS, pressure depth) runs entirely on standard
messages with no vendor packages:

```bash
ros2 launch sonar_slam_cpp slam.launch.xml \
  payload_config:=$(ros2 pkg prefix --share sonar_slam_cpp)/config/payloads/deeptrekker_revolution.yaml \
  enable_gyro:=false
```

The full driver matrix and how to add a payload are in `docs/PAYLOADS.md`.

## Run

```bash
ros2 launch sonar_slam_cpp slam.launch.xml
# CPU-only forcing (e.g. to A/B against the GPU path):
SONAR_SLAM_FORCE_CPU=1 ros2 launch sonar_slam_cpp slam.launch.xml
# self-test:
ros2 run sonar_slam_cpp parity_check
```

## Parity verification against the Python stack

`test/parity_driver.py` (dev-only; requires the Python bruce_slam workspace)
feeds identical fixtures through the original pybind/scipy functions and the
C++ ports:

```bash
python3 test/parity_driver.py gen /tmp/slam_parity
ros2 run sonar_slam_cpp parity_vs_python /tmp/slam_parity
python3 test/parity_driver.py compare /tmp/slam_parity
```

Verified 2026-07-10 (GPU and CPU-forced): CFAR masks, cloud ops, KNN match,
full ICP and the global scan-match cost are **bit-exact** vs the originals;
threshold factors match scipy; FAST-MCD agrees with the planted ground truth.

## Intentional deviations from the Python stack

- `shgo` is replaced by Sobol sampling + Nelder–Mead: same sample budget
  (`initialization_params`), same cost function, but the local-refine path
  differs numerically. Both stacks feed the sampled poses into the same
  covariance-ICP machinery, so behavior is equivalent in practice.
- `MinCovDet` is a reimplementation of FAST-MCD, deterministic (fixed seed)
  rather than sklearn's randomized subsets.
- `cv2.applyColorMap(img, 2)` → `cv::COLORMAP_JET` (same map, same id).
- Python's `ApproximateTimeSynchronizer` is replaced by an explicit
  primary-stream synchronizer (DVL paces dead reckoning; the feature cloud
  paces SLAM), matching the effective pairing of the original.
