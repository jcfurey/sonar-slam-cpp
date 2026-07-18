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
| `mapping_node` | keyframe-anchored, loop-closure-correctable occupancy grid + intensity/backscatter mosaic | `mapping.py` |
| `parity_check` | CPU/GPU parity + perf self-test | — |
| `map_metrics` | map-quality metrics (wall thickness, doubled-wall fraction) over the slam cloud | — |

`mapping_node` consumes the latched `/bruce/slam/slam/traj` (whole optimized
trajectory) plus the ping + feature cloud, and re-renders its 2D map products
when a loop closure moves keyframes (`dec`/`inc` per moved tile) — the map
correction the live accumulator lacks. See `SONAR_MAPPING_ARCHITECTURE.md` §5.
Its `mapping/enu_world` MUST match `slam/enu_world`.

Not ported: the offline bag-pump mode (use `ros2 bag play` instead).

## GPU acceleration & CPU fallback

CUDA kernels (compiled only when a CUDA toolchain exists; the build degrades
to CPU-only silently otherwise):

- **CFAR** (CA/SOCA/GOCA/OS) over the polar image — one thread per pixel.
- **Polar→Cartesian remap** of the detection mask (bilinear, `cv::remap`
  semantics).
- **Batched global scan-match cost** — all Sobol candidate poses of the
  SSM/NSSM initialization evaluated in one launch.
- **Overlap/correspondence 1-NN** (`cloud_ops match()`) — exact brute force,
  same id/-1 + dist²/inf contract as the KDTree; serves the SSM/NSSM overlap
  gates, the NSSM target-key refinement, and the Censi correspondences.
  Size-gated (small queries stay on the CPU KDTree).

Not a kernel but related: `parallel_cov_samples` (default true) runs the
"sampled" covariance method's 30 registrations per scan match across a
per-thread pool of identically-configured libpointmatcher engines instead of
one core sequentially — per-guess results unchanged (deterministic chain);
see `docs/DIVERGENCES.md` #8. The registrations themselves stay on the CPU by
decision (GPU ICP would rewrite deployed registration math — repo-root
`GPU_ACCELERATION.md` §4).

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

Inside the `nautilus-robot-gpu-1` container (ROS 2 Jazzy, CUDA 12.8 toolkit —
nvidia-smi's "13.x" is the driver UMD version, not nvcc — RTX 3090 = `sm_86`):

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
colcon build --packages-select sonar_slam_cpp --merge-install
```

The CUDA architecture defaults to `86-real;86-virtual`; a stale cached
`CMAKE_CUDA_ARCHITECTURES` below sm_75 (the observed case: 52, which nvcc 12.8
still accepts and ships as PTX the sm_86 device must JIT) self-heals to 86 on
reconfigure — see the repo-root `GPU_ACCELERATION.md` for the full audit and
the sonar_proc GPU/OpenMP work.

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

## Bag replay — time discipline (required)

Replay MUST run in bag time on every node, or arrival-stamped data, republish
stamps, and TF lookups split across two clock domains:

```bash
ros2 bag play <bag> --clock
ros2 launch sonar_slam_cpp slam.launch.xml use_sim_time:=true [...]
```

The sensor drivers stamp from independent device clocks. If the sync layers
report no-match starvation (an ERROR naming `stamp_offset`), measure the
constant offset between the streams and set the per-sensor
`<sensor>.stamp_offset` parameter (seconds) — a constant offset δ otherwise
biases every keyframe pose by v·δ / ω·δ, silently. See
`docs/SLAM_EFFECTIVENESS_AUDIT.md`.

## Run

```bash
ros2 launch sonar_slam_cpp slam.launch.xml
# CPU-only forcing (e.g. to A/B against the GPU path):
SONAR_SLAM_FORCE_CPU=1 ros2 launch sonar_slam_cpp slam.launch.xml
# self-test:
ros2 run sonar_slam_cpp parity_check
```

### Hand correction

The operator can correct the SLAM estimate live: use RViz's **2D Pose
Estimate** button (publishes `/initialpose` in the `map` frame) to place the
vehicle where it actually is. The fix is applied as a position-tight,
yaw-soft prior on the newest keyframe; the trajectory re-optimizes, the
`map->odom` TF jumps, and the mapping node re-renders its tiles from the
republished trajectory. Topic and trust sigmas are `manual_correction_topic`
/ `manual_correction_sigmas` in `config/slam.yaml`.

A mis-click is not permanent — corrections form an undo stack:

```bash
ros2 service call /slam/undo_manual_correction std_srvs/srv/Trigger
```

removes the most recent manual prior and relaxes the trajectory back; call
repeatedly to peel earlier corrections.

### Field diagnostics

`slam_node` and `dead_reckoning_node` publish `/diagnostics`
(`diagnostic_msgs/DiagnosticArray`, 1 Hz): sync pairing health (the
stamp-offset starvation signature raises ERROR), the NSSM
accept/reject/revert funnel, and the manual-correction counters — watch with
`rqt_runtime_monitor` instead of grepping logs mid-deployment.

### Map-quality metrics

`ros2 run sonar_slam_cpp map_metrics` subscribes to the aggregated SLAM
cloud (latched — works live or against `ros2 bag play` of a recorded run)
and prints one line per map update: total wall length, local wall thickness
(median / p95), and the doubled-wall fraction at the 0.3–3 m probe scale —
the `docs/MAP_DOUBLING_FIX_PLAN.md` §5 numbers, so before/after replays
compare as numbers instead of screenshots.

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
