# sonar_slam_cpp

[![ci](https://github.com/jcfurey/sonar-slam-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/jcfurey/sonar-slam-cpp/actions/workflows/ci.yml)

All-C++/CUDA runtime port of the `bruce_slam` sonar SLAM stack — nodes are
`rclcpp` and the launch files are XML. Python is used only by the optional
cross-project parity harness under `test/`, never at runtime. Every GPU kernel
has a CPU twin selected at runtime, so the same binary runs on the
Jetson/desktop GPU or on a CPU-only machine with identical behavior.

## Quick demo (no hardware, no bags)

```bash
ros2 launch sonar_slam_cpp demo.launch.xml
```

A synthetic pool simulator (`sim_payload`) drives the full standard pipeline
— CFAR feature extraction, dead reckoning, scan matching, loop closures,
mapping — with RViz up. The simulated DVL carries a small bias, so dead
reckoning drifts visibly and loop closures pull the trajectory back. Use the
RViz **2D Pose Estimate** button for a live hand correction;
`ros2 service call /bruce/slam/slam/undo_manual_correction std_srvs/srv/Trigger` takes
it back. The same synthetic world backs the end-to-end pipeline test
(`test/test_pipeline_e2e.cpp`), which CI runs on pushes to `main` and on
pull requests — **not** on pushes to work branches, and not at all when this
package is built as part of `nautilus_ws` (see *Tests* below).

## Drop-in compatibility

Topic names and YAML layouts are **identical** to the Python `bruce_slam`
package, and every parameter the Python stack has keeps its name and meaning
— so the configs under `config/` are drop-in compatible and existing tuning
(including `src/settings/params/localization/sonar_slam/*.yaml`) applies
unchanged. The two stacks are interchangeable per-node: you can run the C++
feature extractor against the Python SLAM node or vice versa.

The C++ stack is a strict **superset**: it adds parameters the Python
original has no equivalent for — the NSSM degeneracy and compass gates
(`nssm/max_sigma`, `max_anisotropy`, `max_yaw_vs_compass`,
`max_translation_vs_dr`, `min_revisit_sep`, `min_overlap_ratio`), the
post-loop verification (`post_loop_max_yaw_rms`,
`post_loop_max_link_error_sigma`), map persistence, USBL input, operator hand
correction, and the `map->odom` republish timer. Those are additive; a Python
config simply leaves them at their defaults.

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
| `stamp_probe` | cross-device stamp-offset measurement (prints the `stamp_offset` lines to set) | — |
| `sim_payload` | synthetic pool payload simulator for `demo.launch.xml` | — |

`mapping_node` consumes the latched `/bruce/slam/slam/traj` (whole optimized
trajectory) plus the ping + feature cloud, and re-renders its 2D map products
when a loop closure moves keyframes (`dec`/`inc` per moved tile) — the map
correction the live accumulator lacks.
Its `mapping/enu_world` MUST match `slam/enu_world` (both ship `true`; the
`bluerov_legacy` payload flips both to the legacy z-down convention). Under
`bringup_localization_erdc` this is no longer left to two separate YAMLs
agreeing by hand — both nodes take the value from one `slam_enu_world` launch
argument, declared after the YAML loads, the same interlock pattern used for
`publish_tf`. A silent mismatch mirrors every map product about the x axis
while the SLAM pose stays correct, so it reads as a calibration fault rather
than a config one.

Not ported: the offline bag-pump mode (use `ros2 bag play` instead).

## GPU acceleration & CPU fallback

CUDA kernels (compiled only when a CUDA toolchain exists; the build degrades
to CPU-only silently otherwise):

- **CFAR** (CA/SOCA/GOCA/OS) over the polar image — one thread per pixel.
- **Polar→Cartesian remap** (`cv::remap` semantics): nearest-neighbour for the
  binary detection mask and bilinear for grayscale visualization/intensity.
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
decision — GPU ICP would rewrite deployed registration math.

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
reference (11× in the documented four-core reference measurement; benchmark
results vary by host and OpenMP configuration; proofs in `docs/MATH_NOTES.md`).

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
| `scipy.interpolate.interp1d` | linear / not-a-knot cubic spline (`interp.hpp`) |
| `networkx`-style `find_cliques` | Bron–Kerbosch with pivoting (`slam_core.cpp`) |
| `message_filters` (Python ATS) | slop-based `ApproxSync2/3` (`approx_sync.hpp`) |
| gtsam (Python wheel) | libgtsam-dev 4.2 (system) |
| `bruce_slam.pcl` pybind module | direct libpointmatcher/PCL calls (`cloud_ops.cpp`) |

ICP factor covariance is estimated with sampled registrations + FAST-MCD, from
the **Sobol** population only. Nelder–Mead refinement probes used to be
recorded alongside them; because NM converges toward the optimum its probes
carried the lowest costs and sorted to the front of the cost-ordered seed
list, so the covariance measured sensitivity to a *tiny* perturbation rather
than to real initialization uncertainty. The resulting over-confidence drove
ISAM2 indeterminate and silently dropped keyframes (7/67 in the end-to-end
test). `result.delta` still comes from the refined NM optimum, so
registration quality is unchanged — only the covariance population moved.

`cov_method: censi` is gated on the **loaded** ICP chain rather than assumed.
The Censi helper builds a point-to-**point** J'J Hessian, so it is valid only
against a point-to-point minimizer: `config/icp.yaml` (this package's default)
is `PointToPlaneErrorMinimizer` and rejects it, while
`settings_erdc/params/localization/sonar_slam/icp.yaml` is
`PointToPointErrorMinimizer` and accepts it. `sampled` remains the default
either way — censi is selectable, not validated.

Note that Censi is **not** a degeneracy detector here. With correspondences
held fixed its translation block is `N·I` regardless of geometry, so it rates
a flat wall as *better* conditioned than an L-corner (measured 2.92 vs 3.15);
only a point-to-plane information matrix goes singular on the wall. Sampling
detects sliding precisely because restarting ICP lets the data association
change — which is why the sampled path, not a closed form, feeds the
`nssm/max_sigma` + `max_anisotropy` gates.

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
reconfigure.

Requires standard ROS 2 messages (`marine_acoustic_msgs`, `sensor_msgs`,
`geometry_msgs`, `nav_msgs`, `std_msgs`) plus system
`ros-jazzy-libpointmatcher`, `libgtsam-dev`, PCL and OpenCV. The legacy
BlueROV driver message stubs (`sonar_oculus`, `rti_dvl`, `bar30_depth`,
`kvh_gyro`) are **optional** — `find_package(... QUIET)` compiles their
adapters when present and the build degrades to standard-message drivers
otherwise, so a standard-payload deployment needs none of them.

### Tests — skipped by the nautilus_ws build

`.github/workflows/ci.yml` builds and runs the suite on pushes to `main` and
on every pull request, so the tests are healthy upstream.

They are **silently skipped when built as part of `nautilus_ws`**: that
workspace's `colcon_defaults.yaml` sets `-DBUILD_TESTING=OFF`, and
`CMakeLists.txt` gates every test on `BUILD_TESTING`, so `ctest -N` in
`build/sonar_slam_cpp` reports **`Total Tests: 0`**. Since this package is
consumed there as a submodule pinned to a work branch (`cam_wip`), and CI
triggers only on `main`/PRs, day-to-day integration work runs against no
tests at all unless they are asked for explicitly:

```bash
colcon build --packages-select sonar_slam_cpp \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
colcon test --merge-install --packages-select sonar_slam_cpp
colcon test-result --all --verbose
```

`colcon_defaults.test.yaml` in the workspace root carries the same settings
if you would rather swap profiles (the pattern `Release.Dockerfile` uses for
`colcon_defaults.release.yaml`).

Eight tests, ~10 s total. `test_pipeline_e2e` is the one that matters most:
it drives a synthetic rectangular pool through CFAR, SSM, NSSM (every
defence gate), ISAM2, map save/load + relocalization, and a USBL-style
position prior, asserting that the loop closes and the trajectory error
beats raw dead reckoning. It runs with the **deployed** covariance settings
(`cov_samples: 30`, `sampled`, `min_overlap_ratio: 0.1`) so the
ICP-covariance → degeneracy-gate chain is actually exercised — it used to
run at `cov_samples: 0`, leaving that whole chain untested.

`nautilus_ws/.gitlab-ci.yml` is docker-buildx-bake only and has no colcon
stage, so nothing runs the suite on the integration side.

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

Inside `nautilus_ws` this is handled for you, but conditionally: a shell hook
(`bashrc.d/pre/98-PS1_state_icon.bashrc`) exports `use_sim_time=True` when it
detects a running `rosbag2_transport/player`, and the bag launch reads that
env var. It is evaluated **per prompt**, so a shell opened before the player
started — or a non-interactive script — sees `False`. Export
`USE_SIM_TIME=True` to lock it on rather than relying on the detection.

The `map->odom` republish timer does not depend on getting this right: it
stamps in the DATA clock domain (last frame stamp advanced by elapsed node
time) rather than at `now()`, so it stays consistent with the rest of TF in
either domain.

The sensor drivers stamp from independent device clocks. If the sync layers
report no-match starvation (an ERROR naming `stamp_offset`), MEASURE the
offset instead of guessing:

```bash
ros2 run sonar_slam_cpp stamp_probe --topics /oculus/sonar_image /dvl/data
```

reports each stream's median stamp-vs-arrival offset, jitter, and drift, and
prints the exact `<sensor>.stamp_offset` value to set (a constant offset δ
otherwise biases every keyframe pose by v·δ / ω·δ, silently — see
`docs/SLAM_EFFECTIVENESS_AUDIT.md`). It works live and against
`ros2 bag play` at rate 1.0. If the probe reports a DRIFTING offset, a
constant `stamp_offset` cannot fully fix that stream.

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
ros2 service call /bruce/slam/slam/undo_manual_correction std_srvs/srv/Trigger
```

removes the most recent manual prior and relaxes the trajectory back; call
repeatedly to peel earlier corrections.

### Field diagnostics

`slam_node` and `dead_reckoning_node` publish `/diagnostics`
(`diagnostic_msgs/DiagnosticArray`, 1 Hz): sync pairing health (the
stamp-offset starvation signature raises ERROR), the NSSM
accept/reject/revert funnel, and the manual-correction counters — watch with
`rqt_runtime_monitor` instead of grepping logs mid-deployment.

### DVL-outage coast

`dvl_coast` (dead reckoning, seconds; 0 = off) bridges DVL dropouts —
bottom-lock loss, `require_valid` drops, a stalled secondary stream — by
dead-reckoning on the last good body velocity rotated through the live
attitude stream, then holding when the budget is spent. Without it the
estimate freezes for the whole outage while the vehicle keeps moving. The
Revolution preset arms 3 s.

### Map persistence & relocalization (multi-session)

```bash
ros2 service call /bruce/slam/slam/save_map std_srvs/srv/Trigger   # end of mission 1
# mission 2:
ros2 launch sonar_slam_cpp slam.launch.xml ... # with slam/map_load_path set
```

`save_map` serializes the whole keyframe map (poses + clouds) to
`map_save_path`; loading it on a later mission restores the map and arms
relocalization — the first dense feature frame is globally scan-matched
against the loaded map (start near previously mapped area), then SLAM
continues in the SAME map frame, closing loops against the previous
session's keyframes.

### USBL / acoustic absolute positioning

Point `usbl/topic` (slam.yaml) at a map-frame position feed
(`PoseWithCovarianceStamped` or `Odometry`). Each fix becomes a
position-only prior on the stamp-nearest keyframe — innovation-gated
(multipath outliers rejected), one per keyframe — turning bounded-drift
SLAM into globally-anchored SLAM. Heading is never taken from USBL.

### Georeferenced deliverables

Give the mapping node a survey `datum` (`[lat, lon, bearing-of-map-x]`,
mapping.yaml) and call `/bruce/slam/mapping/export_map` (Trigger): the occupancy and
intensity grids are written as PNG + UTM world files (`.pgw` + `.prj` —
QGIS/ArcGIS open them as georeferenced rasters) and the trajectory as a
WGS84 GeoJSON LineString.

### Live CFAR tuning

The feature node's `CFAR.*` and `filter.threshold` parameters are dynamic:
`ros2 param set /bruce/slam/feature_extraction CFAR.Pfa 0.005` rebuilds the detector on
the next ping — tune at sea without relaunching. Match the declared types:
`Pfa` takes a double literal (`0.005`), `Ntc`/`Ngc`/`rank`/`filter.threshold`
take integers.

### Map-quality metrics

`ros2 run sonar_slam_cpp map_metrics` subscribes to the aggregated SLAM
cloud (latched — works live or against `ros2 bag play` of a recorded run)
and prints one line per map update: total wall length, local wall thickness
(median / p95), the doubled-wall fraction at the 0.3–3 m probe scale, and the
skeleton cell count — the `docs/MAP_DOUBLING_FIX_PLAN.md` §5 numbers, so
before/after replays compare as numbers instead of screenshots.

> **Check `skel=` before trusting a run.** Thinning can *collapse* walls
> instead of thinning them, and every metric then reads ~0 — including
> `doubled`, i.e. a perfect score for a doubled map. Measured on synthetic
> clouds at `--grid 0.05`:
>
> | input | occupied | skeleton | wall_len | truth |
> | --- | --- | --- | --- | --- |
> | horizontal 10 m wall | 200 | 200 | 9.95 m | ok |
> | single 45° 10 m wall | 142 | 142 | 9.97 m | ok |
> | 30° 10 m wall | 261 | 259 | 13.15 m | **+32%** |
> | two parallel 45° walls, 1 m apart | 566 | **2** | **0.00 m** | **20 m erased** |
>
> A collapse now prints a WARNING naming the ratio, so it cannot pass
> silently — but the underlying thinning is **not fixed**. A single wall of
> either orientation measures correctly, so the trigger is a
> rasterisation/thinning interaction rather than orientation alone. The 30°
> over-read is the staircase problem: the length sum corrects the pure-45°
> case only, so intermediate angles over-count by up to 41%.
>
> Practically: run it on a known cloud first and confirm `skel` is a healthy
> fraction of `cells`. If it warns, fix the thinning before using the numbers
> to judge a tuning change.

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
