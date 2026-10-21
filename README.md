# sonar_slam_cpp

ROS 2 sonar scan matching and pose graphs with a maintained constrained
x/y/yaw model and an opt-in physical SE(3) model for measured XYZ.

This package intentionally owns only:

- keyframe selection;
- sequential and non-sequential scan matching;
- ICP covariance, degeneracy gates, PCM and ISAM2 optimization;
- relocalization, graph persistence and absolute/operator priors;
- `map -> odom`, optimized pose, odometry, trajectory and graph constraints.

The surrounding workspace owns the rest:

- `sonar_proc` decodes sonar images, suppresses artifacts, selects candidate
  returns and publishes `/sensor/sonar/sonar0/proc_points`;
- the Sonar 3D-15 driver supplies measured body-axis XYZ;
- the fused odometry stack publishes timestamped `odom -> base_link` TF;
- `sonar_map_assembler` retains occupancy/survey evidence and applies the
  optimized trajectory to historical map products.

## Timing model

Oculus is a flash-imaging sonar. Every point in one `proc_points` message is
treated as one rigid acquisition at `header.stamp`; there is no invented
per-point scan time and no nearest-odometry-message approximation.

For each ping, `slam_node` requests both:

- `base_link <- cloud frame` at the ping stamp; and
- `odom <- base_link` at the same stamp.

If either exact transform is unavailable, the ping is dropped and reported on
`/diagnostics`. In `registration_model: constrained_xyyaw` (the default),
the cloud is projected through the live head transform, then leveled with
fused roll/pitch. XYZ is retained through
keyframe accumulation, correspondence search, trimming, overlap checks and
ICP. Source elevation is shifted into the target keyframe's pressure-depth
chart before matching.

This registration motion model is constrained to `x/y/yaw`:
elevation decides *which* returns correspond, but sonar factors cannot modify
depth, roll or pitch. Those states remain owned by pressure and the IMU. This
avoids both failures of the previous flattened matcher—different elevations
aliasing in XY and floor/head-sweep geometry being discarded—without allowing
the wide Oculus elevation aperture to invent a free SE(3) solution. A positive
`max_head_pitch` remains available as an emergency/operator gate; its default
is disabled.

The Sobol global initializer still searches only `x/y/yaw` on an XY occupancy
seed. Every seed is subsequently refined and validated with constrained XYZ
ICP, XYZ overlap, sampled covariance and the existing graph-defense gates.

With `registration_model: se3`, calibrated extrinsics place measured returns
in the physical body frame. Fused attitude is not applied to the cloud, and
pressure depth is not prealigned. Registration optimizes all three translations
and all three rotations. The physical graph uses soft fused-odometry increments
with `se3.odom_sigmas`, not zero-attitude or fixed-depth unary factors. TF does
not supply uncertainty; these six sigmas require commissioning. Raw IMU and
pressure factors are not added to their already-fused output.

SE3 uses trimmed, robust point-to-plane residuals with cached target normals
and search trees. A scaled six-dimensional information test rejects weak
geometry such as isolated walls/floors. Comparable separated fits are rejected
using their full pose covariances. Loop PCM uses the six-dimensional 99-percent
chi-square threshold 16.8119 and full adjoint covariance transport; post-loop
verification checks all six link residuals. Rejected rounds restore the
committed graph, including after operator/absolute corrections.

## Open-water behavior

Sparse sonar is treated as absence of an acoustic pose measurement. A scan must
have enough finite returns distributed across distinct XYZ voxels and horizontal
azimuth bins before its cloud can enter registration history. With valid
timestamped transforms and odometry, the normal keyframe motion/time/head gates
still create odometry-only trajectory anchors, including the first frame. The
mapping assembler can attach occupancy and survey evidence to those anchors;
their empty registration clouds cannot supply sequential or loop-closure evidence.
The loaded-map relocalization gate still requires an informative acoustic match
before any new state enters the saved map's coordinate frame.

`~/mapping_traj` supplies additional pose-only anchors at
`mapping_anchor_interval` acquisition seconds (default 2 s), even during a
stationary interval. Each anchor retains its exact valid DR pose/stamp and its
parent graph keyframe. Its published pose is recomputed as
`parent.pose3 * parent.dr_pose3.inverse() * anchor.dr_pose3`, so loop, manual and
other graph corrections propagate through the mapping history. This creates no
extra graph states, acoustic evidence, or registration factors; `~/traj` remains
the graph-keyframe trajectory.

Selected mapping anchors wait one acquisition interval before publication so
same-ping map/survey clouds arrive before the assembler advances its association
cutoff. After valid input stops for one wall-time interval, a wall timer flushes
pending anchors and the final cached observation using their original stamps.
It never creates a pose at the current time from stale input. Rewind clears both
histories and their retained messages. Zero/negative or unnormalized acquisition
timestamps are rejected before TF lookup.

Intake, graph processing and TF publication run in independent callback groups.
Mapping anchors advance while registration is busy, then adopt newer graph
parents as optimization catches up. The live registration queue drops its
oldest pending ping when full. With `input_reliable: true`, excess registration
payloads spill to a temporary directory and replay in order; `input_queue_depth`
bounds resident payloads. Set `TMPDIR` to storage with room for a replay backlog.
Disk failures are reported as errors. Registration backlog/drop/spill counts,
processing age, and stale input are exposed on `/diagnostics`.

`registration.max_points` bounds each retained candidate cloud (default 5000).
Larger inputs are voxel sampled and deterministically capped before entering
the queue or graph. This limit applies to registration evidence; the mapper
continues to receive its original survey cloud. Each constrained ICP worker
reuses its target search tree across iterations and restarts, with parallelism
bounded by `constrained_3d.max_threads` (default 4). Reaching the iteration limit
is a registration failure. The CPU XY initializer uses at most four workers;
`OMP_NUM_THREADS` can reduce that and bounds the legacy planar pool.
SE3 caps accumulated sources at `registration.max_points`, caps local targets
at four times that value, and accepts source/target windows of at most 16
frames. Relocalization streams a bounded XY proposal index
(`se3.max_seed_points`, default 250000), then reuses physical reference-frame
matchers. It does not build a dense whole-map normal/search structure.
The shared XY proposal raster keeps the usual `point_noise / 10` resolution
for local footprints and coarsens larger footprints to at most 16 MiB of
byte cells and 16384 cells per axis. Coarsening affects
proposals; XYZ refinement and its observability/ambiguity gates still determine
registration acceptance. `GlobalInitResult` reports the actual raster extent
and resolution. Coarsening can reduce candidate discrimination on large
footprints; local registration still needs enough geometric overlap to converge.
Sobol sampling is capped at one million candidates per attempt,
local refinement respects the configured bounds, and zero coarse overlap is
an initialization failure. Invalid sampling settings fail startup in both models.
Graph and mapping history still grow with
mission length; the per-cloud limit does not bound total mission memory.

Sequential scan matching then requires both a meaningful overlap fraction and a
finite, observable ICP covariance. Failure adds only a span-scaled odometry link.
Admission counts, rejection reasons, input mode and degenerate SSM rejections are
published on `/diagnostics`, alongside separate `odometry_only_keyframes`,
`registration_keyframes`, and retained `registration_points` counts. An assembled
map supported only by odometry remains subject to odometry drift; map production
does not imply successful acoustic registration.
`mapping_anchors` and `mapping_pending_anchors` expose the separate mapping history.

## Public interfaces

Inputs:

- `/sensor/sonar/sonar0/proc_points` (`sensor_msgs/PointCloud2`);
- timestamped TF for the cloud frame, `base_link`, and `odom`;
- optional `/initialpose` and USBL/LBL position fixes.

Outputs:

- `~/pose`;
- `~/odom`;
- `~/traj`;
- `~/mapping_traj`;
- `~/constraint`;
- `map -> odom` TF when `publish_tf` is enabled.

The supplied launch keeps the `/bruce/slam/slam` prefix. `map_frame`,
`odom_frame` and `base_frame` configure frame names. Trajectories and constraints
are reliable, transient-local snapshots with depth one. Missing or invalid
marginals publish variance `1e6` and a diagnostic warning. Constrained output
marks depth/roll/pitch as unobserved. SE3 publishes a full positive definite
6x6 covariance, converted from the source-body right tangent
`[rx,ry,rz,tx,ty,tz]` to ROS position and small fixed-axis rotation errors in
map axes about the pose origin. `registration.hpp` documents the result
contract and includes a constrained adapter with exactly zero information in
unobserved directions.
Constrained between-keyframe output propagates parent heading uncertainty
through the displacement and rotates odometry noise into the destination's
map axes, retaining position/heading cross covariance. Covariance validation
normalizes numeric scale before symmetry/eigenvalue tests; an invalid or
overflowing publication conversion uses unknown uncertainty.

Manual poses and USBL positions must use the configured `map_frame`; other
reference frames are rejected. `usbl.frame_id` explicitly names the static
transponder mount. Empty means a position already corrected to `base_frame`.
A configured unresolved mount rejects the fix. USBL motion and mount offsets
use exact fix-time TF and the current map correction. Accepted corrections
republish pose, TF and both trajectories immediately.
In SE3, a USBL factor observes the transponder point on the physical pose;
its lever arm remains coupled to attitude during optimization. Fix-time
odometry uncertainty is included for fixes between keyframes. RViz planar
corrections remain the default. `manual_correction_mode: se3` instead requires
a full map-frame pose with a positive definite message covariance.
Nonzero manual-pose stamps identify the fix acquisition time and require exact
`odom <- base_link` TF at that stamp. The correction is brought back to the
newest graph keyframe using that motion, even while sonar processing lags or
sonar input has stopped. Invalid stamps and unavailable fix-time TF reject the
correction. A zero stamp retains the last-processed-sonar-pose convention.
Planar corrections in SE3 retain predicted depth and attitude at the fix time;
full-pose corrections also transport their covariance and configured DR noise.

Loaded-map relocalization requires converged, distinguishable global hypotheses,
observable local covariance, refined XYZ overlap and at least
`relocalize_min_confirmations` newer consistent observations (default 2).
Duplicate stamps cannot confirm a match. Confirmation reduces false matches;
representative replay is still needed to commission thresholds for a venue.

`relocalize_max_ms` sets a shared steady-clock budget for each attempt (default
5000 ms; supported range 1–60000). Proposals, reference preparation, hypotheses
and covariance estimation share that budget. Timeouts reject incomplete
searches and retain the loaded graph for retry. Full SE3 chooses its nearest
references from keyframes with enough sonar points; empty odometry anchors
cannot displace usable references. Sampled registration's existing 2-second
budget also covers robust covariance estimation, and a truncated population
cannot supply measurement covariance. Full SE3 registration includes normal
estimation and initialization in its 2-second search budget.

These are cooperative budgets: an individual library/kernel call or graph
update cannot be interrupted, so elapsed callback time can exceed the limit.
Measure acquisition-to-publication latency on representative growing maps.

Maps declare their measurement model, cloud chart, calibration ID and depth
datum. Constrained version-2 maps retain the established reference-shape
semantics. Loading a metadata-free version-1 imaging map requires an explicit
`allow_legacy_map: true` migration; saving it writes version 2. SE3 maps preserve
the physical factors, their noise, manual-undo history and full pose/covariance
with an integrity checksum. SE3 relocalization adds a between factor to a
matched reference pose, preserving the map's gauge uncertainty. SE3 persistence
requires a nonempty `map_calibration_id`; update that ID when the calibration
changes, and set `map_depth_datum` consistently across sessions. Incompatible
maps are rejected, and any configured map-load failure stops startup.

Algorithm parameters are explicitly declared read-only startup settings. Restart
to change them; unknown overrides and contradictory covariance configurations
fail startup. `use_sim_time` retains the ROS clock parameter behavior.

Library callers must serialize operations on a `Slam` instance. Call
`configure()` on an empty session, prepare the first frame with `add_prior()`,
and commit that same frame with `update_factor_graph(frame)`. Later frames use
`add_odometry()` or `add_sequential_scan_matching()` before the corresponding
commit. Admission validates the cloud chart, finite XYZ/poses and strictly
increasing acquisition stamps. An empty `update_factor_graph()` is a no-op.

`has_pending_update()` reports prepared work. A mismatched commit or conflicting
correction/map operation rejects without consuming it. Commit the original
frame or call `discard_pending_update()` to cancel before proceeding. The node
uses that cancellation after unexpected staging exceptions; solver failures
roll back and rebuild from committed factors. Sequential acceptance counters
include only committed measurements. Reconfiguration requires an empty session
after `resetSession()`; map loading also requires no pending update.

Mapping clouds and occupancy grids belong to the workspace mapping packages.

## Measured 3D sonar and full pose SLAM

Select the sensor model independently of the registration model. For an
isolated full-pose replay evaluation:

```bash
ros2 launch sonar_slam_cpp slam.launch.xml \
  input_model:=measured_xyz points_topic:=/sensor/sonar/sonar3d/points \
  registration_model:=se3 namespace:=/evaluation/sonar3d \
  map_frame:=evaluation_map use_sim_time:=true publish_tf:=false
```

`imaging` (default) uses optical +Z boresight; `measured_xyz` uses body +X
boresight and omits the imaging fan-angle shortcut in loop-target selection.
`constrained_xyyaw` retains pressure/IMU ownership of depth/attitude; `se3`
allows sonar to correct all six pose directions. SE3 requires `measured_xyz`;
imaging elevation hypotheses cannot select it. Supply calibrated sensor-to-base TF and exact acquisition-time
odometry. The example leaves TF ownership configurable for evaluation alongside
an existing localization stack; a deployed graph-backed map needs one
`map -> odom` authority and external mapping trajectories in that same frame.

The workspace profiles coordinate the complete path:

| Profile | Estimator | Global map/view frame |
| --- | --- | --- |
| `sonar3d` | Exact odometry baseline | `odom` |
| `sonar3d_constrained` | Measured XYZ, constrained x/y/yaw graph | `map` |
| `sonar3d_se3` | Measured XYZ, physical six-axis graph | `map` |

Both graph profiles route external mapping anchors, keep local obstacles in
`odom`, and use the localization TF ownership interlock. The SE3 map/earth
data stage consumes the corrected six-axis pose. Recording/replay preserves
the selected model, mount, calibration ID and depth datum. The workspace
`docs/MAPPING_PROFILES.md` gives the bringup commands.

SE3 is opt-in and validated with synthetic geometry and ROS integration tests.
Its local covariance is conditional on correspondences/normals and includes
configured noise floors; it is not a calibrated sonar accuracy model. Global
proposals use bounded yaw, depth and attitude priors, followed by six-axis
refinement and confirmation. Moving, synchronized sonar/pose recordings and
independent geometric reference are still needed for hardware accuracy and
sustained mission-latency validation.

## Build and test

```bash
colcon build --packages-select sonar_slam_cpp \
  --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select sonar_slam_cpp
colcon test-result --all --verbose
```

The node is also registered as the `sonar_slam::SlamNode` component. Use a
multithreaded container with at least three workers to serve its callback groups.
Installed CMake targets are `sonar_slam_cpp::sonar_slam_core`,
`sonar_slam_cpp::sonar_slam_ros` and `sonar_slam_cpp::sonar_slam_component`.
The test suite compiles an external consumer against those installed targets.

CUDA accelerates the XY global-initialization seed and legacy planar
nearest-neighbor queries. Constrained XYZ and physical SE3 registration use CPU search trees.
Diagnostics report the registration and seed backends separately.
Set `SONAR_SLAM_FORCE_CPU=1` to force CPU equivalents of the CUDA kernels.
