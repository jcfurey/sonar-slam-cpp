# sonar_slam_cpp

Tilt-aware constrained-3D sonar scan matching and pose-graph optimization for
ROS 2.

This package intentionally owns only:

- keyframe selection;
- sequential and non-sequential scan matching;
- ICP covariance, degeneracy gates, PCM and ISAM2 optimization;
- relocalization, graph persistence and absolute/operator priors;
- `map -> odom`, optimized pose, odometry, trajectory and graph constraints.

The surrounding workspace owns the rest:

- `sonar_proc` decodes sonar images, suppresses artifacts, selects candidate
  returns and publishes `/sensor/sonar/sonar0/proc_points`;
- the fused odometry stack publishes timestamped `odom -> base_link` TF;
- `vdb_mapping_ros2` consumes `map_points`, `survey_points`, `free_points` and
  the optimized trajectory to produce corrected occupancy and survey maps.

## Timing model

Oculus is a flash-imaging sonar. Every point in one `proc_points` message is
treated as one rigid acquisition at `header.stamp`; there is no invented
per-point scan time and no nearest-odometry-message approximation.

For each ping, `slam_node` requests both:

- `base_link <- cloud frame` at the ping stamp; and
- `odom <- base_link` at the same stamp.

If either exact transform is unavailable, the ping is dropped and reported on
`/diagnostics`. The input cloud is projected once through the live head
transform, then leveled with fused roll/pitch. XYZ is retained through
keyframe accumulation, correspondence search, trimming, overlap checks and
ICP. Source elevation is shifted into the target keyframe's pressure-depth
chart before matching.

The registration motion model is deliberately constrained to `x/y/yaw`:
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

- `/bruce/slam/slam/pose`;
- `/bruce/slam/slam/odom`;
- `/bruce/slam/slam/traj`;
- `/bruce/slam/slam/mapping_traj`;
- `/bruce/slam/slam/constraint`;
- `map -> odom` TF when `publish_tf` is enabled.

Mapping clouds and occupancy grids are deliberately not published here.

## Build and test

```bash
colcon build --packages-select sonar_slam_cpp \
  --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select sonar_slam_cpp
colcon test-result --all --verbose
```

CUDA accelerates only global-initialization cost evaluation and nearest-neighbor
matching. Set `SONAR_SLAM_FORCE_CPU=1` to force their CPU equivalents.
