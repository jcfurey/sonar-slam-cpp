# sonar_slam_cpp

Sonar scan matching and pose-graph optimization for ROS 2.

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
`/diagnostics`. The input cloud is projected once through the sensor transform,
then leveled with fused roll/pitch for planar registration. Large head sweeps
are rejected by boresight elevation while sonar_proc's mapping streams continue
unaffected.

## Public interfaces

Inputs:

- `/sensor/sonar/sonar0/proc_points` (`sensor_msgs/PointCloud2`);
- timestamped TF for the cloud frame, `base_link`, and `odom`;
- optional `/initialpose` and USBL/LBL position fixes.

Outputs:

- `/bruce/slam/slam/pose`;
- `/bruce/slam/slam/odom`;
- `/bruce/slam/slam/traj`;
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
