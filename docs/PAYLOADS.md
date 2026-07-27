# Sensor payload configurations

The stack is payload-agnostic: every node selects its sensor drivers by
**parameter string**, and each driver is a small adapter that normalizes a raw
driver message into the pipeline's internal reading type. Supporting a new
vehicle means picking the right driver names + topics in a config overlay — no
code changes.

Ready overlays are provided under `config/payloads/`:

| Payload | Overlay | Vendor packages needed at build |
| --- | --- | --- |
| Legacy BlueROV (bruce_slam original) | `bluerov_legacy.yaml` | `sonar_oculus`, `rti_dvl`, `bar30_depth`, `kvh_gyro` |
| Modern BlueROV2 (Water Linked A50) | `bluerov2_waterlinked.yaml` | **none** (standard messages only) |
| **Deep Trekker Revolution** | `deeptrekker_revolution.yaml` | **none** (standard messages only) |
| Synthetic demo (no hardware, no bags) | `sim_demo.yaml` | **none** — driven by `sim_payload`; see `launch/demo.launch.xml` and the README quick demo |
| Default no-op marker | `none.yaml` | **none** — sets only `sonar_slam_payload`, so the per-node base configs apply unchanged |

The two standard-message payloads differ only in their default topics and
frame conventions (the Revolution's internal AHRS vs. the BlueROV2's
MAVROS/flight-controller IMU) — both run the same standard `dvl_standard` /
`fluid_pressure` / `enu` / `projected_sonar` adapters, so neither needs a
vendor package.

## The build no longer hard-requires the vendor packages

`sonar_oculus`, `rti_dvl`, `bar30_depth` and `kvh_gyro` are the four
BlueROV-era vendor message packages. They are now found **`QUIET`** by CMake
(`find_package(... QUIET)`); each one present defines `HAVE_<PKG>`, which
compiles in its driver adapter. A build without them succeeds and simply omits
those adapters — selecting one at runtime then raises a clear error
(`driver '<name>' needs the '<pkg>' message package, which was not present at
build time`) instead of crashing.

So a Deep Trekker Revolution deployment builds against nothing but the standard
ROS 2 message packages the port already depends on
(`marine_acoustic_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`,
`std_msgs`), with no BlueROV vendor stubs in the workspace at all.

`rosdep` note: the vendor stubs (and `dvl_msgs`) are still listed in
`package.xml` (as `<depend>`) so `colcon` orders them first *when they are in
the workspace*. In a standard-only workspace they are absent; run
`rosdep install --from-paths src --skip-keys "sonar_oculus rti_dvl bar30_depth kvh_gyro dvl_msgs"`.

## Running the Revolution

```bash
ros2 launch sonar_slam_cpp slam.launch.xml \
    payload_config:=$(ros2 pkg prefix --share sonar_slam_cpp)/config/payloads/deeptrekker_revolution.yaml \
    enable_gyro:=false
```

`payload_config` is an overlay merged **after** each node's base config, so it
overrides only the driver/topic/frame parameters. `enable_gyro:=false` omits
the FOG integration node (the Revolution has no fibre-optic gyro). If you leave
it on, the gyro node detects the missing `kvh_gyro` driver and stays idle
rather than aborting.

Edit the topic names in the overlay to match your driver launch.

## Revolution sensor → driver mapping

| Sensor | Device | Publishes | Driver (`*/driver`) | Notes |
| --- | --- | --- | --- | --- |
| DVL | Water Linked DVL-A50 | `marine_acoustic_msgs/Dvl` | `dvl_standard` | Official `waterlinked_dvl` driver. Gates on `beam_velocities_valid` + rejects water-track (`require_valid`, default true). |
| DVL (alt) | Water Linked DVL-A50 | `dvl_msgs/DVL` | `dvl_a50` | Community paagutie/ndahn driver. Needs the optional `dvl_msgs` package at build time. |
| Attitude | internal ENU AHRS | `sensor_msgs/Imu` | `enu` | REP-145 ENU; adapter rotates to the pipeline's z-down. `imu_pose` identity for an aligned mount. |
| Depth | internal pressure | `sensor_msgs/FluidPressure` | `fluid_pressure` | `depth = (P − atmospheric_pressure)/(ρg)`. Set `atmospheric_pressure: 0.0` for a gauge-pressure driver. |
| Sonar | Blueprint Oculus M750d / M1200d | `marine_acoustic_msgs/ProjectedSonarImage` | `projected_sonar` | Fan geometry read from the message; beam order normalized to ascending bearing. |

## All available driver adapters

Selected per sensor via the `<sensor>/driver` parameter (`dvl/driver`,
`depth/driver`, `imu/driver`, `gyro/driver`, `sonar/driver`).

### DVL — `dvl/driver`
| Name | Message | Package | Validity gating |
| --- | --- | --- | --- |
| `rti_dvl` | `rti_dvl/DVL` | `rti_dvl` (vendor) | none |
| `dvl_standard` (`marine_dvl`) | `marine_acoustic_msgs/Dvl` | standard | `beam_velocities_valid`, rejects `DVL_MODE_WATER`, finite check |
| `dvl_a50` (`dvl_msgs`) | `dvl_msgs/DVL` | `dvl_msgs` (optional) | `velocity_valid`, finite check |
| `twist_stamped` | `geometry_msgs/TwistStamped` | standard | none |
| `twist_cov` | `geometry_msgs/TwistWithCovarianceStamped` | standard | none |

`dvl.require_valid` (default `true`) turns the validity gating on/off for the
`dvl_standard`/`dvl_a50` adapters. Caveat for `dvl_standard`: the message spec
defines `beam_velocities_valid` as "the optional per-beam data is populated",
but the official `waterlinked_dvl` driver publishes its bottom-lock validity in
it — the default gating assumes that convention. A spec-conforming publisher
that only reports the derived velocity leaves the flag `false` on every
message; the adapter logs a throttled warning naming the remedy — set
`dvl.require_valid: false` for such a driver.

### Depth — `depth/driver`
| Name | Message | Package | Conversion |
| --- | --- | --- | --- |
| `bar30` | `bar30_depth/Depth` | `bar30_depth` (vendor) | depth used directly |
| `fluid_pressure` | `sensor_msgs/FluidPressure` | standard | `(P − atmospheric_pressure)/(ρg)` |
| `odom` | `nav_msgs/Odometry` | standard | `depth = z_sign · z` (`depth.z_sign` default −1, ENU up) |
| `float` | `std_msgs/Float64` | standard | value used directly, metres positive-down (stamped on arrival) |

### IMU — `imu/driver` (all use `sensor_msgs/Imu`)
| Name | Frame handling |
| --- | --- |
| `vn100` | legacy VectorNav (imu_pose roll −90 + fixed offsets applied downstream) |
| `enu` (`3dm_gx5`, `microstrain`) | ENU→NED + FLU→FRD to the pipeline z-down |
| `ned` | already z-down; pass-through, no conversion, no legacy offset |

### Gyro — `gyro/driver`
| Name | Message | Package |
| --- | --- | --- |
| `kvh_gyro` | `kvh_gyro/Gyro` | `kvh_gyro` (vendor) |
| `vector3_stamped` | `geometry_msgs/Vector3Stamped` (delta angles, rad) | standard |

### Sonar — `sonar/driver`
| Name | Message | Package | Notes |
| --- | --- | --- | --- |
| `oculus_compressed` | `sonar_oculus/OculusPing` | `sonar_oculus` (vendor) | JPEG-in-message, decoded with a corrupt-frame guard |
| `oculus_uncompressed` | `sonar_oculus/OculusPingUncompressed` | `sonar_oculus` (vendor) | raw image via cv_bridge |
| `projected_sonar` | `marine_acoustic_msgs/ProjectedSonarImage` | standard | beam-major message payload decoded to the internal range-major image; bearings from `beam_directions`, order normalized |
| `image` | `sensor_msgs/Image` | standard | plain polar image; needs `sonar/range_resolution` (+ optional `sonar/horizontal_fov`) |

Leaving `sonar/driver` blank auto-selects `oculus_compressed` /
`oculus_uncompressed` from `compressed_images`, preserving the historic
default.

## Adding another payload

1. Copy `config/payloads/deeptrekker_revolution.yaml`, adjust the driver names
   and topics for your hardware.
2. If your sensor speaks a message the table above doesn't cover, add an
   adapter branch in `src/ros/sensors.cpp` (guard it with `#ifdef HAVE_<PKG>`
   and an optional `find_package(<pkg> QUIET)` in `CMakeLists.txt` if it needs
   a non-standard message package).
3. Standard `marine_acoustic_msgs` / `sensor_msgs` messages need no new
   dependency and always compile in.

The internal reading types the adapters target (`DvlReading`, `DepthReading`,
`ImuReading`, `GyroReading`, `SonarPing`) are defined in
`include/sonar_slam_cpp/sensors.hpp` and `sonar_geometry.hpp`.
