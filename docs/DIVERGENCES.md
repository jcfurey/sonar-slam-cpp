# Intentional divergences from `bruce_slam`

This port aims for drop-in behavioral parity with the original Python
`bruce_slam` (`jake3991/sonar-SLAM`). A few places deliberately **diverge** —
each one fixes a latent logic bug that exists in the upstream Python. They are
collected here so the divergences are explicit and auditable, and so the
"identical behavior" claim elsewhere is understood to mean *identical except
for these documented corrections*.

Line numbers refer to `bruce_slam/src/bruce_slam/*.py` in `jake3991/sonar-SLAM`
at the time of comparison.

## Divergences that fix an upstream bug

### 1. Gyro initial roll — degrees used as radians
- **Upstream:** `gyro.py:21` `self.roll, self.yaw, self.pitch = 90., 0., 0.`
  then `gyro.py:71` `gtsam.Rot3.Ypr(self.yaw, self.pitch, self.roll)`.
  `Rot3.Ypr` takes **radians**, so the literal `90.` is 90 rad (~5157°), not
  90°.
- **Here:** `gyro_node.cpp` starts `roll_ = M_PI / 2.0`.
- **Impact:** limited — dead reckoning consumes only the gyro's *yaw*, and
  `Ypr(y,p,r).yaw()` round-trips `y` away from pitch gimbal lock — but the value
  was still wrong.

### 2. Gyro earth-rate compensation — deg/s added to a radian integrator
- **Upstream:** `gyro.py:38` `earth_rate = -15.04107 * sin(lat) / 3600` (deg/s),
  added at `gyro.py:63` to `delta_roll`, which is integrated in radians.
- **Here:** `gyro_node.cpp` converts to rad/s
  (`-15.04107 * sin(lat) * pi / (180*3600)`), unit-consistent with the radian
  integration.

### 3. NSSM search anchored on a stale `current_frame`
- **Upstream:** `slam.py:854` `ret.source_pose = self.current_frame.pose`, but
  `slam_ros.py:211` assigns `self.current_frame = frame` *after*
  `add_nonsequential_scan_matching()` runs — so during NSSM it still holds the
  previous callback's frame, while `ret.source_key` is the newest keyframe and
  `source_points` live in that keyframe's frame. The global-init search is thus
  anchored up to a keyframe step away from its own geometry.
- **Here:** `slam_core.cpp` uses `ret.source_pose = keyframes[ret.source_key]->pose`.

### 4. NSSM bounds — loop-variable leak
- **Upstream:** `slam.py:878-932` reassigns `cov = self.keyframes[source_frame].cov`
  inside the fan-selection loop, then computes the ±5σ search bounds from the
  leaked `cov` **after** the loop (`slam.py:929`) — i.e. the *oldest* source
  frame's covariance, whichever the loop ended on.
- **Here:** `slam_core.cpp` derives the bounds from `ret.cov`
  (`keyframes[source_key]`, the anchor keyframe), consistent with
  `ret.source_pose`. That keyframe's marginal was just refreshed by
  `update_factor_graph`, so it is current.

### 5. SSM `ret.cov` — standard deviation used as variance
- **Upstream:** `slam.py:634` `ret.cov = np.diag(self.odom_sigmas)` (unsquared),
  while the same file squares correctly elsewhere (`slam.py:383`
  `np.diag(icp_odom_sigmas)**2`).
- **Here:** `slam_core.cpp` uses `odom_sigmas.array().square()`. This field is
  currently vestigial in the C++ (the simplified global-init cost dropped the
  pose-prior term that consumed it upstream), so the correction is inert today
  but kept correct for any future use.

## Carried-over limitations left in place (with rationale)

### A. Only the newest keyframe's covariance is refreshed
- **Upstream:** `slam.py:1232-1234` (`# Only update latest cov`), and the
  authors' own `slam.py:51` `# TODO propagate cov from previous keyframe`.
- **Here:** same behavior (`slam_core.cpp update_factor_graph`). Recomputing
  every keyframe's marginal on every update is O(n²) over a mission and would
  starve the single-threaded callback. The main consumer of a *fresh*
  covariance — the NSSM search bounds — was fixed to use the just-refreshed
  anchor-keyframe covariance (divergence #4), so the residual staleness only
  affects the second-order fan **padding** used for NSSM candidate
  pre-selection. Left as-is pending a cheaper incremental-covariance approach.

## Parity testing

`parity_vs_python.cpp` / `test/parity_driver.py` compare per-function numeric
output against the Python originals. Divergences #1-#5 change behavior only in
the specific code paths above; the CFAR, ICP, downsample, MCD, transform, and
global-init-cost parity fixtures are unaffected.

`test/interp_spline_test.cpp` checks `Interp1d` against `scipy.interp1d`: LINEAR
and the not-a-knot CUBIC both match scipy to ~1e-15 (incl. a uniform grid, the
zero-pivot case). `test/censi_covariance_test.cpp` Monte-Carlo-validates the
opt-in Censi ICP covariance.
