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

### 6. OS-CFAR threshold factor — order-statistic off-by-one
- **Upstream:** `CFAR.py:116-121` solves the threshold factor from
  `Γ(N+1)/Γ(N−rank+1)·Γ(τ+N−rank+1)/Γ(τ+N+1)` — the false-alarm model for the
  **rank-th smallest** training cell — while the detector (`cpp/cfar.cpp`,
  reproduced by this port's CPU and CUDA paths) thresholds on `train[rank]`
  after `nth_element`, the **(rank+1)-th smallest**. The solved τ therefore
  over-thresholds: realized P_FA = target × (N−rank)/(τ+N−rank). At the
  shipped config (Ntc 40, rank 10, Pfa 0.1) that is ≈ 0.0767 instead of 0.1
  (Monte Carlo: 0.0766).
- **Here:** `pfa_os()` uses the product with rank+1 factors
  (`Γ(N−rank)`, `Γ(τ+N−rank)`), matching the detector's actual statistic;
  realized P_FA hits the target (Monte Carlo: 0.0999). Full derivation and
  validation: `docs/MATH_NOTES.md` §5, `test/cfar_math_test.cpp`. Note the
  shipped default `alg: SOCA` is unaffected; only OS runs change. The
  `parity_vs_python` harness is also unaffected (it consumes τ values from
  the Python fixture).

### 7. Mapping grid column-growth — wrong axis for the intensity/counter grids
- **Upstream:** `mapping.py:557-565, 576-583` — in `adjust_bounds`, the two
  COLUMN-growth branches grow the intensity and counter grids with `np.r_`
  (row concatenation) and `self.inc_r`, i.e. they pad rows when they must pad
  columns. Only the logodds grid (grown with `np.c_`) is correct. The bug is
  latent upstream because `pub_intensity` defaults False, so the intensity
  branch never runs.
- **Here:** `src/core/mapping.cpp adjust_bounds` grows all three grids
  (logodds, intensity, counter) by the SAME column primitive (`grow_cols` /
  `grow_rows`) with the matching increment, so the intensity mosaic stays
  aligned as the map expands sideways. The port enables `pub_intensity` (the
  backscatter mosaic is a shipped product), so the branch is now live and had
  to be correct. (`mapping_node`, the port of `mapping.py`; see
  `SONAR_MAPPING_ARCHITECTURE.md` §5.)

### 8. Sampled-covariance registrations run in parallel (`parallel_cov_samples`)
- **Upstream:** `slam.py` runs the `cov_samples` (30) ICP registrations of the
  "sampled" covariance method sequentially on one core with a 2 s wall-clock
  cap — with SSM + NSSM both on `sampled`, that is up to 60 registrations
  (≈4 s worst case) per keyframe on a single core, the node's dominant CPU
  cost.
- **Here:** `parallel_cov_samples` (default true) runs the registrations
  across a per-OpenMP-thread pool of identically-configured libpointmatcher
  engines (`ICP::compute_batch`). Each guess's registration is IDENTICAL to
  the sequential result — the configured chain (`icp.yaml`) has no sampling
  filters, so registration is deterministic given (source, target, guess) —
  and samples are collected in guess order. The only behavioral divergence is
  the 2 s cap: sequentially it could truncate the sample set mid-way; in
  parallel it rarely fires, so FAST-MCD typically sees all 30 samples (a
  strictly better-populated covariance estimate). Set false to restore the
  historical one-core loop. GPU note: the registrations stay CPU
  (libpointmatcher) by decision;
  the overlap/correspondence 1-NN (`cloud_ops match()`) is GPU-dispatched
  (exact brute force, same -1/inf contract as the KDTree).

### 9. NSSM target = clustered revisit, not raw argmax overlap
- **Upstream:** `slam.py` selects the loop-closure target keyframe as the single
  frame with the maximum in-fan overlap (`argmax(counts)`), among all keyframes
  outside the `min_st_sep` exclusion.
- **Here:** `slam_core.cpp` `initialize_nonsequential_scan_matching` requires the
  target to be a genuine revisit: (1) a `min_revisit_sep` **floor** — the target
  must be at least that many keyframes older than the source, which excludes the
  current pass's recent trail; then (2) the survivors are segmented into
  contiguous index **runs**, one per visiting episode, the trailing
  (highest-index) run is dropped as the current pass's own tail when more than
  one run exists, and **exactly one** run — the one carrying the most in-fan
  points — becomes the target submap. The floor is primary so the fix is robust
  when the sonar range covers the whole environment and candidates form one
  unbroken run with no gap to cluster on (e.g. a small pool).

  Upstream's argmax anchors every closure to a near-sequential trailing frame
  ("same area, same time") because those carry the most overlap, so out-and-back
  passes are never tied together and walls render doubled.

  Keeping only ONE run matters (2026-07-25): an earlier form of this fix dropped
  just the trailing run, so with three or more runs two distinct earlier
  episodes survived together and `get_points` aggregated both into a single
  target cloud **using current graph estimates**. If those episodes were
  mis-aligned — exactly the situation the closure exists to fix — ICP registered
  against a doubled target and produced a compromise transform, reintroducing
  the wall doubling this whole path exists to prevent. rtabmap has no such
  exposure: `getPaths` segments candidates into neighbour-link-connected runs
  and attempts one detection **per path**, never merging clouds across paths.

  This change only narrows *which* frames become the target; all acceptance
  gates (compass, degeneracy, PCM, DCS, post-loop revert) are unchanged.

### 10. Occupancy free-space: a row-0 hit erased its whole bearing column
- **Upstream:** `mapping.py:218-223` locates the first return per bearing column
  with `argmax`, which returns 0 both for "hit in row 0" and for "no hit at all"
  (an all-equal array), then treats 0 as the latter and frees the entire column.
  A single row-0 return therefore wiped every hit in that column and stamped
  miss-logodds out to max range — a radial free-space stripe carved straight
  through real structure, in the occupancy product only, silently.
- **Here:** `mapping.cpp` `add_keyframe` uses an explicit scan that distinguishes
  the two cases (`first` stays `sub_rows_` only when no cell exceeded the
  threshold), so a row-0 hit frees nothing before it and a hitless column is
  still freed entirely.
- **Reachability:** row 0 is not a corner case. The row binding clamps, so any
  feature at or inside `fan_range_min + 2*range_resolution` lands there, and the
  Gaussian inflation spreads a hit up to `inflation_range` (0.3 m deployed) into
  it. Near-field ringdown passing CFAR is enough.
- **Impact:** occupancy grid only — the pose graph never consumes this tile, so
  the failure shows up as a corrupted map rather than a SLAM divergence, which
  is why it could persist unnoticed.

### 11. Feature extraction reads the POLAR mask, not its Cartesian remap

- **Upstream:** `feature_extraction.py` remaps the binary CFAR mask to a
  Cartesian image with nearest-neighbour interpolation, calls `findNonZero` on
  that, and converts the resulting pixel indices to metres.
- **Here:** `feature_extraction_node.cpp` runs `findNonZero` on the polar mask
  and converts each `(range bin, beam)` exactly —
  `range = range_min + row*res`, `bearing = bearings[beam]` — using the same
  axis convention `generate_map_xy` defines, so cloud orientation is unchanged.
  `filter/extract_polar: false` restores the upstream order.
- **Why:** nearest-neighbour resampling is a *pull*, so a polar cell reaches
  the cloud only if some destination pixel happens to sample it. Inside the
  range where beam arc spacing is finer than the Cartesian cell (6.8 m for an
  Oculus at 0.03 m, 9.4 m for the Revolution preset) cells compete and lose:
  68.7% and 73.9% of near-field cells respectively are unreachable, and the
  survivors arrive snapped to the grid. Measurements and the bounded-impact
  analysis are in `docs/SONAR_FRONTEND_REVIEW.md` §5 — the practical damage is
  confined to roughly the inner 3 m because `filter/resolution` (0.5 m)
  downsamples far more aggressively than either grid, so this is a
  correctness/near-field fix, not a claimed accuracy jump.
- **Cost:** negative — the mask remap disappears entirely. The Cartesian remap
  survives only for the `feature_img` visualization, which is already gated on
  a subscriber.
- **Verification:** 35/35 probe cells across the fan agree between the two
  conversions to within one Cartesian cell (worst 0.032 m against a 0.030 m
  cell), which is exactly the grid snap being removed — confirming no mirror
  or axis swap was introduced.

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
retained point-to-point covariance math; runtime selection is disabled because
the shipped ICP objective is now point-to-plane.

## Angle-innovation wrapping in the Kalman node (deliberate, correctness over parity)

kalman.py feeds the raw `yaw - yaw0` (and roll) measurement into the update
step with no wrapping, so every +-pi crossing of the vehicle's heading drives
a ~2*pi residual through the gain and slews all coupled states for the
convergence window. The C++ port wraps the roll/yaw innovations into
(-pi, pi] against the predicted measurement (`H_imu * x_pred`) before
correcting. Behavior is identical away from the wrap; at the wrap the C++
filter is continuous where the Python one glitches.
