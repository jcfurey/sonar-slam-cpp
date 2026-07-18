# Why SLAM never produced a helpful correction — systemic audit

**Date:** 2026-07-18
**Trigger:** no meaningful/helpful SLAM correction observed on any bag tested,
despite the loop-closure defense stack, the map-doubling fixes, and healthy
sensor streams. This audit takes the system apart end-to-end instead of
hunting more per-line bugs. Two independent failure families were found; both
are now addressed in code, and §4 gives the replay protocol that verifies the
fix and discriminates what remains.

---

## 1. Family A — the correction path was structurally muted

Each layer below was individually reasonable (most were responses to real
field disasters), but stacked they made accepting a *meaningful* correction
nearly impossible. Numbers use the shipped configs.

### A1. DCS on loop factors suppressed exactly the corrections that matter
`gtsam::noiseModel::mEstimator::DCS(φ=1.0)` scales a factor's weight by
`2φ/(φ+χ²)`. A **genuine** closure correcting drift `D` with ICP sigma `σ`
starts at `χ² = (D/σ)²` — large *by design*; that is what a correction is.
With the sampled/Censi covariances of a confident lock (σ ≈ 3–10 cm):

| drift D | σ = 5 cm → χ² | DCS weight |
|---|---|---|
| 0.25 m | 25 | 7.7 % |
| 0.5 m | 100 | 2.0 % |
| 1.0 m | 400 | **0.5 %** |

Corrections small enough to keep full weight (≲ σ) are invisible; corrections
worth having are suppressed 10–1000×. The optimizer moved millimetres,
`map→odom` never meaningfully changed, and every acceptance gate happily
reported "accepted".
**Change:** DCS is now opt-in (`nssm/use_dcs`, default **false**;
`nssm/dcs_phi`). Outlier damage-limitation is the optimize-then-verify
revert + quarantine, which judges the *actual* optimized graph rather than
pre-shrinking every correction.

### A2. ISAM2 defaults under-converged whatever survived
`ISAM2()` defaults are batch-throughput settings: `relinearizeThreshold 0.1`,
`relinearizeSkip 10`, one Gauss-Newton pass per `update()`. A metre-scale
correction "staircases" over many updates — and worse:

### A3. …and the post-loop verification judged the transient
The yaw-RMS / chain-tear checks run immediately after the single update. An
under-converged genuine correction looks exactly like a localized fold
(stretched links mid-chain), so verification **reverted** it — and since the
quarantine change, permanently. Genuine closures could be executed, judged on
their transient, reverted, and blacklisted.
**Change:** `relinearizeThreshold 0.01`, `relinearizeSkip 1` (both ISAM2
construction sites), plus `loop_extra_iterations` (default 3) extra
`isam_.update()` passes on loop rounds **before** the verification runs.

### A4. Context: the acceptance funnel was already narrow
Pre-3a, the yaw alias basin + compass gate rejected 56 % of candidates (~4
accepted/run). 3a (compass-clamped init yaw) fixes the front of the funnel;
A1–A3 fixed the back. Watch both in the status line.

## 2. Family B — time-domain inconsistencies (full inventory in this commit)

The stack has two legitimate clock domains — message stamps (driver/bag) and
the node clock — and mixed them at exactly the wrong places. Complete
inventory ran over every `now()`/stamp/timer usage; the material items:

1. **`use_sim_time` was not plumbed at all.** No launch arg, no param, so bag
   replay ran every node clock on wall time while all data/TF sat at bag
   time. The head-tilt interpolation *never* ran on replay (its 5 s domain
   guard tripped on every lookup, falling back to the stale latch — the exact
   mechanism behind admitted floor-bowl frames), and the republished map
   cloud was restamped to wall time, which TF-timed consumers reject.
   **Fixed:** `use_sim_time` launch arg applied to every node; republish
   keeps the data-domain stamp; README documents `ros2 bag play --clock`.
2. **Cross-device stamp offsets had no remedy and no symptom.** Sonar, DVL,
   IMU, gyro, depth all stamp from independent clocks; every synchronizer
   pairs on those stamps with slop 0.1 s (DR) / 0.5 s (slam). A constant
   offset δ biases *every* keyframe pose by v·δ and ω·δ (δ = 0.1 s at 0.5 m/s,
   0.3 rad/s → 5 cm + 1.7°; 1.7° of heading is 0.6 m of lateral feature
   displacement at 20 m range — bigger than the 0.5 m ICP resolution, every
   frame). Past the slop, primaries were popped **with no callback and no log
   line**: streams alive, zero pairs, SLAM silently mute.
   **Fixed:** per-sensor `<sensor>.stamp_offset` parameters at the adapter
   choke point; ApproxSync `set_nomatch_callback` wired to throttled ERRORs in
   dead_reckoning and slam_node that name the offset remedy; backward-jump
   (bag-loop) queue resets.
3. **Nominal-rate integration.** kalman advanced x/y by `v·dt_imu(nominal)`
   per *message* (a 1 % rate mismatch or best-effort drop rate = 1 % scale
   error ≈ 18 m/h at 0.5 m/s); gyro's earth-rate term likewise. **Fixed:**
   measured stamp dt (clamped, re-anchoring on gaps/negative); gyro warns
   when sample gaps imply lost delta angles.
4. **Bag-loop robustness:** `is_keyframe` returned false for an entire looped
   pass after the time jump; viz/republish throttles suppressed for a pass;
   sync queues blocked behind stale large stamps. All now reset on negative
   elapsed time.

## 3. What was NOT wrong (checked and cleared)

- The per-keyframe DR unaries are wide (σ = 10 m/rad on x/y/yaw): rubber-band
  effect on a 1 m correction over 50 keyframes is χ² ≈ 0.5 — negligible.
  Depth/attitude pinning is appropriate for the horizon chart.
- The sampled-covariance un-rotation, PCM clique machinery (after the earlier
  frame fix), and the SSM/NSSM candidate plumbing are sound.
- DR itself is time-consistent (stamp-dt integration, gap re-anchoring) — the
  best time code in the stack, and the reason "SLAM ≈ DR" looked acceptable.

## 4. Replay protocol — verify the fix, discriminate what remains

1. Rebuild `sonar_slam_cpp`; replay CHL_Pool **with** `--clock` +
   `use_sim_time:=true` (now mandatory).
2. Watch startup for the new no-match ERRORs. If they fire, measure the
   stream offset (median stamp delta) and set `<sensor>.stamp_offset`;
   re-run. *Do not tune anything else until pairing is clean.*
3. In the status line: `NSSM accepted` should rise (3a) AND `map→odom` should
   now actually move when a closure lands (A1–A3). `REVERTED` entries should
   be rare and cite real folds.
4. Metrics vs MAP_DOUBLING_FIX_PLAN §5: doubled-wall fraction from 23 % down;
   local wall thickness ≤ 10 cm (no new smear).
5. If corrections now land but maps still double: the remaining suspects are
   registration quality itself (feature repeatability across passes) and the
   SSM fixed-noise chain (`ssm/cov_samples: 0` still ships fixed 10 cm
   confidence — consider 30 with the parallel pool, as the deployed config
   notes already suggest).

## 5. Residual risks / follow-ups

- Removing DCS raises the stakes on the verify layer: watch the first replays
  for reverts citing real folds (that is the layer doing DCS's old job).
- `loop_extra_iterations` trades callback latency for convergence on loop
  rounds only (~3 extra linear solves on a ≤1000-node graph; negligible).
- kalman's `A_imu` still carries the *configured* nominal dt inside the
  transition matrix (config data, structure unknown to the node); only the
  position integration uses measured dt. Second-order; documented here.
- If cross-device offsets turn out to DRIFT (not constant), stamp_offset is
  insufficient — that would need arrival-time regression per stream. Measure
  first.
