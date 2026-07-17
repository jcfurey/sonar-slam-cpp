# SLAM Map-Doubling Fix Plan

**Date:** 2026-07-17
**Bag:** `bags/2026-04-16_Callisto_CHL_Pool_19-20-45_UTC_corrected_v2` (CHL_Pool, `robot_slam_venue=pool`)
**Goal:** eliminate the ~1 m wall **doubling** in the graph-anchored map so the corrected map has single, crisp walls.

---

## 1. Problem statement

The assembled/SLAM map is **locally sharp but globally doubled**: the same physical
wall appears as two crisp lines ~1 m apart.

Measured on the final corrected frame (`/tmp/analyze_map.py`, `/tmp/analyze_double.py`):

| Cloud | local wall thickness | doubled-wall fraction | doubling gap |
|---|---|---|---|
| `/bruce/slam/slam/cloud` (n=133) | 8.3 cm (sharp) | **23%** | median 105 cm, max 515 cm |
| `~/survey_pointcloud` (n=13*) | 11 cm (sharp) | 85%* | median 120 cm |

\*small clean-line sample; trust the slam-cloud 23%. Both agree doubling is present and material.

Sharp lines + ~1 m separation ⇒ **not blur, but two offset passes** of the same wall.

---

## 2. Root cause (established by elimination)

| Hypothesis | Test | Verdict |
|---|---|---|
| Loop closure (NSSM) destabilizes the graph | A/B: `nssm/enable` on vs off | ❌ on ≈ off (map→odom ±2.5 m either way) |
| Feature dropout drives the instability | corr(feature-drop, map→odom motion) = **−0.215**; biggest snaps have healthy features | ❌ not the trigger |
| Head-pitch gate misfiring on a bad signal | NaN-sentinel timing: 10 sustained bands, 0 singletons | ❌ genuine head sweeps; gate is correct by design (`feature_extraction_node.cpp:375-376`) |
| map→odom "oscillation" is the defect | map→odom dips near the origin revisit, grows far from it | ⚠️ largely expected geometry, not the defect |
| **Loop closure is ineffective → revisit drift uncorrected** | doubling = crisp offset passes; SSM is sequential and cannot close a revisit gap | ✅ **root cause** |

**Why NSSM is ineffective:** on the near-square pool the loop-closure ICP aliases at
~90°. The rejection histogram (added to the status log this session) shows **56% of
candidates compass-rejected** (`Large transformation (compass)`), leaving only ~4
accepted closures over ~100 keyframes — far too few to close the revisit drift, so the
~1 m gap between passes is never corrected → doubling.

**Aliasing source:** the NSSM Sobol global-init searches yaw over `±5·rotation_std`
(`src/packages/localization/sonar_slam_cpp/src/core/slam_core.cpp:683-686`). When a
keyframe's marginal yaw uncertainty is loose, that window spans the ~90° alias basin,
so the search lands on a rotated-but-plausible match that the compass gate then kills.

---

## 3. Fix

### 3a. Compass-clamped NSSM init yaw (primary)

**File:** `src/packages/localization/sonar_slam_cpp/src/core/slam_core.cpp`, in
`initialize_nonsequential_scan_matching`, the bounds block at ~683-686.

Change the rotation bounds from `±5·rotation_std` to
`±min(5·rotation_std, nssm_max_yaw_vs_compass)` (~0.15 rad). Keep translation bounds
(`±5·translation_std`) unchanged.

Rationale: both keyframes' DR yaws are compass-anchored and drift-free (the same
premise the compass gate at `:783-795` relies on), so the true closure yaw is always
within the compass band. Narrowing the search keeps full translation freedom but
prevents the Sobol search from ever entering the 90° alias basin.

Effect: compass-*rejected* candidates become compass-*consistent accepted* closures →
revisit passes pull together → doubling drops.

### 3b. `is_keyframe` min-point gate (supporting, cheap)

**File:** `slam_core.cpp:307-319` (`is_keyframe`).

Add a minimum point-count condition so a near-empty (0-row) feature cloud does not
become a 0-pt keyframe. NaN-sentinel frames already become non-keyframes; genuinely
empty clouds (~8% of pings) currently slip through and dilute the graph. Odometry
still carries the pose across the gap.

### 3c. Head pitch — no code change

The 26% head-pitch drops are genuine sweeps of a floor/surface-dominated fan; the gate
is correct (`feature_extraction_node.cpp:375-383`) and the 2D registration points are
not pitch-compensated, so **raising `max_head_pitch` would admit corrupting frames**.
Mitigation is operational: keep the camera head level while surveying.

---

## 4. Safeties (why 3a is safe despite the 90°-corruption history)

The change *narrows* the search (makes aliasing less likely); it does not touch any
acceptance gate. All existing guards remain:

- Compass-consistency gate (`slam_core.cpp:783-795`)
- PCM clique consistency (`min_pcm`, `pcm_queue_size`)
- Degeneracy gate (`max_sigma`, `max_anisotropy`)
- DCS robust kernel on loop factors (`:851-855` / `:868`)
- Post-loop revert (`post_loop_max_yaw_rms`, `post_loop_max_translation_err`)

---

## 5. Build & test

1. Rebuild package: **`sonar_slam_cpp`** (user builds; do not run colcon here).
2. Replay `robot_slam_venue=pool` on the CHL_Pool bag.
3. Record: `ros2 bag record -o /tmp/map_quality_fix /bruce/slam/slam/cloud ~/mapping/map_assembler/survey_pointcloud /tf /tf_static`
4. Also capture the `SLAM status:` line for the `NSSM accepted` count and the
   `NSSM rejects [...]` histogram (compass bar should shrink, accepted should rise).

### Metrics (before → target)

| Metric | Before | Target |
|---|---|---|
| doubled-wall fraction (slam cloud, `analyze_double.py`) | 23% | well below 23% |
| NSSM accepted (whole run) | ~4 | materially higher |
| `Large transformation (compass)` share of rejects | 56% | much lower |
| local wall thickness | 8 cm | stay ≲ 10 cm (no new smear) |

Expectation: doubling drops **materially but not to zero** — head-pitch DR-only gaps
and residual SSM drift still contribute.

---

## 6. Analysis helpers (in `/tmp`, run inside `nautilus-robot-gpu-1`, source ROS)

- `analyze_corr.py` — map→odom stability + trajectory retroactive correction
- `analyze_feat.py` / `analyze_feat2.py` — feature-count timeline; feature-drop vs map→odom correlation
- `analyze_tilt.py` — NaN-sentinel (head-pitch) temporal clustering
- `analyze_map.py` — final-cloud extent, occupancy, local wall thickness
- `analyze_double.py` — wall-doubling probe at the 0.3–3 m scale

---

## 7. Status / changes already made this session

- `settings/params/mapping/map_assembler.yaml`: `survey_occupancy_mask: false` (+ prob tweaks) — fixed survey sparseness (confirmed good).
- `settings/params/localization/sonar_slam/slam.yaml`: `nssm/max_rotation 0.10 → 0.20` (safe, benign; loop-closure tuning is not the map lever).
- `slam_node.cpp` / `slam_core.{hpp,cpp}`: added the **NSSM rejection histogram** to the status log.
- `nssm/enable` toggled off for the A/B, then **reverted to true**.

## 8. Open / follow-up

- If 3a under-delivers, revisit SSM constraint weighting (sampled ICP covariance vs
  `odom_sigmas`/`icp_odom_sigmas`) as a secondary lever.
- Consider gating NSSM candidate acceptance on keyframe density during head-pitch gaps.
