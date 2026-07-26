# Sonar front-end review: CFAR and polar image processing (2026-07)

Scope: the detection front-end — `src/core/cfar.cpp`, `include/sonar_slam_cpp/cfar.hpp`,
`src/cuda/cfar_cuda.cu`, and the polar→Cartesian→cloud path in
`src/nodes/feature_extraction_node.cpp`. The back-end (pose graph, loop
closure, verification) is covered by `SLAM_EFFECTIVENESS_AUDIT.md` and
`LITERATURE_REVIEW_2026.md`; `RESEARCH.md` §2 already carries the provenance
of the *detector variants*. This document covers what those do not: whether the
detector's assumptions hold on real sonar, what its shipped operating point
actually does, and what the image-processing chain around it costs.

Evidence tags:
- **[MEASURED]** — measured on this repo's own code in this sandbox; method
  given so it can be re-run.
- **[SOURCED]** — from named literature, read at least at abstract level.
- **[DERIVED]** — follows from repo code plus arithmetic, no new measurement.

Measurement harness: a standalone program linking `sonar_slam_core`, driving
`CFAR::detect()` directly and replicating `generate_map_xy()`'s map geometry
exactly. 716×512 and 300×256 polar frames. It is not in-tree (see §6).

---

## 1. The detector implements its own model correctly [MEASURED]

On synthetic **exponential** intensity clutter — square-law detected Rayleigh,
which is precisely the model `docs/MATH_NOTES.md` §1 derives the threshold
factors under — the realized per-pixel false-alarm fraction at the shipped
`Pfa: 0.1` is:

| variant | realized | nominal |
|---|---|---|
| CA | 0.1027 | 0.10 |
| SOCA | 0.1031 | 0.10 |
| GOCA | 0.1023 | 0.10 |
| OS (rank 10) | 0.1131 | 0.10 |

CA/SOCA/GOCA track nominal to within 3%. This is an end-to-end check of the
threshold-factor solving, the detector kernels, and the CFAR property itself,
and it passes. **No defect found in the detector.**

The OS outlier is explained in §3.

## 2. The shipped operating point is not what the config appears to say

### 2.1 `Pfa: 0.1` is 5–50× looser than sonar CFAR practice [SOURCED] [DERIVED]

A per-pixel false-alarm probability of 0.1 means one detection in ten
background pixels. On a 716×512 Oculus frame that is **~32,000 false
detections per ping** before any other stage. Published sonar CFAR work
operates far tighter: the SOW-CFAR lake trials
([Zhang et al., *J. Mar. Sci. Eng.* 13(4):819, 2025](https://www.mdpi.com/2077-1312/13/4/819))
use `Pfa = 2%` as their nominal, and radar practice is conventionally 1e-3 and
below.

This is not automatically wrong — this front-end is not doing target
*declaration*, it is producing a dense point cloud for ICP, and a permissive
detector followed by geometric filtering is a defensible design. But it does
mean the CFAR stage is not the thing rejecting clutter, which matters for §2.2.

### 2.2 The fixed intensity gate destroys the CFAR property [MEASURED] [SOURCED]

`filter/threshold: 65` is ANDed into the detector (`cfar.cpp` folds it in so
the CPU and GPU twins agree). It is an **absolute, non-adaptive** threshold.
Measured pass rates on the same clutter, once homogeneous and once with a
range-decaying mean (a residual-TVG profile, mean 30→12):

| | CFAR alone | gate alone |
|---|---|---|
| homogeneous | 0.103 | 0.111 |
| range-decaying | 0.105 | **0.048** |

The CFAR stage holds its rate constant across the level change — that is its
entire purpose. The gate's pass rate drops by more than half. Since the two are
ANDed, the *combined* operating point moves with absolute signal level, i.e.
with gain setting, water conditions, TVG configuration and range. The stage is
therefore only nominally constant-false-alarm.

This is the textbook distinction: *"TVG differs from normalization in that it
is not data dependent and does not produce a CFAR"*
([ScienceDirect, Signal Detection](https://www.sciencedirect.com/topics/physics-and-astronomy/signal-detection)).
A fixed absolute gate is on the non-data-dependent side of that line.

**Recommendation.** Keep the gate — with `Pfa: 0.1` something has to carry the
load — but treat the pair as one operating point and tune them together against
a bag, not independently. If the gate is doing the real work, that is worth
knowing explicitly rather than by accident. The principled alternative is to
tighten `Pfa` (0.01–0.02, matching the sonar literature) and lower or remove the
absolute gate, which moves the decision back onto the adaptive statistic. That
is a behaviour change, so it belongs on a CHL_Pool replay, not in a default.

### 2.3 Clutter is heavier-tailed than the model [SOURCED]

Every variant here derives from an exponential (Rayleigh-envelope) background.
Measured forward-looking sonar echo data in shallow water is not that: **both
the K and generalized Pareto distributions fit better, with GP better in the
tail** ([Statistical Analyses of Measured Forward-Looking Sonar Echo Data in a
Shallow Water Environment, IEEE](https://ieeexplore.ieee.org/document/9827817/)),
and K-distribution is the standard model for sonar image background
([Abu & Diamant, *IET Radar Sonar Navig.*, 2020](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2020.0230)).
Because the mismatch is in the tail, the realized false-alarm rate exceeds
nominal: the SOW-CFAR authors measured **3.643% against a 2% nominal** on real
data, ~80% high, and attribute it to model mismatch in reverberation regions.

Consequence for us: §1's clean agreement is agreement with the *model*, not
with the sea. On real bags the realized rate should be expected to run high,
and the direction of the error is toward more false alarms, never fewer. This
argues for tightening `Pfa` rather than loosening it, and against trusting the
nominal number as a calibrated quantity.

## 3. `CFAR/rank: 10` is the wrong end of the order statistic [MEASURED] [SOURCED]

`RESEARCH.md` §2 flags this as an open question — the literature recommends
k ≈ 3N/4–4N/5 while we ship `rank: 10` of `Ntc: 40` (N/4). Rohling's
recommendation is k ≈ 0.75N. Measured on homogeneous exponential clutter:

| rank | τ | realized Pfa (nominal 0.10) |
|---|---|---|
| 10 (0.25N) | 8.089 | 0.1131 |
| 20 (0.50N) | 3.338 | 0.1040 |
| 30 (0.75N) | 1.660 | **0.1004** |
| 32 (0.80N) | 1.427 | 0.1005 |

Rohling's 3N/4 tracks nominal to 0.4%; the shipped N/4 runs **13% high**. The
mechanism is quantization, and it is specific to our data type: at rank 10 the
reference statistic is the 11th-smallest of 40 **uint8** samples — a small
integer, typically single digits against a mean of 30 — so `tau * train[rank]`
inherits a large relative quantization step, and the large τ (8.089) multiplies
it. At rank 30 the statistic sits near the distribution's bulk and the relative
step is small.

**Recommendation.** `rank: 30` when `alg: OS` is selected. This is a
config-only change and it moves both the literature agreement and the measured
agreement in the same direction. It does not affect the shipped default
(`alg: 'SOCA'`), so it is low-risk. Note the *derivation* is already correct
here — `pfa_os()` was fixed to model the (rank+1)-th smallest and validated by
Monte Carlo (`MATH_NOTES.md`, `DIVERGENCES.md`); this is about which rank to
ask for, not about the formula.

## 4. The window is 1-D in range; sonar practice is 2-D [DERIVED] [SOURCED]

`detect_cpu` slides along rows (range) within each beam column
(`cfar.cpp`: "the window slides along the range axis (rows) within each beam
column"), so no training cell ever comes from a neighbouring beam. Sonar
imagery CFAR generally uses a two-dimensional window, "consider[ing] the noise
characteristics in the vertical range and horizontal azimuth dimensions
simultaneously" ([Zhang et al. 2025](https://www.mdpi.com/2077-1312/13/4/819)).

Practical read: 1-D range-only is the cheaper and more parallel choice, and it
is what bruce_slam did, so this is a *parity* decision rather than a defect.
Its real cost is that azimuthal structure — beam-to-beam gain variation, a
cross-beam reverberation ridge — is invisible to the estimator, so it lands in
the detection statistic instead of being normalized out. Worth revisiting only
if bag replay shows beam-correlated false alarms. Not recommended now.

## 5. Detection is extracted *after* Cartesian resampling [MEASURED]

`feature_extraction_node.cpp` runs CFAR on the polar image, then remaps the
**binary mask** to Cartesian with nearest-neighbour interpolation, then calls
`findNonZero` and converts pixel indices to metres. Detections are therefore
sampled onto the Cartesian grid before they become points.

Nearest-neighbour resampling is a *pull*: a polar cell survives only if some
destination pixel's nearest source is that cell. Replicating `generate_map_xy`
exactly and counting reachable polar cells:

**Revolution sim preset** (res 0.083 m, 300 bins × 256 beams, 130°):

| range band | polar cells | reachable | lost |
|---|---|---|---|
| 0–5 m | 15360 | 4008 | **73.9%** |
| 5–10 m | 15360 | 11478 | 25.3% |
| 10–15 m | 15360 | 14942 | 2.7% |
| 15–25 m | 30720 | 30696 | ~0% |

**Oculus M750d-like** (res 0.03 m, 716 bins × 512 beams): 68.7% lost inside
4.3 m, 16.3% in 4.3–8.6 m, ~0 beyond 8.6 m.

The crossover is where beam arc spacing equals the Cartesian cell — 9.37 m and
6.76 m respectively. Inside it the sonar's angular resolution is finer than the
grid it is being resampled onto, so cells compete for destination pixels and
lose; outside it one polar cell wins several pixels (duplication, which the
voxel downsample then collapses harmlessly).

**How much does this actually matter?** Less than those percentages suggest,
and I checked rather than assumed. Reachability is deterministic, so the
longest run of consecutive unreachable beams bounds the widest compact target
that can be erased outright:

| range (Revolution) | reachable | max erasable target |
|---|---|---|
| 0–3.1 m | 16.0% | whole rows (apex) |
| 3.1–6.2 m | 48.8% | 0.110 m across |
| 6.2–9.3 m | 78.0% | 0.111 m across |
| >9.3 m | 94–100% | ≤ 1 beam |

So the strong version of the concern — that pool posts or thin structure
vanish — **is not supported**: anything wider than ~11 cm survives beyond ~3 m,
and `filter/resolution: 0.5` voxel-downsamples the cloud far more aggressively
than this anyway. The honest conclusion is narrower:

- Below ~3 m the remap discards most of the angular resolution and can erase
  compact returns entirely. For a pool deployment that band is not empty.
- Extracted points are snapped to the Cartesian grid, adding up to ~res/√2
  position error (0.06 m Oculus, 0.06 m Revolution) — small against the 0.5 m
  voxel, but it is pure loss for nothing.
- The ordering blocks any future move to finer clouds: lowering
  `filter/resolution` cannot recover detail the remap already discarded.

**Recommendation.** Extract in polar and convert exactly:
`findNonZero` on the polar mask, then `range = (row + 0.5) * res + range_min`,
`bearing = bearings[col]`, then `x = range·cos(bearing)`, `y = range·sin(bearing)`.
This is *cheaper* than the current path (no mask remap at all — the Cartesian
remap stays only for the `feature_img` visualization, which is already
subscription-gated) and it is exact. Deferred here rather than done because it
changes the shipped cloud on every platform and so needs a bag replay against
`map_metrics`, not a green e2e test.

## 6. The e2e fixture never stresses the detector [MEASURED]

`SyntheticWorld::polar_image` draws background from **N(30, 6)** and wall
returns from N(190, 15). Measured detection fractions on that fixture versus on
exponential clutter, same detector, same config:

| background | CFAR alone | gate alone | both |
|---|---|---|---|
| repo fixture (Gaussian σ=6) | 0.0030 | 0.0030 | 0.0030 |
| exponential (realistic) | 0.103 | 0.111 | 0.088 |

The fixture understates the false-alarm load by **~30×**. A tight Gaussian
background has no tail, so `cell > τ · mean_train` essentially never fires on
noise, and the fixture's `[1] CFAR: 119/128 wall beams, 0 stray detections`
line is measuring a regime the sensor never operates in. That result is not
wrong — it is a valid *geometric* check that the wall is found — it just
carries no information about clutter rejection, which is what a CFAR test
should assert.

This also explains why §2's operating-point concerns have never shown up in
CI: nothing in-tree exercises them.

**Recommendation.** Give `polar_image` an exponential background option and add
a detector stage asserting realized false-alarm rate against nominal on pure
clutter. That is a genuinely new test rather than a tightened one, and it is
the natural home for the §1 measurement so it stops being a one-off. The
harness used for this document should become that test rather than living in a
scratch directory.

---

## Summary: what to change, in order

| # | Change | Status |
|---|---|---|
| 1 | `CFAR/rank: 10 → 30` for `alg: OS` | **DONE** — `feature.yaml`; inert for the SOCA default |
| 2 | Realized-P_FA assertion on uint8 clutter (§6) | **DONE** — `cfar_math_test` stage [5] |
| 3 | Extract in polar, drop the mask remap (§5) | **DONE** — `filter/extract_polar: true`, legacy path retained |
| 4 | Tune `Pfa` / `filter.threshold` as one operating point (§2.2) | **OPEN** — needs a bag; documented in `feature.yaml` |
| 5 | 2-D range-azimuth window (§4) | **NOT PLANNED** — no evidence of need |

Nothing here is a defect in the detector itself (§1). The findings are about
operating point, an inherited processing order, and a test fixture that cannot
see any of it.

### What was implemented, and how it was verified (2026-07-26)

- **#1** `feature.yaml` ships `rank: 30`. Only `alg: OS` reads it, and the
  default is `SOCA`, so this changes no shipped behaviour — it fixes the value
  for anyone who selects OS.
- **#2** `cfar_math_test` stage [5] drives `CFAR::detect()` on uint8
  exponential clutter and asserts realized P_FA against nominal for
  CA/SOCA/GOCA (all within 0.01), asserts the shipped OS rank holds nominal,
  and asserts 3N/4 stays closer to nominal than N/4 so the finding cannot
  silently invert. It reproduces this document's numbers independently:
  CA 0.1023, SOCA 0.1031, GOCA 0.1020, OS rank 10 **0.1140**, OS rank 30
  **0.1012**. Note stage [2]'s existing Monte Carlo — continuous doubles —
  reports OS at 0.0999, which is exactly why the quantization effect was
  invisible before: it is not a property of the formula, only of uint8 input.
- **#3** implemented in `feature_extraction_node.cpp` behind
  `filter/extract_polar` (default true). The legacy Cartesian path is kept
  intact for a one-line revert. Conventions were verified numerically rather
  than by inspection, because a sign or axis error here would mirror the map:
  a harness replicating `generate_map_xy` and `cv::remap(..., INTER_NEAREST)`
  ran both conversions over 35 probe cells spanning the fan — **35/35 agree**,
  worst delta 0.032 m against a 0.030 m cell, i.e. exactly the one-cell grid
  snap the polar path removes and nothing more.

The §6 recommendation to give `polar_image` an exponential-background option
was **not** taken. The detector assertion belongs in the detector test, where
it is self-contained; switching the *pipeline* fixture's background to
exponential would put ~10% false alarms into the e2e cloud and require
re-tuning `Pfa`/`filter.threshold` inside the test — which is open item #4,
and needs a bag rather than a guess. The e2e fixture therefore stays Gaussian
deliberately, and §6's caveat about what its `0 stray detections` line means
still stands.

## Sources

- [Zhang et al., "A Constant False Alarm Rate Detection Method for Sonar Imagery Targets Based on Segmented Ordered Weighting," *J. Mar. Sci. Eng.* 13(4):819, 2025](https://www.mdpi.com/2077-1312/13/4/819)
- [Abu & Diamant, "CFAR detection algorithm for objects in sonar images," *IET Radar, Sonar & Navigation*, 2020](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2020.0230)
- [Statistical Analyses of Measured Forward-Looking Sonar Echo Data in a Shallow Water Environment, IEEE, 2022](https://ieeexplore.ieee.org/document/9827817/)
- [Rohling rank guidance, Purdue ECE OS-CFAR notes](https://engineering.purdue.edu/~mrb/resources/AltLectureF/Session_22.pdf)
- [Signal Detection overview — normalization vs TVG and the CFAR property, ScienceDirect](https://www.sciencedirect.com/topics/physics-and-astronomy/signal-detection)
- [Westman & Kaess, "Degeneracy-aware Imaging Sonar SLAM," IEEE JOE](https://www.cs.cmu.edu/~kaess/pub/Westman20joe.pdf) — context for the FLS elevation-ambiguity constraint this front-end works under
- Existing in-repo coverage: `RESEARCH.md` §2 (variant provenance: Rohling 1983, Gandhi–Kassam 1988, VI-CFAR, Mu et al. 2024), `MATH_NOTES.md` (threshold derivations), `DIVERGENCES.md` (the `pfa_os` correction)
