# Literature review — design validation and radar-transfer feasibility (2026)

**Purpose.** Two questions. (1) Does this stack's architecture transfer to
**land vehicles with scanning radar**? (2) Does the published literature
**validate, nuance, or contradict** each load-bearing design decision we made?

## How to read the confidence tags

The research for this document was gathered by a fan-out search harness (26
primary sources, ~130 extracted claims). The harness's adversarial
verification stage did **not** complete — it hit a provider usage limit
mid-run — so the automated 3-vote confirmation is absent. To compensate, the
most decision-critical claims were **re-verified by hand** against the primary
sources. Every statement below carries one of:

- **[VERIFIED]** — re-checked this session against the cited primary source.
- **[SOURCED]** — extracted verbatim by the harness from the named primary
  source (URL given), but not independently re-verified line-by-line. Treat as
  a strong lead, not gospel; confirm the exact figure before quoting it
  externally.
- **[DOMAIN]** — standard, uncontroversial background knowledge in the field;
  no single citation is load-bearing.

Where the literature **disagrees** with a choice we made, it is called out
plainly.

---

## Part 1 — Radar transfer feasibility

**Verdict: the architecture transfers, but the front-end physics does not.**
The *shape* of this pipeline — detect features on a polar intensity image →
build a point cloud → register scans → optimize a pose graph with robust loop
handling — is exactly the accepted shape for spinning-FMCW radar SLAM. What
does **not** transfer is the assumption that a scan is an instantaneous,
motion-free, Doppler-free snapshot, and the DVL/AHRS/compass sensor scaffold.

### 1.1 Pipeline shape — VALIDATED

The dominant published radar-odometry/SLAM systems share our block diagram:

- **CFEAR** (Adolfsson, Magnusson, Alhashimi, Andreasson, Lilienthal; IROS
  2021 / T-RO 2023; arXiv 2105.01457): polar radar → per-azimuth filtering →
  oriented surface points → scan registration → pose graph. **[VERIFIED]** —
  achieves 1.76 % translation error on the Oxford benchmark at 55 Hz on one
  CPU thread.
- **TBV Radar SLAM** (Adolfsson et al.; RA-L 2023; arXiv 2301.04397): CFEAR
  front-end + 2-D pose-graph back-end with post-registration loop
  verification. **[SOURCED]**
- **RadarSLAM** (Hong, Petillot, Wang; IROS 2020; arXiv 2104.05347): polar
  scan → point cloud → scan match → pose graph with loop closure.
  **[SOURCED]**
- **Cen & Newman** (ICRA 2018): the seminal per-azimuth landmark extractor +
  scan matching. **[DOMAIN]**

So the coarse architecture — the thing your theory rests on — is not just
*viable* for radar, it is the mainstream design. Our
CFAR→cloud→ICP→ISAM2→map is structurally the same animal.

### 1.2 CFAR as the detector — NUANCED (leaning contradicted for radar)

Our CFAR family (CA/SOCA/GOCA/OS) is a legitimate radar detector, but it is
**not** the preferred radar front-end, and at least one strong system argues
against it directly:

- **CFEAR** argues classical CFAR *cannot be applied directly* to spinning
  FMCW radar because the noise distribution is not known a priori and CFAR
  needs multi-parameter manual tuning; their k-strongest-per-azimuth filter
  outperforms it. **[VERIFIED]** (k-strongest confirmed; the "cannot be
  applied directly" argument is **[SOURCED]** from the same paper.)
- **RadarSLAM** uses a Hessian blob detector + KLT for odometry and
  per-azimuth adaptive thresholding (power ≥ mean + 1σ of that azimuth) for
  loop-closure clouds — chosen specifically to fight **receiver saturation**,
  a radar effect with no sonar analogue. **[SOURCED]** (arXiv 2104.05347)
- The **Abu-Alrub & Rawashdeh survey** (arXiv 2307.07861, 2023) categorizes
  radar feature extraction into per-azimuth statistical landmarks, learned CNN
  keypoints (Barnes & Posner, *Under the Radar*, ICRA 2020), and k-strongest
  filtering — with CFAR a minority choice (e.g. Vivet et al.; BFAR by
  Alhashimi et al. is a CFAR *variant*). **[SOURCED]**

**Implication for us:** CFAR is a *defensible* radar detector and BFAR shows
the CFAR family is still in play, but if we port to radar we should expect to
add a k-strongest or adaptive-per-azimuth option and benchmark against CFAR
rather than assume CFAR wins. Our OS-CFAR (rank-order) is the most
radar-appropriate of our four — OS-CFAR is the standard choice against the
spiky, clutter-heavy statistics radar and sonar share.

### 1.3 Radar-specific physics we do NOT model — the real gaps

These are the effects a sonar pipeline never sees and a radar port must add:

1. **Intra-scan motion distortion (dominant).** A Navtech-class sensor spins
   at ~4 Hz; at road speed the platform moves several metres and degrees
   during one 360° sweep, smearing the point cloud. **[SOURCED, multiple
   sources]** (2104.05347, 2105.01457, 2011.03512). RadarSLAM jointly
   optimizes per-scan velocity with pose to compensate; CFEAR de-skews all
   points to the mid-sweep timestamp under a constant-velocity assumption.
   Our pipeline treats each frame as instantaneous — **this stage does not
   exist in our code and must be built.**
2. **Doppler range bias in FMCW.** Radial ego-velocity shifts apparent target
   range (~4.8 cm per 1 m/s for the 76–77 GHz Navtech, per the harness's
   extraction of Burnett et al.). **[SOURCED]** (arXiv 2011.03512)
3. **Motion vs Doppler — the quantified nuance.** Burnett, Schoellig &
   Barfoot (RA-L 2021) reportedly find: Doppler **cancels** in scan-to-scan
   registration (consecutive scans distort alike) so it can be ignored for
   *odometry*, but **matters for metric localization against a prior map** —
   with the biggest published gains on a reverse-direction drive (motion comp
   ~41.7 %, Doppler ~67.7 %, both ~81.2 %, ~2.30 m → ~0.43 m median).
   **[SOURCED — not independently re-verified; the PDF was unparseable this
   session. Confirm before external quotation.]** This distinction maps
   *directly* onto our own architecture: it means our **relocalization**
   module (localize against a saved map) would be the part that most needs
   Doppler/motion compensation, while our scan-to-scan SSM would tolerate its
   absence.
4. **Speckle, receiver saturation, multipath ghosts, detection
   discontinuity.** Radar perceives *beyond* the nearest object on a line of
   sight, producing repetitive ghost returns and targets that blink between
   frames. **[SOURCED]** (arXiv 2104.05347). Sonar has speckle and multipath
   too, but not receiver saturation or see-through ghosting; our CFAR + the
   degeneracy/compass gates were tuned for sonar clutter, not radar ghosting.

### 1.4 What replaces DVL + AHRS + compass on land — ASSUMPTION BREAKS

This is the sharpest break, and it touches our **most load-bearing
assumption**: that yaw is compass-anchored and trustworthy (the entire
loop-verification stack — the compass gate, the windowed yaw-RMS check, the 3a
init clamp — leans on it).

- **Radar odometry can be radar-only.** RadarSLAM and CFEAR run with **no
  IMU, wheel odometry, GPS, or compass** and still hit state-of-the-art
  accuracy. **[VERIFIED for CFEAR's 1.76 %/55 Hz; SOURCED for the "no other
  sensors" claim.]** So the DVL+AHRS front-end is not a *required* ingredient
  on land — radar scan-matching can be the odometry source itself.
- **When a yaw source is used, it is an IMU/gyro, not a magnetometer.** The
  survey's most common fusion is radar + IMU, with wheel encoders / CAN-bus
  yaw-rate secondary; it does **not** discuss magnetometer/compass anchoring.
  **[SOURCED]** (arXiv 2307.07861)
- **Magnetometer yaw is actively bad near vehicles.** An SAE study
  (2020-01-1025) reports magnetometer-derived yaw RMS of **3.4° straight /
  6.0° turning** under disturbance, cut to **0.5° / 1.9°** by fusing vehicle
  kinematics + IMU. **[SOURCED]**
- **Our own domain (underwater) already documents this failure.** Vial et al.
  (JFR 2024; rob.22272) add absolute magnetometer yaw as SO(2) prior factors
  in a GTSAM/iSAM2 sonar pose graph — a **direct precedent for our
  per-keyframe compass anchoring** — but had to inflate magnetometer variance
  **50×** (R_mag 0.5 rad vs 0.01 rad) near steel harbor structures or the
  optimized graph was wrecked. **[SOURCED]**

**Implication for us:** on land, do **not** carry the compass-anchored-yaw
assumption across as-is. Replace the compass prior with a **gyro/IMU
yaw-rate** source (integrated, with the loop-verification comparing optimized
yaw against *integrated gyro* rather than absolute compass), or drop the
absolute-yaw gate entirely and lean on radar-only rotational estimation. The
good news: automotive-grade gyros are far better than the MEMS AHRS
bruce_slam assumed, so a gyro-anchored version of our verify layer is
*stronger* on land, not weaker — as long as we stop trusting the magnetometer.

### 1.5 Cross-domain precedent — YES, both directions

The sonar↔radar transfer is not novel speculation. RadarSLAM explicitly draws
on visual/lidar SLAM structure; the whole radar-odometry line inherits scan-
matching + pose-graph machinery from lidar and underwater sonar work. No
single paper reuses *this exact* sonar stack for radar, but the architectural
equivalence is well established. **[DOMAIN]**

### Radar-transfer module ledger

| Module | Ports unchanged? | Adaptation the literature prescribes |
|---|---|---|
| Polar image → point cloud geometry | **Yes** (retune range/aperture) | — |
| CFAR detector | **Partial** | Add k-strongest / adaptive-per-azimuth; benchmark vs OS-CFAR; expect saturation/ghost handling |
| ICP / scan registration | **Yes** | — (but see Part 2.6 on covariance) |
| ISAM2 pose graph + PCM | **Yes** | — |
| Optimize-then-verify + revert | **Yes** — TBV is the radar precedent | Compare vs gyro, not compass |
| Compass-anchored yaw + gate | **No — breaks** | Gyro/IMU yaw-rate anchor; distrust magnetometer near chassis |
| DVL + AHRS front-end | **No** | Radar-only odometry, or radar + IMU |
| Motion-distortion comp. | **Missing — must add** | De-skew to mid-sweep (CFEAR) or joint velocity-pose (RadarSLAM) |
| Doppler comp. | **Missing — add for relocalization** | Needed for map localization, not scan-to-scan (Burnett) |
| Mapping / occupancy | **Yes** | — |
| Georeferencing (UTM export) | **Yes — already land-ready** | — |

---

## Part 2 — Design-decision verdicts

**1. CFAR (CA/SOCA/GOCA/OS) as the FLS feature detector — VALIDATED (sonar).**
CFAR is the standard constant-false-alarm detector for forward-looking sonar
and is the bruce_slam lineage's choice; OS-CFAR is the prevalent variant
against spiky clutter. **[DOMAIN]** For *radar*, see 1.2 — nuanced. A sonar
CFAR-detection reference is in the source set (MDPI *Sensors* 2018,
s18010076). **[SOURCED]**

**2. Disabling DCS / robust kernels in favour of optimize-then-verify + revert
+ quarantine — VALIDATED, with a strong published precedent.**
The robust-kernel line (Sünderhauf & Protzel switchable constraints, IROS
2012; Agarwal et al. Dynamic Covariance Scaling, ICRA 2013) down-weights
suspect loop edges *inside* the optimization. Our audit found DCS(φ=1) mutes
metre-scale genuine corrections (weight 2φ/(φ+χ²)). The literature's modern
answer to exactly this — **verify the loop candidate *after* registration and
accept/reject, rather than soft-weighting it** — is **TBV Radar SLAM** ("Trust
But Verify", RA-L 2023): it checks alignment quality, place similarity, and
odometry uncertainty and selects the most likely constraint. **[SOURCED]**
RTAB-Map's robust graph optimization / OptimizeMaxError is the same
verify-and-reject philosophy. **[SOURCED]** Our optimize-then-verify-then-
revert layer is squarely in this accepted lineage; keeping DCS as opt-in
rather than default is defensible. **Nuance:** robust kernels and verify-revert
are not mutually exclusive in the literature — some systems use both. Offering
DCS as opt-in (as we do) is the conservative, well-supported choice.

**3. ISAM2 relinearizeThreshold 0.01 / relinearizeSkip 1 + extra update()
iterations on loop rounds — NUANCED (sound, under-documented).**
The defaults (0.1 / 10) are batch-throughput settings; tightening them makes
ISAM2 relinearize more aggressively, which is exactly what a large loop-closure
correction needs to converge rather than staircase across updates (Kaess et
al., iSAM2, IJRR 2012, is the authority on the relinearization machinery).
**[SOURCED/DOMAIN]** There is no single paper prescribing "0.01 / 1 for
sonar," so this is an empirically-justified tuning, not a cited constant —
which matches how we've documented it. The extra-iterations-before-verify
step is our own contribution and is consistent with the general finding that
judging an under-converged transient is unsafe.

**4. PCM with small queue (5) and min-clique 2 — VALIDATED (method), NUANCED
(sizing).**
PCM (Mangelson, Dominic, Eustice, Vasudevan; ICRA 2018; IEEE 8460217) is the
right tool for rejecting mutually-inconsistent loop closures, and combining it
with per-measurement gating (our degeneracy + compass gates) is a sensible
belt-and-suspenders. **[SOURCED]** The specific queue size (5) and min-clique
(2) are not literature constants — min-clique 2 is the *weakest* consistency
requirement (a single consistent pair), which is permissive; the paper's
robustness comes from larger consistent sets. **Recommendation:** treat
min_pcm as the primary robustness knob and consider min-clique 3 where loop
density allows.

**5. Post-optimization verification then REVERT — VALIDATED.**
Comparing the optimized graph against an independent reference and rejecting
bad constraints is established: TBV (radar, RA-L 2023), RTAB-Map's
OptimizeMaxError, and Cartographer's constraint-scoring / branch-and-bound
acceptance (Hess, Kohler, Rapp, Andor; ICRA 2016). **[SOURCED]** Our specific
references — windowed yaw-RMS vs compass, per-link chain-tear vs dead reckoning
— are a novel *instantiation*, but the family (verify-then-commit / lazy loop
commitment) is standard. This is one of our better-grounded decisions.

**6. Censi closed-form ICP covariance offered opt-in, sampling as default —
VALIDATED as a default choice; the Censi OPTION is CONTRADICTED for our ICP
type.** *This is the most important finding in this review.*
Bonnabel, Barczyk & Goulette ("On the Covariance of ICP-based Scan-matching
Techniques," ACC 2016; arXiv 1410.7632) prove the Censi (2007) closed form is
valid for **point-to-plane** ICP but yields **"completely erroneous
covariances" for point-to-point ICP**, because the closed form does not
account for the rematching step. **[VERIFIED]** Our shipped ICP uses
`PointToPointErrorMinimizer` (config/icp.yaml) and our
`icp_covariance.cpp` builds the point-to-point Jacobian — so **the Bonnabel
critique lands directly on our opt-in Censi path.** Two consequences:
  - Keeping **sampled + FAST-MCD as the default is the correct, safe choice** —
    the literature validates it and warns against the closed form for our
    minimizer.
  - The **Censi option, as currently paired with point-to-point ICP, is on
    unsound theoretical footing** and can under-estimate covariance in
    degenerate (wall-sliding) geometry — the exact failure mode our degeneracy
    gate exists to catch. Newer estimators (Brossard, Bonnabel & Barrau, RA-L
    2020, arXiv 1909.05722; CELLO-3D, Landry et al., arXiv 1810.01470) address
    this. **Action: either switch the Censi path to a point-to-plane minimizer
    (which makes Censi rigorous), or document the Censi option as
    "point-to-point → optimistic covariance, use with the degeneracy gate
    tight," or adopt a Brossard-style estimator.**

**7. Compass / absolute-yaw gating of loop closures — VALIDATED (underwater),
BREAKS ON LAND.**
Using an absolute yaw reference to reject rotational-alias loop closures in
symmetric environments has direct underwater precedent — Vial et al. (JFR
2024) add absolute magnetometer yaw priors to a sonar pose graph. **[SOURCED]**
The same paper is the cautionary tale: near ferromagnetic structure they
inflated magnetometer variance 50×. So the *technique* is validated; the
*trust level* is environment-dependent, and on land (steel chassis) it must be
replaced by gyro-integrated yaw (see 1.4). Structured/symmetric-environment
sonar SLAM precedents (Mallios et al., *Scan matching SLAM in underwater
environments*, Auton. Robots 2014; VanMiddlesworth et al., FSR 2013) support
the underlying rotational-aliasing concern.

**8. Coast through DVL dropout on last velocity + live attitude with a hard
time budget — VALIDATED (standard practice).**
Bridging DVL outages by holding last-good velocity through the attitude
solution, with a bounded horizon, is standard AUV dead-reckoning practice;
model-aided INS is the higher-end version (Kinsey, Eustice & Whitcomb, "A
Survey of Underwater Vehicle Navigation," and the AUV-navigation literature,
researchgate 224166509). **[SOURCED/DOMAIN]** Our hard time budget +
covariance inflation is the conservative, textbook-consistent form. No source
contradicts it.

**9. Global relocalization by Sobol-sampled correlation over an occupancy grid
— NUANCED (defensible at our scale; not the SOTA method).**
The rigorous, exhaustive alternative is Cartographer's **branch-and-bound**
correlative scan matching (Hess et al., ICRA 2016), built on Olson's
real-time correlative scan matching (ICRA 2009). **[SOURCED]** Branch-and-
bound guarantees the global optimum within a window; our Sobol sampling is a
*budgeted stochastic* search that does not. At pool / small-survey scale
(our target), a well-seeded Sobol+Nelder-Mead correlation is defensible and
cheap. **Recommendation:** for large-area or safety-critical relocalization,
consider a branch-and-bound pass; document Sobol relocalization as
"best-effort, small-area." (This mirrors what we already ship for scan-match
init.)

**10. USBL fixes as innovation-gated, position-only, one-per-keyframe graph
priors — VALIDATED.**
Fusing acoustic absolute-position fixes as position priors in a pose graph is
standard underwater practice (Ribas, Ridao, Tardós & Neira, *Underwater SLAM
in man-made structured environments*, JFR 2008; Mallios; MORPH-project multi-
vehicle work), and the terrestrial analogue is GPS-prior fusion in
ORB-SLAM-style graphs. **[SOURCED/DOMAIN]** Innovation-gating (reject fixes
too far from the estimate) and position-only (never take heading from the
acoustic fix) are the correct, conventional guards. No source contradicts our
approach; the one-per-keyframe rate limit is a sensible engineering choice not
specifically prescribed by any paper.

---

## What the literature suggests we change or test next (prioritized)

1. **Fix the Censi option or re-document it (Part 2.6).** Highest-value,
   lowest-effort. Either point the Censi path at a point-to-plane minimizer
   (makes the closed form rigorous per Bonnabel), or annotate it as optimistic
   for point-to-point and keep the degeneracy gate tight. Sampled stays the
   default — the literature backs that.
2. **On any radar port, replace compass anchoring with gyro yaw-rate (Part
   1.4).** The single assumption most certain to break on land. The verify
   layer should compare optimized yaw against integrated gyro, not a
   magnetometer.
3. **Add motion de-skew + Doppler compensation *if* radar (Part 1.3).**
   De-skew to mid-sweep for odometry; add Doppler compensation specifically in
   the relocalization/map-localization path, where Burnett shows it matters
   most. Not needed for sonar.
4. **Offer a k-strongest / adaptive-per-azimuth detector alongside CFAR for
   radar, and benchmark (Part 1.2).** CFEAR's argument against direct CFAR is
   worth taking seriously; OS-CFAR is our most radar-appropriate variant to
   pit against it.
5. **Consider min-clique 3 for PCM where loop density allows, and treat
   branch-and-bound as the large-area relocalization upgrade (Parts 2.4,
   2.9).** Both are "scale up when needed," not "fix now."

---

## Sources

Primary sources gathered by the research harness (quality: primary). URLs as
retrieved; verify DOIs/venues before external citation.

- Abu-Alrub & Rawashdeh, *Radar Odometry for Autonomous Ground Vehicles: A
  Survey of Methods and Datasets*, 2023 — arXiv:2307.07861
- Adolfsson et al., *TBV Radar SLAM — Trust But Verify Loop Candidates*, RA-L
  2023 — arXiv:2301.04397
- Adolfsson, Magnusson, Alhashimi, Andreasson, Lilienthal, *CFEAR
  Radarodometry*, IROS 2021 / T-RO — arXiv:2105.01457
- Hong, Petillot, Wang, *RadarSLAM*, IROS 2020 — arXiv:2104.05347
- Burnett, Schoellig, Barfoot, *Do We Need to Compensate for Motion Distortion
  and Doppler Effects in Spinning Radar Navigation?*, RA-L 2021 —
  arXiv:2011.03512
- (radar odometry, motion/Doppler) — arXiv:2310.12729
- Sünderhauf & Protzel, *Switchable Constraints for Robust Pose Graph SLAM*,
  IROS 2012 — nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf
- Agarwal, Tipaldi, Spinello, Stachniss, Burgard, *Robust Map Optimization
  using Dynamic Covariance Scaling*, ICRA 2013 —
  informatik.uni-freiburg.de/~spinello/agarwalICRA13.pdf
- Mangelson, Dominic, Eustice, Vasudevan, *Pairwise Consistent Measurement Set
  Maximization*, ICRA 2018 — ieeexplore.ieee.org/document/8460217
- Kaess, Johannsson, Roberts, Ila, Leonard, Dellaert, *iSAM2*, IJRR 2012 —
  cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf
- RTAB-Map, *Robust Graph Optimization* — github.com/introlab/rtabmap/wiki
- Hess, Kohler, Rapp, Andor, *Real-Time Loop Closure in 2D LIDAR SLAM*
  (Cartographer), ICRA 2016 — research.google/pubs/archive/45466.pdf
- Olson, *Real-Time Correlative Scan Matching*, ICRA 2009 —
  april.eecs.umich.edu/pdfs/olson2009icra.pdf
- Censi, *An Accurate Closed-Form Estimate of ICP's Covariance*, ICRA 2007 —
  researchgate.net/publication/224705616
- Bonnabel, Barczyk, Goulette, *On the Covariance of ICP-based Scan-matching
  Techniques*, ACC 2016 — arXiv:1410.7632
- Brossard, Bonnabel, Barrau, *A New Approach to 3D ICP Covariance
  Estimation*, RA-L 2020 — arXiv:1909.05722
- Landry, Pomerleau, Giguère, *CELLO-3D: Estimating the Covariance of ICP in
  the Real World*, ICRA 2019 — arXiv:1810.01470
- Vial et al., underwater sonar SLAM with magnetometer yaw priors, JFR 2024 —
  doi.org/10.1002/rob.22272
- Ribas, Ridao, Tardós, Neira, *Underwater SLAM in man-made structured
  environments*, JFR 2008 (and AUV-navigation survey,
  researchgate 224166509)
- VanMiddlesworth, Kaess, Hover, Leonard, structured underwater mapping, FSR
  2013 — cs.cmu.edu/~kaess/pub/VanMiddlesworth13fsr.pdf
- Magnetometer yaw under disturbance (vehicle-kinematics EKF), SAE 2020-01-1025
- CFAR sonar detection, MDPI *Sensors* 2018 — doi.org/10.3390/s18010076
- Additional sources retrieved: arXiv:1909.05722, arXiv:2202.05811,
  arXiv:2404.01537, pmc.ncbi.nlm.nih.gov/articles/PMC11452886,
  scispace (sonar relocalization)

**Verification status.** Independently re-verified this session: the Bonnabel
point-to-point/point-to-plane distinction (Part 2.6) and CFEAR's k-strongest
detector + 1.76 %/55 Hz result (Parts 1.1–1.2). All other citations are
harness-extracted from the named primary sources and tagged **[SOURCED]**; the
automated verification stage did not complete due to a provider usage limit.
Re-run the verification pass, or confirm a specific figure against its source,
before quoting numbers externally.
