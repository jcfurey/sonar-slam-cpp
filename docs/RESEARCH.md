# Research notes: papers, tech reports, and reference implementations

Literature and open-source provenance behind `sonar_slam_cpp`, categorized by the
part of the codebase each item underpins. Each entry carries a short note tying it
to specific code. Maturity figures (stars, releases) are as of 2026-07-11.

> Assembled from a multi-source research sweep with partial adversarial
> verification: of the fact-check votes that completed, all upheld their claims
> except one, which corrected the repo lineage (recorded below). Citation
> details (venues, DOIs, page ranges) were extracted from primary sources.

---

## 1. Direct lineage: BRUCE / bruce_slam (Robust Field Autonomy Lab)

The port's provenance chain is:
`jinkunw/bruce` (original BRUCE) → `jake3991/sonar-SLAM` (the maintained
`bruce_slam`, **direct upstream of this port**) → `sonar_slam_cpp`.
(A verification pass confirmed the files this repo ports — `CFAR.py`,
`feature_extraction.py`, `slam.py`/`slam_ros.py`/`slam_objects.py`,
`dead_reckoning.py`, `gyro.py`, `kalman.py`, `sensors.py` — exist in
`jake3991/sonar-SLAM`, not in the older `jinkunw/bruce` tree.)

- **jake3991/sonar-SLAM** — <https://github.com/jake3991/sonar-SLAM>
  The upstream `bruce_slam`: DVL/IMU dead reckoning + forward-looking-sonar
  observations, CFAR feature extraction, ICP scan matching, GTSAM/ISAM2 pose
  graph, PCM loop-closure verification. Originally by Jinkun Wang; documented
  and maintained by John McConnell. MIT license, ~296 stars / 41 forks.
  *Every node in `src/nodes/` ports a module of this repo (see README table).*

- **jinkunw/bruce** — <https://github.com/jinkunw/bruce>
  The original "BlueROV Underwater SLAM and Exploration (BRUCE)" codebase
  (BSD-3, created 2019); the ancestor of `bruce_slam` and origin of the name.
  Small research repo (~18 stars). *Historical provenance only — the ported
  sources live in jake3991/sonar-SLAM.*

- **J. Wang, F. Chen, Y. Huang, J. McConnell, T. Shan, B. Englot,
  "Virtual Maps for Autonomous Exploration of Cluttered Underwater
  Environments," IEEE Journal of Oceanic Engineering, 2022.**
  arXiv: <https://arxiv.org/abs/2202.08359>
  The canonical citation named by the upstream repo; its exploration framework
  is built on the imaging-sonar SLAM stack this repo ports, validated on a
  BlueROV in a real harbor. *Anchor citation for the whole pipeline.*

- **J. McConnell, Y. Huang, P. Szenher, I. Collado-Gonzalez, B. Englot,
  "DRACo-SLAM: Distributed Robust Acoustic Communication-efficient SLAM for
  Imaging Sonar Equipped Underwater Robot Teams," IROS 2022.**
  arXiv: <https://arxiv.org/abs/2210.00867> · code: <https://github.com/jake3991/DRACo-SLAM>
  Adopts Bruce-SLAM as its single-robot subsystem and gives one of the clearest
  published descriptions of the exact pipeline: CFAR-detected point clouds, ICP
  sequential scan matching factors, GTSAM/iSAM2 back-end, PCM verification.
  *Best single paper to read next to `src/core/slam_core.cpp`.*

- **DRACO-SLAM2 (IROS 2025)** — <https://github.com/RobustFieldAutonomyLab/DRACO-SLAM2> ·
  arXiv:2507.23629. The lab's actively developed successor line (object-graph
  matching for robot teams); evidence the ecosystem around this stack is alive,
  and a comparable open-source sonar SLAM stack for benchmarking.

- **E. Westman, A. Hinduja, M. Kaess, "Feature-based SLAM for Imaging Sonar
  with Under-constrained Landmarks," ICRA 2018.**
  PDF via CMU RI: <https://www.ri.cmu.edu/publications/feature-based-slam-for-imaging-sonar-with-under-constrained-landmarks/>
  Foundational imaging-sonar SLAM work on the elevation-ambiguity /
  under-constrained landmark problem; context for why the bruce_slam lineage
  chose dense CFAR + ICP scan matching over feature-landmark SLAM.

- **Robust Field Autonomy Lab, Stevens Institute of Technology** (PI: Brendan
  Englot) — <https://robustfieldautonomylab.github.io/> — publication list
  anchoring the citation tree (BlueROV sonar SLAM, DRACo line, companion
  vehicle software `jake3991/Argonaut`).

## 2. Detection front-end: CFAR

Underpins `include/sonar_slam_cpp/cfar.hpp`, `src/core/cfar.cpp`,
`src/cuda/cfar_cuda.cu`, and `src/nodes/feature_extraction_node.cpp`
(`enum Alg { CA, SOCA, GOCA, OS }`, threshold-factor solving, per-column
sliding window).

- **H. Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target
  Situations," IEEE Trans. Aerospace and Electronic Systems, AES-19(4):
  608–621, 1983.** DOI: 10.1109/TAES.1983.309350
  The foundational OS-CFAR paper (~1.4k citations): rank-selected order
  statistic instead of an average, robust to both interfering targets and
  clutter edges. *Primary citation for the `OS` variant and the
  `pfa_os`/root-finding threshold factor in `cfar.cpp`.*

- **P. P. Gandhi, S. A. Kassam, "Analysis of CFAR processors in
  nonhomogeneous background," IEEE Trans. Aerospace and Electronic Systems,
  24(4):427–445, 1988.** DOI: 10.1109/7.7185
  The canonical unified analysis of exactly the four families the repo
  implements — CA, greatest-of (GOCA), smallest-of (SOCA), OS — with
  closed-form false-alarm/threshold relations under an exponential noise
  model. *Best single citation for the CA/SOCA/GOCA/OS enum and the
  per-algorithm threshold factors.*

- **"Variability Index CFAR for Sonar Target Detection," ICSCN 2008.**
  DOI: 10.1109/ICSCN.2008.4447176
  Sonar-domain paper whose introduction crisply motivates the variant
  trade-offs (CA optimal only in homogeneous background; GO controls false
  alarms at reverberation edges; SO tolerates multiple targets). VI-CFAR
  adaptively switches between them — *a candidate future extension beyond the
  fixed `CFAR/alg` config option in `config/feature.yaml`.*

- **Mu, Chen, Wang, Qin, Zhu, "AUV SLAM method based on SO-CFAR and ADT
  feature extraction," Science Progress, 2024.**
  Open access: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11452886/>
  Recent independent validation that smallest-of CFAR is the preferred variant
  on forward-looking sonar (multi-target, reverberation-heavy scenes), feeding
  weighted ICP and a GTSAM/iSAM2 back-end — essentially this repo's
  architecture. *Supports the `alg: 'SOCA'` default in `feature.yaml`.*

- **Purdue ECE lecture notes on OS-CFAR** —
  <https://engineering.purdue.edu/~mrb/resources/AltLectureF/Session_22.pdf>
  Accessible derivations of the CA/GOCA/SOCA/OS threshold factors with the
  primary-source trail (Hansen 1973 for GO; Weiss 1982 for SO/GO; Rohling 1983
  for OS), including the closed-form OS-CFAR P_FA product the repo's
  `pfa_os()` implements. Notes that ranks near 3N/4–4N/5 are recommended —
  worth comparing against the repo's default `rank: 10` of `Ntc: 40` (N/4)
  when tuning.

## 3. Scan matching: ICP, libpointmatcher, and ICP covariance

Underpins the `ICP` wrapper in `src/core/cloud_ops.cpp`, `config/icp.yaml`,
and `Slam::compute_icp_with_cov` in `src/core/slam_core.cpp`.

- **F. Pomerleau, F. Colas, R. Siegwart, S. Magnenat, "Comparing ICP variants
  on real-world data sets: Open-source library and experimental protocol,"
  Autonomous Robots 34(3):133–148, 2013.**
  Free PDF: <https://hal.science/hal-01143458/document>
  The libpointmatcher paper: modular ICP pipeline (data filters, matcher,
  outlier rejectors, error minimizer) that `config/icp.yaml` configures
  verbatim. *The direct algorithmic reference for the registration stack.*

- **A. Censi, "An accurate closed-form estimate of ICP's covariance,"
  ICRA 2007, pp. 3167–3172.** DOI: 10.1109/ROBOT.2007.363961
  The canonical closed-form ICP covariance estimator (Hessian of the
  registration error, correlated correspondences). The original
  point-to-point helper remains in `src/core/icp_covariance.cpp`
  (`censi_icp_covariance`) as reference math. Under an isotropic, independent
  per-point noise model that estimate reduces to
  `cov = 2·σ²·(Σ Jᵢᵀ Jᵢ)⁻¹`, validated by Monte Carlo (predicted vs. empirical
  spread agree to ~1.5%). It is no longer runtime-selectable: the shipped ICP
  now uses a point-to-plane objective, and mixing it with this point-to-point
  Hessian would attach invalid confidence to graph factors. `Slam::configure`
  rejects `cov_method: censi`; sampled + FAST-MCD is the runtime path until a
  covariance implementation consumes the same normals and objective.

- **F. Pomerleau, F. Colas, R. Siegwart, "A Review of Point Cloud Registration
  Algorithms for Mobile Robotics," Foundations and Trends in Robotics, 2015.**
  Companion survey to the libpointmatcher paper; broader registration context.

## 4. Back-end: factor graphs, ISAM2, GTSAM

Underpins `gtsam::ISAM2`, `NonlinearFactorGraph`, `BetweenFactor`/`PriorFactor`
usage in `src/core/slam_core.cpp` and `include/sonar_slam_cpp/slam_core.hpp`.

- **M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. Leonard, F. Dellaert,
  "iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree,"
  International Journal of Robotics Research 31(2):216–235, 2012.**
  Author PDF: <https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf>
  The exact incremental optimizer the back-end calls (`isam_.update(...)` in
  `Slam::update_factor_graph`): Bayes-tree factorization, incremental variable
  re-ordering, fluid relinearization — no periodic batch steps.

- **F. Dellaert, M. Kaess, "Factor Graphs for Robot Perception," Foundations
  and Trends in Robotics, 2017.**
  PDF: <http://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf>
  The standard monograph explaining why odometry, sequential scan matching,
  and loop closures are modeled as factors over a pose graph — the mental
  model behind `add_odometry` / `add_sequential_scan_matching` /
  `add_nonsequential_scan_matching`.

- **GTSAM (software citation):** F. Dellaert and GTSAM Contributors,
  *borglab/gtsam*, Georgia Tech Borg Lab. DOI: 10.5281/zenodo.5794541 ·
  <https://github.com/borglab/gtsam>

## 5. Loop-closure verification: PCM

Underpins `Slam::verify_pcm` and the Bron–Kerbosch `find_cliques` port in
`src/core/slam_core.cpp`.

- **J. G. Mangelson, D. Dominic, R. M. Eustice, R. Vasudevan, "Pairwise
  Consistent Measurement Set Maximization for Robust Multi-Robot Map
  Merging," ICRA 2018, pp. 2916–2923.** DOI: 10.1109/ICRA.2018.8460217 ·
  open access: <https://par.nsf.gov/biblio/10354834>
  PCM: select the largest pairwise internally consistent set of loop closures
  by solving a maximum-clique problem over a consistency graph, using a
  Mahalanobis-distance test whose normalized error is chi-squared distributed.
  *Explains both the max-clique search and the `md < 11.34` gate
  (chi2.ppf(0.99, 3)) in `verify_pcm`. Reported to significantly outperform
  DCS, SCGP, and RANSAC on synthetic and real data.*

- **B. Forsgren, M. Kaess, R. Vasudevan, T. W. McLain, J. G. Mangelson,
  "Group-k Consistent Measurement Set Maximization," IJRR 2024.**
  DOI: 10.1177/02783649241256970
  The PCM authors' generalization to group-k consistency over hypergraphs.
  *Not used here — further reading / potential upgrade path for the
  loop-closure gate.*

## 6. Robust statistics and global optimization

Underpins `src/core/mcd.cpp` (FAST-MCD) and `src/core/global_init.cpp`
(Sobol sampling + Nelder–Mead, replacing `scipy.optimize.shgo`).

- **P. J. Rousseeuw, K. Van Driessen, "A Fast Algorithm for the Minimum
  Covariance Determinant Estimator," Technometrics 41(3):212–223, 1999.**
  FAST-MCD (~2.8k citations): C-steps with the monotone-determinant guarantee
  (Theorem 1: det(S2) ≤ det(S1)), selective iteration, reweighting. *The
  primary citation for `min_cov_det()`; the C-step monotonicity is the
  correctness property to test in the port. The repo mirrors scikit-learn's
  `MinCovDet` (support-fraction floor, consistency correction, chi-squared
  0.975 reweighting) because that is what slam.py called.*

- **M. Hubert, M. Debruyne, "Minimum covariance determinant," WIREs
  Computational Statistics 2:36–43, 2010.** DOI: 10.1002/wics.61
  Accessible survey of MCD breakdown properties and the reweighting step —
  good secondary source when reviewing `mcd.cpp`.

- **S. C. Endres, C. Sandrock, W. W. Focke, "A simplicial homology algorithm
  for Lipschitz optimisation," Journal of Global Optimization 72:181–217,
  2018.** DOI: 10.1007/s10898-018-0645-y ·
  open access: <https://repository.up.ac.za/bitstream/2263/64460/1/Endres_Simplicial_2018.pdf> ·
  reference implementation: <https://github.com/Stefan-Endres/shgo>
  The SHGO paper behind `scipy.optimize.shgo(sampling_method="sobol")`, which
  slam.py used for global scan-match initialization. *`global_init.cpp`
  replaces it with a Sobol sweep + Nelder–Mead refinement over the same
  dilated-occupancy-grid cost; this is the citation for what that code
  approximates.*

- **S. Joe, F. Y. Kuo, "Constructing Sobol sequences with better
  two-dimensional projections," SIAM J. Scientific Computing 30(5):2635–2654,
  2008.** The standard source of Sobol direction numbers; the `Sobol3` class
  in `global_init.cpp` uses Joe–Kuo direction numbers for its 3-dim sequence.
  *(Added from the code comment's own reference; not part of the verified
  sweep.)*

## 7. Underwater navigation and SLAM surveys

Context for the dead-reckoning front-end (`dead_reckoning_node.cpp`,
`gyro_node.cpp`, `kalman_node.cpp`) and the overall architecture.

- **L. Paull, S. Saeedi, M. Seto, H. Li, "AUV Navigation and Localization:
  A Review," IEEE Journal of Oceanic Engineering 39(1), 2014.**
  Author PDF: <https://people.csail.mit.edu/lpaull/publications/Paull_JOE_2013.pdf>
  The canonical AUV navigation survey: dead-reckoning error growth is
  unbounded (motivating sonar loop closures); DVL velocity noise ~0.3–0.8 cm/s;
  gyro drift spans 0.0001°/hr (RLG/FOG) to 60°/hr (MEMS) — the reason this
  stack fuses a KVH FOG with the VN-100 MEMS IMU. *Grounds the four
  localization modes in `dead_reckoning_node.cpp` and the sensor suite in
  `config/*.yaml` (Oculus M750d/M1200, Rowe SeaPilot DVL, VN-100, KVH
  DSP-1760 — the upstream repo's reference hardware).*

- **"Advancements in Sensor Fusion for Underwater SLAM: A Review on Enhanced
  Navigation and Environmental Perception," Sensors, 2024.**
  DOI: 10.3390/s24237490 · <https://pmc.ncbi.nlm.nih.gov/articles/PMC11644431/>
  Modern survey situating graph-based back-ends among underwater SLAM
  approaches; documents the per-modality failure modes (vision in turbidity,
  sonar resolution limits, IMU drift) that motivate sonar-corrected dead
  reckoning.

- **Cheng, Wang, Yang, Liu, Zhang, "Underwater Localization and Mapping Based
  on Multi-Beam Forward Looking Sonar," Frontiers in Neurorobotics, 2022.**
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8791027/>
  An architecturally different FLS SLAM system (RBPF + occupancy grid,
  strongest-return-per-beam extraction instead of CFAR, EKF dead reckoning) —
  useful as a contrast point showing the design space around this repo's
  CFAR + ICP + factor-graph choices, not a foundation for them.

## 8. Well-proven open-source repositories

| Repository | Role for this repo | Maturity evidence |
| --- | --- | --- |
| [borglab/gtsam](https://github.com/borglab/gtsam) | Factor-graph/ISAM2 back-end linked by `slam_core.cpp` | ~3.6k stars / ~960 forks, BSD, active releases (4.2.x line current), Python/MATLAB wrappers; de-facto standard SLAM back-end |
| [norlab-ulaval/libpointmatcher](https://github.com/norlab-ulaval/libpointmatcher) | ICP engine wrapped by `cloud_ops.cpp`, configured by `icp.yaml` | ~1.8k stars, BSD-3, v1.4.4 (Dec 2024), ETH-ASL origin → NorLab maintained; adopters include NASA Ames Stereo Pipeline, CGAL, ANYbotics |
| [jake3991/sonar-SLAM](https://github.com/jake3991/sonar-SLAM) | Direct upstream (`bruce_slam`) this repo ports | ~296 stars / 41 forks, MIT, backed by the IEEE JOE 2022 paper; field-validated on a BlueROV |
| [jinkunw/bruce](https://github.com/jinkunw/bruce) | Original BRUCE ancestor of bruce_slam | Small research repo (BSD-3, 2019–2025); provenance value |
| [jake3991/DRACo-SLAM](https://github.com/jake3991/DRACo-SLAM) / [RobustFieldAutonomyLab/DRACO-SLAM2](https://github.com/RobustFieldAutonomyLab/DRACO-SLAM2) | Same-lab multi-robot stacks built on the ported pipeline | IROS 2022 / IROS 2025 papers with released code and datasets; comparable systems for benchmarking |
| [Stefan-Endres/shgo](https://github.com/Stefan-Endres/shgo) | Reference implementation of the optimizer `global_init.cpp` approximates | Merged into SciPy as `scipy.optimize.shgo` — the behavior baseline for parity |
| scikit-learn `MinCovDet` / SciPy | Reference implementations `mcd.cpp` and `global_init.cpp` were written to match | Industry-standard libraries; `test/parity_driver.py` diffs against them directly |

---

### Cross-cutting notes

- The clearest end-to-end published description of the exact ported pipeline is
  the DRACo-SLAM paper (§1), which names CFAR [Rohling/Gandhi–Kassam], ICP
  [Pomerleau], iSAM2 [Kaess], and PCM [Mangelson] in one system diagram.
- Two entries double as tuning references rather than provenance: the OS-CFAR
  rank guidance (§2, 3N/4–4N/5 vs. the shipped N/4) and Censi's closed-form
  ICP covariance (§3) as a cheaper alternative to the sampled-ICP + FAST-MCD
  path.

---

## Findings status: what was acted on

The research surfaced mostly **citations** (the bibliography above) plus a few
**candidate algorithmic extensions**. Because this repo's contract is drop-in
behavioral parity with `bruce_slam`, extensions are added opt-in (default
config → unchanged behavior) and only when their correctness can be validated.

**Retained as reference math, disabled at runtime:**

- **Censi closed-form ICP covariance** (§3) — `src/core/icp_covariance.cpp`,
  contains the earlier point-to-point closed form. The covariance formula is
  Monte-Carlo validated
  (`test/censi_covariance_test.cpp`; predicted vs. empirical spread within
  ~1.5%), but this validates only the isolated known-correspondence math.

  An older integration check showed that the point-to-point implementation
  produced finite graph marginals, but that does not validate it against the
  current point-to-plane error minimizer. `Slam::configure` now rejects the
  mode. Re-enabling it requires a point-to-plane covariance implementation and
  an end-to-end degeneracy test using the configured surface normals.

**Documented, not changed (would break `bruce_slam` parity):**

- **OS-CFAR rank** (§2) — the literature favours k ≈ 3N/4–4N/5, whereas the
  shipped `feature.yaml` uses `rank: 10` with `Ntc: 40` (N/4), inherited
  verbatim from `bruce_slam`. Left as-is to preserve tuning parity; the value
  is already a config knob (`CFAR/rank`) for anyone who wants to follow the
  guidance. The code's negative-`rank` fallback stays at N/2 to match
  `CFAR.py`'s `rank=None`.

**Deferred (new algorithms; correctness not verifiable in this environment):**

- **VI-CFAR** (§2) — an adaptive CA/GO/SO detector. Would be a genuinely new
  detector variant (no `bruce_slam` reference to match), and its decision
  truth table needs validation against a reference implementation before it
  can be trusted on real sonar. Tracked as a candidate `CFAR/alg: VI`.
- **Group-k consistent measurement set maximization** (§5) — a research-grade
  generalization of the PCM loop-closure gate; the current PCM
  (`verify_pcm`) is correct and sufficient. Noted as a future upgrade path.
