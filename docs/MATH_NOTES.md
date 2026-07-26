# Mathematical notes: proofs behind the core algorithms and their improvements

Rigorous derivations for the CFAR detector (threshold factors, the OS
calibration fix, the exact prefix-sum optimization) and proof-backed proposals
for the other core components. Everything marked **implemented** is validated
by an in-repo test; everything marked **proposal** carries its proof here so it
can be implemented later without re-deriving.

Notation: the polar sonar image after square-law detection is modeled, per
cell under noise-only conditions (H0), as i.i.d. exponential: `X ~ Exp(mu)`
with unknown clutter power `mu`. `N = Ntc` training cells, `h = N/2` per
half-window, `tau` the threshold factor, and the detector fires when
`cell > tau * Z` for a statistic `Z` of the training cells.

---

## 1. The CFAR property (why "constant false alarm rate" holds at all)

**Lemma 1.** For every statistic used here (mean, half-window sums with
min/max, order statistics), `Z` is positively homogeneous: `Z(cY) = c Z(Y)`
for `c > 0`. Hence for `X, Y_i ~ Exp(mu)`,
`P(X > tau Z(Y)) = P(X/mu > tau Z(Y/mu))` with `X/mu, Y/mu ~ Exp(1)` — the
false-alarm probability is independent of the clutter power `mu`.

*Proof.* Sums, minima/maxima of sums, and order statistics all commute with
positive scaling; dividing both sides of the comparison by `mu` gives the
unit-rate model. ∎

All derivations below therefore take `mu = 1` WLOG.

## 2. CA-CFAR threshold factor (closed form)

The detector fires when `cell > tau * S/N`, `S = sum of N training cells ~
Gamma(N, 1)`. Since `P(X > y) = e^{-y}` for `X ~ Exp(1)` independent of `S`:

    P_FA = E[e^{-(tau/N) S}] = (1 + tau/N)^{-N}        (Gamma MGF)

Solving for `tau`: `tau = N (P_FA^{-1/N} - 1)` — exactly
`calc_threshold_factor_ca()` in `cfar.cpp`.

## 3. SOCA threshold factor (full derivation)

Statistic: `min(S1, S2)/h` with `S1, S2 ~ Gamma(h, 1)` i.i.d. (leading /
lagging half-windows), `a = tau/h`. By symmetry (ties have measure zero):

    P_FA = 2 P(X > a S1, S1 < S2)
         = 2 ∫_0^∞ (s^{h-1} e^{-s} / Γ(h)) e^{-as} P(S2 > s) ds

With `P(S2 > s) = e^{-s} Σ_{j=0}^{h-1} s^j / j!` (Gamma survival, integer h):

    P_FA = 2 Σ_{j=0}^{h-1} 1/(Γ(h) j!) ∫_0^∞ s^{h-1+j} e^{-(2+a)s} ds
         = 2 Σ_{j=0}^{h-1} C(h+j-1, j) (2+a)^{-(h+j)}

which is precisely `pfa_gosoca_core()` (`exp(lgamma(h+k) - lgamma(k+1) -
lgamma(h)) = C(h+k-1, k)`) doubled, i.e. the root equation
`core(tau) = P_FA/2` in `pfa_soca()`. ∎

## 4. GOCA threshold factor (inclusion–exclusion)

With events `A = {X > a S1}`, `B = {X > a S2}`:
`{X > a·max} = A ∩ B` and `{X > a·min} = A ∪ B`, so

    P_FA^GOCA = P(A) + P(B) - P(A ∪ B) = 2 (1+a)^{-h} - P_FA^SOCA

using `P(A) = E[e^{-a S1}] = (1+a)^{-h}`. This is exactly `pfa_goca()`:
`temp - core = P_FA/2` with `temp = (1+a)^{-h}`. ∎

## 5. OS-CFAR: the Rényi product, and the carried-over calibration bug

**Theorem 2 (Rényi representation).** For `Z_(1) ≤ … ≤ Z_(N)` the order
statistics of N i.i.d. `Exp(1)` variables, the spacings are independent with
`Z_(j) = Σ_{i=1}^{j} E_i / (N - i + 1)`, `E_i ~ Exp(1)` i.i.d. (Memorylessness:
after the i-th smallest arrives, the remaining `N-i` excesses are again i.i.d.
`Exp(1)`, and their minimum is `Exp(N-i)`.)

**Corollary.** `P(X > tau Z_(j)) = E[e^{-tau Z_(j)}]
= Π_{i=1}^{j} (N-i+1)/(N-i+1+tau) = Π_{i=0}^{j-1} (N-i)/(N-i+tau)`.
(Sanity: `j=1` gives `N/(N+tau)`, the min of N exponentials. ∎)

**The bug.** The detector (CPU `cfar.cpp`, GPU `cfar_cuda.cu`, and the
*original* `bruce_slam` `cpp/cfar.cpp`) thresholds on `train[rank]` after
`nth_element` — the **(rank+1)-th smallest** (0-indexed), i.e. `Z_(rank+1)`.
But `CFAR.py`'s `calc_WGN_pfa_OS` (and the port before this fix) solved

    Γ(N+1)/Γ(N-rank+1) · Γ(tau+N-rank+1)/Γ(tau+N+1)
      = Π_{i=0}^{rank-1} (N-i)/(N-i+tau)  =  P(X > tau Z_(rank))

— the model for the **rank-th** smallest: one factor short. Since
`Z_(rank+1) ≥ Z_(rank)`, the solved `tau` over-thresholds and the realized
false-alarm rate is

    P_actual = P_target · (N - rank)/(tau + N - rank)  <  P_target.

At the shipped config (`N=40, rank=10, P_FA=0.1`) this predicts
`P_actual ≈ 0.0767` — a 23% deficit. **Monte Carlo confirms it exactly**
(`test/cfar_math_test.cpp`: empirical 0.0766 with the old tau vs. prediction
0.0767; 0.0999 with the fix). The fix (`pfa_os()` now uses `Γ(N-rank)` /
`Γ(tau+N-rank)`, i.e. the product with `rank+1` factors) makes the model match
the detector's actual statistic. Recorded as a divergence from `bruce_slam`
in `docs/DIVERGENCES.md`. **Implemented + validated.**

## 6. Uniqueness of the threshold roots (bisection correctness)

**Proposition.** Each P_FA expression above is continuous and strictly
decreasing on `(0, ∞)` with `P_FA(0+) = 1` and `P_FA(∞) = 0`; hence for any
target in `(0,1)` the root is unique and `solve_root`'s sign-change scan +
bisection converges to it.

*Proof.* Each is `E[e^{-tau W}]` for a positive random variable `W`
(`S/N`, `min/h`, `Z_(j)`) or, for GOCA, `E[e^{-tau·max/h}]` — expectations of
strictly decreasing positive integrands, so strictly decreasing; monotone
convergence gives the limits. ∎

## 7. Detection probability in closed form (principled parameter choice)

For a Swerling-I target with mean SNR `S`, the H1 cell after square-law
detection is `Exp(1/(1+S))`-distributed, so `P(X_1 > tau Z) =
E[e^{-tau Z/(1+S)}]`: **every formula in §2–5 gives P_D by substituting
`tau -> tau/(1+S)`**, e.g.

    P_D^OS = Π_{i=0}^{rank} (N-i)/(N-i + tau/(1+S)).

This yields exact ROC curves for all four detectors and lets `rank` (and the
CA/SOCA/GOCA choice) be selected analytically instead of by folklore — e.g.
quantifying the earlier observation that literature favors `rank ≈ 3N/4`
while the shipped config uses `N/4`. **Derived; ready to use for tuning.**

## 8. Exact integer arithmetic in float, and the prefix-sum CFAR

**Theorem 3 (exactness).** (i) Every integer `|n| ≤ 2^24` is exactly
representable in IEEE-754 binary32. (ii) If `a, b` are such integers and
`|a ± b| ≤ 2^24`, then the float operation is exact (correct rounding must
return the representable true result). (iii) By induction, any summation —
in any order — of `m` values from `{0,…,255}` with `m·255 ≤ 2^24` is exact
and order-independent.

**Theorem 4 (prefix-sum equivalence).** For a CV_8UC1 image, per column let
`P[k] = Σ_{i<k} img[i]` in int32 (exact for any realistic column length,
`rows ≤ 2^31/255`). Then for each pixel the training sums

    leading = P[r-g] - P[r-b],   lagging = P[r+b+1] - P[r+g+1]

(`b = train_hs+guard_hs`, `g = guard_hs`) equal the naive float accumulations
of `detect_cpu` **exactly** (both are the same integer, ≤ Ntc·255 < 2^24, by
Thm 3), and casting to float is exact. Since the subsequent threshold
expressions are evaluated identically, **the masks are bit-identical.** ∎

**Corollary 5 (CPU/GPU order-invariance).** For integer-valued images the GPU
kernel's summation order is irrelevant (every partial sum is exact), so the
CPU (naive or prefix) and GPU CFAR masks agree bit-for-bit — the
"float summation order can flip a borderline pixel" caveat in `parity_check`
applies only to genuinely fractional float inputs.

**Complexity.** Naive: `O(rows·cols·(2b+1))` adds (`2b+1 = 51` at the shipped
`Ntc=40, Ngc=10`). Prefix: `O(rows·cols)` — one add per pixel for the prefix
plus O(1) per statistic. Measured on 716×512 SOCA: **3.9 ms → 0.36 ms (11×)**
on 4 CPU cores. Implemented as the uint8 fast path in `CFAR::detect`
(`detect_cpu` retained unchanged for float inputs, OS, and the parity tools);
bit-exactness is asserted over 30 size/alg/gate combinations in
`test/cfar_math_test.cpp`. **Implemented + validated.**

## 9. Proposals with proofs (not yet implemented)

**9a. Sliding-histogram OS-CFAR.** For integer data, maintain a 256-bin count
of the training multiset; advancing one row changes the window by O(1) cells
(insert/remove at the leading and lagging edges), and the rank query walks at
most 256 bins. Exactness is trivial (integer counts); complexity drops from
`O(Ntc)` per pixel (`nth_element` average) to `O(1)` amortized update +
`O(256)` worst-case select — profitable for large `Ntc`, marginal at
`Ntc = 40`, which is why it is deferred.

**9b. Distance-transform scan-match cost with a Lipschitz certificate.** The
current global-init cost is a hit count on a dilated binary grid — piecewise
constant, so Nelder–Mead has no descent information on plateaus. Replacing it
with a (truncated) distance-transform cost `Σ_i min(DT(T_p q_i), d_max)` gives:
since any point-set distance function is 1-Lipschitz
(`|DT(u) - DT(v)| ≤ ‖u-v‖`), and a pose perturbation moves a point by at most
`‖Δt‖ + 2 R sin(|Δθ|/2) ≤ ‖Δt‖ + R|Δθ|` (`R` = max point radius), the cost is
Lipschitz with constant `L ≤ N(1 + R)` in `(t, θ)`. Corollaries: (i) grid
resolution `ε` bounds the discretization error by `Nε`; (ii) branch-and-bound
over pose boxes with per-box uncertainty radius `r = δt + 2R sin(δθ/2)`
(lower bound = center cost evaluated on a grid eroded by `r`) is admissible,
giving a **certified global optimum** for loop-closure initialization,
Go-ICP-style. Behavior-changing → would ship opt-in like `cov_method`.

**9c. Sobol correctness invariant.** The Gray-code generator (Antonov–Saleev)
enumerates, within each block of `2^m` consecutive indices, exactly the same
point set as the radical-inverse definition; with the (nonsingular,
upper-triangular bit-matrix) Joe–Kuo direction numbers, each dimension's first
`2^m` points hit each dyadic interval `[k 2^{-m}, (k+1) 2^{-m})` exactly once.
This is a machine-checkable stratification invariant for `Sobol3` (currently
TU-local; expose if a regression test is wanted).

**9d. Kalman state pruning (provably behavior-preserving).** In the 12-state
filter, states 0–1 (x, y) are never measured (`H_*` columns 0–1 are zero in
every model) and influence no other state (`A_imu` columns 0–1 have support
only in rows 0–1). The `(x,y)`-block therefore forms an unobservable,
non-influencing sink: the Riccati recursion of the remaining 10 states closes
on itself, and every published quantity (which uses states 2–11 plus the
externally integrated x/y) is unchanged if rows/cols 0–1 are deleted. Left
unimplemented purely for config parity (the 12-state YAML layout).

**9e. Anisotropic Censi covariance.** The implemented estimator is the
isotropic special case `cov = 2σ²(Σ JᵢᵀJᵢ)^{-1}`. The general GLS form for
per-correspondence noise `Σᵢ` is `cov = (Σ JᵢᵀΣᵢ^{-1}Jᵢ)^{-1}` — for a
forward-looking sonar, `Σᵢ` grows with range in the bearing direction
(`σ_bearing ≈ r·σ_θ`), so far points would be down-weighted correctly.
Standard weighted-least-squares variance result; extend
`icp_covariance_math.hpp` when range metadata is plumbed through.

## 10. Validation index

| Claim | Where proved | Where validated |
| --- | --- | --- |
| CA/SOCA/GOCA/OS τ solve their closed forms | §2–5 | `cfar_math_test` [1], 1e-12 |
| OS calibration off-by-one + magnitude | §5 | `cfar_math_test` [2], MC matches theory to 4 decimals |
| Realized P_FA = design P_FA for the actual detector statistics | §1–5 | `cfar_math_test` [2] |
| Prefix path bit-identical on uint8 | §8 | `cfar_math_test` [3], 0/30 mismatches |
| 11× CPU speedup | §8 | `cfar_math_test` [4] |
| Retained point-to-point covariance constant `2σ²` | RESEARCH.md §3 | `censi_covariance_test`, ~1.5% |
| Not-a-knot spline ≡ scipy | DIVERGENCES.md | `interp_spline_test`, ~1e-15 |
