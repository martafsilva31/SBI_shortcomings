# Dimensionality study — design + first results (Phase 2)

> Status: **first results in** (2026-07-10, same day as design — Marta approved defaults + wider
> μ window). Goal of the phase: show the Asimov-property breakdown is driven by per-bin occupancy
> (effective dimensionality), not by machine learning per se.
> Started: 2026-07-10. Builds on: docs/hist-reproduction.md (1D machinery + metrics),
> docs/context.md (yields), code walkthrough (reusable engine).
>
> **▶ Canonical artifact: [`notebooks/dimensionality_study.ipynb`](../notebooks/dimensionality_study.ipynb)**
> — runnable, all 32 points (8 configs × 4 lumis), plots inline, parameters in one cell.

## Results (2026-07-10, first pass)

Full metrics table + figures in the notebook; scan tensors under `output/dim/`. Headlines:

1. **The Asimov property breaks in a plain histogram once it has enough cells — no ML involved.**
   At 1000 fb⁻¹ (5656 expected events) the width ratio median(toys)/Asimov falls from **1.08 (1D,
   10 bins) to 0.44 (4D, 10⁴ bins)** — same direction as the NSBI failure (toys narrower =
   overconfidence). The NSBI's measured 0.874 at 300 fb⁻¹ sits in the range of a 2D–3D histogram.
2. **Mean occupancy alone does not collapse the effect**: at fixed mean occupancy the
   higher-d/higher-L point is consistently worse (occ 0.57: 0.71 @ 3D/100 fb⁻¹ vs 0.44 @
   4D/1000 fb⁻¹). The steep m4ℓ spectrum makes the occupancy *distribution* (many near-empty
   cells) the relevant object — a refined effective-occupancy variable is the next design item.
3. **Control at fixed 16 total bins stays near 1** (0.76–1.06 across d) — the bulk of the effect
   tracks cell count, not dimensionality per se. Second-order structure is within ~200-toy median
   noise; needs more toys to resolve.
4. Asimov μ̂ = 1.000 at all 32 points; zero non-finite toys (the ν = 0-bin concern didn't
   materialise with μ ∈ [0, 8]).

Follow-ups: effective-occupancy metric for the collapse plot; toy-count scaling check; NSBI
overlay on the same panels once path-B training lands.

## 1. Physics question and the knob being turned

The histogram analysis satisfies Asimov ≈ median(toys) at all luminosities (Phase 1, reproduced);
the NSBI does not at 1–10 fb⁻¹. The working hypothesis (context.md, foundations §11) is that the
NN's effective partition of the 16-dimensional event space is so fine that every "cell" is deeply
non-asymptotic even when the *total* event count looks comfortable. A histogram with more axes is
the cleanest interpolation between the two regimes: same likelihood, same data, same toys — only
the bin occupancy changes. If Asimov-vs-median divergence appears in a plain 4D histogram at the
occupancies where NSBI fails, the ML wrapper is exonerated and the non-asymptotic mechanism is
established with no neural network in sight.

## 2. Occupancy arithmetic (what makes the grid interesting)

Expected SBI events (context.md): 5.656 × (L / 1 fb⁻¹). Mean occupancy = N_tot / N_bins with 10
bins per axis:

| mean occ. per bin | 1D (10 bins) | 2D (10²) | 3D (10³) | 4D (10⁴) |
|---|---|---|---|---|
| 1 fb⁻¹ (N = 5.66) | 0.57 | 0.057 | 0.0057 | 0.00057 |
| 10 fb⁻¹ (N = 56.6) | 5.7 | 0.57 | 0.057 | 0.0057 |
| 100 fb⁻¹ (N = 566) | 57 | 5.7 | 0.57 | 0.057 |
| 1000 fb⁻¹ (N = 5656) | 566 | 57 | 5.7 | 0.57 |

The table is the design: **constant occupancy runs along the diagonals**. If occupancy is the
driver, (1000 fb⁻¹, 4D) should misbehave like (1 fb⁻¹, 1D) behaves — and the Asimov-vs-median
mismatch should be a function of N/N_bins, collapsing the 16-point grid onto a single curve. That
collapse (or its failure) is the headline plot of the phase. Occupancy is of course non-uniform
(the m4ℓ spectrum falls steeply — the 1 fb⁻¹ Asimov histogram spans 1.85 → 0.006 events/bin), so
the analysis should also report the *effective* occupancy (e.g. weight-weighted mean, or the
fraction of expected events in bins with ν above a threshold), not only N/N_bins.

## 3. Observable menu (all from existing columns or the existing calculator)

| Axis | Observable | Source | Notes |
|---|---|---|---|
| 1 | m4ℓ | CSV `4l_mass` | the Phase-1 axis; 10 log bins in [180, 1000] GeV |
| 2 | mZ2 | CSV `Z2_mass` | off-shell Z pairing; complementary to m4ℓ |
| 3 | cos θ* | `AngularVariables` calculator | production angle; flat-ish, good occupancy behaviour |
| 4 | mZ1 (alt: y4ℓ) | CSV `Z1_mass` (alt: `4l_rapidity`) | on-shell Z window; rapidity as fallback if mZ1 too peaked |

- The decay angles (`cth_star`, `cth_1`, `cth_2`, `phi`, `phi_1`) are **not** CSV columns but are
  computable per event with the sub-repo's own `AngularVariables()` calculator
  (`physics/analysis/zz4l.py:125`, applied via `Process.calculate`, `mcfm.py:142`) — no new
  physics code needed.
- **Excluded: `4l_pT`** — identically zero at LO parton level (checked in the CSV: values are
  1e-5 GeV numerical noise), so it carries no information here. `4l_energy` is highly correlated
  with m4ℓ and y4ℓ; not a primary choice.
- Binning per axis: 10 bins is the 1D baseline; edges to be fixed **from 1D occupancy plots in the
  implementation notebook** (log bins for masses, uniform for cos θ*), not invented here. For the
  fixed-total-bins control (§4), coarser per-axis binnings (4², ~2×2×3, 2⁴) reuse the same edges
  merged.

## 4. Two scans, one primary and one control

1. **Primary — occupancy scan:** 10 bins/axis, d = 1…4, lumi ∈ {1, 10, 100, 1000} fb⁻¹ (the §2
   grid). This mimics how NSBI's effective partition outruns the data.
2. **Control — fixed total bins:** hold N_bins ≈ 16 while varying d (16¹, 4², 2×2×4, 2⁴). If the
   Asimov property depends only on occupancy, these should all behave like the 1D case — separating
   "dimension" from "cells".

## 5. Implementation route (reuse, no new engine)

- Fill a d-dimensional `hist.Hist` (the library supports multi-axis fills natively) for the Asimov
  weights `w_sm × L`, the 101-point μ-scan weights `w(μ) × L` (`sigstr.scale`, as in Phase 1), and
  the 200 Poisson toys (`toys.generate`, seed 42 — same recipe).
- Flatten: `h.values().flatten()` → `stats/binned.py` works **unchanged** for any d:
  `poisson_log_likelihood_scan(h_obs (N_bins,), h_exp (N_bins, N_scan))` and
  `poisson_log_likelihood_scan_toys((N_bins, N_toys), (N_bins, N_scan)) → (N_scan, N_toys)`.
  Zero-expectation bins are already handled (−inf only when a toy observes an event in a ν = 0
  bin; the toys variant clamps expectations at 1e-20) — at 4D/1 fb⁻¹ this *will* be exercised;
  the rate of −inf toys per μ point must be monitored and reported, not silently masked.
- Metrics: exactly the Phase-1 set (μ̂ per curve, Δ(−2lnL) = 1 widths where contained, curve-level
  max |A − M|), plus the occupancy-collapse plot of §2. The μ scan window may need widening beyond
  [0, 4] at low occupancy (Phase-1 lesson: intervals do not close at 1–10 fb⁻¹ in 1D).
- Memory/runtime: worst case is the toys tensor at 4D, (10⁴ bins × 200 toys) plus the scan
  (10⁴ × 101) — ~24 MB in float32; trivial on CPU. Event-loop cost identical to Phase 1
  (300k rows). Minutes per grid point.
- New artifacts when implemented (needs approval then): one container-level analysis notebook (the
  sub-repo stays untouched, same pattern as Phase 1), `output/dim/lumi_<L>_d<d>/` result dirs, and
  a `docs/dimensionality-study.md` results section appended to this note.

## 6. Expected outcomes and what each would mean

- **Collapse onto occupancy** (mismatch grows as N/N_bins ↓, same direction as NSBI: median
  narrower than Asimov): the failure is a generic non-asymptotic effect; NSBI is "just" a
  histogram with astronomically many cells → mitigations should target the statistics
  (Bartlett-type corrections, ensemble training), not the network.
- **No mismatch even at 4D/5.7e-4 occupancy:** the histogram's extended likelihood handles empty
  cells gracefully and the NSBI failure needs the *learned continuous* ratio (training-sample
  noise) — points at the estimator, strengthens the toy-ensemble-training hypothesis.
- **Mismatch in the opposite direction** (Asimov narrower): would contradict the Phase-1 NSBI
  direction and force a rethink of the mechanism — important to leave falsifiable.

## 7. Open design decisions (for Marta before implementation)

1. Confirm the axis order/menu of §3 (in particular mZ1 vs y4ℓ as the 4th axis).
2. Scan-window policy at low occupancy (widen μ grid vs report open intervals).
3. Whether the control scan (§4.2) is in scope for the first pass or a follow-up.
4. Toy count: 200 (Phase-1 parity) vs more (median stability at 4D will be poor — the Phase-1
   μ̂_M wobble says medians get noisy exactly where this study looks).
