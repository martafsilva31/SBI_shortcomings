# Phase 3 — proposed fixes (running log)

> Status: **in progress**. Goal: test mitigations for the non-asymptotic Asimov-property failure.
> Three prongs (docs/plan.md Phase 3): (1) Bartlett/alternative test statistics, (2) MC-stat
> uncertainties, (3) toy-ensemble training. Started 2026-07-11.
>
> **▶ Notebooks:** `notebooks/bartlett_correction.ipynb` (prong 1; includes the 2026-07-11 audit).

## Prong 1 — Bartlett correction: first pass + audit (2026-07-11)

**Design:** coverage test on the saved toy ensembles (histogram + dimensionality runs, no
re-running). Toys thrown at SM truth μ=1; per toy, the 68% interval is where t(μ) crosses the
threshold — 1 (naive χ²₁) vs the toy-ensemble mean E[t_μ] (Bartlett). Coverage = fraction of
intervals containing μ=1; nominal 0.683.

**Audit (same day, at Marta's request).** The first pass was checked adversarially: interval-walk
coverage re-derived directly from t(1) (agrees to ~0.01), tensors clean, but **one interpretation
error and one missed mechanism were found and fixed**:

- ❌ *First-pass error:* "E[t]@1σ ≈ 1.8–2.3 vs asymptotic 1" conflated the noncentral separation term
  (asymptotic expectation at the Asimov 1σ boundary is ≈ 2, not 1) with inflation. The true Bartlett
  factor is **E[t(1)] = 1.33–1.36 at 1000 fb⁻¹ and 0.95 (no inflation) at 100 fb⁻¹**.
- ✅ *Coverage results stand:* naive 68% intervals under-cover at 1000 fb⁻¹ (0.585–0.600 direct);
  thresholding at E[t(1)] restores 0.655–0.685. At 100 fb⁻¹ nothing to fix.

**The actual mechanism (found in the audit): interference-induced rate degeneracy.** The destructive
interference makes the total rate stationary in μ at the SM: dν_tot/dμ|₁ = σ_S + σ_I/2 = 0.0067 fb
(×46 suppressed; ν_tot minimised at μ* = (σ_I/2σ_S)² = 0.957, two-to-one map around it). First-order
information ~vanishes at truth; the likelihood is locally quartic-like. Confirmed signatures: μ̂ std
scales ≈ L^(−1/4) (measured 0.70→0.42, ratio 0.60 vs 0.56 predicted; regular = 0.32); μ̂ biased low
(mean 0.86 at 1000 fb⁻¹); var[t(1)] ≈ 3.2 (χ²₁: 2); effect *grows* with luminosity (more data
resolves the fold) and is occupancy-independent (lives in the μ-parametrization, not the binning).

**Consequences for the project:**
1. There are now **two distinct failure mechanisms**: (a) cell-count/occupancy (Phase-2 result,
   central Asimov≈median behaviour) and (b) interference rate degeneracy (coverage/tail behaviour,
   luminosity-driven). The plain histogram passes the Asimov-property check yet under-covers at high
   L — the diagnostics probe different pathologies.
2. Mechanism (b) applies **identically to NSBI** (same μ parametrization) and complicates the
   "Poisson term takes over at high lumi" narrative: near the SM the rate term carries almost no
   μ-information (dν/dμ ≈ 0); it constrains only the far tails.
3. Bartlett-at-truth empirically fixes the 68% quantile, but the model is *non-regular* at μ≈1, so
   Bartlett's guarantees don't apply — treat as a patch at this quantile, not a validated fix.

### M1 executed (plan items 1–4, 2026-07-11) — prong-1 verdict

- **Under-coverage at 1000 fb⁻¹ confirmed with error bars** (1000 per-bin toys; per-bin ≡ per-event
  proven and cross-validated): naive 68% coverage 0.620/0.615 ± 0.015 (1D/4D); E[t(1)] = 1.17 ± 0.07.
- **Bartlett = partial fix**: 0.585 → 0.637 (at-truth) → **0.641 (proper per-hypothesis)** vs 0.683
  nominal — residual ~4% under-coverage; mean-rescaling can't fix the shape distortion near the fold.
- **Miscoverage is localised at the fold** (multi-truth toys): dip to ~0.60 at μ_true ∈ {0.957, 1},
  recovery to 0.65–0.71 at {0.5, 2}; μ̂ attracted toward μ* from both sides. The fold, not occupancy,
  drives coverage failure.
- **⚠ Median-convention discovery**: the Phase-2 width-ratio headline is convention-dependent —
  occ_4d@1000 = 0.44 (torch.median of raw; even-n lower-middle quirk) / 0.71 (np.median of raw) /
  **1.10 (median of per-toy Δ curves — no dimensionality effect under this convention)**. Tei's NSBI
  direction check is robust (0.87 all conventions). Convention decision needed before further use
  (foundations §8; slide-5 warning now demonstrably material). Phase-2 note carries the caveat.

### M2 executed (items 5–6) — mechanism characterised

`notebooks/rate_degeneracy.ipynb`: per-bin folds μ*_b ∈ [0.46, 6.7] (shape partially rescues the
global fold); the rate term holds **0.0%** of the Fisher information at μ=1; reparametrisation
(κ=√μ) provably cannot help (LR invariance verified; dν/dκ also suppressed ×23) — the fold is
physical. Foundations §7 + §13 carry dated caveats. Mitigations must target the statistic's
calibration or the estimator, not the parametrization.

### M3 executed (item 8) — prong-2 gate numbers

Per-bin MC errors (dimensionality notebook §10): 1D 1.2% median / 2D 6.4% / 3D 22% (5.5% of yield in
>10%-error bins) / 4D 51% (20% of yield). **Recommendation: document-and-skip BB-lite for 1D–2D work;
mandatory only if 4D becomes quantitative** (currently moot pending the convention decision). Marta
decides.

**Next (updated 2026-07-13 — decisions taken):** median convention = Δ-primary + np-median-raw
secondary, torch.median retired (provisional, ⚠ not yet team-ratified — goes into the item-7 Tei
message); BB-lite = GO in every scenario (Marta overrides the document-and-skip recommendation;
items 8a–8d added). Remaining: Tei message (item 7); NSBI coverage audit when path A lands;
1000-toy coverage-vs-level as standing validation.

## Prong 2 — MC statistical uncertainties — gating quantification running (plan item 8)

Per-bin effective MC sample sizes n_eff = (Σw)²/Σw² and fractional MC errors across the occupancy
configs — dimensionality notebook §10. Decision (Marta) after numbers land: implement
Barlow–Beeston-lite (needs profiling machinery in `stats/`) or document-and-skip.

## Prong 3 — toy-ensemble deep-set prototype: DESIGN NOTE (plan item 9 — files named for approval)

**Goal (v0, deliberately minimal):** demonstrate that a permutation-invariant network trained on many
toy datasets learns *luminosity-aware uncertainty* — the thing single-Asimov CARL training cannot
(foundations §14). Not a full NSBI replacement.

**Architecture:** DeepSets ρ(Σφ). φ: per-event MLP, input = 16 kinematics + the toy weight (17),
2×64 hidden; sum-pool (sum, not mean — carries N); ρ: 64→64→2 outputs (μ̂, log σ) — heteroscedastic
Gaussian regression, loss = Gaussian NLL. v1 option (later): output the full 161-point curve.

**Training data:** on-the-fly generator; per sample draw μ_true ~ U[0,4] and L ~ log-U[1,1000],
per-event Poisson toy weights on the train split of the existing 300k-event sample (engine:
`sigstr.scale` + `toys.generate`), keep only events with non-zero toy weight (variable-size set,
padded + masked).

**Evaluation (the point of the exercise):** `notebooks/deepset_coverage.ipynb` — pull distributions
(μ̂−μ_true)/σ and coverage of μ̂±σ intervals across (μ_true, L); v0 success criterion = coverage
consistent with 68% at all four luminosities at μ_true=1 (network reproduces the L-scaling of the
uncertainty, including the fold region's L^(−1/4) behaviour — a sharp test given M2's finding).

**New files (approval requested):**
- `asimov-aymptotics-check/models/deepset.py` (LightningModule)
- `asimov-aymptotics-check/datasets/toyset.py` (IterableDataset toy generator)
- `scripts/deepset_train.sh` + `scripts/deepset_train.sub` (condor GPU, pattern of nsbi_pathB)
- `notebooks/deepset_coverage.ipynb` (evaluation)

GPU cost estimate: a few GPU-hours (generator-bound; num_workers tunable). Items 10–12 start only on
Marta's approval of this file list.

**Status 2026-07-12 — approved and launched (items 10–11):**
- `datasets/toyset.py` + `models/deepset.py` written; local CPU smoke test passed (30 steps, loss
  7.2→~1, scaler + logs written). One design deviation, documented in the eval notebook: ρ also
  receives log L and log1p(N) as auxiliary inputs (L is exactly known in any real analysis) and the
  sum-pool is scaled ×0.01 for optimisation sanity — sum semantics preserved.
- Negative reweighted rates w(μ)·L are clipped at 0 in the generator (same spirit as the upstream
  negative-weight zeroing); the clipped fraction is quantified per μ_true in the eval notebook.
- Training submitted: condor cluster **5074149** (20k steps × 64 sets, μ~U[0,4], L~log-U[1,1000],
  first 200k events; 1 GPU + 8 CPU). Monitor armed (val_loss milestones, tight error signatures).
- `notebooks/deepset_coverage.ipynb` built and syntax-validated — evaluates on the **held-out last
  100k events** with fresh toys (500/point over μ_true × L incl. the fold): coverage, pulls, σ(L)
  scaling vs L^(−1/4)/L^(−1/2). Executes when the checkpoint lands (item 12).

**v0 verdict (2026-07-12, items 11–12 complete):** training clean (20k steps, best val 0.0249);
**coverage criterion FAILED, instructively**. The network learned genuinely luminosity-aware
uncertainty — σ: 1.15→0.39 across L=1→1000 fb⁻¹, matching the classical toy scatter (0.42) at
1000 fb⁻¹, the very thing single-Asimov CARL cannot do — but Gaussian-NLL regression over the
training prior converges to the *posterior under U[0,4]*: at 1 fb⁻¹ it predicts the prior
(μ̂≈1.96, σ≈1.155) for every truth (coverage 0.00 at μ_true=0.5), and at 1000 fb⁻¹ a residual −0.2
bias near the fold drags coverage to 0.65. Classic Bayes-vs-frequentist calibration gap.
**v1 executed (approved 2026-07-12): `notebooks/deepset_neyman.ipynb` — the mitigation works.**
Neyman belts on the deep-set statistic (41 hypotheses × 400 toys per L, held-out events, disjoint
validation streams): **no under-coverage anywhere** (0.67–0.87 vs v0's 0.00–1.00), exact at the
informative points (0.684/0.694 at 1000 fb⁻¹ near the fold), conservative elsewhere (hull
inflation — the safe direction; Feldman–Cousins ordering listed as the refinement). Widths are
honest: ≈ full μ-range at 1 fb⁻¹ (6 events genuinely tell you nothing), ≈ 0.9 at 1000 fb⁻¹.
**Recipe for the write-up: learned L-aware summary statistic + classical toy-based calibration —
immune by construction to both failure mechanisms (cell count/convention, interference fold).**
Prong 3 closed.
