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

### Prong 2 executed (items 8a–8c, 2026-07-13) — BB-lite in everywhere, conclusions unchanged

Decision (Marta, 2026-07-13): MC-stat uncertainties in **every** scenario, overriding the
document-and-skip recommendation. Implemented as Barlow–Beeston-lite with closed-form per-bin
profiling γ̂_b = (n_b+τ_b)/(ν_b(μ)+τ_b), τ_b = n_eff,b (`stats/binned.py`, new
`poisson_log_likelihood_scan[_toys]_bblite`; constraint written relative to γ=1 for float32
stability; second modified file in the sub-repo). No minimiser needed — the scans stay fully
vectorised. Exactness checks pass: n_eff→∞ reproduces the plain scan; Asimov μ̂ = 1.000 at all
32 grid points.

- **32-point overlay** (dimensionality notebook §11, Δ-convention metrics, paired on the same
  toys): Asimov 1σ widths grow ≤0.2% at 100 fb⁻¹ and 1.8–2.6% at 1000 fb⁻¹; ratio_D shifts by
  ≤0.013. Even at 4D (51% median per-bin MC error) the *data* statistical error per bin dominates
  — BB-lite alters no conclusion of the dimensionality study.
- **Coverage spot-check** (bartlett notebook §10, paired 1000 fresh toys per point): naive 68%
  coverage hist_1d@1000 0.594→0.608 (±0.016) and occ_4d@1000 0.644→0.649 with BB-lite; at
  100 fb⁻¹ no change (0.717→0.719, 0.705→0.706). E[t(1)] drops only 1.19→1.15 / 1.07→1.05. The
  under-coverage **persists with MC-stat nuisances profiled — confirming the audit's fold
  attribution** (the falsifiable expectation held).
- Same run: the dimensionality metrics table is now Δ-primary + np-raw secondary (provisional
  convention, see Next block); at 1000 fb⁻¹ ratio_D = 1.06–1.10 flat across 1D–4D, i.e. no
  dimensionality effect under the primary convention, as the item-2/4 decomposition predicted.
- Both notebooks re-executed end-to-end in the container (headless nbconvert via the
  `output/pyextra` toolchain); pre-existing §8.1 numbers reproduced identically (seed-stable).

### Validation re-run (plan item 14, 2026-07-13) — closed

All Phase-3 notebooks re-executed end-to-end in-container the same day: bartlett + dimensionality
(with the Δ/BB-lite updates, above), then rate_degeneracy, deepset_coverage, deepset_neyman
(CPU; checkpoints loaded map_location='cpu'). Documented numbers reproduce: per-bin fold range
[0.46, 6.7] and 0.0% rate Fisher share with ×23 κ-suppression (M2); v0 prior collapse (coverage
0.000 at μ_true=0.5 @1 fb⁻¹, σ 1.15→0.29–0.39 across L) and the −0.2 fold bias at 1000 fb⁻¹
(coverage 0.642/0.646); v1 Neyman no-under-coverage 0.668–0.866 with the fold points 0.684/0.694
@1000 fb⁻¹ byte-identical to the 2026-07-12 run. Remaining open in this phase: items 7 (Marta),
13 and 8d (blocked on Tei), 15 (gated on format sign-off).

### 2026-07-13 — the "Wi-Fi ensemble" is identified: wᵢfᵢ ensembles (item 7 prep; unblocks 8d's design)

The April meeting's *"new Wi-Fi ensemble method"* — flagged in foundations §15/§18 as a garbled
auto-transcription needing team confirmation — **is not garbled**. It is **wᵢfᵢ ("wifi") ensembles**,
Benevedes & Thaler, arXiv:2506.00113 (v2, hep-ph, MIT-CTP/5874) — a pun on weights **wᵢ** × basis
functions **fᵢ**. PDF downloaded to `docs/refs/`, title page checked, written up as **Ref. 17** in
literature.md and as **foundations §15.1**. Tei's confirmation is now a courtesy, not a dependency.

**Method in one paragraph.** Train M networks fᵢ(x) (bootstrap resamples or a partition of the
training set), *freeze* them, and treat them as a **basis**, not as M competing answers: model
`log r̃(x|w) = Σᵢ wᵢ fᵢ(x)` and fit only the M weights on a **held-out** sample with a convex
(symmetrised MLC) loss. Linearity in w buys closed forms: sandwich covariance `C = V⁻¹UV⁻¹`, pointwise
ratio uncertainty `σ²(x) = fᵢ(x)Cᵢⱼfⱼ(x)`, and Gong–Samaniego propagation to the POI,
`σ²_GS = σ²_MLE(1 + σ²_MLE AᵢCᵢⱼAⱼ)`. That is why MC-stat uncertainties are *"trivial to apply"*:
**no bootstrap loop around the whole analysis** — it is an analytic *inflation* of σ²_MLE.

**Two boundaries, both load-bearing for us:**
1. **Scope.** C ~ O(1/N_fit) is the noise of the finite **fitting** sample. The paper is explicit that
   it does *not* cover mismodelling from network expressivity, training stochasticity, finite
   *training* data, or simulator systematics. It closes one term of condition (C), not the condition.
2. **Regime.** Every guarantee is **asymptotic** (fixed M, N_fit → ∞; Gong–Samaniego is a delta-method
   inflation presupposing the Wald picture). Our failure mode *is* the breakdown of asymptotics at
   O(100) events — so wifi is a **candidate mitigation to be tested, not a fix to be assumed**.

**Bonus finding, relevant to prong 3 and to §13.2:** the paper shows the **naive uniform-weight
ensemble undercovers and is biased**, while the fitted-weight version reaches nominal coverage (M=32,
quark/gluon). Published support for a claim this project keeps making: ensembling-as-averaging is not
a calibration fix.

**Consequence for item 8d (NSBI MC-stat design).** The network-side route is now named: template-side
= BB-lite on the rate/shape inputs (done for the histogram family, items 8a–8c); network-side = wifi
ensembles on the CARL ratios. **Execution stays blocked on path A** — wifi needs M *working* basis
networks and our path-B r_sbi/bkg never left the ln2 floor. But the design is no longer blocked on the
method's *identity*, and there is an unblocked prototype route (below) that Marta has not yet approved.

**Proposed (NOT started — needs Marta's approval): prototype wifi on the ratio that does train.**
Path B's `sig_over_bkg` is a keeper (val_loss 0.419, r rising 0.82 → 11.6 across m4ℓ quartiles); only
`sbi_over_bkg` failed. Training M ≈ 8–32 bootstrap replicas of `sig_over_bkg` and running the wifi
machinery on it would build and validate the whole apparatus (basis freeze → weight fit → sandwich C →
Gong–Samaniego on μ) against a ratio we trust, so that when Tei's checkpoints land the method is ready
to apply rather than being written from scratch. It would also answer the sharp question of foundations
§17.3 in a controlled setting: *does an asymptotically-derived inflation restore coverage where
asymptotics fail?* Cost: M GPU trainings (each ~ the 2026-07-10 run) + one notebook.

### 2026-07-13 — item 8e: wifi prototype on `sig_over_bkg` (APPROVED by Marta; running)

**What is being inferred, and what is not.** Not μ. The NSBI μ-scan needs `r_sbi/bkg`, which path B never
learned (ln2 floor) and which path A is blocked on. So the prototype uses the ratio that *does* train,
`sig_over_bkg`, and infers the **mixture fraction κ** of signal in a sig+bkg mixture — which is exactly the
task of Ref. 17's own LHC demonstration (their quark fraction), via their Eq. (27),
`log L(x|κ) = log(κ r(x) + (1-κ))`. This is a **methodology testbed, not a physics measurement**: κ is a
mixture fraction, μ is a signal strength with interference, and they are not the same parameter. What
transfers is the machinery and the answer to the §17.3 question.

**The control Benevedes & Thaler could not build.** Our events carry per-event matrix elements, so the
**truth** ratio is known analytically: r_true(x) ∝ msq_sig(x)/msq_bkg(x), normalised so E_bkg[r_true] = 1
(their Eq. 6). They had no truth ratio for jets. With it we can separate ratio **noise** (which wifi
propagates) from ratio **bias** (which it explicitly does not) — the two halves of condition (C).

**Implementation (2026-07-13).**
- `Process.resample()` — implemented the sub-repo's own **"For wifi ensembles. TODO: IMPLEMENT ME"** stub in
  `physics/simulation/mcfm.py`. Its pre-existence is independent corroboration of the identification: the
  method was already on this codebase's roadmap. Poisson bootstrap of a weighted sample (weight × Poisson(1),
  cross-section renormalised). Verified: 37.3% zero-weight events (e⁻¹ = 36.8% ✓), Σw preserved,
  seed-reproducible, kinematics untouched, n_eff → 0.58 × original.
- `datasets/balanced.py` — bootstrap wired into the **training split only** (env `CARL_BOOTSTRAP_SEED`;
  unset = original behaviour). Validation deliberately NOT resampled, so early stopping stays comparable
  across members. Also `CARL_NUM_WORKERS` (default 8 = unchanged) — see the condor note below.
- `scripts/wifi_train_basis.{sh,sub}` — M = 16 basis nets, one GPU job each, proven path-B recipe unchanged
  (n_layers 3, n_nodes 128, lr 1e-3); **initialisation held fixed at seed 42 on purpose**, so ensemble
  diversity comes from the bootstrap (what wifi propagates) and not from init noise (what Ref. 16 measures).
- `scripts/wifi_fit.py` — the engine: symmetrised MLC loss (Eq. 8) minimised by Newton with backtracking,
  sandwich covariance C = V⁻¹UV⁻¹ (Eqs. 11/18), σ²(x) = fᵢCᵢⱼfⱼ (Eq. 20), κ̂ by bisection (boundary-aware),
  and σ²_GS = σ²_MLE(1 + σ²_MLE AᵢCᵢⱼAⱼ) with A analytic (Eqs. 22–24; A_i = Σ ρ fᵢ/g²).
- `notebooks/wifi_ensembles.ipynb` — narrative + figures (imports the engine, does not reimplement it).

**Engine closure test — PASSED** (n = N(1,1), d = N(0,1), basis [x, 1] ⇒ well-specified with w* = (1, −½)):
ŵ = (1.0008, −0.4972) at N=2k and (1.0034, −0.5035) at N=200k; pulls O(1) (0.02–1.16), so the sandwich
covariance is honest; σ_w = 0.0426 → 0.0138 → 0.0044 per decade, i.e. ratios 3.09 / 3.14 against √10 = 3.16
— **C ~ O(1/N) confirmed exactly as Ref. 17 states**. κ inference unbiased (bias ≤ 0.009), coverage 0.650
(400 trials, ±0.024). The machinery is correct independently of any network.

**Data-path validation — PASSED (before spending the GPU time).**
- *Datamodule*: the bootstrap fires on the **training split only** (36.8% zero-weight events, exactly e⁻¹);
  the no-bootstrap control has 0.0% zeros; batches finite; `CARL_NUM_WORKERS=0` respected.
- *Truth ratio*: E_bkg[r_true] = 1.000000 **by construction**, but the *independent* second constraint of
  Ref. 17 Eq. (6) — E_sig[1/r_true], which is **not** enforced anywhere — comes out **0.989**, i.e. right to
  1%. That is a real check that the matrix-element ratio is a genuine density ratio, not a normalisation
  artefact. Physics sanity: median r_true rises 0.138 → 0.156 → 0.218 → 1.114 across m4ℓ quartiles — the
  signal sits in the off-shell high-mass tail, as it must.

**Condor note (cost us two resubmissions).** The first submission (cluster 5086708, 8 CPUs/job) matched
**zero** slots: the Nikhef GPU nodes had free GPUs but **zero free CPUs**, and `condor_q -better-analyze`
showed the "≥8 CPUs" and "≥1 GPU" conditions each matching 5 slots but never *the same* slot. 2 CPUs (cluster
5086738) still did not match. **Cluster 5086805** = 1 CPU + `CARL_NUM_WORKERS=0` (the dataset is already an
in-memory numpy array, so worker processes buy nothing and cost a core) — matched immediately, 5 running
within minutes. GPU-node CPU contention, not a fault in the setup. Results and verdict to follow in the
notebook.

### 2026-07-13 — item-7 message to Tei: DRAFT (Marta sends; parked while he is unresponsive)

Kept here so it survives outside chat. Four asks, none of them blocking any more; every number traces
to this note, dimensionality-study.md, or env-setup §8 Q3.

> Hi Tei,
>
> A few updates and asks from the shortcomings study — nothing needs code from you, just confirmations.
>
> **1. The high-luminosity coverage failure is an interference rate fold, not occupancy.** The destructive interference makes the total rate stationary in μ at the SM: dν_tot/dμ at μ=1 is σ_S + σ_I/2 ≈ 0.0067 fb (×46 suppressed), with ν_tot minimised at μ* = (σ_I/2σ_S)² ≈ 0.957 — a two-to-one map in μ right at the SM. Confirmed signatures on the histogram analysis: σ(μ̂) scales like L^(−1/4) (measured 0.70 → 0.42, ratio 0.60 vs 0.56 predicted), μ̂ biased low (mean 0.86 at 1000 fb⁻¹), var[t(1)] ≈ 3.2 vs the χ²₁ value of 2, and the miscoverage is localised at the fold: 68% intervals cover ~0.60 at μ_true ∈ {0.957, 1} at 1000 fb⁻¹, recovering to 0.65–0.71 at {0.5, 2}. Per-bin folds span μ*_b ∈ [0.46, 6.7], so shape partially rescues the global fold, but the rate term carries 0.0% of the Fisher information at μ=1. The effect grows with luminosity, is binning-independent, and lives in the μ-parametrisation — so it applies identically to the NSBI setup. A κ=√μ reparametrisation provably cannot help (the likelihood ratio is invariant under it). Bartlett correction is only a partial patch: coverage 0.585 → 0.641 against 0.683 nominal.
>
> **2. Median convention — please ratify.** The Asimov-vs-median width ratio is convention-dependent. Decomposed on identical scan tensors (4D histogram, 1000 fb⁻¹): **0.44** with torch.median of raw curves (torch takes the lower middle element at even toy counts), **0.71** with np.median of raw curves, **1.10** with the median of per-toy min-subtracted (Δ) curves — i.e. no dimensionality effect at all under the Δ convention. We provisionally adopted on 2026-07-13: Δ curves as the primary comparator, np.median-of-raw reported alongside as an estimator-variance observation, torch.median retired. Your 300 fb⁻¹ NSBI direction check is robust under all three conventions (0.87). OK to make this the team convention?
>
> **3. "wifi ensemble" method — found it, just confirming.** We believe the April-meeting method is Benevedes & Thaler, "Frequentist Uncertainties on Neural Density Ratios with wᵢfᵢ Ensembles" (arXiv:2506.00113) — frozen basis networks with fitted weights and an analytic sandwich covariance, propagated to the POI by Gong–Samaniego, no bootstraps. Is that the one? We have written it into our notes on the assumption that it is.
>
> **4. The [2, 1.83] cross-section prefactors.** Our reading: 2 = channel counting (distinct-flavour 2e2μ → all e/μ 4ℓ channels), 1.83 = a flat higher-order QCD k-factor applied commonly to SIG/INT/BKG. Can you confirm, and give the exact provenance of the 1.83?
>
> (And whenever the run/h4l checkpoints plus your r_sbi/bkg training recipe are ready, path A is set up to consume them unchanged.)
>
> Thanks!
> Marta

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

### 2026-07-14 — item 8e RESULTS: the wifi verdict (after an f₀ protocol fix caught by review)

**Basis training landed overnight.** Cluster 5086805 (the 1-CPU resubmit): 16/16 jobs exit 0 by 04:22.
True best val_loss per member (read from the ModelCheckpoint callback state — the 2-dp filenames all say
"0.42") spans **0.41928–0.42331** (ln 2 = 0.693): every member learned, and the spread is real (best epochs
range 15–199 despite the shared init — the bootstrap alone diversifies the trainings). Median per-event
basis spread 0.170 in log-ratio units.

**Run 1 (no f₀) and the review that killed its headline.** The first full fit (~5 min CPU on stbc-i2;
E_bkg[r_true] = 1.000000, saturation 0.000%, Newton 3–7 iterations) showed wifi coverage collapsing to
**0.35–0.59 at N_mix = 10⁴** with GS inflation ≤ 1.024 — one section away from a "bias dominates, wifi
fails" write-up. A 23-agent adversarial review of the engine (lanes: paper-math, statistical design, output
contract; each finding attacked by two independent refuters) confirmed three defects first:

1. **[major] Missing f₀(x) = 1.** Ref. 17 Sec. III.A includes the constant basis function in *all* wifi
   experiments — its weight w₀ is what lets the fitted ratio satisfy the Eq. (6) normalisation. Our engine
   fitted only the M logits (the constant is not in their span: R² = 0.22), so a common normalisation offset
   of the basis (measured: E_d[e^u] = 0.994 vs E_n[e^−u] = 1.013) survived as an unabsorbable κ bias.
2. **[major] The notebook's GS-scaling narrative was backwards.** σ²_MLE·ACA ~ N_mix/N_fit **grows** with
   N_mix (A is a sum over trial events); "inflation largest at small N" contradicted the engine's own
   output. Corrected in notebook §6 and promoted to the structural half of the verdict.
3. **[minor] Filename-parsed val_losses in notebook §2** (all quantized to "0.42"); the engine now exports
   the true scores (`member_val_loss` in the results JSON).

Ten further attacks were refuted **with measurements**: shared-200k-pool trial correlation ≤ 1.7 pp,
common-mode incl. the truth row; frozen-ŵ/C conditional coverage second-order (≤ few pp, N = 10⁴ only —
the paper refits per trial, we do not); source-ESS effect on C; the shared init seed (bootstrap diversity
is real). Equation-level fidelity of loss/scores/Hessian/sandwich/κ-MLE/GS to the paper: clean.

**Run 2 (with f₀): run 1's collapse was the protocol deviation, not wifi.** w₀ = +0.039…+0.043 across
M ∈ {4, 8, 16}; Eq. (6) stationarity exact (E_n[e^−u] = E_d[e^u] = 0.998–0.999). At N_mix = 10⁴, wifi(GS)
coverage is **0.64–0.72 across all κ and M** (nominal 0.683, MC error ±0.021) vs 0.35–0.59 in run 1;
κ̂ bias **+0.001…+0.003** (was −0.005…−0.007), compatible with the truth control (±0.002). The RMS
log-ratio error is **unchanged** (0.352 ≈ 15× the propagated noise 0.023): f₀ removed only the
normalisation component of the bias — precisely the component κ inference feels. The **naive ensemble**
(uniform weights, no f₀ — the paper's own Naive protocol) still collapses: 0.28–0.63 at N = 10⁴,
κ̂ bias −0.011 at κ = 0.5. Ref. 17's central claim, confirmed on physics data.

**Verdict (answers foundations §17.3).**
- **Structurally: GS cannot act at small N_mix.** Relative inflation ~ N_mix/N_fit — measured 1.0001 at
  N = 30, max 1.037 at N = 10⁴. Whenever the ratio is fitted on more events than the measurement uses —
  the universal NSBI situation — the inflation is invisible in the non-asymptotic regime *by construction*.
  **Hypothesis 3 (sample-size blindness) survives its strongest challenger.**
- **Empirically: on this rate-style task there was nothing at small N to fix** (truth-ratio coverage nominal
  at every N_mix; the κ = 0.05/N = 30 overcoverage is the Wald-at-boundary artefact — ~⅓ of fits pinned at
  κ̂ = 0 — common to all methods).
- **wifi + f₀ is validated as MC-stat *propagation*** on a ratio that trains: honest sandwich covariance,
  load-bearing Eq. (6). It enters the 8d design as the network-side treatment — but it is not, and cannot
  be, a mitigation of the Asimov ≠ median(toys) failure.
- Lessons: Eq. (6)/f₀ is not a convenience; and pointwise ratio bias ≠ parameter bias — what matters is the
  projection of the ratio error onto the parameter's score.

**Files.** Engine `scripts/wifi_fit.py` (f₀ basis, Eq. (6) printout, val_loss export); results
`output/wifi/results/{wifi_results.json, coverage.csv, diagnostics.npz, fit.log}` (run 2; run 1's log
preserved as `fit_run1_no_f0.log`); narrative `notebooks/wifi_ensembles.ipynb` (fully executed, 5 figures,
§7 verdict).

⚠ **For the item-7 Tei message:** §3 can be upgraded from "found it, just confirming" to *validated on our
data* (κ-mixture task; coverage nominal once the paper's f₀ term is included) — one-line addendum available
when Marta sends it.
