# NSBI reproduction — path B (own CARL training)

> Status: **running log** (task in progress)
> Task: reproduce the NSBI Asimov-vs-median failure with independently trained CARL classifiers
> (path B — Tei's checkpoints not yet delivered), then add the extended Poisson term and sweep
> luminosity. Prep + patch drafts: [nsbi-repro-prep](nsbi-repro-prep.md).
> Started: 2026-07-10
>
> **▶ Canonical artifact (when results land): `notebooks/nsbi_reproduction.ipynb`** — runnable,
> plots inline, same standard as `notebooks/histogram_reproduction.ipynb`.

## Running log

### 2026-07-10 — training submitted to the Nikhef condor GPUs

- Two jobs, cluster **5064754** (`sig_over_bkg`, `sbi_over_bkg`), submitted from
  `scripts/nsbi_pathB_train.sub` + `scripts/nsbi_pathB_train.sh` (both git-tracked).
- Nikhef schedd specifics discovered on submission: `+SingularityImage` (job runs *inside*
  `ml4fp-sbi-tutorial.sif`, condor handles GPU passthrough — no nested apptainer) and
  `+JobCategory = "medium"` are mandatory.
- Config per docs/nsbi-repro-prep.md §1.4 (approved by Marta 2026-07-10): n_layers 3, n_nodes 128,
  lr 1e-3, batch 1024, max 200 epochs (early stopping patience 20), seed 42; full CSVs
  (numerator sig/sbi vs denominator bkg); 1 GPU + 8 CPU + 32 GB per job.
- **Hyperparameters are our choices** — Tei's are unrecoverable (config hunt in the env-setup note),
  so this is an *independent* reproduction of the phenomenon, not a bit-exact one.
- Run layout satisfies `carl.utils.load_results`: everything lands in
  `output/nsbi-pathB/h4l/{sig_over_bkg,sbi_over_bkg}/` (pickled splits, scaler, checkpoints).
- Monitoring: condor state + epoch progress + error signatures watched continuously; logs in
  `output/nsbi-pathB/condor/`.

### 2026-07-10 — first attempt failed fast: container lacks jsonargparse

- Cluster 5064754 matched (both jobs on wn-pijl-003, one GPU each) but died in seconds:
  `ModuleNotFoundError: Requirement 'jsonargparse[signatures]>=4.27.7' not met` — LightningCLI's
  parser dependency is **not in `container.def`** (Tei evidently had it outside the container on
  his cluster). GPU passthrough and /project binding under condor's `+SingularityImage` were fine.
- Fix (same pattern as the step-0 tooling): `pip install --target=output/pyextra
  'jsonargparse[signatures]>=4.27.7'`; wrapper now exports `PYTHONPATH=$ROOT/output/pyextra`.
- Validated before resubmission with `python -m nsbi.carl fit --print_config` (full argument set,
  no training): parses clean, the 16-feature list resolves item-by-item.
- **Resubmitted as cluster 5064851**; monitor re-armed.

### 2026-07-10 — training finished; sbi_over_bkg failed to learn (diagnosed)

Both jobs completed (cluster 5064851, exit 0). Full-precision val_loss (baseline ln2 = 0.693147):

| classifier | best val_loss | vs ln2 | best epoch / stopped | verdict |
|---|---|---|---|---|
| sig_over_bkg | 0.419166 | −0.274 | 170 / ran to ~181 | learned strongly ✅ |
| sbi_over_bkg | 0.692612 | −0.0005 | 11 / early-stopped 31 | at random floor ❌ |

Ratio diagnostic (CPU forward pass on 40k test events, `output/nsbi-pathB/ratio_diagnostic.png`):
- **sig/bkg**: r rises 0.82 → 11.6 across m4ℓ quartiles — correctly places signal in the off-shell
  high-mass tail. Physically sensible, usable.
- **sbi/bkg**: r ≈ 1 flat (0.999 ± 0.05; quartile medians 1.02, 1.01, 1.01, 0.96 — highest-m4ℓ bin
  even dips *below* 1, the wrong sign). The classifier learned essentially nothing.

**Diagnosis: under-training, not a bug.** Separating SBI (≈95 % background at μ=1) from background is a
far harder, low-signal task than sig-vs-bkg; the shared hyperparameters + patience-20 early stopping
cut it off at epoch 16 (sig needed ~170). Setup mechanics are sound (identical pattern worked for
sig). Consequence if used as-is: near-μ=1 the NSBI per-event likelihood would carry almost no shape
information → an insensitive / non-reproducing scan. **sig_over_bkg is a keeper; sbi_over_bkg needs a
retrain with task-appropriate settings** (more capacity, lower LR, longer patience). Ultimate arbiter
remains the NLL scan vs Tei's committed 300 fb⁻¹ reference — but the flat ratio predicts a poor scan.

### 2026-07-10 — sbi_over_bkg retrain submitted + reproduction notebook built

- **Retrain** (cluster 5065281, `scripts/nsbi_pathB_retrain_sbi.sub`): sbi_over_bkg only, beefed up
  for the hard low-signal task — n_layers 3→4, n_nodes 128→256, lr 1e-3→3e-4, early-stop patience
  20→50. sig_over_bkg kept as-is. Failed v1 archived to `h4l/sbi_over_bkg_failed_v1`.
- **Code change** (our fork, approved): `models/carl.py` early-stop patience is now
  `int(os.environ.get("CARL_EARLYSTOP_PATIENCE", 20))` — Tei's default 20 preserved; the retrain sets
  50 via the condor `environment`. Wrapper `nsbi_pathB_train.sh` now takes n_layers/n_nodes/lr as
  optional args (defaults = original). **Sub-repo now has 1 modified file** (previously clean).
- **`notebooks/nsbi_reproduction.ipynb` built** (CPU inference — no GPU needed to run it): loads both
  classifiers → SM ratios → μ-mixture → as-is shape-only scan vs Tei's 300 fb⁻¹ reference → adds the
  extended Poisson term (checks μ̂→1.000, NLL(0)≠0) → luminosity sweep with chunk-free toy NLL. All
  code cells syntax-validated; awaiting the retrained checkpoint to execute.
- Monitor watching the retrain for val_loss dropping below the 0.693 floor (the go/no-go signal).

### 2026-07-10 — retrain STILL flat; reproduction fails; path B blocked without Tei's recipe

- **Beefed-up retrain did not help.** n_layers 4, n_nodes 256, lr 3e-4, patience 50 → best val_loss
  0.69262 over 58 epochs — identical floor to the first attempt. r_sbi/bkg stays flat (mean 0.998,
  std 0.049 on the SBI test split). More capacity / lower LR / more patience does not crack the
  SBI-vs-background separation.
- **Reproduction executed anyway** (`notebooks/nsbi_reproduction.ipynb`, CPU) to see whether the good
  r_sig carries it. It does not:

  | quantity | Tei reference (300 fb⁻¹) | our path B |
  |---|---|---|
  | Asimov μ̂ (shape-only) | 0.920 | 0.080 |
  | Asimov width | 1.267 | 0.175 (~7× too narrow) |
  | Asimov μ̂ (+ Poisson term) | — | 0.120 (must be ≈1.000 if ratios were right) |

  The scan collapses to a razor-thin well near μ=0 (fig `output/nsbi-pathB/asis_vs_reference.png`).
  Decisive tell: **even with the extended Poisson term the fit sits at μ̂=0.12**, whereas the full
  extended likelihood forced μ̂=1.000 exactly in the histogram case — so the NSBI *shape* term (built
  from the flat r_sbi) is actively wrong, not merely missing the rate anchor.
- **Diagnosis:** path B is blocked not by infrastructure but by our inability to reproduce Tei's
  r_sbi network quality without his exact training recipe. r_sig trains fine; r_sbi (the hard,
  ~95%-background separation) does not, across two hyperparameter settings.
- **Value retained:** this is a real, documented finding — the NSBI reproduction is *sensitive to
  r_sbi quality*, and a poorly-trained ratio yields a wildly overconfident, mis-located scan. It also
  sharpens the ask to Tei: his r_sbi/bkg training recipe (epochs, LR schedule, any preprocessing) is
  the missing piece, on top of the checkpoints themselves.

**Decision → path A** (Marta, 2026-07-10). Path-B experiments **parked** (no more training runs). The
reproduction notebook consumes Tei's checkpoints unchanged (same `load_results` layout) — drop them
in `output/nsbi-pathA/h4l/`, point `RUN` at it, execute. Sharpened request sent to Tei (checkpoints
**+ his r_sbi/bkg training recipe and its achieved val_loss floor** — the missing piece path B exposed).
Path-B artifacts kept for the record: `output/nsbi-pathB/` (scans, diagnostics, sig_over_bkg keeper,
failed sbi checkpoints).

### Planned once training lands (order agreed with Tei)

1. `notebooks/nsbi_reproduction.ipynb`: load both checkpoints, CARL ratios on the SBI test split,
   NLL scan **as-is (shape-only)** at 300 fb⁻¹ → qualitative comparison with Tei's committed curves
   (`output/reference_tei/`); direction + width-ratio check (his: median/Asimov = 0.874, μ̂_A = 0.92).
2. Add the extended Poisson term (prep §2). Falsifiable predictions: NLL(0) ≠ 0; Asimov μ̂ → 1.000.
3. Luminosity sweep 1/10/100/1000 fb⁻¹ — the NSBI counterpart of the histogram notebook, expecting
   Asimov ≠ median(toys) at 1–10 fb⁻¹ (chunk-free toy NLL via prep §3).

### 2026-07-16 — from-scratch redo: checkpoints arrived, NSBI reproduced three ways (plan approved by Marta)

*(Note on the name: earlier entries in this file call the author "Tei"; Marta corrected the name
to **Tae** on 2026-07-16 — new text uses Tae.)*

**The checkpoints arrived.** Marta supplied the CERNBox link (token `Shv8zta4oZ1aOL4`); the share
was mirrored to `asimov-aymptotics-check/run/h4l/` (29 files, 7.9 GB, sizes verified; training
splits skipped) by `scripts/asimov_scan.py --download-checkpoints`. Contents per network
(`sig_over_bkg`, `sbi_over_bkg`): 6 checkpoints, `config.yaml`, `hparams.yaml`, `metrics.csv`
(so the hyperparameters previously believed unrecoverable are now on disk: 16 layers × 1024
nodes), the train/val/test splits, the scaler, and `fit-carl-4l.sh`.

**Protocol.** Same redo protocol as the histogram note (this date, there): statistic
t(μ) = −2[lnL(μ) − lnL(μ̂)] per toy; comparison of measured vs predicted (noncentral χ², 1 dof,
noncentrality = Asimov t_A(μ)) quantiles; 2000 toys, one stream per toy; μ ∈ [0, 4] × 161;
11 luminosities 1–3000 fb⁻¹; **event-count term included everywhere**. All scans by
`scripts/asimov_scan.py` on Tae's own test split (576 265 events, Σw = 5.6565 fb = the SBI
cross-section); notebook = `notebooks/nsbi_reproduction.ipynb` (rewritten from scratch).

**Finding 1 — the "ln 2 floor" was never a failed training.** From the exact matrix-element
ratio evaluated on Tae's validation events: the best achievable balanced classification loss for
SBI-vs-BKG is **0.692401**, only 7.5×10⁻⁴ below ln 2 = 0.693147 (exact-ratio ROC area 0.513 —
the two samples are near-indistinguishable event by event). Tae's network reached 0.692582 ≈ 76%
of that achievable margin; our path-B `sig_over_bkg` reached 97% of its own achievable margin
(0.41917 vs 0.41116 optimal). Every "stalled at ln 2" training, his and ours, was in fact close
to optimal — invisible at the 2-decimal precision of checkpoint filenames.

**Finding 2 — the shared `sig_over_bkg` is from the wrong channel (erratum).** Its
`config.yaml` records numerator/denominator = `zz2l2v/ggZZ_{sig,bkg}` with 10 input features
including `met`/`met_phi`, contradicting `fit-carl-4l.sh` in the same share (16 features, 4ℓ
CSVs). It cannot process 4ℓ events, so it cannot be the network behind the committed 300 fb⁻¹
scans, and **a bit-exact reproduction of those scans is impossible with the shared material**.
The nsbi-nn runs therefore use his `sbi_over_bkg` + our near-optimal path-B `sig_over_bkg`
(hybrid, documented in the notebook), plus a variant with the exact r_sig that isolates his
sbi network's contribution.

**Finding 3 — with the exact ratio, there is no dramatic NSBI failure.** The exact-ratio
unbinned analysis (a perfect network) behaves like the histogram: Asimov μ̂ = 1.000 at all 11
luminosities; the median-curve width ratio rises from ~0.96 (30 fb⁻¹) to the asymptotic 1.185
by ~3000 fb⁻¹; coverage dips to ~0.6 at 100–1000 fb⁻¹ (the known interference rate degeneracy,
not an NSBI effect). The slide-7 effect is therefore **not intrinsic to unbinned statistics at
these event counts**.

**Finding 4 — the network version distorts the comparison, in the opposite direction to the
slides.** With the network ratio the Asimov interval narrows spuriously (common-mode sub-percent
errors in ln r × ~1700 events at 300 fb⁻¹) while the toys spread around it: median-curve width
ratio up to 2.09 ± 0.04 at 300 fb⁻¹ (vs 1.17 exact, expectation 1.185), coverage down to 0.53.
Under Tae's own raw-median convention at 300 fb⁻¹ the closest-faithful configuration gives ratio
**1.38 (median wider than Asimov)** against his committed **0.874 (narrower)** — the direction
of his headline flips with working ingredients.

**Finding 5 — the missing event-count term is quantified (erratum).** Without it the statistic
has no normalisation anchor and the shape carries almost no information (ROC 0.513): the
exact-ratio shape-only Asimov best fit moves from **μ̂ ≈ 0** (cross-section-table
normalisation) to **μ̂ = 1.000** (sample normalisation) — a sub-percent convention change moves
the minimum across the entire range. His committed μ̂_A = 0.920 sits inside that fragility. With
the term included the exact-ratio Asimov best fit is 1.000 everywhere, by construction.

**Follow-up check (same day, prompted by Marta).** Suspecting the wrong share had been
downloaded, Tae's *dataset* share (token `AeKNoWslVzwmAdt` — the one the CSVs came from on
09 Jul) was fully listed: 124 entries, training CSVs for both channels (`zz2l2v/`, `zz4l/`,
one `analyzed.csv` per component) plus 2 × 48 observed pseudo-experiment CSVs at μ = 0.4 and
μ = 2.5 — **no checkpoints or run directories anywhere**. Its `zz4l/ggZZ_*/analyzed.csv` are
byte-identical in size and header to the local `data/zz4l/ggzz4l_sm/ggzz4l_*.csv`. Conclusion:
the download was not wrong — the checkpoint share is the only source of checkpoints, and the
2ℓ2ν `sig_over_bkg` is a training-side mix-up on Tae's end (both channels' `analyzed.csv` sit
side by side in his data area; his job was launched with the 2ℓ2ν paths). Since
`fit-carl-4l.sh` + `config.yaml` fully specify his intended 4ℓ recipe (16 features, 16 layers ×
1024 nodes, lr 1e-4, batch 1024, seed 42, sample_size 10⁷, early-stop patience 20) and the 4ℓ
CSVs are already local, the missing network can be trained with his exact recipe — one condor
GPU job — **proposed, awaiting Marta's approval** (retraining is explicitly outside the redo's
approved scope). New material noted for later: the observed_mu0p4 / observed_mu2p5 pseudo-data.

### 2026-07-16 (evening) — sig_over_bkg retrained with Tae's exact recipe (approved); his committed result reproduced and explained

- **Training** (condor cluster 5106594, 1 GPU + 1 CPU, 2.2 h, normal termination): Tae's exact
  4ℓ recipe — 16 features, 16 layers × 1024 nodes, lr 10⁻⁴, batch 1024, seed 42, early-stop
  patience 20 (max 500 epochs; stopped at 60). Output `output/nsbi-pathB/h4l/sig_over_bkg_taerecipe/`.
  One documented no-op deviation: his config's `sample_size: 10⁷` cap (our datamodule version has
  no such argument; the CSVs hold fewer rows). New optional `max_epochs` argument added to
  `scripts/nsbi_pathB_train.sh` (default unchanged).
- **Best val_loss 0.419194 — statistically identical to the 3×128 network's 0.419166** (both 97%
  of the achievable margin 0.411164): on this task the architecture is irrelevant to the loss,
  and the remaining margin is not a capacity limit.
- **But the two same-loss networks distort the analysis in qualitatively different ways** (full
  sweeps, 2000 toys, same events/toys): the Tae-recipe network biases every best fit low (Asimov
  μ̂ = 0.875 at all luminosities; fraction of toys whose 68% interval contains μ = 1 falls
  monotonically to 0.42 at 3000 fb⁻¹), while the 3×128 network biases high (μ̂ = 1.025) and
  instead inflates the toy spread (median-curve width ratio up to 2.09 ± 0.04 at 300 fb⁻¹).
  Both are kept in the notebook (§5–§6) as the sharpest demonstration that the discrepancy is a
  property of the particular trained network, not of its loss and not of the event count.
- **Tae's committed 300 fb⁻¹ result is now closely reproduced — by his architecture only.**
  Under his exact statistic/convention/toys, his `sbi_over_bkg` + the Tae-recipe `sig_over_bkg`
  give median/Asimov width ratio **0.897** (his committed: **0.874**; Asimov widths 1.30 vs 1.27),
  and shape-only Asimov μ̂ = 1.000. The same configuration with the same-loss 3×128 network gives
  **1.38** — opposite direction. Conclusion for the finding-4 entry above: the direction measured
  there was the 3×128 accident; the slide-7 effect follows Tae's particular network, and neither
  direction reflects unbinned statistics (the exact ratio shows none of it).
- Notebook regenerated and re-executed with the fifth scan set (`nsbi-nn (3×128 sig)`); §§2, 6, 7
  and the errata updated accordingly.

**Files.** Engine `scripts/asimov_scan.py` (also the downloader); scans under
`output/asimov_scan/{nsbi_exact_testsplit,nsbi_nn,nsbi_nn_sigexact,nsbi_nn_anchor300,`
`nsbi_exact_shapeonly300,nsbi_exact_shapeonly300_samplenorm}/` (gitignored);
notebook `notebooks/nsbi_reproduction.ipynb` (definitions → network verdict → exact vs nn
measurements → decomposition → Tae comparison → errata → reproduce). Checkpoints in
`asimov-aymptotics-check/run/h4l/` (gitignored `run/`).
