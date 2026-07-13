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
