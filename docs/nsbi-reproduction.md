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

### Planned once training lands (order agreed with Tei)

1. `notebooks/nsbi_reproduction.ipynb`: load both checkpoints, CARL ratios on the SBI test split,
   NLL scan **as-is (shape-only)** at 300 fb⁻¹ → qualitative comparison with Tei's committed curves
   (`output/reference_tei/`); direction + width-ratio check (his: median/Asimov = 0.874, μ̂_A = 0.92).
2. Add the extended Poisson term (prep §2). Falsifiable predictions: NLL(0) ≠ 0; Asimov μ̂ → 1.000.
3. Luminosity sweep 1/10/100/1000 fb⁻¹ — the NSBI counterpart of the histogram notebook, expecting
   Asimov ≠ median(toys) at 1–10 fb⁻¹ (chunk-free toy NLL via prep §3).
