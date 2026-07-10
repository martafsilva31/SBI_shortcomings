# NSBI reproduction prep — path B launch kit + patch drafts

> Status: **prep note** — nothing in here has been run or applied; it exists so NSBI reproduction
> is instant the moment it unblocks (Tei's files for path A, or a GPU session for path B).
> Covers plan tasks: path B launch kit, extended-Poisson patch draft, chunk-over-toys patch draft.
> Started: 2026-07-10. Sources: code walkthrough (docs/env-setup-and-code-walkthrough.md) and
> direct reads of `nsbi/carl/`, `models/carl.py`, `datasets/balanced.py`, the NSBI notebook.

## 1. Path B launch kit — train our own CARL pair

### 1.1 Context

Path A (Tei's checkpoints) gives a bit-exact reproduction and is inference-only. Path B trains our
own two classifiers (`sig_over_bkg`, `sbi_over_bkg`) with sensible hyperparameters — it cannot match
Tei's curves bit-for-bit (his hyperparameters are unrecoverable) but reproduces the *phenomenon*
independently, which is arguably the stronger statement. Both need a GPU node.

### 1.2 What the CLI actually accepts (read from code, not guessed)

- Entry point `nsbi/carl/__main__.py` = LightningCLI over `CARL` + `BalancedDataModule`, with two
  links: `model.n_features` ← `len(data.features)` (never pass it) and `data.random_state` ←
  `seed_everything`.
- `CARL.__init__` (`models/carl.py`): `--model.n_layers`, `--model.n_nodes`,
  `--model.learning_rate` — **all REQUIRED, no defaults**. Optimiser NAdam +
  ReduceLROnPlateau(factor=0.1, patience=5) on val_loss.
- Callbacks are **hard-coded** in `CARL.configure_callbacks()`: ModelCheckpoint(top-5 on val_loss,
  `filename="{epoch:02d}-{val_loss:.2f}"`), ModelCheckpoint(top-1 on train_loss), EarlyStopping
  (patience 20 on val_loss).
- `BalancedDataModule` (`datasets/balanced.py`): `--data.numerator_events` / `--data.denominator_events`
  (CSV paths, glob-expanded), `--data.features` (list; **default is a 9-var angular list whose
  `cth_star, cth_1, cth_2, phi_1, phi` are NOT CSV columns → KeyError** — always pass the 16
  lepton features explicitly), `--data.batch_size` (default 32), `--data.data_dir` (default `./`,
  where `scaler.pkl` + `events_{numerator,denominator}_{train,val,test}.pkl` land — **the code does
  not create it; mkdir first**), optional `--data.{numerator,denominator}_reweight` msq-reweight
  tuples (not needed: dedicated sig/bkg CSVs exist). Split 6:2:2 sequential, hard-coded;
  class-balanced weights (each side normalised to Σw = 1, shape-only); `num_workers=8` hard-coded;
  **no `n_rows` exposure → full CSVs are read** (sig 1.52M + bkg 2.83M rows for classifier 1).
- Packaging trap: `pyproject.toml` installs only `physics` and `stats` — `nsbi`, `models`,
  `datasets` resolve **only with cwd = sub-repo root** (or that path on `PYTHONPATH`).

### 1.3 The `load_results` layout contract (must be satisfied by training-time paths)

`carl.utils.load_results(run_dir, run_name)` (`nsbi/carl/utils.py:11-46`) requires
`<run_dir>/<run_name>/` to contain the four `events_*_{val,test}.pkl`, `scaler.pkl`, and
`lightning_logs/version_*/checkpoints/epoch=*-val_loss=*.ckpt`; it picks the **latest** `version_N`
and, of the top-5 val_loss checkpoints, the **largest-epoch** one (quirk — preserve it when
comparing anything). Consequence: `--data.data_dir` **and** the CSVLogger `save_dir` must both
point at `<run_dir>/<run_name>`. Keep the run names exactly `sig_over_bkg` / `sbi_over_bkg` so the
analysis notebook works with a single `run_dir =` line changed.

### 1.4 Launch commands (draft — do not run without approval + GPU node)

```bash
cd /project/atlas/Users/mfernand/SBI_shortcomings/asimov-aymptotics-check
RUN=/project/atlas/Users/mfernand/SBI_shortcomings/output/nsbi-pathB/h4l   # gitignored
DATA=/project/atlas/Users/mfernand/SBI_shortcomings/data/zz4l/ggzz4l_sm
FEATS="[l1_pt,l1_eta,l1_phi,l1_energy,l2_pt,l2_eta,l2_phi,l2_energy,l3_pt,l3_eta,l3_phi,l3_energy,l4_pt,l4_eta,l4_phi,l4_energy]"
mkdir -p $RUN/sig_over_bkg $RUN/sbi_over_bkg

# inside: apptainer shell --nv ml4fp-sbi-tutorial.sif   (with PYTHONNOUSERSITE=1)
for PAIR in "sig_over_bkg $DATA/ggzz4l_sig.csv" "sbi_over_bkg $DATA/ggzz4l_sbi.csv"; do
  set -- $PAIR
  python -m nsbi.carl fit \
    --seed_everything 42 \
    --model.n_layers 3 --model.n_nodes 128 --model.learning_rate 1e-3 \
    --data.numerator_events "$2" \
    --data.denominator_events "$DATA/ggzz4l_bkg.csv" \
    --data.features "$FEATS" \
    --data.batch_size 1024 \
    --data.data_dir $RUN/$1 \
    --trainer.accelerator gpu --trainer.devices 1 \
    --trainer.max_epochs 200 \
    --trainer.logger.init_args.save_dir $RUN/$1
done
# afterwards, in the NSBI notebook (a copy in output/): run_dir = '<RUN>' and everything else works
```

Read-from-code vs guessed:

| Setting | Status |
|---|---|
| Required arg names, feature list, layout contract, split, callbacks, num_workers | read from code |
| Run names `sig_over_bkg`/`sbi_over_bkg`; numerator = the sig/sbi CSV, denominator = bkg CSV | read from notebook usage; dedicated-CSV choice assumed (msq-reweighting also possible) |
| n_layers=3, n_nodes=128, lr=1e-3, max_epochs=200, seed=42 | **guess** (sensible defaults; Tei's values unrecoverable) |
| batch_size=1024 | **guess** (default 32 → ~135k steps/epoch on 4.3M rows; 1024 matches the notebook's inference loaders) |
| `--trainer.logger.init_args.save_dir` override syntax | standard jsonargparse form, **untested here** |

## 2. Extended-Poisson-term patch draft (apply only AFTER as-is reproduction)

### 2.1 What is missing and where

The NSBI NLL (notebook cell `e9ffa5e9`) is shape-only:

```python
nll_scan = -2 * torch.sum( lumi * torch.tensor(events_sbi_test.weights)[:,None] * torch.log(r_sbi_over_bkg_mu), dim=0)
```

The extended (total-rate) term `−2 (N_obs log ν_tot(μ) − ν_tot(μ))` is absent; the μ-dependent total
cross-section is already computed as the *denominator* of the mixture in cell `82fdbe3f`
(`xs_sig_sm * poi_scan + xs_int_sm * torch.sqrt(poi_scan) + xs_bkg_sm`), so `ν_tot(μ)` is just
`lumi ×` that expression. Subtlety from the walkthrough: part of the −N log ν behaviour enters
implicitly through the ratio's denominator; the **linear +ν(μ) piece is entirely absent** — hence
NLL(μ=0) = 0 exactly in the committed scan.

### 2.2 The patch (draft)

```python
# --- new lines, after the mixture cell 82fdbe3f ---
xs_tot_mu = xs_sig_sm * poi_scan + xs_int_sm * torch.sqrt(poi_scan) + xs_bkg_sm   # (101,)
nu_tot_mu = lumi * xs_tot_mu                                                      # (101,)

# --- Asimov NLL (replaces cell e9ffa5e9) ---
w_test  = torch.tensor(events_sbi_test.weights)          # per-event xs weights [fb]
N_obs   = lumi * w_test.sum()                            # Asimov observed total
shape   = torch.sum(lumi * w_test[:,None] * torch.log(r_sbi_over_bkg_mu), dim=0)  # (101,)
nll_scan = -2 * ( shape + N_obs * torch.log(nu_tot_mu) - nu_tot_mu )

# --- toy NLL (replaces cell 4ca22ecc; uses the chunk-free form of §3) ---
N_obs_toys    = nu_toys.sum(dim=0)                                                # (200,)
shape_toys    = torch.log(r_sbi_over_bkg_mu).to(torch.float64).T @ nu_toys        # (101, 200)
nll_scan_toys = -2 * ( shape_toys
                       + N_obs_toys[None,:] * torch.log(nu_tot_mu)[:,None].to(torch.float64)
                       - nu_tot_mu[:,None].to(torch.float64) )
```

μ-independent constants (log N_obs! terms) are dropped — they cancel in ΔNLL. Output names and
shapes (`nll_scan` (101,), `nll_scan_toys` (101, 200), saved as `nll_scan_asimov.pt` /
`nll_scan_toys.pt` / `poi_scan.pt`) are unchanged, so the compare notebook keeps working.

### 2.3 Predictions to check when it runs (falsifiable)

- NLL(μ=0) ≠ 0 after the patch (the committed scan has exactly 0 there).
- The NSBI **Asimov μ̂ should move from 0.920 to 1.000** — the histogram runs showed the full
  extended likelihood recovers the injected μ exactly at every luminosity; the 0.920 offset is
  consistent with the missing linear +ν(μ) piece tilting the shape-only minimum.
- Recompute the 300 fb⁻¹ width ratio (median/Asimov = 0.874 pre-patch) — Tei expects convergence
  to be driven by the rate term at high lumi, so the ratio should move toward 1 at 1000 fb⁻¹.

## 3. Chunk-over-toys patch draft (needed before any toy scan on a normal node)

Cell `4ca22ecc` as committed materialises the full broadcast before reducing:

```python
nll_scan_toys = -2 * torch.sum( nu_toys[:,None,:] * torch.log(r_sbi_over_bkg_mu)[:,:,None], dim=0)
```

Intermediate tensor (576265, 101, 200) float64 ≈ **93.1 GB** → OOM anywhere but a fat node. The sum
over events is a matrix product, so the giant intermediate is unnecessary:

```python
nll_scan_toys = -2 * ( torch.log(r_sbi_over_bkg_mu).to(torch.float64).T @ nu_toys )   # (101,200)
```

Peak extra memory = the float64 log-ratio copy, 576265 × 101 × 8 B ≈ 0.47 GB. Result is identical
(same reduction, float64 accumulation as before — the committed version promoted to float64 via the
toys). If even ~0.5 GB is tight, the fallback is a loop over toy slices `nu_toys[:, t0:t1]`
accumulating into a preallocated (101, 200) output. Shapes and filenames unchanged.

## 4. Order of operations when unblocked

1. Path A files arrive → copy under `output/nsbi-pathA/h4l/`, run a **copy** of the NSBI notebook
   (cwd in `output/`, `run_dir` pointed at the copy) **as-is** on the GPU node → must match the
   committed `.pt` triple bit-for-bit (same test split, same largest-epoch checkpoint quirk).
2. Apply §3 (chunking — pure refactor, results identical) if toy scans are needed on small nodes.
3. Apply §2 (Poisson term), re-run Asimov + toys at 300 fb⁻¹, check §2.3 predictions.
4. Sweep lumi ∈ {1, 10, 100, 1000} in μ mode → the NSBI counterpart of docs/hist-reproduction.md.
5. Path B (§1) in parallel or as fallback — same checks, plus hyperparameter sensitivity note.
