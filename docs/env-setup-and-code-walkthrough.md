# Environment setup + code walkthrough

> Status: **running log** (task in progress — will be cleaned into a summary study-note)
> Task: get the project runnable (container + dataset) and understand Tei's code before reproducing his results.
> Started: 2026-07-09

## Running log

### 2026-07-09 — environment survey

- Sub-repo `asimov-aymptotics-check/` already cloned (2026-06-02) with real code AND Tei's committed
  reference results at repo root: `nll_scan_asimov.pt`, `nll_scan_toys.pt`, `mu_scan.pt`, `poi_scan.pt`
  plus plots (`with_int.png`, `without_int.png`, `theta_lin.png`, `theta_quad.png`). These are the
  ground truth to reproduce against.
- Node check: Apptainer available (`/bin/apptainer`); **no GPU** on this node (`nvidia-smi` absent);
  **EOS not mounted** — local dataset copy needed.
- `datasets/` and `models/` inside the sub-repo are *Python packages*, not data/checkpoints.

### 2026-07-09 — dataset obtained (from Tei, via Marta)

- CERNBox public share: `https://cernbox.cern.ch/s/AeKNoWslVzwmAdt` (all of Tei's MCFM datasets).
  Asimov-check data in `zz4l/ggzz4l_sm/`. Share browsable + downloadable **anonymously** via WebDAV:
  `https://cernbox.cern.ch/remote.php/dav/public-files/AeKNoWslVzwmAdt/<path>` (user = share token, empty password).
- Downloaded to `data/zz4l/ggzz4l_sm/` (gitignored; `$DATASET_DIR` in setup.sh):
  `ggzz4l_sig.csv` (3.8 GB), `ggzz4l_bkg.csv` (7.0 GB), `ggzz4l_sbi.csv` (7.2 GB), `ggzz4l_xs.json`.
  All byte sizes match the WebDAV listing exactly.
- `ggzz4l_xs.json`: `{"sbi": [2, 1.83, 1.5454912568], "sig": [2, 1.83, 0.08467131148], "bkg": [2, 1.83, 1.6265120219], "int": [2, 1.83, -0.1656920766]}` —
  per Tei: normalise the sum of weights to the total cross-section, then multiply by luminosity.
  Cross-check: yields at 1 fb⁻¹ in context.md = 2 × 1.83 × xs for every component
  (e.g. SBI 5.656 = 3.66 × 1.5455) → the `[2, 1.83]` factors multiply the xs. Meaning of 2 and 1.83
  (flavour combinations × k-factor?) to be confirmed in code.
- CSV format: 6-particle final-state kinematics + per-Z lepton 4-vectors + derived variables
  (`Z1_mass, Z2_mass, l{1..4}_{pt,eta,phi,energy,charge}, 4l_mass, 4l_rapidity, 4l_pT, 4l_energy`),
  matrix elements `msq_{sig,int,sbi,bkg}_sm`, **45 BSM variations** `msq_{sig,int,sbi}_bsm_{1..45}`
  (EFT reweighting — relevant for Phase 4), and event weight `wt`.

### 2026-07-09 — container build

- First build attempt failed: Apptainer cache defaults to `$HOME/.apptainer/cache` and the home
  quota is exceeded. Fix: `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` → `output/apptainer-{cache,tmp}`
  (project area, gitignored). Rebuild in progress.
- Reminder: **no GPU on this node** — training will need a GPU node; to be checked what actually
  requires GPU vs runs on CPU (see walkthrough).

### 2026-07-09 — heads-up from Tei (relayed by Marta)

- Tei realised the **NSBI NLL-scan code is missing the extended Poisson (total-rate) term**.
  Agreed procedure: reproduce his results with the notebooks **as-is** first, then add the term.
  Recorded in `docs/context.md` (Known code issues) and `docs/plan.md` (Phase 1 backlog).
- Code walkthrough started: module-by-module maps + verification of the missing-Poisson-term
  claim with file:line evidence. Results appended below.

## Code walkthrough (2026-07-09)

### Verified: the missing extended Poisson term (Tei's claim — CONFIRMED)

- **NSBI NLL as implemented** (`notebooks/nsbi/fit_asimov_and_toys.ipynb`, cell `e9ffa5e9`, raw line 258):
  `nll_scan = -2 * Σ_i ν_i · log r(x_i|μ)` with `ν_i = lumi · w_i` and `r = p_SBI(x|μ)/p_B(x)`
  (xs-normalised mixture of the two CARL ratios, cell `82fdbe3f`). **Pure shape — no ν(μ) − n·log ν(μ)
  rate term anywhere in the NSBI path** (repo-wide grep). A binned-Poisson alternative is literally
  commented out just above the NLL line. Consistently, NLL(μ=0) = 0 exactly in the saved scan.
- **Histogram NLL is fine**: `stats/binned.py:14` (Asimov) and `:48` (toys) use the full per-bin
  Poisson log-pmf `n·log ν − ν − lgamma(n+1)` summed over all bins → the extended rate term is
  implicitly and correctly included (expectations are absolute yields: fills use `w·lumi`, Σw = xs).
- **Where the term goes when we add it later** (after as-is reproduction, per Tei):
  add `-2·(N_obs·log ν_tot(μ) − ν_tot(μ))` with `ν_tot(μ) = lumi·(μ·σ_S + √μ·σ_I + σ_B)` — exactly
  lumi × the denominator already computed in cell `82fdbe3f`; `N_obs = lumi·Σw` for Asimov, per-toy
  totals for toys. Note: the −N·log ν(μ)-like piece partially enters via the ratio's denominator;
  the linear +ν(μ) piece is entirely absent.

### How the pipeline works (condensed)

- **Data**: `mcfm.from_csv` reads the CSVs (208 cols: p1..p6 four-vectors, derived lepton/Z/4ℓ
  variables, `msq_{sig,int,sbi,bkg}_sm`, 45 EFT points `msq_*_bsm_{1..45}` on a (c6,ct,cg)=5×3×3 grid,
  weight `wt`), zeroes negative weights (documented hack), and normalises Σw = xs [fb]
  (`physics/simulation/mcfm.py:106-107`). xs = `np.prod` of the json lists (so 5.656 = 2·1.83·1.5455 fb
  for SBI — matches context.md yields). Luminosity enters **only at inference** (`ν_i = w_i·lumi`).
- **POI reweighting** (single SBI sample → any μ): `physics/hstar/sigstr.py:25`
  `w(μ) = w_sm·(μ·msq_sig + √μ·msq_int + msq_bkg)/msq_sbi`. Theta mode exists too
  (`physics/hstar/theta.py:31`, `w(θ) ∝ θ²·SIG + θ·INT + BKG`) — **θ=0 is background-only, θ=1 is SM;
  the docstring saying θ=(μ−1)² is wrong.**
- **Toys**: `toys.generate` = per-event `torch.poisson(λ = w·lumi)`, 200 toys, seed 42.
- **Histogram analysis**: `notebooks/hist/fit_asimov_and_toys.ipynb` — 3e5 rows of the SBI CSV,
  10 log bins in m4ℓ ∈ [180, 1000] GeV, scan grid linspace(0,4,101) in μ (or −2..2 in θ), full binned
  Poisson NLL. **CPU-only, minutes, <1 GB RAM.** Committed state: `poi='theta'`, `lumi=1000`.
  (`notebooks/generate_toy_datasets.ipynb` is md5-identical to the hist notebook — a duplicate.)
- **NSBI analysis**: 2 CARL classifiers = plain MLPs (SiLU, sigmoid; weighted BCE; NAdam +
  ReduceLROnPlateau; checkpoints top-5 on val_loss; early stopping patience 20), trained via
  `python -m nsbi.carl fit` (LightningCLI) on **balanced** class weights (shape-only), 6:2:2 split.
  Inference: likelihood-ratio trick `r = s/(1−s)·C`, calibration constants set to 1.0 (measured
  1.0034/0.9995 commented out). Fit notebook committed state: `poi='mu'`, `lumi=300`,
  2e5 rows, test split = 576,265 events.
- **Committed reference results** at sub-repo root (`nll_scan_asimov.pt`, `nll_scan_toys.pt`,
  `poi_scan.pt`; `mu_scan.pt` is an orphan duplicate of `poi_scan.pt`): produced by the **NSBI**
  notebook in μ mode at **300 fb⁻¹**. Careful — hist and NSBI notebooks write the **same filenames**
  into the cwd (mutual clobbering); archive before any run.

### Reproduction readiness

**Histogram (unblocked once the container is built; CPU ok):**
1. Bridge the data path: notebooks hardcode `./data/sm/` relative to kernel cwd = sub-repo root →
   `ln -s ../../data/zz4l/ggzz4l_sm asimov-aymptotics-check/data/sm`.
2. Archive Tei's committed `.pt` reference files first (clobbering, see above).
3. Set `poi='mu'` (committed state is θ-mode, whose Asimov is background-only) and sweep
   `lumi ∈ {1, 10, 100, 1000}` by hand (no lumi loop exists); archive the `.pt` triple per point.
4. Run `notebooks/compare_median_vs_asimov.ipynb` from the same cwd for the overlay plot.

**NSBI (blocked on two things):**
1. **CARL hyperparameters missing and mandatory** — `CARL.__init__(n_features, n_layers, n_nodes,
   learning_rate)` has no defaults (`models/carl.py:9`) and no config yaml/ckpt is committed
   (run/, jobs/ gitignored). → **Ask Tei** for `config.yaml`/`hparams.yaml` or any checkpoint from
   `/ptmp/mpp/taepa/stat/ml4fp-sbi-tutorial/run/h4l/{sig_over_bkg,sbi_over_bkg}` (+ seed, epochs,
   and whether numerator samples were the separate CSVs or msq-reweighted SBI).
2. **GPU required as written** — `accelerator='gpu'` hardcoded (`nsbi/carl/utils.py:49`, fit-notebook
   cell `cc1313c1`); training is ~5.7M balanced samples/epoch. Current node has no GPU.
3. Also: training must pass `--data.features` explicitly (the default feature list contains angular
   variables `cth_star…phi` that are **not** CSV columns → KeyError), and each classifier must be
   launched from its own `run/h4l/<name>/` dir so `lightning_logs/` sits next to the pickles for
   `load_results` (which picks the largest-epoch of the top-5 checkpoints, not the best — preserve
   this quirk when comparing). The toy-NLL cell materialises a (576265, 101, 200) float64 tensor
   ≈ **93 GB RAM** — needs a fat node or a chunk-over-toys edit (approval before editing).

### 2026-07-09 — git state

- Container repo `SBI_shortcomings`: local-only, no remote yet — to be published as a **private**
  GitHub repo (`git@github.com:martafsilva31/SBI_shortcomings.git`). SSH auth to GitHub works on
  this node; `gh` is not installed, so the empty repo is created via the web UI, then pushed.
- Sub-repo `asimov-aymptotics-check`: this IS the fork of Tae's code
  (`origin = github.com/martafsilva31/asimov-aymptotics-check.git`, already on GitHub; 4 commits,
  all authored by Tae Hyoun Park). No `upstream` remote to his original (optional — for pulling
  his future updates).

### 2026-07-09 — Tei's slides (`asimov_vs_medtoys_test_statistic_studies.pdf`, 7 pp.)

- Confirms the whole setup: SBI mixture model (both forms in context.md), yields table (exact match),
  toy recipe `w_toy ~ P(λ=w)`, and the two likelihoods. Slide 3 writes the **intended** NSBI
  likelihood *with* the Poisson term `L = P(n|ν)·∏ p_SBI/p_ref` and notes the rate term is computed
  analytically (not NN-estimated) — consistent with the code *omitting* it being a bug, not design.
- Target figures: **slide 6** = Asimov ≈ median(toys) for the **histogram** (good, all lumi; note the
  slide title says "NSBI" but the bullet + curves are the histogram). **Slide 7** = the **NSBI**
  result: median(toys) departs from Asimov at 1/10/100 fb⁻¹, converging only at 1000 fb⁻¹ "when the
  Poisson term takes over." Tei's own read: NN mis-specification, and an NN estimator can't be exactly
  unbiased (unlike histogramming).
- **⚠ Direction to verify:** on slide 7 at 1 fb⁻¹ the **median-of-toys NLL is steeper (narrower) than
  the Asimov** (blue above orange) — i.e. the single Asimov scan is the *wider/less-constraining* one.
  context.md currently says "NSBI NLL scan is systematically narrower than the median … (overconfident)",
  which reads as the opposite sign. Resolve during reproduction before it propagates to Notion; these
  are the buggy-code (no-Poisson-term) curves, so re-check after adding the term too.

## Open questions for Tei

1. CARL training configs (hyperparameters/seed/epochs or a checkpoint) — hard blocker for NSBI.
2. Which (poi, lumi) points to reproduce exactly? Committed notebooks have hist@(θ,1000) and
   NSBI@(μ,300), but the headline result is μ at 1–10 fb⁻¹.
3. Meaning of the `[2, 1.83]` factors in `ggzz4l_xs.json` (flavour combinations × k-factor?).

## Next

- Verify container build + Python env inside it (home quota may block `pip install --user`;
  fallback: `PYTHONPATH=<sub-repo>` — pip install only covers `physics`+`stats` anyway).
- On approval: run the histogram reproduction (μ mode, lumi sweep) — the Asimov ≈ median(toys)
  sanity check — and log results here.
- Ask Tei the open questions above; then NSBI reproduction on a GPU node.
