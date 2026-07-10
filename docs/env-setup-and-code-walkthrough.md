# Environment setup & code walkthrough

> **TL;DR** — This task took the SBI_shortcomings project from "empty container repo + a fork of Tei's code" to "fully runnable and fully understood". The MCFM gg → ZZ → 4ℓ dataset (18 GB of per-event CSVs) was located on Tei's CERNBox share and downloaded locally; the Apptainer software container was built and verified on a CPU-only node; and Tei's code was walked through module by module, with every claim pinned to file:line evidence. Two load-bearing findings: (1) Tei's own warning is **confirmed** — the NSBI NLL-scan notebook implements a *shape-only* likelihood and omits the extended Poisson (total-rate) term, while the histogram path implements the full extended likelihood correctly; (2) the CARL network hyperparameters/checkpoints exist **only** on Tei's MPP cluster, so exact NSBI reproduction needs his `run/h4l/` directory (path A) — or we retrain with our own settings (path B). The histogram analysis is runnable on CPU immediately (and was subsequently reproduced — see the related note).
>
> **Status:** ✅ Done (2026-07-09, addenda 2026-07-10) · **Project:** SBI_shortcomings · **Related:** [hist-reproduction](hist-reproduction.md), [notes/sbi_foundations](notes/sbi_foundations.md), [notes/literature](notes/literature.md)

| Item | Value |
|---|---|
| Question | Can Tei's `asimov-aymptotics-check` analysis run in our environment, and how exactly does the code work? |
| Dataset(s) | MCFM parton-level gg → ZZ → 4ℓ CSVs, 18 GB total, CERNBox → local `data/zz4l/ggzz4l_sm/` |
| Method | Anonymous WebDAV download; Apptainer container build + import verification; module-by-module code walkthrough with file:line evidence |
| Headline result | Everything runnable (histogram path CPU-only); missing extended-Poisson-term claim CONFIRMED in the NSBI notebook; CARL hyperparameters not recoverable locally — NSBI needs Tei's run dir (path A) or a retrain (path B) |
| Environment | Apptainer `ml4fp-sbi-tutorial.sif` (4.3 GB): torch 2.2.1, lightning 2.6.5, sklearn 1.7.2; no GPU on this node; no EOS mount |
| Status | ✅ setup complete · ⏳ NSBI reproduction blocked on Tei's `run/h4l/` |

## 1. Background & goal

This project studies why Neural Simulation-Based Inference (NSBI) violates the *Asimov property* — the assumption that a fit to the Asimov dataset (the dataset in which every observable takes exactly its expected value) reproduces the median result of an ensemble of toy pseudo-experiments — in the non-asymptotic (low event count) regime. The concrete starting point is Tei's demonstration on off-shell Higgs production, gg → H* → ZZ → 4ℓ, where at 1–10 fb⁻¹ the NSBI's toy median visibly departs from its Asimov curve while a plain histogram analysis of the same events shows no such departure. Before anything can be reproduced or extended, three things had to happen, and they are this task: obtain the dataset, build a working software environment, and understand Tei's code well enough to trust and modify it.

"Tei's code" is the sub-repo `asimov-aymptotics-check/` (our GitHub fork of his repository, 4 commits, all authored by him). It contains the physics/statistics Python packages, the two analysis notebooks (histogram and NSBI), and — importantly — his committed **reference results** (`nll_scan_asimov.pt`, `nll_scan_toys.pt`, `poi_scan.pt`, `mu_scan.pt` at the repo root), which are the ground truth any reproduction must match.

## 2. Setup

### 2.1 Node constraints (lxplus-like project node)

| Constraint | Consequence |
|---|---|
| No NVIDIA GPU (`nvidia-smi` absent) | CARL training/inference (hardcoded `accelerator='gpu'`) cannot run here; histogram path is pure CPU torch and runs fine |
| No EOS mount | Dataset must be a local copy, not an EOS path |
| `$HOME` quota exceeded | Apptainer cache and pip caches must be redirected off `$HOME` (see 3.3); stray `~/.local` python packages also leak into the container (see the hist-reproduction note for the fix, `PYTHONNOUSERSITE=1`) |
| Apptainer available at `/bin/apptainer` | Container workflow is viable |

### 2.2 Dataset

The dataset lives on Tei's CERNBox public share (all his MCFM datasets): `https://cernbox.cern.ch/s/AeKNoWslVzwmAdt`, folder `zz4l/ggzz4l_sm/`. The share is browsable and downloadable **anonymously** over WebDAV — `https://cernbox.cern.ch/remote.php/dav/public-files/AeKNoWslVzwmAdt/<path>` with username = the share token and empty password — so no CERN credentials are needed.

| File | Size | Content |
|---|---|---|
| `ggzz4l_sig.csv` | 3.8 GB | signal-only events (S) |
| `ggzz4l_bkg.csv` | 7.0 GB | continuum-background events (B) |
| `ggzz4l_sbi.csv` | 7.2 GB | signal+background+interference events (SBI) |
| `ggzz4l_xs.json` | 156 B | per-component cross-section normalisations |

Local copy: `data/zz4l/ggzz4l_sm/` (gitignored; exported as `$DATASET_DIR` by `setup.sh`). All byte sizes match the WebDAV listing exactly.

**CSV format** (208 columns): 6-particle final-state four-vectors `p1..p6`, per-Z lepton four-vectors `Z{1,2}_l{1,2}_*`, derived kinematics (`Z1_mass`, `Z2_mass`, `l{1..4}_{pt,eta,phi,energy,charge}`, `4l_mass`, `4l_rapidity`, `4l_pT`, `4l_energy`), SM matrix elements `msq_{sig,int,sbi,bkg}_sm`, **45 BSM variations** `msq_{sig,int,sbi}_bsm_{1..45}` on a (c6, ct, cg) = 5×3×3 EFT grid (per-event reweighting — the Phase-4 hook), and the event weight `wt`.

**Cross-section JSON** — `{"sbi": [2, 1.83, 1.5454912568], "sig": [2, 1.83, 0.08467131148], "bkg": [2, 1.83, 1.6265120219], "int": [2, 1.83, -0.1656920766]}`. The code takes `np.prod` of each list, so the usable cross-section is `2 × 1.83 × xs_raw` (fb). Per Tei: normalise the sum of event weights to this cross-section, then multiply by the tested luminosity. Cross-check against the yields table in `context.md`: every 1 fb⁻¹ yield equals `2 × 1.83 × xs_raw` exactly (e.g. SBI: 3.66 × 1.5455 = 5.656). The meaning of the two prefactors is open question 3 below (proposed answer: channel counting × k-factor).

### 2.3 Software container

`asimov-aymptotics-check/container.def` builds `ml4fp-sbi-tutorial.sif` (4.3 GB). Verified inside: torch 2.2.1 (`cuda.is_available() = False`, expected on this node), lightning 2.6.5, sklearn 1.7.2, scikit-hep (`hist`, `vector`) all import. The project's own top-level packages (`physics`, `stats`, `datasets`, `nsbi`) import from the sub-repo root cwd **without any `pip install`** — the home-quota problem is moot for running the code.

## 3. What was done

### 3.1 Environment survey

Checked what already existed: the sub-repo was cloned 2026-06-02 with real code **and** Tei's committed reference results (the four `.pt` files at the repo root, plus result plots `with_int.png`, `without_int.png`, `theta_lin.png`, `theta_quad.png`). Established the node constraints of 2.1. One trap documented for posterity: `datasets/` and `models/` inside the sub-repo are Python *packages*, not data/checkpoint directories.

### 3.2 Dataset acquisition

Located the Asimov-check data in `zz4l/ggzz4l_sm/` on the share (relayed by Tei), discovered the share is anonymously WebDAV-accessible, downloaded all four files, and verified byte-exact sizes against the listing. While searching the share for CARL configs (3.6), two additional datasets were catalogued for possible later use: `zz4l/ggZZ_{sig,bkg,sbi,int}/analyzed.csv` (per-component analyzed files) and `zz4l/observed_mu{0p4,2p5}/observed_{0..47}.csv` — 48 pre-generated pseudo-datasets at injected μ = 0.4 and μ = 2.5 each, presumably for observed-data fits (open question 4).

### 3.3 Container build

First build attempt failed: Apptainer's cache defaults to `$HOME/.apptainer/cache` and the home quota is exceeded. Fix: `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` → `output/apptainer-{cache,tmp}` (project area, gitignored). Rebuild succeeded; verification as in 2.3.

### 3.4 Code walkthrough — how the pipeline works

**Data loading** — `physics/simulation/mcfm.py`: `mcfm.from_csv` reads the CSVs, zeroes negative weights (a documented hack), and normalises `Σw = xs` in fb (`mcfm.py:106-107`). Luminosity enters **only at inference** as `ν_i = w_i · lumi` — training never sees it.

**POI reweighting** — a single SBI sample serves any tested μ via per-event matrix-element reweighting, `physics/hstar/sigstr.py:25`:

$$ w(\mu) = w_{\mathrm{sm}} \cdot \frac{\mu \, \mathrm{msq}_{\mathrm{sig}} + \sqrt{\mu} \, \mathrm{msq}_{\mathrm{int}} + \mathrm{msq}_{\mathrm{bkg}}}{\mathrm{msq}_{\mathrm{sbi}}} $$

A θ mode exists too (`physics/hstar/theta.py:31`, `w(θ) ∝ θ^2 \cdot \mathrm{SIG} + θ \cdot \mathrm{INT} + \mathrm{BKG}`); note **θ = 0 is background-only and θ = 1 is the SM — the docstring saying θ = (μ−1)² is wrong**.

**Toys** — `toys.generate`: per-event `torch.poisson(λ = w · lumi)`, 200 toys, seed 42. A toy pseudo-experiment is therefore a Poisson-fluctuated copy of the per-event weight vector, not a resampled event list.

**Histogram analysis** — `notebooks/hist/fit_asimov_and_toys.ipynb`: 3×10⁵ rows of the SBI CSV, 10 log-spaced bins in m4ℓ ∈ [180, 1000] GeV, scan grid `linspace(0, 4, 101)` in μ (or −2..2 in θ), full binned Poisson NLL. CPU-only, minutes, < 1 GB RAM. Committed state: `poi='theta'`, `lumi=1000`. (`notebooks/generate_toy_datasets.ipynb` is md5-identical to this notebook — a duplicate.)

**NSBI analysis** — `notebooks/nsbi/fit_asimov_and_toys.ipynb`: two CARL classifiers (plain MLPs: SiLU hidden activations, sigmoid output; weighted BCE; NAdam + ReduceLROnPlateau; top-5 val_loss checkpoints; early stopping patience 20) estimate the density ratios p_sig/p_bkg and p_sbi/p_bkg, trained via `python -m nsbi.carl fit` (LightningCLI) on **balanced** class weights (shape-only), 6:2:2 sequential split. Inference applies the likelihood-ratio trick `r = s/(1−s) · C` with calibration constants set to 1.0 (measured values 1.0034/0.9995 left commented out). Committed notebook state: `poi='mu'`, `lumi=300`, 2×10⁵ rows per CSV, test split = 576,265 events.

**Committed reference results** — the root-level `.pt` files were produced by the **NSBI** notebook in μ mode at **300 fb⁻¹** (`mu_scan.pt` is an orphan duplicate of `poi_scan.pt`). ⚠ The hist and NSBI notebooks write the **same output filenames** into the cwd, so runs clobber each other and the committed files — archive before any run (done: `output/reference_tei/`).

### 3.5 The missing extended Poisson term (Tei's claim — CONFIRMED)

Tei flagged on 2026-07-09 that the NSBI NLL-scan code omits the extended Poisson (total-rate) term. Verified by inspection, with the agreed procedure being: reproduce his results with the notebooks **as-is** first, then add the term and re-run.

The NSBI NLL as implemented (`notebooks/nsbi/fit_asimov_and_toys.ipynb`, cell `e9ffa5e9`, raw line 258) is pure shape:

$$ \mathrm{NLL}(\mu) = -2 \sum_i \nu_i \log r(x_i \mid \mu), \qquad \nu_i = L \, w_i, \qquad r = \frac{p_{\mathrm{SBI}}(x \mid \mu)}{p_B(x)} $$

where r is the cross-section-normalised mixture of the two CARL ratios (cell `82fdbe3f`). A repo-wide grep confirms **no** `ν(μ) − n log ν(μ)` rate term anywhere in the NSBI path; a binned-Poisson alternative is literally commented out just above the NLL line, and consistently NLL(μ=0) = 0 exactly in the saved scan. The **histogram NLL is fine**: `stats/binned.py:14` (Asimov) and `:48` (toys) use the full per-bin Poisson log-pmf `n log ν − ν − lgamma(n+1)`, so the rate term is implicitly and correctly included. Tei's slides (3.7) write the *intended* NSBI likelihood **with** the Poisson term — the omission is a bug, not a design choice. Where the term goes when added later (after as-is reproduction): add `−2 (N_{obs} \log ν_{tot}(\mu) − ν_{tot}(\mu))` with `ν_{tot}(\mu) = L (\mu σ_S + \sqrt{\mu} σ_I + σ_B)` — exactly L times the denominator already computed in cell `82fdbe3f`; `N_{obs} = L Σw` for the Asimov, per-toy totals for toys. The full patch draft lives in [nsbi-repro-prep](nsbi-repro-prep.md).

### 3.6 CARL config hunt — hyperparameters are NOT recoverable

`CARL.__init__(n_features, n_layers, n_nodes, learning_rate)` has **no defaults** (`models/carl.py:9`) and no config yaml or checkpoint is committed (`run/`, `jobs/` are gitignored). Searched every place they could hide: the local working tree (no `*.yaml`/`*.ckpt`/`hparams*`), the **full git history** of all 4 commits (nothing config-like ever committed or deleted), the committed `.pt` files (plain scan tensors — shapes (101,), (101,), (101,200) — no state_dict, no hparams), the notebook outputs (models are loaded via `carl.utils.load_results('/ptmp/mpp/taepa/stat/ml4fp-sbi-tutorial/run/h4l', …)` → `CARL.load_from_checkpoint`, so the hyperparameters live only *inside* those cluster-side `.ckpt` files via `save_hyperparameters()`, `models/carl.py:11`), and the CERNBox share (only CSVs). Conclusion: **only** Tei's MPP-cluster run directory has them.

Important correction to the initial "hard blocker" framing: Tei's exact config is required only for a *bit-exact* match of his curves, **not** to reproduce the *phenomenon* — the NSBI-overconfidence effect is claimed to be intrinsic, and every checkpoint self-describes. Hence two paths: **(A)** copy Tei's `run/h4l/` (bit-exact, inference-only, fast); **(B)** train our own CARL with sensible hyperparameters (independent reproduction). Both need a GPU node (`accelerator='gpu'` hardcoded at `nsbi/carl/utils.py:49` and in the fit notebook).

### 3.7 Cross-check against Tei's slides

`docs/presentations/asimov_vs_medtoys_test_statistic_studies.pdf` (7 pp., first meeting 2026-05-20) confirms the whole setup: the SBI mixture model (both algebraic forms in `context.md`), the yields table (exact match), the toy recipe `w_toy ∼ P(λ = w)`, and the two likelihoods — slide 3 includes the analytic Poisson rate term in the intended NSBI likelihood. Target figures: slide 6 = histogram Asimov ≈ median(toys) at all luminosities (the slide title says "NSBI" but bullet + curves are the histogram); slide 7 = the NSBI failure, median departing from Asimov at 1/10/100 fb⁻¹ and converging only at 1000 fb⁻¹ "when the Poisson term takes over".

A direction ambiguity was spotted here — slide 7 reads as *median narrower than Asimov*, while some project wording said the opposite. **Resolved 2026-07-10** by measuring Tei's committed 300 fb⁻¹ tensors: median(toys) is indeed the **narrower/steeper** curve (Δ(−2lnL)=1 width ratio median/Asimov = 0.874), i.e. each toy analysis is overconfident and the Asimov *understates* the toys' claimed sensitivity. Full numbers in [hist-reproduction](hist-reproduction.md); `context.md` wording clarified; the inverted sentence in the project `CLAUDE.md` is flagged for Marta (not edited).

### 3.8 Git state

Container repo `SBI_shortcomings` published as a **private** GitHub repo (`git@github.com:martafsilva31/SBI_shortcomings.git`), `main` pushed, zero AI traces. Sub-repo `asimov-aymptotics-check` is our fork of Tei's code (origin on GitHub, 4 commits all authored by him); no `upstream` remote configured (optional, for pulling his future updates).

## 4. Results

### 4.1 Walkthrough verdicts (each pinned to evidence)

| Claim | Verdict | Evidence |
|---|---|---|
| NSBI NLL omits the extended Poisson term | ✅ CONFIRMED | `notebooks/nsbi/fit_asimov_and_toys.ipynb` cell `e9ffa5e9` line 258; NLL(μ=0)=0 in saved scan |
| Histogram NLL is a full extended likelihood | ✅ | `stats/binned.py:14` and `:48` (per-bin Poisson log-pmf incl. rate term) |
| θ-mode docstring | ❌ wrong | `physics/hstar/theta.py:31` — θ=0 is bkg-only, θ=1 is SM; docstring says θ=(μ−1)² |
| `generate_toy_datasets.ipynb` | duplicate | md5-identical to the hist fit notebook |
| CARL hyperparameters recoverable locally | ❌ no | exhaustive hunt, 3.6 |
| Committed `.pt` reference results | NSBI, μ mode, 300 fb⁻¹ | notebook committed state + walkthrough |
| `load_results` checkpoint choice | largest-epoch, not best val_loss | `nsbi/carl/utils.py:11-46` — a quirk to preserve when comparing |
| NSBI toy-NLL memory | ~93 GB tensor | un-reduced (576265, 101, 200) float64 broadcast — needs the chunking patch before toy scans on a normal node |

### 4.2 Reproduction readiness

| Analysis | Status | Needs |
|---|---|---|
| Histogram (μ mode, any lumi) | ✅ unblocked, CPU-only | data-path bridge + archive-first discipline (done in the follow-up task) |
| NSBI path A (bit-exact) | ⏳ blocked | Tei's full `run/h4l/{sig_over_bkg,sbi_over_bkg}/` (list in §8) + GPU node (inference only, no training) |
| NSBI path B (independent) | ready to launch | GPU node + our own hyperparameters (launch kit in [nsbi-repro-prep](nsbi-repro-prep.md)) |

## 5. Discussion & conclusions

The environment is no longer a risk: dataset local and size-verified, container built and import-verified, histogram path proven CPU-runnable. The code is understood to file:line depth, which already paid off twice — confirming the missing-rate-term bug (and locating exactly where the fix goes), and catching the notebook-output clobbering hazard before it cost us Tei's reference results. The one genuine external dependency is Tei's `run/h4l/` directory for path-A NSBI; everything else on the critical path (histogram reproduction, patch drafts, path-B prep) proceeds without him. The direction ambiguity in the key finding is settled the slide-7 way (toys narrower than Asimov), which matters because every downstream write-up will phrase the failure mode from it.

Caveats: the walkthrough conclusions about the NSBI path rest on the committed notebook state (μ, 300 fb⁻¹) and static inspection; behaviour at other luminosities is extrapolated until path A/B runs. The `[2, 1.83]` normalisation prefactors have a proposed but unconfirmed interpretation (§8, Q3).

## 6. How to reproduce

```bash
cd /project/atlas/Users/mfernand/SBI_shortcomings
source setup.sh   # prints the canonical commands; does NOT launch anything

# 1. dataset (anonymous WebDAV: user = share token, empty password)
mkdir -p data/zz4l/ggzz4l_sm && cd data/zz4l/ggzz4l_sm
for f in ggzz4l_xs.json ggzz4l_sig.csv ggzz4l_bkg.csv ggzz4l_sbi.csv; do
  curl -u 'AeKNoWslVzwmAdt:' -O "https://cernbox.cern.ch/remote.php/dav/public-files/AeKNoWslVzwmAdt/zz4l/ggzz4l_sm/$f"
done
cd -

# 2. container build (cache off $HOME — quota!)
export APPTAINER_CACHEDIR=$PWD/output/apptainer-cache APPTAINER_TMPDIR=$PWD/output/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build asimov-aymptotics-check/ml4fp-sbi-tutorial.sif asimov-aymptotics-check/container.def

# 3. verify (PYTHONNOUSERSITE=1 keeps ~/.local from leaking in — see hist-reproduction note)
apptainer exec --bind $PWD --env PYTHONNOUSERSITE=1 asimov-aymptotics-check/ml4fp-sbi-tutorial.sif \
  python3 -c "import torch, lightning, sklearn, hist, vector; print('torch', torch.__version__)"
```

## 7. Files, inputs & outputs

| What | Path |
|---|---|
| Dataset (gitignored) | `data/zz4l/ggzz4l_sm/{ggzz4l_sig,ggzz4l_bkg,ggzz4l_sbi}.csv`, `ggzz4l_xs.json` |
| Container image (gitignored) | `asimov-aymptotics-check/ml4fp-sbi-tutorial.sif` (4.3 GB) |
| Container recipe | `asimov-aymptotics-check/container.def` |
| Tei's reference results (archived copies) | `output/reference_tei/{nll_scan_asimov,nll_scan_toys,mu_scan,poi_scan}.pt` |
| Tei's slides | `docs/presentations/asimov_vs_medtoys_test_statistic_studies.pdf` |
| Sub-repo (fork of Tei's code) | `asimov-aymptotics-check/` — `github.com/martafsilva31/asimov-aymptotics-check` |
| This note's source task log | this file (running log rewritten 2026-07-10 by /writeup; raw log in git history, commit `326954b`) |

## 8. Open questions / next

1. **To Tei — path A files.** The entire `…/run/h4l/` tree, i.e. for each of `sig_over_bkg` and `sbi_over_bkg`: `events_{numerator,denominator}_{val,test}.pkl`, `scaler.pkl`, `lightning_logs/version_*/checkpoints/epoch=*-val_loss=*.ckpt` (`load_results` contract, `nsbi/carl/utils.py:11-46`). With these, no separate hyperparameter list is needed — they are baked into the checkpoints.
2. **To Tei — which (poi, lumi) points to reproduce exactly?** Committed notebooks have hist@(θ, 1000) and NSBI@(μ, 300), but the headline result is μ at 1–10 fb⁻¹.
3. **To Tei — the `[2, 1.83]` factors.** Proposed answer (2026-07-10): the identical pair multiplies all four components, so it must be channel counting × common QCD normalisation — **2** = distinct-flavour 2e2μ → all e/μ 4ℓ channels (σ(4e) + σ(4μ) + σ(2e2μ) = 2 σ(2e2μ) in BR counting, identical-flavour interference neglected; the CSV's fixed Z1(l1,l2)/Z2(l3,l4) pairing with no flavour labels is the MCFM distinct-flavour process), and **1.83** = a flat higher-order QCD k-factor on the LO MCFM gg → ZZ processes applied commonly to S/I/B, as ATLAS does (Run-1 off-shell used K_NLO = 1.7; the 2024 NSBI measurement applies NLO k-factors × 1.2 NNLO/NLO × 1.1 N3LO/NNLO common to all gg → ZZ components — a product in the ~1.8 ballpark). Only the exact provenance of 1.83 needs Tei.
4. **To Tei — the extra CERNBox sets.** What are `observed_mu{0p4,2p5}/` (48 pseudo-datasets each at injected μ = 0.4 / 2.5) and `ggZZ_*/analyzed.csv` for — a separate observed-data study?
5. GPU-node access details (for path A inference or path B training) — with Marta.

### Appendix — key commits

| Repo | Commit | One-liner |
|---|---|---|
| SBI_shortcomings | `318ebc4` | initialise container (docs, env setup, project context) |
| SBI_shortcomings | `326954b` | env-setup log and plan updates (raw running log of this note) |
| SBI_shortcomings | `4205876` | align container structure (slides to docs/presentations) |
| SBI_shortcomings | `8623943` | curated literature note and project foundations |
| asimov-aymptotics-check | `c36bbaa` | First commit (Tei) |
| asimov-aymptotics-check | `f8f01f4` / `44d1141` | Fitting toys (Tei) |
| asimov-aymptotics-check | `aee4d9c` | results for signal-strength + EFT modes, NSBI model changes (Tei) |
