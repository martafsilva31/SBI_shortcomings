# Histogram reproduction — μ mode, luminosity sweep

> Status: **running log** (task in progress — will be cleaned into a summary study-note)
> Task: reproduce Tei's histogram NLL scans (Asimov vs median of toys) at 1/10/100/1000 fb⁻¹ and
> verify Asimov ≈ median(toys); settle the NSBI direction question from his committed 300 fb⁻¹ result.
> Started: 2026-07-10
>
> **▶ Canonical artifact: [`notebooks/histogram_reproduction.ipynb`](../notebooks/histogram_reproduction.ipynb)** —
> one verbose, runnable, adjustable notebook (loops all four luminosities, plots inline, restyle
> without re-running the analysis). This note is the written companion; the notebook is the thing you
> open, run, and re-plot.

## Running log

### 2026-07-10 — method (sub-repo stays byte-for-byte untouched)

- Everything runs on **copies** of Tei's notebooks placed under gitignored `output/` — no symlink,
  no edits, no clobbering of his committed `.pt` files inside `asimov-aymptotics-check/` (verified
  `git status` clean before and after).
- Tei's four committed reference files archived: `asimov-aymptotics-check/{nll_scan_asimov,nll_scan_toys,mu_scan,poi_scan}.pt`
  → `output/reference_tei/` (byte copies, Jun 2 timestamps preserved). These are the **NSBI** μ-mode
  results at 300 fb⁻¹ (per code walkthrough), kept as ground truth.
- Per lumi point X ∈ {1, 10, 100, 1000}: copy of `notebooks/hist/fit_asimov_and_toys.ipynb` in
  `output/hist/lumi_X/` with exactly three lines patched (each patch asserted to hit exactly once):
  `data_dir` → absolute `data/zz4l/ggzz4l_sm/`, `lumi = X.`, `poi='mu'`. Outputs cleared before run.
  Everything else (300k rows, 10 log bins in m4ℓ ∈ [180, 1000] GeV, 200 toys, seed 42,
  μ grid linspace(0, 4, 101)) is Tei's committed state.
- Per dir (incl. `reference_tei/`): copy of `compare_median_vs_asimov.ipynb` with
  `map_location='cpu'` added to the three `torch.load` calls, plus one appended cell that computes
  the Δ(−2 lnL) = 1 crossings and prints which curve is narrower (quantitative Asimov-vs-median
  verdict, since the original notebook only overlays the two curves visually).
- Execution: headless `python3 -m nbconvert --to notebook --execute --inplace` inside the Apptainer
  container, cwd = the respective `output/` dir (so the `.pt` outputs land next to the executed
  notebook), `PYTHONPATH` = `output/pyextra` (nbconvert install, step 0) + sub-repo root (for the
  `physics`/`stats` packages). CPU only — the walkthrough established the hist path has no GPU calls.
- Copy prep script: session scratchpad `prep_notebooks.py` (temp helper, not part of the repo).

### 2026-07-10 — direction check on Tei's committed NSBI result (⚠ resolved)

- Executed `output/reference_tei/compare_median_vs_asimov.ipynb` on the archived committed scans
  (NSBI, μ mode, 300 fb⁻¹, no-Poisson-term code). Quantitative Δ(−2 lnL) = 1 crossings:

| Curve | μ̂ | interval | width |
|---|---|---|---|
| Asimov | 0.920 | [0.062, 1.329] | 1.267 |
| median(toys) | 0.760 | [0.085, 1.192] | 1.107 |

- **Verdict: median(toys) is NARROWER/steeper than the Asimov** (width ratio 0.874) — same
  direction as the slide-7 reading in the walkthrough. So the correct statement of the failure is:
  each toy analysis claims *more* sensitivity than the Asimov predicts (toy-level overconfidence);
  the single Asimov scan is the wider/less-constraining curve. Wordings that say "the NSBI (Asimov)
  scan is narrower than the median of toys" have the direction inverted.
- Also noted: the **Asimov best fit is μ̂ = 0.920, not 1.0**, and median(toys) μ̂ = 0.760 — with a
  correct extended likelihood the Asimov minimum should sit at the injected μ = 1 exactly; plausibly
  an artifact of the missing Poisson term (shape-only NLL). Re-check after the term is added.
- Caveats: single luminosity point (300 fb⁻¹, the only NSBI result available locally), buggy
  (no-rate-term) likelihood, medians over 200 toys; the low-lumi points come with the NSBI
  reproduction (path A, waiting on Tei's `run/h4l/`).
- Follow-up applied: clarified the ambiguous direction sentence in `docs/context.md` (Key finding);
  flagged (not applied) the inverted sentence in the project `CLAUDE.md` for Marta's approval.

### 2026-07-10 — first launch failed: home-dir site-packages leak into the container

- All four lumi runs died at the first `hist` plot call: `import mplhep` (lazy, inside
  `hist.plot`) aborts with FreeType `Can not load face (error 0x6)`.
- Root cause: Apptainer binds `$HOME`, and Python's **user site-packages
  (`~/.local/lib/python3.10/site-packages/`) shadow the container's packages**. The home dir has a
  stray `mplhep_data` install whose `fonts/firamath/FiraMath-Light.otf` (87 kB) is corrupt —
  plausibly truncated by the known home-quota overflow. The container's own `mplhep_data`
  (98 fonts, all load fine) never got used. `uhi` is also shadowed from `~/.local` (version-skew
  risk for `hist`), so this wasn't just a fonts problem.
- Fix: **`--env PYTHONNOUSERSITE=1` on every `apptainer exec`** — makes the container hermetic.
  Verified: `mplhep_data` then resolves to `/opt/conda`, `import mplhep` OK. Adopted for all runs
  in this task. (Proposal for Marta, not applied: bake this flag into the `setup.sh` instructions,
  and clean the stray `~/.local` packages — the corrupt font can bite any tool that scans it.)
- Notebooks/`.pt` outputs were untouched by the failed attempts (nbconvert aborts before writing;
  the failing cell precedes every `torch.save`), so the relaunch started from clean patched copies.

### 2026-07-10 — results: Asimov ≈ median(toys) confirmed at all four luminosities

All four points executed cleanly (fit + compare notebooks, 4+1 inline figures each; sub-repo
`git status` clean after, container repo touched only in `docs/`). Quantitative summary computed
from the archived `.pt` triples (A = Asimov curve, M = median-of-200-toys curve, both −2 lnL,
min-subtracted; widths from Δ = 1 crossings; maxdiff = max |A − M| over the full μ ∈ [0, 4] grid):

| lumi [fb⁻¹] | μ̂_A | μ̂_M | width_A | width_M | ratio M/A | maxdiff over grid |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.280 | — (open) | — (open) | — | 0.362 |
| 10 | 1.000 | 0.800 | — (open) | — (open) | — | 0.274 |
| 100 | 1.000 | 0.960 | 1.805 | 1.777 | **0.985** | 1.174 |
| 1000 | 1.000 | 1.080 | 0.805 | 0.848 | **1.054** | 1.708 |

- **The histogram Asimov best fit is μ̂ = 1.000 exactly at every luminosity** — the full extended
  per-bin Poisson likelihood recovers the injected value, in contrast to the NSBI (shape-only,
  no-rate-term) Asimov μ̂ = 0.920 at 300 fb⁻¹ from the direction check above.
- Where the 1σ intervals close inside the scan window (100/1000 fb⁻¹), **Asimov and median agree
  to ~2–5% in width** (0.985 / 1.054) — compare the NSBI ratio 0.874 (13% narrower) at 300 fb⁻¹.
- At 1–10 fb⁻¹ the Δ = 1 crossings fall outside μ ∈ [0, 4] (≈ 5.7 / 57 expected events → very
  shallow curves; the anticipated window-clipping caveat), so widths are not defined there.
  Curve-level agreement holds: max |A − M| ≈ 0.27–0.36 units of −2ΔlnL across the whole window —
  small on the Δ = 1 scale. μ̂_M wanders at low lumi (0.28 / 0.80) because the median curve's
  minimum is extremely shallow — its *position* is ill-determined, while the curve *shape* still
  tracks the Asimov; the overlay figures in the compare notebooks are the visual evidence.
- **Conclusion: reproduces Tei's slide 6 — the histogram analysis satisfies the Asimov property
  at all tested luminosities, including the 1–10 fb⁻¹ regime where the NSBI does not.**
- Yield sanity check: the 1 fb⁻¹ Asimov m4ℓ histogram integrates to ≈ 5.65 events (context.md
  expectation 5.656 for SBI at 1 fb⁻¹, minus the [180, 1000] GeV window edges). ✓

### Files

- **`notebooks/histogram_reproduction.ipynb`** (git-tracked) — the canonical artifact: verbose
  runnable notebook, lumi loop, inline plots, isolated plotting cells. Calls Tei's engine
  (`physics`/`stats`); adds nothing physics-wise. Executed 2026-07-10; both figures embedded.
- `output/hist/lumi_{1,10,100,1000}/` (gitignored) — the `.pt` scan triple per point, regenerated
  by the notebook's save cell.
- `output/hist/summary_asimov_vs_median.png` (gitignored) — standalone copy of the main plot.
- `output/reference_tei/` (gitignored) — archived committed scans + executed direction-check notebook.
- Environment: `output/pyextra/` (nbconvert/ipykernel, step 0); `PYTHONNOUSERSITE=1` mandatory
  when running the notebook headlessly or in the container (see incident above).

### 2026-07-10 — consolidated into a single runnable notebook (per Marta's standing rule)

- The first pass ran four *copies* of Tei's notebook headlessly and left results as numbers in this
  note — results weren't in a notebook Marta could open and re-plot. Superseded by
  `notebooks/histogram_reproduction.ipynb`: one narrated notebook that loads the SBI sample once,
  loops the four luminosities (using histogram linearity in the weights — the expensive 101-point
  fill runs once, scaled by L), throws the toys per luminosity, and plots inline. The redundant
  per-lumi notebook copies under `output/hist/` were removed; the `.pt` triples remain (the notebook
  regenerates them). Numbers identical to the first pass (μ̂ = 1.000 ∀L; ratios 0.985 / 1.054).

### Open follow-ups

- μ scan window [0, 4] too narrow to contain 1σ intervals at 1–10 fb⁻¹ — if we want closed
  histogram intervals there, extend the grid in a future run (deliberate deviation, needs approval).
- NSBI direction check is single-lumi (300 fb⁻¹); low-lumi NSBI points come with path A/B.
- Proposed (not applied): fix inverted direction sentence in project `CLAUDE.md`; add
  `PYTHONNOUSERSITE=1` to setup.sh instructions; clean corrupt `~/.local` site-packages.
