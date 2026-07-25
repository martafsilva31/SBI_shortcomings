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

### 2026-07-13 — notebook review (Marta): band convention, and the Δ-median pedestal

Marta reviewed `notebooks/histogram_reproduction.ipynb` and raised three things. All three were real;
the physics conclusion is unchanged, but the way it was being plotted and scored was not defensible.

**1. Our plot did not look like Tei's because the band is a different statistic.** His slide 6 (titled
"NSBI" but the bullet underneath says "For histogram analysis…") builds the band from the **raw** toy
curves shifted by one common constant, so it goes *negative* — his slide-5 caveat says exactly this
("the y-axis is not consistent between toys; some toy curves should go negative w.r.t. the nominal
Asimov curve"). Ours was built from **per-toy min-subtracted** curves, which are ≥ 0 by construction.
Measured on our own scans at 1 fb⁻¹: Tei's convention spans −2.74 … +4.49, ours 0.00 … +0.81 — and
`set_ylim(bottom=0)` was hiding the difference. His slide-6 panel spans ≈ −2.5 … +5.5, so the
common-offset convention reproduces it and per-toy Δ cannot.

**2. "toys 16–84%" vs "±1σ quantile" is only a label — but it exposed a real bug.** 16–84% *is* the
±1σ quantile. The bug behind the question: the **band and the median line were computed from different
quantities** — band from the per-toy Δ curves, line from `torch.median` of the *raw* curves. So the
plotted line was not the 50% quantile of the family the band bracketed, and it fell **outside its own
band at 13% of the μ points at 1 fb⁻¹** (18% at 10 fb⁻¹). Tei's version never had this problem:
subtracting one common constant commutes with taking quantiles, so his band and line are one statistic.
`torch.median` was also still in use here despite being retired on 2026-07-13 (lower-middle-element
bias at even toy counts).

**3. ⚠ The Δ-median curve has a pedestal and can never lie on the Asimov curve.** This is the
substantive finding of the review. Min-subtraction is a **per-toy** operation — each toy is shifted by
*its own* minimum, a different random amount — so it does not commute with the median, and the result is
not a likelihood curve. At μ = μ_true every toy has Δᵢ ≥ 0, with equality only if that toy's μ̂ᵢ landed
*exactly* on the truth, which none do; hence med(Δ) > 0 strictly while the Asimov curve is 0 there by
definition. **A quantity that is positive for every toy cannot have a median of zero.** This is an
inequality — it needs no asymptotics and no model, and it is what makes the convention split below robust.

**How big is the pedestal — and a ⚠ correction to the first draft of this entry.** *Asymptotically*
(μ̂ ~ N(μ_true, σ), parabolic curves) Δᵢ(μ_true) ~ χ²₁, so the pedestal would be med(χ²₁) = 0.455,
**flat in luminosity**. The measured `ped(D)` does **not** do that:

| | 1 fb⁻¹ | 10 fb⁻¹ | 100 fb⁻¹ | 1000 fb⁻¹ | asymptotic |
|---|---|---|---|---|---|
| `ped(D)` | 0.027 | 0.145 | 0.458 | 0.662 | 0.455 (flat) |
| `ratio(D)` | — | — | 0.990 | 1.052 | 1.185 |

It rises monotonically and overshoots — so the 0.458 at 100 fb⁻¹ is a curve crossing 0.455 **on its way
up, not converging to it**. The first draft of this entry reported that agreement as a ✓ confirmation of
the asymptotic prediction; **that was wrong and is retracted.** The width says the same: a direct
simulation of ideal parabolic toys gives pedestal 0.457 ✓ (so the analytics are right) but width ratio
**1.185** — the Δ-median curve *should* be 18% wider than the Asimov. We measure 0.99–1.05. **Neither
asymptotic prediction is reproduced at any luminosity we test, 1000 fb⁻¹ included.**

That is not a bug — it is the project's thesis appearing in its own control channel: these toys are not
asymptotic, and §10 of the notebook says why (the μ̂ distribution is bimodal, sitting on the two fold
branches, not Gaussian). **⚠ Open, not explained:** why `ratio(D)` lands at ~1 instead of the asymptotic
1.185. Do not quote 0.455 as a validated prediction of this analysis.

> **Consequence for the adopted convention.** The Δ convention (adopted 2026-07-13,
> `docs/dimensionality-study.md`) is correct for **widths** — it is what an experiment quotes — but it
> is the **wrong comparator for the "Asimov = median" overlay**, which can only be read in Tei's
> common-offset convention. The 2026-07-13 decision did not draw this distinction. **This goes in the
> Tei message (plan item 7) alongside the median-convention issue already drafted there.**

**Fixes applied to the notebook** (engine untouched; μ grid deliberately left at Tei's committed [0, 4]):

- §6 is now a 2×4 figure: row 1 = per-toy Δ (project convention), row 2 = common offset (reproduces
  slide 6). In both rows the median line is the 50% quantile of the curves the band brackets. `np.median`
  throughout; no `set_ylim`, so row 2's band is free to go negative.
- §8 summary table rebuilt: "1σ half-widths" → **full** 68% interval width (the prose was simply wrong —
  the code always computed `hi − lo`); width ratio now carries a **bootstrap error bar**; ratio reported
  under both conventions; new `ped(D)` column measuring the pedestal; `max|A−M|` measured after removing
  it, so it is the metric that carries the 1–10 fb⁻¹ points where the widths are `nan`.

**Results after the fix** (200 toys, seed 42, μ ∈ [0, 4]):

| lumi [fb⁻¹] | μ̂_A | med(μ̂_toy) | width_A | ratio(Δ) | ratio(raw) | ped(Δ) | max\|A−M\| |
|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.680 | — (open) | — | — | 0.027 | 0.312 |
| 10 | 1.000 | 0.740 | — (open) | — | — | 0.145 | 0.317 |
| 100 | 1.000 | 0.640 | 1.805 | **0.990 ± 0.051** | 0.986 | 0.458 | 0.868 |
| 1000 | 1.000 | 0.920 | 0.805 | **1.052 ± 0.067** | 1.052 | 0.662 | 0.991 |

- **Both width ratios are consistent with 1 within their MC error** — the first time this could be said
  at all. The old bare 0.985 / 1.054 had no error bar and could not be judged against toy noise.
  Conclusion (histogram satisfies Asimov ≈ median at all four luminosities) **stands**.
- Small shifts vs the 2026-07-10 table (0.985 → 0.986, 1.054 → 1.052) are the `torch.median` →
  `np.median` change, as expected.

**⚠ Flagged, not resolved — the toy μ̂ distribution is not centred on 1.** med(μ̂_toy) = 0.64 at
100 fb⁻¹ because **54 of the 200 toys (27%) minimise at or near the μ = 0 boundary**; the distribution is
strongly skewed (quartiles 0.20 / 0.64 / 1.36), not Gaussian. This looks like the rate fold already
characterised in `notebooks/rate_degeneracy.ipynb` and blamed for the under-coverage in
`notebooks/bartlett_correction.ipynb`. It does not undermine the curve-level Asimov agreement (which is
what the property is about) but it **should be reconciled against those two notebooks** rather than taken
at face value from this table. *(Resolved the next day — see below.)*

### 2026-07-14 — reconciliation: the pile-up is the fold's unresolved lower branch (notebook §10)

The flag above is **resolved**; the analysis lives in `notebooks/histogram_reproduction.ipynb` §10
(new stats cell + two figures, computed from the §5 scans plus the seed-42 μ ∈ [0,8] dimensionality
tensors). What was established:

- **First, the definitions the flag glossed over.** "27% at or near μ = 0" = μ̂ ≤ 0.24; exactly *at*
  the boundary is only 7.5%, μ̂ ≤ 0.08 is 18.5%. The criterion matters because the mechanism puts the
  toys *near* zero, not at it.
- **Grid- and binning-independent.** The same seed-42 toys on the μ ∈ [0,8] grid with two independent
  binnings (`occ_1d`, `ctl_1d`) give med(μ̂) = 0.65/0.70 and 24.5–25.5% ≤ 0.24 at 100 fb⁻¹, nothing at
  μ ≥ 4. Physics against the physical μ ≥ 0 floor, not the [0,4] window. (The 1 fb⁻¹ pile-ups are a
  different, flat-likelihood effect: median curve depth NLL(1)−NLL(min) = 0.06, noise minima on *both*
  edges — 26% at μ = 4.)
- **It is the fold, and the attractor is the *shape-weighted* fold pair.** W-shaped toys (two local
  minima: 0% → 7% → 22% → 31% for L = 1 → 1000 fb⁻¹) have branches at 0.076 ± 0.066 / 1.65 ± 0.32
  (100 fb⁻¹) and 0.175 ± 0.069 / 1.28 ± 0.20 (1000 fb⁻¹), mirror-symmetric in κ = √μ about
  κ_v = 0.768 at both luminosities → an **effective fold at μ_eff = κ_v² ≈ 0.59**, well below the
  total-rate fold μ* = 0.957. This
  is where the per-bin folds of this very axis point: the information-carrying high-mass bins fold at
  μ*_b = 0.46–0.68 (`rate_degeneracy`; Fisher-weighted vertex κ ≈ 0.87), while the 0.957 total-rate
  figure is dominated by low-mass bins with no shape information. "μ̂ attracted toward μ*" (bartlett)
  refines to "attracted toward the effective fold pair of the full likelihood".
- **100 → 1000 fb⁻¹ is branch resolution, not disappearance.** At 100 the lower branch (0.08 ± 0.07)
  is pressed against zero → the 27%. At 1000 it detaches (0.18 ± 0.07), the boundary empties, ~13% of
  toys (μ̂ ≤ 0.24) sit on the resolved branch: bimodal, left-skewed, med 0.92, mean 0.859 — **identical to
  `bartlett_correction`'s 0.859 ± 0.417 on the same tensors**, and consistent with `rate_degeneracy`'s
  L^(−1/4) and 0.86-bias narrative. Neither notebook had measured a median, quartile, or boundary
  fraction, so nothing was ever in contradiction — the flag added a new statistic to a known mechanism.
- **Consequences.** The §8 med(mu_toy) column is fold-shaped, not a calibration bias; the curve-level
  Asimov ≈ median conclusion is untouched; the bartlett coverage dip at μ_true ∈ {0.957, 1} is the same
  lower branch escaping the Δ = 1 interval. Nothing new to fix (κ = √μ provably cannot remove the fold;
  the working mitigation remains prong 3's Neyman construction).

### Open follow-ups
- **Convention split (overlay vs width) into the Tei message** (new, 2026-07-13 — plan item 7).
- μ scan window [0, 4] too narrow to contain 1σ intervals at 1–10 fb⁻¹ — if we want closed
  histogram intervals there, extend the grid in a future run (deliberate deviation, needs approval;
  Marta declined the widening on 2026-07-13 to keep the byte-for-byte reproduction of Tei's setup).
- NSBI direction check is single-lumi (300 fb⁻¹); low-lumi NSBI points come with path A/B.
- Proposed (not applied): fix inverted direction sentence in project `CLAUDE.md`; add
  `PYTHONNOUSERSITE=1` to setup.sh instructions; clean corrupt `~/.local` site-packages.

### 2026-07-16 — from-scratch redo (plan approved by Marta; notebook rewritten)

*(Note on the name: earlier entries call the author "Tei"; Marta corrected the name to **Tae**
on 2026-07-16 — new text uses Tae.)*

Everything above this line describes the old 200-toy reproduction and its conventions; it is kept
as history. The study was redone from scratch today with a protocol that fixes the problems Marta
raised (toy count, per-toy best-fit subtraction, luminosity grid, band meaning) plus the ones the
review surfaced. `notebooks/histogram_reproduction.ipynb` was **rewritten in place** (the old
executed state survives in git before commit `adf067b`'s successor).

**Protocol.** Per toy, t(μ) = −2[lnL(μ) − lnL(μ̂)] with the per-toy best fit subtracted; the
Asimov-property test is *measured vs predicted quantiles*: in the large-sample limit t(μ) follows
a noncentral χ² (1 dof) with noncentrality t_A(μ), so the median and 16/84% quantiles of t across
toys are compared against that distribution's quantiles at every μ. The known 0.455 offset at
μ = 1 and the 1.185 width factor are *inside the prediction* — no convention war, no common-offset
shifts. 2000 toys (one random stream each, master seed 42), 11 luminosities {1, 10, 30, 100, 140,
200, 300, 500, 700, 1000, 3000} fb⁻¹, μ ∈ [0, 4] with 161 points (range kept at Tae's on Marta's
decision; open intervals reported as open). Compute in `scripts/asimov_scan.py` (the one new file
of the redo); the notebook only loads, summarises, plots.

**Closure.** The legacy configuration (200 toys, single seed block, 101-point grid, Tae's
float32 grid values) reproduces the pre-redo tensors to float32 rounding (max |Δ| ≤ 4×10⁻³ on
curves of magnitude 10⁴–10⁵, ≈ 2×10⁻⁷ relative) and the published numbers exactly: width ratios
0.990 / 1.052 at 100/1000 fb⁻¹, Asimov μ̂ = 1.000, open intervals at 1–10 fb⁻¹. The comparator
machinery passes its self-test on ideal parabolic likelihoods (0.456 vs 0.455; 1.177 vs 1.185
within grid resolution).

**Results (2000 toys).** Asimov μ̂ = 1.000 at all 11 points. The sharper toy statistics change
the headline: the median-curve width ratio (large-sample expectation **1.185**, not 1) is
0.936 ± 0.016 at 100 fb⁻¹, 0.972 ± 0.018 at 300, 1.091 ± 0.020 at 1000, and reaches
1.157 ± 0.017 only at 3000 fb⁻¹ — i.e. **the histogram analysis is measurably below its
large-sample behaviour everywhere up to ~1000 fb⁻¹**, a departure invisible at the old ±0.05
precision (where "consistent with 1" was the best that could be said, against the wrong
reference value). The measured median of t(1) rises from 0.05 (1 fb⁻¹) through 0.71 (500 fb⁻¹)
back to 0.54 (3000 fb⁻¹) against a flat predicted 0.455; the 68% interval covers μ = 1 for 92%
of toys at 1 fb⁻¹ (over-coverage) and only ~0.60 at 200–700 fb⁻¹ (the known interference rate
degeneracy). The fraction of toys with μ̂ = 0 falls from ~11–15% (1–30 fb⁻¹) to 0 above
700 fb⁻¹. All numbers with definitions in the notebook's §4 table; errata table in §7.

**Files.** `scripts/asimov_scan.py` (engine + downloader + statistic definitions, importable by
both notebooks); `output/asimov_scan/{hist,closure}/` (gitignored scans);
`notebooks/histogram_reproduction.ipynb` (rewritten, executed).
