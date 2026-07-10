# SBI_shortcomings — Task Board

## In progress

- [ ] Phase 1 NSBI reproduction — **path A** (Tei's checkpoints); waiting on his `run/h4l/` dir (see
      docs/env-setup-and-code-walkthrough.md §8). GPU node available. Ready when it lands: patch
      drafts + path B launch kit in docs/nsbi-repro-prep.md
- [ ] Env-setup Logbook entry — local draft rewritten via /writeup (2026-07-10); awaiting Marta's
      format sign-off (first entry of the project) + Notion connector auth before the push

## Backlog

### Phase 1 — Reproduce Tei's results

- [ ] Reproduce NSBI NLL scan at 1/10/100/1000 fb⁻¹ — verify Asimov ≠ median(toys) at 1–10 fb⁻¹
- [ ] Add the missing extended Poisson term to the NSBI NLL (flagged by Tei 2026-07-09) and re-run — only after reproducing his results as-is (patch drafted: docs/nsbi-repro-prep.md §2)
- [ ] Document reproduction results in first logbook entry (hist study note ready: docs/hist-reproduction.md; env-setup entry drafted, see In progress)

### Phase 2 — Dimensionality study

- [x] Implement 1D–4D histogram analysis, 4 luminosities + fixed-16-bin control (2026-07-10):
      notebooks/dimensionality_study.ipynb — headline: width ratio 1.08 (1D) → 0.44 (4D) at
      1000 fb⁻¹, Asimov property breaks with cell count, no ML involved
- [ ] Refine the occupancy metric (mean occupancy does not collapse the grid — distribution matters)
- [ ] Toy-count scaling check (200 → 1000 toys) on the extreme points
- [ ] NSBI overlay on the same panels (after path-B reproduction)

### Phase 3 — Proposed fixes

- [ ] Add MC statistical uncertainties to both histogram and NSBI likelihoods
- [ ] Investigate alternative test statistics (Bartlett correction or equivalent) for non-asymptotic regime
- [ ] Prototype toy-ensemble training: train CARL NN on multiple pseudo-experiments (deep sets or equivalent) instead of single Asimov dataset

### Phase 4 — Dataset extension (coordinate with Lydia)

- [ ] Check FCC group central repository for available parton-level samples with EFT effects
- [ ] Evaluate candidate process H→SS at FCC centre-of-mass energy
- [ ] Decide whether to switch dataset or extend study to both

### Phase 5 — Write-up

- [ ] Write short paper / ATLAS internal note documenting the Asimov failure mode
- [ ] Include proposed fixes with quantitative comparison
- [ ] Target venue: [TBD — decide with team]

## Done

- [x] Histogram reproduction (2026-07-10): μ mode at 1/10/100/1000 fb⁻¹ — Asimov ≈ median(toys)
      confirmed at all four points (1σ width ratios 0.985/1.054 where intervals close; Asimov
      μ̂ = 1.000 exactly everywhere). Study note: docs/hist-reproduction.md
- [x] NSBI direction question resolved (2026-07-10): median(toys) is the narrower/steeper curve
      (width ratio 0.874 at 300 fb⁻¹, measured on Tei's committed scans); context.md clarified
- [x] NSBI prep drafts (2026-07-10): path B launch kit + extended-Poisson-term + chunk-over-toys
      patch drafts — docs/nsbi-repro-prep.md (nothing applied or trained)
- [x] Phase 2 design note (2026-07-10): docs/dimensionality-study.md (occupancy grid, observables,
      reuse-only implementation route)
- [x] Foundations extended (2026-07-10): new §§12–14 (SBI overconfidence & coverage, Bartlett
      correction, toy-ensemble + deep sets) in docs/notes/sbi_foundations.md
- [x] [2, 1.83] xs factors (2026-07-10): proposed answer documented (channel counting × common
      gg→ZZ k-factor) — awaiting Tei's confirmation
- [x] First team meeting (2026-05-20): established key finding, agreed on methodology and next steps
- [x] Bootstrap SBI_shortcomings project container (2026-06-02)
- [x] Clone asimov-aymptotics-check fork with real code + Tei's reference results (2026-06-02)
- [x] Obtain CERNBox dataset location from Tei (2026-07-09): `zz4l/ggzz4l_sm/` in his public share
- [x] Environment set up (2026-07-09): dataset downloaded locally, Apptainer container built + verified
      (imports OK, CPU histogram path runnable), Tei's code walkthrough documented
- [x] Publish container repo to GitHub (2026-07-09): private `martafsilva31/SBI_shortcomings`, clean history
- [x] Confirm CARL configs not recoverable locally/CERNBox/history (2026-07-09) → NSBI via path A
