# SBI_shortcomings — Task Board

## In progress

- [ ] Set up environment: build Apptainer container + download local dataset copy from Tei's CERNBox share (link received 2026-07-09; see docs/context.md)

## Backlog

### Phase 1 — Reproduce Tei's results

- [ ] Reproduce histogram NLL scan at 1/10/100/1000 fb⁻¹ — verify Asimov ≈ median(toys)
- [ ] Reproduce NSBI NLL scan at same luminosities — verify Asimov ≠ median(toys) at 1–10 fb⁻¹
- [ ] Add the missing extended Poisson term to the NSBI NLL (flagged by Tei 2026-07-09) and re-run — only after reproducing his results as-is
- [ ] Document reproduction results in first logbook entry

### Phase 2 — Dimensionality study

- [ ] Implement 4D histogram analysis (extend beyond 1D M₄ℓ) at 1 fb⁻¹
- [ ] Compare Asimov vs. median(toys) for 1D, 2D, 3D, 4D histograms to isolate dimensionality effect
- [ ] Plot NLL scan curves for all histogram dimensionalities

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

- [x] First team meeting (2026-05-20): established key finding, agreed on methodology and next steps
- [x] Bootstrap SBI_shortcomings project container (2026-06-02)
- [x] Clone asimov-aymptotics-check fork with real code + Tei's reference results (2026-06-02)
- [x] Obtain CERNBox dataset location from Tei (2026-07-09): `zz4l/ggzz4l_sm/` in his public share
