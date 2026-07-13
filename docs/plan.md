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
      notebooks/dimensionality_study.ipynb — headline REVISED 2026-07-11: the 0.44 (4D) ratio is
      convention-dependent (0.44 torch-median-raw / 0.71 np-median-raw / 1.10 per-toy-Δ median — no
      dimensionality effect under the Δ convention); convention DECIDED provisionally 2026-07-13
      (Marta): Δ-primary + np-median-raw secondary, torch.median retired — ⚠ not yet team-ratified,
      goes into the item-7 Tei message
- [ ] Refine the occupancy metric (mean occupancy does not collapse the grid — distribution matters)
- [x] Toy-count scaling check (2026-07-11, via Phase-3 item 2 — and it exposed the convention issue)
- [ ] NSBI overlay on the same panels (after path-B reproduction)

### Phase 3 — Proposed fixes (plan approved 2026-07-11; docs/phase3-fixes.md is the study note)

M1 — Bartlett/coverage prong (notebooks/bartlett_correction.ipynb):
- [x] 1. Coverage-vs-level curves on existing toys
- [x] 2. 1000-toy sharpening (coverage + width-ratio error bars)
- [x] 3. Multi-truth toy ensembles (μ_true grid incl. the fold μ*=0.957)
- [x] 4. Proper per-hypothesis Bartlett intervals → prong-1 verdict

M2 — rate-degeneracy mechanism (notebooks/rate_degeneracy.ipynb):
- [x] 5. Per-bin fold characterisation + foundations addendum
- [x] 6. κ=√μ reparametrisation check (LR-invariance demo + what changes)
- [ ] 7. Tei message (Marta sends): fold mechanism + median-convention issue + ratification of the
      provisional Δ-primary convention (adopted 2026-07-13)

M3 — prong-2 gate:
- [x] 8. MC-stat relevance per config (dimensionality notebook §10) → BB-lite go/no-go (Marta)
      → DECIDED 2026-07-13 (Marta): MC-stat uncertainties go in, in every scenario
- [ ] 8a. BB-lite closed-form profiling in stats/binned.py (sub-repo; approved 2026-07-13)
- [ ] 8b. BB-lite overlay in the dimensionality notebook (μ̂ + Δ-convention width ratios, 32 points)
- [ ] 8c. BB-lite coverage spot-check (bartlett notebook, 1D/4D @ 100/1000 fb⁻¹)
- [ ] 8d. NSBI MC-stat treatment — design note (gated on path A)

M4 — prong-3 deep-set prototype:
- [x] 9. Design note (files named for approval) — docs/phase3-fixes.md §prong-3
- [x] 10. Generator + training script (2026-07-12; smoke-tested)
- [x] 11. Condor GPU training (2026-07-12: full 20k steps, best val_loss 0.0249)
- [x] 12. Coverage evaluation (2026-07-12): v0 fails coverage (prior-dominated at low L, fold bias
      at high L) BUT learns L-aware sigma
- [x] 12b. v1 Neyman calibration (approved + done 2026-07-12): notebooks/deepset_neyman.ipynb —
      no under-coverage anywhere (0.67–0.87), exact at the informative points, honest widths
      (full-range at 1 fb⁻¹, 0.9 at 1000 fb⁻¹). Prong 3 closed with a working mitigation.

Blocked / closure:
- [ ] 13. NSBI coverage+Bartlett audit on path A (blocked on Tei)
- [ ] 14. Re-execute all Phase-3 notebooks end-to-end (validation)
- [ ] 15. Logbook entries + commits (gated on Notion auth + format sign-off)

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
