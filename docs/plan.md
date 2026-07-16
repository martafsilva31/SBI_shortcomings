# SBI_shortcomings — Task Board

## In progress

- [ ] Phase 1 NSBI reproduction — **path A** (Tei's checkpoints); waiting on his `run/h4l/` dir (see
      docs/env-setup-and-code-walkthrough.md §8). GPU node available. Ready when it lands: patch
      drafts + path B launch kit in docs/nsbi-repro-prep.md. ⚠ **Tei unresponsive as of 2026-07-13**
      — everything downstream of path A (items 13, 8d-execution, the NSBI overlay) is parked; work
      is being routed to the unblocked prongs meanwhile
- [ ] Env-setup Logbook entry — **published to Notion 2026-07-13** (page ID in docs/notion_map.json;
      connector auth resolved, 4 Logbook + 2 Notes entries live). Remaining: Marta's format sign-off
      on it as the template for the rest — that sign-off is what still gates item 15

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
- [x] Toy-count scaling check (2026-07-11, via Phase-3 item 2 — and it exposed the convention issue)
- [ ] NSBI overlay on the same panels (after path A lands), incl. an effective-occupancy axis for
      the comparison — absorbs the retired "refine the occupancy metric" item (2026-07-13: its
      width-ratio motivation vanished under the Δ convention; the surviving μ̂-scatter signal is
      noted in the study note)

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
      provisional Δ-primary convention (adopted 2026-07-13) + wifi-method + [2, 1.83] confirmations.
      **Drafted 2026-07-13 — full text parked in docs/phase3-fixes.md** (Tei unresponsive; send when
      he resurfaces). Nothing in the project now blocks on his reply to it: the wifi ask (§3 of the
      draft) was self-answered — see below

M3 — prong-2 gate:
- [x] 8. MC-stat relevance per config (dimensionality notebook §10) → BB-lite go/no-go (Marta)
      → DECIDED 2026-07-13 (Marta): MC-stat uncertainties go in, in every scenario
- [x] 8a. BB-lite closed-form profiling in stats/binned.py (2026-07-13: γ̂ closed form, exactness
      checks pass, sub-repo file #2 modified)
- [x] 8b. BB-lite overlay in the dimensionality notebook (2026-07-13: widths +≤2.6% at 1000 fb⁻¹,
      ratio_D shift ≤0.013 — no conclusion changes; §11)
- [x] 8c. BB-lite coverage spot-check (2026-07-13: 0.594→0.608 / 0.644→0.649 at 1000 fb⁻¹ —
      under-coverage persists, fold attribution confirmed; bartlett §10)
- [x] 8c-bis. **"Wi-Fi ensemble" method identified (2026-07-13)** — not a garbled transcription:
      **wᵢfᵢ ensembles**, Benevedes & Thaler, arXiv:2506.00113 (frozen basis nets fᵢ, fitted weights
      wᵢ, analytic sandwich covariance → Gong–Samaniego inflation of σ²_μ; no bootstraps). Written up
      as Ref. 17 (literature.md) + foundations §15.1; PDF in docs/refs/. ⚠ Its guarantees are
      **asymptotic** — a candidate mitigation to be *tested*, not assumed, in our regime
- [ ] 8d. NSBI MC-stat treatment — design note. **Method side settled and now validated end-to-end**
      (template = BB-lite; network = wifi + f₀, prototyped in 8e); *execution* still gated on path A
      (needs Tei's working r_sbi/bkg basis; our path-B never left the ln2 floor)
- [x] 8e. **wifi prototype on `sig_over_bkg` — DONE 2026-07-14.** §17.3 answered: the GS inflation
      scales as N_mix/N_fit (measured 1.0001→1.037) — structurally invisible in the non-asymptotic
      regime whenever the ratio trains on more events than the measurement uses; **hypothesis 3
      survives its strongest challenger**. With Ref. 17's f₀=1 basis term — omitted in run 1, caught
      by a 23-agent adversarial engine review; run-1's "wifi collapses" (0.35–0.59 at N=10⁴) was a
      protocol artefact — wifi+GS coverage is nominal at all N_mix (0.64–0.72 at N=10⁴, ±0.021),
      κ̂ bias +0.001…+0.003 ≈ truth control; the naive ensemble still collapses (0.28–0.63), exactly
      as the paper warns. wifi enters 8d as MC-stat *propagation*, not as an Asimov-failure
      mitigation. Basis: 16/16 nets (condor 5086805, true val_loss 0.41928–0.42331). Study note:
      phase3-fixes.md §2026-07-14; verdict: notebooks/wifi_ensembles.ipynb §7

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
- [x] 14. Re-execute all Phase-3 notebooks end-to-end (2026-07-13: bartlett + dimensionality with
      the Δ/BB-lite updates, then rate_degeneracy, deepset_coverage, deepset_neyman — all headless
      in-container; documented numbers reproduced: fold range, 0.0% rate Fisher share, v0 prior
      collapse, v1 Neyman 0.684/0.694 at the fold byte-identical)
- [ ] 15. Logbook entries + commits — Notion auth is no longer a gate (6 entries live since
      2026-07-13); remaining gates: Marta's format sign-off, and the Phase-3 docs/notebooks are still
      uncommitted in the working tree (8 files + this session's 4 doc updates — commit proposed)

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
