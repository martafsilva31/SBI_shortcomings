# SBI_shortcomings — Task Board

> Roadmap only. The discrete task list and its live status are in Notion (Tasks + Plan Steps);
> this file is the human-readable map. Last reconciled with the boards: 2026-07-25.

## Where the project stands (2026-07-25)

The 2026-07-21 audit found the **statistical engine sound and the narrative partly wrong**, and
turned up an exact result that reframes the whole question. The 2026-07-22 HWW meeting put the
resulting open questions to the room; **none was resolved**, so the project is in a
re-verification pause rather than moving forward.

**What is solid.** The per-toy test statistic really does subtract each toy's own best fit (every
per-toy minimum asserts to exactly zero on the stored tensors); the noncentral χ² comparator
matches scipy to machine precision; toy streams are genuinely independent; and every histogram
headline number reproduces under a reimplementation that never imports `scripts/asimov_scan.py`
(Asimov best fit 1.000 at all 11 luminosities, width ratios 0.936 / 0.972 / 1.091 / 1.157,
median of t(1) rising 0.054 → 0.710 → 0.542, coverage 0.917 at 1 fb⁻¹).

**The new headline — the mean identity.** The Asimov dataset is by construction the mean dataset
and −2lnL is affine in the observed counts, so the toy-average of −2lnL(μ) equals the Asimov curve
plus a μ-independent constant — exactly, at any sample size, with no asymptotic assumption.
Verified on the project's own tensors to Monte-Carlo noise for both the histogram and the
exact-ratio unbinned analysis. The **median** obeys no such identity: it lies strictly above the
Asimov curve at every μ even in the perfect asymptotic limit, so the Asimov-vs-median overlay can
never agree by construction.

**What the audit contradicted** (details in `docs/context.md` § Key finding):
1. the histogram is not a clean control — width ratio 0.92–1.16 vs a 1.185 large-sample
   expectation, and undefined at 1–10 fb⁻¹ (2000/2000 toys open at 1 fb⁻¹);
2. the exact-vs-network contrast is confounded by a 0.19% rate-term normalisation mismatch, which
   alone moves the Asimov best fit to 0.175 and inflates the interval by 34% at 300 fb⁻¹;
3. the 1.38 "direction flips" figure is 1.2σ on 200 toys and appears only in prose;
4. "his committed 0.874" is our own re-measurement of Tae's tensors — his slides carry no
   Asimov-vs-median numbers.

⚠ Audit findings are mostly **singly sourced** (the re-check phase covered 31 of 83 findings, 29
surviving). Treat them as checkable leads; each carries a reproducible script. That is what the
verification walkthrough below is for.

## In progress

- [ ] **Verification walkthrough** — eight checks, ~85 min, none yet ticked. Gates every claim that
      goes on a slide or into a write-up. Order: (1) the mean identity on one luminosity; (2) the
      median-gap proof by hand; (3) Cowan Eq. 32 paragraph; (4) per-toy subtraction assertion;
      (5) hist notebook §4 table; (6) the 1.185 self-test; (7) the 0.19% rate-term mismatch;
      (8) the 0.897 ratio in nsbi cell 16. Deliberately deferred: the 1e-300 clamp census, the
      200 fb⁻¹ dip in the width ratio, the med(w_t) column definition, the checkpoint-continuity
      question in nsbi §7
- [ ] **Fix the NSBI rate-term normalisation and re-run the network scans** — take the total rate
      from the event sample rather than `ggzz4l_xs.json`, confirm the Asimov best fit returns to
      1.000 under an exact ratio, then re-measure every exact-vs-network comparison. Comparisons
      *among* the three network variants share a normalisation and stay clean
- [ ] **Commit + write up the Phase-3 and redo work** — gated only on Marta's Logbook format
      sign-off on the 2026-07-13 env-setup entry
- [ ] **Logbook format sign-off** (Marta) — the smallest open item, blocking the most

## Open questions put to the HWW meeting, 2026-07-22 — all still open

1. How should the median be taken — subtract each toy's own best fit first (Δ-primary, provisional
   since 2026-07-13), or take the median of the raw curves as Tae did? The two give materially
   different conclusions, and no one in the room was sure.
2. Are the NLL curves supposed to agree at all, or only the quantiles / extracted CLs?
3. Are these 11-panel overlays the best way to study when the property holds or breaks?
4. Does signal–background interference make this process a poor candidate for Asimov to hold?

Until (1) is settled the Δ-primary convention stays **provisional and un-ratified**, and every
width-ratio number in the project carries that caveat.

## Backlog

### Phase 1 — Reproduce Tae's results (absorbed into the 2026-07-16 redo)

- [x] Reproduce NSBI NLL scan — done 2026-07-16, three ways (exact ratio / his sbi net + our sig
      net / decomposition), 11 luminosities × 2000 toys
- [x] Extended Poisson term — included from the start in every redo scan; its omission quantified
      as an erratum (nsbi notebook §7)
- [x] The shared `sig_over_bkg` is a 2ℓ2ν network, not a 4ℓ one — bit-exact closure of Tae's
      committed scans is impossible; agreement can only ever be statistical
- [x] The ln 2 loss floor on path B was near-optimal, not a training failure (achievable margin
      7.5e-4, ROC 0.513) — this is why every path-A-gated item stays blocked
- [ ] ⚠ **Pending re-measurement**: all exact-vs-network conclusions from the redo, once the
      rate-term normalisation is fixed. The earlier claims ("exact ratio shows no dramatic
      failure, the network distorts network-specifically", 0.897, 1.38) are not safe as they stand
- [ ] Document the reproduction — `/writeup` of the two study notes after the walkthrough passes

### Phase 2 — Dimensionality study

- [x] 1D–4D histogram analysis, 4 luminosities + fixed-16-bin control (2026-07-10):
      `notebooks/dimensionality_study.ipynb` — headline REVISED 2026-07-11: the 0.44 (4D) ratio is
      convention-dependent (0.44 torch-median-raw / 0.71 np-median-raw / 1.10 per-toy-Δ median — no
      dimensionality effect under the Δ convention); convention provisional since 2026-07-13,
      still un-ratified (see open question 1)
- [x] Toy-count scaling check (2026-07-11, via Phase-3 item 2 — it exposed the convention issue)
- [ ] NSBI overlay on the same panels, incl. an effective-occupancy axis — **blocked on path A**

### Phase 3 — Proposed fixes (plan approved 2026-07-11; `docs/phase3-fixes.md` is the study note)

M1 — Bartlett/coverage prong (`notebooks/bartlett_correction.ipynb`):
- [x] 1. Coverage-vs-level curves on existing toys
- [x] 2. 1000-toy sharpening (coverage + width-ratio error bars)
- [x] 3. Multi-truth toy ensembles (μ_true grid incl. the fold μ*=0.957)
- [x] 4. Proper per-hypothesis Bartlett intervals → prong-1 verdict

M2 — rate-degeneracy mechanism (`notebooks/rate_degeneracy.ipynb`):
- [x] 5. Per-bin fold characterisation + foundations addendum
- [x] 6. κ=√μ reparametrisation check (LR-invariance demo + what changes)
- [ ] 7. Tei message (Marta sends) — drafted 2026-07-13, full text parked in
      `docs/phase3-fixes.md`; Tei unresponsive, send when he resurfaces. Nothing blocks on his
      reply: the wifi ask (§3 of the draft) was self-answered

M3 — MC-stat prong:
- [x] 8. MC-stat relevance per config → decided 2026-07-13 (Marta): MC-stat uncertainties go in,
      in every scenario
- [x] 8a. BB-lite closed-form profiling in `stats/binned.py` (γ̂ closed form, exactness checks pass)
- [x] 8b. BB-lite overlay in the dimensionality notebook (widths +≤2.6% at 1000 fb⁻¹, ratio_D shift
      ≤0.013 — no conclusion changes; §11)
- [x] 8c. BB-lite coverage spot-check (0.594→0.608 / 0.644→0.649 at 1000 fb⁻¹ — under-coverage
      persists, fold attribution confirmed; bartlett §10)
- [x] 8c-bis. "Wi-Fi ensemble" identified as **wᵢfᵢ ensembles**, Benevedes & Thaler,
      arXiv:2506.00113 (frozen basis nets fᵢ, fitted weights wᵢ, analytic sandwich covariance →
      Gong–Samaniego inflation of σ²_μ; no bootstraps). Ref. 17 + foundations §15.1. ⚠ Its
      guarantees are **asymptotic** — a candidate to be tested, not assumed, in our regime
- [x] 8e. wifi prototype on `sig_over_bkg` (2026-07-14): GS inflation scales as N_mix/N_fit
      (1.0001→1.037) — structurally invisible in the non-asymptotic regime whenever the ratio
      trains on more events than the measurement uses; **hypothesis 3 survives its strongest
      challenger**. With Ref. 17's f₀=1 basis term, wifi+GS coverage is nominal at all N_mix
      (0.64–0.72 at N=10⁴, ±0.021), κ̂ bias +0.001…+0.003; the naive ensemble still collapses
      (0.28–0.63). wifi enters as MC-stat *propagation*, not as an Asimov-failure mitigation
- [ ] 8d. NSBI MC-stat design note — method settled (template = BB-lite, network = wifi + f₀);
      execution **blocked on path A**

M4 — deep-set prototype:
- [x] 9–11. Design note, generator + training script, Condor GPU training (best val_loss 0.0249)
- [x] 12. Coverage evaluation: v0 fails coverage (prior-dominated at low L, fold bias at high L)
      but learns an L-aware sigma
- [x] 12b. v1 Neyman calibration: `notebooks/deepset_neyman.ipynb` — no under-coverage anywhere
      (0.67–0.87), exact at the informative points, honest widths. Prong 3 closed with a working
      mitigation

Blocked / closure:
- [ ] 13. NSBI coverage + Bartlett audit on path A — **blocked on Tei**
- [x] 14. Re-execute all Phase-3 notebooks end-to-end (2026-07-13, headless in-container;
      documented numbers reproduced byte-identically at the fold)
- [ ] 15. Logbook entries + commits — gated on Marta's format sign-off

### Phase 4 — Dataset extension (coordinate with Lydia)

- [ ] Check FCC group central repository for available parton-level samples with EFT effects
- [ ] Evaluate candidate process H→SS at FCC centre-of-mass energy
- [ ] Decide whether to switch dataset or extend study to both

### Phase 5 — Write-up

- [ ] Short paper / ATLAS internal note documenting the Asimov failure mode
- [ ] Include proposed fixes with quantitative comparison
- [ ] Target venue: [TBD — decide with team]

## Done

- [x] **HWW status update presented (2026-07-22)** — 8-slide Keynote deck, histogram only, built
      around the median-convention question and the two result grids; four questions put to the
      room, none resolved. `docs/presentations/HWW_status_update_22072026.pdf`
- [x] **Independent audit of the study's numbers (2026-07-21)** — 83 findings, 31 adversarially
      re-checked, 29 surviving; engine sound, narrative partly contradicted; the mean identity
      found in the process
- [x] Histogram reproduction (2026-07-10, redone 2026-07-16) — study note
      `docs/hist-reproduction.md`. ⚠ its "Asimov ≈ median at all luminosities" reading is retired
- [x] NSBI prep drafts (2026-07-10): path B launch kit, extended-Poisson-term and chunk-over-toys
      patch drafts — `docs/nsbi-repro-prep.md`
- [x] Phase 2 design note (2026-07-10): `docs/dimensionality-study.md`
- [x] Foundations extended (2026-07-10): §§12–14 in `docs/notes/sbi_foundations.md`
- [x] [2, 1.83] xs factors (2026-07-10): channel counting × common gg→ZZ k-factor — awaiting Tei's
      confirmation
- [x] First team meeting (2026-05-20): key finding, methodology, next steps
- [x] Bootstrap project container (2026-06-02) + clone `asimov-aymptotics-check` fork
- [x] Obtain CERNBox dataset location from Tei (2026-07-09): `zz4l/ggzz4l_sm/`
- [x] Environment set up (2026-07-09): dataset downloaded, Apptainer container built + verified
- [x] Publish container repo to GitHub (2026-07-09): private `martafsilva31/SBI_shortcomings`
- [x] Confirm CARL configs not recoverable (2026-07-09) → NSBI via path A
