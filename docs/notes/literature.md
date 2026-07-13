# Literature — NSBI Shortcomings in the Non-Asymptotic Regime

> Annotated bibliography for the **SBI_shortcomings** project. PDFs in
> [`docs/refs/`](../refs/) (gitignored — local artifacts, re-downloadable from the links).
>
> Format: each entry has **(a) full citation**, **(b) one-paragraph annotation**
> explaining what it does and why it matters for *this* project.
>
> Status legend: **[CORE]** = directly underpins the methodology or the failure mode
> under study; **[SUPPORTING]** = statistics/ML background and mitigation anchors.
>
> All arXiv IDs and journal references verified against arxiv.org / publisher pages
> on 2026-07-09; every PDF title page checked against its citation. Refs. 14–16
> verified against Inspire / Crossref / arXiv on 2026-07-13 (no local PDFs yet).
> Ref. 17 verified against arXiv on 2026-07-13 (PDF downloaded, title page checked).

---

## A. Founding evidence (internal)

### [CORE] Ref. 0 — Tei's slides (2026-05-20)

**Tae Hyoun Park**, *Asimov == median of toy pseudo-experiments check for histogram &
NSBI-based test statistics*, 7 slides, shared at the first team meeting 2026-05-20.
Local: [`docs/presentations/asimov_vs_medtoys_test_statistic_studies.pdf`](../presentations/asimov_vs_medtoys_test_statistic_studies.pdf).

> The founding evidence of the project. Slide 2: the SBI mixture model (both forms);
> slide 3: the two likelihoods side by side, with the note that the "total rate" Poisson
> term is analytic (not NN-estimated); slide 4: the yields table (⚠ typo'd column headers,
> cite numbers from `docs/context.md`); slide 5: the toy recipe w_toy ~ 𝒫(λ=w) and its
> y-axis caveat; slide 6: **histogram** result, Asimov ≈ median(toys) at all four
> luminosities (⚠ the slide title says "NSBI" but the bullet and curves are the histogram
> analysis); slide 7: the **NSBI** result — median(toys) departs from Asimov at 1–100 fb⁻¹,
> converging only at 1000 fb⁻¹ "when the Poisson term takes over". ⚠ All curves predate
> the extended-Poisson-term fix (see `docs/context.md` § Known code issues).

---

## B. Core methodology

### [CORE] Ref. 1 — CARL (Cranmer, Pavez, Louppe, 2015)

**K. Cranmer, J. Pavez, G. Louppe**, *Approximating Likelihood Ratios with Calibrated
Discriminative Classifiers* (2015).
arXiv: [1506.02169](https://arxiv.org/abs/1506.02169) — **arXiv-only, no journal version;
do not let a citation manager invent a venue.**
Local: [`refs/1506.02169_CARL_calibrated_classifiers.pdf`](../refs/1506.02169_CARL_calibrated_classifiers.pdf).

> The CARL method: a classifier trained to separate two samples yields the density ratio
> via the likelihood-ratio trick r(x) = s(x)/(1−s(x)), plus the calibration step that
> gives the method its name. This is exactly what `models/carl.py` implements — the two
> classifiers estimating p_sig/p_ref and p_sbi/p_ref (p_ref = p_bkg) are CARL networks.
> The project's key finding is a statement about the finite-sample behaviour of this
> procedure, so this paper defines the object under test.

### [CORE] Ref. 2 — Asymptotic formulae (Cowan, Cranmer, Gross, Vitells, 2011)

**G. Cowan, K. Cranmer, E. Gross, O. Vitells**, *Asymptotic formulae for likelihood-based
tests of new physics*, **Eur. Phys. J. C 71 (2011) 1554**; Erratum: **Eur. Phys. J. C 73
(2013) 2501** (corrects Eqs. (74)–(75); always cite together).
arXiv: [1007.1727](https://arxiv.org/abs/1007.1727).
DOI: [10.1140/epjc/s10052-011-1554-0](https://doi.org/10.1140/epjc/s10052-011-1554-0),
erratum [10.1140/epjc/s10052-013-2501-z](https://doi.org/10.1140/epjc/s10052-013-2501-z).
Local: [`refs/1007.1727_Cowan_asymptotic_formulae.pdf`](../refs/1007.1727_Cowan_asymptotic_formulae.pdf),
[`refs/1007.1727_erratum_EPJC73_2501.pdf`](../refs/1007.1727_erratum_EPJC73_2501.pdf).

> Defines the **Asimov dataset** (Sec. 3.2: replace the data by their expectation values)
> and proves that, *asymptotically*, the test statistic evaluated on it reproduces the
> **median** significance/limits of an ensemble of experiments. The property whose
> breakdown this project studies is the operational version of exactly this statement.
> Careful when writing: what is proven is a median property of the *test statistic* in
> the asymptotic limit — not equality of full NLL scan curves at ~6 events (see
> `sbi_foundations.md` §3).

### [CORE] Ref. 3 — ATLAS NSBI implementation (2025)

**ATLAS Collaboration**, *An implementation of neural simulation-based inference for
parameter estimation in ATLAS*, **Rep. Prog. Phys. 88 (2025) 067801**.
arXiv: [2412.01600](https://arxiv.org/abs/2412.01600).
DOI: [10.1088/1361-6633/add370](https://doi.org/10.1088/1361-6633/add370).
Local: [`refs/2412.01600_ATLAS_NSBI_implementation.pdf`](../refs/2412.01600_ATLAS_NSBI_implementation.pdf).

> The production-grade ATLAS NSBI methodology: per-event density-ratio likelihoods built
> from classifier ensembles, calibration, diagnostics, and the treatment of systematics.
> This is the "grown-up" version of the pipeline in `asimov-aymptotics-check/` and the
> methodological context in which a non-asymptotic failure mode would matter most —
> any conclusion here is implicitly a caveat (or a validation task) for this framework.

### [CORE] Ref. 4 — ATLAS off-shell H*→ZZ→4ℓ with NSBI (2025)

**ATLAS Collaboration**, *Measurement of off-shell Higgs boson production in the
H*→ZZ→4ℓ decay channel using a neural simulation-based inference technique in 13 TeV
pp collisions with the ATLAS detector*, **Rep. Prog. Phys. 88 (2025) 057803**.
arXiv: [2412.01548](https://arxiv.org/abs/2412.01548).
DOI: [10.1088/1361-6633/adcd9a](https://doi.org/10.1088/1361-6633/adcd9a).
Local: [`refs/2412.01548_ATLAS_offshell_H4l_NSBI.pdf`](../refs/2412.01548_ATLAS_offshell_H4l_NSBI.pdf).

> The physics twin of this project: the same process (off-shell gg → H* → ZZ → 4ℓ), the
> same signal-strength parametrisation with √μ interference scaling, analysed with NSBI —
> in the real-data regime where the method is trusted. Use it for the physics framing
> (why off-shell, why interference matters) and as the reference point for "what NSBI
> claims to deliver" against which the non-asymptotic breakdown is judged.

### [CORE] Ref. 5 — SBI overconfidence (Hermans et al., 2022)

**J. Hermans, A. Delaunoy, F. Rozet, A. Wehenkel, V. Begy, G. Louppe**, *A Trust Crisis
In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful*,
**Transactions on Machine Learning Research (2022)**.
arXiv: [2110.06581](https://arxiv.org/abs/2110.06581).
⚠ Two title variants exist: the arXiv PDF (which carries the "Published in TMLR (11/2022)"
header) uses the title above; the TMLR/OpenReview listing uses *"A Crisis In Simulation-Based
Inference? Beware, Your Posterior Approximations Can Be Unfaithful"*. Cite the TMLR version.
Local: [`refs/2110.06581_Hermans_SBI_trust_crisis.pdf`](../refs/2110.06581_Hermans_SBI_trust_crisis.pdf).

> The ML-community evidence for the same disease seen from the Bayesian side: across SBI
> algorithms (including neural ratio estimation, the family CARL belongs to), posterior
> approximations are systematically **overconfident** unless explicitly diagnosed and
> corrected. Directly relevant to §11 of `sbi_foundations.md` (why would NSBI be
> overconfident?) and the strongest external corroboration that Tei's observation is a
> real, generic phenomenon rather than a local training artefact.

---

## C. Statistics background & mitigation anchors

### [SUPPORTING] Ref. 6 — Wilks (1938)

**S. S. Wilks**, *The Large-Sample Distribution of the Likelihood Ratio for Testing
Composite Hypotheses*, **Ann. Math. Statist. 9 (1938) 60–62**.
DOI: [10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360).
Local: [`refs/Wilks1938_AnnMathStat9_60_LR_distribution.pdf`](../refs/Wilks1938_AnnMathStat9_60_LR_distribution.pdf).

> The χ² limiting distribution of −2 ln λ under the null — the theorem that underlies
> every "asymptotic" claim this project tests. Three pages; worth actually reading to see
> how strong the large-sample assumptions are.

### [SUPPORTING] Ref. 7 — Wald (1943)

**A. Wald**, *Tests of Statistical Hypotheses Concerning Several Parameters When the
Number of Observations is Large*, **Trans. Amer. Math. Soc. 54 (1943) 426–482**.
Local: [`refs/Wald1943_TAMS54_426_large_sample_tests.pdf`](../refs/Wald1943_TAMS54_426_large_sample_tests.pdf).

> Source of the quadratic (Wald) approximation of the test statistic,
> −2 ln λ ≈ (μ̂−μ)²/σ² — the step that lets an Asimov NLL curve be read off as expected
> sensitivity (Ref. 2 builds directly on it). When the quadratic approximation fails —
> visibly the case for the non-parabolic scan curves at 1 fb⁻¹ — everything downstream
> of it is suspect.

### [SUPPORTING] Ref. 8 — Bartlett (1937)

**M. S. Bartlett**, *Properties of sufficiency and statistical tests*, **Proc. R. Soc.
Lond. A 160 (1937) 268–282**.
DOI: [10.1098/rspa.1937.0109](https://doi.org/10.1098/rspa.1937.0109).
**No local PDF** — paywalled at the Royal Society; citation-only.

> Origin of the **Bartlett correction**: rescaling the LR statistic by its finite-sample
> mean so its distribution approaches χ² faster (O(1/n²) instead of O(1/n)). Literature
> anchor for mitigation #2 in `docs/context.md` (alternative/corrected test statistics
> for the non-asymptotic regime).

### [SUPPORTING] Ref. 9 — Cowan (2019), Bartlett in HEP

**G. Cowan**, *Statistical Models with Uncertain Error Parameters*, **Eur. Phys. J. C 79
(2019) 133**.
arXiv: [1809.05778](https://arxiv.org/abs/1809.05778).
DOI: [10.1140/epjc/s10052-019-6644-4](https://doi.org/10.1140/epjc/s10052-019-6644-4).
Local: [`refs/1809.05778_Cowan_uncertain_error_parameters.pdf`](../refs/1809.05778_Cowan_uncertain_error_parameters.pdf).

> A worked HEP example where the LR statistic measurably departs from its asymptotic χ²
> form and **Bartlett-corrected** statistics are constructed and deployed (in the body of
> the paper, not the abstract). The practical bridge from Ref. 8 to an implementation
> this project could copy for mitigation #2.

### [SUPPORTING] Ref. 14 — Barlow & Beeston (1993)

**R. Barlow, C. Beeston**, *Fitting using finite Monte Carlo samples*, **Comput. Phys.
Commun. 77 (1993) 219–228**.
DOI: [10.1016/0010-4655(93)90005-W](https://doi.org/10.1016/0010-4655(93)90005-W).
*(Verified against Inspire 2026-07-13; no local PDF yet.)*

> The standard treatment of MC statistical uncertainties in template fits: one nuisance
> parameter per bin per source for the unknown true MC expectation, profiled in the fit.
> The "lite" simplification (one effective parameter per bin, effective sample size
> n_eff = (Σw)²/Σw²) is what modern HEP fits deploy. Literature anchor for mitigation #3.

### [SUPPORTING] Ref. 15 — Neyman (1937)

**J. Neyman**, *Outline of a Theory of Statistical Estimation Based on the Classical
Theory of Probability*, **Phil. Trans. R. Soc. Lond. A 236 (1937) 333–380**.
DOI: [10.1098/rsta.1937.0005](https://doi.org/10.1098/rsta.1937.0005).
*(Verified against Crossref 2026-07-13; no local PDF yet.)*

> The origin of confidence intervals and of the **Neyman construction**: build, for every
> hypothesis value, the acceptance region containing the desired probability of the data;
> the confidence set is every hypothesis whose region contains the observation. Coverage
> is guaranteed **by construction**, with no asymptotic assumption — which is exactly why
> toy-based calibration (a Monte Carlo Neyman construction) is the fallback when
> Wilks/Wald fail, and the theory anchor for the foundations note's coverage section.

### [SUPPORTING] Ref. 18 — Feldman & Cousins (1998)

**G. J. Feldman and R. D. Cousins**, *A Unified Approach to the Classical Statistical
Analysis of Small Signals*, **Phys. Rev. D 57 (1998) 3873**.
arXiv: [physics/9711021](https://arxiv.org/abs/physics/9711021);
DOI: [10.1103/PhysRevD.57.3873](https://doi.org/10.1103/PhysRevD.57.3873).
*(Verified against arXiv/Crossref 2026-07-13; no local PDF yet.)*

> Supplies the **ordering rule** that the Neyman construction (Ref. 15) leaves free: rank
> outcomes by the likelihood ratio P(x|μ)/P(x|μ̂_phys), with the best-fit μ restricted to the
> physical region, and fill each acceptance region in decreasing order of that ratio. Cures
> two pathologies of small-signal limits: **flip-flopping** (choosing upper-limit vs central
> interval after seeing the data, which undercovers — 85% for a nominal 90%) and **empty
> intervals** under a downward fluctuation near a physical boundary. Caveats: over-covers for
> discrete data; does not stop a downward-fluctuating experiment from quoting a limit better
> than its sensitivity (the motivation for CLs); expensive with nuisance parameters.
> Directly relevant to conditions (A) and (D) of the foundations note — the μ ≥ 0 boundary is
> live at 1 fb⁻¹ — and the ordering behind mitigation #6 (toy-calibrated inference).
> Table IV (Poisson with known background) is the numerical cross-check for the belt figure
> in §6.3: n=0 gives [0, 2.44] at b=0, [0, 1.61] at b=1, [0, 1.08] at b=2.

---

## D. SBI / ML methods context

### [SUPPORTING] Ref. 10 — Frontier of SBI (2020)

**K. Cranmer, J. Brehmer, G. Louppe**, *The frontier of simulation-based inference*,
**PNAS 117 (2020) 30055–30062**.
arXiv: [1911.01429](https://arxiv.org/abs/1911.01429).
DOI: [10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117).
Local: [`refs/1911.01429_frontier_of_SBI.pdf`](../refs/1911.01429_frontier_of_SBI.pdf).

> The canonical SBI review; fixes the terminology (simulator, ratio estimation,
> diagnostics) and situates CARL/NSBI within the broader method landscape. Use its
> vocabulary in write-ups so external readers can place the project.

### [SUPPORTING] Ref. 11 — Mining gold (2020)

**J. Brehmer, G. Louppe, J. Pavez, K. Cranmer**, *Mining gold from implicit models to
improve likelihood-free inference*, **PNAS 117 (2020) 5242–5249**.
arXiv: [1805.12244](https://arxiv.org/abs/1805.12244).
DOI: [10.1073/pnas.1915980117](https://doi.org/10.1073/pnas.1915980117).
Local: [`refs/1805.12244_mining_gold.pdf`](../refs/1805.12244_mining_gold.pdf).

> The successor idea to CARL: exploit joint score/ratio information from the simulator
> (RASCAL/SALLY family) for markedly more sample-efficient ratio estimation. Relevant
> both to position CARL within modern NSBI and as a possible better-estimator direction
> if the failure mode turns out to be estimator-variance-driven.

### [SUPPORTING] Ref. 12 — Deep Sets (2017)

**M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Póczos, R. Salakhutdinov, A. J. Smola**,
*Deep Sets*, **Advances in Neural Information Processing Systems 30 (NIPS 2017)**.
arXiv: [1703.06114](https://arxiv.org/abs/1703.06114).
Local: [`refs/1703.06114_deep_sets.pdf`](../refs/1703.06114_deep_sets.pdf).

> The permutation-invariant architecture for set-valued inputs — the natural network for
> mitigation #1 (toy-ensemble training: feed the network whole pseudo-experiments so its
> confidence can scale with the available statistics).

### [SUPPORTING] Ref. 13 — Balanced NRE (2022)

**A. Delaunoy, J. Hermans, F. Rozet, A. Wehenkel, G. Louppe**, *Towards Reliable
Simulation-Based Inference with Balanced Neural Ratio Estimation*, **NeurIPS 2022**.
arXiv: [2208.13624](https://arxiv.org/abs/2208.13624).
Local: [`refs/2208.13624_BNRE_balanced_NRE.pdf`](../refs/2208.13624_BNRE_balanced_NRE.pdf).

> The closest existing fix in the ML literature: a modified (balanced) ratio-estimation
> objective that makes estimators **conservative rather than overconfident**, from the
> same group as Ref. 5. A candidate fifth mitigation for this project and the natural
> baseline to compare any fix against.

### [SUPPORTING] Ref. 16 — Deep Ensembles (2017)

**B. Lakshminarayanan, A. Pritzel, C. Blundell**, *Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles*, **Advances in Neural Information
Processing Systems 30 (NIPS 2017)**.
arXiv: [1612.01474](https://arxiv.org/abs/1612.01474).
*(Verified against arXiv 2026-07-13; no local PDF yet.)*

> The canonical reference for **ensembling as uncertainty estimation**: train M networks
> from independent initialisations, treat the spread of their outputs as (epistemic)
> model uncertainty. Anchors the foundations note's distinction between what ensembles
> estimate (training/model variability) and what toys estimate (data statistics) — the
> two are complementary, and neither substitutes for the other.

### [SUPPORTING] Ref. 17 — wᵢfᵢ ("wifi") ensembles (Benevedes & Thaler, 2025)

**S. Benevedes, J. Thaler**, *Frequentist Uncertainties on Neural Density Ratios with
wᵢfᵢ Ensembles*, MIT-CTP/5874.
arXiv: [2506.00113](https://arxiv.org/abs/2506.00113) (v1 2025-05-30; v2 2025-09-24, hep-ph).
Code: [github.com/benevedes/wifi-ensembles](https://github.com/benevedes/wifi-ensembles).
Local: [`refs/2506.00113_wifi_ensembles.pdf`](../refs/2506.00113_wifi_ensembles.pdf).
*(Verified against arXiv 2026-07-13; title page checked.)*

> **This is the April meeting's "Wi-Fi ensemble method"** — the auto-transcription was
> right and the name is a pun: *wᵢfᵢ* = weights **wᵢ** on basis functions **fᵢ**. Train M
> networks fᵢ(x) (bootstrap resamples or a partition of the training set), **freeze them**,
> model the log density ratio as a linear combination `log r̃(x|w) = Σᵢ wᵢ fᵢ(x)`, and fit
> only the M weights on a *held-out* sample D_fit with a convex (symmetrised MLC) loss.
> Because the loss is linear in w, the weight covariance follows in closed form — a
> sandwich estimator `C = V⁻¹ U V⁻¹` (V = Hessian, U = score covariance) — giving
> `σ²(x) = fᵢ(x) Cᵢⱼ fⱼ(x)` on the learned log-ratio, propagated to the POI by the
> Gong–Samaniego pseudo-MLE result `σ²_GS = σ²_MLE (1 + σ²_MLE Aᵢ Cᵢⱼ Aⱼ)`. Hence
> "trivial to apply": **no bootstrap loop over the whole analysis** — the uncertainty is an
> analytic inflation of the usual MLE variance.
>
> **What it does and does not buy us.** It captures the statistical noise of the **finite
> fitting sample** (C ~ O(1/N_fit)); the paper is explicit that it does *not* capture
> mismodelling from limited network expressivity, training stochasticity, finite *training*
> data, or simulator systematics. Two facts make it directly relevant here: (i) it inflates
> the POI variance, i.e. it pushes against exactly the overconfidence this project measures;
> (ii) the paper shows the **naive uniform-weight ensemble (Ref. 16 style) undercovers and
> is biased**, while the fitted-weight version reaches nominal coverage (M = 32, quark/gluon)
> — a published statement that plain deep ensembling is *not* a calibration fix.
>
> ⚠ **The caveat that matters for this project:** every guarantee above is **asymptotic**
> (fixed M, N_fit → ∞; Gong–Samaniego is a delta-method inflation of σ²_MLE). Our failure
> mode *is* the breakdown of asymptotics at O(100) events. So wifi's coverage guarantee does
> not automatically transfer to the 1–10 fb⁻¹ regime — it is a **candidate mitigation to be
> tested, not a fix to be assumed**, and testing it is itself an interesting experiment (a
> method whose uncertainties are asymptotic, applied where asymptotics fail).

---

## E. Not yet verified — do not cite before checking

*(empty — the former entry, Barlow–Beeston, was verified 2026-07-13 and promoted to Ref. 14.)*

---

## F. Cross-references: refs ↔ project

| Reference | Anchors |
|-----------|---------|
| Ref. 0 (Tei's slides) | The founding evidence; reproduction target (Phase 1, `docs/plan.md`) |
| Ref. 1 (CARL) | `asimov-aymptotics-check/models/carl.py`, the two density-ratio NNs |
| Ref. 2 (Cowan et al.) | The Asimov property under test; `stats/binned.py` NLLs |
| Ref. 5 (Hermans et al.) | Key-finding root-cause discussion (`docs/context.md`) |
| Ref. 8 + 9 (Bartlett) | Mitigation #2 — alternative test statistics (Phase 3) |
| Ref. 12 (Deep Sets) | Mitigation #1 — toy-ensemble training (Phase 3) |
| Ref. 13 (BNRE) | Candidate mitigation #5 — conservative training objective |
| Ref. 14 (Barlow–Beeston) | Mitigation #3 — MC stat uncertainties (Phase 3) |
| Ref. 15 (Neyman) | Coverage & the Neyman construction (foundations note, coverage section) |
| Ref. 16 (Deep Ensembles) | Ensembling as model-uncertainty estimation (foundations note) |
| Ref. 17 (wifi ensembles) | Mitigation #3, network side — the April meeting's method, identified 2026-07-13 (Phase 3, prong 2 / plan item 8d) |
| Ref. 18 (Feldman–Cousins) | The ordering rule for the Neyman construction; mitigation #6 (toy-calibrated inference), and the μ ≥ 0 boundary behaviour that condition (D) needs (foundations note §6.3) |

---

## G. Citation standards (for the future note/paper)

```bibtex
@article{Cranmer:2015bka,
    author        = {Cranmer, Kyle and Pavez, Juan and Louppe, Gilles},
    title         = {{Approximating Likelihood Ratios with Calibrated Discriminative Classifiers}},
    year          = {2015},
    eprint        = {1506.02169},
    archivePrefix = {arXiv},
    primaryClass  = {stat.AP},
    note          = {arXiv only}
}

@article{Cowan:2010js,
    author        = {Cowan, Glen and Cranmer, Kyle and Gross, Eilam and Vitells, Ofer},
    title         = {{Asymptotic formulae for likelihood-based tests of new physics}},
    journal       = {Eur. Phys. J. C},
    volume        = {71},
    pages         = {1554},
    year          = {2011},
    doi           = {10.1140/epjc/s10052-011-1554-0},
    note          = {Erratum: Eur. Phys. J. C 73 (2013) 2501},
    eprint        = {1007.1727},
    archivePrefix = {arXiv},
    primaryClass  = {physics.data-an}
}

@article{ATLAS:2024ynn,
    author        = {{ATLAS Collaboration}},
    title         = {{An implementation of neural simulation-based inference for parameter estimation in ATLAS}},
    journal       = {Rep. Prog. Phys.},
    volume        = {88},
    pages         = {067801},
    year          = {2025},
    doi           = {10.1088/1361-6633/add370},
    eprint        = {2412.01600},
    archivePrefix = {arXiv},
    primaryClass  = {physics.data-an}
}

@article{ATLAS:2024yzr,
    author        = {{ATLAS Collaboration}},
    title         = {{Measurement of off-shell Higgs boson production in the $H^{*}\to ZZ\to 4\ell$ decay channel using a neural simulation-based inference technique in 13 TeV $pp$ collisions with the ATLAS detector}},
    journal       = {Rep. Prog. Phys.},
    volume        = {88},
    pages         = {057803},
    year          = {2025},
    doi           = {10.1088/1361-6633/adcd9a},
    eprint        = {2412.01548},
    archivePrefix = {arXiv},
    primaryClass  = {hep-ex}
}

@article{Hermans:2021trust,
    author        = {Hermans, Joeri and Delaunoy, Arnaud and Rozet, Fran{\c c}ois and Wehenkel, Antoine and Begy, Volodimir and Louppe, Gilles},
    title         = {{A Crisis In Simulation-Based Inference? Beware, Your Posterior Approximations Can Be Unfaithful}},
    journal       = {Transactions on Machine Learning Research},
    year          = {2022},
    eprint        = {2110.06581},
    archivePrefix = {arXiv},
    primaryClass  = {stat.ML},
    note          = {arXiv version titled ``A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful''}
}
```

⚠ The BibTeX keys above follow Inspire style but were **not** pulled from Inspire —
regenerate from Inspire before submitting anything official.

---

## H. Recommended reading order (for onboarding)

1. **Ref. 0** (Tei's slides) — the phenomenon, 15 minutes.
2. **Ref. 2 §§2–3** (Cowan et al.) — what the Asimov dataset is and what the median
   property actually says. ~2 hours.
3. **Ref. 1** (CARL) — the density-ratio trick and calibration. ~1.5 hours.
4. **Ref. 4** (ATLAS off-shell NSBI) — the same physics done for real; skim the
   statistical-procedure sections. ~1 hour.
5. **Ref. 5** (Hermans et al.) — the overconfidence evidence; read the conclusions and
   Fig. 1 at minimum. ~1 hour.
6. **Ref. 13** (BNRE) — what a fix can look like. Skim. ~30 minutes.
7. Refs. 6–9 as needed when working on mitigation #2 (test statistics).
