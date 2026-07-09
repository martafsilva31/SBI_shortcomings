# NSBI and the Asimov Property — Foundations

<!-- Figures: none embedded yet. The evidence figures live in docs/presentations/asimov_vs_medtoys_test_statistic_studies.pdf (Tei's slides); once the reproduction produces our own curves, publish them to the public figures repo and embed via GitHub-raw URLs, as in QT/Isolation's foundations note. -->

*A self-contained theory primer. It assumes no prior knowledge of simulation-based inference or of frequentist HEP statistics, and builds, section by section, everything needed to understand why the Asimov dataset is trusted, why that trust rests on asymptotics, and why a neural per-event likelihood can forfeit it while a histogram does not. Concepts and founding evidence only — operational details (environment, code paths, run instructions) live in `docs/env-setup-and-code-walkthrough.md`; the annotated bibliography lives in [`literature.md`](literature.md).*

---

## Table of contents

| § | Section | One-line takeaway |
|---|---------|-------------------|
| 1 | [The question this project asks](#1-the-question-this-project-asks) | Is one perfectly average dataset an honest stand-in for the median of many noisy ones? For NSBI at low luminosity, apparently not. |
| 2 | [Likelihood-ratio tests and their asymptotic guarantees](#2-likelihood-ratio-tests-and-their-asymptotic-guarantees) | Wilks and Wald turn −2Δln L into a χ²-distributed ruler — but only in the large-sample limit. |
| 3 | [The Asimov dataset and what is actually proven](#3-the-asimov-dataset-and-what-is-actually-proven) | Cowan et al. prove a *median* property of the *test statistic*, *asymptotically*; the scan-curve equality tested here is a stronger, operational statement. |
| 4 | [Why histograms are asymptotic and NSBI is not](#4-why-histograms-are-asymptotic-and-nsbi-is-not) | A histogram lets you widen bins until counts are large; a per-event likelihood hands every event its own bin. |
| 5 | [The physics stage: off-shell gg → H* → ZZ → 4ℓ](#5-the-physics-stage-off-shell-gg--h--zz--4ℓ) | Signal and box background interfere, so the density is a μ-, √μ-weighted three-component mixture with ~5.7 expected events at 1 fb⁻¹. |
| 6 | [Learning the likelihood ratio: CARL and NSBI](#6-learning-the-likelihood-ratio-carl-and-nsbi) | A classifier's output encodes a density ratio; two CARL networks give the full μ-dependent mixture ratio. |
| 7 | [The two likelihoods, side by side](#7-the-two-likelihoods-side-by-side) | Both are extended likelihoods; the Poisson rate term carries the luminosity dependence, the shape term the per-event information. |
| 8 | [Toy pseudo-experiments and the test protocol](#8-toy-pseudo-experiments-and-the-test-protocol) | Per-event Poisson weight resampling; "median of toys" means the pointwise-per-μ median of the toy scan curves — a definition the study must own. |
| 9 | [The evidence](#9-the-evidence) | Histogram: Asimov ≈ median(toys) at every luminosity. NSBI: a clear departure at 1–100 fb⁻¹, gone at 1000 fb⁻¹. |
| 10 | [Known caveats on the evidence](#10-known-caveats-on-the-evidence) | The NSBI code omits the extended Poisson term, and the two project docs disagree on the sign of the effect — both flagged, neither resolved. |
| 11 | [Why would NSBI be overconfident? Hypotheses](#11-why-would-nsbi-be-overconfident-hypotheses) | NN mis-specification, no exactly unbiased NN estimator, training blind to sample-size fluctuations — and the ML literature says SBI is generically overconfident. |
| 12 | [Proposed mitigations](#12-proposed-mitigations) | Toy-ensemble training, Bartlett-type corrections, MC-stat uncertainties, a dimensionality study — each with a literature anchor. |
| 13 | [Open questions](#13-open-questions) | Sign and size after the Poisson-term fix; a scalar breakdown metric; whether the pointwise median is even the right comparator. |
| 14 | [Glossary](#14-glossary) | Every term in one line. |
| 15 | [References](#15-references) | The canonical papers behind each section. |

---

## 1. The question this project asks

Before an experiment runs, physicists want to know what sensitivity to *expect*. The gold-standard answer is brute force: simulate many pseudo-experiments ("toys"), analyse each exactly as real data would be analysed, and report the median of the resulting measures — median significance, median limit, median likelihood-scan width. The convenient answer is to analyse a single artificial dataset in which every observable is set exactly to its expectation value — the **Asimov dataset** — and read the same measures off that one fit. The entire convenience rests on a claimed equivalence: *the Asimov result equals the median of the toys*.

This project asks whether that equivalence survives when the analysis is a **neural simulation-based inference (NSBI)** analysis in a low-statistics regime. The operational form of the claim tested here — the project's working definition of the **Asimov property** — is: *the −2Δln L scan curve computed on the Asimov dataset equals, pointwise in μ, the median of the scan curves of an ensemble of toy pseudo-experiments*. Tei's slide 5 states the expectation in exactly these terms: "the NLL of nominal Asimov dataset corresponds to the median of throwing these pseudo-experiments."

The founding observation (first team meeting, 2026-05-20; `docs/context.md` § Key finding) is that a histogram analysis of off-shell Higgs production satisfies this property at every luminosity tried, while the NSBI analysis of the *same events* visibly violates it at 1–100 fb⁻¹, converging only at ~1000 fb⁻¹. Everything below builds the concepts needed to understand why that could happen — and why it is plausible that it *must* happen.

---

## 2. Likelihood-ratio tests and their asymptotic guarantees

The statistical machinery is frequentist and likelihood-based throughout. For a dataset of $n$ observed events with observables $x_1,\dots,x_n$ and a parameter of interest $\mu$ (here the signal-strength modifier, $\mu=1$ being the Standard Model), the **extended likelihood** is

$$\mathcal{L}(\mu) \;=\; \mathcal{P}\big(n \,\big|\, \nu(\mu)\big)\;\prod_{i=1}^{n} p(x_i|\mu),$$

where $\nu(\mu)$ is the total expected event count and $p(x|\mu)$ the per-event probability density. The first factor is the **rate (Poisson) term**; the product is the **shape term**. Inference uses the likelihood-ratio statistic

$$t_\mu \;=\; -2\ln\lambda(\mu), \qquad \lambda(\mu) = \frac{\mathcal{L}(\mu)}{\mathcal{L}(\hat\mu)},$$

with $\hat\mu$ the maximum-likelihood estimate. (This project currently has no nuisance parameters, so no profiling is involved; the general story is identical with $\lambda$ replaced by the profile likelihood ratio.)

Two classical theorems make $t_\mu$ useful:

- **Wilks (1938)**: under the hypothesis $\mu$, $t_\mu$ converges in distribution to a $\chi^2$ with one degree of freedom (one POI) as the sample size grows. This is what lets "$-2\Delta\ln L = 1$" be read as a 68% interval.
- **Wald (1943)**: in the same limit, $t_\mu \approx (\mu - \hat\mu)^2/\sigma^2$ — the scan curve is a parabola whose curvature *is* the measurement uncertainty. This quadratic form is what makes a single scan curve summarise an experiment's sensitivity.

> ⚠️ Both are **large-sample** statements. Their regularity conditions (estimator asymptotically Gaussian, likelihood locally quadratic, expected counts large where it matters) are exactly what "asymptotic regime" means in this note. Nothing in either theorem survives by fiat at 5.7 expected events — whether the conclusions survive *approximately* is a quantitative question, and it is the question of this project.

A visible symptom in the founding evidence: the NSBI scan curves at 1 fb⁻¹ (slide 7, left panel) are manifestly non-parabolic, so the Wald reading of curve width as sensitivity is already on thin ice there.

---

## 3. The Asimov dataset and what is actually proven

Cowan, Cranmer, Gross and Vitells (2011) define the **Asimov dataset** as the artificial dataset in which every observable takes exactly its expected value — every histogram bin holds its expected yield, and the total count equals $\nu(\mu')$ for the assumed true $\mu'$. Its defining property: evaluating the estimator on it returns the true parameters exactly, $\hat\mu_A = \mu'$.

What the paper *proves* (its Sec. 3.2, within the Wald approximation) is: the test statistic evaluated on the Asimov dataset, $t_{\mu,A}$, equals the **median** of the $t_\mu$ distribution under $\mu'$, **asymptotically**; equivalently, the "Asimov significance" $\sqrt{t_{\mu,A}}$ is the median expected significance. This is the theorem that made "fit the Asimov, skip the toys" standard practice in HEP.

> 🎯 **What is proven vs. what is tested here.** The proven statement is a median property of the *scalar test statistic*, in the *asymptotic limit*. The property tested in this project — pointwise equality of the whole Asimov NLL *curve* with the median of toy *curves*, at ~6 expected events — is a stronger, operational statement, and it is the project's own definition (attribute it to this study, not to Cowan et al.). Its violation therefore does not contradict any theorem; it demonstrates that a regime assumed asymptotic is not. The histogram analysis satisfying the operational property at all luminosities is what makes the comparison meaningful: same events, same toys, same comparator — only the analysis method changes.

---

## 4. Why histograms are asymptotic and NSBI is not

The heart of the project's hypothesis (`docs/context.md` § Background) is a bin-counting argument.

A **histogram analysis** compresses each event to one observable (here $m_{4\ell}$) and bins it. The analyst controls the bin width: if some bin has too few expected events for Poisson-Gaussian asymptotics, merge bins until it doesn't. Coarse binning discards information but *buys asymptotic validity*. That knob is why the histogram column of the founding evidence behaves at every luminosity.

An **NSBI analysis** evaluates a per-event likelihood ratio: every event contributes its own factor $p_{\rm SBI}(x_i|\mu)/p_{\rm ref}(x_i)$ at its exact 16-dimensional phase-space point. This is the limit of an infinitely finely binned histogram — in 16 dimensions. The effective "bin" occupied by each event contains, in expectation, far less than one event; there is no bin-widening knob, because giving up resolution is precisely what NSBI exists to avoid.

> 💡 **Analogy.** A histogram lets you choose the bucket size; NSBI hands every event its own bucket. You can always make buckets big enough that each holds a crowd — but a bucket sized for one occupant is never crowded, no matter how many buckets there are.

The consequence, and the project's framing hypothesis: *for high-dimensional per-event analyses, the non-asymptotic regime extends to far larger total event counts than one-dimensional intuition suggests* — here, an analysis with O(100) events (10 fb⁻¹) still misbehaves. Mitigation #4 (§12) is designed to test that this is really about dimensionality/effective-bin occupancy rather than about neural networks per se: a 4-dimensional *histogram* at 1 fb⁻¹ should start failing the same way.

---

## 5. The physics stage: off-shell gg → H\* → ZZ → 4ℓ

The test case is off-shell Higgs production at parton level: $gg \to H^* \to ZZ \to 4\ell$ (slide 2 shows the diagrams — the Higgs-mediated triangle and the continuum $gg\to ZZ$ box). Above the $2m_Z$ threshold the Higgs is far off shell, the signal amplitude is small, and it **interferes destructively** with the box background. Because the signal-strength modifier scales the signal *amplitude* by $\sqrt\mu$, the cross terms scale as $\sqrt\mu$: the probability density of an event under the signal+background+interference (SBI) hypothesis is

$$p_{\rm SBI}(x|\mu) \;=\; \frac{\mu\,\nu_S\,p_S(x) + \sqrt{\mu}\,\nu_I\,p_I(x) + \nu_B\,p_B(x)}{\mu\,\nu_S + \sqrt{\mu}\,\nu_I + \nu_B},$$

where $\nu_S, \nu_I, \nu_B$ are the component yields and $p_S, p_I, p_B$ the component densities. Since the interference "density" $p_I$ is signed (its yield $\nu_I < 0$), the code and slides use an equivalent re-arrangement in terms of the physical SM mixture $p_{\rm SBI} \equiv p_{\rm SBI}(x|\mu{=}1)$ that avoids negative weights:

$$p_{\rm SBI}(x|\mu) \;=\; \frac{(\mu-\sqrt{\mu})\,\nu_S\,p_S(x) + \sqrt{\mu}\,\nu_{\rm SBI}\,p_{\rm SBI}(x) + (1-\sqrt{\mu})\,\nu_B\,p_B(x)}{(\mu-\sqrt{\mu})\,\nu_S + \sqrt{\mu}\,\nu_{\rm SBI} + (1-\sqrt{\mu})\,\nu_B}.$$

Expected yields at $\mu=1$ (`docs/context.md`; slide 4 shows the same numbers with typo'd column headers, so cite context.md):

| Component | 1 fb⁻¹ | 10 fb⁻¹ | 100 fb⁻¹ | 1000 fb⁻¹ |
|-----------|--------|---------|----------|-----------|
| SIG       | 0.310  | 3.099   | 30.990   | 309.897   |
| BKG       | 5.953  | 59.530  | 595.303  | 5953.034  |
| INT       | −0.606 | −6.064  | −60.643  | −606.433  |
| SBI total | 5.656  | 56.565  | 565.650  | 5656.498  |

So the headline regime, 1 fb⁻¹, is an experiment expecting **~5.7 events**, of which the signal contribution is a *net negative* −0.3 (destructive interference beats the tiny signal). Events are MCFM parton-level 4-lepton final states, per-event reweightable to any $\mu$ (and to 45 EFT points, for later phases) via stored matrix elements; the NSBI observables are the 16 components of the four lepton four-vectors, the histogram observable is $m_{4\ell}$. This is the same process and parametrisation that the real ATLAS off-shell NSBI measurement uses (literature Ref. 4) — which is what makes a failure mode here worth understanding precisely.

---

## 6. Learning the likelihood ratio: CARL and NSBI

The densities $p_S$, $p_B$, $p_{\rm SBI}$ in 16 dimensions are not known in closed form — only samples of them exist. Simulation-based inference solves this with the **likelihood-ratio trick** (CARL; literature Ref. 1): train a classifier $s(x)$ to distinguish samples of $p_{\rm num}$ from samples of $p_{\rm ref}$; the optimally trained classifier satisfies

$$\frac{s(x)}{1-s(x)} \;=\; \frac{p_{\rm num}(x)}{p_{\rm ref}(x)},$$

so a *discriminative* network yields a *density ratio*. CARL's second ingredient is **calibration** — diagnosing and correcting the residual distortion of an imperfectly trained classifier's ratio.

This analysis (slide 3, NSBI column) chooses the reference $p_{\rm ref} = p_B$ and trains **two** CARL networks: one for $p_S/p_B$, one for $p_{\rm SBI}/p_B$. Plugging both into the mixture formula of §5 gives the full μ-dependent ratio $p_{\rm SBI}(x|\mu)/p_B(x)$ for *any* μ — the reference density cancels in the ratio of ratios, so its arbitrary choice is harmless. Networks are trained once, on balanced (shape-only) samples with no luminosity information; luminosity enters only at inference time. The larger ATLAS NSBI framework (literature Ref. 3) is the production-grade version of this construction, with ensembling and calibration diagnostics this toy setup does not have.

---

## 7. The two likelihoods, side by side

Both analyses are extended-likelihood fits of the same μ (slide 3):

**Histogram:** $\;\mathcal{L}(\mu) = \prod_{\rm bins} \mathcal{P}\big(n_b \,\big|\, \nu_b(\mu)\big)$ — ten log-spaced $m_{4\ell}$ bins; the product of per-bin Poissons automatically contains the total-rate information (the sum of bin expectations is the total expectation).

**NSBI (as intended):** $\;\mathcal{L}(\mu) = \mathcal{P}\big(n \,\big|\, \nu(\mu)\big) \prod_{\rm events} \dfrac{p_{\rm SBI}(x_i|\mu)}{p_{\rm ref}(x_i)}$ — the per-event shape product, times one global Poisson rate term with $\nu(\mu) = \mathrm{lumi}\cdot(\mu\,\sigma_S + \sqrt{\mu}\,\sigma_I + \sigma_B)$. Dividing the shape product by $p_{\rm ref}$ is again harmless — it is μ-independent and drops out of $-2\Delta\ln L$. Slide 3's own emphasis: "the 'total rate' Poisson term, is not estimated using NNs" — it is analytic; the networks only ever supply shape information.

> 🎯 **The rate term carries all the luminosity dependence.** The shape term gains information per event; the rate term gains information as $\nu(\mu)$ grows with luminosity, and it is exact (no NN approximation touches it). This is the natural reading of the founding evidence's convergence pattern — NSBI's Asimov-vs-median agreement is restored at 1000 fb⁻¹ "when the Poisson term takes over" (slide 7): as luminosity grows, an exact, well-behaved term increasingly dominates an approximate, badly behaved one. It is also why the missing-rate-term bug (§10) is more than a detail: the curves in the founding evidence are *pure shape*, i.e. pure NN.

---

## 8. Toy pseudo-experiments and the test protocol

The dataset is weighted (each simulated event carries a weight $w_i$ such that weights sum to the expected yields). Toys are thrown per event weight (slide 5): $w_{\rm toy} \sim \mathcal{P}(\lambda = w)$ — each event's weight is replaced by a Poisson draw with that mean, which fluctuates both the total count and the shape simultaneously. The code implementation draws $\mathcal{P}(\lambda = w_i \cdot \mathrm{lumi})$ per event, 200 toys, fixed seed; the founding evidence quotes 100–200 toys per luminosity point.

For each toy, the full analysis (histogram or NSBI) produces a $-2\Delta\ln L$ scan over $\mu \in [0,4]$; the comparator is then the **pointwise-per-μ median** of the toy curves, with a ±1σ quantile band, overlaid on the Asimov curve.

> ⚠️ **The definition must be owned precisely.** Slide 5 itself warns of a normalisation subtlety: "Here, the y-axis is not consistent between toys — some toy curves should go negative w.r.t. nominal Asimov curve." Whether each toy curve is minimum-subtracted (its own $\hat\mu$ as reference) or referenced to the nominal Asimov likelihood changes what "the median curve" means, materially, in exactly the low-statistics regime under study. Related: §3's theorem concerns the median of a *scalar statistic*; a pointwise median of *curves* is not the median of any single experiment's curve. The reproduction (Phase 1) must fix and document one convention before quantitative statements are made.

---

## 9. The evidence

The founding evidence is two figure grids in Tei's slides (Asimov = orange dashed, median of toys = solid blue, ±1σ band shaded; panels at 1, 10, 100, 1000 fb⁻¹):

**Slide 6 — histogram.** Median(toys) tracks the Asimov curve closely in all four panels. Tei's bullet: "For histogram analysis, the median of toys seems to indeed correspond to that of the single representative Asimov dataset results." ⚠️ The slide *title* reads "Asimov vs. Median(Toys): NSBI", but the bullet and content are the histogram analysis — the title is mislabeled; do not be misled when citing the PDF.

**Slide 7 — NSBI.** At 1 fb⁻¹ the two curves visibly part ways: the blue median-of-toys curve rises to ≈ 0.47 at $\mu = 4$ while the orange Asimov curve reaches only ≈ 0.16 — the *median of toys* is the steeper of the two (see §10 for why this sign matters). The departure shrinks at 10 fb⁻¹, is small at 100 fb⁻¹, and is gone at 1000 fb⁻¹. Tei's bullets, verbatim: "For NSBI analysis, the convergence only happens when the Poisson term takes over."; "Why does the median not correspond to the Asimov?"; "(Personal thought) the fact that as results → Poisson term the convergence is better seems to suggest that NN mis-specification could be a cause."; "But of course, it is impossible to have a completely 'un-biased' estimator with NNs (even with ensembling), in contrast to e.g. histogramming."

Same events, same toy recipe, same comparator — the only change between slide 6 and slide 7 is the analysis method. That controlled contrast is the strength of the evidence; §10 lists the caveats on it.

---

## 10. Known caveats on the evidence

**(i) The NSBI code omits the extended Poisson term.** Tei flagged this on 2026-07-09 (`docs/context.md` § Known code issues) and the code walkthrough confirmed it with file-level evidence (`docs/env-setup-and-code-walkthrough.md` § Verified): the NSBI notebook computes a weighted pure-shape NLL, $-2\sum_i \nu_i \ln r(x_i|\mu)$, with no $\nu(\mu) - n\ln\nu(\mu)$ rate term anywhere in the NSBI path — consistently, the saved scan has $\mathrm{NLL}(\mu{=}0) = 0$ exactly. The histogram path is correct (full per-bin Poisson log-pmf). Subtlety worth preserving: a $-n\ln\nu(\mu)$-like piece *partially* enters through the mixture-ratio denominator, but the linear $+\nu(\mu)$ piece is entirely absent. Consequence: **all slide-6/7 curves are pre-fix, pure-shape curves**, and slide 7's convergence at 1000 fb⁻¹ happened *without* the explicit rate term (the rate information entering only via the per-event weights). The agreed procedure is to reproduce the curves as-is first, then add the term and re-run — both the size *and the sign* of the discrepancy may change.

**(ii) The direction of the effect is stated inconsistently across the project's own docs — UNRESOLVED.** Verbatim, both sides:

> `docs/context.md` § Key finding: "At 1–10 fb⁻¹, the NSBI median(toys) diverges significantly from the Asimov prediction: NLL curves are systematically steeper (NSBI is overconfident)."

> `docs/env-setup-and-code-walkthrough.md` § Direction to verify: "on slide 7 at 1 fb⁻¹ the **median-of-toys NLL is steeper (narrower) than the Asimov** (blue above orange) — i.e. the single Asimov scan is the *wider/less-constraining* one. context.md currently says 'NSBI NLL scan is systematically narrower than the median … (overconfident)', which reads as the opposite sign."

The context.md sentence leaves the subject of "steeper" ambiguous (and note the env-setup doc's inner quotation of context.md is itself not verbatim — no such sentence appears in the current context.md). Direct inspection of slide 7 supports the env-setup reading of the *figure*: the median of toys is the steeper curve, i.e. the Asimov scan *understates* the toys' median constraint. Which of the two curves deserves the label "overconfident" is then an interpretation question layered on top ("the Asimov underestimates" vs "the toys overestimate"), and it interacts with the median-definition caveat of §8 and the missing rate term of (i). **Resolution path (agreed): reproduce the curves, fix the median convention, add the Poisson term, re-check — before any statement about the direction propagates further** (this note deliberately quotes both sides rather than adjudicating).

**(iii) Do not source evidence from the repo's committed scan files.** The `.pt` tensors committed at the sub-repo root are NSBI, μ-mode, at **300 fb⁻¹** — a different luminosity grid from the slides. They are reproduction targets, not the founding evidence.

---

## 11. Why would NSBI be overconfident? Hypotheses

Four candidate mechanisms, not mutually exclusive:

1. **NN mis-specification** (Tei's own reading, slide 7): the learned ratios $\hat r(x)$ deviate from the true $r(x)$; a wrong likelihood is being scanned, and there is no theorem about its curvature. The convergence exactly when the (exact, analytic) Poisson term dominates points the finger at the NN-supplied piece.
2. **No exactly unbiased NN estimator** (also slide 7): a histogram's bin counts are unbiased estimators of bin expectations by construction; a trained network is not an unbiased estimator of anything, ensembling or not. Some bias term in the learned log-ratio is unavoidable and its effect on the scan curvature is uncontrolled.
3. **Training is blind to sample-size fluctuations** (`docs/context.md` § Key finding): the networks are trained on a single, large, Asimov-like balanced sample. The learned ratio is a statement about *densities*, carrying no representation of what a 6-event dataset does to shape estimates. The root-cause paragraph in context.md frames this as "a systematic bias intrinsic to the CARL/NSBI procedure in the non-asymptotic regime — not a training artefact that can be fixed by better regularisation alone."
4. **SBI is generically overconfident** (literature Refs. 5, 13): Hermans et al. showed across SBI algorithm families — including neural ratio estimation, CARL's family — that the resulting approximate posteriors are systematically *less conservative than the truth* unless explicitly diagnosed and corrected; BNRE (Delaunoy et al.) modifies the training objective to buy conservativeness. That is the Bayesian-coverage view of what the Asimov-vs-median check probes frequentistly. The convergence of the two literatures on the same word — *overconfident* — is the strongest external hint that Tei's observation is a real, structural phenomenon.

> 💡 Note the tension worth keeping in mind: hypotheses 1–2 are about the NN being *wrong*; hypothesis 3 would apply, at some level, even to a *perfectly* learned ratio, because the issue is what a finite dataset does to the statistic's distribution, not only the quality of the density estimate. Disentangling these is precisely what the mitigation studies are for.

---

## 12. Proposed mitigations

From `docs/context.md` § Proposed fixes, each with its literature anchor (numbers = `literature.md` refs):

1. **Toy-ensemble training** — train on many pseudo-experiments rather than one Asimov-like sample, with a permutation-invariant set architecture (**Deep Sets**, Ref. 12), so the network can learn that confidence must scale with available statistics. Attacks hypothesis 3 head-on.
2. **Alternative / corrected test statistics** — Bartlett-type corrections rescale the LR statistic by its finite-sample mean, accelerating the approach to χ² (Bartlett 1937, Ref. 8; a worked HEP deployment in Cowan 2019, Ref. 9). Attacks the symptom at the test-statistic level, agnostic to its cause.
3. **MC statistical uncertainties** — incorporate the finite size of the simulation samples into both likelihoods (Barlow–Beeston-type treatment; anchor **not yet verified**, see literature.md §E), making both analyses more conservative.
4. **Dimensionality study** — run a 4-dimensional *histogram* analysis at 1 fb⁻¹. If it starts failing the Asimov property like NSBI does, the effect is about effective-bin occupancy (§4), not about machine learning per se. This is the cleanest discriminator between "NN problem" and "dimensionality problem".
5. *(candidate, not yet in the plan)* **Conservative training objectives** — BNRE (Ref. 13) exists precisely to make ratio estimators err conservative; it is the natural external baseline for mitigation 1 and a possible fix in its own right.

---

## 13. Open questions

Conceptual questions this note leaves open (operational ones live in `docs/env-setup-and-code-walkthrough.md` § Open questions):

- **Sign and size after the fix.** Does adding the extended Poisson term shrink the 1–10 fb⁻¹ discrepancy, flip its apparent direction, or leave it untouched? (§10 i–ii.)
- **A breakdown metric.** What scalar quantifies "Asimov-property violation" as a function of luminosity and analysis dimensionality — a curve-distance, a coverage error, an interval-width ratio? Needed for the planned quantitative output (`docs/context.md` § Outputs).
- **Is the pointwise median of curves the right comparator** when the statistic is far from χ²-distributed and toy curves need not be parabolic (§8)? Median-of-curves, curve-of-medians, and median-of-scalar-summaries can all differ at 6 events.
- **Where does the histogram break?** At what effective dimensionality/bin occupancy does the histogram analysis itself start violating the property (mitigation 4)? That locates the boundary of the asymptotic regime empirically.
- **Frequentist ↔ Bayesian bridge.** Can the Asimov-vs-median check be related quantitatively to the SBI-calibration diagnostics of Refs. 5/13 (expected coverage), so the two literatures' evidence can be pooled?

---

## 14. Glossary

| Term | One line |
|------|----------|
| Asimov dataset | Artificial dataset in which every observable equals its expectation; fitting it returns the true parameters exactly. |
| Asimov property (operational, this project) | Asimov −2Δln L scan curve == pointwise-per-μ median of toy pseudo-experiment scan curves. |
| Asymptotic regime | Sample sizes large enough that Wilks/Wald hold: LR statistic ~ χ², scan curves parabolic. |
| Extended likelihood | Poisson probability of the total count times the product of per-event densities. |
| Rate (Poisson) term | The $\mathcal{P}(n|\nu(\mu))$ factor; analytic in this analysis, carries the luminosity dependence. |
| Shape term | The per-event product; the only part the neural networks touch. |
| NLL scan | $-2\Delta\ln L$ as a function of the POI over a grid; its width encodes sensitivity in the asymptotic regime. |
| POI, μ | Parameter of interest: signal-strength modifier scaling the signal amplitude ($\sqrt\mu$) and rate (μ); μ=1 is SM. |
| SIG / BKG / INT / SBI | Signal, background, (destructive) interference components, and their physical sum. |
| Toy pseudo-experiment | Synthetic dataset with per-event weights resampled as $w_{\rm toy}\sim\mathcal{P}(\lambda=w)$; fluctuates rate and shape together. |
| Median of toys | Pointwise-per-μ median of the toy scan curves (convention to be pinned down — §8). |
| SBI / NSBI | (Neural) simulation-based inference: inference when the likelihood is only accessible through simulation samples. |
| CARL | Likelihood-ratio estimation via calibrated classifiers: $s/(1-s) = p_{\rm num}/p_{\rm ref}$. |
| Density ratio, $p_{\rm ref}$ | The quantity CARL learns; the fixed reference density (here $p_B$) that cancels in ratios. |
| Likelihood-ratio trick | The identity turning an optimal classifier's output into a density ratio. |
| Overconfidence | Reported uncertainty smaller than true dispersion; in SBI literature, posteriors narrower than faithful ones. |
| Coverage | Fraction of repeated experiments whose reported interval contains the truth; the Bayesian-diagnostic mirror of the Asimov check. |
| Bartlett correction | Rescaling of the LR statistic by its finite-sample mean, improving the χ² approximation from O(1/n) to O(1/n²). |
| Deep Sets | Permutation-invariant NN architecture for set-valued inputs (a dataset of events is a set). |
| BNRE | Balanced neural ratio estimation: a training objective that makes learned ratios conservative. |

---

## 15. References

Full annotated bibliography with local PDFs: [`literature.md`](literature.md). Section anchors:

- §2 — Wilks 1938 (Ref. 6), Wald 1943 (Ref. 7).
- §3 — Cowan, Cranmer, Gross, Vitells 2011 + erratum (Ref. 2).
- §5 — ATLAS off-shell H\*→ZZ→4ℓ NSBI measurement (Ref. 4); `docs/context.md` for yields and model.
- §6 — CARL (Ref. 1), ATLAS NSBI implementation (Ref. 3), frontier of SBI (Ref. 10), mining gold (Ref. 11).
- §§1, 7–9 — Tei's slides (Ref. 0): model p. 2, likelihoods p. 3, yields p. 4, toys p. 5, histogram result p. 6, NSBI result p. 7.
- §10 — `docs/context.md` § Known code issues; `docs/env-setup-and-code-walkthrough.md` §§ Verified / Direction to verify.
- §11 — Hermans et al. (Ref. 5), BNRE (Ref. 13).
- §12 — Deep Sets (Ref. 12), Bartlett 1937 (Ref. 8), Cowan 2019 (Ref. 9), Barlow–Beeston (unverified, literature.md §E).
