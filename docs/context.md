# SBI_shortcomings — Project Context

## Goal

Characterise and correct the failure of Neural Simulation-Based Inference (NSBI) to satisfy the
Asimov property (Asimov dataset == median of toy pseudo-experiments) in the non-asymptotic regime,
and identify mitigation strategies suitable for rare-process analyses.

## Background

Standard NSBI analyses assume the Asimov dataset gives the expected experimental sensitivity.
This assumption holds only in the asymptotic regime — i.e., when every "bin" in the test-statistic
space has sufficient statistics. For high-dimensional neural-network-based analyses the effective
per-event "bin" is tiny, so even with O(100) total signal events the analysis is non-asymptotic.
Histogram-based analyses satisfy the Asimov property because physical bins can always be chosen
wide enough to be asymptotic; NSBI cannot be tuned this way.

The starting point is a concrete demonstration by Tae Hyoun (Tei) Park using off-shell Higgs
production (gg → H* → ZZ → 4ℓ), where the effect is large and unambiguous at 1–10 fb⁻¹.

### Signal probability model

The probability density under the SBI hypothesis (incorporating signal-background-interference):

```
p_SBI(x|μ) = [μ ν_S p_S(x) + √μ ν_I p_I(x) + ν_B p_B(x)]
             / [μ ν_S + √μ ν_I + ν_B]
```

Equivalently re-arranged to avoid negative interference:

```
p_SBI(x|μ) = [(μ-√μ) ν_S p_S(x) + √μ ν_SBI p_SBI(x) + (1-√μ) ν_B p_B(x)]
             / [(μ-√μ) ν_S + √μ ν_SBI + (1-√μ) ν_B]
```

### Observed yields at μ=1 (parton level, no detector simulation)

| Component | 1 fb⁻¹ | 10 fb⁻¹ | 100 fb⁻¹ | 1000 fb⁻¹ |
|-----------|--------|---------|----------|-----------|
| SIG       | 0.310  | 3.099   | 30.990   | 309.897   |
| BKG       | 5.953  | 59.530  | 595.303  | 5953.034  |
| INT       | -0.606 | -6.064  | -60.643  | -606.433  |
| SBI total | 5.656  | 56.565  | 565.650  | 5656.498  |

## Inputs

- **Dataset**: parton-level gg → ZZ → 4ℓ events with per-event matrix element weights (signal,
  background, interference). Stored on CERNBox due to size: Tei's public share
  `https://cernbox.cern.ch/s/AeKNoWslVzwmAdt` (all his MCFM datasets), folder `zz4l/ggzz4l_sm/`
  with `ggzz4l_{sig,bkg,sbi}.csv` + `ggzz4l_xs.json` (total cross-sections — normalise the sum
  of weights to these values, then multiply by the tested luminosity). Local copy:
  `data/zz4l/ggzz4l_sm/` (gitignored; `$DATASET_DIR` in setup.sh).
- **Code repository**: fork of Tei's code at
  `https://github.com/martafsilva31/asimov-aymptotics-check.git`
  (clone into `asimov-aymptotics-check/` inside this container)
- **Observables**: 4-momenta of the 4 final-state leptons (16 input features for NNs);
  M₄ℓ for the histogram analysis

## Outputs

- NLL scan curves: Asimov vs. median(toys) ± 1σ, for histogram and NSBI, at 1/10/100/1000 fb⁻¹
- Quantitative Asimov-breakdown metric as a function of luminosity and analysis dimensionality
- (Future) NLL scan curves after proposed fixes (toy-ensemble training, alternative test statistics)
- (Future) Short paper or ATLAS note documenting the failure mode and mitigations

## Analysis methods

### Histogram

- Bin M₄ℓ distribution for SIG, BKG, INT components
- Likelihood = product of Poisson probabilities per bin: ℒ = ∏_bins 𝒫(n|ν)
- Asimov satisfies median(toys) property reliably across all luminosity values

### NSBI

- Reference density: p_ref = p_bkg
- Two CARL classifiers estimate density ratios: p_sig/p_ref and p_sbi/p_ref
- Likelihood = Poisson(total rate) × ∏_events [p_SBI(x|μ)/p_ref(x)]
- Poisson term is NOT estimated by NNs (uses true total rate)
- Training: balanced datasets, no luminosity information — luminosity applied only at inference
- Architecture: PyTorch Lightning, 16 input features
- Toy generation: w_toy ~ Poisson(λ=w) for each event weight
- Validation: 100–200 toy pseudo-experiments per luminosity point

## Key finding (first meeting, 2026-05-20)

At 1–10 fb⁻¹, the NSBI median(toys) diverges significantly from the Asimov prediction:
NLL curves are systematically steeper (NSBI is overconfident). Histogram analysis shows
Asimov ≈ median(toys) at all luminosities. Convergence of NSBI only occurs at ~1000 fb⁻¹
where the Poisson rate term dominates. Root cause: NN density ratios are trained on a fixed
Asimov-like sample and cannot account for statistical fluctuations in sparse high-dimensional
event space. This is a systematic bias intrinsic to the CARL/NSBI procedure in the non-asymptotic
regime — not a training artefact that can be fixed by better regularisation alone.

## Known code issues

- **Missing extended Poisson term in the NSBI NLL** (Tei, 2026-07-09): Tei realised the NSBI
  NLL-scan code omits the extended Poisson (total-rate) term. Agreed procedure: first reproduce
  his results with the notebooks as-is, then add the term and re-run.

## Proposed fixes to study

1. **Toy-ensemble training**: train NNs on multiple pseudo-experiments (deep sets architecture),
   so the network learns that confidence should scale with available statistics
2. **Alternative test statistics**: Bartlett-corrected or other non-asymptotic test statistics
3. **MC stat uncertainties**: add MC statistical uncertainties to both histogram and NSBI to
   make analyses more conservative
4. **Dimensionality study**: test 4D histogram analysis at 1 fb⁻¹ to show the problem arises
   from high dimensionality, not from machine learning per se

## Future dataset considerations

- Current dataset: off-shell Higgs production (parton level, no detector sim)
- Candidate FCC dataset: H→SS decay (non-asymptotic regime, potentially measurable at FCC not LHC)
  — requires quantum interference with background or EFT effects
- Lydia (FCC global EFT group convener) will coordinate on available samples
- For now: continue with current dataset at 1 fb⁻¹ to develop methodology

## Team

- **Marta Fernandes da Silva** — lead, reproducing + extending
- **Tae Hyoun (Tei) Park** — original code author, will help onboard Marta after teaching
- **Lydia** — collaborator, FCC dataset coordination (convener of FCC global EFT group)

## Key references

- Cranmer, Pavez, Louppe — "Approximating Likelihood Ratios with Calibrated Discriminative
  Classifiers" (CARL) [REF NEEDED]
- Cowan et al. — "Asymptotic formulae for likelihood-based tests of new physics" [REF NEEDED]
- Tei's slides: `asimov_vs_medtoys_test_statistic_studies.pdf` (shared at first meeting 2026-05-20)
