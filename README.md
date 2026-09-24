# SBI_shortcomings

Study of Neural Simulation-Based Inference (NSBI) shortcomings in the non-asymptotic regime,
using off-shell Higgs production (gg → H* → ZZ → 4ℓ) as a concrete demonstration case.

## Key finding (revised 2026-07-25)

The Asimov dataset is by construction the *mean* dataset, and −2 ln L is affine in the observed
counts, so the expectation over toys of −2 ln L(μ) equals the Asimov curve plus a μ-independent
constant, at any sample size. The *median* of the toys obeys no such identity: it lies above the
Asimov curve at every μ even in the asymptotic limit, so an Asimov-versus-median overlay cannot
agree by construction. This reframes the original question (the May 2026 statement that NSBI
diverges from Asimov at 1–10 fb⁻¹ while histograms agree at all luminosities is superseded).
Open since the 2026-07-22 HWW meeting: whether to subtract each toy's own best fit before
comparing, and whether curves or quantiles are the right comparison. Full statement and the
three retracted claims: `docs/context.md`, "Key finding".

## Sub-repos

- `asimov-aymptotics-check/` — training and analysis code (PyTorch Lightning). Fork of Tei Park's
  original code. Clone from `https://github.com/martafsilva31/asimov-aymptotics-check.git`.

## Quick start

```bash
source setup.sh
# First time: clone the sub-repo (see setup.sh for the git clone command)
cd asimov-aymptotics-check
# See asimov-aymptotics-check/README.md for training + inference commands
```

## Documentation

- `docs/context.md` — physics goal, signal model, datasets, key finding, proposed fixes
- `docs/plan.md` — roadmap (current focus + next + blockers + refs)
- `docs/worklogs/` — read-only exports of the Notion worklogs (the worklog lives on the Notion task page)

## Team

- Marta Fernandes da Silva (lead)
- Tae Hyoun (Tei) Park (original code)
- Lydia (FCC dataset coordination)

