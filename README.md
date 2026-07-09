# SBI_shortcomings

Study of Neural Simulation-Based Inference (NSBI) shortcomings in the non-asymptotic regime,
using off-shell Higgs production (gg → H* → ZZ → 4ℓ) as a concrete demonstration case.

## Key finding

At low luminosities (1–10 fb⁻¹), NSBI analysis shows significant discrepancy between the
Asimov expectation and the median of toy pseudo-experiments. Histogram-based analysis satisfies
the Asimov property at all luminosities. The NSBI overconfidence problem arises because the
high-dimensional NN event space is non-asymptotic even when total event counts appear adequate.

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
- `docs/<task>.md` — per-task study-note (running log → summary → Notion Logbook)

## Team

- Marta Fernandes da Silva (lead)
- Tae Hyoun (Tei) Park (original code)
- Lydia (FCC dataset coordination)

First meeting: 2026-05-20. Next meeting: 2026-05-27 at 16:00 CEST.
