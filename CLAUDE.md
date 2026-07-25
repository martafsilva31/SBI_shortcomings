# SBI_shortcomings — Claude Code Instructions

> Container repo. See `asimov-aymptotics-check/` for code-level conventions.
> Expert slash command: `/sbi-shortcomings` — invoke when working anywhere under this container.
> Read the global instructions at `../CLAUDE.md` for workspace-wide rules.

## Project layout

```
SBI_shortcomings/
  .claude/commands/sbi-shortcomings.md ← expert slash command
  docs/context.md                      ← physics goal, methods, key finding
  docs/plan.md                         ← roadmap (focus + next + blockers + refs)
  docs/worklogs/<slug>.md              ← read-only worklog snapshots (AUTO-EXPORTED by /workend)
  asimov-aymptotics-check/             ← sub-repo (gitignored; independent git)
    ← PyTorch Lightning training + analysis code
    ← dataset accessed via CERNBox (not committed)
```

## Environment

```bash
source setup.sh   # shows the commands; does NOT launch the container
```

The code runs inside an **Apptainer container** with an NVIDIA GPU:

```bash
# Build once (needs Apptainer, ~10 min):
apptainer build asimov-aymptotics-check/ml4fp-sbi-tutorial.sif asimov-aymptotics-check/container.def

# Every session:
apptainer shell --nv asimov-aymptotics-check/ml4fp-sbi-tutorial.sif

# Inside container, first time:
pip install -e asimov-aymptotics-check/
```

No ATLAS Athena or lsetup required.

## Sub-repos

- `asimov-aymptotics-check/` — all training and inference code. See its `README.md` for
  run instructions. Clone from: `https://github.com/martafsilva31/asimov-aymptotics-check.git`
  (GitHub credentials required; see setup.sh for the clone command).

## Worklogs rule

Worklogs are Notion-native: each task's dated, append-only running log lives on its
Notion task / plan-step page — per-task `docs/<task>.md` running logs are retired.
`/workend` auto-exports read-only snapshots to `docs/worklogs/<slug>.md` (headed
AUTO-EXPORTED, never hand-edited), at the container level — never inside the sub-repo.
Logbook entries go via `/writeup` only. The discrete task list + status live in Notion
Tasks; `docs/plan.md` is the roadmap. No daily logbook.

## Physics context

Study of NSBI (Neural Simulation-Based Inference) failure in the non-asymptotic regime.
Process: off-shell gg → H* → ZZ → 4ℓ (parton level, per-event reweightable).
Key result (revised 2026-07-25 after the 2026-07-21 audit — the earlier headline is retired):
the Asimov dataset is by construction the *mean* dataset and −2lnL is affine in the observed
counts, so the toy-average of −2lnL(μ) equals the Asimov curve plus a μ-independent constant —
exactly, at any sample size, no asymptotic assumption. Verified on the project's own tensors for
both the histogram and the exact-ratio unbinned analysis. The *median* has no such identity: it
sits strictly above the Asimov curve at every μ even in the perfect asymptotic limit, so an
Asimov-vs-median overlay cannot agree by construction.

Consequences for the older claims. The histogram is **not** the clean control the earlier notes
described: where the 1σ width ratio is defined it runs 0.92–1.16 against a large-sample
expectation of 1.185, and at 1–10 fb⁻¹ it is undefined (2000/2000 toys stay open at 1 fb⁻¹).
The NSBI exact-vs-network contrast is confounded by a 0.19% rate-term normalisation mismatch
(network path takes its rate from the cross-section table, toys are thrown from the event sample)
— every network number is pending re-measurement. The 0.874 width ratio is our own re-measurement
of Tae's tensors, not a number from his slides.

Open, as of the 2026-07-22 HWW meeting: whether to subtract each toy's own best fit before
comparing (the Δ-primary convention, provisional since 2026-07-13), and whether curves or
quantiles are the right thing to compare at all. Every width-ratio number carries that caveat.

POI: μ (signal strength modifier). No NPs in current setup.

Software stack: PyTorch Lightning, Python 3, no ATLAS Athena required for this project.

## Known issues

- Dataset is on CERNBox — requires EOS/krb5 credentials on lxplus. Not available offline.
- Requires an NVIDIA GPU — check availability on the lxplus node before launching the container (`nvidia-smi`).
- Sub-repo `asimov-aymptotics-check/` cloned from fork on 2026-06-02. Real code and results present.
