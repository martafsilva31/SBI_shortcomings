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
  docs/<task>.md                       ← per-task study-note (running log → summary → Notion)
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

## Study-notes rule

Each meaningful task gets one `SBI_shortcomings/docs/<task>.md` study-note (running
log → cleaned summary → Notion Logbook entry), at the container level — never inside
the sub-repo. The discrete task list + status live in Notion Tasks; `docs/plan.md` is
the roadmap. No daily logbook.

## Physics context

Study of NSBI (Neural Simulation-Based Inference) failure in the non-asymptotic regime.
Process: off-shell gg → H* → ZZ → 4ℓ (parton level, per-event reweightable).
Key result: at 1–10 fb⁻¹, the median of the toy NLL scans is systematically narrower/steeper than
the NSBI Asimov scan (toy-level overconfidence; direction measured 2026-07-10 — width ratio 0.874
at 300 fb⁻¹), while histogram analysis satisfies Asimov ≈ median(toys) at all luminosities. Root cause: high-dimensional NN space is non-asymptotic even with
O(100) total events.

POI: μ (signal strength modifier). No NPs in current setup.

Software stack: PyTorch Lightning, Python 3, no ATLAS Athena required for this project.

## Known issues

- Dataset is on CERNBox — requires EOS/krb5 credentials on lxplus. Not available offline.
- Requires an NVIDIA GPU — check availability on the lxplus node before launching the container (`nvidia-smi`).
- Sub-repo `asimov-aymptotics-check/` cloned from fork on 2026-06-02. Real code and results present.
