# SBI_shortcomings — Copilot Instructions

> Container repo. See each sub-repo for code-level conventions.
> Expert agent: `@SBI_shortcomings` — invoke from any sub-repo within this workspace.

## Project layout
```
SBI_shortcomings/
  .github/agents/SBI_shortcomings.agent.md     ← expert agent (Copilot)
  .claude/commands/SBI_shortcomings.md         ← slash command (Claude Code)
  docs/context.md                      ← physics goal, inputs, outputs
  docs/plan.md                         ← roadmap (focus + next + blockers + refs)
  docs/<task>.md                       ← per-task study-note (running log → summary → Notion)
  asimov-aymptotics-check/  ← sub-repo (gitignored; independent git)
```

## Environment
```bash
source setup.sh
```

## Sub-repos
- `asimov-aymptotics-check/` — see `asimov-aymptotics-check/README.md` for build/run instructions

## Logbook rule
All logbook entries go to `SBI_shortcomings/docs/logbook/`, not inside any sub-repo.

## Physics context
<!-- 2–4 lines: what is being measured / studied, why -->

## Known issues
<!-- Persistent gotchas -->
