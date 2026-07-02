# Active Context — traffic_flow_models

> Living document. Overwrite (don't indefinitely accumulate) each session with the current
> focus and recent changes. Last updated: 2026-07-02.

## Current branch state

- Branch: `dev-js`, clean working tree.
- 4 commits ahead of `master`, none yet released:
  - `7a3ae22` — fix: add spatial resampling to microscopic simulation results if necessary (#100)
  - `d008ff3` — fix(SumoPipeline): update microsimulation step length
  - `690bd77` — chore: update docstrings to follow consistent structure across codebase (#99)
  - `798b0ef` — chore: update README
- Latest published release: **v1.1.0** (2026-05-23). Everything on `dev-js` is unreleased delta
  on top of it.

## This session's work

Rewrote `AGENTS.md`/`CLAUDE.md` (which had been copy-pasted from an unrelated "FEDORA Platform"
project template) to reflect this repo's actual architecture, tooling, and conventions. Then
populated `.agent_docs/` and `.memory-bank/` for the first time — both directories previously
existed but were empty scaffolding.

## Known parallel branches (not merged into `dev-js`/`master`)

Noted briefly so future agents don't assume `dev-js` is the only active thread — not deep-dived:

- `dev-fd-calibration` — fundamental-diagram parameter fitting from microsimulation data (#87).
- `hero-ramp-metering` — coordinated ramp-metering scaffold (see `Onramp.control_status` in
  `systemPatterns.md`).
- `paper-macro-run-figures` — figure/results generation for the STRC 2026 paper (see
  `projectbrief.md`).
- Also present: local branch `dev`, remotes `demand` and `stepwise`.

## Things worth double-checking before relying on them

- `docs/development/contributing.md` states PRs should target `dev`, but the actual current work
  is happening on `dev-js` (branched off `master`). Worth clarifying with the maintainer rather
  than assuming one or the other is stale/wrong.
- The MkDocs `docs/` site is very recent (introduced in v1.1.0, #98) — verify it's still in sync
  before treating it as ground truth for anything changed on `dev-js` or the other branches above
  since.

## See also

- `progress.md` — what's verified working vs. fragile
- `projectbrief.md` / `techContext.md` / `systemPatterns.md` — stable background context
