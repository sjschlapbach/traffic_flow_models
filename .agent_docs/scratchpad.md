# Scratchpad — traffic_flow_models

> Per-session working memory (tasks, learnings, dead ends). Update continuously during every
> session. Do not delete previous entries — accumulate knowledge over time.
>
> This file is the sole scratchpad for the project going forward. The legacy root `scratchpad.md`
> (gitignored, untracked raw session log) has been removed — the durable lessons it contained were
> curated into the "Migrated Context" entry below before deletion.

## Format

```markdown
## Task: <short description>

### Plan

- [ ] Step 1
- [ ] Step 2

### Learnings

- YYYY-MM-DD: <learning>

### Decisions

- <decision>: <rationale>

### Dead Ends

- <approach>: <why it failed>

### Open Questions

- <question>
```

---

## Migrated Context (from legacy root scratchpad.md, pre-2026-07-02)

Curated from the legacy root `scratchpad.md` during the 2026-07-02 session that introduced
`.agent_docs/`/`.memory-bank/`. Not a raw migration — only durable, still-relevant lessons and
open items were pulled forward.

### Learnings

- For METANET, `kappa` and `delta` are the most sensitive knobs when oscillations appear in a
  simulation; reducing `delta` and increasing `kappa` are usually the first stabilizing changes
  to try.
- Run tests via the venv interpreter (activate the venv, or `venv/bin/python -m pytest`) to avoid
  system-Python import errors.
- matplotlib's frame-buffer API for video export is backend-inconsistent (`buffer_rgba()` fails
  on some backends, `tostring_rgb()` fails on macOS) — use the backend-agnostic
  `fig.canvas.print_to_buffer()` instead.
- Short-link CFL-stability handling rule of thumb (in `NetworkArbitrator.handle_short_links`):
  always stretch ramps; only fuse uniform-type, same-direction, non-ramp segments.
- SUMO detector aggregation moved from discrete time bins to rolling-window aggregation (detector
  frequency 900s → 15s, 2-minute rolling window) for smoother demand/turning-rate estimates.
  `TurningRateAggregator` deliberately mirrors `DemandAggregator`'s pattern for consistency —
  reuse that pattern for any future aggregator added to `arbitrator/`.

### Dead Ends / Open Questions

- CTM calibration exhibited weak parameter identifiability in past debugging sessions:
  regularized fits barely moved from the initial guess, with |corr| ≈ 0.99 between parameters.
  Flagged root causes at the time included CTM supply terms not scaled by `lanes` post-refactor,
  hardcoded `free_flow_speed`/`jam_density` constants baked into micro-aggregation, and a units
  mismatch in the destination boundary-condition fallback (jam density used where a flow was
  expected). **Treat CTM calibration results with skepticism until independently re-verified
  against the current code** — some of these may already be fixed by the later loop-detector
  cell-flow fix (#88) or the backbone-aggregator default-masking fix (#90), but this hasn't been
  re-confirmed as of this migration. See also `.memory-bank/progress.md`.

---

<!-- New session entries go below this line. -->
