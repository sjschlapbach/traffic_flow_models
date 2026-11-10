# Progress — traffic_flow_models

> Living document. Update every session with what works and what's broken/fragile.
> Last updated: 2026-11-10.

## Works / stable

CTM, METANET, `Network` topology + validation, `Simulation.run` + save/load JSON round-trip,
VKT/VHT metrics, plotting, video visualization, and all four controllers (`AlineaController`,
`MetalineController`, `FlowController`, `CustomController`) — all covered by the test suite under
`tests/` (see `.agent_docs/STRUCTURE.md` for the per-file table). CI is green on
`python_testing.yml`/`lint.yml` for `master`/`dev`.

## Experimental / actively evolving — treat with extra care

- `Calibrator` — introduced #45, generalized to be model-agnostic #85, normalization added #92,
  divergence failsafe added #89. Still evolving on the `dev-fd-calibration` branch.
- `Simulation.run(reference_results_path=...)` reference-data overlay — new on
  `pipeline-removal` (replaces `plot_micro_results`), covered by `tests/simulation_test.py`.

## Known fragile/sensitive areas — be cautious editing

- **METANET node-processing order/causality**: multiple historical fixes (#26–#82) around
  merge/diverge virtual upstream-speed and downstream-density computation. Now numerically stable,
  but this is the area with the highest historical bug density in the whole codebase — read
  `systemPatterns.md`'s state-packing section and the relevant `metanet.py` docstrings carefully
  before changing node-processing logic.
- **CTM calibration weak parameter identifiability**: past debugging sessions (mined from the
  legacy scratchpad — see `.agent_docs/scratchpad.md`) found regularized CTM calibration fits
  barely move from the initial guess, with |corr| ≈ 0.99 between parameters. This is a known open
  *modeling* limitation, not obviously a code bug — treat CTM calibration results with skepticism
  until independently re-verified against the current code (some flagged root causes at the time
  may have since been fixed by #88/#90, but this hasn't been re-confirmed).

## Recently fixed (v1.1.0 → dev-js)

- Saved results whose per-link cell count differs from the network are spatially resampled on
  load (#100). This still applies to user-provided reference data.
- (#88, #90 and the SUMO step-length fix concerned the removed SUMO pipeline.)

## Removed (pipeline-removal branch, unreleased)

- SUMO/OSM pipeline (`simulator/`, `arbitrator/`, `run_pipeline.py`, SUMO scenario configs, SUMO
  CI workflows, `osmnx`). See ADR-0001 in `.agent_docs/DECISIONS.md`.

## Not yet implemented / scaffolding only

- Coordinated ramp metering (HERO scheme) — `Onramp.control_status` and the onramp-neighbor
  discovery machinery (`Network.upstream_onramps`/`downstream_onramps`) exist but aren't consumed
  by any controller yet. See `systemPatterns.md` and the `hero-ramp-metering` branch noted in
  `activeContext.md`.

## See also

- `activeContext.md` — current branch/session focus
- `systemPatterns.md` — architectural context behind the fragile areas above
- `.agent_docs/scratchpad.md` — mined historical lessons/dead-ends
