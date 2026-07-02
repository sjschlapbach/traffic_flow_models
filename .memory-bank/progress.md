# Progress — traffic_flow_models

> Living document. Update every session with what works and what's broken/fragile.
> Last updated: 2026-07-02.

## Works / stable

CTM, METANET, `Network` topology + validation, `Simulation.run` + save/load JSON round-trip,
VKT/VHT metrics, plotting, video visualization, and all four controllers (`AlineaController`,
`MetalineController`, `FlowController`, `CustomController`) — all covered by the test suite under
`tests/` (122 test functions across 14 files as of this audit). CI is green on
`python_testing.yml`/`lint.yml` for `master`/`dev`.

## Experimental / actively evolving — treat with extra care

- `Calibrator` — introduced #45, generalized to be model-agnostic #85, normalization added #92,
  divergence failsafe added #89. Still evolving on the `dev-fd-calibration` branch.
- `NetworkArbitrator` / SUMO pipeline — continuous stream of fixes through v1.1.0 and into
  `dev-js`: cell-flow-from-detector fix #88, backbone-aggregator default-masking fix #90,
  microsimulation step-length fix, spatial resampling fix #100.

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

- Cell flow is now computed from loop-detector density/speed via the fundamental relation, instead
  of a previously incorrect method (#88).
- Removed backbone-aggregator default values that caused "undesired masking effects" (#90).
- SUMO microsimulation results now get spatial resampling when needed (#100) and a corrected
  step length.

## Not yet implemented / scaffolding only

- Coordinated ramp metering (HERO scheme) — `Onramp.control_status` and the onramp-neighbor
  discovery machinery (`Network.upstream_onramps`/`downstream_onramps`) exist but aren't consumed
  by any controller yet. See `systemPatterns.md` and the `hero-ramp-metering` branch noted in
  `activeContext.md`.

## See also

- `activeContext.md` — current branch/session focus
- `systemPatterns.md` — architectural context behind the fragile areas above
- `.agent_docs/scratchpad.md` — mined historical lessons/dead-ends
