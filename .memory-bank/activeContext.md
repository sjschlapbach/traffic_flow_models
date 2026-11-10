# Active Context — traffic_flow_models

> Living document. Overwrite (don't indefinitely accumulate) each session with the current
> focus and recent changes. Last updated: 2026-11-10.

## Current branch state

- Branch: `pipeline-removal` (branched from `dev-js`). Changes are uncommitted; the maintainer
  commits manually.
- Latest published release: **v1.1.0**. This branch is a **breaking** change and should ship as a
  major version (`feat!`/`BREAKING CHANGE:` footer so git-cliff bumps correctly).

## This session's work

Removed the whole SUMO/OSM data-generation pipeline (ADR-0001 in `.agent_docs/DECISIONS.md`):
- deleted `simulator/`, `arbitrator/`, `src/demo/run_pipeline.py`,
  `src/demo/road_params_config.json` and `src/demo/scenarios/`;
- deleted the SUMO tests, the two SUMO CI workflows, the SUMO docs pages and the `osmnx` dependency.

The package now covers CTM/METANET, network components, controllers and (experimental)
calibration against user-provided results-JSON files. The ground-truth format is documented in
`docs/experimental/calibration.md`.

`Simulation.run(plot_micro_results=True)` became `run(reference_results_path=...)`. The legend
label is now "Reference" and the `micro_*` names became `reference_*`. The `"MICRO"` metadata
marker is kept for backward compatibility.

## Known parallel branches (not merged)

- `dev-fd-calibration` — FD fitting from microsimulation data (#87). It **depends on the removed
  pipeline**, so it needs rework before merging.
- `hero-ramp-metering` — coordinated ramp-metering scaffold.
- `paper-macro-run-figures` — STRC 2026 paper figures. May use `plot_micro_results`, so check
  before rebasing.

## Things worth double-checking

- `docs/development/contributing.md` says PRs target `dev`, but work happens on `dev-js`.

## See also

- `progress.md` — what's verified working vs. fragile
- `projectbrief.md` / `techContext.md` / `systemPatterns.md` — stable background context
