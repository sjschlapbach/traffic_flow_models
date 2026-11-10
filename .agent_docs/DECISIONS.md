# Architectural Decision Records — traffic_flow_models

This file is **forward-only**: it captures decisions made from the introduction of
`.agent_docs/` onward (2026-07-02). Historical architectural decisions are not backfilled here —
durable lessons mined from the project's history live in `.agent_docs/scratchpad.md` instead.

Add a non-trivial technical or design choice as a new entry below, most recent first.

## Template

### ADR-XXXX: <short title>

- **Date**: YYYY-MM-DD
- **Status**: Proposed | Accepted | Superseded by ADR-YYYY
- **Context**: <what problem/tension prompted this>
- **Decision**: <what was decided>
- **Consequences**: <trade-offs accepted, follow-up work implied>

---

### ADR-0001: Remove the SUMO/OSM pipeline and `arbitrator/` from the package

- **Date**: 2026-11-10
- **Status**: Accepted
- **Context**: The package mixed two concerns: the macroscopic models, controllers and
  calibration, and an experimental data-generation pipeline (OSM fetch via `osmnx` → `netconvert`
  → SUMO demand/detectors/run → `arbitrator/` aggregation into a macroscopic `Network` and
  ground-truth JSON). The pipeline needed an external SUMO install, `SUMO_HOME`, and its own CI
  workflows, and it carried most of the maintenance burden. Nothing in `model/`, `controller/` or
  `network/` imported it. The only coupling was the ground-truth results JSON
  (`metadata.model_type="MICRO"`, mainline-only) plus a SUMO-labelled plot overlay in
  `Simulation.run(plot_micro_results=True)`.
- **Decision**: Delete `simulator/`, `arbitrator/`, `src/demo/run_pipeline.py`,
  `src/demo/road_params_config.json`, `src/demo/scenarios/` (SUMO configs), the two SUMO test
  files, the `scripts_sumo_pipeline`/`scripts_scenarios_pipeline` workflows, the SUMO docs pages
  and the `osmnx` dependency. The package now assumes ground-truth files are provided in the
  results-JSON format (documented in `docs/experimental/calibration.md`). Calibration stays
  experimental. The plot overlay is generalized, not deleted:
  `run(reference_results_path=...)` replaces `plot_micro_results` with its hardcoded
  `results_dir/micro_results.json`, and the legend reads "Reference". The `"MICRO"`
  `model_type` marker and `load_mainline_only` are kept unchanged, so existing mainline-only
  files still load in `visualize`/`visualize_comparison`.
- **Consequences**: Breaking public API change. `SUMOSimulation`, `SUMOPipeline`,
  `NetworkArbitrator`, `LoopDetectorGenerator`, `DemandAggregator`, `TurningRateAggregator` and
  `BackboneStateAggregator` are no longer exported, and `plot_micro_results` is gone. The next
  release should be a major version (`feat!`/`BREAKING CHANGE:` footer for git-cliff). Users who
  need data generation must convert their data to the results JSON themselves. The removed code
  remains in git history, and the `dev-fd-calibration` branch still depends on it.

---

<!-- New ADRs go above this line, most recent first. -->
