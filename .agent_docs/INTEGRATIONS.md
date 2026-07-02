# Integrations — traffic_flow_models

> External tools, simulators, and their config patterns. Update when an external dependency
> changes. Deliberately excludes pip-installed Python libraries like CasADi/numpy/scipy — those
> belong in `.memory-bank/techContext.md`; this file is for things with their own CLI/binary/
> config surface outside the Python packaging system.

## SUMO microsimulation

The experimental SUMO integration lives in `src/traffic_flow_models/simulator/` (single-run
wrapper + pipeline orchestration) and `src/traffic_flow_models/arbitrator/` (SUMO↔macroscopic
bridge — see `.memory-bank/systemPatterns.md` for the pipeline diagram).

**Binaries required** (must be on `PATH`): `netconvert` (OSM → SUMO network conversion),
`duarouter` (validates/expands raw `<trip>` XML into `<route>`s, used with
`--repair --ignore-errors`), `sumo` (headless microsimulation run — **not** `sumo-gui`).

**Environment**: `SUMO_HOME` env var is required by `SUMOPipeline.generate_demand()` (raises
`EnvironmentError("Please set the 'SUMO_HOME' environment variable.")` if unset) — used to locate
`$SUMO_HOME/tools/randomTrips.py`, invoked as `sys.executable randomTrips.py ...`.

**No TraCI**: no `traci`/`sumolib` Python bindings are used anywhere in this codebase. All SUMO
interaction is via `subprocess.run`/`subprocess.CalledProcessError` to the CLI binaries above,
plus manual `xml.etree.ElementTree` parsing of SUMO's XML output — there is no live/step-by-step
control loop into SUMO.

**CI install recipe** (the canonical way to get a working SUMO environment; see
`.github/workflows/scripts_sumo_pipeline.yml` and `scripts_scenarios_pipeline.yml`):
```bash
sudo add-apt-repository -y ppa:sumo/stable
sudo apt-get update
sudo apt-get install -y sumo sumo-tools
export SUMO_HOME=/usr/share/sumo
sumo --version && netconvert --version   # sanity check
```
On macOS, the SUMO installer may not set `SUMO_HOME` correctly — point it at something like
`/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo` (see
`README.md`'s installation section).

**File formats read/written** (all under `SUMOPipeline.output_dir`, named `f"{name}.<ext>"`):

| File | Produced by | Consumed by |
| --- | --- | --- |
| `<name>.osm` | `SUMOPipeline.fetch_OSM()` (via `osmnx`) | `convert_to_sumo()` |
| `<name>.net.xml` | `netconvert` (`convert_to_sumo()`) | most downstream stages; also parsed directly via `ET.parse` for edge filtering |
| `<name>.rou.xml` | `generate_demand()` (merges urban + highway trip files) | `SUMOSimulation` run |
| `<name>.sumocfg` | `SUMOSimulation.write_config()` | `sumo` CLI run |
| `edge_data_config.add.xml` | `SUMOSimulation.write_config()` | `sumo` run (declares `<edgeData>` output, period 60s) |
| `edge_data_output.xml` | SUMO run | `build_destination_bc_from_sumo_edges()` |
| `<name>_stats.xml` | SUMO run (`--statistic-output`) | `SUMOSimulation.print_summary()` |
| `<name>detectors.xml`, `detectors_output.xml`, `<name>_detectors_spec.csv` | `LoopDetectorGenerator.generate()` | `DemandAggregator`, `TurningRateAggregator`, `BackboneStateAggregator` |
| `road_params_config_path` (JSON) | user-provided | `NetworkArbitrator` (per-road-type `lane_capacity`/`jam_density`/`free_flow_speed`, keys: `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `default`) |

**End-to-end pipeline** (see `src/demo/run_pipeline.py` for the canonical worked example):
`fetch_OSM` → `convert_to_sumo` → `generate_demand` → `create_consolidated_network` (via
`NetworkArbitrator`) → `generate_detectors` (via `LoopDetectorGenerator`) → run SUMO
(`SUMOSimulation.write_config()`/`.run_simulation()`) → `DemandAggregator.run(...)` →
`build_destination_bc_from_sumo_edges(...)` → `compute_splits()` (via `TurningRateAggregator`,
falling back to `NetworkArbitrator.compute_lane_based_splits` per-node when detector data is
missing) → `BackboneStateAggregator.run(...)` (produces ground-truth `micro_results.json`) →
dispatch to `Simulation.run(...)` (CTM) or `Calibrator.calibrate_model_params(...)` then simulate
(METANET).

## OpenStreetMap via `osmnx`

`SUMOPipeline.fetch_OSM()` calls `osmnx.graph_from_place(location, ...)` — this is the only step
in the pipeline that requires live network access to fetch real-world data. The alternative
`--sumo-cfg-dir <dir>` mode (see `run_pipeline.py`) bypasses this entirely by using pre-provided
`.net.xml`/`.rou.xml`/`.sumocfg` files, needing no network access — this is what CI's
`scripts_scenarios_pipeline.yml` uses for reproducibility.

## git-cliff (changelog automation)

Config: `cliff.toml`. `conventional_commits = true`, `filter_unconventional = true`,
`require_conventional = false`. Commit-prefix → changelog-section mapping: `feat`→Features,
`enhance`→Enhancements, `fix`→Bug Fixes, `doc`→Documentation, `perf`→Performance,
`refactor`→Refactor, `style`→Styling, `test`→Testing, `chore`/`ci`→Miscellaneous Tasks,
`revert`→Revert, commit body matching `.*security`→Security, everything else→Other.
`skip = true` parsers exclude `chore(release): prepare for`, `chore(deps...)`, `chore(pr)`,
`chore(pull)` commits from the changelog entirely. `sort_commits = "oldest"`,
`topo_order_commits = true`, `initial_tag = "0.1.2"`, `breaking_always_bump_major = true`.

Used both for `git-cliff --bump` (local version-bump step before tagging a release) and inside
`release.yml` (`git-cliff --bump --output CHANGELOG.md`, tag-triggered). See the release mechanics
already summarized in `.memory-bank/techContext.md` — never hand-edit `CHANGELOG.md`.

## MkDocs Material + mkdocstrings (docs site)

Config: `mkdocs.yml`. Theme: `material` (indigo palette, light/dark toggle, tabs/sections
navigation, search, code copy/annotate). `mkdocstrings` python handler config: `paths: [src]`,
Google docstring style, `show_source: true`, members ordered by source. MathJax enabled via
`pymdownx.arithmatex` for rendering model equations. Nav mirrors the `docs/` directory structure
(see `.agent_docs/STRUCTURE.md`).

Built by `.github/workflows/docs_build.yml` (`mkdocs build`, every push/PR — verifies the site
builds) and deployed by `docs_release.yml` (`mkdocs gh-deploy --force`, tag-triggered, publishes
to GitHub Pages at `https://sjschlapbach.github.io/traffic_flow_models/`). `docs/` is user-facing
and must be kept in sync with the implementation per `AGENTS.md`'s mandatory update protocol.
