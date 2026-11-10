# Structure — traffic_flow_models

> Annotated directory tree with module responsibilities, down to key classes/functions per file.
> Update whenever a file/folder is added, moved, or removed. This is the detailed companion to
> the higher-level "Project Layout" tree in `AGENTS.md` — that one is the quick-scan summary,
> this one is the "where exactly do I find X" reference.

## `src/traffic_flow_models/` — the installable package

### `model/` — mathematical traffic-flow models (built on CasADi symbolic graphs)

- **`ctm.py`** — `CTM` class (first-order Cell Transmission Model, stateless).
  `network_update_function(network, num_flows, num_densities, num_speeds, num_origins,
  num_onramps, num_offramps, num_splits, num_destinations, dt) -> casadi.Function` (named
  `"ctm_network_step"`). Fundamental-diagram helpers: `critical_density(...)`,
  `backward_wave_speed(...)`. Calibration-interface methods (`get_default_calibration_params`,
  etc.) all raise `NotImplementedError` by design — CTM has no calibratable parameters, its FD is
  fully determined by link geometry/capacity. Internal helpers: `_update_motorway_link` (Godunov
  demand/supply update), `_compute_normalized_splits`, `_compute_node_maximum_outflows`,
  `_compute_offramp_outflows`. Docstring flags a known causality caveat: first-cell supply
  restriction uses the *previous-step* density of the outgoing link, not a fully self-consistent
  next-step value.
- **`metanet.py`** — `METANET` class (second-order model). `METANETParams` /
  `METANETSymbolicParams` TypedDicts hold `tau` (relaxation time), `nu` (speed diffusion),
  `kappa` (density smoothing), `delta` (onramp-merge speed-drop factor), `phi` (lane-drop
  capacity-drop factor), `alpha` (FD shape, scalar or per-link dict). Full calibration interface
  implemented (`get_default_calibration_params`, `get_calibration_bounds`, `validate_model_params`,
  `model_params_to_vec`/`model_params_vec_to_dict`, `set_up_symbolic_model_params`, etc.). Core
  per-cell update: `cell_update(params, link, cell, upstream_flow, previous_flow,
  previous_density, downstream_density, upstream_speed, upstream_onramp_inflows, previous_speed,
  dt) -> (density, speed, flow)` — relaxation + convection + anticipation − onramp-merge drop −
  lane-drop deceleration. `network_update_function(...)` builds
  `casadi.Function("metanet_network_step", ...)`, processing nodes with destinations-last
  ordering so all incoming flows are known before destination outflow is derived.
- **`helpers.py`** — model-agnostic shared functions: `store_and_forward_update(capacity,
  jam_density, backward_wave_speed, density, demand, queue, dt, metering_rate=None) -> (inflow,
  updated_queue)`, `update_queue(...)`, `compute_node_outflows(node, flows, node_splits) ->
  dict[outgoing_link_id, outflow]`.

### `network/` — topology, simulation execution, calibration

- **`cell.py`** — `Cell`: minimal container (`length`, `upcoming_lane_drop`, doubly-linked-list
  `upstream`/`downstream` pointers set by `MotorwayLink`). No FD parameters (those live on the
  parent link).
- **`motorway_link.py`** — `MotorwayLink(length, lanes, lane_capacity, free_flow_speed,
  jam_density, id=None, origin_node_id=None, destination_node_id=None)`: manages cells as a
  doubly-linked list. `add_cell(...)`, `partition_link(preferred_cell_size, dt,
  upcoming_lane_drop=0)` (CFL-condition discretization — see `systemPatterns.md`),
  `first_cell()`/`last_cell()`/`get_cell(index)`/`enumerate_cells()`.
- **`origin.py` / `destination.py`** — trivial virtual containers, just `id` +
  `destination_node_id`/`origin_node_id`.
- **`onramp.py`** — `Onramp(length, lanes, lane_capacity, free_flow_speed, jam_density, id=None,
  controller=None, ...)`: single-cell store-and-forward link with an optional `controller`.
  `control_status: Literal["unset","hero_master","hero_slave"]` and
  `upstream_onramps`/`downstream_onramps` exist as scaffolding for coordinated ramp metering —
  see `systemPatterns.md`, not yet consumed by any controller.
- **`offramp.py`** — `Offramp`: same FD-param pattern as `Onramp`, no controller, no length
  (store-and-forward "point" link).
- **`node.py`** — `Node(id=None, incoming=None, outgoing=None)`: junction connecting links.
  `add_incoming`/`add_outgoing`/`remove_*_by_id`/`set_incoming`/`set_outgoing` auto-manage
  back-references; `validate()` enforces allowed incoming/outgoing types.
- **`network.py`** — `Network(nodes=None)`: main topology container. State/disturbance vector
  (de)serialization for CasADi (`network_dict_to_state_vec`, `state_vec_to_network_dict`,
  `network_dict_to_disturbance_vec`, `disturbance_vec_to_network_dict` — see the packing-order
  contract in `systemPatterns.md`). `set_onramp_relations`/`update_all_onramp_neighbors` (BFS
  onramp-neighbor discovery). `validate() -> bool` (structural rules: connectivity, exactly one
  Origin/Destination per boundary node, etc.). `_compute_upcoming_lane_drop(link)`.
  `save_to_txt`/`save_to_json`/`load_from_json`. `plot(figsize, show, save_path)`.
- **`simulation.py`** — `Simulation(network, model, model_params=None)` (largest file, ~3800
  lines). Main entry `run(duration, dt, origin_demands, turning_rates, destination_flow_bc,
  destination_density_bc, ..., preferred_cell_size=0.5, plot_results=True,
  reference_results_path=None, results_dir=None) -> (time_array, state_history,
  disturbance_history)`; `reference_results_path` overlays a mainline-only reference file on the
  link plots. Also: `save_results`/`load_results` (JSON, with a `load_mainline_only` mode for
  externally provided ground truth; files marked `metadata.model_type == "MICRO"` are
  auto-loaded that way by the visualizers),
  `resample_results_file(source_filepath, dest_filepath, target_time_array)` (classmethod, linear
  interpolation onto a new time grid), `compute_metrics(states, dt, timesteps, ignore_queues=False)
  -> (VKT, VHT, overall_avg_speed)`, `plot_results(...)`, `visualize(results_filepath,
  output_filepath, fps=25, subsampling=1, ...)` (AVI export via OpenCV, color-coded by density),
  `visualize_comparison(...)` (side-by-side comparison of several results files).
- **`calibrator.py`** — `Calibrator(network)` (~1500 lines). `calibrate_model_params(
  ground_truth_filepath, model, window_size=50, stride=None, regularization_weight=0.0,
  use_parameter_search=False, ..., use_disturbance_from_file=True, origin_demands_fn=None,
  turning_rates_fn=None, ...) -> (calibrated_params, OptimizeResult, param_history)` — sliding-
  window multi-step-prediction residuals fit via `scipy.optimize.least_squares(method='trf')`.
  `use_parameter_search=True` mode does Latin Hypercube multi-start (`scipy.stats.qmc
  .LatinHypercube`, seed 42). `analyze_parameter_correlation` (Jacobian → condition
  number/correlation matrix, flags condition number > 1e6 as poor identifiability). Model-agnostic
  via the calibration interface each model class implements — see `systemPatterns.md`.

### `controller/` — ramp-metering control strategies (duck-typed, no ABC)

- **`flow_controller.py`** — `FlowController(onramp, flow)`: fixed-rate metering.
- **`alinea.py`** — `AlineaController(onramp, measurement_link_id, measurement_cell_idx, gain,
  density_setpoint)`: classic ALINEA feedback law.
- **`metaline.py`** — `MetalineController(onramp, measurement_cells, gain_matrix,
  density_setpoints)`: MIMO feedback generalization of ALINEA.
- **`custom_controller.py`** — `CustomController(onramp, controller_fn, params=None)`: wraps an
  arbitrary user callable; the sanctioned extension point for custom metering logic.

All expose `compute_regulated_flow(onramp_queues, flows, densities) -> casadi.SX`, called from
inside the model's `network_update_function` when `onramp.controller is not None`.

## `src/demo/` — demo scripts and example networks (not part of the installed package)

- **`ctm_simulation.py` / `metanet_simulation.py`** — CLI demos (`--scenario A..E2`, `--no-plot`,
  `--generate-video`) using prebuilt networks/demand from `scenarios.py`.
- **`scenarios.py`** (~684 lines) — library of hand-built demo `Network`s (`setup_network_a`
  through `_e2`); canonical example of manual topology construction and all 4 controller types
  attached post-hoc.
- **`plot_demo_network.py`** — minimal topology-only example (no simulation).
- **`metanet_calibration.py`** (~1049 lines) — dedicated METANET calibration demo, exercising
  `Calibrator.calibrate_model_params` with diagnostic plotting.

## `tests/` — flat pytest suite at repo root

| File | Tests under | Roughly covers |
| --- | --- | --- |
| `cell_test.py` | `Cell`/`MotorwayLink` partitioning | 2 |
| `controller_test.py` | all 4 controllers + `store_and_forward_update` | 9 |
| `ctm_test.py` | `CTM` model | 6 |
| `destination_test.py` | `Destination` | 2 |
| `metanet_test.py` | `METANET` model | 6 |
| `motorway_link_test.py` | `MotorwayLink` linked-list structure | 5 |
| `node_test.py` | `Node` validation | 6 |
| `network_test.py` | `Network` (largest, CRUD/validate/packing/persistence) | 64 |
| `offramp_test.py` | `Offramp` | 4 |
| `onramp_test.py` | `Onramp` + neighbor discovery | 5 |
| `origin_test.py` | `Origin` | 1 |
| `simulation_test.py` | `Simulation` metrics + reference-data overlay | 4 |
| `visualization_test.py` | `Simulation` visualization/video export | 9 |

No `conftest.py`. No tests are skipped/xfail and none need external tools.

## Other top-level directories

- **`docs/`** — user-facing MkDocs documentation (deployed to GitHub Pages):
  `getting-started/` (installation, quickstart), `concepts/` (network, state-space),
  `models/` (ctm, metanet, comparison), `control/` (ramp-metering), `simulation/` (running,
  metrics, visualization), `experimental/` (index/stability tiers, calibration),
  `api/` (mkdocstrings reference), `development/` (contributing, changelog).
- **`results/`** — generated simulation outputs (not authoritative source; macroscopic
  JSON/PNG/AVI output; gitignored).
- **`.github/workflows/`** — CI; see `techContext.md` for the full table.
- **`pyproject.toml`** — package metadata, runtime/dev/docs dependencies.
- **`cliff.toml`** — git-cliff changelog generation config; see `.agent_docs/INTEGRATIONS.md`.
- **`.pylintrc`** — minimal config, mostly `cv2` false-positive suppression.
- **`mkdocs.yml`** — MkDocs site config/nav; see `.agent_docs/INTEGRATIONS.md`.
