# System Patterns — traffic_flow_models

> Update when architecture changes. This is the "how the pieces fit together" reference —
> read this before making non-trivial changes to model/network internals.

## Unified state-space abstraction

Both `CTM` and `METANET` implement the same discrete-time update contract
`x(k+1) = f(x(k), u(k), d(k), θ)`. Concretely, each model's `network_update_function(...)`
(in `model/ctm.py` / `model/metanet.py`) builds a symbolic **CasADi** computation graph once per
network topology and timestep, compiled into `casadi.Function("<model>_network_step",
[params, x, d], [x_next])`. `Simulation._run_simulation_loop` (`network/simulation.py`) just
evaluates this compiled function numerically at each timestep — it contains no model physics
itself.

**Implication for future changes**: changing model equations (CTM demand-supply rule, METANET
speed ODE, etc.) means editing the symbolic graph-construction code in `ctm.py`/`metanet.py`, not
a numeric Python loop. `casadi.fmin`/`fmax` are used throughout instead of Python's `min`/`max`
because everything must stay differentiable/symbolic until the graph is compiled.

## Centralized, order-dependent state packing — read this before touching model internals

`Network` (`network/network.py`) is the single source of truth for how the flat state vector `x`
and disturbance vector `d` are laid out:

- `network_dict_to_state_vec(flow_dict, density_dict, speed_dict, origin_queue_dict,
  onramp_queue_dict, offramp_queue_dict) -> (x, num_flows, num_densities, ...)`
- `state_vec_to_network_dict(x) -> (flows, densities, speeds, origin_queues, onramp_queues,
  offramp_queues)`
- `network_dict_to_disturbance_vec(...)` / `disturbance_vec_to_network_dict(d) -> (...)`

Packing order is driven by `network.list_nodes()` iteration order and **must stay stable across
calls** — both `CTM`/`METANET`'s `network_update_function`s and `Simulation`/`Calibrator` all rely
on this same ordering contract to unpack `x`/`d` correctly inside the CasADi graph. This is the
single riskiest invariant in the codebase to accidentally break (e.g. by changing `Network`
iteration/sorting logic without updating the packing functions in lockstep).

## Store-and-forward virtual links vs. spatially-discretized links

`Origin`, `Onramp`, `Offramp` are all single-scalar "virtual queue" links modeled via
store-and-forward dynamics (`model/helpers.py::store_and_forward_update`). Only `MotorwayLink` is
spatially discretized into a chain of `Cell` objects. `Cell` itself carries no fundamental-diagram
parameters — those live on the parent `MotorwayLink`/`Onramp`/`Offramp`, since the FD shape
(triangular for CTM, exponential for METANET) is model-dependent, not link-dependent.

## CFL-driven cell discretization

`MotorwayLink.partition_link(preferred_cell_size, dt, upcoming_lane_drop=0)` enforces the CFL
condition `min_cell_length = free_flow_speed * dt`. Any change touching `dt` handling, cell
partitioning, or `free_flow_speed` must preserve this invariant, or simulations become
numerically unstable/invalid. `Simulation.run` calls this automatically before starting a run.

## Duck-typed controller interface

There is no controller ABC. Any object exposing
`compute_regulated_flow(onramp_queues: dict[str, SX], flows: dict[str, SX], densities: dict[str,
SX]) -> casadi.SX` can be assigned to `Onramp.controller` (constructor arg or later attribute
assignment). Both model update functions call `onramp.controller.compute_regulated_flow(...)`
internally when present, then clamp with `casadi.fmin(Qc, r_k, ...)`.
`controller/custom_controller.py::CustomController` is the sanctioned extension point for
arbitrary user-defined metering logic (wraps any callable, auto-detects whether it accepts a
`params` argument via `inspect.signature`).

## Model-agnostic calibration interface

`Calibrator` (`network/calibrator.py`) never touches model internals directly. It delegates all
model-specific packing/bounds/param-name logic to methods each model class must implement:
`get_default_calibration_params`, `get_calibration_bounds`, `prepare_calibration_params`,
`parse_calibration_params`, `prepare_system_params`, `get_calibration_param_names`.

- `CTM` implements all six as `NotImplementedError` **by design** — its fundamental diagram is
  fully determined by link geometry/capacity, so there is nothing to calibrate.
- `METANET` implements the full interface over `METANETParams` (`tau, nu, kappa, delta, phi,
  alpha`), where `alpha` may be a global scalar or per-link (`dict[link_id, float]`).
- The `model_options` dict (e.g. `{"link_specific_alpha": bool}`) is the established pattern for
  passing model-specific calibration knobs through `Calibrator` without leaking them into the
  generic `Network`/`Calibrator` API surface. Follow this pattern for any new calibratable model.

## SUMO ↔ macroscopic bridge pipeline (`arbitrator/` subpackage)

One-directional pipeline turning a raw SUMO network + microsimulation output into the inputs
`Simulation.run`/`Calibrator.calibrate_model_params` need:

```
NetworkArbitrator            SUMO .net.xml → macroscopic Network object
   (merges serial edges into synthetic "merged_<A>_<B>" link IDs)
        ↓
LoopDetectorGenerator        places SUMO detectors at macro/micro interface + backbone points
        ↓
DemandAggregator / TurningRateAggregator / BackboneStateAggregator
   (SUMO detector output → origin_demands / turning_rates / ground-truth state callables)
```

**Critical gotcha**: `NetworkArbitrator` produces `sumo_edge_id_map`, the mandatory bridge from
synthetic merged macroscopic link IDs back to real SUMO edge IDs. Anything touching detector
placement or aggregation (`LoopDetectorGenerator`, the three aggregators) must use this map —
several docstrings in the codebase explicitly flag this as an easy correctness pitfall.

## Known incomplete scaffolding — not a bug

`Onramp.control_status` (`Literal["unset", "hero_master", "hero_slave"]`) and
`Network.upstream_onramps` / `downstream_onramps` (populated by
`Network.update_all_onramp_neighbors()`) exist but are **not consumed by any controller yet**.
This is preparation for a coordinated ramp-metering scheme (HERO — see the `hero-ramp-metering`
branch in `activeContext.md`), not finished functionality. Don't assume it's wired up; don't
"clean it up" as dead code without checking that branch first.

## Known cleanup candidate

`arbitrator/network_arbitrator.py`, inside `instantiate_network` (roughly lines 884–921), contains
a block of dead/commented-out code duplicating onramp/offramp classification logic that was
re-implemented, uncommented, just below it — looks like an abandoned refactor. Minor cleanup
candidate, not a functional bug.

## See also

- `projectbrief.md` — what this project is
- `techContext.md` — dependencies and CI
- `.agent_docs/STRUCTURE.md` — per-file class/function catalog
- `.agent_docs/INTEGRATIONS.md` — SUMO/OSM integration detail
