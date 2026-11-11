# Project Overview

**traffic_flow_models** is a modular Python 3.13 library for **macroscopic traffic flow
simulation on highway networks**. It implements two macroscopic models on one network
representation, with model-agnostic simulation, calibration and visualization interfaces:

- **Cell Transmission Model (CTM)**, first-order;
- **METANET**, second-order.

Network components: `Network`, `Node`, `MotorwayLink` (split into `Cell`s), `Onramp`, `Offramp`,
`Origin`, `Destination`. Ramp-metering controllers: `AlineaController`, `MetalineController`,
`FlowController` (fixed rate), `CustomController` (wraps a user callable). Also included:
performance metrics (VKT, VHT, average speed), plots, video export, and an experimental
`Calibrator` that fits METANET parameters to user-provided ground-truth files in the
results-JSON format (see `docs/experimental/calibration.md`). Code layout:
`mem:codebase_structure`.

Published to PyPI as `traffic-flow-models` (version in `pyproject.toml`), MIT licensed. User docs
(MkDocs, GitHub Pages): `https://sjschlapbach.github.io/traffic_flow_models/`.

## Academic context

Companion codebase for J. Schlapbach, K. K. Vuppala Narasimha, A. Kouvelas, M. A. Makridis,
*"Macroscopic Traffic Flow Modelling on Highway Networks: An Open-Source Computational
Framework,"* 26th Swiss Transport Research Conference (STRC), Ascona, May 2026. Written at IVT, ETH
Zurich, as part of the maintainer's doctoral research. Published research claims may rest on the
simulation and calibration output, so correctness and reproducibility matter a lot.

## Scope boundary

The package covers macroscopic models, network components, ramp-metering controllers and
calibration. **Out of scope:** importing real topologies (OSM), running microsimulations (SUMO)
and aggregating detector data into ground truth. Users build networks in code (or with
`Network.load_from_json`) and supply ground truth as results-JSON files.

## Stability tiers (`docs/experimental/index.md`)

- **Stable**: `CTM`, `METANET`, `Simulation`, all controllers (and the network components they
  build on). The test suite covers all of them.
- **Experimental (API may change without notice)**: `Calibrator`.

## Fragile areas: read before editing

- **METANET node-processing order and causality** (`model/metanet.py`): this area has the most
  bugs on record (merge/diverge virtual upstream speed, downstream density). It is numerically
  stable at present. Read the state-packing contract in `mem:codebase_structure` and the
  `metanet.py` docstrings before you change node processing.
- **CTM is not calibratable**: its calibration-interface methods (`model/ctm.py`) raise
  `NotImplementedError`, pinned by `tests/ctm_test.py`. Only METANET supports calibration.
- **CFL condition**: see `MotorwayLink.partition_link` in `mem:codebase_structure`.
- **Public positional argument order**: constructors are public API since v1.0. For example,
  `Onramp.__init__(..., id, controller, origin_node_id, destination_node_id)` is pinned by
  `test_init_keeps_released_positional_order`. Never reorder positional parameters.

## Scaffolding, not dead code

`Onramp.control_status` (`"unset" | "hero_master" | "hero_slave"`) and
`Onramp.upstream_onramps` / `downstream_onramps` (filled by
`Network.update_all_onramp_neighbors()`) prepare coordinated HERO ramp metering (branch
`hero-ramp-metering`). No controller uses them yet. Do not remove them as dead code.

## Branches and releases

Pull requests target `dev` (`docs/development/contributing.md`). CI runs on pushes and PRs to
`master` and `dev`. Releases are tag-triggered, and the release workflow pushes the generated
`CHANGELOG.md` to `master` (`mem:suggested_commands`).
