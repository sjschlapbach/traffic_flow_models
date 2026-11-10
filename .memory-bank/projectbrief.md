# Project Brief — traffic_flow_models

> Read-only unless project scope changes. This is the foundational "what is this project" file.

## What this is

`traffic_flow_models` is a modular Python 3.13 library for **macroscopic traffic flow
simulation on highway networks**. It implements two widely-used macroscopic models — the
**Cell Transmission Model (CTM)**, a first-order model, and **METANET**, a second-order model —
on a unified network representation (`Network`, `MotorwayLink`, `Onramp`, `Offramp`, `Origin`,
`Destination`, `Cell`) with consistent, model-agnostic simulation, calibration, and visualization
interfaces. Calibration (experimental) fits METANET parameters to user-provided ground-truth
files in the results-JSON format. Since 2026-11-10 the package no longer ships tooling to
*generate* such data (the former SUMO/OSM pipeline was removed, see ADR-0001).

Published to PyPI as `traffic-flow-models`, MIT licensed. Docs:
`https://sjschlapbach.github.io/traffic_flow_models/`.

## Academic context

This is the companion codebase for:

> J. Schlapbach, K. K. Vuppala Narasimha, A. Kouvelas, M. A. Makridis, *"Macroscopic Traffic Flow
> Modelling on Highway Networks: An Open-Source Computational Framework,"* 26th Swiss Transport
> Research Conference (STRC), Ascona, May 2026 (DOI pending).

Authored at IVT (Institute for Transport Planning and Systems), ETH Zurich. The working directory
path (`.../Doctorate/git/traffic_flow_models`) and author match strongly suggest this is the
repository owner's own doctoral research codebase, with the STRC 2026 paper as a milestone
deliverable. Treat correctness and reproducibility of simulation/calibration results as
high-stakes — this isn't just a library to keep tidy, its output may back published research
claims.

## Stability tiers

Per `docs/experimental/index.md`, components fall into two explicit tiers:

- **Stable**: `CTM`, `METANET`, `Network`, `Simulation`, controllers (`AlineaController`,
  `MetalineController`, `FlowController`, `CustomController`).
- **Experimental — API may change without notice**: `Calibrator`.

## Scope boundary

This is a *macroscopic* modeling library: models, network components, ramp-metering controllers
and calibration. Importing real-world topologies (OSM), running microsimulations (SUMO) and
aggregating detector data into ground truth are **out of scope**. Users supply networks in code
(or via `Network.load_from_json`) and ground-truth data as results-JSON files.

## See also

- `techContext.md` — dependencies, environment, tooling
- `systemPatterns.md` — recurring architectural patterns
- `activeContext.md` — current focus and recent changes
- `progress.md` — what works, what's fragile
