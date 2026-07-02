# Project Brief — traffic_flow_models

> Read-only unless project scope changes. This is the foundational "what is this project" file.

## What this is

`traffic_flow_models` is a modular Python 3.13 library for **macroscopic traffic flow
simulation on highway networks**. It implements two widely-used macroscopic models — the
**Cell Transmission Model (CTM)**, a first-order model, and **METANET**, a second-order model —
on a unified network representation (`Network`, `MotorwayLink`, `Onramp`, `Offramp`, `Origin`,
`Destination`, `Cell`) with consistent, model-agnostic simulation, calibration, and visualization
interfaces. It also includes an experimental SUMO microsimulation integration used to derive
real-world network topologies, demand, and "ground truth" states for calibrating the macroscopic
models.

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
- **Experimental — API may change without notice**: `Calibrator`, `NetworkArbitrator`, and the
  SUMO pipeline tooling (`SUMOPipeline`, `SUMOSimulation`, and the rest of the `arbitrator/`
  subpackage).

## Scope boundary

This is a *macroscopic* modeling library. The SUMO microsimulation integration exists solely to
(a) import real-world highway topologies and demand, and (b) generate ground-truth aggregated
states to calibrate macroscopic model parameters against — it is not intended to be a general
microsimulation platform in its own right, and its API is explicitly experimental/volatile
compared to the core CTM/METANET/Network/Simulation stack.

## See also

- `techContext.md` — dependencies, environment, tooling
- `systemPatterns.md` — recurring architectural patterns
- `activeContext.md` — current focus and recent changes
- `progress.md` — what works, what's fragile
