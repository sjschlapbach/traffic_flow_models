# traffic-flow-models

A modular Python library for macroscopic traffic flow simulation on highway networks.
It implements both the **Cell Transmission Model (CTM)** and the **METANET** model on a
unified network representation, with consistent interfaces for demand definition, ramp
metering control, and result visualisation.

The library can be installed from [PyPI](https://pypi.org/project/traffic-flow-models/) through the following command, while the source code is available on [GitHub](https://github.com/sjschlapbach/traffic_flow_models):

```
pip install traffic-flow-models
```

---

## Why macroscopic models?

Microscopic simulators track individual vehicles and capture detailed dynamics, but their
computational cost scales with vehicle count, making them impractical for large networks,
long time horizons, or real-time optimisation. Macroscopic models aggregate traffic into
three continuum variables — **flow** \(q\) (veh/h), **density** \(\rho\) (veh/km/lane),
and **mean speed** \(v\) (km/h) — and operate orders of magnitude faster while remaining
mostly compatible with loop-detector measurements.

---

## What this library provides

**Traffic flow models**

- **CTM** — First-order model based on the LWR theory and a triangular fundamental diagram.
- **METANET** — Second-order model with an explicit speed dynamics equation, capturing
  driver anticipation, speed relaxation, and merging effects.

Both models operate on an identical network representation and use consistent input interfaces, making
their outputs directly comparable under the same conditions.

**Simulation infrastructure**

- `Network`, `Node`, `MotorwayLink`, `Onramp`, `Offramp`, `Origin`, `Destination`
- `Simulation` — time-stepping, metrics computation, result save/load, visualisation

**Ramp metering control**

- `AlineaController` — local I-type feedback (ALINEA)
- `MetalineController` — coordinated multi-ramp feedback (METALINE)
- `FlowController` — fixed-rate metering
- `CustomController` — base class for user-defined strategies

**Experimental components**

- `Calibrator` — sliding-window least-squares parameter estimation
- `SUMOPipeline` _(requires SUMO installation)_ — end-to-end micro-to-macro benchmarking pipeline

---

## Quickstart

```python
from traffic_flow_models import (
    Network, Node, MotorwayLink, Origin, Destination,
    CTM, Simulation,
)

# 1. Build a minimal single-link network
origin = Origin(id="o_main")
dest   = Destination(id="d_main")
link   = MotorwayLink(id="m1", length=4.0, lanes=2,
                      lane_capacity=2000, free_flow_speed=100, jam_density=180)
n_in   = Node(id="n_in",  incoming=[origin], outgoing=[link])
n_out  = Node(id="n_out", incoming=[link],   outgoing=[dest])
net    = Network(nodes=[n_in, n_out])

# 2. Run a CTM simulation (1-hour, 10-second steps)
sim = Simulation(network=net, model=CTM())
time_arr, states, disturbances = sim.run(
    duration=1.0,
    dt=10 / 3600,
    origin_demands={"o_main": lambda t: 1500.0},
    turning_rates={
        "n_in":  lambda t: {"m1":     1.0},
        "n_out": lambda t: {"d_main": 1.0},
    },
    destination_flow_bc={"d_main": lambda t: 2000.0},
    destination_density_bc={"d_main": lambda t: 0.0},
    plot_results=False,
)
```

See the [Quickstart guide](getting-started/quickstart.md) for a complete worked example.

---

## Citation

If you use this library in your research, please cite:

> J. Schlapbach, K. K. Vuppala Narasimha, A. Kouvelas, M. A. Makridis,
> _"Macroscopic Traffic Flow Modelling on Highway Networks:
> An Open-Source Computational Framework"_,
> 26th Swiss Transport Research Conference (STRC), Ascona, May 2026.
> DOI: `[to be added upon publication]`

---

## Licence

[MIT](https://github.com/sjschlapbach/traffic_flow_models/blob/master/LICENSE) ·
Authors: Julius Schlapbach, Krishna Kanth Vuppala Narasimha, Anastasios Kouvelas,
Michail A. Makridis (IVT, ETH Zurich)
