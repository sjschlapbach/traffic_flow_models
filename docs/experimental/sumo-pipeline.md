# SUMO Pipeline

!!! warning "Experimental"

    The SUMO pipeline is experimental and requires SUMO (≥ 1.19) to be installed
    separately. See the [SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html).

The SUMO pipeline automates the generation of macroscopic input data from
**microscopic SUMO simulations**, and vice versa. This enables:

- Generating synthetic detector datasets for calibration.
- Validating macroscopic model predictions against microscopic ground truth.

---

## Pipeline steps

```
Scenario definition
        │
        ▼
  Generate SUMO network + routes (sumolib / netconvert)
        │
        ▼
  Place virtual detectors on all links (E1 inductive loops and E2 area detectors)
        │
        ▼
  Run SUMO (CLI is sufficient)
        │
        ▼
  Abstract network topology → traffic_flow_models Network
        │
        ▼
  Aggregate detector outputs to macroscopic time step
        │
        ▼
  Run macroscopic calibration and simulation
```

---

## Input / output summary

| Step                | Input                            | Output                        |
| ------------------- | -------------------------------- | ----------------------------- |
| Network abstraction | SUMO XML files                   | `Network` object              |
| Detector placement  | `Network`, link geometry         | E1/E2 detector XML            |
| SUMO run            | `.sumocfg`                       | Raw detector CSV              |
| Aggregation         | Detector CSV, aggregation window | Density / flow / speed arrays |
| Macroscopic run     | Arrays above                     | Simulation results            |
