# Experimental Features

!!! warning "Experimental"

    The components described in this section are **experimental**. APIs may change
    without notice between releases. They are not subject to the same stability
    guarantees as the core traffic flow models.

    These features require **SUMO** to be installed separately. See the
    [Eclipse SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html)
    for instructions.

## What's in this section

| Page                              | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| [Calibration](calibration.md)     | Sliding-window nonlinear least-squares parameter calibration |
| [SUMO Pipeline](sumo-pipeline.md) | Automated microscopic ↔ macroscopic data pipeline via SUMO   |

## Stability guarantees

| Component                                         | Status           |
| ------------------------------------------------- | ---------------- |
| `CTM`, `METANET`, `Simulation`, all controllers   | **Stable**       |
| `Calibration`, `Network Arbitrator`, SUMO tooling | **Experimental** |
