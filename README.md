# Macroscopic Traffic Flow Models

A Python library for simulating and analyzing macroscopic traffic flow on highway networks. This package implements two widely-used traffic flow models—**Cell Transmission Model (CTM)** and **METANET**—along with network infrastructure components, ramp metering controllers (ALINEA), and visualization tools.

## Features

- **Traffic Flow Models**: CTM (first-order) and METANET (second-order) macroscopic models
- **Network Components**: Flexible highway network structure with cells, onramps, and offramps
- **Control Strategies**: ALINEA ramp metering controller
- **Visualization**: Network topology plotting and simulation result visualization

## Installation

**Requirements**: Python 3.13 or later

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"
```

On macOS, you may need to install Python 3.13 explicitly:

```bash
brew install python@3.13
python3.13 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
traffic_flow_models/
├── src/
│   ├── traffic_flow_models/     # Main package
│   │   ├── model/               # Traffic flow models (CTM, METANET)
│   │   ├── network/             # Network components
│   │   └── controller/          # Control strategies (ALINEA)
│   └── demo/                    # Demo scripts
└── tests/                       # Unit tests
```

## Quick Start

Run a pre-configured demo simulation:

```bash
python -m src.demo.ctm_simulation
python -m src.demo.metanet_simulation
```

## Testing

```bash
pytest
```

## Development

Contributions are welcome! Please ensure that:

1. All tests pass (`pytest`)
2. Code is formatted with [Black](https://github.com/psf/black)
3. New features include appropriate tests

## Usage Examples

### Creating a Network

```python
from traffic_flow_models import Network, Onramp

network = Network()
network.add_cell(length=0.5, lanes=3, lane_capacity=2000,
                 free_flow_speed=100, jam_density=180)
network.add_cell(length=0.5, lanes=3, lane_capacity=2000,
                 free_flow_speed=100, jam_density=180,
                 onramp=Onramp(lanes=1, lane_capacity=2000,
                              free_flow_speed=100, jam_density=180))
```

### Running Simulations

```python
from traffic_flow_models import CTM, METANET
import numpy as np

# CTM simulation
ctm = CTM()
density, flow, speed, *_ = network.simulate(
    duration=1.0, dt=10.0/3600, model=ctm,
    mainline_demand=lambda t: 4000,
    onramp_demand=lambda t, n: np.array([0, 2000] + [0]*(n-2)),
    plot_results=True
)

# METANET simulation
metanet = METANET(tau=22/3600, nu=15, kappa=10, delta=1.4, phi=10, alpha=2)
density, flow, speed, *_ = network.simulate(
    duration=1.0, dt=10.0/3600, model=metanet,
    mainline_demand=lambda t: 4000,
    onramp_demand=lambda t, n: np.array([0, 2000] + [0]*(n-2)),
    plot_results=True
)
```

### ALINEA Ramp Metering

```python
from traffic_flow_models import AlineaController

onramp = Onramp(lanes=1, lane_capacity=2000, free_flow_speed=100,
                jam_density=180,
                controller=AlineaController(gain=5.0, setpoint=20.0,
                                           measurement_cell=3))
```

## License

TBD

<!-- TODO: add license here before open-sourcing -->

## References

- **CTM**: Daganzo, C. F. (1994). The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory. Transportation Research Part B, 28(4), 269-287.
- **METANET**: Messmer, A., & Papageorgiou, M. (1990). METANET: A macroscopic simulation program for motorway networks. Traffic Engineering & Control.
- **ALINEA**: Papageorgiou, M., Hadj-Salem, H., & Blosseville, J. M. (1991). ALINEA: A local feedback control law for on-ramp metering. Transportation Research Record, 1320, 58-64.
