# Macroscopic Traffic Flow Models

[![Python Testing](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/python_testing.yml/badge.svg)](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/python_testing.yml)
[![Package Build](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/build.yml/badge.svg)](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/build.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

A Python library for simulating and analyzing macroscopic traffic flow on highway networks. This package implements two widely-used traffic flow models—**Cell Transmission Model (CTM)** and **METANET**—along with network infrastructure components, ramp metering controllers (ALINEA), and visualization tools.

## Features

- **Traffic Flow Models**: CTM (first-order) and METANET (second-order) macroscopic models
- **Network Components**: Flexible highway network structure with cells, onramps, and offramps
- **Control Strategies**: ALINEA ramp metering controller
- **Visualization**: Network topology plotting and simulation result visualization

## Installation

The latest stable version of the package can be easily installed through pip. For more information on how to run the code in this repository, including the demo scripts, please refer to the [development section](#development) below.

```bash
pip install traffic-flow-models
```

## Usage Examples

### Creating a Network

```python
from traffic-flow-models import Network, Onramp

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
from traffic-flow-models import CTM, METANET
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
from traffic-flow-models import AlineaController

onramp = Onramp(lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180,
               controller=AlineaController(gain=5.0, setpoint=20.0,measurement_cell=3))
```

## Development

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

Quickstart: Run one of the demo scripts to see the package in action:

```bash
python -m src.demo.ctm_simulation
python -m src.demo.metanet_simulation
```

Contributions are welcome! Before opening a pull request, please ensure that:

1. All tests pass (`pytest`)
2. Code is formatted with [Black](https://github.com/psf/black)
3. New features include appropriate tests

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

## Testing

All tests in this project are written using [pytest](https://docs.pytest.org/en/stable/). To run the tests, execute the following command in the project root directory:

```bash
pytest
```

## Release

To release a new version of the macroscopic traffic flow package, please make sure that all tests and builds are passing and follow these steps:

#### 0. Prerequisites (important!)

Locally change to the `master` branch and make sure that it contains all the latest changes that should be part of the release

```bash
git checkout master
git pull origin master
```

Also, make sure you have the necessary permissions to push tags to the repository.

#### 1. Update version in pyproject.toml

Update the version of the package in the pyproject file according to the next release version according to the conventional commit guidelines. To make sure you update the version correctly, you can run the following command:

```bash
git-cliff --bump
```

⚠️ **CAUTION:** Do not commit any changelog updates, as these will be handled automatically by the release workflow. Only commit the version change in `pyproject.toml`. If the version in the pyproject configuration and the changelog are not in sync, the release workflow will fail.

#### 2. Commit the version change

```bash
git checkout master                     # safeguard to ensure you are on the master branch
git commit -m "chore(release): v1.0.0"  # replace v1.0.0 with the current version
```

#### 3. Create and push tag (triggers workflow)

```bash
git checkout master                                     # safeguard to ensure you are on the master branch
git tag -a v1.0.0 -m "chore(release): version 1.0.0"    # replace v1.0.0 with the current version
git push origin v1.0.0                                  # push the new tag to GitHub to trigger the release workflow
```

#### 4. GitHub Actions automatically:

- Validates all versions match
- Generates changelog
- Builds distribution packages
- Publishes to PyPI
- Creates GitHub release

## License

This project is licensed under the GNU Affero General Public License version 3 (AGPL-3.0). See the `LICENSE` file in the repository for the full license text.

## References

- **CTM**: Daganzo, C. F. (1994). The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory. Transportation Research Part B, 28(4), 269-287.
- **METANET**: Messmer, A., & Papageorgiou, M. (1990). METANET: A macroscopic simulation program for motorway networks. Traffic Engineering & Control.
- **ALINEA**: Papageorgiou, M., Hadj-Salem, H., & Blosseville, J. M. (1991). ALINEA: A local feedback control law for on-ramp metering. Transportation Research Record, 1320, 58-64.
