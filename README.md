# Macroscopic Traffic Flow Models

[![PyPI version](https://img.shields.io/pypi/v/traffic-flow-models)](https://pypi.org/project/traffic-flow-models/)
[![Python Testing](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/python_testing.yml/badge.svg)](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/python_testing.yml)
[![Package Build](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/build.yml/badge.svg)](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/build.yml)
[![Pipeline](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/scripts_sumo_pipeline.yml/badge.svg)](https://github.com/sjschlapbach/traffic_flow_models/actions/workflows/scripts_sumo_pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python library for simulating and analyzing macroscopic traffic flow on highway networks. This package implements two widely-used traffic flow models—**Cell Transmission Model (CTM)** and **METANET**—along with network infrastructure components, ramp metering controllers (ALINEA), and visualization tools.

## Features

- **Traffic Flow Models**: CTM (first-order) and METANET (second-order) macroscopic models
- **Network Components**: Flexible highway network structure with motorway links, nodes, onramps, offramps, origins, and destinations
- **Control Strategies**: Generic control interfaces for local and coordinated ramp metering strategies, including implementations for ALINEA, METALINE, and fixed-rate metering
- **Performance Metrics**: Extensive set of network performance metrics (VKT, VHT, average speed, density, flow, etc.) computed from simulation results
- **Visualization**: Network topology plotting, simulation result visualization, and video export
- **SUMO Integration**: Pipeline components for importing and benchmarking real-world highway networks
- **Calibration & Parameter Estimation**: Tools for calibrating macroscopic model parameters from aggregated observation data. Includes regularized least-squares calibration, multi-start parameter search, parameter-correlation analysis, and plotting utilities.

## Installation

The latest [stable version of the package](https://pypi.org/project/traffic-flow-models/) can be easily installed through pip. For more information on how to run the code in this repository, including the demo scripts, please refer to the [development section](#development) below.

```bash
pip install traffic-flow-models
```

If you plan to use the pipeline components involving SUMO (e.g. for benchmarking highway networks of a specific city), please refer to the [SUMO installation guide](https://sumo.dlr.de/docs/Installing/index.html) for instructions on how to install SUMO on your system (not included in the package).

For the pipeline to be fully functional, auxiliary command line commands such as `netconvert` and `sumo` need to be accessible from your system PATH. For installations on macOS the SUMO installer might not set the SUMO_HOME environment variable automatically / correctly. We recommend checking manually that the variable is set to your library installation (something like `/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo`).

## Usage Examples

### Creating a Network

Networks are built using nodes that connect motorway links, onramps, offramps, origins, and destinations. Onramps should be fed by their own origin via an upstream node, and every offramp should lead into a downstream node whose single outgoing link is a destination:

```python
from traffic_flow_models import (
    Network, Node, MotorwayLink, Onramp, Offramp, Origin, Destination
)

# Create entry/exit points
o_main = Origin(id="o_main")
o_ramp = Origin(id="o_ramp")
d_main = Destination(id="d_main")
d_off = Destination(id="d_off")

# Define motorway and ramp links
m1 = MotorwayLink(
    id="m1", length=1.2, lanes=3, lane_capacity=2000,
    free_flow_speed=100, jam_density=180
)
m2 = MotorwayLink(
    id="m2", length=1.0, lanes=3, lane_capacity=2000,
    free_flow_speed=100, jam_density=180
)
m3 = MotorwayLink(
    id="m3", length=1.5, lanes=3, lane_capacity=2000,
    free_flow_speed=100, jam_density=180
)
r1 = Onramp(
    id="r1", lanes=1, lane_capacity=1800,
    free_flow_speed=90, jam_density=170
)
f1 = Offramp(
    id="f1", lanes=1, lane_capacity=1500,
    free_flow_speed=80, jam_density=160
)

# Connect components using nodes (with positions for visualization)
n_entry = Node(id="n_entry", incoming=[o_main], outgoing=[m1])
n_entry.position = (0.0, 0.0)

n_ramp = Node(id="n_ramp", incoming=[o_ramp], outgoing=[r1])
n_ramp.position = (0.4, -0.1)

n_merge = Node(id="n_merge", incoming=[m1, r1], outgoing=[m2])
n_merge.position = (1.0, 0.0)

n_split = Node(id="n_split", incoming=[m2], outgoing=[m3, f1])
n_split.position = (2.0, 0.0)

n_off = Node(id="n_off", incoming=[f1], outgoing=[d_off])
n_off.position = (2.3, -0.2)

n_exit = Node(id="n_exit", incoming=[m3], outgoing=[d_main])
n_exit.position = (3.0, 0.0)

# Build the network (place a junction first so the connectivity check sees every branch)
network = Network(
    nodes=[n_merge, n_entry, n_ramp, n_split, n_off, n_exit]
)
```

### Running Simulations

Simulations use dictionaries of time-dependent demand functions. Ramp inflows are modeled as additional origins that feed their onramp through a node:

```python
from traffic_flow_models import CTM, METANET, METANETParams, Simulation
from typing import Callable

# Define demand functions
def mainline_demand(t: float) -> float:
    return 4200.0 if t < 0.8 else 3200.0

def ramp_demand(t: float) -> float:
    return 1800.0 if 0.2 < t < 0.9 else 400.0

# Map demands to network components (origins only)
origin_demands: dict[str, Callable[[float], float]] = {
    "o_main": mainline_demand,
    "o_ramp": ramp_demand,
}
destination_flow_bc: dict[str, Callable[[float], float]] = {
    "d_main": lambda t: 7000.0,
    "d_off": lambda t: 2000.0,
}
destination_density_bc: dict[str, Callable[[float], float]] = {
    "d_main": lambda t: 0.0,
    "d_off": lambda t: 0.0,
}
turning_rates: dict[str, Callable[[float], dict[str, float]]] = {
    "n_entry": lambda t: {"m1": 1.0},
    "n_ramp": lambda t: {"r1": 1.0},
    "n_merge": lambda t: {"m2": 1.0},
    "n_split": lambda t: {"m3": 0.8, "f1": 0.2},
    "n_off": lambda t: {"d_off": 1.0},
    "n_exit": lambda t: {"d_main": 1.0},
}

# CTM simulation
ctm = CTM()
sim = Simulation(network, ctm)
time, states, disturbances = sim.run(
    duration=1.0,
    dt=10.0/3600,
    preferred_cell_size=0.5,
    origin_demands=origin_demands,
    turning_rates=turning_rates,
    destination_flow_bc=destination_flow_bc,
    destination_density_bc=destination_density_bc,
    plot_results=True,
    results_dir="results/ctm_run",
)

# METANET simulation with parameters
metanet = METANET()
model_params: METANETParams = {
    "tau": 22/3600,
    "nu": 15,
    "kappa": 10,
    "delta": 1.4,
    "phi": 10,
    "alpha": 2,
}

sim = Simulation(network, metanet, model_params)
time, states, disturbances = sim.run(
    duration=1.0,
    dt=10.0/3600,
    preferred_cell_size=0.5,
    origin_demands=origin_demands,
    turning_rates=turning_rates,
    destination_flow_bc=destination_flow_bc,
    destination_density_bc=destination_density_bc,
    plot_results=True,
    results_dir="results/metanet_run",
)
```

### Model Calibration

```python
from traffic_flow_models import Calibrator, METANET, Simulation

# Calibrate METANET using aggregated microscopic simulation results
calibrator = Calibrator(network)
metanet = METANET()
calibrated_params, opt_result, _ = calibrator.calibrate_model_params(
    ground_truth_filepath="results/micro_results.json",
    model=metanet,
    window_size=30,
    stride=15,
    regularization_weight=0.01,
    use_disturbance_from_file=False,
    origin_demands_fn=origin_demands,
    turning_rates_fn=turning_rates,
)

# Run macroscopic simulation with calibrated parameters
sim = Simulation(network=network, model=metanet, model_params=calibrated_params)
time, states, disturbances = sim.run(
    duration=1.0, dt=10.0/3600, preferred_cell_size=0.5,
    origin_demands=origin_demands, turning_rates=turning_rates,
    destination_flow_bc=destination_flow_bc, destination_density_bc=destination_density_bc,
)
```

### Performance Metrics and Visualization

```python
# Compute network performance metrics
VKT, VHT, avg_speed = sim.compute_metrics(
    states=states,
    dt=10.0/3600,
    timesteps=len(time)
)
print(f"Total VKT: {VKT:.2f} veh-km")
print(f"Total VHT: {VHT:.2f} veh-h")
print(f"Average Speed: {avg_speed:.2f} km/h")

# Generate video visualization
sim.visualize(
    results_filepath="results/simulation_results.json",
    output_filepath="results/simulation.avi",
    fps=30,
    subsampling=1
)
```

### Ramp Metering

Ramp metering strategies regulate on-ramp inflow to improve mainline traffic performance. The package includes multiple ramp metering controllers (for example, ALINEA), as well as an interface for the definition of custom controllers. The code below is provided as an example demonstrating how to attach an ALINEA controller to an on-ramp and measure density on a downstream motorway link cell.

```python
from traffic_flow_models import Onramp, AlineaController

onramp = Onramp(
    id="onramp",
    lanes=1,
    lane_capacity=2000,
    free_flow_speed=100,
    jam_density=180,
    controller=None,
)

# attach an ALINEA controller that measures density on downstream link 'm2' cell 0
onramp.controller = AlineaController(
    onramp=onramp,
    measurement_link_id="m2",
    measurement_cell_idx=0,
    gain=5.0,
    density_setpoint=30.0,
)
```

Other controllers are available in the `controller` directory — for example, [METALINE](src/traffic_flow_models/controller/metaline.py). See [src/traffic_flow_models/controller](src/traffic_flow_models/controller) for more controllers and the structure of custom controllers.

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
│   │   ├── network/             # Network components (links, nodes, cells)
│   │   ├── controller/          # Control strategies (ALINEA)
│   │   ├── simulator/           # SUMO simulation and pipeline
│   │   └── arbitrator/          # Demand aggregation and loop detectors
│   └── demo/                    # Demo scripts and scenarios
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
git push origin master
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

This project is licensed under an MIT License. See the `LICENSE` file in the repository for the full license text.

## References

- **CTM**: Daganzo, C. F. (1994). The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory. Transportation Research Part B, 28(4), 269-287.
- **METANET**: Messmer, A., & Papageorgiou, M. (1990). METANET: A macroscopic simulation program for motorway networks. Traffic Engineering & Control.
- **ALINEA**: Papageorgiou, M., Hadj-Salem, H., & Blosseville, J. M. (1991). ALINEA: A local feedback control law for on-ramp metering. Transportation Research Record, 1320, 58-64.
