# Macroscopic Traffic Flow Models

A Python library for simulating and analyzing macroscopic traffic flow on highway networks. This package implements two widely-used traffic flow models—**Cell Transmission Model (CTM)** and **METANET**—along with network infrastructure components, ramp metering controllers (ALINEA), and comprehensive visualization tools.

## Features

- **Traffic Flow Models**
  - **CTM (Cell Transmission Model)**: A first-order macroscopic model that updates traffic density based on cell transmission principles
  - **METANET**: A second-order macroscopic model with dynamics for both density and speed
  
- **Network Components**
  - Flexible highway network structure with cells, onramps, and offramps
  - Lane drops and capacity constraints
  - Customizable cell parameters (length, lanes, capacity, speeds, densities)

- **Control Strategies**
  - **ALINEA Controller**: Feedback-based ramp metering algorithm for traffic regulation

- **Visualization & Analysis**
  - Network topology plotting
  - Simulation result visualization (density, flow, speed over time)
  - Demand profile generation

- **Pre-configured Scenarios**
  - Multiple traffic demand scenarios (A, B, C)
  - Example network configurations
  - Demo scripts for quick start

## Installation

### Requirements

- Python 3.13 or later
- pip 25.3 or later

### Setup Instructions

It is recommended to set up a virtual environment for this project:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

### For Development

To install the package with development dependencies (includes pytest):

```bash
pip install -e ".[dev]"
```

### macOS with Pre-installed Python

On macOS or other systems with pre-installed Python, you may need to install Python 3.13 explicitly:

```bash
brew install python@3.13
python3.13 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Environment Variable (if needed)

In some cases, you may need to set the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=.
```

## Project Structure

```
traffic_flow_models/
├── src/
│   ├── traffic_flow_models/     # Main package
│   │   ├── model/               # Traffic flow models (CTM, METANET)
│   │   ├── network/             # Network components (Cell, Network, Onramp, Offramp)
│   │   ├── controller/          # Control strategies (ALINEA)
│   │   └── simulator/           # Simulation utilities
│   └── demo/                    # Demo scripts and scenarios
│       ├── ctm_simulation.py    # CTM simulation example
│       ├── metanet_simulation.py # METANET simulation example
│       ├── scenarios.py         # Pre-configured network scenarios
│       ├── demand.py            # Demand profile utilities
│       └── plot_network.py      # Network visualization example
├── tests/                       # Unit tests
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Usage

### Basic Example: Creating a Network

```python
from traffic_flow_models import Network, Onramp

# Create a new highway network
network = Network()

# Add mainline cells
network.add_cell(
    length=0.5,           # km
    lanes=3,
    lane_capacity=2000,   # vehicles/hour/lane
    free_flow_speed=100,  # km/h
    jam_density=180       # vehicles/km/lane
)

# Add a cell with an onramp
network.add_cell(
    length=0.5,
    lanes=3,
    lane_capacity=2000,
    free_flow_speed=100,
    jam_density=180,
    onramp=Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180
    )
)
```

### Running a CTM Simulation

```python
from traffic_flow_models import CTM

# Initialize the CTM model
ctm = CTM()

# Define demand functions
def mainline_demand(time):
    return 4000  # vehicles/hour

def onramp_demand(time, network_length):
    import numpy as np
    demands = np.zeros(network_length)
    demands[1] = 2000  # vehicles/hour at cell 1
    return demands

# Run simulation
density, flow, speed, input_flow, input_queue, onramp_flow, onramp_queue = \
    network.simulate(
        duration=1.0,              # hours
        dt=10.0/3600,             # time step in hours
        model=ctm,
        mainline_demand=mainline_demand,
        onramp_demand=onramp_demand,
        plot_results=True
    )
```

### Running a METANET Simulation

```python
from traffic_flow_models import METANET

# Initialize METANET with model parameters
metanet = METANET(
    tau=22/3600,    # Relaxation time scale
    nu=15,          # Anticipation coefficient
    kappa=10,       # Density smoothing constant
    delta=1.4,      # Onramp influence weight
    phi=10,         # Lane drop coefficient
    alpha=2         # Velocity function shape parameter
)

# Run simulation (same interface as CTM)
density, flow, speed, input_flow, input_queue, onramp_flow, onramp_queue = \
    network.simulate(
        duration=1.0,
        dt=10.0/3600,
        model=metanet,
        mainline_demand=mainline_demand,
        onramp_demand=onramp_demand,
        plot_results=True
    )
```

### Using ALINEA Ramp Metering

```python
from traffic_flow_models import AlineaController

# Create an onramp with ALINEA controller
onramp_with_control = Onramp(
    lanes=1,
    lane_capacity=2000,
    free_flow_speed=100,
    jam_density=180,
    controller=AlineaController(
        gain=5.0,                    # Controller gain
        setpoint=20.0,               # Target density (veh/km/lane)
        measurement_cell=3           # Cell to measure density
    )
)
```

### Running Demo Simulations

The package includes pre-configured demo scenarios:

```bash
# Run CTM simulation (Scenario A)
python -m src.demo.ctm_simulation

# Run METANET simulation (Scenario A)
python -m src.demo.metanet_simulation

# Visualize a sample network
python -m src.demo.plot_network
```

You can modify the scenario by editing the `scenario` variable in the demo files:
- `"A"`: Base scenario
- `"B"`: Higher onramp demand
- `"C"`: Different network configuration with lane drop

## Testing

The project includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/ctm_test.py
```

### Test Coverage

The test suite covers:
- Cell and network creation
- Onramp and offramp functionality
- CTM and METANET model computations
- ALINEA controller behavior
- Network topology validation

## Development

### Code Formatting

This project uses [Black](https://github.com/psf/black) for code formatting. A GitHub Actions workflow automatically checks formatting on pull requests.

### Continuous Integration

The project includes GitHub Actions workflows for:
- **Testing**: Runs pytest on Python 3.13
- **Formatting**: Checks code style with Black

## Dependencies

- `numpy>=2.3.4`: Numerical computations
- `scipy>=1.16.3`: Scientific computing utilities
- `matplotlib>=3.10.7`: Plotting and visualization

## Contributing

Contributions are welcome! Please ensure that:
1. All tests pass (`pytest`)
2. Code is formatted with Black
3. New features include appropriate tests
4. Documentation is updated as needed

## License

This project was developed by Julius Schlapbach (juliussc@ethz.ch).

## References

- **CTM**: Daganzo, C. F. (1994). The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory. Transportation Research Part B, 28(4), 269-287.
- **METANET**: Papageorgiou, M., Blosseville, J. M., & Hadj-Salem, H. (1990). Modelling and real-time control of traffic flow on the southern part of Boulevard Périphérique in Paris. Transportation Research Part A, 24(5), 345-359.
- **ALINEA**: Papageorgiou, M., Hadj-Salem, H., & Blosseville, J. M. (1991). ALINEA: A local feedback control law for on-ramp metering. Transportation Research Record, 1320, 58-64.
