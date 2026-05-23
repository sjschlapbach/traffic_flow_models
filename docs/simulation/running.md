# Running Simulations

## Basic usage

A `Simulation` binds a `Network` to a model:

```python
from traffic_flow_models import Simulation, CTM

sim = Simulation(
    network=net,
    model=CTM(),
    # model_params=model_params  # required for METANET, must be None for CTM
)
```

Call `run()` to execute the time-stepping loop. All four demand/BC dicts are required
and must map **string IDs** to **callables** `(t: float) -> float` (or, for turning
rates, `(t: float) -> dict[str, float]`):

```python
time_arr, states, disturbances = sim.run(
    duration=1.0,                       # total simulation time [hours]
    dt=10 / 3600,                       # time step [hours] (10 seconds)
    origin_demands={
        "o_main": lambda t: 4200.0,     # [veh/h]
        "o_ramp": lambda t: 600.0,
    },
    turning_rates={
        "n_entry": lambda t: {"m1": 1.0},
        "n_merge": lambda t: {"m2": 1.0},
        "n_split": lambda t: {"m3": 0.8, "f1": 0.2},
        "n_exit":  lambda t: {"d_main": 1.0},
        "n_ramp":  lambda t: {"r1": 1.0},
        "n_off":   lambda t: {"d_off": 1.0},
    },
    destination_flow_bc={
        "d_main": lambda t: 7000.0,     # unconstrained exit
        "d_off":  lambda t: 2000.0,
    },
    destination_density_bc={
        "d_main": lambda t: 0.0,
        "d_off":  lambda t: 0.0,
    },
    preferred_cell_size=0.5,            # target cell length [km]; default 0.5
    plot_results=False,                 # set True to auto-save plots + JSON
    results_dir="results/my_scenario",  # used when plot_results=True
)
```

`run()` returns a tuple `(time_array, state_history, disturbance_history)`:

- `time_array` — 1-D NumPy array of time points [hours], length = steps + 1
- `state_history` — 2-D NumPy array, shape `(state_size, steps + 1)`
- `disturbance_history` — 2-D NumPy array, shape `(disturbance_size, steps)`

The time step must satisfy the CFL condition for all links. An error is raised if it
is violated.

## Saving and loading results

Call `save_results()` after `run()` to persist results to a JSON file:

```python
sim.save_results("results/my_scenario/results.json")
```

Load with the class method:

```python
time_arr, states, disturbances, metadata = Simulation.load_results(
    filepath="results/my_scenario/results.json",
    network=net,
)
```

## Setting initial conditions

By default the network starts empty (zero density, free-flow speed). Override per link:

```python
time_arr, states, disturbances = sim.run(
    ...,
    initial_densities={"m1": 20.0, "m2": 50.0},    # scalar applies to all cells
    initial_speeds={"m1": 95.0},                   # free-flow speed used elsewhere
    initial_origin_queues={"o_main": 100.0},
)
```
