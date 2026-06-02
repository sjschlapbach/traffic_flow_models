# Quickstart

This guide walks through a complete CTM simulation, introducing
the main building blocks: network topology, model selection, simulation execution,
and result inspection.

## 1. Build a network

A network consists of **links** connected by **nodes**. Virtual `Origin` and `Destination`
nodes represent demand sources and sinks at the boundary of the modelled area.

The example below models a simple motorway stretch with a bottleneck (lane drop in the
middle) and an on-ramp.

```python
from traffic_flow_models import (
    Network, Node,
    MotorwayLink, Onramp,
    Origin, Destination,
)

# Origins (virtual demand sources)
o_main = Origin(id="o_main")
o_ramp = Origin(id="o_ramp")

# Destination (virtual sink)
dest = Destination(id="d_main")

# Links — every link needs lane_capacity, free_flow_speed, jam_density
link1  = MotorwayLink(id="m1", length=2.0, lanes=3,
                      lane_capacity=2000, free_flow_speed=100, jam_density=180)
onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180)
link2  = MotorwayLink(id="m2", length=1.0, lanes=1,   # bottleneck
                      lane_capacity=2000, free_flow_speed=100, jam_density=180)
link3  = MotorwayLink(id="m3", length=1.0, lanes=3,
                      lane_capacity=2000, free_flow_speed=100, jam_density=180)

# Nodes
n0 = Node(id="n0", incoming=[o_main],          outgoing=[link1])
n1 = Node(id="n1", incoming=[link1, onramp],   outgoing=[link2])
n2 = Node(id="n2", incoming=[o_ramp],          outgoing=[onramp])
n3 = Node(id="n3", incoming=[link2],           outgoing=[link3])
n4 = Node(id="n4", incoming=[link3],           outgoing=[dest])

net = Network(nodes=[n0, n1, n2, n3, n4])
```

## 2. Define demand

Demand is a **callable** `(t: float) -> float` for each `Origin`, where `t` is the
current simulation time in hours. Here we use constant or piecewise-constant demand.

```python
main_demand = lambda t: 1500.0                              # constant 1500 veh/h
ramp_demand = lambda t: 800.0 if t < 0.5 else 400.0         # steps down after 30 min
```

The simulation also requires turning rates at each node (fractions of flow routed into
each outgoing link) and boundary conditions at each destination:

```python
origin_demands = {
    "o_main": main_demand,
    "o_ramp": ramp_demand,
}

turning_rates = {
    "n0": lambda t: {"m1": 1.0},
    "n1": lambda t: {"m2": 1.0},
    "n2": lambda t: {"r1": 1.0},
    "n3": lambda t: {"m3": 1.0},
    "n4": lambda t: {"d_main": 1.0},
}

destination_flow_bc    = {"d_main": lambda t: 6000.0}  # unconstrained exit
destination_density_bc = {"d_main": lambda t: 0.0}
```

## 3. Choose a model and run

=== "CTM"

    ```python
    from traffic_flow_models import CTM, Simulation

    # CTM takes no constructor arguments;
    # fundamental-diagram parameters are set on each MotorwayLink.
    sim = Simulation(network=net, model=CTM())
    time_arr, states, disturbances = sim.run(
        duration=1.0,
        dt=10 / 3600,
        origin_demands=origin_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
    )
    ```

=== "METANET"

    ```python
    from traffic_flow_models import METANET, METANETParams, Simulation

    # METANETParams holds the second-order coefficients only;
    # fundamental-diagram parameters (vf, Qc_lane, rho_jam) stay on the links.
    model_params: METANETParams = {
        "tau":   22 / 3600,  # relaxation time [h]
        "nu":    15.0,       # anticipation coefficient [km²/h]
        "kappa": 40.0,       # density smoothing [veh/km/lane]
        "delta": 0.012,      # on-ramp speed-drop coefficient
        "phi":   1.0,        # lane-drop anticipation coefficient
        "alpha": 1.8,        # FD shape exponent
    }

    sim = Simulation(network=net, model=METANET(), model_params=model_params)
    time_arr, states, disturbances = sim.run(
        duration=1.0,
        dt=10 / 3600,
        origin_demands=origin_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
    )
    ```

## 4. Inspect results

```python
VKT, VHT, avg_speed = sim.compute_metrics(
    states=states, dt=10 / 3600, timesteps=len(time_arr)
)
print(f"Total VKT : {VKT:.0f} veh·km")
print(f"Total VHT : {VHT:.1f} veh·h")
print(f"Mean speed: {avg_speed:.1f} km/h")
```

## 5. Save results and generate plots

Pass `plot_results=True` (and optionally `results_dir`) to `run()` to automatically
save plots and a JSON results file:

```python
time_arr, states, disturbances = sim.run(
    duration=1.0,
    dt=10 / 3600,
    origin_demands=origin_demands,
    turning_rates=turning_rates,
    destination_flow_bc=destination_flow_bc,
    destination_density_bc=destination_density_bc,
    plot_results=True,
    results_dir="results/quickstart",
)
```

To generate a basic video from saved results:

```python
sim.visualize(
    results_filepath="results/quickstart/results.json",
    output_filepath="results/quickstart/simulation.avi",
    fps=10,
)
```

---

Next steps:

- [Network Structure](../concepts/network.md) — understand the topology model in depth
- [CTM](../models/ctm.md) / [METANET](../models/metanet.md) — model equations and parameter reference
- [Ramp Metering](../control/ramp-metering.md) — add a controller to the simulation
