import argparse
from typing import Callable

from traffic_flow_models import METANET, METANETParams
from demo.scenarios import (
    mainline_demand_a,
    mainline_demand_b,
    mainline_demand_c,
    onramp_demand_a,
    onramp_demand_b,
    onramp_demand_c,
    setup_network_ab,
    setup_network_c,
    mainline_demand_d,
    onramp_demand_d,
    setup_network_d,
)


if __name__ == "__main__":
    # ! simulation configuration parameters
    scenario = "A"
    alinea_ramp_control = False
    alinea_gain = 5.0
    dt = 10.0 / 3600
    duration = 5000.0 / 3600

    # ! METANET model parameters
    tau = 22 / 3600
    nu = 15
    kappa = 10
    delta = 1.4
    phi = 10
    alpha = 2

    # check if plotting is disabled through command line argument (CI environment)
    parser = argparse.ArgumentParser(description="METANET Simulation Demo")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting for CI/automated runs",
    )
    args = parser.parse_args()
    plot_enabled = not args.no_plot

    # select the appropriate scenario functions
    if scenario == "A":
        mainline_demand = mainline_demand_a
        onramp_demand = onramp_demand_a
        setup_network = setup_network_ab
    elif scenario == "B":
        mainline_demand = mainline_demand_b
        onramp_demand = onramp_demand_b
        setup_network = setup_network_ab
    elif scenario == "C":
        mainline_demand = mainline_demand_c
        onramp_demand = onramp_demand_c
        setup_network = setup_network_c
    elif scenario == "D":
        mainline_demand = mainline_demand_d
        onramp_demand = onramp_demand_d
        setup_network = setup_network_d
    else:
        raise ValueError(f"Scenario {scenario} is not defined.")

    # initialize the network with the correct structure (optionally with ALINEA ramp metering)
    network, metadata = setup_network()

    # build disturbance dictionaries expected by the new simulate signature
    origin_ids = metadata.get("origin_ids", [])
    onramp_ids = metadata.get("onramp_ids", [])
    destination_ids = metadata.get("destination_ids", [])
    splits = metadata.get("splits", {})

    origin_demands: dict[str, Callable[[float], float]] = {
        oid: mainline_demand for oid in origin_ids
    }
    onramp_demands: dict[str, Callable[[float], float]] = {
        rid: onramp_demand for rid in onramp_ids
    }
    destination_boundary_conditions: dict[str, Callable[[float], float]] = {
        did: (lambda _: 0.0) for did in destination_ids
    }

    # turning rates: create callables that return the provided split mapping (time-invariant here)
    turning_rates: dict[str, Callable[[float], dict[str, float]]] = {
        nid: (lambda _t, s=splits[nid]: s) for nid in splits.keys()
    }

    # initialize the METANET model parameters
    model_params: METANETParams = {
        "tau": tau,
        "nu": nu,
        "kappa": kappa,
        "delta": delta,
        "phi": phi,
        "alpha": alpha,
    }

    # run a simulation of the network using the METANET model
    metanet = METANET()
    time, states, disturbances = network.simulate(
        duration=duration,
        dt=dt,
        model=metanet,
        model_params=model_params,
        preferred_cell_size=0.5,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=turning_rates,
        destination_boundary_conditions=destination_boundary_conditions,
        plot_results=True,
        show_plots=plot_enabled,
    )

    # compute performance metrics and illustrate them
    VKT, VHT, avg_speed = network.compute_performance_metrics(
        states=states,
        dt=dt,
        timesteps=len(time),
    )
    print(f"Total VKT: {VKT:.2f} veh-km")
    print(f"Total VHT: {VHT:.2f} veh-h")
    print(f"Overall Average Speed: {avg_speed:.2f} km/h")
