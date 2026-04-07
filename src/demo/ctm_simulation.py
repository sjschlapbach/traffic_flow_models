import os
import argparse
from typing import Callable
from datetime import datetime

from traffic_flow_models import CTM, Simulation
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
    alinea_ramp_control = False
    alinea_gain = 5.0
    dt = 10.0 / 3600
    duration = 5000.0 / 3600

    # check if plotting is disabled through command line argument (CI environment)
    parser = argparse.ArgumentParser(description="CTM Simulation Demo")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting for CI/automated runs",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate video visualization of simulation results",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "C", "D"],
        default="A",
        help="Select the scenario to simulate (default: A)",
    )
    args = parser.parse_args()
    plot_enabled = not args.no_plot
    generate_video = args.generate_video
    scenario = args.scenario

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
    destination_ids = metadata.get("destination_ids", [])
    splits = metadata.get("splits", {})

    origin_demands: dict[str, Callable[[float], float]] = {
        "origin": mainline_demand,
        "origin_onr": onramp_demand,
    }
    destination_flow_bc: dict[str, Callable[[float], float]] = {
        did: (lambda _: 6000.0) for did in destination_ids
    }
    destination_density_bc: dict[str, Callable[[float], float]] = {
        did: (lambda _: 0.0) for did in destination_ids
    }

    # turning rates: create callables that return the provided split mapping (time-invariant here)
    turning_rates: dict[str, Callable[[float], dict[str, float]]] = {
        nid: (lambda _t, s=splits[nid]: s) for nid in splits.keys()
    }

    # initialize the results directory
    timestamp = datetime.now().strftime("simulation_results_%Y-%m-%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # run a simulation of the network using the CTM model
    ctm = CTM()
    sim = Simulation(network=network, model=ctm)
    time, states, disturbances = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=0.5,
        origin_demands=origin_demands,
        turning_rates=turning_rates,
        destination_density_bc=destination_density_bc,
        destination_flow_bc=destination_flow_bc,
        plot_results=True,
        show_plots=plot_enabled,
        results_dir=results_dir,
    )

    # compute performance metrics and illustrate them
    VKT, VHT, avg_speed = sim.compute_metrics(
        states=states,
        dt=dt,
        timesteps=len(time),
    )
    print(f"Total VKT: {VKT:.2f} veh-km")
    print(f"Total VHT: {VHT:.2f} veh-h")
    print(f"Overall Average Speed: {avg_speed:.2f} km/h")

    # generate video visualization if requested
    if generate_video:
        video_path = os.path.join(results_dir, "simulation.avi")
        print(f"\nGenerating video visualization...")
        sim.visualize(
            results_filepath=os.path.join(results_dir, "simulation_results.json"),
            output_filepath=video_path,
            fps=30,
            subsampling=1,
        )
        print(f"Video saved to: {video_path}")
