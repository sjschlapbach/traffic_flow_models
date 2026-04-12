import os
import argparse
from typing import Callable
from datetime import datetime

from traffic_flow_models import CTM, Simulation
from demo.scenarios import (
    setup_network_a,
    setup_network_b,
    setup_network_c,
    setup_network_c1,
    setup_network_c2,
    setup_network_c3,
    setup_network_c4,
    setup_network_d,
    setup_network_e,
    setup_network_e1,
    setup_network_e2,
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
        choices=["A", "B", "C", "C1", "C2", "C3", "C4", "D", "E", "E1", "E2"],
        default="A",
        help="Select the scenario to simulate (default: A). The subversions (if available) contain different ramp metering controllers for the on-ramp(s) in the scenario.",
    )
    args = parser.parse_args()
    plot_enabled = not args.no_plot
    generate_video = args.generate_video
    scenario = args.scenario

    # select the appropriate scenario functions
    if scenario == "A":
        setup_network = setup_network_a
    elif scenario == "B":
        setup_network = setup_network_b
    elif scenario == "C":
        setup_network = setup_network_c
    elif scenario == "C1":
        setup_network = setup_network_c1
    elif scenario == "C2":
        setup_network = setup_network_c2
    elif scenario == "C3":
        setup_network = setup_network_c3
    elif scenario == "C4":
        setup_network = setup_network_c4
    elif scenario == "D":
        setup_network = setup_network_d
    elif scenario == "E":
        setup_network = setup_network_e
    elif scenario == "E1":
        setup_network = setup_network_e1
    elif scenario == "E2":
        setup_network = setup_network_e2
    else:
        raise ValueError(f"Scenario {scenario} is not defined.")

    # initialize the network and get the origin demand callables mapping
    network, metadata, origin_demands = setup_network()

    destination_ids = metadata.get("destination_ids", [])
    splits = metadata.get("splits", {})
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
