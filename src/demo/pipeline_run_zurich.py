import argparse
import os
from datetime import datetime

from traffic_flow_models import (
    CTM,
    SUMOPipeline,
    SUMOSimulation,
    DemandAggregator,
    Simulation,
)
from traffic_flow_models.arbitrator.backbone_aggregator import BackboneStateAggregator

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run the Zurich demo scenario.")
    args.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of results (useful for CI or headless environments)",
    )
    args.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate video visualization of simulation results",
    )
    args.add_argument(
        "--vehicle-demand",
        type=int,
        help="Vehicle demand for the Zurich scenario (default: 20000)",
    )
    parsed_args = args.parse_args()

    # scenario definitions
    name = "zurich"
    location = "Zurich, Switzerland"
    vehicle_demand = (
        parsed_args.vehicle_demand if parsed_args.vehicle_demand is not None else 20000
    )

    # general macroscopic simulation settings
    dt = 10.0 / 3600
    duration = 5000.0 / 3600
    plot_enabled = not parsed_args.no_plot
    generate_video = parsed_args.generate_video
    preferred_cell_size = 0.5  # km

    # compute minimum link length for CFL stability
    # must account for both CFL condition (vf * dt) and preferred cell size
    max_free_flow_speed = 120.0  # km/h
    cfl_minimum = max_free_flow_speed * dt  # CFL condition: cell_length >= vf * dt
    min_link_length = max(cfl_minimum, preferred_cell_size) + 0.01  # km

    # path to road parameters configuration
    road_params_config_path = os.path.join(
        os.path.dirname(__file__), "road_params_config.json"
    )

    # run the pipeline to generate files for the SUMO simulation
    pipeline = SUMOPipeline(
        name=name, location=location, road_params_config_path=road_params_config_path
    )
    pipeline.fetch_OSM()
    pipeline.convert_to_sumo()
    pipeline.create_consolidated_network(min_link_length=min_link_length)
    detector_def_file, detector_output_file, spec_file = pipeline.generate_detectors()
    pipeline.generate_demand(vehicle_count=vehicle_demand)
    (
        network,
        origin_ids,
        onramp_ids,
        destination_ids,
        road_params,
        diverge_node_info,
    ) = pipeline.get_consolidated_network()

    # run the SUMO simulation
    sim = SUMOSimulation(
        name=name,
        net_file=pipeline.net_file,
        detector_file=pipeline.detector_file,
        rou_file=pipeline.rou_file,
        output_dir=pipeline.output_dir,
    )
    sim.write_config()
    sim.run_simulation()

    # aggregate demands from detector data
    demand_generator = DemandAggregator(
        detector_output_path=detector_output_file, detector_spec_path=spec_file
    )
    origin_demands, onramp_demands = demand_generator.run(
        origin_ids=origin_ids,
        onramp_ids=onramp_ids,
        sumo_network_path=pipeline.net_file,
    )

    combined_demands = {**origin_demands, **onramp_demands}

    # initialize the results directory
    timestamp = datetime.now().strftime("simulation_results_%Y-%m-%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    backbone_state_path = os.path.join(results_dir, "backbone_state.json")
    backbone_aggregator = BackboneStateAggregator(
        detector_output_path=detector_output_file,
        detector_spec_path=spec_file,
        window_size_minutes=2.0,
    )
    backbone_aggregator.run(
        output_path=backbone_state_path,
        time_step_minutes=1.0,
    )

    # compute splits (turning rates) from detector data
    # This is the primary source of splits - detector-based with lane-based fallback
    # Uses rolling window aggregation (2 minutes by default) over small detector intervals (15 seconds)
    splits = pipeline.compute_splits(window_size_minutes=2.0)

    # TODO: replace these, once they can be obtained from data
    destination_density_bc = {dest_id: lambda _t: 10.0 for dest_id in destination_ids}
    destination_flow_bc = {dest_id: lambda _t: 6000.0 for dest_id in destination_ids}

    # plot the network
    network.plot(save_path="results/zurich/network.png", show=plot_enabled)

    # run a simulation of the network using the CTM model
    ctm = CTM()
    sim = Simulation(network=network, model=ctm)
    time, states, disturbances = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=preferred_cell_size,
        # origin_demands=origin_demands,
        # onramp_demands=onramp_demands,
        origin_demands=combined_demands,
        turning_rates=splits,
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
        print("\nGenerating video visualization...")
        sim.visualize(
            results_filepath=os.path.join(results_dir, "simulation_results.json"),
            output_filepath=video_path,
            fps=30,
            subsampling=1,
        )
        print(f"Video saved to: {video_path}")
