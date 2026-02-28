import argparse

from traffic_flow_models import (
    CTM,
    SUMOPipeline,
    SUMOSimulation,
    DemandAggregator,
    Simulation,
)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run the Zurich demo scenario.")
    args.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of results (useful for CI or headless environments)",
    )
    parsed_args = args.parse_args()

    # scenario definitions
    name = "zurich"
    location = "Zurich, Switzerland"
    vehicle_demand = 2000

    # general macroscopic simulation settings
    dt = 10.0 / 3600
    duration = 5000.0 / 3600
    plot_enabled = not parsed_args.no_plot

    # run the pipeline to generate files for the SUMO simulation
    pipeline = SUMOPipeline(name=name, location=location)
    pipeline.fetch_OSM()
    pipeline.convert_to_sumo()
    pipeline.create_consolidated_network()
    detector_file, spec_file = pipeline.generate_detectors()
    pipeline.generate_demand(vehicle_count=vehicle_demand)
    (
        network,
        origin_ids,
        onramp_ids,
        destination_ids,
        splits,
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

    # initialize a demand generator and run it to obtain origin and onramp demands
    demand_generator = DemandAggregator(
        detector_output_path=detector_file, detector_spec_path=spec_file
    )
    origin_demands, onramp_demands = demand_generator.run(
        origin_ids=origin_ids,
        onramp_ids=onramp_ids,
        sumo_network_path=pipeline.net_file,
    )

    # TODO: replace these, once they can be obtained from data
    destination_density_bc = {dest_id: lambda t: 0.1 for dest_id in destination_ids}
    destination_flow_bc = {dest_id: lambda t: 0.1 for dest_id in destination_ids}

    # plot the network
    network.plot(save_path="results/zurich/network.png", show=plot_enabled)

    # run a simulation of the network using the CTM model
    ctm = CTM()
    sim = Simulation(network=network, model=ctm)
    time, states, disturbances = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=0.5,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=splits,
        destination_density_bc=destination_density_bc,
        destination_flow_bc=destination_flow_bc,
        plot_results=True,
        show_plots=plot_enabled,
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
