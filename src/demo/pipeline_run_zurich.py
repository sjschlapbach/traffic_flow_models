from traffic_flow_models import SUMOPipeline, SUMOSimulation

if __name__ == "__main__":
    # scenario definitions
    name = "zurich"
    location = "Zurich, Switzerland"
    vehicle_demand = 2000

    # run the pipeline to generate files for the SUMO simulation
    network = SUMOPipeline(name=name, location=location)
    network.fetch_OSM()
    network.covert_to_sumo()
    network.create_consolidated_network()
    network.generate_detectors()
    network.generate_demand(vehicle_count=vehicle_demand)

    # run the SUMO simulation
    sim = SUMOSimulation(
        name=name,
        net_file=network.net_file,
        detector_file=network.detector_file,
        rou_file=network.rou_file,
        output_dir=network.output_dir,
    )
    sim.write_config()
    sim.run_simulation()
