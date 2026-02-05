from traffic_flow_models import SUMOPipeline, SUMOSimulation

if __name__ == "__main__":
    # scenario definitions
    name = "zurich"
    location = "Zurich, Switzerland"
    vehicle_demand = 2000

    # run the pipeline to generate files for the SUMO simulation
    network = SUMOPipeline(name, location)
    network.fetch_OSM()
    network.covert_to_sumo()
    network.generate_detectors()
    network.generate_demand(vehicle_demand)

    # run the SUMO simulation
    sim = SUMOSimulation(name, network.net_file, network.detector_file, network.rou_file, network.output_dir)
    sim.write_config()
    sim.run_simulation()
