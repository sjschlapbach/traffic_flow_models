"""
Run the entire pipeline for a given location or set of SUMO files, culminating in a macroscopic simulation and analysis.

Example calls:
python src/demo/run_pipeline.py --location "Zurich, Switzerland" --vehicle-demand 2000 --highway-vehicle-demand 5000 --no-plot
python src/demo/run_pipeline.py --location "Zurich, Switzerland" --vehicle-demand 2000 --highway-vehicle-demand 5000 --demand-profile "[[0.0,0.3],[0.3,0.5],[0.8,0.2]]" --no-plot
python src/demo/run_pipeline.py --sumo-cfg-dir "src/demo/scenarios/example" --no-plot
"""

import argparse
import os
from datetime import datetime

from traffic_flow_models import (
    CTM,
    METANET,
    SUMOPipeline,
    SUMOSimulation,
    DemandAggregator,
    Simulation,
    Calibrator,
    BackboneStateAggregator,
    NetworkArbitrator,
)

if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Run entire SUMO and Macroscopic Simulation Pipeline for an arbitrary City (Nomatim standard strings)."
    )

    # ! Option 1: Choose location (Nomatim string) and demand profile settings
    args.add_argument(
        "--location",
        type=str,
        default="Zurich, Switzerland",
        help="Location for the scenario (default: 'Zurich, Switzerland')",
    )
    args.add_argument(
        "--vehicle-demand",
        type=int,
        default=20000,
        help="Vehicle demand for the scenario (default: 20000)",
    )
    args.add_argument(
        "--highway-vehicle-demand",
        type=int,
        default=0,
        help="Number of additional vehicles to place directly on highway origins (simulating inflow from upstream highway links).",
    )
    args.add_argument(
        "--demand-profile",
        type=str,
        default=None,
        help=(
            "Piecewise-linear demand profile as a matrix of [time, fraction] pairs. "
            "Times are relative (0.0-1.0). Fractions must sum to 1.0. "
            "Example: '[[0.0,0.3],[0.3,0.5],[0.8,0.2]]'. "
            "Default: uniform distribution."
        ),
    )

    # ! Option 2: Provide required SUMO files directly (net, route, sumocfg)
    args.add_argument(
        "--sumo-cfg-dir",
        type=str,
        help="Path to the directory containing the SUMO network, route, and configuration files",
    )

    # remaining arguments / settings
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
        "--model",
        type=str,
        choices=["CTM", "METANET"],
        default="CTM",
        help="Traffic flow model to use for the simulation (default: CTM)",
    )
    parsed_args = args.parse_args()

    # check if the location string or the prepared SUMO files should be used
    location_mode = True
    sumo_net_file_provided = ""
    sumo_route_file_provided = ""
    sumo_cfg_file_provided = ""

    if parsed_args.sumo_cfg_dir:
        location_mode = False

        # check that the provided SUMO configuration directory contains the required files
        # (and only exactly one of each type of file)
        net_files = [
            f for f in os.listdir(parsed_args.sumo_cfg_dir) if f.endswith(".net.xml")
        ]
        route_files = [
            f for f in os.listdir(parsed_args.sumo_cfg_dir) if f.endswith(".rou.xml")
        ]
        sumocfg_files = [
            f for f in os.listdir(parsed_args.sumo_cfg_dir) if f.endswith(".sumocfg")
        ]

        if len(net_files) != 1 or len(route_files) != 1 or len(sumocfg_files) != 1:
            raise ValueError(
                f"SUMO configuration directory must contain exactly one .net.xml file, one .rou.xml file, and one .sumocfg file. Found: {len(net_files)} .net.xml files, {len(route_files)} .rou.xml files, {len(sumocfg_files)} .sumocfg files."
            )

        print("Using provided SUMO files:")
        print(f"  Net file: {net_files[0]}")
        print(f"  Route file: {route_files[0]}")
        print(f"  SUMO config file: {sumocfg_files[0]}")
        sumo_net_file_provided = os.path.join(parsed_args.sumo_cfg_dir, net_files[0])
        sumo_route_file_provided = os.path.join(
            parsed_args.sumo_cfg_dir, route_files[0]
        )
        sumo_cfg_file_provided = os.path.join(
            parsed_args.sumo_cfg_dir, sumocfg_files[0]
        )

        # set the name of the scenario for pipeline initialization
        name = sumocfg_files[0].replace(".sumocfg", "")

    else:
        # set the name of the scenario for pipeline initialization
        name = parsed_args.location.split(",")[0].strip().lower().replace(" ", "_")

    # path to road parameters configuration
    road_params_config_path = os.path.join(
        os.path.dirname(__file__), "road_params_config.json"
    )

    # run the pipeline to generate files for the SUMO simulation
    pipeline = SUMOPipeline(
        name=name,
        location=parsed_args.location,
        road_params_config_path=road_params_config_path,
        output_dir=(
            os.path.join("results", name) if location_mode else parsed_args.sumo_cfg_dir
        ),
        clean_output_dir=location_mode,  # only clean output directory if we are generating new files (i.e., in location mode)
    )

    if location_mode:
        print(f"Running pipeline for location: {parsed_args.location}")

        # fetch the required data from OSM and prepare the SUMO network files
        pipeline.fetch_OSM()
        pipeline.convert_to_sumo()

    else:
        print(
            "Running pipeline with provided SUMO files (skipping OSM fetch and conversion steps)."
        )

        # set SUMO network and route files directly from provided arguments
        pipeline.net_file = sumo_net_file_provided
        pipeline.rou_file = sumo_route_file_provided

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

    # if the preferred cell size is smaller than the CFL minimum, notify the user
    if preferred_cell_size < cfl_minimum:
        raise ValueError(
            f"Preferred cell size ({preferred_cell_size} km) is too small for CFL stability with the given timestep (dt={dt} h) and maximum free-flow speed ({max_free_flow_speed} km/h). Minimum cell size for stability is {cfl_minimum:.3f} km. Please increase the preferred cell size or adjust the timestep."
        )

    # create an instance of the macroscopic traffic flow model network
    pipeline.create_consolidated_network(
        min_link_length=min_link_length, target_cell_length=preferred_cell_size
    )
    detector_def_file, detector_output_file, spec_file = pipeline.generate_detectors(
        cell_size=preferred_cell_size
    )

    # if the route file was not provided, generate it using the pipeline's demand generator
    if location_mode:
        # generate vehicles routes (and corresponding rou file) using the pipeline's demand generator
        pipeline.generate_demand(
            urban_count=int(parsed_args.vehicle_demand),
            duration_seconds=duration * 3600,
            highway_count=int(parsed_args.highway_vehicle_demand),
            demand_profile=SUMOPipeline.parse_demand_profile(
                parsed_args.demand_profile
            ),
        )

    # get the consolidated network as well as additional relevant node
    # and link IDs and parameters for the simulation
    (
        network,
        origin_ids,
        onramp_ids,
        offramp_ids,
        destination_ids,
        road_params,
        diverge_node_info,
        backbone_node_ids,
    ) = pipeline.get_consolidated_network()
    print(f"Origins:  {len(origin_ids)} → {origin_ids}")
    print(f"Onramps:  {len(onramp_ids)} → {onramp_ids}")
    print(f"Offramps: {len(offramp_ids)} → {offramp_ids}")
    print(f"Destinations: {len(destination_ids)}")

    # run the SUMO simulation
    sim = SUMOSimulation(
        name=name,
        simulation_end_time=int(duration * 3600),
        net_file=pipeline.net_file,
        detector_file=pipeline.detector_file,
        rou_file=pipeline.rou_file,
        cfg_file=(
            os.path.join(pipeline.output_dir, f"{name}.sumocfg")
            if location_mode
            else sumo_cfg_file_provided
        ),
        output_dir=pipeline.output_dir,
    )
    sim.write_config()
    sim.run_simulation()

    # aggregate demands from detector data
    demand_generator = DemandAggregator(
        detector_output_path=detector_output_file, detector_spec_path=spec_file
    )
    urban_demands = demand_generator.run(
        origin_ids=origin_ids,
        onramp_ids=onramp_ids,
        sumo_network_path=pipeline.net_file,
    )

    # compute boundary conditions from microscopic simulation results
    edge_data_path = os.path.join(pipeline.output_dir, "edge_data_output.xml")
    destination_flow_bc, destination_density_bc = (
        pipeline.build_destination_bc_from_sumo_edges(edge_data_path=edge_data_path)
    )

    # compute splits (turning rates) from detector data
    # This is the primary source of splits - detector-based with lane-based fallback
    # Uses rolling window aggregation (2 minutes by default) over small detector intervals (15 seconds)
    splits = pipeline.compute_splits(window_size_minutes=2.0)

    # initialize the results directory
    timestamp = datetime.now().strftime("simulation_results_%Y-%m-%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # ── Backbone state estimation ──────────────────────────────────────────────
    road_params = NetworkArbitrator._load_road_params_from_json(road_params_config_path)
    micro_results_path = os.path.join(results_dir, "micro_results.json")
    backbone_aggregator = BackboneStateAggregator(
        detector_output_path=detector_output_file,
        detector_spec_path=spec_file,
        window_size_minutes=10.0,
    )
    _, highway_demands = backbone_aggregator.run(
        output_path=micro_results_path,
        urban_demands=urban_demands,
        time_step_minutes=1.0,
        free_flow_speed=road_params["motorway"]["free_flow_speed"],
        jam_density=road_params["motorway"]["jam_density"],
        preferred_cell_size=preferred_cell_size,
        sumo_network_path=pipeline.net_file,
        origin_ids=origin_ids,
    )

    # combine the highway demands and urban demands into one origin demands dictionary
    # urban_demands technically contains demands for highway origins, these should be overwritten
    origin_demands = {**urban_demands, **highway_demands}

    # log the generated demands for use in the macroscopic simulation
    print("Demand keys:", sorted(origin_demands.keys()))
    print("Missing:", [k for k in origin_ids if k not in origin_demands])

    # run a simulation of the network using the selected model
    if parsed_args.model.upper() == "CTM":
        ctm = CTM()
        sim = Simulation(network=network, model=ctm)
        time, states, disturbances = sim.run(
            duration=duration,
            dt=dt,
            preferred_cell_size=preferred_cell_size,
            origin_demands=origin_demands,
            turning_rates=splits,
            destination_density_bc=destination_density_bc,
            destination_flow_bc=destination_flow_bc,
            plot_results=True,
            show_plots=plot_enabled,
            results_dir=results_dir,
        )

    elif parsed_args.model.upper() == "METANET":
        # Use the backbone state file for ground-truth states and forward the
        # callable disturbance functions (origin_demands, splits) to the
        # calibrator. The calibrator will build the disturbance history by
        # sampling these callables on the simulation time grid.
        print("Running METANET calibration using backbone aggregated states...")
        calibrator = Calibrator(network=network)
        metanet_model = METANET()
        calibrated_params, result, _ = calibrator.calibrate_model_params(
            ground_truth_filepath=micro_results_path,
            model=metanet_model,
            initial_params=None,
            window_size=30,
            stride=15,
            model_options={"link_specific_alpha": False},
            regularization_weight=0.01,
            verbose=True,
            use_parameter_search=False,
            save_dir=results_dir,
            use_disturbance_from_file=False,  # we will provide disturbance callables instead of using the file-based disturbances
            origin_demands_fn=origin_demands,
            turning_rates_fn=splits,
            flow_boundary_conditions_fn=destination_flow_bc,
            density_boundary_conditions_fn=destination_density_bc,
        )

        print(
            "Calibration complete — running METANET simulation with calibrated parameters"
        )
        metanet = METANET()
        sim = Simulation(network=network, model=metanet, model_params=calibrated_params)
        time, states, disturbances = sim.run(
            duration=duration,
            dt=dt,
            preferred_cell_size=preferred_cell_size,
            origin_demands=origin_demands,
            turning_rates=splits,
            destination_density_bc=destination_density_bc,
            destination_flow_bc=destination_flow_bc,
            plot_results=True,
            show_plots=plot_enabled,
            results_dir=results_dir,
        )

    else:
        raise ValueError(
            f"Unknown MODEL: {parsed_args.model}. Choose 'CTM' or 'METANET'."
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

        # also generate a backbone (microsimulation) video when available
        backbone_video_path = os.path.join(results_dir, "backbone_simulation.avi")
        try:
            print("\nGenerating backbone (microsimulation) video...")

            sim.visualize(
                results_filepath=micro_results_path,
                output_filepath=backbone_video_path,
                fps=30,
                subsampling=1,
            )
            print(f"Backbone video saved to: {backbone_video_path}")
        except Exception as e:
            print(f"Could not generate backbone video: {e}")

        # generate side-by-side comparison video (micro vs macro)
        comparison_video_path = os.path.join(results_dir, "simulation_comparison.avi")
        try:
            print("\nGenerating comparison video (micro vs macro)...")

            # ensure the backbone (micro) results are on the same time grid as
            # the macro simulation results. Resample the backbone file onto the
            # macro time array and pass the resampled file to visualize_comparison.
            macro_time_array, _, _, _ = Simulation.load_results(
                filepath=os.path.join(results_dir, "simulation_results.json"),
                network=network,
            )

            subsampled_micro_data = os.path.join(
                results_dir, "subsampled_micro_data.json"
            )
            Simulation.resample_results_file(
                source_filepath=micro_results_path,
                dest_filepath=subsampled_micro_data,
                target_time_array=macro_time_array,
            )

            sim.visualize_comparison(
                result_filepaths=[
                    subsampled_micro_data,
                    os.path.join(results_dir, "simulation_results.json"),
                ],
                labels=["Backbone (MICRO)", f"Macro {parsed_args.model.upper()}"],
                output_filepath=comparison_video_path,
                fps=30,
                subsampling=1,
            )
            print(f"Comparison video saved to: {comparison_video_path}")
        except Exception as e:
            print(f"Could not generate comparison video: {e}")
