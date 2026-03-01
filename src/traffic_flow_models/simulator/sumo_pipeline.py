import os
import subprocess
import sys
import shutil
import osmnx as ox
from functools import wraps
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable, Union
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.network_arbitrator import (
    NetworkArbitrator,
    RoadParamsConfig,
)
from traffic_flow_models.network.network import Network


def skip_if_exists(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if os.path.exists(getattr(self, attr_name)):
                print(f"[SKIP] {getattr(self, attr_name)} already exists.")
                return
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class SUMOPipeline:
    """A pipeline for fetching OSM data and preparing SUMO simulation files.

    Attributes:
        name: Name identifier for the simulation.
        location: Geographic location to fetch OSM data from.
        output_dir: Directory where output files are stored.
        osm_file: Path to the downloaded OSM file.
        net_file: Path to the SUMO network file.
        detector_file: Path to the SUMO loop detectors file.
        rou_file: Path to the SUMO route file.
        consolidated_network: Network object representing the macroscopic network.
        arbitrator: NetworkArbitrator instance used for network conversion.
        origin_ids: List of origin node IDs in the network.
        onramp_ids: List of onramp node IDs in the network.
        destination_ids: List of destination node IDs in the network.
        splits: Dictionary mapping node IDs to their outgoing link split ratios.
    """

    def __init__(self, name: str, location: str, road_params_config_path: str):
        """Initialize the SUMO pipeline.

        Args:
            name: Name identifier for the simulation.
            location: Geographic location to fetch OSM data from.
            road_params_config_path: Path to JSON configuration file containing
                road parameters (lane_capacity, jam_density, free_flow_speed for each road type).
        """
        self.name: str = name
        self.location: str = location
        self.road_params_config_path: str = road_params_config_path

        # set up output directory
        self.output_dir: str = os.path.join("results", name)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.osm_file: str = os.path.join(self.output_dir, f"{name}.osm")
        self.net_file: str = os.path.join(self.output_dir, f"{name}.net.xml")
        self.detector_file: str = os.path.join(self.output_dir, f"{name}detectors.xml")
        self.rou_file: str = os.path.join(self.output_dir, f"{name}.rou.xml")

        self.detector_spec_path: str = os.path.join(
            self.output_dir, f"{name}_detectors_spec.csv"
        )
        self.consolidated_network: Optional[Network] = None
        self.arbitrator: Optional[NetworkArbitrator] = None
        self.origin_ids: Optional[list[str]] = None
        self.onramp_ids: Optional[list[str]] = None
        self.destination_ids: Optional[list[str]] = None
        self.splits: Optional[dict[str, Callable[[float], dict[str, float]]]] = None
        self.road_params: Optional[RoadParamsConfig] = None

    @skip_if_exists("osm_file")
    def fetch_OSM(self) -> None:
        """Download OSM data for the specified location.

        Fetches road network data from OpenStreetMap, plots the network,
        and saves it as an OSM XML file.
        """
        ox.settings.all_oneway = True
        graph = ox.graph_from_place(self.location, network_type="drive", simplify=False)
        fig, ax = ox.plot_graph(
            graph,
            bgcolor="white",
            node_color="blue",
            node_size=2,
            edge_color="gray",
            edge_linewidth=0.3,
            show=False,
            close=False,
        )
        ax.set_title(f"{self.location} Road Network", fontsize=16)
        fig.savefig(os.path.join(self.output_dir, "network_plot.png"), dpi=500)
        plt.close(fig)
        ox.save_graph_xml(graph, filepath=self.osm_file)
        print(f"OSM data downloaded for {self.location}")

    @skip_if_exists("net_file")
    def convert_to_sumo(self) -> None:
        """Convert OSM file to SUMO network format.

        Converts the downloaded OSM data to a SUMO .net.xml file using netconvert
        with various processing options for geometry simplification, junction handling,
        and traffic signal detection.
        """
        cmd = [
            "netconvert",
            "--osm-files",
            self.osm_file,
            "--output-file",
            self.net_file,
            "--geometry.remove",
            "true",
            "--junctions.join",
            "true",
            "--roundabouts.guess",
            "true",
            "--tls.discard-simple",
            "true",
            "--verbose",
            "true",
            "--ramps.guess",
            "true",
            "--remove-edges.isolated",
            "true",
            "--tls.guess-signals",
            "true",
            "--tls.join",
            "true",
            "--tls.ignore-internal-junction-jam",
            "true",
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"{self.net_file} file generated.")

    # @skip_if_exists('rou_file')
    def generate_demand(self, vehicle_count: int) -> None:
        """Generate traffic demand and create route file.

        Creates random trips using SUMO's randomTrips.py tool and generates
        a route file with the specified vehicle count.

        Args:
            vehicle_count: Number of vehicles to generate in the simulation.
        """
        if "SUMO_HOME" not in os.environ:
            print("Error: Please set the 'SUMO_HOME' environment variable.")
            return

        random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

        cmd = [
            sys.executable,
            random_trips,
            "-n",
            self.net_file,
            "-o",
            "temp_trips.xml",
            "--route-file",
            self.rou_file,
            "--period",
            str(3600 / vehicle_count),
            "--fringe-factor",
            "10",
            "--validate",
            "--remove-loops",
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"{self.rou_file} file generated.")

            if os.path.exists("temp_trips.xml"):
                os.remove("temp_trips.xml")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while generating demand: {e}")

    def create_consolidated_network(
        self,
        min_link_length: float,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        dict[str, Callable[[float], dict[str, float]]],
        RoadParamsConfig,
    ]:
        """Create consolidated network from SUMO network.

        Instantiates a NetworkArbitrator to convert the SUMO microscopic network
        into a consolidated macroscopic network. The arbitration process includes
        filtering roads, merging serial edges, handling roundabouts, handling short
        links for CFL stability, and assigning appropriate parameters.

        Args:
            min_link_length: Minimum acceptable link length in kilometers for CFL stability.
                If specified, links shorter than this threshold are either stretched (if > 50% of minimum)
                or fused by contracting their nodes (if <= 50% of minimum).

        Returns:
            A tuple containing:
                - consolidated_network: Network object representing the macroscopic network.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - splits: Dictionary mapping node IDs to their outgoing link split ratios as functions of time.
                - road_params: Road parameters configuration used for the network.
        """
        self.arbitrator = NetworkArbitrator(
            net_xml_path=os.path.normpath(self.net_file),
            road_params_config_path=self.road_params_config_path,
            min_link_length=min_link_length,
        )
        (
            self.consolidated_network,
            self.origin_ids,
            self.onramp_ids,
            self.destination_ids,
            self.splits,
            self.road_params,
        ) = self.arbitrator.run()

        return (
            self.consolidated_network,
            self.origin_ids,
            self.onramp_ids,
            self.destination_ids,
            self.splits,
            self.road_params,
        )

    def generate_detectors(self) -> Tuple[str, str]:
        """Generate loop detectors at network interface points.

        Creates loop detectors at the boundaries between the macroscopic network
        and the SUMO microscopic network. Detectors are placed to measure inflow
        and outflow at these interface points, enabling demand aggregation for
        the macroscopic flow simulation.

        If the consolidated network has not been created yet, this method will
        automatically create it first.

        Returns:
            A tuple containing:
                - detector_file: Path to the generated SUMO detector XML file.
                - detector_spec_path: Path to the detector specification CSV file.

        Raises:
            ValueError: If the consolidated network has not been initialized.
        """
        if self.consolidated_network is None:
            raise ValueError(
                "Please first generate the consolidated network using the create_consolidated_network() method before generating detectors."
            )

        # ensure network parameters exist
        if (
            self.origin_ids is None
            or self.onramp_ids is None
            or self.destination_ids is None
        ):
            raise ValueError(
                "Network parameters must be initialized before generating detectors"
            )

        # generate detectors
        generator = LoopDetectorGenerator(
            sumo_network_path=self.net_file,
            origin_ids=self.origin_ids,
            onramp_ids=self.onramp_ids,
            destination_ids=self.destination_ids,
            output_dir=self.output_dir,
        )
        self.detector_file, self.detector_spec_path = generator.generate()

        return self.detector_file, self.detector_spec_path

    def get_consolidated_network(
        self,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        dict[str, Callable[[float], dict[str, float]]],
        RoadParamsConfig,
    ]:
        """Retrieve the consolidated macroscopic network and metadata.

        Provides access to the previously generated network and its
        associated metadata. This method should be called after either
        generate_detectors() or create_consolidated_network() has been executed.

        Returns:
            A tuple containing:
                - consolidated_network: Network object representing the macroscopic network.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - splits: Dictionary mapping node IDs to their outgoing link split ratios
                    as a time-varying function.
                - road_params: Road parameters configuration used for the network.

        Raises:
            ValueError: If consolidated network has not been created yet.
        """
        if self.consolidated_network is None:
            raise ValueError(
                "Please first compute the consolidated network using the create_consolidated_network() method."
            )

        if (
            self.origin_ids is None
            or self.onramp_ids is None
            or self.destination_ids is None
            or self.splits is None
            or self.road_params is None
        ):
            raise ValueError("Network parameters have not been properly initialized.")

        return (
            self.consolidated_network,
            self.origin_ids,
            self.onramp_ids,
            self.destination_ids,
            self.splits,
            self.road_params,
        )
