import os
import subprocess
import sys
import shutil
import osmnx as ox
from functools import wraps
import logging
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.turning_rate_aggregator import TurningRateAggregator
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
        detector_file: Path to the SUMO loop detector definition file.
        detector_output_file: Path to the SUMO detector output file (written by SUMO).
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
        self.detector_output_file: Optional[str] = None  # Set by generate_detectors()
        self.rou_file: str = os.path.join(self.output_dir, f"{name}.rou.xml")

        self.detector_spec_path: str = os.path.join(
            self.output_dir, f"{name}_detectors_spec.csv"
        )
        self.detector_output_path: str = os.path.join(
            self.output_dir, "detectors_output.xml"
        )
        self.consolidated_network: Optional[Network] = None
        self.arbitrator: Optional[NetworkArbitrator] = None
        self.origin_ids: Optional[list[str]] = None
        self.onramp_ids: Optional[list[str]] = None
        self.offramp_ids: Optional[list[str]] = None
        self.destination_ids: Optional[list[str]] = None
        self.backbone_node_ids: Optional[set[str]] = None
        self.road_params: Optional[RoadParamsConfig] = None
        self.diverge_node_info: Optional[dict[str, list[str]]] = None

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
    def generate_demand(
        self,
        vehicle_count: int,
        duration_seconds: float,
        backbone_vehicle_count: int = 0,
        seed: int = 42,
    ) -> None:
        """Generate traffic demand combining random trips and backbone-direct demand.

        Creates random trips using SUMO's randomTrips.py tool and optionally
        adds vehicles departing directly from backbone origin inflow edges.
        Both sets of vehicles are merged into a single route file, with
        backbone vehicles spread uniformly across the simulation duration.

        Args:
            vehicle_count: Number of random vehicles to generate across the
                full SUMO network.
            duration_seconds: Simulation duration in seconds. Used to spread
                backbone departures uniformly over time.
            backbone_vehicle_count: Number of additional vehicles to place
                directly on backbone origin inflow edges. Default 0 (disabled).
            seed: Random seed for backbone trip generation.

        Raises:
            ValueError: If backbone_vehicle_count > 0 but consolidated_network
                is not initialized.
        """

        logger = logging.getLogger(__name__)

        if "SUMO_HOME" not in os.environ:
            print("Error: Please set the 'SUMO_HOME' environment variable.")
            return

        random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

        # ------------------------------------------------------------------
        # 1. Generate random trips as before
        # ------------------------------------------------------------------
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
            print(f"{self.rou_file} generated ({vehicle_count} random vehicles).")
            if os.path.exists("temp_trips.xml"):
                os.remove("temp_trips.xml")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while generating random demand: {e}")
            return

        # ------------------------------------------------------------------
        # 2. Optionally inject backbone-direct vehicles
        # ------------------------------------------------------------------
        if backbone_vehicle_count <= 0:
            return

        if self.consolidated_network is None:
            raise ValueError(
                "consolidated_network is not initialized. "
                "Call create_consolidated_network() before using backbone_vehicle_count > 0."
            )

        from traffic_flow_models.network import Origin, Destination

        rng = random.Random(seed)

        # collect backbone origin and destination node IDs
        backbone_origin_node_ids: set[str] = set()
        backbone_destination_node_ids: set[str] = set()

        for node in self.consolidated_network:
            for link in node.incoming:
                if isinstance(link, Origin):
                    backbone_origin_node_ids.add(link.destination_node_id)
            for link in node.outgoing:
                if isinstance(link, Destination):
                    backbone_destination_node_ids.add(link.origin_node_id)

        logger.debug(
            "Backbone origin nodes: %d, destination nodes: %d",
            len(backbone_origin_node_ids),
            len(backbone_destination_node_ids),
        )

        # find inflow/outflow edges from the SUMO network XML
        tree = ET.parse(self.net_file)
        root = tree.getroot()

        origin_inflow_edges: list[str] = []
        destination_outflow_edges: list[str] = []

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            edge_id = edge.get("id")
            edge_type = edge.get("type", "").lower()
            from_node = edge.get("from")
            to_node = edge.get("to")
            is_motorway = "motorway" in edge_type

            if not is_motorway and to_node in backbone_origin_node_ids:
                origin_inflow_edges.append(edge_id)
                logger.debug("Backbone inflow edge: '%s' → node '%s'", edge_id, to_node)

            if not is_motorway and from_node in backbone_destination_node_ids:
                destination_outflow_edges.append(edge_id)
                logger.debug(
                    "Backbone outflow edge: '%s' ← node '%s'", edge_id, from_node
                )

        if not origin_inflow_edges:
            raise ValueError(
                "No backbone inflow edges found in SUMO network. "
                "Check that the SUMO network and consolidated network are consistent."
            )
        if not destination_outflow_edges:
            raise ValueError("No backbone outflow edges found in SUMO network.")

        # build backbone trips spread uniformly over the simulation duration
        interval = duration_seconds / backbone_vehicle_count
        backbone_trips: list[ET.Element] = []

        for i in range(backbone_vehicle_count):
            depart_time = i * interval
            from_edge = rng.choice(origin_inflow_edges)
            to_edge = rng.choice(destination_outflow_edges)

            attempts = 0
            while to_edge == from_edge and attempts < 10:
                to_edge = rng.choice(destination_outflow_edges)
                attempts += 1

            trip = ET.Element("trip")
            trip.set("id", f"backbone_veh_{i}")
            trip.set("depart", f"{depart_time:.2f}")
            trip.set("from", from_edge)
            trip.set("to", to_edge)
            trip.set("departLane", "best")
            trip.set("departSpeed", "max")
            backbone_trips.append(trip)

        logger.debug(
            "Generated %d backbone trips over %.1fs (interval=%.2fs)",
            backbone_vehicle_count,
            duration_seconds,
            interval,
        )

        # ------------------------------------------------------------------
        # 3. Merge backbone trips into the existing route file
        # ------------------------------------------------------------------
        existing_tree = ET.parse(self.rou_file)
        existing_root = existing_tree.getroot()

        for trip in backbone_trips:
            existing_root.append(trip)

        # re-sort all trips and vehicles by depart time so SUMO does not
        # complain about non-monotonic departure order
        children = list(existing_root)
        children.sort(key=lambda el: float(el.get("depart", "0")))
        existing_root[:] = children

        ET.indent(existing_root, space="  ")
        existing_tree.write(self.rou_file, encoding="utf-8", xml_declaration=True)

        print(
            f"{self.rou_file} updated with {backbone_vehicle_count} backbone vehicles "
            f"({vehicle_count + backbone_vehicle_count} total)."
        )

    def create_consolidated_network(
        self,
        min_link_length: float,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        list[str],
        RoadParamsConfig,
        dict[str, list[str]],
        set[str],
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
                - road_params: Road parameters configuration used for the network.
                - diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs.
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
            self.offramp_ids,
            self.destination_ids,
            self.road_params,
            self.diverge_node_info,
            self.backbone_node_ids,
        ) = self.arbitrator.run()

        return (
            self.consolidated_network,
            self.origin_ids,
            self.onramp_ids,
            self.offramp_ids,
            self.destination_ids,
            self.road_params,
            self.diverge_node_info,
            self.backbone_node_ids,
        )

    def generate_detectors(self, cell_size: float) -> Tuple[str, str, str]:
        """Generate loop detectors at network interface points.

        Creates loop detectors at the boundaries between the macroscopic network
        and the SUMO microscopic network. Detectors are placed to measure inflow
        and outflow at these interface points, enabling demand aggregation for
        the macroscopic flow simulation.

        If the consolidated network has not been created yet, this method will
        automatically create it first.

        Returns:
            A tuple containing:
                - detector_file: Path to the generated SUMO detector definition XML file.
                - detector_output_file: Path to the detector output XML file (where SUMO writes data).
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
            or self.offramp_ids is None
            or self.destination_ids is None
            or self.backbone_node_ids is None
        ):
            raise ValueError(
                "Network parameters must be initialized before generating detectors"
            )

        # generate detectors
        generator = LoopDetectorGenerator(
            sumo_network_path=self.net_file,
            origin_ids=self.origin_ids,
            onramp_ids=self.onramp_ids,
            offramp_ids=self.offramp_ids,
            destination_ids=self.destination_ids,
            backbone_node_ids=self.backbone_node_ids,
            output_dir=self.output_dir,
            diverge_node_info=(
                self.diverge_node_info if self.diverge_node_info is not None else {}
            ),
            target_cell_length_km=cell_size,
        )
        self.detector_file, self.detector_output_file, self.detector_spec_path = (
            generator.generate()
        )

        return self.detector_file, self.detector_output_file, self.detector_spec_path

    def compute_splits(
        self, window_size_minutes: float = 2.0
    ) -> dict[str, Callable[[float], dict[str, float]]]:
        """Compute split ratios (turning rates) from SUMO detector data.

        This is the primary method to obtain splits for the macroscopic network.
        It processes loop detector outputs from SUMO simulations placed at diverge
        nodes to compute time-varying split ratios based on actual observed traffic
        distribution using rolling window temporal aggregation.

        If detector data is unavailable for some diverge nodes, falls back to
        lane-based splits for those nodes.

        This method should be called after running the SUMO simulation with
        detector outputs available.

        Args:
            window_size_minutes: Rolling window size in minutes for temporal aggregation (default: 2.0).
                At query time t, vehicle counts from [t - window/2, t + window/2] are aggregated.

        Returns:
            Dictionary mapping diverge node IDs to split functions.
            Each function takes time in hours and returns a dictionary mapping
            edge IDs to their split ratios (fractions between 0 and 1).

        Raises:
            ValueError: If detector files have not been generated.
        """
        if not os.path.exists(self.detector_output_path):
            raise ValueError(
                f"Detector output file not found: {self.detector_output_path}. "
                "Please run the SUMO simulation first."
            )

        if not os.path.exists(self.detector_spec_path):
            raise ValueError(
                f"Detector specification file not found: {self.detector_spec_path}. "
                "Please generate detectors first using generate_detectors()."
            )

        if not self.diverge_node_info:
            print(
                "Warning: No diverge nodes found in network. "
                "Returning empty splits dictionary."
            )
            return {}

        # compute detector-based turning rates
        aggregator = TurningRateAggregator(
            detector_output_path=self.detector_output_path,
            detector_spec_path=self.detector_spec_path,
            window_size_minutes=window_size_minutes,
        )
        detector_based_splits = aggregator.run()

        # fall back to lane-based splits for nodes without detector data
        if self.consolidated_network and self.arbitrator:
            lane_based_splits = self.arbitrator.compute_lane_based_splits(
                network=self.consolidated_network
            )

            # use detector-based splits where available, lane-based as fallback
            for node_id in self.diverge_node_info:
                if (
                    node_id not in detector_based_splits
                    and node_id in lane_based_splits
                ):
                    detector_based_splits[node_id] = lane_based_splits[node_id]
                    print(f"  Using lane-based fallback for diverge node {node_id}")
                elif (
                    node_id not in detector_based_splits
                    and node_id not in lane_based_splits
                ):
                    raise ValueError(
                        f"No split data available for diverge node {node_id} from either detectors or lane-based estimation."
                    )

        return detector_based_splits

    def get_consolidated_network(
        self,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        list[str],
        RoadParamsConfig,
        dict[str, list[str]],
        set[str],
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
                - road_params: Road parameters configuration used for the network.
                - diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs.

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
            or self.offramp_ids is None
            or self.destination_ids is None
            or self.backbone_node_ids is None
            or self.road_params is None
            or self.diverge_node_info is None
        ):
            raise ValueError("Network parameters have not been properly initialized.")

        return (
            self.consolidated_network,
            self.origin_ids,
            self.onramp_ids,
            self.offramp_ids,
            self.destination_ids,
            self.road_params,
            self.diverge_node_info,
            self.backbone_node_ids,
        )
