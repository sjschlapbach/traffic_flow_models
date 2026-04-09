import os
import subprocess
import sys
import shutil
import osmnx as ox
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Callable
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.turning_rate_aggregator import TurningRateAggregator
from traffic_flow_models.arbitrator.network_arbitrator import (
    NetworkArbitrator,
    RoadParamsConfig,
)
from traffic_flow_models.network.network import Network, Destination, MotorwayLink
from traffic_flow_models import Simulation


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

    def build_destination_boundary_conditions(
        self,
        backbone_state_path: str,
    ) -> Tuple[
        dict[str, Callable[[float], float]],
        dict[str, Callable[[float], float]],
    ]:
        """Build destination boundary condition callables from backbone state data.
 
        For each destination, walks the network topology to find the upstream
        MotorwayLink and reads the flow and density of its last cell from the
        backbone state file produced by ``BackboneStateAggregator.run()``.
        Returns time-interpolating callables suitable for passing directly to
        ``Simulation.run()`` and ``Calibrator.calibrate_model_params()``.
 
        This method is the downstream counterpart to ``compute_splits()`` and
        ``DemandAggregator.run()`` — all three together fully specify the
        boundary conditions required by the macroscopic simulation.
 
        Args:
            backbone_state_path: Path to the backbone_state.json written by
                ``BackboneStateAggregator.run()``.
 
        Returns:
            A tuple ``(destination_flow_bc, destination_density_bc)`` where each
            is a dict mapping destination ID to a callable ``f(t) -> float`` that
            returns the boundary value at simulation time ``t`` (in hours).
 
        Raises:
            ValueError: If the consolidated network has not been created yet.
        """
        if self.consolidated_network is None or self.destination_ids is None:
            raise ValueError(
                "Please first generate the consolidated network using "
                "create_consolidated_network() before building boundary conditions."
            )
 
        # load backbone state via the same interface used by the Calibrator
        time_h, state_history, _, _ = Simulation.load_results(
            filepath=backbone_state_path, network=self.consolidated_network
        )
        T = len(time_h)
 
        # map each Destination link ID -> the last upstream MotorwayLink.
        # origin_node_id on the Destination identifies the node it is attached
        # to; the upstream MotorwayLink is the last incoming link of that node.
        node_by_id = {node.id: node for node in self.consolidated_network}
        dest_to_upstream_link: dict[str, MotorwayLink] = {}
        for node in self.consolidated_network:
            for link in node.outgoing:
                if not isinstance(link, Destination):
                    continue
                host_node = node_by_id.get(link.origin_node_id)
                if host_node is None:
                    continue
                upstream = [l for l in host_node.incoming if isinstance(l, MotorwayLink)]
                if upstream:
                    dest_to_upstream_link[link.id] = upstream[-1]
 
        destination_flow_bc: dict[str, Callable[[float], float]] = {}
        destination_density_bc: dict[str, Callable[[float], float]] = {}
 
        for dest_id in self.destination_ids:
            link = dest_to_upstream_link.get(dest_id)
 
            if link is None:
                print(
                    f"  Warning: no upstream MotorwayLink found for destination "
                    f"'{dest_id}' — using constant fallback values."
                )
                destination_flow_bc[dest_id] = lambda _t: 6000.0
                destination_density_bc[dest_id] = lambda _t: 10.0
                continue
 
            # extract the last-cell flow and density for every backbone timestep
            flow_series = np.empty(T)
            density_series = np.empty(T)
            for t_idx in range(T):
                state_dict = self.consolidated_network.state_vec_to_network_dict(
                    state_history[:, t_idx]
                )
                link_state = state_dict.get(link.id, {})
                rho = link_state.get("density", [10.0])
                q = link_state.get("flow", [6000.0])
                density_series[t_idx] = rho[-1] if len(rho) > 0 else 10.0
                flow_series[t_idx] = q[-1] if len(q) > 0 else 6000.0
 
            # build interpolating callables.
            # default-argument capture (_th, _q, _rho) avoids the late-binding
            # closure pitfall when creating lambdas inside a loop.
            destination_flow_bc[dest_id] = (
                lambda t, _th=time_h, _q=flow_series: float(np.interp(t, _th, _q))
            )
            destination_density_bc[dest_id] = (
                lambda t, _th=time_h, _rho=density_series: float(
                    np.interp(t, _th, _rho)
                )
            )
 
        return destination_flow_bc, destination_density_bc

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
