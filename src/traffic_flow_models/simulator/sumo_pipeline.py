import os
import sys
import json
import shutil
import random
import json
import subprocess
import osmnx as ox
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import json
import warnings
from functools import wraps
from typing import Optional, Tuple, Callable

from traffic_flow_models.network import Network, MotorwayLink
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.turning_rate_aggregator import TurningRateAggregator
from traffic_flow_models.arbitrator.network_arbitrator import (
    NetworkArbitrator,
    RoadParamsConfig,
    RoadTypeParams,
)


def skip_if_exists(attr_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # If the target output file already exists, skip the work to avoid
            # re-downloading or regenerating expensive SUMO inputs.
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
        offramp_ids: List of offramp node IDs in the network.
        destination_ids: List of destination node IDs in the network.
        backbone_node_ids: Set of node IDs that form the motorway backbone.
        diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs.
    """

    def __init__(
        self,
        name: str,
        location: str,
        road_params_config_path: str,
        output_dir: str,
        clean_output_dir: bool = False,
    ) -> None:
        """Initialize the SUMO pipeline.

        Args:
            name: Name identifier for the simulation.
            location: Geographic location to fetch OSM data from.
            road_params_config_path: Path to JSON configuration file containing
                road parameters (lane_capacity, jam_density, free_flow_speed for each road type).
            output_dir: Directory where all intermediate and final output files are stored.
            clean_output_dir: If True, deletes and recreates output_dir before starting.
        """
        self.name: str = name
        self.location: str = location
        self.road_params_config_path: str = road_params_config_path

        # set up output directory
        self.output_dir: str = output_dir
        if os.path.exists(self.output_dir) and clean_output_dir:
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # paths for intermediate and final SUMO inputs/outputs
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
        self.v_type_file: Optional[str] = None
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
        # Force all edges to be treated as one-way in the OSM graph to match
        # SUMO's directed edge semantics for network conversion.
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
            # Use aggressive topology cleanup and signal inference to create a
            # cleaner SUMO network from raw OSM input.
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

    def _sample_from_profile(
        self,
        count: int,
        duration_seconds: float,
        demand_profile: list[tuple[float, float]],
    ) -> list[float]:
        """Helper to scale relative profile percentages to absolute departure times."""
        # Scale relative times -> absolute seconds
        # Convert normalized demand profile fractions into actual departure times
        # and keep the matching fraction values for each segment.
        abs_profile = [
            (t_percentage * duration_seconds, f) for t_percentage, f in demand_profile
        ]

        times = [t for t, _ in abs_profile]
        fractions = [f for _, f in abs_profile]

        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(
                f"demand_profile fractions must sum to 1.0, got {sum(fractions):.6f}"
            )

        departures: list[float] = []
        for i in range(len(times)):
            t_start = times[i]
            t_end = times[i + 1] if (i + 1) < len(times) else duration_seconds

            n = round(fractions[i] * count)
            if n <= 0:
                continue

            if t_end > t_start:
                interval = (t_end - t_start) / n
                for j in range(n):
                    departures.append(t_start + j * interval)
            else:
                for _ in range(n):
                    departures.append(t_start)

        # Handle rounding errors to ensure we return exactly 'count' vehicles
        while len(departures) < count:
            departures.append(duration_seconds - 1.0)

        return sorted(departures[:count])

    def strip_node_prefix(self, node_id: str) -> str:
        """Return the raw SUMO junction ID by stripping any known role prefix."""
        NODE_ID_PREFIXES = ("origin_", "destination_", "dest_", "onramp_", "offramp_")

        for prefix in NODE_ID_PREFIXES:
            if node_id.startswith(prefix):
                return node_id[len(prefix) :]
        return node_id

    def _build_highway_trips_xml(
        self,
        departures: list[float],
        from_edges: list[str],
        to_edges: list[str],
        rng: random.Random,
    ) -> ET.Element:
        """Build a bare <routes> XML tree of <trip> elements for the highway stream.

        Each trip is randomly paired (from_edge → to_edge) using *rng*, so the
        result is fully reproducible given the same seed.  A trivial same-edge
        pairing is retried up to 20 times where possible.

        Args:
            departures: Pre-computed, sorted departure times in seconds.
            from_edges: Pool of valid departure edges (outgoing from origin nodes).
            to_edges:   Pool of valid arrival edges (incoming to dest/offramp nodes).
            rng:        Seeded :class:`random.Random` instance.

        Returns:
            An ``ET.Element`` rooted at ``<routes>`` containing ``<trip>`` children.
            The caller is responsible for writing this to disk and running
            duarouter to validate/expand the trips into full routes.
        """
        root = ET.Element("routes")
        for i, t in enumerate(departures):
            from_edge = rng.choice(from_edges)
            to_edge = rng.choice(to_edges)
            # Avoid trivial same-edge trips (not always avoidable on tiny networks)
            # by retrying random selection a few times.
            for _ in range(20):
                if to_edge != from_edge:
                    break
                to_edge = rng.choice(to_edges)

            trip = ET.SubElement(
                root, "trip", id=f"hw_{i}", depart=f"{t:.2f}", type="passenger_car"
            )
            trip.set("from", from_edge)
            trip.set("to", to_edge)
            trip.set("departPos", "0")
            trip.set("departLane", "random")
        return root

    def _validate_with_duarouter(
        self,
        trips_path: str,
        routes_out_path: str,
        seed: int,
    ) -> None:
        """Convert a raw trips file into validated routes using ``duarouter``.

        Unreachable trips are silently dropped (``--ignore-errors``) rather than
        aborting the whole run.  The caller should compare expected vs. actual
        vehicle counts in the output if strict counts matter.

        Args:
            trips_path:     Path to the input ``<routes>``/``<trip>`` XML file.
            routes_out_path: Destination path for the validated ``<routes>`` output.
            seed:           Passed to duarouter for deterministic internal routing.

        Raises:
            subprocess.CalledProcessError: If duarouter exits with a non-zero code.
        """
        # Run duarouter to expand simple trip definitions into full routes based
        # on the current SUMO network topology.
        cmd = [
            "duarouter",
            "-n",
            self.net_file,
            "--route-files",
            trips_path,
            "-o",
            routes_out_path,
            "--repair",
            "--ignore-errors",  # drop unroutable trips
            "--seed",
            str(seed),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _merge_route_files(self, paths: list[str], output_path: str) -> int:
        """Merge several route XML files into one file sorted by departure time.

        Extracts ``<vehicle>`` and ``<trip>`` elements from each source file
        (preserving nested ``<route>`` children produced by duarouter), prepends
        ``<vType>`` definitions for both ``passenger_car`` and ``urban`` vehicle
        classes, and writes the merged tree to *output_path*.

        Args:
            paths:       Ordered list of source route/trip XML file paths.
            output_path: Destination path for the merged output file.

        Returns:
            Total number of vehicle/trip elements written.
        """
        merged_root = ET.Element("routes")
        # ET.SubElement(merged_root, "vType", id="urban", vClass="passenger")
        ET.SubElement(merged_root, "vType", id="passenger_car", vClass="passenger")
        urban_vtype = ET.SubElement(
            merged_root, "vType", id="urban", vClass="passenger"
        )
        urban_vtype.set("maxSpeed", "50")
        urban_vtype.set("accel", "2.6")
        urban_vtype.set("decel", "4.5")
        urban_vtype.set("length", "5.0")
        urban_vtype.set("minGap", "2.5")
        urban_vtype.set("sigma", "0.5")

        all_vehicles: list[ET.Element] = []
        for path in paths:
            tree = ET.parse(path)
            for el in tree.getroot():
                if el.tag in ("vehicle", "trip"):
                    all_vehicles.append(el)

        all_vehicles.sort(key=lambda x: float(x.get("depart", "0")))
        merged_root.extend(all_vehicles)

        ET.ElementTree(merged_root).write(
            output_path, encoding="utf-8", xml_declaration=True
        )
        return len(all_vehicles)

    # @skip_if_exists("rou_file")

    def generate_demand(
        self,
        urban_count: int,
        duration_seconds: float,
        highway_count: int = 0,
        demand_profile: list[tuple[float, float]] | None = None,
        seed: int = 42,
    ) -> None:
        """
        Generates urban and highway demand. Highway demand is strictly filtered
         to only inject onto edges directly connected to motorway mainlines.
        """
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")

        rng = random.Random(seed)
        temp_urban_trips = os.path.join(self.output_dir, "_temp_urban_trips.xml")
        temp_urban_rou = os.path.join(self.output_dir, "_temp_urban.rou.xml")
        temp_hw_trips = os.path.join(self.output_dir, "_temp_hw_trips.xml")
        temp_hw_rou = os.path.join(self.output_dir, "_temp_hw.rou.xml")

        try:
            # --- 1. URBAN DEMAND GENERATION ---
            random_trips_script = os.path.join(
                os.environ["SUMO_HOME"], "tools", "randomTrips.py"
            )
            cmd = [
                sys.executable,
                random_trips_script,
                "-n",
                self.net_file,
                "-o",
                temp_urban_trips,
                "--route-file",
                temp_urban_rou,
                "--end",
                str(duration_seconds),
                "--period",
                str(duration_seconds / max(urban_count, 1)),
                "--validate",
                "--remove-loops",
                "--seed",
                str(seed),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Reshape urban times based on profile
            urban_tree = ET.parse(temp_urban_rou)
            urban_root = urban_tree.getroot()
            urban_elements = [el for el in urban_root if el.tag in ("vehicle", "trip")]

            if demand_profile:
                u_times = self._sample_from_profile(
                    len(urban_elements), duration_seconds, demand_profile
                )
            else:
                u_interval = duration_seconds / max(len(urban_elements), 1)
                u_times = [i * u_interval for i in range(len(urban_elements))]

            for idx, (el, t) in enumerate(zip(urban_elements, u_times)):
                el.set("depart", f"{t:.2f}")
                el.set("type", "urban")
                el.set("id", f"urban_{idx}")
                el.set("departLane", "random")

            urban_elements.sort(key=lambda x: float(x.get("depart")))  # type: ignore
            urban_root[:] = urban_elements
            urban_tree.write(temp_urban_rou, encoding="utf-8", xml_declaration=True)

            # --- 2. HIGHWAY DEMAND GENERATION (WITH MAINLINE FILTERING) ---
            routed_hw_count = 0
            if highway_count > 0:
                net_tree = ET.parse(self.net_file)
                net_root = net_tree.getroot()

                # Build topological index: node_id -> List[Edge Elements]
                node_to_out_edges = {}
                node_to_in_edges = {}

                for edge in net_root.findall("edge"):
                    if edge.get("function") == "internal":
                        continue
                    f, t = edge.get("from"), edge.get("to")
                    if f:
                        node_to_out_edges.setdefault(f, []).append(edge)
                    if t:
                        node_to_in_edges.setdefault(t, []).append(edge)

                # --- Highway origins: mainline cuts only (edges of the map) ---
                # Urban demand covers onramps; highway demand represents traffic
                # entering from outside the map on the motorway itself. Exclude
                # onramp nodes explicitly (using the arbitrator's own classification)
                # and verify the connector edge is a pure motorway, not a ramp link.
                onramp_id_set = {str(n) for n in (self.onramp_ids or [])}
                offramp_id_set = {str(n) for n in (self.offramp_ids or [])}

                verified_from_edges: list[str] = []
                origin_nodes = [
                    n for n in (self.origin_ids or []) if n.startswith("origin_")
                ]

                for node_name in origin_nodes:
                    stripped_id = self.strip_node_prefix(node_name)

                    # Onramp origins are served by urban demand. Never use them
                    # as highway entries, regardless of what lies downstream.
                    if stripped_id in onramp_id_set:
                        continue

                    for c_edge in node_to_out_edges.get(stripped_id, []):
                        etype = c_edge.get("type", "")
                        # Must sit on the motorway mainline itself, not a ramp stub.
                        if "motorway" in etype and "link" not in etype:
                            verified_from_edges.append(c_edge.get("id"))

                if not verified_from_edges:
                    print(
                        "[WARN] No mainline-cut origins found; skipping highway demand."
                    )
                    highway_count = 0

                # --- Highway destinations: mainline cuts only, symmetric to origins ---
                verified_to_edges: list[str] = []
                dest_nodes = [
                    n for n in (self.destination_ids or []) if n.startswith("dest")
                ]

                for node_name in dest_nodes:
                    stripped_id = self.strip_node_prefix(node_name)

                    # Offramp destinations feed urban streets. A highway through-trip
                    # should exit at a mainline cut, not dive off into the city.
                    if stripped_id in offramp_id_set:
                        continue

                    for e in node_to_in_edges.get(stripped_id, []):
                        etype = e.get("type", "")
                        if "motorway" in etype and "link" not in etype:
                            verified_to_edges.append(e.get("id"))

                if highway_count > 0 and (
                    not verified_from_edges or not verified_to_edges
                ):
                    raise ValueError(
                        "No valid mainline origin/destination pairs found after filtering. "
                        "Highway demand requires at least one mainline-cut origin AND one "
                        "mainline-cut destination on this network."
                    )

                # Generate Highway Departures
                if demand_profile:
                    hw_deps = self._sample_from_profile(
                        highway_count, duration_seconds, demand_profile
                    )
                else:
                    hw_interval = duration_seconds / highway_count
                    hw_deps = [i * hw_interval for i in range(highway_count)]

                # Build and Route Highway Trips
                hw_trips_root = self._build_highway_trips_xml(
                    hw_deps, verified_from_edges, verified_to_edges, rng
                )
                ET.ElementTree(hw_trips_root).write(
                    temp_hw_trips, encoding="utf-8", xml_declaration=True
                )

                self._validate_with_duarouter(temp_hw_trips, temp_hw_rou, seed)

                # Count successfully routed vehicles
                hw_tree = ET.parse(temp_hw_rou)
                routed_hw_count = sum(
                    1 for el in hw_tree.getroot() if el.tag in ("vehicle", "trip")
                )

                if routed_hw_count < highway_count:
                    print(
                        f"[WARN] Highway: {highway_count - routed_hw_count}/{highway_count} dropped (unroutable)."
                    )

            # --- 3. MERGE AND CLEANUP ---
            files_to_merge = [temp_urban_rou]
            if highway_count > 0:
                files_to_merge.append(temp_hw_rou)

            total = self._merge_route_files(files_to_merge, self.rou_file)
            print(f"[OK] Generated {total} vehicles -> {self.rou_file}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SUMO Subprocess failed: {e.stderr}") from e
        finally:
            for path in [temp_urban_trips, temp_urban_rou, temp_hw_trips, temp_hw_rou]:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def parse_demand_profile(raw: str | None) -> list[tuple[float, float]] | None:

        if raw is None:
            return None
        try:
            matrix = json.loads(raw)
            profile = [(float(row[0]), float(row[1])) for row in matrix]
        except (json.JSONDecodeError, IndexError, TypeError, ValueError):
            raise ValueError(
                f"Invalid demand profile format: '{raw}'. "
                "Expected a matrix e.g. '[[0.0,0.3],[0.3,0.5],[0.8,0.2]]'"
            )
        total = sum(f for _, f in profile)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"demand_profile fractions must sum to 1.0, got {total:.6f}"
            )
        return profile

    def create_consolidated_network(
        self,
        min_link_length: float,
        target_cell_length: float = 0.3,
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
                - offramp_ids: List of offramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - road_params: Road parameters configuration used for the network.
                - diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs.
                - backbone_node_ids: Set of node IDs that form the motorway backbone.
        """
        self.arbitrator = NetworkArbitrator(
            net_xml_path=os.path.normpath(self.net_file),
            road_params_config_path=self.road_params_config_path,
            min_link_length=min_link_length,
            target_cell_length=target_cell_length,
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

        motorway_links = [
            link
            for node in self.consolidated_network
            for link in node.outgoing
            if isinstance(link, MotorwayLink)
        ]

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
            motorway_links=motorway_links,
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

    def get_edge_lane_counts(self) -> dict[str, int]:
        """Parse the net_file to map edge IDs to their number of lanes."""
        tree = ET.parse(self.net_file)
        root = tree.getroot()
        lane_counts = {}

        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue

            eid = edge.get("id")
            if eid:
                lanes = edge.findall("lane")
                lane_counts[eid] = len(lanes)

        return lane_counts

    @staticmethod
    def bc_flow_from_density(
        rho_lane: float,
        n_lanes: int,
        road_params: RoadTypeParams,
    ) -> float:
        q_max_lane = road_params["lane_capacity"]  # veh/h/lane
        rho_jam_lane = road_params["jam_density"]  # veh/km/lane
        v_f = road_params["free_flow_speed"]  # km/h

        # Per-lane density for the FD calculation
        if rho_lane < 0.0 or rho_lane > rho_jam_lane:
            warnings.warn(
                f"Density BC value {rho_lane:.2f} veh/km/lane is out of bounds "
                f"(0 to {rho_jam_lane:.2f}). Clipping to valid range."
            )
            rho_lane_capped = max(0.0, min(rho_lane, rho_jam_lane))
        else:
            rho_lane_capped = rho_lane

        rho_c = q_max_lane / v_f  # Critical density
        w = q_max_lane / (rho_jam_lane - rho_c)  # Wave speed

        # The Hybrid Equation:
        if rho_lane_capped <= rho_c:
            # Freeflow: Boundary is wide open at capacity
            q_lane = q_max_lane
        else:
            # Congestion: Boundary flow is restricted by the FD slope
            q_lane = q_max_lane - w * (rho_lane_capped - rho_c)

        return max(0.0, q_lane) * n_lanes

    def build_destination_bc_from_sumo_edges(
        self,
        edge_data_path: str,
    ) -> tuple[dict[str, Callable], dict[str, Callable]]:
        """Build time-varying flow and density boundary conditions for each destination node.

        Parses SUMO edge data output to extract per-interval density time series for the
        edges immediately upstream of each destination. Density is aggregated across all
        upstream edges weighted by lane count, then converted to a boundary flow via the
        triangular fundamental diagram defined in road_params.

        Args:
            edge_data_path: Path to a SUMO ``<edgeData>`` output XML file containing
                per-interval density measurements (``density`` attribute on ``<edge>`` elements).

        Returns:
            A tuple of two dictionaries, both keyed by destination node ID:
                - destination_flow_bc: Maps each destination to a callable ``f(t) -> float``
                  returning total flow in veh/h at time *t* (hours).
                - destination_density_bc: Maps each destination to a callable ``f(t) -> float``
                  returning per-lane density in veh/km at time *t* (hours).

        Raises:
            ValueError: If destination_ids or road_params have not been initialized.
        """
        if self.destination_ids is None:
            raise ValueError(
                "Destination boundary conditions cannot be computed in case of missing destinations."
            )
        if self.road_params is None:
            raise ValueError("Road parameters are not initialized.")

        # Parse SUMO edge data — collect density time series per edge
        tree = ET.parse(edge_data_path)
        root = tree.getroot()
        edge_density_ts: dict[str, list[tuple[float, float]]] = {}
        for interval in root.findall("interval"):
            t_mid = (
                (float(interval.get("begin", 0)) + float(interval.get("end", 0))) / 2.0
            ) / 3600.0
            for edge_el in interval.findall("edge"):
                eid = edge_el.get("id")
                d = edge_el.get("density")
                if eid and d is not None:
                    edge_density_ts.setdefault(eid, []).append((t_mid, float(d)))

        # Build junction → incoming edges index directly from net XML
        net_tree = ET.parse(self.net_file)
        to_index: dict[str, list[str]] = {}
        for edge in net_tree.getroot().findall("edge"):
            if edge.get("function") == "internal":
                continue
            eid = edge.get("id")
            t = edge.get("to")
            if eid and t:
                to_index.setdefault(t, []).append(eid)

        lane_counts = self.get_edge_lane_counts()
        motorway_params = self.road_params.get("motorway") or next(
            iter(self.road_params.values())
        )

        destination_flow_bc: dict[str, Callable] = {}
        destination_density_bc: dict[str, Callable] = {}

        for dest_id in self.destination_ids:
            raw = self.strip_node_prefix(dest_id)
            upstream_edges = to_index.get(raw, [])

            # Aggregate density time series across all upstream edges
            ts_agg: dict[float, list[float]] = {}
            matched = []
            for eid in upstream_edges:
                if eid in edge_density_ts:
                    matched.append(eid)
                    for t, d in edge_density_ts[eid]:
                        ts_agg.setdefault(t, []).append(d)

            if not matched:
                warnings.warn(
                    f"No density data found for any upstream edges of destination '{dest_id}'. "
                    "Using fallback constant BCs: density=10 veh/km, flow=capacity."
                )
                destination_density_bc[dest_id] = lambda t: 10.0
                destination_flow_bc[dest_id] = (
                    lambda t: motorway_params["lane_capacity"]
                    * sum(lane_counts.values())
                    / len(lane_counts)
                )
                continue

            t_vals = sorted(ts_agg.keys())
            total_lanes = sum(lane_counts.get(eid, 1) for eid in matched)
            d_vals_per_lane = [sum(ts_agg[t]) / total_lanes for t in t_vals]

            # Density BC — per lane, interpolated
            def get_rho_lane(t, _t=t_vals, _d=d_vals_per_lane) -> float:
                return float(np.interp(t, _t, _d))

            destination_density_bc[dest_id] = get_rho_lane

            # Flow BC — per link total, derived from FD
            def get_q_total(
                t, _rho_fn=get_rho_lane, lanes=total_lanes, _params=motorway_params
            ) -> float:
                return self.bc_flow_from_density(_rho_fn(t), lanes, _params)

            destination_flow_bc[dest_id] = get_q_total

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
        associated metadata. This method should be called after
        create_consolidated_network() has been executed.

        Returns:
            A tuple containing:
                - consolidated_network: Network object representing the macroscopic network.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - offramp_ids: List of offramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - road_params: Road parameters configuration used for the network.
                - diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs.
                - backbone_node_ids: Set of node IDs that form the motorway backbone.

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
