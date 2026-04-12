import os
import sys
import json
import shutil
import random
import logging
import subprocess
import osmnx as ox
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import json
import warnings
from functools import wraps
from typing import Optional, Tuple, Callable

from traffic_flow_models.network import Network, Origin, Destination
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.turning_rate_aggregator import TurningRateAggregator
from traffic_flow_models.arbitrator.network_arbitrator import (
    NetworkArbitrator,
    RoadParamsConfig,
)
from traffic_flow_models.network.network import (
    Network,
    Destination,
    MotorwayLink,
    Offramp,
)
from traffic_flow_models import Simulation
import logging

logging.basicConfig(level=logging.DEBUG)


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

    # ---------------------------------------------------------------------------
    # Sentinel suffixes for synthetic ramp edges injected by netconvert --ramps.guess
    # These edges model ramp geometry and must never be used as backbone
    # inflow / outflow edges.
    # ---------------------------------------------------------------------------
    SYNTHETIC_RAMP_SUFFIXES: tuple[str, ...] = (
        "-AddedOffRampEdge",
        "-AddedOnRampEdge",
    )

    def _is_synthetic_ramp(edge_id: str) -> bool:
        """Return True if *edge_id* was synthesised by netconvert's ramp guesser."""
        return any(edge_id.endswith(s) for s in SUMOPipeline.SYNTHETIC_RAMP_SUFFIXES)

    # ============================================================================
    # generate_demand
    # ============================================================================

    def generate_demand(
        self,
        vehicle_count: int,
        duration_seconds: float,
        backbone_vehicle_count: int = 0,
        demand_profile: list[tuple[float, float]] | None = None,
        seed: int = 42,
    ) -> None:
        """Generate traffic demand combining random trips and backbone-direct demand.

        Both random trips and backbone vehicles share the same demand_profile,
        producing a physically consistent time-of-day demand shape across the
        full network.

        Args:
            vehicle_count: Number of random vehicles to generate across the
                full SUMO network.
            duration_seconds: Simulation duration in seconds.
            backbone_vehicle_count: Number of additional vehicles to place
                directly on backbone origin inflow edges. Default 0 (disabled).
                These vehicles model highway through-traffic entering the zone
                from outside; they are injected on plain motorway fringe edges only
                and must NOT be placed on on-ramp or synthetic ramp edges.
            demand_profile: Piecewise-linear list of (time_percentage, fraction)
                pairs shaping vehicle departures. Fractions must sum to 1.0.
                None = uniform distribution.
                Example: [(0.2, 0.3), (0.3, 0.5), (0.5, 0.2)].
            seed: Random seed passed to both randomTrips *and* duarouter (via
                --duarouter-option) to make the entire OD + routing step
                fully reproducible across runs.

        Raises:
            ValueError: If backbone_vehicle_count > 0 but consolidated_network
                is not initialised.
            ValueError: If demand_profile fractions do not sum to 1.0.
        """
        logger = logging.getLogger(__name__)

        if "SUMO_HOME" not in os.environ:
            print("Error: Please set the 'SUMO_HOME' environment variable.")
            return

        # Reproducible RNG for backbone edge selection.
        rng = random.Random(seed)

        # ------------------------------------------------------------------
        # Helper: sample departure times from a piecewise-linear profile
        # ------------------------------------------------------------------
        def sample_departure_times(count: int) -> list[float]:
            if demand_profile is None:
                interval = duration_seconds / count
                return [i * interval for i in range(count)]

            abs_profile = [(t_pct * duration_seconds, f) for t_pct, f in demand_profile]
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

            while len(departures) < count:
                departures.append(duration_seconds - 1.0)
            departures = sorted(departures[:count])

            logger.debug(
                "Sampled %d departure times from demand_profile (first=%.2f, last=%.2f)",
                len(departures),
                departures[0],
                departures[-1],
            )
            return departures

        # ------------------------------------------------------------------
        # 1. Generate random trips via randomTrips.py
        #
        # IMPORTANT — reproducibility:
        #   --seed seeds randomTrips' own OD-pair sampling.
        #   --duarouter-option --seed=N seeds the duarouter call that
        #   --validate triggers internally. Without the second flag, two runs
        #   with the same --seed can still produce different route files.
        # ------------------------------------------------------------------
        random_trips_script = os.path.join(
            os.environ["SUMO_HOME"], "tools", "randomTrips.py"
        )

        cmd = [
            sys.executable,
            random_trips_script,
            "-n",
            self.net_file,
            "-o",
            "temp_trips.xml",
            "--route-file",
            self.rou_file,
            "--end",
            str(duration_seconds),
            "--period",
            str(duration_seconds / vehicle_count),
            "--fringe-factor",
            "10",
            "--validate",
            "--remove-loops",
            "--seed",
            str(seed),
            # Seed duarouter (called internally by --validate) for full reproducibility.
            "--duarouter-option",
            f"--seed={seed}",
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
        # 2. Post-process: overwrite random-trip depart times with profile times
        # ------------------------------------------------------------------
        random_departures = sample_departure_times(vehicle_count)

        existing_tree = ET.parse(self.rou_file)
        existing_root = existing_tree.getroot()

        random_elements = [
            el
            for el in existing_root
            if el.tag in ("vehicle", "trip")
            and not el.get("id", "").startswith("backbone_")
        ]

        if len(random_elements) != len(random_departures):
            logger.warning(
                "randomTrips produced %d elements but expected %d — "
                "profile reshaping may be approximate.",
                len(random_elements),
                vehicle_count,
            )
            random_departures = sample_departure_times(len(random_elements))

        for el, t in zip(random_elements, random_departures):
            el.set("depart", f"{t:.2f}")

        logger.debug(
            "Rewrote depart times for %d random vehicles using demand_profile=%s",
            len(random_elements),
            "uniform" if demand_profile is None else demand_profile,
        )

        # ------------------------------------------------------------------
        # 3. Optionally inject backbone-direct vehicles
        #
        # Goal: vehicles that model highway through-traffic entering the zone
        # from outside. They must depart from a non-motorway fringe edge that
        # leads directly into a true motorway (not an on-ramp / motorway_link),
        # and arrive at an equivalent fringe edge on the other side.
        #
        # Two-pass approach:
        #   Pass A — identify which backbone origin/destination nodes are
        #            connected to a *plain* motorway (not motorway_link, not
        #            a synthetic ramp edge). This filters out on-ramp nodes.
        #   Pass B — for those filtered nodes, find the actual SUMO fringe
        #            edges (non-motorway, non-synthetic) used for trip endpoints.
        # ------------------------------------------------------------------
        if backbone_vehicle_count > 0:
            if self.consolidated_network is None:
                raise ValueError(
                    "consolidated_network is not initialised. "
                    "Call create_consolidated_network() before using backbone_vehicle_count > 0."
                )

            # Collect backbone origin / destination node IDs from the macroscopic network.
            # Origin link → its destination_node_id is the first backbone node.
            # Destination link → its origin_node_id is the last backbone node.
            backbone_origin_node_ids: set[str] = set()
            backbone_destination_node_ids: set[str] = set()

            for node in self.consolidated_network:
                for link in node.incoming:
                    if isinstance(link, Origin):
                        if link.destination_node_id is None:
                            raise ValueError(
                                f"Backbone origin link {link.id} missing destination_node_id"
                            )
                        backbone_origin_node_ids.add(link.destination_node_id)

                for link in node.outgoing:
                    if isinstance(link, Destination):
                        if link.origin_node_id is None:
                            raise ValueError(
                                f"Backbone destination link {link.id} missing origin_node_id"
                            )
                        backbone_destination_node_ids.add(link.origin_node_id)

            logger.debug(
                "Backbone origin nodes (all): %d, destination nodes (all): %d",
                len(backbone_origin_node_ids),
                len(backbone_destination_node_ids),
            )

            tree = ET.parse(self.net_file)
            root = tree.getroot()

            # ------------------------------------------------------------------
            # Pass A — Origins:
            # Keep only origin nodes from which at least one outgoing edge is a
            # plain motorway (contains "motorway" but NOT "link") and is not a
            # synthetic ramp edge.  On-ramp entry nodes are excluded here because
            # their outgoing edge type is "highway.motorway_link".
            # ------------------------------------------------------------------
            direct_backbone_origin_node_ids: set[str] = set()
            for edge in root.findall("edge"):
                if edge.get("function") == "internal":
                    continue
                edge_id = edge.get("id", "")
                if SUMOPipeline._is_synthetic_ramp(edge_id):
                    continue
                edge_type = edge.get("type", "").lower()
                from_node = edge.get("from")
                is_plain_motorway = "motorway" in edge_type and "link" not in edge_type
                if is_plain_motorway and from_node in backbone_origin_node_ids:
                    direct_backbone_origin_node_ids.add(from_node)

            logger.debug(
                "Direct-mainline backbone origin nodes: %d / %d "
                "(%d on-ramp origin nodes excluded)",
                len(direct_backbone_origin_node_ids),
                len(backbone_origin_node_ids),
                len(backbone_origin_node_ids) - len(direct_backbone_origin_node_ids),
            )

            # ------------------------------------------------------------------
            # Pass A — Destinations:
            # Mirror the origin logic: keep only destination nodes where at least
            # one *incoming* edge is a plain motorway (not motorway_link, not
            # synthetic). This excludes off-ramp exit nodes whose last incoming
            # edge is a motorway_link.
            # ------------------------------------------------------------------
            direct_backbone_destination_node_ids: set[str] = set()
            for edge in root.findall("edge"):
                if edge.get("function") == "internal":
                    continue
                edge_id = edge.get("id", "")
                if SUMOPipeline._is_synthetic_ramp(edge_id):
                    continue
                edge_type = edge.get("type", "").lower()
                to_node = edge.get("to")
                is_plain_motorway = "motorway" in edge_type and "link" not in edge_type
                if is_plain_motorway and to_node in backbone_destination_node_ids:
                    direct_backbone_destination_node_ids.add(to_node)

            logger.debug(
                "Direct-mainline backbone destination nodes: %d / %d "
                "(%d off-ramp destination nodes excluded)",
                len(direct_backbone_destination_node_ids),
                len(backbone_destination_node_ids),
                len(backbone_destination_node_ids)
                - len(direct_backbone_destination_node_ids),
            )

            # ------------------------------------------------------------------
            # Pass B — Find fringe inflow / outflow edges.
            #
            # Inflow edge:  non-motorway, non-synthetic edge whose *to* node is
            #               a direct-mainline origin node.  This is the access
            #               road from outside the zone that leads onto the
            #               motorway at the boundary.
            # Outflow edge: non-motorway, non-synthetic edge whose *from* node is
            #               a direct-mainline destination node.  Symmetric exit.
            #
            # Note: "not is_motorway" here uses the broad check
            # ("motorway" in edge_type) intentionally — we want the fringe
            # connector, never the motorway itself.  Synthetic edges are caught
            # by the explicit suffix guard before the type check.
            # ------------------------------------------------------------------
            origin_inflow_edges: list[str] = []
            destination_outflow_edges: list[str] = []

            for edge in root.findall("edge"):
                if edge.get("function") == "internal":
                    continue

                edge_id = edge.get("id", "")
                if not edge_id:
                    raise ValueError("Edge without ID found in SUMO network XML.")

                # Skip synthetic ramp edges regardless of anything else.
                if SUMOPipeline._is_synthetic_ramp(edge_id):
                    logger.debug("Pass B: skipping synthetic ramp edge '%s'", edge_id)
                    continue

                edge_type = edge.get("type", "").lower()
                from_node = edge.get("from")
                to_node = edge.get("to")
                is_motorway = "motorway" in edge_type  # broad: covers plain + link

                if not is_motorway and to_node in direct_backbone_origin_node_ids:
                    origin_inflow_edges.append(edge_id)
                    logger.debug(
                        "Backbone inflow edge: '%s' (type='%s') → node '%s'",
                        edge_id,
                        edge_type,
                        to_node,
                    )

                if (
                    not is_motorway
                    and from_node in direct_backbone_destination_node_ids
                ):
                    destination_outflow_edges.append(edge_id)
                    logger.debug(
                        "Backbone outflow edge: '%s' (type='%s') ← node '%s'",
                        edge_id,
                        edge_type,
                        from_node,
                    )

            if not origin_inflow_edges:
                raise ValueError(
                    "No backbone inflow edges found in SUMO network. "
                    "Check that the SUMO network and consolidated network are consistent, "
                    "and that --ramps.guess was used during netconvert."
                )
            if not destination_outflow_edges:
                raise ValueError(
                    "No backbone outflow edges found in SUMO network. "
                    "Check that the SUMO network and consolidated network are consistent."
                )

            logger.debug(
                "Pass B complete: %d inflow edges, %d outflow edges",
                len(origin_inflow_edges),
                len(destination_outflow_edges),
            )

            # Sample backbone departure times from the same profile as random trips.
            backbone_departures = sample_departure_times(backbone_vehicle_count)
            backbone_trips: list[ET.Element] = []

            for i, depart_time in enumerate(backbone_departures):
                from_edge = rng.choice(origin_inflow_edges)
                to_edge = rng.choice(destination_outflow_edges)

                # Avoid trivially same-edge OD pairs.
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
                "Generated %d backbone trips (profile=%s)",
                backbone_vehicle_count,
                "uniform" if demand_profile is None else "custom",
            )

            for trip in backbone_trips:
                existing_root.append(trip)

        # ------------------------------------------------------------------
        # 4. Re-sort all elements by depart time and write final .rou.xml
        # ------------------------------------------------------------------
        children = list(existing_root)
        children.sort(key=lambda el: float(el.get("depart", "0")))
        existing_root[:] = children

        ET.indent(existing_root, space="  ")
        existing_tree.write(self.rou_file, encoding="utf-8", xml_declaration=True)

        total = vehicle_count + backbone_vehicle_count
        print(
            f"{self.rou_file} finalised — {vehicle_count} random + "
            f"{backbone_vehicle_count} backbone = {total} total vehicles. "
            f"Profile: {'uniform' if demand_profile is None else 'custom'}."
        )

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

    # ============================================================================
    # build_destination_boundary_conditions
    # ============================================================================

    def build_destination_boundary_conditions(
        self,
        backbone_state_path: str,
    ) -> tuple[
        dict[str, Callable[[float], float]], dict[str, Callable[[float], float]]
    ]:
        """Build time-varying flow and density boundary conditions for all destinations.

        For each destination node the function resolves which upstream MotorwayLink
        to read from, then builds a linear interpolant over the backbone simulation's
        time array.

        Resolution rule (Offramp checked BEFORE direct MotorwayLink):
        1. If the host node has an Offramp in its incoming links, walk back
            through the offramp to its origin node and take the MotorwayLink
            feeding *that* node.  This gives the pre-diverge (pre-split) flow.
        2. If the host node has a direct MotorwayLink in its incoming links
            (no Offramp), use that link directly.

        Offramp is checked first so that, when a node sits at the end of an
        off-ramp that branches from a diverge, we read the backbone link *before*
        the split rather than the (shorter, lower-flow) off-ramp link.

        The last spatial cell of the resolved link is used for both flow and
        density, since that cell is closest to the downstream boundary.

        Args:
            backbone_state_path: Path to the backbone_state.json file produced
                by a prior macroscopic simulation step.

        Returns:
            (destination_flow_bc, destination_density_bc): two dicts mapping
            destination IDs to callables ``f(t_hours) -> float``.
        """
        logger = logging.getLogger(__name__)

        _FALLBACK_FLOW: float = 6000.0  # veh/h — used when no data is available
        _FALLBACK_DENSITY: float = 10.0  # veh/km — used when no data is available

        if self.consolidated_network is None:
            raise ValueError(
                "consolidated_network is not initialised. "
                "Call create_consolidated_network() before "
                "build_destination_boundary_conditions()."
            )

        if self.destination_ids is None:
            raise ValueError(
                "destination_ids is None. "
                "Call create_consolidated_network() before "
                "build_destination_boundary_conditions()."
            )

        # ------------------------------------------------------------------
        # Load backbone state
        # ------------------------------------------------------------------
        with open(backbone_state_path, "r", encoding="utf-8") as fh:
            backbone_state = json.load(fh)

        time_array: list[float] = backbone_state["time_array"]
        flows_ts: dict[str, list[list[float]]] = backbone_state["state_time_series"][
            "flows"
        ]
        densities_ts: dict[str, list[list[float]]] = backbone_state[
            "state_time_series"
        ]["densities"]
        t_arr = np.array(time_array)
        n_steps = len(time_array)

        logger.debug(
            "backbone_state loaded: %d time steps, %d flow links, %d density links",
            n_steps,
            len(flows_ts),
            len(densities_ts),
        )

        # Warn if most of the backbone state is zeros — this is the likely root
        # cause of very low BC values and is worth surfacing loudly.
        for link_id, rows in flows_ts.items():
            non_zero_steps = sum(1 for row in rows if row and any(v > 0.0 for v in row))
            if non_zero_steps < n_steps * 0.1:
                logger.debug(
                    "backbone_state WARNING: link '%s' has only %d / %d non-zero "
                    "time steps — BCs derived from this link will be near-zero "
                    "for most of the simulation. Check that vehicles remain in "
                    "the network throughout the simulation, not just at t=0.",
                    link_id,
                    non_zero_steps,
                    n_steps,
                )

        node_by_id = {node.id: node for node in self.consolidated_network}

        # ------------------------------------------------------------------
        # _last_cell_series
        #
        # Extract the value of the *last spatial cell* at every time step for
        # a given link.  The last cell is the one at the downstream end of the
        # link, closest to the destination boundary.
        #
        # Empty rows (no cells recorded for that time step) are replaced with
        # 0.0 rather than skipped, so the returned series always has exactly
        # len(time_array) entries.  Skipping them would produce a shorter
        # series which _make_interp would silently treat as "no data", falling
        # back to the constant fallback value.
        # ------------------------------------------------------------------
        def _last_cell_series(
            link_id: str,
            ts: dict[str, list[list[float]]],
        ) -> list[float] | None:
            rows = ts.get(link_id)
            if not rows:
                return None
            # Preserve zeros for empty rows — do NOT use `if row` filter.
            extracted = [row[-1] if row else 0.0 for row in rows]
            return extracted

        # ------------------------------------------------------------------
        # _make_interp
        #
        # Build a linear interpolant over time_array.  If the series is
        # missing or has a length mismatch, fall back to a constant function
        # and log the reason so callers can detect the issue.
        # ------------------------------------------------------------------
        def _make_interp(
            series: list[float] | None,
            fallback: float,
            label: str = "",
        ) -> Callable[[float], float]:
            if series is None:
                logger.debug(
                    "_make_interp [%s]: series is None — using constant fallback %.2f",
                    label,
                    fallback,
                )
                return lambda _t, _f=fallback: _f
            if len(series) != n_steps:
                logger.debug(
                    "_make_interp [%s]: series length %d != time_array length %d "
                    "— using constant fallback %.2f",
                    label,
                    len(series),
                    n_steps,
                    fallback,
                )
                return lambda _t, _f=fallback: _f
            v_arr = np.array(series)
            return lambda t, _t=t_arr, _v=v_arr: float(np.interp(t, _t, _v))

        # ------------------------------------------------------------------
        # Main loop — resolve upstream link for each destination
        # ------------------------------------------------------------------
        destination_flow_bc: dict[str, Callable[[float], float]] = {}
        destination_density_bc: dict[str, Callable[[float], float]] = {}

        for dest_id in self.destination_ids:
            host_node_id = dest_id.removeprefix("dest_")
            host_node = node_by_id.get(host_node_id)

            if host_node is None:
                raise ValueError(
                    f"[build_destination_bc] Host node '{host_node_id}' for destination "
                    f"'{dest_id}' not found in consolidated network."
                )

            upstream_link: MotorwayLink | None = None
            link_source: str = "unknown"

            # ----------------------------------------------------------------
            # Traversal — Offramp checked FIRST.
            #
            # Rationale: a destination that sits at the end of an off-ramp must
            # read from the pre-diverge backbone link (the MotorwayLink that
            # feeds the diverge node where the off-ramp branches off).  If we
            # checked MotorwayLink first, we might accidentally pick up a short
            # post-diverge highway segment that also terminates at the same node,
            # producing a lower (already-split) flow reading.
            # ----------------------------------------------------------------
            offramp_incoming: Offramp | None = None
            motorway_incoming: MotorwayLink | None = None

            for incoming in host_node.incoming:
                if isinstance(incoming, Offramp) and offramp_incoming is None:
                    offramp_incoming = incoming
                elif isinstance(incoming, MotorwayLink) and motorway_incoming is None:
                    motorway_incoming = incoming

            if offramp_incoming is not None:
                # Case 1: destination is at the end of an off-ramp.
                # Walk back to the MotorwayLink that feeds the diverge node.
                logger.debug(
                    "dest '%s' → Offramp '%s', walking upstream to backbone diverge node",
                    dest_id,
                    offramp_incoming.id,
                )
                offramp_origin = node_by_id.get(offramp_incoming.origin_node_id)
                if offramp_origin is None:
                    raise ValueError(
                        f"[build_destination_bc] Offramp '{offramp_incoming.id}' origin node "
                        f"'{offramp_incoming.origin_node_id}' not found in consolidated network "
                        f"(destination '{dest_id}')."
                    )

                for upstream in offramp_origin.incoming:
                    if isinstance(upstream, MotorwayLink):
                        upstream_link = upstream
                        link_source = f"backbone (via offramp '{offramp_incoming.id}')"
                        logger.debug(
                            "dest '%s' → Offramp '%s' → MotorwayLink '%s' (pre-diverge)",
                            dest_id,
                            offramp_incoming.id,
                            upstream.id,
                        )
                        break

                if upstream_link is None:
                    raise ValueError(
                        f"[build_destination_bc] No upstream MotorwayLink found behind "
                        f"offramp '{offramp_incoming.id}' at diverge node "
                        f"'{offramp_incoming.origin_node_id}' (destination '{dest_id}')."
                    )

            elif motorway_incoming is not None:
                # Case 2: destination is directly at the end of a MotorwayLink
                # (no off-ramp involved).
                upstream_link = motorway_incoming
                link_source = "backbone (direct)"
                logger.debug(
                    "dest '%s' → direct MotorwayLink '%s'",
                    dest_id,
                    motorway_incoming.id,
                )

            else:
                raise ValueError(
                    f"[build_destination_bc] Could not resolve upstream MotorwayLink for "
                    f"destination '{dest_id}': host node '{host_node_id}' has neither an "
                    f"Offramp nor a MotorwayLink in its incoming links."
                )

            # ----------------------------------------------------------------
            # Extract time series and build interpolants
            # ----------------------------------------------------------------
            flow_series = _last_cell_series(upstream_link.id, flows_ts)
            density_series = _last_cell_series(upstream_link.id, densities_ts)

            if flow_series is None:
                warnings.warn(
                    f"[build_destination_bc] Link '{upstream_link.id}' ({link_source}) "
                    f"has no detector coverage in backbone state for destination '{dest_id}'. "
                    f"Flow BC will use fallback constant {_FALLBACK_FLOW} veh/h."
                )
            if density_series is None:
                warnings.warn(
                    f"[build_destination_bc] Link '{upstream_link.id}' ({link_source}) "
                    f"has no detector coverage in backbone state for destination '{dest_id}'. "
                    f"Density BC will use fallback constant {_FALLBACK_DENSITY} veh/km."
                )

            if flow_series is not None:
                non_zero = sum(1 for v in flow_series if v > 0)
                logger.debug(
                    "dest '%s' link '%s' flow series: %d steps, %d non-zero, "
                    "min=%.1f max=%.1f mean=%.1f veh/h",
                    dest_id,
                    upstream_link.id,
                    len(flow_series),
                    non_zero,
                    min(flow_series),
                    max(flow_series),
                    sum(flow_series) / len(flow_series),
                )
            if density_series is not None:
                non_zero = sum(1 for v in density_series if v > 0)
                logger.debug(
                    "dest '%s' link '%s' density series: %d steps, %d non-zero, "
                    "min=%.4f max=%.4f mean=%.4f veh/km",
                    dest_id,
                    upstream_link.id,
                    len(density_series),
                    non_zero,
                    min(density_series),
                    max(density_series),
                    sum(density_series) / len(density_series),
                )

            flow_label = f"{dest_id}/flow/{upstream_link.id}"
            density_label = f"{dest_id}/density/{upstream_link.id}"

            destination_flow_bc[dest_id] = _make_interp(
                flow_series, _FALLBACK_FLOW, label=flow_label
            )
            destination_density_bc[dest_id] = _make_interp(
                density_series, _FALLBACK_DENSITY, label=density_label
            )

            # Spot-check the interpolant at 5 evenly spaced times so that
            # near-zero BCs are immediately visible in the debug log.
            if t_arr.size > 0:
                sample_times = np.linspace(t_arr[0], t_arr[-1], 5)
                logger.debug(
                    "dest '%s' BC spot-check — flow (veh/h): %s | density (veh/km): %s",
                    dest_id,
                    [f"{destination_flow_bc[dest_id](t):.1f}" for t in sample_times],
                    [f"{destination_density_bc[dest_id](t):.3f}" for t in sample_times],
                )

            print(
                f"  [dest BC] '{dest_id}' ← link '{upstream_link.id}' ({link_source})"
            )

        logger.debug(
            "build_destination_boundary_conditions complete: %d destinations resolved",
            len(destination_flow_bc),
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
