import os
import sys
import json
import shutil
import random
import json
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

from traffic_flow_models.network import Network, MotorwayLink, Offramp
from traffic_flow_models.arbitrator.loop_detector_generator import LoopDetectorGenerator
from traffic_flow_models.arbitrator.turning_rate_aggregator import TurningRateAggregator
from traffic_flow_models.arbitrator.network_arbitrator import (
    NetworkArbitrator,
    RoadParamsConfig,
)


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

    def _sample_from_profile(
        self,
        count: int,
        duration_seconds: float,
        demand_profile: list[tuple[float, float]],
    ) -> list[float]:
        """Helper to scale relative profile percentages to absolute departure times."""
        # Scale relative times -> absolute seconds
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

    # @skip_if_exists('rou_file')
    # def generate_demand(
    #     self,
    #     vehicle_count: int,
    #     duration_seconds: float,
    #     demand_profile: list[tuple[float, float]] | None = None,
    #     seed: int = 42,
    # ) -> None:
    #     """
    #     Generates demand using randomTrips.py and reshapes timing to match a profile.
    #     """
    #     if "SUMO_HOME" not in os.environ:
    #         print("Error: Please set the 'SUMO_HOME' environment variable.")
    #         return

    #     random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

    #     # 1. Run randomTrips to generate VALID routes
    #     # We use a uniform period here just to get the routes created
    #     cmd = [
    #         sys.executable,
    #         random_trips,
    #         "-n", self.net_file,
    #         "-o", "temp_trips.xml", # We keep trips separate from routes
    #         "--route-file", self.rou_file,
    #         "--end", str(duration_seconds),
    #         "--period", str(duration_seconds / vehicle_count),
    #         "--fringe-factor", "10",  # Prioritizes motorway boundaries
    #         "--validate",             # Ensures every trip is physically possible
    #         "--remove-loops",
    #         "--seed", str(seed),
    #     ]

    #     try:
    #         subprocess.run(cmd, check=True)

    #         # 2. Reshape departure times to match your Profile
    #         if demand_profile:
    #             tree = ET.parse(self.rou_file)
    #             root = tree.getroot()

    #             # Find all vehicle/trip elements
    #             elements = [el for el in root if el.tag in ("vehicle", "trip")]

    #             # Sample new times from your profile helper
    #             new_times = self._sample_from_profile(len(elements), duration_seconds, demand_profile)

    #             # Update the XML elements with the profile-based times
    #             for el, t in zip(elements, new_times):
    #                 el.set("depart", f"{t:.2f}")
    #                 # Optional: Force a standard car type to avoid permission errors
    #                 el.set("type", "passenger_car")

    #             # Add the vType definition at the top
    #             vtype = ET.Element("vType", id="passenger_car", vClass="passenger")
    #             root.insert(0, vtype)

    #             # Sort by departure time (Required for SUMO)
    #             elements.sort(key=lambda x: float(x.get("depart")))
    #             root[:] = [vtype] + elements

    #             tree.write(self.rou_file, encoding="utf-8", xml_declaration=True)

    #         print(f"Successfully generated {vehicle_count} validated trips.")

    #         # Cleanup
    #         if os.path.exists("temp_trips.xml"):
    #             os.remove("temp_trips.xml")

    #     except subprocess.CalledProcessError as e:
    #         print(f"An error occurred while generating demand: {e}")

    def _strip_node_prefix(self, node_id: str) -> str:
        """Return the raw SUMO junction ID by stripping any known role prefix."""
        NODE_ID_PREFIXES = ("origin_", "destination_", "dest_", "onramp_", "offramp_")

        for prefix in NODE_ID_PREFIXES:
            if node_id.startswith(prefix):
                return node_id[len(prefix) :]
        return node_id

    def _get_fringe_edges(
        self,
        node_ids: list[str],
        direction: str,
        *,
        allow_reverse_fallback: bool = False,
    ) -> list[str]:
        """Collect edge IDs in the SUMO net adjacent to the given node IDs.

        Handles three kinds of entries in *node_ids*:

        1. **Prefixed node IDs** (e.g. ``'origin_177009495'``, ``'dest_260747056'``) —
        the prefix is stripped and the raw ID is looked up as a SUMO junction.
        2. **Raw junction IDs** (plain numerics like ``'298859194'``) — looked up
        directly as junctions.
        3. **Raw SUMO edge IDs** (contain ``'#'`` or match an edge in the net,
        e.g. ``'210731931#0'``) — used as-is without any junction lookup.

        For junction-based lookups the preferred *direction* is tried first; if
        nothing is found and *allow_reverse_fallback* is ``True``, the opposite
        direction is attempted and a warning is printed.

        Internal junction edges (``function="internal"``) are always skipped.

        Args:
            node_ids: Mixed list of prefixed node IDs, raw junction IDs, or raw
                edge IDs — as stored by NetworkArbitrator.
            direction: Preferred search direction for junction-based lookups.
                    ``'from'`` → outgoing edges; ``'to'`` → incoming edges.
            allow_reverse_fallback: Retry with the opposite direction when the
                preferred direction yields nothing for a junction.

        Returns:
            Deduplicated list of resolved edge IDs.
        """
        tree = ET.parse(self.net_file)
        root = tree.getroot()

        # Build junction→edges index and a set of all known edge IDs in one pass.
        from_index: dict[str, list[str]] = {}
        to_index: dict[str, list[str]] = {}
        all_edge_ids: set[str] = set()
        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            eid = edge.get("id")
            if not eid:
                continue
            all_edge_ids.add(eid)
            f = edge.get("from")
            t = edge.get("to")
            if f:
                from_index.setdefault(f, []).append(eid)
            if t:
                to_index.setdefault(t, []).append(eid)

        primary_index = from_index if direction == "from" else to_index
        fallback_index = to_index if direction == "from" else from_index
        fallback_dir = "to" if direction == "from" else "from"

        edges: list[str] = []
        seen: set[str] = set()
        fallback_nodes: list[str] = []
        direct_edge_nodes: list[str] = []

        for node_id in node_ids:
            raw = self._strip_node_prefix(node_id)

            # Case 3: stripped ID is itself a valid SUMO edge — use it directly.
            if raw in all_edge_ids:
                direct_edge_nodes.append(raw)
                if raw not in seen:
                    seen.add(raw)
                    edges.append(raw)
                continue

            # Cases 1 & 2: treat as junction ID and look up adjacent edges.
            found = primary_index.get(raw, [])
            if not found and allow_reverse_fallback:
                found = fallback_index.get(raw, [])
                if found:
                    fallback_nodes.append(raw)
            for eid in found:
                if eid not in seen:
                    seen.add(eid)
                    edges.append(eid)

        if direct_edge_nodes:
            print(
                f"[INFO] _get_fringe_edges: {len(direct_edge_nodes)} ID(s) resolved "
                f"as direct edge references (not junctions): {direct_edge_nodes}"
            )
        if fallback_nodes:
            print(
                f"[WARN] _get_fringe_edges(direction='{direction}'): "
                f"{len(fallback_nodes)} junction(s) had no '{direction}' edge — "
                f"used '{fallback_dir}' fallback: {fallback_nodes}"
            )

        return edges

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
            for _ in range(20):
                if to_edge != from_edge:
                    break
                to_edge = rng.choice(to_edges)

            trip = ET.SubElement(
                root, "trip", id=f"hw_{i}", depart=f"{t:.2f}", type="passenger_car"
            )
            trip.set("from", from_edge)
            trip.set("to", to_edge)
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
            # Seed duarouter (called internally by --validate) for full reproducibility.
            # "--duarouter-option",
            # f"--seed={seed}",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _merge_route_files(self, paths: list[str], output_path: str) -> int:
        """Merge several route XML files into one file sorted by departure time.

        Extracts ``<vehicle>`` and ``<trip>`` elements from each source file
        (preserving nested ``<route>`` children produced by duarouter), prepends
        a single ``<vType id="passenger_car">`` definition, and writes the merged
        tree to *output_path*.

        Args:
            paths:       Ordered list of source route/trip XML file paths.
            output_path: Destination path for the merged output file.

        Returns:
            Total number of vehicle/trip elements written.
        """
        merged_root = ET.Element("routes")
        vtype = ET.SubElement(
            merged_root, "vType", id="passenger_car", vClass="passenger"
        )

        all_vehicles: list[ET.Element] = []
        for path in paths:
            tree = ET.parse(path)
            for el in tree.getroot():
                if el.tag in ("vehicle", "trip"):
                    all_vehicles.append(el)

        all_vehicles.sort(key=lambda x: float(x.get("depart", "0")))
        merged_root[:] = [vtype] + all_vehicles

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
        """Generate two-stream SUMO demand and write a single merged route file.

        **Urban stream** (always active)
            Uses ``randomTrips.py`` to produce topologically valid random trips
            across the full road network, biased toward motorway fringes.
            Departure times are reshaped to match *demand_profile* if supplied.

        **Highway stream** (enabled when ``highway_count > 0``)
            Injects vehicles directly at motorway fringe *origin* nodes
            (``self.origin_ids``) and routes them to any valid *destination* or
            *offramp* node (``self.destination_ids ∪ self.offramp_ids``).  This
            simulates through traffic entering the map from outside the modelled
            area.  Trips are randomly paired per the seeded RNG and then validated
            with ``duarouter``; unreachable pairings are silently dropped.

        Both streams share the same *seed* and *demand_profile*, so their temporal
        shapes are identical.  The two streams are merged into a single
        ``self.rou_file``, sorted by departure time, with a single
        ``<vType id="passenger_car">`` definition at the top.

        Vehicle IDs are namespaced: ``urban_<n>`` and ``hw_<n>`` to avoid
        collisions when SUMO reads the merged file.

        Args:
            urban_count:     Number of vehicles in the urban random-trip stream.
            duration_seconds: Total simulation window in seconds.
            highway_count:   Number of vehicles in the highway fringe stream.
                            Pass ``0`` (default) to disable the highway stream.
            demand_profile:  Piecewise-linear temporal profile as a list of
                            ``(relative_time, fraction)`` pairs.  Times must be
                            in ``[0.0, 1.0]``; fractions must sum to ``1.0``.
                            Example: ``[(0.0, 0.3), (0.3, 0.5), (0.8, 0.2)]``.
                            Pass ``None`` for a uniform distribution.
            seed:            Integer seed for reproducibility.  Controls both
                            ``randomTrips.py`` routing and the RNG used for
                            highway origin→destination pairing.

        Raises:
            EnvironmentError: If the ``SUMO_HOME`` environment variable is not set.
            ValueError: If ``highway_count > 0`` but ``origin_ids``,
                        ``destination_ids``, or ``offramp_ids`` have not been
                        populated (i.e. ``create_consolidated_network()`` has not
                        been called).
            ValueError: If no valid departure or arrival edges can be resolved
                        from the given origin/destination node IDs.
            subprocess.CalledProcessError: Propagated if ``randomTrips.py`` or
                        ``duarouter`` exits with a non-zero status.
        """
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please set the 'SUMO_HOME' environment variable.")

        rng = random.Random(seed)

        temp_urban_trips = os.path.join(self.output_dir, "_temp_urban_trips.xml")
        temp_urban_rou = os.path.join(self.output_dir, "_temp_urban.rou.xml")
        temp_hw_trips = os.path.join(self.output_dir, "_temp_hw_trips.xml")
        temp_hw_rou = os.path.join(self.output_dir, "_temp_hw.rou.xml")

        try:
            # ── Stream 1: Urban random trips ─────────────────────────────────────
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
                str(duration_seconds / urban_count),
                "--fringe-factor",
                "10",
                "--validate",
                "--remove-loops",
                "--seed",
                str(seed),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Reshape urban departure times + apply vehicle type + namespace IDs.
            urban_tree = ET.parse(temp_urban_rou)
            urban_root = urban_tree.getroot()
            elements = [el for el in urban_root if el.tag in ("vehicle", "trip")]

            if demand_profile:
                new_times = self._sample_from_profile(
                    len(elements), duration_seconds, demand_profile
                )
            else:
                # Uniform: re-space whatever randomTrips produced across the window.
                interval = duration_seconds / max(len(elements), 1)
                new_times = [i * interval for i in range(len(elements))]

            for idx, (el, t) in enumerate(zip(elements, new_times)):
                el.set("depart", f"{t:.2f}")
                el.set("type", "passenger_car")
                el.set("id", f"urban_{idx}")

            elements.sort(key=lambda x: float(x.get("depart")))  # type: ignore - should fail in case of missing attribute
            urban_root[:] = elements
            urban_tree.write(temp_urban_rou, encoding="utf-8", xml_declaration=True)

            # ── Stream 2: Highway fringe demand ──────────────────────────────────
            routed_count = 0
            if highway_count > 0:
                # Validate prerequisites
                if not self.origin_ids:
                    raise ValueError(
                        "origin_ids is empty.  Call create_consolidated_network() "
                        "before generating highway demand."
                    )
                dest_node_ids = (self.destination_ids or []) + (self.offramp_ids or [])
                if not dest_node_ids:
                    raise ValueError(
                        "Both destination_ids and offramp_ids are empty.  "
                        "At least one destination or offramp node is required for "
                        "the highway stream."
                    )

                raw_origins = [self._strip_node_prefix(n) for n in self.origin_ids]
                raw_dests = [self._strip_node_prefix(n) for n in dest_node_ids]
                print(f"[DEBUG] Highway stream — searching net XML for:")
                print(f"  origin  raw IDs  ({len(raw_origins)}): {raw_origins}")
                print(f"  dest    raw IDs  ({len(raw_dests)}):   {raw_dests}")

                from_edges = self._get_fringe_edges(self.origin_ids, "from")
                to_edges = self._get_fringe_edges(dest_node_ids, "to")

                if not from_edges:
                    raise ValueError(
                        f"No edges (in either direction) found for origin nodes "
                        f"{raw_origins} in {self.net_file}.\n"
                        "Possible causes:\n"
                        "  * create_consolidated_network() was not called before generate_demand().\n"
                        "  * The node IDs don't match any junction in the net XML."
                    )
                if not to_edges:
                    # Collect the first 20 junction IDs from the net for comparison.
                    _tree = ET.parse(self.net_file)
                    _net_junctions = [
                        j.get("id")
                        for j in _tree.getroot().findall("junction")
                        if j.get("type") != "internal"
                    ][:20]
                    raise ValueError(
                        f"No edges (in either direction) found for "
                        f"{len(dest_node_ids)} destination/offramp nodes after "
                        f"prefix stripping.\n"
                        f"  Searched raw IDs : {raw_dests}\n"
                        f"  Net junctions (first 20): {_net_junctions}\n"
                        "Check whether the stripped IDs appear in that list.  "
                        "If not, the node IDs produced by NetworkArbitrator do not "
                        "correspond 1-to-1 with SUMO junction IDs."
                    )
                print(
                    f"[INFO] Highway stream edge pools — "
                    f"departure: {len(from_edges)}, arrival: {len(to_edges)}"
                )

                # Departure times — same profile logic, independent count
                if demand_profile:
                    hw_departures = self._sample_from_profile(
                        highway_count, duration_seconds, demand_profile
                    )
                else:
                    hw_interval = duration_seconds / highway_count
                    hw_departures = [i * hw_interval for i in range(highway_count)]

                # Build raw trips and validate with duarouter
                hw_trips_root = self._build_highway_trips_xml(
                    hw_departures, from_edges, to_edges, rng
                )
                ET.ElementTree(hw_trips_root).write(
                    temp_hw_trips, encoding="utf-8", xml_declaration=True
                )
                self._validate_with_duarouter(temp_hw_trips, temp_hw_rou, seed)

                # Report how many highway trips survived routing
                hw_tree = ET.parse(temp_hw_rou)
                routed_count = sum(
                    1 for el in hw_tree.getroot() if el.tag in ("vehicle", "trip")
                )
                if routed_count < highway_count:
                    print(
                        f"[WARN] Highway stream: {highway_count - routed_count} of "
                        f"{highway_count} trips were unroutable and dropped by "
                        "duarouter."
                    )

            # ── Merge both streams into the final route file ──────────────────────
            files_to_merge = [temp_urban_rou]
            if highway_count > 0:
                files_to_merge.append(temp_hw_rou)

            total = self._merge_route_files(files_to_merge, self.rou_file)

            parts = [f"urban={urban_count}"]
            if highway_count > 0:
                parts.append(f"highway={routed_count}/{highway_count} routed")
            print(
                f"[OK] Generated {total} vehicles ({', '.join(parts)}) "
                f"→ {self.rou_file}"
            )

        except subprocess.CalledProcessError as e:
            # Re-raise with full stderr visible to the caller rather than silencing.
            stderr = e.stderr.strip() if e.stderr else "(no stderr)"
            raise RuntimeError(
                f"Subprocess failed (exit {e.returncode}): {' '.join(e.cmd)}\n{stderr}"
            ) from e

        finally:
            # Always clean up temp files, even on error
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

    # def build_destination_boundary_conditions(
    #     self,
    #     backbone_state_path: str,
    # ) -> tuple[
    #     dict[str, Callable[[float], float]], dict[str, Callable[[float], float]]
    # ]:
    #     """Build time-varying flow and density boundary conditions for all destinations.

    #     For each destination node the function resolves which upstream MotorwayLink
    #     to read from, then builds a linear interpolant over the backbone simulation's
    #     time array.

    #     Resolution rule (Offramp checked BEFORE direct MotorwayLink):
    #     1. If the host node has an Offramp in its incoming links, walk back
    #         through the offramp to its origin node and take the MotorwayLink
    #         feeding *that* node.  This gives the pre-diverge (pre-split) flow.
    #     2. If the host node has a direct MotorwayLink in its incoming links
    #         (no Offramp), use that link directly.

    #     Offramp is checked first so that, when a node sits at the end of an
    #     off-ramp that branches from a diverge, we read the backbone link *before*
    #     the split rather than the (shorter, lower-flow) off-ramp link.

    #     The last spatial cell of the resolved link is used for both flow and
    #     density, since that cell is closest to the downstream boundary.

    #     Args:
    #         backbone_state_path: Path to the backbone_state.json file produced
    #             by a prior macroscopic simulation step.

    #     Returns:
    #         (destination_flow_bc, destination_density_bc): two dicts mapping
    #         destination IDs to callables ``f(t_hours) -> float``.
    #     """
    #     logger = logging.getLogger(__name__)

    #     _FALLBACK_FLOW: float = 6000.0  # veh/h — used when no data is available
    #     _FALLBACK_DENSITY: float = 10.0  # veh/km — used when no data is available

    #     if self.consolidated_network is None:
    #         raise ValueError(
    #             "consolidated_network is not initialised. "
    #             "Call create_consolidated_network() before "
    #             "build_destination_boundary_conditions()."
    #         )

    #     if self.destination_ids is None:
    #         raise ValueError(
    #             "destination_ids is None. "
    #             "Call create_consolidated_network() before "
    #             "build_destination_boundary_conditions()."
    #         )

    #     # ------------------------------------------------------------------
    #     # Load backbone state
    #     # ------------------------------------------------------------------
    #     with open(backbone_state_path, "r", encoding="utf-8") as fh:
    #         backbone_state = json.load(fh)

    #     time_array: list[float] = backbone_state["time_array"]
    #     flows_ts: dict[str, list[list[float]]] = backbone_state["state_time_series"][
    #         "flows"
    #     ]
    #     densities_ts: dict[str, list[list[float]]] = backbone_state[
    #         "state_time_series"
    #     ]["densities"]
    #     t_arr = np.array(time_array)
    #     n_steps = len(time_array)

    #     logger.debug(
    #         "backbone_state loaded: %d time steps, %d flow links, %d density links",
    #         n_steps,
    #         len(flows_ts),
    #         len(densities_ts),
    #     )

    #     # Warn if most of the backbone state is zeros — this is the likely root
    #     # cause of very low BC values and is worth surfacing loudly.
    #     for link_id, rows in flows_ts.items():
    #         non_zero_steps = sum(1 for row in rows if row and any(v > 0.0 for v in row))
    #         if non_zero_steps < n_steps * 0.1:
    #             logger.debug(
    #                 "backbone_state WARNING: link '%s' has only %d / %d non-zero "
    #                 "time steps — BCs derived from this link will be near-zero "
    #                 "for most of the simulation. Check that vehicles remain in "
    #                 "the network throughout the simulation, not just at t=0.",
    #                 link_id,
    #                 non_zero_steps,
    #                 n_steps,
    #             )

    #     node_by_id = {node.id: node for node in self.consolidated_network}

    #     # ------------------------------------------------------------------
    #     # _last_cell_series
    #     #
    #     # Extract the value of the *last spatial cell* at every time step for
    #     # a given link.  The last cell is the one at the downstream end of the
    #     # link, closest to the destination boundary.
    #     #
    #     # Empty rows (no cells recorded for that time step) are replaced with
    #     # 0.0 rather than skipped, so the returned series always has exactly
    #     # len(time_array) entries.  Skipping them would produce a shorter
    #     # series which _make_interp would silently treat as "no data", falling
    #     # back to the constant fallback value.
    #     # ------------------------------------------------------------------
    #     def _last_cell_series(
    #         link_id: str,
    #         ts: dict[str, list[list[float]]],
    #     ) -> list[float] | None:
    #         rows = ts.get(link_id)
    #         if not rows:
    #             return None
    #         # Preserve zeros for empty rows — do NOT use `if row` filter.
    #         extracted = [row[-1] if row else 0.0 for row in rows]
    #         return extracted

    #     # ------------------------------------------------------------------
    #     # _make_interp
    #     #
    #     # Build a linear interpolant over time_array.  If the series is
    #     # missing or has a length mismatch, fall back to a constant function
    #     # and log the reason so callers can detect the issue.
    #     # ------------------------------------------------------------------
    #     def _make_interp(
    #         series: list[float] | None,
    #         fallback: float,
    #         label: str = "",
    #     ) -> Callable[[float], float]:
    #         if series is None:
    #             logger.debug(
    #                 "_make_interp [%s]: series is None — using constant fallback %.2f",
    #                 label,
    #                 fallback,
    #             )
    #             return lambda _t, _f=fallback: _f
    #         if len(series) != n_steps:
    #             logger.debug(
    #                 "_make_interp [%s]: series length %d != time_array length %d "
    #                 "— using constant fallback %.2f",
    #                 label,
    #                 len(series),
    #                 n_steps,
    #                 fallback,
    #             )
    #             return lambda _t, _f=fallback: _f
    #         v_arr = np.array(series)
    #         return lambda t, _t=t_arr, _v=v_arr: float(np.interp(t, _t, _v))

    #     # ------------------------------------------------------------------
    #     # Main loop — resolve upstream link for each destination
    #     # ------------------------------------------------------------------
    #     destination_flow_bc: dict[str, Callable[[float], float]] = {}
    #     destination_density_bc: dict[str, Callable[[float], float]] = {}

    #     for dest_id in self.destination_ids:
    #         host_node_id = dest_id.removeprefix("dest_")
    #         host_node = node_by_id.get(host_node_id)

    #         if host_node is None:
    #             raise ValueError(
    #                 f"[build_destination_bc] Host node '{host_node_id}' for destination "
    #                 f"'{dest_id}' not found in consolidated network."
    #             )

    #         upstream_link: MotorwayLink | None = None
    #         link_source: str = "unknown"

    #         # ----------------------------------------------------------------
    #         # Traversal — Offramp checked FIRST.
    #         #
    #         # Rationale: a destination that sits at the end of an off-ramp must
    #         # read from the pre-diverge backbone link (the MotorwayLink that
    #         # feeds the diverge node where the off-ramp branches off).  If we
    #         # checked MotorwayLink first, we might accidentally pick up a short
    #         # post-diverge highway segment that also terminates at the same node,
    #         # producing a lower (already-split) flow reading.
    #         # ----------------------------------------------------------------
    #         offramp_incoming: Offramp | None = None
    #         motorway_incoming: MotorwayLink | None = None

    #         for incoming in host_node.incoming:
    #             if isinstance(incoming, Offramp) and offramp_incoming is None:
    #                 offramp_incoming = incoming
    #             elif isinstance(incoming, MotorwayLink) and motorway_incoming is None:
    #                 motorway_incoming = incoming

    #         if offramp_incoming is not None:
    #             # Case 1: destination is at the end of an off-ramp.
    #             # Walk back to the MotorwayLink that feeds the diverge node.
    #             logger.debug(
    #                 "dest '%s' → Offramp '%s', walking upstream to backbone diverge node",
    #                 dest_id,
    #                 offramp_incoming.id,
    #             )
    #             offramp_origin = node_by_id.get(offramp_incoming.origin_node_id)
    #             if offramp_origin is None:
    #                 raise ValueError(
    #                     f"[build_destination_bc] Offramp '{offramp_incoming.id}' origin node "
    #                     f"'{offramp_incoming.origin_node_id}' not found in consolidated network "
    #                     f"(destination '{dest_id}')."
    #                 )

    #             for upstream in offramp_origin.incoming:
    #                 if isinstance(upstream, MotorwayLink):
    #                     upstream_link = upstream
    #                     link_source = f"backbone (via offramp '{offramp_incoming.id}')"
    #                     logger.debug(
    #                         "dest '%s' → Offramp '%s' → MotorwayLink '%s' (pre-diverge)",
    #                         dest_id,
    #                         offramp_incoming.id,
    #                         upstream.id,
    #                     )
    #                     break

    #             if upstream_link is None:
    #                 raise ValueError(
    #                     f"[build_destination_bc] No upstream MotorwayLink found behind "
    #                     f"offramp '{offramp_incoming.id}' at diverge node "
    #                     f"'{offramp_incoming.origin_node_id}' (destination '{dest_id}')."
    #                 )

    #         elif motorway_incoming is not None:
    #             # Case 2: destination is directly at the end of a MotorwayLink
    #             # (no off-ramp involved).
    #             upstream_link = motorway_incoming
    #             link_source = "backbone (direct)"
    #             logger.debug(
    #                 "dest '%s' → direct MotorwayLink '%s'",
    #                 dest_id,
    #                 motorway_incoming.id,
    #             )

    #         else:
    #             raise ValueError(
    #                 f"[build_destination_bc] Could not resolve upstream MotorwayLink for "
    #                 f"destination '{dest_id}': host node '{host_node_id}' has neither an "
    #                 f"Offramp nor a MotorwayLink in its incoming links."
    #             )

    #         # ----------------------------------------------------------------
    #         # Extract time series and build interpolants
    #         # ----------------------------------------------------------------
    #         flow_series = _last_cell_series(upstream_link.id, flows_ts)
    #         density_series = _last_cell_series(upstream_link.id, densities_ts)

    #         if flow_series is None:
    #             warnings.warn(
    #                 f"[build_destination_bc] Link '{upstream_link.id}' ({link_source}) "
    #                 f"has no detector coverage in backbone state for destination '{dest_id}'. "
    #                 f"Flow BC will use fallback constant {_FALLBACK_FLOW} veh/h."
    #             )
    #         if density_series is None:
    #             warnings.warn(
    #                 f"[build_destination_bc] Link '{upstream_link.id}' ({link_source}) "
    #                 f"has no detector coverage in backbone state for destination '{dest_id}'. "
    #                 f"Density BC will use fallback constant {_FALLBACK_DENSITY} veh/km."
    #             )

    #         if flow_series is not None:
    #             non_zero = sum(1 for v in flow_series if v > 0)
    #             logger.debug(
    #                 "dest '%s' link '%s' flow series: %d steps, %d non-zero, "
    #                 "min=%.1f max=%.1f mean=%.1f veh/h",
    #                 dest_id,
    #                 upstream_link.id,
    #                 len(flow_series),
    #                 non_zero,
    #                 min(flow_series),
    #                 max(flow_series),
    #                 sum(flow_series) / len(flow_series),
    #             )
    #         if density_series is not None:
    #             non_zero = sum(1 for v in density_series if v > 0)
    #             logger.debug(
    #                 "dest '%s' link '%s' density series: %d steps, %d non-zero, "
    #                 "min=%.4f max=%.4f mean=%.4f veh/km",
    #                 dest_id,
    #                 upstream_link.id,
    #                 len(density_series),
    #                 non_zero,
    #                 min(density_series),
    #                 max(density_series),
    #                 sum(density_series) / len(density_series),
    #             )

    #         flow_label = f"{dest_id}/flow/{upstream_link.id}"
    #         density_label = f"{dest_id}/density/{upstream_link.id}"

    #         destination_flow_bc[dest_id] = _make_interp(
    #             flow_series, _FALLBACK_FLOW, label=flow_label
    #         )
    #         destination_density_bc[dest_id] = _make_interp(
    #             density_series, _FALLBACK_DENSITY, label=density_label
    #         )

    #         # Spot-check the interpolant at 5 evenly spaced times so that
    #         # near-zero BCs are immediately visible in the debug log.
    #         if t_arr.size > 0:
    #             sample_times = np.linspace(t_arr[0], t_arr[-1], 5)
    #             logger.debug(
    #                 "dest '%s' BC spot-check — flow (veh/h): %s | density (veh/km): %s",
    #                 dest_id,
    #                 [f"{destination_flow_bc[dest_id](t):.1f}" for t in sample_times],
    #                 [f"{destination_density_bc[dest_id](t):.3f}" for t in sample_times],
    #             )

    #         print(
    #             f"  [dest BC] '{dest_id}' ← link '{upstream_link.id}' ({link_source})"
    #         )

    #     logger.debug(
    #         "build_destination_boundary_conditions complete: %d destinations resolved",
    #         len(destination_flow_bc),
    #     )

    #     return destination_flow_bc, destination_density_bc

    def _get_edge_lane_counts(self) -> dict[str, int]:
        """Parse the net_file to map edge IDs to their number of lanes."""
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.net_file)
        root = tree.getroot()
        lane_counts = {}

        for edge in root.findall("edge"):
            # Skip internal edges if they aren't relevant to your data collection
            if edge.get("function") == "internal":
                continue

            eid = edge.get("id")
            if eid:
                lanes = edge.findall("lane")
                lane_counts[eid] = len(lanes)

        return lane_counts

    def build_destination_bc_from_sumo_edges(
        self,
        edge_data_path: str,
    ) -> tuple[
        dict[str, Callable[[float], float]],
        dict[str, Callable[[float], float]],
    ]:
        """Build destination BCs directly from SUMO edgeData output.

        For each destination node, resolves the SUMO edge(s) immediately upstream
        via the net XML, then reads flow and total density time series from the SUMO
        edgeData output file.

        Args:
            edge_data_path: Path to SUMO edgeData XML output.

        Returns:
            (destination_flow_bc, destination_density_bc): two dicts mapping
            destination IDs to callables f(t_hours) -> float.
        """
        import xml.etree.ElementTree as ET
        import numpy as np
        import warnings
        import logging

        logger = logging.getLogger(__name__)

        _FALLBACK_FLOW = 6000.0  # veh/h
        _FALLBACK_DENSITY = 10.0  # veh/km

        if self.consolidated_network is None or self.destination_ids is None:
            raise ValueError(
                "Call create_consolidated_network() before "
                "build_destination_bc_from_sumo_edges()."
            )

        # Pre-fetch lane counts to scale density correctly
        lane_counts = self._get_edge_lane_counts()

        # ------------------------------------------------------------------
        # 1. Parse edgeData XML into per-edge time series
        # ------------------------------------------------------------------
        tree = ET.parse(edge_data_path)
        root = tree.getroot()

        edge_flow_ts: dict[str, list[tuple[float, float]]] = {}
        edge_density_ts: dict[str, list[tuple[float, float]]] = {}

        for interval in root.findall("interval"):
            t_begin = float(interval.get("begin", 0))
            t_end = float(interval.get("end", 0))
            t_mid = ((t_begin + t_end) / 2.0) / 3600.0  # → hours
            interval_h = (t_end - t_begin) / 3600.0

            for edge_el in interval.findall("edge"):
                eid = edge_el.get("id")
                if not eid:
                    continue

                raw_flow = edge_el.get("flow")
                raw_dens = edge_el.get("density")

                # Fallback: compute from nVehContrib if flow attr absent
                if raw_flow is None:
                    n_veh = float(edge_el.get("nVehContrib", 0))
                    raw_flow = str(n_veh / interval_h) if interval_h > 0 else "0"

                edge_flow_ts.setdefault(eid, []).append((t_mid, float(raw_flow)))

                if raw_dens is not None:
                    # SCALE BY LANE COUNT: SUMO density is veh/km/lane
                    num_lanes = lane_counts.get(eid, 1)
                    total_density = float(raw_dens) * num_lanes
                    edge_density_ts.setdefault(eid, []).append((t_mid, total_density))

        logger.debug(
            "edgeData parsed: %d edges with flow data, %d with density data",
            len(edge_flow_ts),
            len(edge_density_ts),
        )

        # ------------------------------------------------------------------
        # 2. For each destination, resolve its upstream SUMO edge(s)
        # ------------------------------------------------------------------
        def _make_interp_from_pairs(
            pairs: list[tuple[float, float]] | None,
            fallback: float,
            label: str = "",
        ) -> Callable[[float], float]:
            if not pairs:
                logger.debug(
                    "_make_interp [%s]: no data — constant fallback %.2f",
                    label,
                    fallback,
                )
                return lambda _t, _f=fallback: _f
            t_arr = np.array([p[0] for p in pairs])
            v_arr = np.array([p[1] for p in pairs])
            return lambda t, _t=t_arr, _v=v_arr: float(np.interp(t, _t, _v))

        destination_flow_bc: dict[str, Callable[[float], float]] = {}
        destination_density_bc: dict[str, Callable[[float], float]] = {}

        for dest_id in self.destination_ids:
            upstream_edges = self._get_fringe_edges(
                [dest_id], direction="to", allow_reverse_fallback=True
            )

            if not upstream_edges:
                warnings.warn(
                    f"[build_destination_bc] No upstream SUMO edges "
                    f"found for destination '{dest_id}'. Using fallback constants."
                )
                destination_flow_bc[dest_id] = lambda _t: _FALLBACK_FLOW
                destination_density_bc[dest_id] = lambda _t: _FALLBACK_DENSITY
                continue

            flow_pairs_agg: dict[float, float] = {}
            density_pairs_agg: dict[float, list[float]] = {}
            matched_edges: list[str] = []

            for eid in upstream_edges:
                if eid in edge_flow_ts:
                    matched_edges.append(eid)
                    for t_mid, f in edge_flow_ts[eid]:
                        flow_pairs_agg[t_mid] = flow_pairs_agg.get(t_mid, 0.0) + f
                    for t_mid, d in edge_density_ts.get(eid, []):
                        density_pairs_agg.setdefault(t_mid, []).append(d)

            if not matched_edges:
                warnings.warn(
                    f"[build_destination_bc] Destination '{dest_id}': "
                    f"upstream edges {upstream_edges} found in net XML but none appear "
                    f"in edgeData output. Using fallback."
                )
                destination_flow_bc[dest_id] = lambda _t: _FALLBACK_FLOW
                destination_density_bc[dest_id] = lambda _t: _FALLBACK_DENSITY
                continue

            flow_pairs = sorted(flow_pairs_agg.items())

            # Since density is already total density per edge, we sum them if multiple edges feed the node.
            # (If you prefer the average density across incoming links, keep your original `sum(vs)/len(vs)`)
            # density_pairs = sorted((t, sum(vs)) for t, vs in density_pairs_agg.items())
            density_pairs = sorted(
                (t, sum(vs) / len(vs)) for t, vs in density_pairs_agg.items()
            )

            flow_label = f"{dest_id}/sumo_flow/{matched_edges}"
            density_label = f"{dest_id}/sumo_density/{matched_edges}"

            destination_flow_bc[dest_id] = _make_interp_from_pairs(
                flow_pairs, _FALLBACK_FLOW, label=flow_label
            )
            destination_density_bc[dest_id] = _make_interp_from_pairs(
                density_pairs if density_pairs else None,
                _FALLBACK_DENSITY,
                label=density_label,
            )

            print(
                f"  [dest BC] '{dest_id}' ← SUMO edges {matched_edges} "
                f"({len(flow_pairs)} time steps)"
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
