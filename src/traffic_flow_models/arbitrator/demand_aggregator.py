import csv
import xml.etree.ElementTree as ET
import networkx as nx
from collections import defaultdict
from typing import Callable, Tuple


class DemandAggregator:
    """Aggregate microscopic SUMO detector data into macroscopic demand functions.

    This class processes loop detector outputs from SUMO simulations and aggregates
    them spatially and temporally to produce demand functions suitable for macroscopic
    entry points (origins and onramps). The aggregation follows the network topology
    to capture all upstream demand feeding into each macroscopic model interface point.

    Attributes:
        detector_output_path: Path to SUMO detector output XML file.
        detector_spec_path: Path to detector specification CSV file.
        time_period_sec: Aggregation time period in seconds.
        detector_intervals: Raw detector readings indexed by detector ID.
        detector_mapping: Maps detector IDs to node IDs and types.
        node_counts: Aggregated vehicle counts per node and time bin.
        max_time: Maximum simulation time observed in detector data.
    """

    def __init__(
        self,
        detector_output_path: str,
        detector_spec_path: str,
        time_period_minutes: int = 15,
    ):
        """Initialize the demand aggregator.

        Args:
            detector_output_path: Path to the SUMO detector output XML file.
            detector_spec_path: Path to the detector specification CSV file.
            time_period_minutes: Time period for aggregation in minutes (default: 15).
        """

        self.detector_output_path: str = detector_output_path
        self.detector_spec_path: str = detector_spec_path
        self.time_period_sec: int = time_period_minutes * 60

        self.detector_intervals: defaultdict[str, list[Tuple[float, int]]] = (
            defaultdict(list)
        )
        self.detector_mapping: dict[str, dict[str, str]] = {}
        self.node_counts: defaultdict[str, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.max_time: float = 0.0

    def parse_detector_output(self) -> None:
        """Parse SUMO detector output XML file and extract interval data.

        Reads the detector output XML file produced by SUMO and extracts vehicle
        count data for each detector over time. Stores raw interval data indexed
        by detector ID and updates the maximum observed simulation time.
        """
        tree = ET.parse(self.detector_output_path)
        root = tree.getroot()

        for interval in root.findall("interval"):
            det_id = interval.get("id")
            begin_str = interval.get("begin")

            # skip malformed entries
            if det_id is None or begin_str is None:
                continue

            begin = float(begin_str)
            count = int(interval.get("nVehEntered", interval.get("nVehContrib", 0)))
            self.detector_intervals[det_id].append((begin, count))
            self.max_time = max(self.max_time, begin)

    def classify_and_map(self) -> None:
        """Map detector IDs to node IDs from CSV specification.

        Reads the detector specification CSV file and creates a mapping between
        detector IDs and their corresponding network nodes. Classifies detectors
        by type (onramp, offramp, origin, destination) and determines the
        appropriate node ID based on the detector type and edge topology.

        The method handles various detector ID formats by creating multiple
        variants to ensure robust matching with the detector output data.
        """
        with open(self.detector_spec_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                det_id = row["detector_id"].strip().strip('"').strip("'")

                det_id_variants = [
                    det_id,
                    det_id.replace("detector_", ""),
                    f"detector_{det_id}",
                ]

                det_type = row["type"].strip().lower()
                from_node = row["from"].strip().strip('"').strip("'")
                to_node = row["to"].strip().strip('"').strip("'")

                if "onramp" in det_type or "origin" in det_type:
                    node_id = to_node
                elif "offramp" in det_type or "destination" in det_type:
                    node_id = from_node
                else:
                    node_id = to_node

                if node_id:
                    # store all variants
                    for variant in det_id_variants:
                        self.detector_mapping[variant] = {
                            "node_id": node_id,
                            "type": det_type,
                        }

    def aggregate_spatially(self) -> None:
        """Aggregate lane-level detector counts into node-level counts.

        Sums vehicle counts from all lane detectors at the same network node
        and organizes them into time bins. This spatial aggregation consolidates
        multi-lane detector data into single node-level measurements suitable
        for macroscopic modeling.

        The aggregation respects the time period specified during initialization
        and creates discrete time bins for temporal aggregation.
        """
        for det_id, intervals in self.detector_intervals.items():
            if det_id not in self.detector_mapping:
                continue

            node_id = self.detector_mapping[det_id]["node_id"]

            for begin, count in intervals:
                time_bin = int(begin / self.time_period_sec)
                self.node_counts[node_id][time_bin] += count

    def aggregate_urban_inflows(
        self,
        origin_ids: list[str],
        onramp_ids: list[str],
        sumo_network_path: str,
    ) -> Tuple[
        dict[str, Callable[[float], float]], dict[str, Callable[[float], float]]
    ]:
        """Aggregate all detector data from roads feeding into macroscopic model entry points.

        Identifies all network nodes upstream of each macroscopic model origin
        and onramp, aggregates their detector counts, and produces time-dependent
        demand functions suitable for macroscopic simulation. This captures the
        full demand from the microscopic network feeding into the macroscopic model.

        Args:
            origin_ids: List of origin node IDs in the network.
            onramp_ids: List of onramp node IDs in the network.
            sumo_network_path: Path to the SUMO network XML file used for
                topology analysis.

        Returns:
            A tuple containing:
                - origin_demands: Dictionary mapping origin IDs to demand functions.
                - onramp_demands: Dictionary mapping onramp IDs to demand functions.
        """
        graph = self._build_network_graph(sumo_network_path)

        origin_node_ids = [oid.replace("origin_", "") for oid in origin_ids]
        onramp_node_ids = [oid.replace("onramp_", "") for oid in onramp_ids]

        origin_demands = {}

        for origin_node in origin_node_ids:
            upstream_nodes = self._find_upstream_nodes(graph, origin_node)
            aggregated_bins = self._aggregate_demand(upstream_nodes)

            origin_id = f"origin_{origin_node}"
            origin_demands[origin_id] = self._make_demand_function(aggregated_bins)

        onramp_demands = {}

        for onramp_node in onramp_node_ids:
            upstream_nodes = self._find_upstream_nodes(graph, onramp_node)
            aggregated_bins = self._aggregate_demand(upstream_nodes)

            onramp_id = f"onramp_{onramp_node}"
            onramp_demands[onramp_id] = self._make_demand_function(aggregated_bins)

        all_detector_vehicles = sum(
            sum(bins.values()) for bins in self.node_counts.values()
        )

        print("AGGREGATION SUMMARY:")
        print(f"  Total detector nodes: {len(self.node_counts)}")
        print(f"  Total detector vehicles: {all_detector_vehicles}")
        print(f"  Network entry points: {len(origin_demands) + len(onramp_demands)}")
        return origin_demands, onramp_demands

    def _build_network_graph(self, sumo_network_path: str) -> nx.DiGraph:
        """Build directed graph from SUMO network XML.

        Parses the SUMO network XML file and constructs a NetworkX directed
        graph representing the network topology. Internal edges (junctions)
        are excluded to focus on the road network structure.

        Args:
            sumo_network_path: Path to the SUMO network XML file.

        Returns:
            NetworkX DiGraph object representing the road network.
        """
        graph = nx.DiGraph()
        tree = ET.parse(sumo_network_path)
        root = tree.getroot()

        for edge in root.findall("edge"):
            if edge.get("function") != "internal":
                from_node = edge.get("from")
                to_node = edge.get("to")
                if from_node and to_node:
                    graph.add_edge(from_node, to_node)

        return graph

    def _find_upstream_nodes(self, graph: nx.DiGraph, target_node: str) -> set[str]:
        """Find all nodes that have a path leading to the target node.

        Identifies all network nodes from which there exists a directed path
        to the target node. This is used to determine which detector data
        should be aggregated for a given macroscopic model entry point.

        Args:
            graph: NetworkX DiGraph representing the network topology.
            target_node: Node ID for which to find upstream nodes.

        Returns:
            Set of node IDs that are upstream of the target node, including
            the target node itself.
        """
        upstream_nodes = {target_node}

        for node in self.node_counts.keys():
            if node == target_node:
                continue

            try:
                if graph.has_node(node) and graph.has_node(target_node):
                    if nx.has_path(graph, node, target_node):
                        upstream_nodes.add(node)
            except nx.NetworkXError:
                continue

        return upstream_nodes

    def _aggregate_demand(self, node_set: set[str]) -> dict[int, int]:
        """Aggregate vehicle counts from multiple nodes.

        Sums vehicle counts across all nodes in the provided set for each
        time bin. This produces a single time-varying demand profile that
        captures the total traffic from multiple upstream measurement points.

        Args:
            node_set: Set of node IDs from which to aggregate demand.

        Returns:
            Dictionary mapping time bins to aggregated vehicle counts.
        """
        aggregated_bins = defaultdict(int)

        for node in node_set:
            if node in self.node_counts:
                for time_bin, count in self.node_counts[node].items():
                    aggregated_bins[time_bin] += count

        return dict(aggregated_bins)

    def _make_demand_function(
        self, aggregated_bins: dict[int, int]
    ) -> Callable[[float], float]:
        """Create a demand function that returns veh/h for given time in hours.

        Constructs a callable function that converts aggregated vehicle count
        data into a time-dependent demand rate in vehicles per hour. The
        function performs linear interpolation within time bins and converts
        counts to hourly rates.

        Args:
            aggregated_bins: Dictionary mapping time bins to vehicle counts.

        Returns:
            A callable function that takes time in hours and returns demand
            in vehicles per hour.
        """
        time_period_sec = self.time_period_sec
        scale = 3600.0 / time_period_sec
        return (
            lambda time_hours: aggregated_bins.get(
                int((time_hours * 3600) / time_period_sec), 0
            )
            * scale
        )

    def run(
        self,
        origin_ids: list[str],
        onramp_ids: list[str],
        sumo_network_path: str,
    ) -> Tuple[
        dict[str, Callable[[float], float]], dict[str, Callable[[float], float]]
    ]:
        """Execute the complete demand aggregation pipeline.

        Orchestrates the full workflow: parsing detector outputs, mapping
        detectors to nodes, performing spatial aggregation, and computing
        upstream demand functions for all macroscopic model entry points.

        Args:
            origin_ids: List of origin node IDs in the network.
            onramp_ids: List of onramp node IDs in the network.
            sumo_network_path: Path to the SUMO network XML file.

        Returns:
            A tuple containing:
                - origin_demands: Dictionary mapping origin IDs to demand functions.
                - onramp_demands: Dictionary mapping onramp IDs to demand functions.

        Raises:
            ValueError: If sumo_network_path is not provided.
        """
        if not sumo_network_path:
            raise ValueError("sumo_network_path is required")

        self.parse_detector_output()
        self.classify_and_map()
        self.aggregate_spatially()

        return self.aggregate_urban_inflows(origin_ids, onramp_ids, sumo_network_path)
