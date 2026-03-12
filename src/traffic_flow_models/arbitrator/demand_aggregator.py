import csv
import warnings
import xml.etree.ElementTree as ET
import networkx as nx
from collections import defaultdict
from typing import Callable, Tuple

from traffic_flow_models.arbitrator.aggregation_helpers import (
    make_single_stream_rolling_window_aggregator,
)


class DemandAggregator:
    """Aggregate microscopic SUMO detector data into macroscopic demand functions.

    This class processes loop detector outputs from SUMO simulations and aggregates
    them spatially and temporally to produce demand functions suitable for macroscopic
    entry points (origins; ramp inflows are mapped to additional origins). The aggregation
    follows the network topology to capture all upstream demand feeding into each
    macroscopic model interface point.

    Uses a rolling window approach for temporal aggregation to smooth demand functions
    while preserving time-varying behavior.

    Attributes:
        detector_output_path: Path to SUMO detector output XML file.
        detector_spec_path: Path to detector specification CSV file.
        window_size_sec: Rolling window size in seconds for temporal aggregation.
        detector_intervals: Raw detector readings indexed by detector ID.
        detector_mapping: Maps detector IDs to node IDs and types.
        node_intervals: Raw interval data per node (not binned).
        max_time: Maximum simulation time observed in detector data.
    """

    def __init__(
        self,
        detector_output_path: str,
        detector_spec_path: str,
        window_size_minutes: float = 2.0,
    ):
        """Initialize the demand aggregator.

        Args:
            detector_output_path: Path to the SUMO detector output XML file.
            detector_spec_path: Path to the detector specification CSV file.
            window_size_minutes: Rolling window size in minutes (default: 2.0).
                At query time t, vehicle counts from [t - window/2, t + window/2] are aggregated.
        """

        self.detector_output_path: str = detector_output_path
        self.detector_spec_path: str = detector_spec_path
        self.window_size_sec: float = window_size_minutes * 60

        self.detector_intervals: defaultdict[str, list[Tuple[float, int]]] = (
            defaultdict(list)
        )
        self.detector_mapping: dict[str, dict[str, str]] = {}
        # store raw intervals per node, not binned counts
        self.node_intervals: defaultdict[str, list[Tuple[float, int]]] = defaultdict(
            list
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

                # match detector types based on spec CSV values:
                # 'inflow', 'outflow', 'ramp_inflow', 'ramp_outflow'
                if "inflow" in det_type:
                    # traffic entering the backbone network
                    node_id = to_node
                elif "outflow" in det_type:
                    # traffic leaving the backbone network
                    node_id = from_node
                elif "turning_rate" in det_type:
                    # turning rate detectors should be ignored for demand aggregation
                    continue
                else:
                    # fallback for any other types
                    warnings.warn(
                        f"Unrecognized detector type '{det_type}' for detector '{det_id}'. "
                        "Defaulting to using 'to_node' as node_id.",
                        stacklevel=2,
                    )
                    node_id = to_node

                if node_id:
                    # store all variants
                    for variant in det_id_variants:
                        self.detector_mapping[variant] = {
                            "node_id": node_id,
                            "type": det_type,
                        }

    def aggregate_spatially(self) -> None:
        """Aggregate lane-level detector counts into node-level intervals.

        Sums vehicle counts from all lane detectors at the same network node
        and stores them as raw intervals (not binned). This spatial aggregation
        consolidates multi-lane detector data into single node-level measurements
        suitable for rolling window temporal aggregation.
        """
        # group intervals by node and timestamp
        node_time_counts: defaultdict[str, defaultdict[float, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for det_id, intervals in self.detector_intervals.items():
            if det_id not in self.detector_mapping:
                continue

            node_id = self.detector_mapping[det_id]["node_id"]

            for begin, count in intervals:
                node_time_counts[node_id][begin] += count

        # convert to list of tuples per node
        for node_id, time_counts in node_time_counts.items():
            self.node_intervals[node_id] = sorted(time_counts.items())

    def aggregate_urban_inflows(
        self,
        origin_ids: list[str],
        onramp_ids: list[str],
        sumo_network_path: str,
    ) -> dict[str, Callable[[float], float]]:
        """Aggregate detector data feeding into macroscopic entry points.

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
            origin_demands: Dictionary mapping origin IDs to demand functions. Onramp
                inflows are converted into additional origins (prefixed with
                "origin_onramp_").
        """
        graph = self._build_network_graph(sumo_network_path)

        origin_node_ids = [oid.replace("origin_", "") for oid in origin_ids]
        onramp_node_ids = [oid.replace("onramp_", "") for oid in onramp_ids]

        origin_demands: dict[str, Callable[[float], float]] = {}

        for origin_node in origin_node_ids:
            upstream_nodes = self._find_upstream_nodes(graph, origin_node)
            aggregated_bins = self._aggregate_demand(upstream_nodes)

            origin_id = f"origin_{origin_node}"
            origin_demands[origin_id] = self._make_demand_function(aggregated_bins)

        for onramp_node in onramp_node_ids:
            upstream_nodes = self._find_upstream_nodes(graph, onramp_node)
            aggregated_bins = self._aggregate_demand(upstream_nodes)

            ramp_origin_id = f"origin_onramp_{onramp_node}"
            origin_demands[ramp_origin_id] = self._make_demand_function(aggregated_bins)

        all_detector_vehicles = sum(
            sum(count for _, count in intervals)
            for intervals in self.node_intervals.values()
        )

        print("AGGREGATION SUMMARY:")
        print(f"  Total detector nodes: {len(self.node_intervals)}")
        print(f"  Total detector vehicles: {all_detector_vehicles}")
        print(f"  Network entry points (origins incl. ramps): {len(origin_demands)}")
        return origin_demands

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

        for node in self.node_intervals.keys():
            if node == target_node:
                continue

            try:
                if graph.has_node(node) and graph.has_node(target_node):
                    if nx.has_path(graph, node, target_node):
                        upstream_nodes.add(node)
            except nx.NetworkXError:
                continue

        return upstream_nodes

    def _aggregate_demand(self, node_set: set[str]) -> list[Tuple[float, int]]:
        """Aggregate vehicle counts from multiple nodes.

        Merges vehicle count intervals from all nodes in the provided set.
        This produces a single time-varying demand profile that captures
        the total traffic from multiple upstream measurement points.

        Args:
            node_set: Set of node IDs from which to aggregate demand.

        Returns:
            List of (begin_time, count) tuples representing aggregated intervals.
        """
        # collect all intervals from the node set and sum by timestamp
        time_counts: defaultdict[float, int] = defaultdict(int)

        for node in node_set:
            if node in self.node_intervals:
                for begin, count in self.node_intervals[node]:
                    time_counts[begin] += count

        # return as sorted list of tuples
        return sorted(time_counts.items())

    def _make_demand_function(
        self, aggregated_intervals: list[Tuple[float, int]]
    ) -> Callable[[float], float]:
        """Create a demand function that returns veh/h for given time in hours.

        Constructs a callable function that converts aggregated vehicle count
        data into a time-dependent demand rate in vehicles per hour using
        rolling window aggregation. This produces smooth, continuous demand
        functions instead of step-wise bin lookups.

        Args:
            aggregated_intervals: List of (begin_time, count) tuples.

        Returns:
            A callable function that takes time in hours and returns demand
            in vehicles per hour.
        """
        return make_single_stream_rolling_window_aggregator(
            intervals=aggregated_intervals,
            window_size_sec=self.window_size_sec,
            max_time=self.max_time,
        )

    def run(
        self,
        origin_ids: list[str],
        onramp_ids: list[str],
        sumo_network_path: str,
    ) -> dict[str, Callable[[float], float]]:
        """Execute the complete demand aggregation pipeline.

        Orchestrates the full workflow: parsing detector outputs, mapping
        detectors to nodes, performing spatial aggregation, and computing
        upstream demand functions for all macroscopic model entry points.

        Args:
            origin_ids: List of origin node IDs in the network.
            onramp_ids: List of onramp node IDs in the network.
            sumo_network_path: Path to the SUMO network XML file.

        Returns:
            origin_demands: Dictionary mapping origin IDs to demand functions.

        Raises:
            ValueError: If sumo_network_path is not provided.
        """
        if not sumo_network_path:
            raise ValueError("sumo_network_path is required")

        self.parse_detector_output()
        self.classify_and_map()
        self.aggregate_spatially()

        return self.aggregate_urban_inflows(origin_ids, onramp_ids, sumo_network_path)
