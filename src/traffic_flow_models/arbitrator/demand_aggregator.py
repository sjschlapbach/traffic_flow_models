import csv
import warnings
import xml.etree.ElementTree as ET
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from typing import Callable, Tuple

from traffic_flow_models.arbitrator.aggregation_helpers import (
    make_single_stream_rolling_window_aggregator,
)


class DemandAggregator:
    """Aggregate SUMO detector output into time-varying demand functions for onramp origins.

    Parses SUMO detector output XML and a detector specification CSV, maps
    lane-level inflow counts to macroscopic network nodes, and produces
    rolling-window demand functions (veh/h) for each onramp origin in the
    macroscopic model.

    Typical usage::

        agg = DemandAggregator(detector_output_path, detector_spec_path)
        demand = agg.run(origin_ids, onramp_ids, sumo_network_path)
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

        intervals = root.findall("interval")
        for interval in tqdm(
            intervals, desc="Parsing detector intervals", unit="interval"
        ):
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
        """Read the detector spec CSV and build self.detector_mapping.

        Retains only ``inflow`` detectors — the urban and ramp inflows entering
        backbone boundary nodes. All other types are excluded:

        - ``backbone_segment``: mainline state measurement, handled by BackboneStateAggregator.
        - ``mainline_origin_interface``: mainline origin demand, handled by BackboneStateAggregator.
        - ``turning_rate``: diverge split ratios, handled by TurningRateAggregator.
        - ``outflow``: exits from the backbone, not required for demand aggregation.

        For each retained detector, resolves the associated network node ID from
        the ``backbone_node`` column, falling back to the ``to`` column when empty.

        Populates self.detector_mapping with a dict per detector ID variant,
        each containing ``{"node_id": ..., "type": ...}``.
        """
        with open(self.detector_spec_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in tqdm(reader, desc="Classifying detectors", unit="detector"):
                det_id = row["detector_id"].strip().strip('"').strip("'")
                det_type = row["type"].strip().lower()

                if det_type != "inflow":
                    continue

                node_id = row.get("backbone_node", "").strip().strip('"').strip("'")
                if not node_id:
                    node_id = row["to"].strip().strip('"').strip("'")

                if not node_id:
                    continue

                for variant in [
                    det_id,
                    det_id.replace("detector_", ""),
                    f"detector_{det_id}",
                ]:
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

        for det_id, intervals in tqdm(
            list(self.detector_intervals.items()),
            desc="Spatially aggregating detectors",
            unit="detector",
        ):
            if det_id not in self.detector_mapping:
                continue

            det_type = self.detector_mapping[det_id]["type"]
            if det_type != "inflow":
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

        graph = self._build_network_graph(sumo_network_path)
        origin_demands: dict[str, Callable[[float], float]] = {}

        onramp_id_set = {str(node_id) for node_id in onramp_ids}

        # Only keep origin keys that correspond to actual on-ramp nodes
        all_entry_points: dict[str, str] = {
            origin_id: origin_id.replace("origin_", "")
            for origin_id in origin_ids
            if origin_id.startswith("origin_")
            and origin_id.replace("origin_", "") in onramp_id_set
        }

        # Use only ramp-origin nodes as boundaries so ramp catchments do not overlap
        raw_origin_nodes = set(all_entry_points.values())

        for demand_key, raw_node_id in tqdm(
            list(all_entry_points.items()),
            desc="Aggregating entry points",
            unit="entry",
        ):
            upstream_nodes = self._find_upstream_nodes(
                graph,
                raw_node_id,
                raw_origin_nodes,
            )
            aggregated_bins = self._aggregate_demand(upstream_nodes)

            if not aggregated_bins:
                raise ValueError(
                    f"No detector intervals found for origin '{demand_key}' "
                    f"(node '{raw_node_id}'). Upstream catchment was "
                    f"{sorted(upstream_nodes)}. This indicates a wiring problem: "
                    f"either the detector spec CSV maps no 'inflow' detectors to "
                    f"any node in this catchment, or SUMO emitted no interval rows "
                    f"for those detectors. A legitimate zero-demand origin should "
                    f"still produce interval rows with count=0 and reach this code "
                    f"with a non-empty bins list."
                )

            demand_fn = self._make_demand_function(aggregated_bins)
            if demand_fn is None:
                raise ValueError(
                    f"Failed to build demand function for origin '{demand_key}' "
                    f"(node '{raw_node_id}') despite having {len(aggregated_bins)} "
                    f"interval bins (total count "
                    f"{sum(c for _, c in aggregated_bins)}). The rolling-window "
                    f"aggregator returned None — inspect "
                    f"make_single_stream_rolling_window_aggregator for edge cases."
                )

            origin_demands[demand_key] = demand_fn

        all_detector_vehicles = sum(
            sum(count for _, count in intervals)
            for intervals in self.node_intervals.values()
        )

        print("AGGREGATION SUMMARY:")
        print(f"  Total detector nodes: {len(self.node_intervals)}")
        print(f"  Total detector vehicles: {all_detector_vehicles}")
        print(f"  Ramp entry points: {len(origin_demands)}")

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

    def _find_upstream_nodes(
        self, graph: nx.DiGraph, target_node: str, origin_node_ids: set[str]
    ) -> set[str]:
        """Find all nodes that have a path leading to the target node.
        Identifies all network nodes from which there exists a directed path
        to the target node. This is used to determine which detector data
        should be aggregated for a given macroscopic model entry point.
        Args:
            graph: NetworkX DiGraph representing the network topology.
            target_node: Node ID for which to find upstream nodes.
            origin_node_ids: A set of all origin node IDs in the graph.
        Returns:
            Set of node IDs that are upstream of the target node, including
            the target node itself.
        """
        if not graph.has_node(target_node):
            raise ValueError(f"Target node '{target_node}' not found in the graph.")

        # find all nodes that can reach the target_node
        ancestors = nx.ancestors(graph, target_node)
        ancestors.add(target_node)

        # filter for nodes that are also detector locations
        relevant_nodes = ancestors.intersection(self.node_intervals.keys())
        upstream_nodes = {target_node}

        # for each relevant upstream node, find the closest origin it flows into
        for node in relevant_nodes:
            if node == target_node:
                continue

            # calculate shortest path lengths from the current node to all other nodes
            try:
                path_lengths = nx.single_source_shortest_path_length(graph, node)
            except nx.NetworkXNoPath:
                continue

            # check if the target_node is reachable
            if target_node not in path_lengths:
                continue

            shortest_to_target = path_lengths[target_node]

            # find the closest origin
            closest_origin = target_node
            closest_dist = shortest_to_target

            for other_origin in origin_node_ids:
                if other_origin == target_node:
                    continue
                if other_origin in path_lengths:
                    dist = path_lengths[other_origin]
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_origin = other_origin

            # if the target_node is the closest origin, claim this upstream node
            if closest_origin == target_node:
                upstream_nodes.add(node)

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
            offramp_ids: List of offramp node IDs in the network.
            backbone_node_ids: Set of node IDs that are part of the backbone network.
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
