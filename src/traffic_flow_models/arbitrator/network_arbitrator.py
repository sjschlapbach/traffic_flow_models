import json
import math
import warnings
import networkx as nx
import xml.etree.ElementTree as ET
from typing import Union, Tuple, Callable, TypedDict, cast


from traffic_flow_models.network import (
    Network,
    Origin,
    Onramp,
    Offramp,
    Destination,
    MotorwayLink,
    Node,
)


class RoadTypeParams(TypedDict):
    """Parameters for a specific road type.

    Attributes:
        lane_capacity: Maximum flow capacity per lane (veh/h/lane).
        jam_density: Maximum density at jam conditions (veh/km/lane).
        free_flow_speed: Speed at free-flow conditions (km/h).
    """

    lane_capacity: float
    jam_density: float
    free_flow_speed: float


class RoadParamsConfig(TypedDict):
    """Configuration for road parameters by road type.

    Must include the following road types: motorway, trunk, primary,
    secondary, tertiary, default.
    """

    motorway: RoadTypeParams
    trunk: RoadTypeParams
    primary: RoadTypeParams
    secondary: RoadTypeParams
    tertiary: RoadTypeParams
    default: RoadTypeParams


class NetworkArbitrator:
    """Convert SUMO microscopic networks into consolidated macroscopic networks.

    This class performs the arbitration between detailed microscopic road networks
    from SUMO and simplified macroscopic representations suitable for macroscopic modeling.
    The process involves filtering roads by type, merging serial edges, handling
    roundabouts, and instantiating network objects with appropriate parameters.

    Attributes:
        path: Path to the SUMO network XML file.
        target_cell_length: Target length for macroscopic link cells in kilometers (default: 0.3).
        min_link_length: Minimum acceptable link length in kilometers for CFL stability.
        graph: NetworkX MultiDiGraph representing the road network.
        roundabouts: List of roundabout node sequences.
        found_types: Set of road types discovered in the network.
        node_coordinates: Dictionary mapping node IDs to (x, y) coordinates.
        selected_types: List of road types selected for the network.
        hwy_filter: Hierarchical list of road type groups for filtering.
        road_params: Road parameters configuration loaded from JSON file.
    """

    MOTORWAY_TYPES: Tuple[str, str] = ("highway.motorway", "highway.motorway_link")

    def __init__(
        self,
        net_xml_path: str,
        road_params_config_path: str,
        target_cell_length: float = 0.3,
        # hwy_filter: Union[list[Tuple[str, str]], None] = None,
        min_link_length: Union[float, None] = None,
    ):
        """Initialize the network arbitrator.

        Args:
            net_xml_path: Path to the SUMO network XML file (.net.xml).
            road_params_config_path: Path to JSON configuration file containing
                road parameters (lane_capacity, jam_density, free_flow_speed for each road type).
            target_cell_length: Target length for cells in the macroscopic network (km).
            hwy_filter: Optional hierarchical list of road type groups for filtering.
                If None, uses default hierarchy: motorway > trunk > primary > secondary > tertiary.
                Format: [["motorway", "motorway_link"], ["trunk", "trunk_link"], ...]
            min_link_length: Minimum acceptable link length in kilometers for CFL stability.
                If specified, links shorter than this threshold are either stretched (if > 50% of minimum)
                or fused by contracting their nodes (if <= 50% of minimum). If None, no short link
                handling is performed. Should be set based on max_free_flow_speed * dt + margin.
        """
        self.path: str = net_xml_path
        self.target_cell_length: float = target_cell_length
        self.min_link_length: Union[float, None] = min_link_length
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.roundabouts: list[list[str]] = []
        self.found_types: set[str] = set()
        self.node_coordinates: dict[str, Tuple[float, float]] = {}
        self.selected_types: Union[Tuple[str, str], None] = None

        # set the hierarchical filter for road types, allowing for
        # flexible specification of type groups and priorities
        # self.hwy_filter: list[Tuple[str, str]] = (
        #     hwy_filter
        #     if hwy_filter is not None
        #     else [
        #         ("motorway", "motorway_link"),
        #         ("trunk", "trunk_link"),
        #         ("primary", "primary_link"),
        #         ("secondary", "secondary_link"),
        #         ("tertiary", "tertiary_link"),
        #     ]
        # )

        # load road parameters from config file
        self.road_params: RoadParamsConfig = (
            NetworkArbitrator._load_road_params_from_json(road_params_config_path)
        )

    @staticmethod
    def _validate_road_params(params: dict) -> None:
        """Validate road parameters configuration.

        Ensures all required road types are present with correct parameter structure.
        Validates that all parameter values are positive numbers.

        Args:
            params: Dictionary containing road parameters to validate.

        Raises:
            ValueError: If required road types are missing, parameters are missing,
                or parameter values are invalid (not positive numbers).
        """
        required_road_types = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "default",
        ]
        required_params = ["lane_capacity", "jam_density", "free_flow_speed"]

        # check all required road types are present
        for road_type in required_road_types:
            if road_type not in params:
                raise ValueError(
                    f"Missing required road type '{road_type}' in road parameters configuration."
                )

            # check all required parameters are present for this road type
            for param in required_params:
                if param not in params[road_type]:
                    raise ValueError(
                        f"Missing required parameter '{param}' for road type '{road_type}'."
                    )

                # validate parameter type and value
                value = params[road_type][param]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Parameter '{param}' for road type '{road_type}' must be a number, got {type(value).__name__}."
                    )
                if value <= 0:
                    raise ValueError(
                        f"Parameter '{param}' for road type '{road_type}' must be positive, got {value}."
                    )

    @staticmethod
    def _load_road_params_from_json(config_path: str) -> RoadParamsConfig:
        """Load and validate road parameters from JSON configuration file.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            Validated road parameters configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If the configuration is invalid (see _validate_road_params).
        """
        try:
            with open(config_path, "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Road parameters configuration file not found: {config_path}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in road parameters configuration file: {config_path}",
                e.doc,
                e.pos,
            )

        # validate the loaded parameters
        NetworkArbitrator._validate_road_params(params)

        return params

    def run(
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
        """Execute the complete network arbitration pipeline.

        Orchestrates the full workflow: parsing SUMO XML, eliminating roundabouts,
        filtering by road type, merging serial edges, handling short links for CFL
        stability, and instantiating the macroscopic network with appropriate parameters.

        Returns:
            A tuple containing:
                - macroscopic_network: Network object representing the consolidated macroscopic network.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - road_params: Road parameters configuration used for the network.
                - diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs
                    for their outgoing motorway links. Use this to compute turning rates from
                    detector data via TurningRateAggregator.

        Raises:
            ValueError: If no edges are found after parsing or if no matching road types exist.
        """
        # Step 1: Parse SUMO XML and build initial graph
        self.parse_sumo_xml()
        if self.graph.number_of_edges() == 0:
            raise ValueError(
                "No edges found after parsing SUMO network. Check the network file or highway filter."
            )

        # Step 2: Eliminate roundabouts by collapsing them into single nodes
        self.eliminate_roundabouts()

        # Step 3: Filter the network to keep only the largest connected component and remove isolated nodes
        self.filter()

        # Step 4: Merge serial edges to simplify the network while preserving junctions
        self.merge_serial_edges()

        # Step 5: Handle short links to ensure CFL stability
        self.handle_short_links()

        # Step 6: Instantiate macroscopic network objects and assign parameters based on road types
        (
            macroscopic_network,
            origin_ids,
            onramp_ids,
            offramp_ids,
            destination_ids,
            diverge_node_info,
            backbone_node_ids,
        ) = self.instantiate_network()
        self._log_network_statistics(macroscopic_network)

        return (
            macroscopic_network,
            origin_ids,
            onramp_ids,
            offramp_ids,
            destination_ids,
            self.road_params,
            diverge_node_info,
            backbone_node_ids,
        )

    def parse_sumo_xml(self) -> None:
        """Parse SUMO .net.xml file and extract network topology.

        Reads the SUMO network XML file and constructs a NetworkX MultiDiGraph
        representation. Performs hierarchical road type filtering to select the
        highest priority roads present in the network. Extracts node coordinates
        and normalizes them to origin. Identifies roundabouts for later processing.

        The method populates self.graph with edges containing attributes: id, length,
        speed, lanes, and type.

        Raises:
            ValueError: If no matching road types are found in the network.
        """
        tree = ET.parse(self.path)
        root = tree.getroot()

        # extract all available road types in the network for filtering
        # available_types = set()
        # for edge in root.findall("edge"):
        #     if edge.get("function") == "internal":
        #         continue
        #     edge_type = edge.get("type", "")
        #     if edge_type:
        #         available_types.add(edge_type)

        # # select highest priority level available
        # for priority_level in self.hwy_filter:
        #     # check if ANY type from this priority exists
        #     matching = [
        #         t
        #         for t in priority_level
        #         if any(t in avail for avail in available_types)
        #     ]

        #     if matching:
        #         self.selected_types = priority_level
        #         print(f"Selected road types: {self.selected_types}")
        #         break

        # if self.selected_types is None:
        #     raise ValueError(
        #         f"No matching road types found in SUMO network. "
        #         f"Available types: {sorted(available_types)}, "
        #         f"Filter priorities: {self.hwy_filter}"
        #     )

        # Restrict selection to motorway types for backbone extraction.
        # This arbitrator currently focuses on motorway/motorway_link as the
        # backbone; the hierarchical filter machinery is present but unused.
        self.selected_types = NetworkArbitrator.MOTORWAY_TYPES

        # extract junction coordinates
        raw_coordinates = {}
        for junction in root.findall("junction"):
            junction_id = junction.get("id")
            x_coord = junction.get("x")
            y_coord = junction.get("y")
            if x_coord is not None and y_coord is not None:
                raw_coordinates[junction_id] = (
                    float(x_coord),
                    float(y_coord),
                )

        # normalize coordinates (shift to origin)
        if raw_coordinates:
            min_x = min(c[0] for c in raw_coordinates.values())
            min_y = min(c[1] for c in raw_coordinates.values())
            self.node_coordinates = {
                junction_id: (c[0] - min_x, c[1] - min_y)
                for junction_id, c in raw_coordinates.items()
            }

        # parse roundabouts
        for roundabout in root.findall("roundabout"):
            self.roundabouts.append(roundabout.get("nodes", "").split())

        # parse edges
        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue

            edge_type = edge.get("type", "")
            self.found_types.add(edge_type)

            if not any(selected in edge_type for selected in self.selected_types):
                continue

            lanes = edge.findall("lane")
            if not lanes:
                continue

            # convert SUMO units to SI units
            length_km = float(lanes[0].get("length", 0)) / 1000.0
            speed_kmh = float(lanes[0].get("speed", 0)) * 3.6

            if length_km <= 0 or speed_kmh <= 0:
                warnings.warn(
                    f"Edge {edge.get('id')} has non-positive length or speed. Skipping.",
                    stacklevel=2,
                )
                continue

            # add edge to graph with attributes
            self.graph.add_edge(
                edge.get("from"),
                edge.get("to"),
                id=edge.get("id"),
                length=length_km,
                speed=speed_kmh,
                lanes=len(lanes),
                type=edge_type,
            )

        if "highway.motorway" not in self.found_types:
            raise ValueError(
                f"No motorway edges found in network '{self.path}'. "
                f"Available types: {sorted(self.found_types)}"
            )

    def eliminate_roundabouts(self) -> None:
        """Collapse roundabouts into single nodes.

        Merges all nodes in each roundabout into a single pivot node positioned
        at the centroid of the roundabout nodes. Internal roundabout edges are
        removed and their lengths are distributed to incident edges to preserve
        total network length.

        This simplification is necessary because macroscopic traffic models do not
        model the circular geometry of roundabouts explicitly. The consolidation
        maintains network connectivity while simplifying the topology.

        The method updates node coordinates to reflect the centroid position and
        uses NetworkX's contracted_nodes operation to merge the graph structure.
        """
        for nodes in self.roundabouts:
            valid_nodes = [n for n in nodes if self.graph.has_node(n)]
            if len(valid_nodes) <= 1:
                continue

            pivot = valid_nodes[0]

            # calculate internal roundabout length
            internal_length = 0
            for node in valid_nodes:
                for _, v, data in self.graph.out_edges(node, data=True):
                    if v in valid_nodes:
                        internal_length += data.get("length", 0)

            # calculate centroid for merged node position
            coordinates_to_merge = [
                self.node_coordinates.get(n, (0, 0)) for n in valid_nodes
            ]
            centroid_x = sum(c[0] for c in coordinates_to_merge) / len(
                coordinates_to_merge
            )
            centroid_y = sum(c[1] for c in coordinates_to_merge) / len(
                coordinates_to_merge
            )
            self.node_coordinates[pivot] = (centroid_x, centroid_y)

            # distribute internal length to incident edges
            extra_length = (internal_length / max(1, len(valid_nodes))) / 2.0

            # contract nodes into pivot
            for other in valid_nodes[1:]:
                if self.graph.has_node(other):
                    self.graph = nx.contracted_nodes(
                        self.graph, pivot, other, self_loops=False
                    )

            # add extra length to incident edges
            for u, v, _, data in self.graph.edges(keys=True, data=True):  # type: ignore
                if u == pivot or v == pivot:
                    data["length"] += extra_length

    def filter(self) -> None:
        """Remove isolated nodes and keep only the largest connected component.

        Cleans the network by removing isolated nodes that have no connections.
        If the network is not weakly connected, extracts and keeps only the
        largest weakly connected component. This ensures the resulting macroscopic
        network is a single connected graph.

        This filtering step is essential for ensuring that the final network
        represents a coherent, connected road system rather than disconnected
        fragments that cannot route traffic between them.
        """
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

        if self.graph.number_of_nodes() > 0 and not nx.is_weakly_connected(self.graph):
            largest = max(nx.weakly_connected_components(self.graph), key=len)
            self.graph = self.graph.subgraph(largest).copy()  # type: ignore
        elif nx.is_weakly_connected(self.graph):
            return
        else:
            raise ValueError(
                "No edges remain after filtering. Check the network file or highway filter."
            )

    def merge_serial_edges(self, max_iterations: int = 1000) -> None:
        """Merge serial edges, preserving junction structure.

        Identifies nodes with exactly one incoming and one outgoing edge (degree-2 nodes)
        and merges them into single longer edges. This simplification reduces the number
        of nodes while preserving the overall network topology and connectivity.

        The method iteratively merges edges until no more degree-2 nodes exist or the
        maximum iteration limit is reached. Edge attributes (length, lanes, speed) are
        appropriately combined during merging. Merging only occurs when adjacent edges
        have compatible characteristics (same number of lanes and similar speeds).

        This consolidation is important for macroscopic modeling as it reduces
        computational complexity while maintaining the essential network structure,
        particularly preserving important junctions where traffic flow decisions occur.
        """

        merge_count = 0
        for _ in range(max_iterations):
            candidates = [
                n
                for n in self.graph.nodes()
                if self.graph.in_degree(n) == 1 and self.graph.out_degree(n) == 1
            ]
            merged = False

            for n in candidates:
                nearby_is_complex = False
                for neighbor in list(self.graph.predecessors(n)) + list(
                    self.graph.successors(n)
                ):
                    if (
                        self.graph.in_degree(neighbor) > 1
                        or self.graph.out_degree(neighbor) > 1
                    ):
                        nearby_is_complex = True
                        break

                if nearby_is_complex:
                    continue

                in_edges = list(self.graph.in_edges(n, data=True))
                out_edges = list(self.graph.out_edges(n, data=True))
                u, _, d_in = in_edges[0]
                _, v, d_out = out_edges[0]

                same_lanes = d_in["lanes"] == d_out["lanes"]
                same_speed = abs(d_in["speed"] - d_out["speed"]) < 5.0

                same_type = d_in.get("type", "") == d_out.get("type", "")

                is_type_transition = ("motorway_link" in d_in.get("type", "")) != (
                    "motorway_link" in d_out.get("type", "")
                )

                if same_lanes and same_speed and same_type and not is_type_transition:
                    new_attr = {
                        "id": f"merged_{d_in['id']}_{d_out['id']}",
                        "length": d_in["length"] + d_out["length"],
                        "speed": min(d_in["speed"], d_out["speed"]),
                        "lanes": d_in["lanes"],
                        "type": d_in.get("type", "default"),
                    }

                    self.graph.add_edge(u, v, **new_attr)
                    self.graph.remove_node(n)

                    if n in self.node_coordinates:
                        del self.node_coordinates[n]

                    merged = True
                    merge_count += 1
                    break

            if not merged:
                break

    def handle_short_links(self) -> None:
        """Handle links shorter than the minimum required length for CFL stability.

        Links are processed using a dual-strategy approach:
        - Links with length > 50% of minimum: stretched to min_link_length
        - Links with length <= 50% of minimum: removed by contracting nodes

        This ensures CFL condition compliance while preserving meaningful network
        segments and removing negligible connectors. The 50% threshold balances
        preservation of network structure against removal of problematic segments.

        The method processes the shortest link at a time and re-evaluates after each
        operation to prevent cascading contractions of adjacent short links. This
        greedy approach ensures links are only merged when truly necessary.
        """
        if self.min_link_length is None:
            raise ValueError(
                "Minimum link length must be specified for short link handling."
            )

        all_stretched_links = []
        all_fused_links = []
        threshold = 0.5 * self.min_link_length
        max_iterations = 1000
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # identify all short links
            short_links = []
            for u, v, key, data in list(
                self.graph.edges(keys=True, data=True)
            ):  # use list() to avoid modification during iteration
                length = data.get("length", 0)
                if length < self.min_link_length:
                    short_links.append((length, u, v, key, data))

            # if no short links found, we're done
            if not short_links:
                break

            # sort by length to process the shortest first
            short_links.sort(key=lambda x: x[0])

            # process only the shortest link
            length, u, v, key, data = short_links[0]
            link_id = data.get("id", f"{u}->{v}")
            link_type = data.get("type", "")

            if length > threshold:
                # stretch the link to minimum length
                old_length = data["length"]
                data["length"] = self.min_link_length
                all_stretched_links.append((link_id, old_length, self.min_link_length))
            else:
                if "motorway_link" in link_type:
                    # preserve onramps/offramps: stretch instead of fusing
                    old_length = data["length"]
                    data["length"] = self.min_link_length
                    all_stretched_links.append(
                        (link_id, old_length, self.min_link_length)
                    )
                    continue

                upstream_uniform_type = self.graph.in_degree(u) > 0 and all(
                    edge_data.get("type", "") == link_type
                    for _, _, edge_data in self.graph.in_edges(u, data=True)
                )

                downstream_uniform_type = self.graph.out_degree(v) > 0 and all(
                    edge_data.get("type", "") == link_type
                    for _, _, edge_data in self.graph.out_edges(v, data=True)
                )

                if upstream_uniform_type or downstream_uniform_type:
                    # fuse the link by contracting its boundary nodes, keeping nearby link types unchanged
                    all_fused_links.append((link_id, length))

                    if self.graph.has_edge(u, v, key):
                        self.graph.remove_edge(u, v, key)

                    if self.graph.has_node(u) and self.graph.has_node(v) and u != v:
                        # keep the node with more outgoing links to preserve turning-rate references
                        pivot, other = (u, v)
                        if self.graph.out_degree(v) > self.graph.out_degree(u):
                            pivot, other = v, u

                        # update coordinates to midpoint
                        if (
                            pivot in self.node_coordinates
                            and other in self.node_coordinates
                        ):
                            pivot_coord = self.node_coordinates[pivot]
                            other_coord = self.node_coordinates[other]
                            self.node_coordinates[pivot] = (
                                (pivot_coord[0] + other_coord[0]) / 2,
                                (pivot_coord[1] + other_coord[1]) / 2,
                            )
                            if other in self.node_coordinates:
                                del self.node_coordinates[other]

                        try:
                            self.graph = nx.contracted_nodes(
                                self.graph, pivot, other, self_loops=False
                            )
                        except nx.NetworkXError:
                            warnings.warn(
                                f"Failed to contract nodes {u} and {v} for link {link_id}. Nodes may have been modified already.",
                                stacklevel=2,
                            )
                    continue

                # fallback: stretch when merging would cross a type boundary
                old_length = data["length"]
                data["length"] = self.min_link_length
                all_stretched_links.append((link_id, old_length, self.min_link_length))

        # log statistics
        if all_stretched_links or all_fused_links:
            print("-" * 60)
            print("Short Link Handling:")
            print(
                f"  Minimum link length threshold: {self.min_link_length:.3f} km (50% threshold: {threshold:.3f} km)"
            )
            print(f"  Iterations: {iteration}")

            if all_stretched_links:
                print(f"  Stretched links: {len(all_stretched_links)}")
                for link_id, old_len, new_len in all_stretched_links[:5]:
                    print(
                        f"    - {link_id}: {old_len:.3f} km → {new_len:.3f} km (stretched)"
                    )
                if len(all_stretched_links) > 5:
                    print(f"    ... and {len(all_stretched_links) - 5} more")

            if all_fused_links:
                print(f"  Fused links (nodes contracted): {len(all_fused_links)}")
                for link_id, old_len in all_fused_links[:5]:
                    print(f"    - {link_id}: {old_len:.3f} km (fused)")
                if len(all_fused_links) > 5:
                    print(f"    ... and {len(all_fused_links) - 5} more")
            print("-" * 60)

    def _nodes_with_nonmotorway_incoming(self) -> set[str]:
        """Return the set of node IDs that have at least one non-motorway
        incoming edge in the full SUMO network."""
        tree = ET.parse(self.path)
        root = tree.getroot()
        reachable: set[str] = set()
        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            etype = edge.get("type", "")
            if "motorway" in etype:  # skip motorway and motorway_link
                continue
            to_node = edge.get("to")
            if to_node:
                reachable.add(to_node)
        return reachable

    def instantiate_network(
        self,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        list[str],
        dict[str, list[str]],
        set[str],
    ]:
        """Create macroscopic network objects from the processed graph.

        Converts the NetworkX graph representation into macroscopic network objects
        (Nodes, MotorwayLinks, Origins, Destinations). Assigns appropriate macroscopic
        model parameters based on road types and divides links into cells of
        appropriate length for numerical simulation.

        The method performs several key operations:
        - Creates Node objects with geographic coordinates
        - Instantiates MotorwayLink objects with parameters from ROAD_PARAMS
        - Divides each link into cells based on target_cell_length
        - Identifies network entry points (origins) and exit points (destinations)
        - Detects onramps (nodes with multiple incoming motorway links)
        - Identifies diverge nodes and tracks their outgoing SUMO edge IDs

        Note: Split ratios (turning rates) should be computed from detector data using
        TurningRateAggregator. Use compute_lane_based_splits() for fallback when
        detector data is unavailable.

        Returns:
            A tuple containing:
                - network: Network object containing all macroscopic nodes with their connections.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - diverge_node_info: Dictionary mapping diverge node IDs to lists of SUMO edge IDs
                    for their outgoing motorway links.
        """

        macro_nodes = {}
        total_cells = 0

        # track diverge node information (node ID -> SUMO edge IDs)
        diverge_node_info: dict[str, list[str]] = {}

        # onramp source: in_degree=0, all outgoing edges are motorway_link
        # offramp sink:  out_degree=0, all incoming edges are motorway_link
        onramp_source_nodes: set[str] = set()
        offramp_sink_nodes: set[str] = set()

        # for nid in self.graph.nodes():
        #     if self.graph.in_degree(nid) == 0:
        #         out_edges = list(self.graph.out_edges(nid, data=True))
        #         if out_edges and all(
        #             "motorway_link" in d.get("type", "") for _, _, d in out_edges
        #         ):
        #             onramp_source_nodes.add(str(nid))

        urban_reachable = self._nodes_with_nonmotorway_incoming()
        # Synthetic scenarios have no urban edges anywhere; in that case the
        # urban-reachability filter is meaningless and would wrongly demote
        # every onramp candidate to a mainline origin.
        topology_only = not urban_reachable

        for nid in self.graph.nodes():
            if self.graph.in_degree(nid) == 0:
                out_edges = list(self.graph.out_edges(nid, data=True))
                if out_edges and all(
                    "motorway_link" in d.get("type", "") for _, _, d in out_edges
                ):
                    if str(nid) not in urban_reachable and not topology_only:
                        warnings.warn(
                            f"Onramp candidate node '{nid}' has no non-motorway incoming "
                            f"edges in the SUMO network. It will be treated as a mainline "
                            f"origin (no urban demand can reach it). Check OSM coverage.",
                            stacklevel=2,
                        )
                        # Do NOT add to onramp_source_nodes — its Onramp link becomes a
                        # MotorwayLink, and the origin receives mainline-type demand only.
                    else:
                        onramp_source_nodes.add(str(nid))

            if self.graph.out_degree(nid) == 0:
                in_edges = list(self.graph.in_edges(nid, data=True))
                if in_edges and all(
                    "motorway_link" in d.get("type", "") for _, _, d in in_edges
                ):
                    offramp_sink_nodes.add(str(nid))

        # create node objects
        for nid in self.graph.nodes():
            n_obj = Node(id=str(nid))
            n_obj.set_position(*self.node_coordinates.get(nid, (0, 0)))
            macro_nodes[nid] = n_obj

        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("type", "default").lower()
            params: RoadTypeParams = cast(
                RoadTypeParams,
                next(
                    (val for key, val in self.road_params.items() if key in edge_type),
                    self.road_params["default"],
                ),
            )

            u_str, v_str = str(u), str(v)
            is_ramp = "highway.motorway_link" in edge_type

            if is_ramp and u_str in onramp_source_nodes:
                link = Onramp(
                    id=str(data["id"]),
                    length=data["length"],
                    lanes=data["lanes"],
                    lane_capacity=params["lane_capacity"],
                    free_flow_speed=params["free_flow_speed"],
                    jam_density=params["jam_density"],
                    origin_node_id=u_str,
                    destination_node_id=v_str,
                )
            elif is_ramp and v_str in offramp_sink_nodes:
                link = Offramp(
                    id=str(data["id"]),
                    lanes=data["lanes"],
                    lane_capacity=params["lane_capacity"],
                    free_flow_speed=params["free_flow_speed"],
                    jam_density=params["jam_density"],
                    origin_node_id=u_str,
                    destination_node_id=v_str,
                )
            else:
                link = MotorwayLink(
                    id=str(data["id"]),
                    length=data["length"],
                    lanes=data["lanes"],
                    lane_capacity=params["lane_capacity"],
                    free_flow_speed=params["free_flow_speed"],
                    jam_density=params["jam_density"],
                    origin_node_id=u_str,
                    destination_node_id=v_str,
                )
                num_cells = max(1, math.floor(data["length"] / self.target_cell_length))
                cell_len = data["length"] / num_cells
                for _ in range(num_cells):
                    link.add_cell(length=cell_len)
                    total_cells += 1

            # connect link to nodes
            macro_nodes[u].add_outgoing(link)
            macro_nodes[v].add_incoming(link)

            # track SUMO edge ID for this link's origin node (for turning rate detection)
            if u_str not in diverge_node_info:
                diverge_node_info[str(u)] = []
            diverge_node_info[str(u)].append(str(data["id"]))

        # assign Origins and Destinations, distinguishing ramp vs mainline
        origin_ids: list[str] = []
        onramp_ids: list[str] = []
        destination_ids: list[str] = []
        offramp_ids: list[str] = []

        for nid, node_obj in macro_nodes.items():
            nid_str = str(nid)

            if not node_obj.incoming:
                if any(isinstance(l, Onramp) for l in node_obj.outgoing):
                    # for l in node_obj.outgoing:
                    onramp_ids.append(nid_str)
                    # onramp_ids.append(l.id)
                    # onramp_ids.append(f'onramp_{nid_str}')

                    orig = Origin(id=f"origin_{nid}", destination_node_id=nid_str)
                    origin_ids.append(orig.id)
                else:
                    orig = Origin(id=f"origin_{nid}", destination_node_id=nid_str)
                    origin_ids.append(orig.id)
                node_obj.add_incoming(orig)

            if not node_obj.outgoing:
                if any(isinstance(l, Offramp) for l in node_obj.incoming):
                    # for l in node_obj.incoming:
                    offramp_ids.append(nid_str)
                    # offramp_ids.append(l.id)
                    # offramp_ids.append(f'offramp_{nid_str}')

                    dest = Destination(id=f"dest_{nid}", origin_node_id=nid_str)
                    destination_ids.append(dest.id)
                else:
                    dest = Destination(id=f"dest_{nid}", origin_node_id=nid_str)
                    destination_ids.append(dest.id)
                node_obj.add_outgoing(dest)

        backbone_node_ids: set[str] = (
            {str(nid) for nid in self.graph.nodes()}
            - onramp_source_nodes
            - offramp_sink_nodes
        )

        return (
            Network(nodes=list(macro_nodes.values())),
            origin_ids,
            onramp_ids,
            offramp_ids,
            destination_ids,
            diverge_node_info,
            backbone_node_ids,
        )

    def _log_network_statistics(self, network: Network) -> None:
        """Log summary statistics about the generated macroscopic network.

        Prints a formatted summary including number of nodes, links, origins,
        destinations, and total network length. This provides visibility into
        the network structure and helps validate the arbitration process.

        The statistics help users understand the scale and characteristics of
        the generated macroscopic network, enabling quick verification that the
        arbitration process produced reasonable results.

        Args:
            network: Network object containing the macroscopic network.
        """
        num_nodes = len(network)
        # num_links = sum(
        #     len(node.outgoing)
        #     for node in network
        #     if node.outgoing and not isinstance(node.outgoing[0], Destination)
        # )

        num_links = sum(
            1
            for node in network
            for link in node.outgoing
            if isinstance(link, MotorwayLink)
        )

        num_origins = sum(
            1
            for node in network
            if node.incoming and isinstance(node.incoming[0], Origin)
        )

        num_destinations = sum(
            1
            for node in network
            if node.outgoing and isinstance(node.outgoing[0], Destination)
        )

        total_length = sum(
            link.length
            for node in network
            for link in node.outgoing
            if isinstance(link, MotorwayLink)
        )

        print("=" * 60)
        print("Macroscopic Network Information:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Links: {num_links}")
        print(f"  Origins: {num_origins}")
        print(f"  Destinations: {num_destinations}")
        print(f"  Total network length: {total_length:.2f} km")
        print("=" * 60)

    def compute_lane_based_splits(
        self, network: Network
    ) -> dict[str, Callable[[float], dict[str, float]]]:
        """Compute lane-based split ratios for nodes as fallback.

        This method provides simple lane-proportional splits when detector data
        is unavailable. The splits are time-invariant and based on the ratio of
        lanes on each outgoing link. For nodes with a single outgoing link,
        returns a constant 1.0 (100% of traffic uses that link).

        Args:
            network: Network object containing all nodes and links.

        Returns:
            Dictionary mapping node IDs to time-invariant split functions.
            Each function returns a dictionary mapping link IDs to their lane-based fractions.
        """
        splits: dict[str, Callable[[float], dict[str, float]]] = {}

        for node in network:
            node_id = node.id

            outgoing_links = [
                link
                for link in node.outgoing
                if isinstance(link, (MotorwayLink, Onramp, Offramp, Destination))
            ]

            if len(outgoing_links) == 0:
                raise ValueError(
                    f"Node {node_id} has no outgoing links, cannot compute lane-based splits."
                )

            elif len(outgoing_links) == 1:
                # single outgoing link: turning rate = 1.0
                splits[node_id] = lambda t, ol=outgoing_links: {ol[0].id: 1.0}

            elif len(outgoing_links) >= 2:
                # multiple outgoing links: use lane-proportional splits
                total_lanes = sum(
                    (link.lanes if isinstance(link, (MotorwayLink, Offramp)) else 1)
                    for link in outgoing_links
                )

                # create time-invariant split function based on lane proportions
                splits[node_id] = lambda t, ol=outgoing_links, tl=total_lanes: {
                    link.id: (
                        (link.lanes if isinstance(link, (MotorwayLink, Offramp)) else 1)
                        / tl
                    )
                    for link in ol
                }

        return splits
