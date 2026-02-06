import math
import warnings
import networkx as nx
import xml.etree.ElementTree as ET
from typing import Union, Tuple, Callable

from traffic_flow_models.network.node import Node
from traffic_flow_models.network.motorway_link import MotorwayLink
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.network import Network


class NetworkArbitrator:
    """Convert SUMO microscopic networks into consolidated macroscopic networks.

    This class performs the arbitration between detailed microscopic road networks
    from SUMO and simplified macroscopic representations suitable for macroscopic modeling.
    The process involves filtering roads by type, merging serial edges, handling
    roundabouts, and instantiating network objects with appropriate parameters.

    The class uses hierarchical road type filtering to automatically select the
    highest priority road types present in the network (e.g., motorways, then trunks,
    then primary roads, etc.).

    Attributes:
        path: Path to the SUMO network XML file.
        target_cell_length: Target length for macroscopic link cells in kilometers (default: 0.3).
        graph: NetworkX MultiDiGraph representing the road network.
        roundabouts: List of roundabout node sequences.
        found_types: Set of road types discovered in the network.
        node_coordinates: Dictionary mapping node IDs to (x, y) coordinates.
        selected_types: List of road types selected for the network.
        hwy_filter: Hierarchical list of road type groups for filtering.
    """

    # TODO: we need to extract these values to a configuration file for better access
    # Capacity per lane, Jam density, Fundamental diagram exponent, Free-flow speed, Relaxation time, Anticipation factor (km²), Lane-changing sensitivity
    ROAD_PARAMS = {
        "motorway": {
            "lane_capacity": 2000.0,
            "jam_density": 150.0,
            "free_flow_speed": 120.0,
        },
        "trunk": {
            "lane_capacity": 1800.0,
            "jam_density": 160.0,
            "free_flow_speed": 100.0,
        },
        "primary": {
            "lane_capacity": 1600.0,
            "jam_density": 180.0,
            "free_flow_speed": 80.0,
        },
        "secondary": {
            "lane_capacity": 1200.0,
            "jam_density": 200.0,
            "free_flow_speed": 50.0,
        },
        "tertiary": {
            "lane_capacity": 1000.0,
            "jam_density": 210.0,
            "free_flow_speed": 30.0,
        },
        "default": {
            "lane_capacity": 1500.0,
            "jam_density": 160.0,
            "free_flow_speed": 60.0,
        },
    }

    def __init__(
        self,
        net_xml_path: str,
        target_cell_length: float = 0.3,
        hwy_filter: Union[list[Tuple[str, str]], None] = None,
    ):
        """Initialize the network arbitrator.

        Args:
            net_xml_path: Path to the SUMO network XML file (.net.xml).
            hwy_filter: Optional hierarchical list of road type groups for filtering.
                If None, uses default hierarchy: motorway > trunk > primary > secondary > tertiary.
                Format: [["motorway", "motorway_link"], ["trunk", "trunk_link"], ...]
        """
        self.path: str = net_xml_path
        self.target_cell_length: float = target_cell_length
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.roundabouts: list[list[str]] = []
        self.found_types: set[str] = set()
        self.node_coordinates: dict[str, Tuple[float, float]] = {}
        self.selected_types: Union[Tuple[str, str], None] = None

        # set the hierarchical filter for road types, allowing for
        # flexible specification of type groups and priorities
        self.hwy_filter: list[Tuple[str, str]] = (
            hwy_filter
            if hwy_filter is not None
            else [
                ("motorway", "motorway_link"),
                ("trunk", "trunk_link"),
                ("primary", "primary_link"),
                ("secondary", "secondary_link"),
                ("tertiary", "tertiary_link"),
            ]
        )

    def run(
        self,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        dict[str, Callable[[float], dict[str, float]]],
    ]:
        """Execute the complete network arbitration pipeline.

        Orchestrates the full workflow: parsing SUMO XML, eliminating roundabouts,
        filtering by road type, merging serial edges, and instantiating the macroscopic
        network with appropriate parameters.

        Returns:
            A tuple containing:
                - macroscopic_network: Network object representing the consolidated macroscopic network.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - splits: Dictionary mapping node IDs to their outgoing link split ratios
                    as time-dependent callable functions.

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

        # Step 5: Instantiate macroscopic network objects and assign parameters based on road types
        (
            macroscopic_network,
            origin_ids,
            onramp_ids,
            destination_ids,
            splits,
        ) = self.instantiate_network()
        self._log_network_statistics(macroscopic_network)

        return macroscopic_network, origin_ids, onramp_ids, destination_ids, splits

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
        available_types = set()
        for edge in root.findall("edge"):
            if edge.get("function") == "internal":
                continue
            edge_type = edge.get("type", "")
            if edge_type:
                available_types.add(edge_type)

        # select highest priority level available
        for priority_level in self.hwy_filter:
            # check if ANY type from this priority exists
            matching = [
                t
                for t in priority_level
                if any(t in avail for avail in available_types)
            ]

            if matching:
                self.selected_types = priority_level
                print(f"Selected road types: {self.selected_types}")
                break

        if self.selected_types is None:
            raise ValueError(
                f"No matching road types found in SUMO network. "
                f"Available types: {sorted(available_types)}, "
                f"Filter priorities: {self.hwy_filter}"
            )

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
                    f"Edge {edge.get('id')} has non-positive length or speed. Skipping."
                )

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

                if same_lanes and same_speed:
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

    def instantiate_network(
        self,
    ) -> Tuple[
        Network,
        list[str],
        list[str],
        list[str],
        dict[str, Callable[[float], dict[str, float]]],
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
        - Computes split ratios at diverge points based on lane counts

        Returns:
            A tuple containing:
                - network: Network object containing all macroscopic nodes with their connections.
                - origin_ids: List of origin node IDs in the network.
                - onramp_ids: List of onramp node IDs in the network.
                - destination_ids: List of destination node IDs in the network.
                - splits: Dictionary mapping node IDs to their outgoing link split ratios
                    as time-dependent callable functions.
        """

        macro_nodes = {}
        total_cells = 0

        for nid in self.graph.nodes():
            n_obj = Node(id=str(nid))
            n_obj.set_position(*self.node_coordinates.get(nid, (0, 0)))
            macro_nodes[nid] = n_obj

        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("type", "default").lower()
            params = next(
                (val for key, val in self.ROAD_PARAMS.items() if key in edge_type),
                self.ROAD_PARAMS["default"],
            )

            # initiate motorway link and directly connect it to the connected nodes
            link = MotorwayLink(
                id=str(data["id"]),
                length=data["length"],
                lanes=data["lanes"],
                lane_capacity=params["lane_capacity"],
                free_flow_speed=params["free_flow_speed"],
                jam_density=params["jam_density"],
                origin_node_id=str(u),
                destination_node_id=str(v),
            )

            # connect link to nodes
            macro_nodes[u].add_outgoing(link)
            macro_nodes[v].add_incoming(link)

            # TODO: remove the creation of cells here if they are not used for the data aggregation and calibration later on -> partitioning performed automatically during simulation
            num_cells = max(1, math.ceil(data["length"] / self.target_cell_length))
            cell_len = data["length"] / num_cells
            for _ in range(num_cells):
                link.add_cell(length=cell_len)
                total_cells += 1

        for nid, node_obj in macro_nodes.items():
            if not node_obj.incoming:
                orig = Origin(id=f"origin_{nid}", destination_node_id=str(nid))
                node_obj.add_incoming(orig)

            if not node_obj.outgoing:
                dest = Destination(id=f"dest_{nid}", origin_node_id=str(nid))
                node_obj.add_outgoing(dest)

        origin_ids: list[str] = [
            node_obj.incoming[0].id
            for node_obj in macro_nodes.values()
            if node_obj.incoming and isinstance(node_obj.incoming[0], Origin)
        ]

        destination_ids: list[str] = [
            node_obj.outgoing[0].id
            for node_obj in macro_nodes.values()
            if node_obj.outgoing and isinstance(node_obj.outgoing[0], Destination)
        ]

        onramp_ids: list[str] = [
            f"onramp_{nid}"
            for nid, node_obj in macro_nodes.items()
            if len(
                [link for link in node_obj.incoming if isinstance(link, MotorwayLink)]
            )
            >= 2
        ]

        # TODO: splits should not be computed based on the number of lanes but on the actual traffic distribution observed from the micro simulation
        # TODO: additionally, turning rates should be time-dependent callable functions (like demand)
        splits: dict[str, Callable[[float], dict[str, float]]] = {}
        for nid, node_obj in macro_nodes.items():
            outgoing_links = [
                link for link in node_obj.outgoing if isinstance(link, MotorwayLink)
            ]

            if len(outgoing_links) >= 2:
                total_lanes = sum(link.lanes for link in outgoing_links)
                # splits[str(nid)] = {
                #     link.id: link.lanes / total_lanes for link in outgoing_links
                # }
                splits[str(nid)] = lambda t, ol=outgoing_links: {
                    link.id: link.lanes / total_lanes for link in ol
                }

        return (
            Network(nodes=list(macro_nodes.values())),
            origin_ids,
            onramp_ids,
            destination_ids,
            splits,
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
        num_links = sum(
            len(node.outgoing)
            for node in network
            if node.outgoing and not isinstance(node.outgoing[0], Destination)
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
