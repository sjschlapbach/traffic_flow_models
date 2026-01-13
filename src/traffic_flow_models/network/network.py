from __future__ import annotations

import numpy as np
import warnings
import casadi
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import TYPE_CHECKING, Iterator, Callable, Mapping, Optional
from numpy.typing import NDArray

from traffic_flow_models.network.node import Node
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.motorway_link import MotorwayLink

if TYPE_CHECKING:
    from traffic_flow_models.model.ctm import CTM
    from traffic_flow_models.model.metanet import METANET


class Network:
    """
    Network class representing the traffic network.

    The network class is the main container for all network components,
    keeping track of all nodes included in the network alongside their connected
    links. Additionally, it is responsible for validating the structure of a network
    before simulation and the initialization of all necessary parameters (including
    demands, split ratios, densities downstream of destinations, etc.).

    Requirements:
        - Each network must have at least one origin link or onramp
        - Each network must have at least one destination
        - Each offramp needs to be connected to a destination
        - Each node needs to have at least one incoming and one outgoing link
          (including origins and destinations beyond regular links)
        - All nodes in the network must be connected through links (no unconnected components)
    """

    def __init__(self, nodes: list[Node] = []) -> None:
        """Initialize the Network object."""
        self._nodes: list[Node] = []

        for n in nodes:
            self.add_node(n)

    def add_node(self, node: Node) -> None:
        """
        Add a `Node` instance to the network.

        Raises:
            TypeError: if `node` is not a Node instance.
            ValueError: if a node with the same `id` is already present.
        """
        if not isinstance(node, Node):
            raise TypeError("Only Node instances may be added to the Network.")

        if any(
            getattr(n, "id", None) == getattr(node, "id", None) for n in self._nodes
        ):
            raise ValueError(f"Node with id {node.id} already present in network.")

        self._nodes.append(node)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the network by object or id.

        Args:
            node_id: The `id` of the `Node` instance to remove.

        Raises:
            ValueError: if no node with the given id is found.
        """
        for n in self.list_nodes():
            if getattr(n, "id", None) == node_id:
                self._nodes.remove(n)
                return

        raise ValueError(f"No node with id {node_id} found in network.")

    def get_node(self, id: str) -> Node | None:
        """Return the node with the given id or None if absent."""
        for n in self.list_nodes():
            if getattr(n, "id", None) == id:
                return n

        return None

    def list_nodes(self) -> list[Node]:
        """Return a shallow copy of the node list."""
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        for n in self.list_nodes():
            yield n

    def validate(self) -> bool:
        """
        Validate network structure according to class requirements.

        Requirements validated:
            - Each network must have at least one origin link or onramp
            - Each network must have at least one destination
            - Each offramp needs to be connected to a destination
            - Each node needs to have at least one incoming and one outgoing link
            - All nodes in the network must be connected through links

        Raises:
            ValueError: if any of the requirements are violated.
        """

        # verify that the network is not empty
        if len(self._nodes) < 2:
            raise ValueError("Network contains less than 2 nodes.")

        # call node-level validation (ensures at least one incoming & outgoing per node)
        for node in self.list_nodes():
            if not isinstance(node, Node):
                raise TypeError("Network contains non-Node object.")

            node.validate()

        # ensure that the network has at least one origin / onramp and one destination
        has_origin_or_onramp = False
        has_destination = False

        # collect all destination instances present in network (to check offramp targets)
        dests: set[Destination] = set()

        for node in self.list_nodes():
            for link in list(node.incoming) + list(node.outgoing):
                if isinstance(link, (Origin, Onramp)):
                    has_origin_or_onramp = True
                if isinstance(link, Destination):
                    has_destination = True
                    dests.add(link)
                if isinstance(link, Offramp) and link.destination is not None:
                    has_destination = True
                    dests.add(link.destination)

        if not has_origin_or_onramp:
            raise ValueError("Network must contain at least one origin or onramp link.")

        if not has_destination:
            raise ValueError("Network must contain at least one destination.")

        # check that every offramp has a destination
        for node in self.list_nodes():
            for link in list(node.incoming) + list(node.outgoing):
                if isinstance(link, Offramp):
                    if link.destination is None:
                        raise ValueError("Offramp is not connected to a destination.")

        # connectivity: use two DFS passes (original and reversed edges)
        # build adjacency (directed) between nodes: A -> B if any link in A.outgoing is in B.incoming
        num_nodes = len(self._nodes)

        # validate that each node's incoming links have their destination_node_id set
        # to this node's id, and each outgoing link has its origin_node_id set to
        # this node's id. This ensures the per-link origin/destination metadata is
        # consistent with the node topology.
        for node in self.list_nodes():
            for link in node.incoming:
                if hasattr(link, "destination_node_id"):
                    if link.destination_node_id is None:
                        raise ValueError(
                            f"Incoming link {getattr(link,'id',repr(link))} has no destination_node_id set for node {node.id}"
                        )
                    if link.destination_node_id != node.id:
                        raise ValueError(
                            f"Incoming link {getattr(link,'id',repr(link))} destination_node_id mismatch: expected {node.id}, got {link.destination_node_id}"
                        )
                else:
                    raise ValueError(
                        f"Incoming link {getattr(link,'id',repr(link))} missing destination_node_id attribute for node {node.id}"
                    )

            for link in node.outgoing:
                if hasattr(link, "origin_node_id"):
                    if link.origin_node_id is None:
                        raise ValueError(
                            f"Outgoing link {getattr(link,'id',repr(link))} has no origin_node_id set for node {node.id}"
                        )
                    if link.origin_node_id != node.id:
                        raise ValueError(
                            f"Outgoing link {getattr(link,'id',repr(link))} origin_node_id mismatch: expected {node.id}, got {link.origin_node_id}"
                        )
                else:
                    raise ValueError(
                        f"Outgoing link {getattr(link,'id',repr(link))} missing origin_node_id attribute for node {node.id}"
                    )

        # build a mapping from node id -> index for quick lookups
        node_id_to_index: dict[str, int] = {
            n.id: i for i, n in enumerate(self.list_nodes())
        }

        # build directed adjacency using each outgoing link's destination id.
        adj_forward: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        for i, node in enumerate(self.list_nodes()):
            for link in node.outgoing:
                dest_id = getattr(link, "destination_node_id", None)
                if dest_id is None:
                    continue

                j = node_id_to_index.get(dest_id)
                if j is not None and i != j:
                    adj_forward[i].add(j)

        # reversed adjacency
        adj_reversed: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        for u, nbrs in adj_forward.items():
            for v in nbrs:
                adj_reversed[v].add(u)

        # choose a start node (index 0)
        start = 0
        vis1 = [False] * num_nodes
        vis2 = [False] * num_nodes

        # iterative DFS on original graph
        stack = [start]
        while stack:
            cur = stack.pop()
            if vis1[cur]:
                continue

            vis1[cur] = True
            for nb in adj_forward.get(cur, ()):
                if not vis1[nb]:
                    stack.append(nb)

        # iterative DFS on reversed graph
        stack = [start]
        while stack:
            cur = stack.pop()
            if vis2[cur]:
                continue
            vis2[cur] = True
            for nb in adj_reversed.get(cur, ()):
                if not vis2[nb]:
                    stack.append(nb)

        # any node that is neither reachable from start nor can reach start is disconnected
        unconnected = [
            self._nodes[i].id for i in range(num_nodes) if (not vis1[i] and not vis2[i])
        ]
        if unconnected:
            raise ValueError(f"Network contains unconnected components: {unconnected}")

        # all checks passed
        return True

    def network_dict_to_state_vec(
        self,
        flow_dict: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        density_dict: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        speed_dict: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        origin_queue_dict: dict[str, float] | dict[str, casadi.SX],
        onramp_queue_dict: dict[str, float] | dict[str, casadi.SX],
        offramp_queue_dict: dict[str, float] | dict[str, casadi.SX],
    ):
        """Convert dictionary-based network state to a packed 1-D state vector.

        The network's per-link and per-origin/onramp state is provided as
        dictionaries keyed by link/origin ids. This helper concatenates those
        arrays/values into a single 1-D array in the deterministic node-
        and link-ordering used by the simulator. The packing order is:
        - For each node: incoming origin/onramp flows and queues (one value each)
        - For each node: outgoing motorway link flows, densities, speeds (values for each cell)
        - For each node: outgoing offramp flow and queue
        - For each node: outgoing destination flow

        Args:
            flow_dict: Mapping link id -> per-cell flow array (veh/h).
            density_dict: Mapping link id -> per-cell density array (veh/km/lane).
            speed_dict: Mapping link id -> per-cell speed array (km/h).
            origin_queue_dict: Mapping origin id -> scalar queue length (veh).
            onramp_queue_dict: Mapping onramp id -> scalar queue length (veh).
            offramp_queue_dict: Mapping offramp id -> scalar queue length (veh).

        Returns:
            System state containing all network variables for simluation

        Raises:
            ValueError: If required arrays/values are missing or have incorrect sizes.
        """

        # combine state from separate arrays into single state vector if required by model
        # structure: [ flows | densities | speeds | origin_queues | onramp_queues ]
        x = (
            np.array([], dtype=np.float64)
            if isinstance(next(iter(flow_dict.values())), np.ndarray)
            else casadi.SX()
        )

        # initialize the counters for all quantities contained in the system vectors
        num_flows = 0
        num_densities = 0
        num_speeds = 0
        num_origin = 0
        num_onramp = 0
        num_offramp = 0
        num_splits = 0
        num_destinations = 0

        for node in self.list_nodes():
            # initialize incoming links (only origins and onramps - only flow and queue)
            for link in node.incoming:
                if isinstance(link, Origin) or isinstance(link, Onramp):
                    init_flow = flow_dict[link.id]
                    if isinstance(init_flow, np.ndarray) and len(init_flow) == 1:
                        x = np.concatenate((x, init_flow))
                        num_flows += 1
                    elif isinstance(init_flow, casadi.SX) and init_flow.size1() == 1:
                        x = casadi.vertcat(x, init_flow)
                        num_flows += 1
                    else:
                        raise ValueError(
                            f"Initial flow for network inflow {link.id} (type: {type(link)}) must be an array of length 1."
                        )

                    if link.id in onramp_queue_dict and isinstance(link, Onramp):
                        num_onramp += 1
                        if isinstance(onramp_queue_dict[link.id], float):
                            x = np.concatenate(
                                (x, np.array([onramp_queue_dict[link.id]]))
                            )
                        elif isinstance(onramp_queue_dict[link.id], casadi.SX):
                            x = casadi.vertcat(x, onramp_queue_dict[link.id])
                        else:
                            raise ValueError(
                                f"Initial queue for onramp {link.id} must be a scalar."
                            )

                    elif link.id in origin_queue_dict and isinstance(link, Origin):
                        num_origin += 1
                        if isinstance(origin_queue_dict[link.id], float):
                            x = np.concatenate(
                                (x, np.array([origin_queue_dict[link.id]]))
                            )
                        elif isinstance(origin_queue_dict[link.id], casadi.SX):
                            x = casadi.vertcat(x, origin_queue_dict[link.id])
                        else:
                            raise ValueError(
                                f"Initial queue for origin {link.id} must be a scalar."
                            )
                    else:
                        raise ValueError(
                            f"Initial queue for network inflow {link.id} (type: {type(link)}) must be provided."
                        )

            # initialize outgoing links (full for motorlinks, queue and flow for offramp, flow for destination)
            for link in node.outgoing:
                num_splits += len(node.outgoing)

                if isinstance(link, MotorwayLink):
                    num_cells = len(link)
                    init_flow = flow_dict[link.id]
                    if (
                        isinstance(init_flow, np.ndarray)
                        and len(init_flow) == num_cells
                    ):
                        x = np.concatenate((x, init_flow))
                        num_flows += num_cells
                    elif (
                        isinstance(init_flow, casadi.SX)
                        and init_flow.size1() == num_cells
                    ):
                        x = casadi.vertcat(x, init_flow)
                        num_flows += num_cells
                    else:
                        raise ValueError(
                            f"Initial flow for motorway link {link.id} must be an array of length {num_cells}."
                        )

                    init_density = density_dict[link.id]
                    if isinstance(init_density, np.ndarray) and link.id in density_dict:
                        x = np.concatenate((x, init_density))
                        num_densities += num_cells
                    elif (
                        isinstance(init_density, casadi.SX)
                        and init_density.size1() == num_cells
                    ):
                        x = casadi.vertcat(x, init_density)
                        num_densities += num_cells
                    else:
                        raise ValueError(
                            f"Initial density for motorway link {link.id} must be an array of length {num_cells}."
                        )

                    init_speed = speed_dict[link.id]
                    if isinstance(init_speed, np.ndarray) and link.id in speed_dict:
                        x = np.concatenate((x, init_speed))
                        num_speeds += num_cells
                    elif (
                        isinstance(init_speed, casadi.SX)
                        and init_speed.size1() == num_cells
                    ):
                        x = casadi.vertcat(x, init_speed)
                        num_speeds += num_cells
                    else:
                        raise ValueError(
                            f"Initial speed for motorway link {link.id} must be an array of length {num_cells}."
                        )

                elif isinstance(link, Offramp):
                    # offramp: store-and-forward model with single cell
                    init_flow = flow_dict[link.id]
                    if isinstance(init_flow, np.ndarray) and len(init_flow) == 1:
                        x = np.concatenate((x, init_flow))
                        num_flows += 1
                    elif isinstance(init_flow, casadi.SX) and init_flow.size1() == 1:
                        x = casadi.vertcat(x, init_flow)
                        num_flows += 1
                    else:
                        raise ValueError(
                            f"Initial flow for offramp {link.id} must be an array of length 1."
                        )

                    init_queue = offramp_queue_dict[link.id]
                    if (
                        isinstance(init_queue, (int, float))
                        and link.id in offramp_queue_dict
                    ):
                        x = np.concatenate((x, np.array([init_queue])))
                        num_offramp += 1
                    elif (
                        isinstance(init_queue, casadi.SX)
                        and link.id in offramp_queue_dict
                    ):
                        x = casadi.vertcat(x, init_queue)
                        num_offramp += 1
                    else:
                        raise ValueError(
                            f"Initial queue for offramp {link.id} must be a scalar."
                        )

                elif isinstance(link, Destination):
                    init_flow = flow_dict[link.id]
                    num_destinations += 1

                    if isinstance(init_flow, np.ndarray) and len(init_flow) == 1:
                        x = np.concatenate((x, init_flow))
                        num_flows += 1
                    elif isinstance(init_flow, casadi.SX) and init_flow.size1() == 1:
                        x = casadi.vertcat(x, init_flow)
                        num_flows += 1
                    else:
                        raise ValueError(
                            f"Initial flow for destination {link.id} must be an array of length 1."
                        )

        return (
            x,
            num_flows,
            num_densities,
            num_speeds,
            num_origin,
            num_onramp,
            num_offramp,
            num_splits,
            num_destinations,
        )

    def network_dict_to_disturbance_vec(
        self,
        origin_demand_dict: dict[str, float],
        onramp_demand_dict: dict[str, float],
        turning_rate_dict: dict[str, dict[str, float]],
        boundary_condition_dict: dict[str, float],
    ):
        """Pack origin/onramp demands and node turning rates into a vector.

        The disturbance vector contains all exogenous inputs required by the
        simulator: origin demands, onramp demands and per-node turning rates
        for outgoing links. The function iterates nodes in the network and
        concatenates the values in a deterministic order.

        Args:
            origin_demand_dict: Mapping origin id -> scalar demand (veh/h).
            onramp_demand_dict: Mapping onramp id -> scalar demand (veh/h).
            turning_rate_dict: Mapping node id -> mapping outgoing link id -> turn rate.
            boundary_condition_dict: Mapping destination id -> downstream density (veh/km/lane).

        Returns:
            A tuple `(d, num_turning_rates)` where `d` is a 1-D NumPy array of
            concatenated disturbances and `num_turning_rates` is the number of
            turning-rate scalars included.

        Raises:
            ValueError: If required demand or turning-rate entries are missing
                or inconsistent with the network topology.
        """

        # "disturbance" variables = network inflows and turning rates
        # structure: [ origin_demands | onramp_demands | turning_rates ]
        d = np.array([], dtype=np.float64)

        for node in self.list_nodes():
            # add turning rates for this node's outgoing links
            node_rates = turning_rate_dict[node.id]
            if node_rates is None:
                raise ValueError(
                    f"Turning rates for node {node.id} not provided in turning_rate_dict."
                )

            if len(node_rates) == 0 and len(node.outgoing) > 1:
                raise ValueError(
                    f"Node {node.id} has multiple outgoing links but no turning rates provided."
                )

            if any(link.id not in node_rates for link in node.outgoing):
                raise ValueError(
                    f"Turning rates for all outgoing links of node {node.id} must be provided."
                )

            for link in node.outgoing:
                d = np.concatenate((d, np.array([node_rates[link.id]])))

            # set values for incoming links (onramps or origins)
            for link in node.incoming:
                if isinstance(link, Onramp):
                    if link.id in onramp_demand_dict:
                        d = np.concatenate((d, np.array([onramp_demand_dict[link.id]])))
                    else:
                        raise ValueError(
                            f"Demand for onramp {link.id} must be provided."
                        )

                # initialize origins (queues only)
                elif isinstance(link, Origin):
                    if link.id in origin_demand_dict:
                        d = np.concatenate((d, np.array([origin_demand_dict[link.id]])))
                    else:
                        raise ValueError(
                            f"Demand for origin {link.id} must be provided."
                        )

            # set values for outgoing links (destinations / destinations connected to offramps)
            for link in node.outgoing:
                if isinstance(link, Destination):
                    if link.id in boundary_condition_dict:
                        d = np.concatenate(
                            (d, np.array([boundary_condition_dict[link.id]]))
                        )
                    else:
                        raise ValueError(
                            f"Boundary condition for destination {link.id} must be provided."
                        )

                if isinstance(link, Offramp):
                    if link.destination is not None:
                        dest_id = link.destination.id
                        if dest_id in boundary_condition_dict:
                            d = np.concatenate(
                                (d, np.array([boundary_condition_dict[dest_id]]))
                            )
                        else:
                            raise ValueError(
                                f"Boundary condition for destination {dest_id} (connected to offramp {link.id}) must be provided."
                            )
                    else:
                        raise ValueError(
                            f"Offramp {link.id} has no destination assigned."
                        )

        return d

    def state_vec_to_network_dict(
        self,
        x: NDArray[np.float64] | casadi.SX,
    ):
        """Unpack a packed state vector into structured dictionaries.

        Reverses the packing performed by `network_dict_to_state_vec`. The
        function accepts either a NumPy 1-D array or a CasADi SX column
        vector and returns dictionaries with per-link and per-origin/onramp
        entries keyed by their ids. The returned tuple has the form:
        ``(flows, densities, speeds, origin_queues, onramp_queues, offramp_queues)``.

        - ``flows``: mapping link id -> 1-D array or CasADi SX slice of per-cell flows
        - ``densities``: mapping motorway link id -> 1-D array or CasADi SX slice
        - ``speeds``: mapping motorway link id -> 1-D array or CasADi SX slice
        - ``origin_queues``: mapping origin id -> scalar queue value
        - ``onramp_queues``: mapping onramp id -> scalar queue value
        - ``offramp_queues``: mapping offramp id -> scalar queue value

        Args:
            x: 1-D NumPy array or CasADi SX column vector containing the packed state.

        Returns:
            Tuple of dictionaries as described above.

        Raises:
            ValueError: If the provided state vector is too short to extract
                the expected entries for the current network topology.
        """

        # initialize the structured dictionary containers for the state vector
        flows = dict[str, NDArray[np.float64] | casadi.SX]()
        densities = dict[str, NDArray[np.float64] | casadi.SX]()
        speeds = dict[str, NDArray[np.float64] | casadi.SX]()
        origin_queues = dict[str, float | casadi.SX]()
        onramp_queues = dict[str, float | casadi.SX]()
        offramp_queues = dict[str, float | casadi.SX]()
        i_state = 0

        # load the vector sizes depending on the data type
        state_size = len(x) if isinstance(x, np.ndarray) else int(x.size1())

        for node in self.list_nodes():
            # split up the state vector entries corresponding to incoming link data (onramps and origins only)
            for link in node.incoming:
                if isinstance(link, Origin) or isinstance(link, Onramp):
                    # verify that the state has enough values remaining (one value for flow and queue length each)
                    # onramps / origins are modeled as single cell store and forward links
                    if i_state + 2 > state_size:
                        raise ValueError(
                            "State vector too short to extract all link states (flow and queue length)."
                        )

                    flows[link.id] = x[i_state : i_state + 1]
                    i_state += 1

                    if isinstance(link, Onramp):
                        onramp_queues[link.id] = x[i_state]
                    else:
                        origin_queues[link.id] = x[i_state]
                    i_state += 1

            # split up the state vector entries corresponding to outgoing link data (full for motorlinks, queue and flow for offramp, flow for destination)
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    num_cells = len(link)

                    # verify that the state has enough values remaining (number of cell values for flow, density, and speed each)
                    if i_state + 3 * num_cells > state_size:
                        raise ValueError(
                            "State vector too short to extract all link states."
                        )

                    flows[link.id] = x[i_state : i_state + num_cells]
                    i_state += num_cells

                    densities[link.id] = x[i_state : i_state + num_cells]
                    i_state += num_cells

                    speeds[link.id] = x[i_state : i_state + num_cells]
                    i_state += num_cells

                elif isinstance(link, Offramp):
                    # verify that the state has enough values remaining (one value for flow and queue length each)
                    if i_state + 2 > state_size:
                        raise ValueError(
                            "State vector too short to extract all link states (flow and queue length for offramps)."
                        )

                    flows[link.id] = x[i_state : i_state + 1]
                    i_state += 1

                    offramp_queues[link.id] = x[i_state]
                    i_state += 1

                elif isinstance(link, Destination):
                    # verify that the state has enough values remaining (one value for flow)
                    if i_state + 1 > state_size:
                        raise ValueError(
                            "State vector too short to extract all link states (flow for destinations)."
                        )

                    flows[link.id] = x[i_state : i_state + 1]
                    i_state += 1

        return flows, densities, speeds, origin_queues, onramp_queues, offramp_queues

    def disturbance_vec_to_network_dict(
        self,
        d: NDArray[np.float64] | casadi.SX,
    ):
        """Unpack a disturbance vector into structured disturbance dictionaries.

        Reverses the packing performed by `network_dict_to_disturbance_vec`.
        Accepts a NumPy 1-D array or a CasADi SX column vector and returns
        four dictionaries keyed by ids:
        - ``origin_demands``: mapping origin id -> scalar demand (veh/time)
        - ``onramp_demands``: mapping onramp id -> scalar demand (veh/time)
        - ``turning_rates``: mapping node id -> (outgoing link id -> rate)
        - ``boundary_conditions``: mapping destination id -> downstream density

        The unpacking follows the node-ordering used in the network and will
        raise ValueError if the disturbance vector is too short or inconsistent
        with the network topology.

        Args:
            d: 1-D NumPy array or CasADi SX column vector containing the packed disturbances.

        Returns:
            Tuple of four dictionaries: ``(origin_demands, onramp_demands, turning_rates, boundary_conditions)``.

        Raises:
            ValueError: If the disturbance vector is too short for the network
                topology or cannot be parsed into the expected entries.
        """

        # initialize the structure dictionary containers for the disturbance vector
        origin_demands = dict[str, float | casadi.SX]()
        onramp_demands = dict[str, float | casadi.SX]()
        turning_rates = dict[str, dict[str, float | casadi.SX]]()
        boundary_conditions = dict[str, float | casadi.SX]()
        i_disturbance = 0

        # load the vector sizes depending on the data type
        disturbance_size = len(d) if isinstance(d, np.ndarray) else int(d.size1())

        for node in self.list_nodes():
            # split up the disturbance vector entries corresponding to the turning rates of this node
            node_turning_rates: dict[str, float | casadi.SX] = {}
            if i_disturbance + len(node.outgoing) > disturbance_size:
                raise ValueError(
                    "Disturbance vector too short to extract all turning rates."
                )

            for link in node.outgoing:
                node_turning_rates[link.id] = d[i_disturbance]
                i_disturbance += 1

            turning_rates[node.id] = node_turning_rates

            # split up the state vector entries corresponding to incoming link data (onramps and origins only)
            for link in node.incoming:
                if isinstance(link, Onramp):
                    if i_disturbance + 1 > disturbance_size:
                        raise ValueError(
                            "Disturbance vector too short to extract all onramp demands."
                        )

                    onramp_demands[link.id] = d[i_disturbance]
                    i_disturbance += 1

                # initialize origins (queues only)
                elif isinstance(link, Origin):

                    if i_disturbance + 1 > disturbance_size:
                        raise ValueError(
                            "Disturbance vector too short to extract all origin demands."
                        )

                    origin_demands[link.id] = d[i_disturbance]
                    i_disturbance += 1

            for link in node.outgoing:
                if isinstance(link, Destination):
                    if i_disturbance + 1 > disturbance_size:
                        raise ValueError(
                            "Disturbance vector too short to extract all destination boundary conditions."
                        )

                    boundary_conditions[link.id] = d[i_disturbance]
                    i_disturbance += 1

                if isinstance(link, Offramp):
                    if link.destination is not None:
                        if i_disturbance + 1 > disturbance_size:
                            raise ValueError(
                                "Disturbance vector too short to extract all destination boundary conditions."
                            )

                        boundary_conditions[link.destination.id] = d[i_disturbance]
                        i_disturbance += 1
                    else:
                        raise ValueError(
                            f"Offramp {link.id} has no destination assigned."
                        )

        return origin_demands, onramp_demands, turning_rates, boundary_conditions

    def simulate(
        self,
        duration: float,
        dt: float,
        # model: Union["CTM", "METANET"], # TODO: re-introduce this union of models as soon as CTM supports the new network structure
        model: METANET,
        origin_demands: dict[
            str, Callable[[float], float]
        ],  # for each origin id, provide a callable function returning the demand at time t
        onramp_demands: dict[
            str, Callable[[float], float]
        ],  # for each onramp id, provide a callable function returning the demand at time t
        turning_rates: dict[
            str, Callable[[float], dict[str, float]]
        ],  # for each node id, provide a callable function returning a dict mapping outgoing link ids to split ratios at time t
        destination_boundary_conditions: dict[
            str, Callable[[float], float]
        ],  # for each destination id, provide a callable function returning the downstream density at time t
        initial_flows: (
            dict[str, float | NDArray[np.float64]] | None
        ) = None,  # for each link id, provide either a float (uniform initial flow) or an array of floats (per-cell initial flows; default: 0)
        initial_densities: (
            dict[str, float | NDArray[np.float64]] | None
        ) = None,  # for each link id, provide either a float (uniform initial density) or an array of floats (per-cell initial densities; default: 0)
        initial_speeds: (
            dict[str, float | NDArray[np.float64]] | None
        ) = None,  # for each link id, provide either a float (uniform initial speed) or an array of floats (per-cell initial speeds; default: free-flow speed)
        initial_origin_queues: (
            dict[str, float] | None
        ) = None,  # for each origin id, provide the initial queue length (default: 0)
        initial_onramp_queues: (
            dict[str, float] | None
        ) = None,  # for each onramp id, provide the initial queue length (default: 0)
        initial_offramp_queues: (
            dict[str, float] | None
        ) = None,  # for each offramp id, provide the initial queue length (default: 0)
        preferred_cell_size: float = 0.5,  # preferred size of link segments (subject to CFL condition and link length divisibility)
        inflows_jam_density: float = 1800.0,  # jam density used for inflow links if initial densities not provided
        inflows_free_flow_speed: float = 100.0,  # free-flow speed used for inflow links if initial speeds not provided
        plot_results: bool = False,
    ):
        # ! 1 - validate all inputs as required
        for node in self.list_nodes():
            # validate node structure
            node.validate()

            # validate that origin demands for each origin are provided
            for link in node.incoming:
                if isinstance(link, Origin):
                    if link.id not in origin_demands:
                        raise ValueError(
                            f"Origin demand function for origin {link.id} not provided."
                        )

            # validate that onramp demands for each onramp are provided
            for link in node.incoming:
                if isinstance(link, Onramp):
                    if link.id not in onramp_demands:
                        raise ValueError(
                            f"Onramp demand function for onramp {link.id} not provided."
                        )

            # validate that turning rates for each node are provided
            if node.id not in turning_rates and (
                len(node.incoming) > 1 or len(node.outgoing) > 1
            ):
                raise ValueError(
                    f"Turning rate function for node {node.id} with multiple incoming and/or outgoing links not provided."
                )

            # validate that destination boundary conditions for each destination are provided
            for link in node.outgoing:
                if isinstance(link, Destination):
                    if link.id not in destination_boundary_conditions:
                        raise ValueError(
                            f"Destination boundary condition function for destination {link.id} not provided."
                        )

            # validate that initial flows are defined for all links if not None
            if initial_flows is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if link.id not in initial_flows:
                        raise ValueError(
                            f"Initial flow for link {link.id} not provided (required for onramp, offramp, motorway links)."
                        )

            # validate that initial densities are defined for all links if not None
            if initial_densities is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if (
                        not isinstance(link, Origin)
                        and not isinstance(link, Destination)
                        and link.id not in initial_densities
                    ):
                        raise ValueError(
                            f"Initial density for link {link.id} not provided (required for onramp, offramp, motorway links)."
                        )

            # validate that initial speeds are defined for all links if not None
            if initial_speeds is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if (
                        not isinstance(link, Origin)
                        and not isinstance(link, Destination)
                        and link.id not in initial_speeds
                    ):
                        raise ValueError(
                            f"Initial speed for link {link.id} not provided (required for onramp, offramp, motorway links)."
                        )

        # ! 2 - discretize mainline motorway links according to preferred cell size and CFL condition -> create cells and link correctly
        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if (
                        link.destination_node_id is None
                        or link.destination_node_id == ""
                    ):
                        raise ValueError(
                            f"Motorway link {getattr(link,'id',repr(link))} has no destination_node_id set."
                        )

                    # if the node at the end of the current link represents a lane drop, set it accordingly
                    upcoming_lane_drop = 0
                    dest_node = self.get_node(link.destination_node_id)
                    if (
                        dest_node is not None
                        and len(dest_node.incoming) == 1
                        and len(dest_node.outgoing) == 1
                        and (
                            isinstance(dest_node.outgoing[0], MotorwayLink)
                            or isinstance(dest_node.outgoing[0], Offramp)
                        )
                    ):
                        downstream_link = dest_node.outgoing[0]
                        if downstream_link.lanes < link.lanes:
                            upcoming_lane_drop = link.lanes - downstream_link.lanes

                    link.partition_link(
                        preferred_cell_size=preferred_cell_size,
                        dt=dt,
                        upcoming_lane_drop=upcoming_lane_drop,
                    )

        # ! 3 - augment the node and link states such that complete information can be guaranteed
        # states ordered according to node ordering and their incoming and outgoing links
        # e.g. flows: [ node1.incoming[0].flows, node1.incoming[1].flows, ..., node1.outgoing[0].flows, ... , nodeN.outgoing[M].flows ]
        # as incoming links, only onramps are considered (motorway links are outgoing from another node)
        # stacking into state vector is done through dedicated helper function
        link_flows_dict: dict[str, NDArray[np.float64]] = {}
        link_densities_dict: dict[str, NDArray[np.float64]] = {}
        link_speeds_dict: dict[str, NDArray[np.float64]] = {}
        origin_queues_dict: dict[str, float] = {}
        onramp_queues_dict: dict[str, float] = {}
        offramp_queues_dict: dict[str, float] = {}

        for node in self.list_nodes():
            # split ratios should be defined for each node (add the ones that are missing for SISO nodes)
            if (
                node.id not in turning_rates
                and len(node.incoming) == 1
                and len(node.outgoing) == 1
            ):
                turning_rates[node.id] = lambda _: {node.outgoing[0].id: 1.0}

            # initialize incoming links (only onramps - no motorway links / origins)
            for link in node.incoming:
                if isinstance(link, Origin) or isinstance(link, Onramp):
                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for link {link.id} (type: {type(link)}) is empty."
                                )

                            if init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for link {link.id} (type: {type(link)}) has incorrect length. Using first value for origin / onramp flow."
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

                    if isinstance(link, Origin) and (
                        initial_origin_queues is None
                        or link.id not in initial_origin_queues
                    ):
                        origin_queues_dict[link.id] = 0.0
                    elif isinstance(link, Onramp) and (
                        initial_onramp_queues is None
                        or link.id not in initial_onramp_queues
                    ):
                        onramp_queues_dict[link.id] = 0.0

            # initialize outgoing links (mainline links, offramps, and destinations)
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    num_cells = len(link)

                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for motorway link {link.id} is empty."
                                )

                            if init_flow.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial flow array for motorway link {link.id} has incorrect length. Using first value for all cells instead."
                                )
                                link_flows_dict[link.id] = np.full(
                                    num_cells, init_flow[0]
                                )
                        else:
                            link_flows_dict[link.id] = np.full(num_cells, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(num_cells)

                    if initial_densities is not None and link.id in initial_densities:
                        init_density = initial_densities[link.id]
                        if isinstance(init_density, np.ndarray):
                            if init_density.shape[0] == 0:
                                raise ValueError(
                                    f"Initial density array for motorway link {link.id} is empty."
                                )

                            if init_density.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial density array for motorway link {link.id} has incorrect length. Using first value for all cells instead."
                                )
                                link_densities_dict[link.id] = np.full(
                                    num_cells, init_density[0]
                                )
                        else:
                            link_densities_dict[link.id] = np.full(
                                num_cells, init_density
                            )
                    else:
                        link_densities_dict[link.id] = np.zeros(num_cells)

                    if initial_speeds is not None and link.id in initial_speeds:
                        init_speed = initial_speeds[link.id]
                        if isinstance(init_speed, np.ndarray):
                            if init_speed.shape[0] == 0:
                                raise ValueError(
                                    f"Initial speed array for motorway link {link.id} is empty."
                                )

                            if init_speed.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial speed array for motorway link {link.id} has incorrect length. Using first value for all cells instead."
                                )
                                link_speeds_dict[link.id] = np.full(
                                    num_cells, init_speed[0]
                                )
                        else:
                            link_speeds_dict[link.id] = np.full(num_cells, init_speed)
                    else:
                        link_speeds_dict[link.id] = np.full(num_cells, link.vf)

                # for offramps, make sure a boundary condition is defined for the connected destination
                # destinations with missing boundary conditions are assigned a constant zero function (downstream in free-flow)
                # additionally, initialize offramp flows and queues
                if isinstance(link, Offramp):
                    if link.destination is not None:
                        dest_id = link.destination.id
                        if dest_id not in destination_boundary_conditions:
                            warnings.warn(
                                f"Destination boundary condition function for destination {dest_id} (connected to offramp {link.id}) not provided. Assuming downstream free flow conditions (zero density)."
                            )
                            destination_boundary_conditions[dest_id] = lambda _: 0.0
                    else:
                        raise ValueError(
                            f"Offramp {link.id} has no destination assigned."
                        )

                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for offramp {link.id} is empty."
                                )

                            if init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for offramp {link.id} has incorrect length. Using first value instead."
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

                    if (
                        initial_offramp_queues is not None
                        and link.id in initial_offramp_queues
                    ):
                        offramp_queues_dict[link.id] = initial_offramp_queues[link.id]
                    else:
                        offramp_queues_dict[link.id] = 0.0

                elif isinstance(link, Destination):
                    # for destinations with missing boundary conditions, assign a constant zero function (downstream in free-flow)
                    if link.id not in destination_boundary_conditions:
                        warnings.warn(
                            f"Destination boundary condition function for destination {link.id} not provided. Assuming downstream free flow conditions (zero density)."
                        )
                        destination_boundary_conditions[link.id] = lambda _: 0.0

                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for destination {link.id} is empty."
                                )

                            if init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for destination {link.id} has incorrect length. Using first value instead."
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

        # combine state from separate arrays into single state vector if required by model
        # disturbance = demands and split ratios will be computed with passed functions at simulation time
        (
            x0,
            num_flows,
            num_densities,
            num_speeds,
            num_origins,
            num_onramps,
            num_offramps,
            num_splits,
            num_destinations,
        ) = self.network_dict_to_state_vec(
            flow_dict=link_flows_dict,
            density_dict=link_densities_dict,
            speed_dict=link_speeds_dict,
            origin_queue_dict=origin_queues_dict,
            onramp_queue_dict=onramp_queues_dict,
            offramp_queue_dict=offramp_queues_dict,
        )

        # ! 4 - generate the model udpate equations according to the selected model and the network structure
        system: casadi.Function = model.network_update_function(
            network=self,
            num_flows=num_flows,
            num_densities=num_densities,
            num_speeds=num_speeds,
            num_origins=num_origins,
            num_onramps=num_onramps,
            num_offramps=num_offramps,
            num_splits=num_splits,
            num_destinations=num_destinations,
            inflows_jam_density=inflows_jam_density,
            inflows_free_flow_speed=inflows_free_flow_speed,
            dt=dt,
        )

        # ! 6 - run the simulation loop, update all link states, and track outputs
        time_array: NDArray[np.float64] = np.arange(
            0, duration + dt, dt, dtype=np.float64
        )

        # initialize variables for state, input and disturbance tracking
        state_history: NDArray[np.float64] = np.zeros(
            (len(x0) if isinstance(x0, np.ndarray) else x0.size1(), len(time_array)),
            dtype=np.float64,
        )
        state_history[:, 0] = x0
        disturbance_history: NDArray[np.float64] = np.zeros(
            (
                num_origins + num_onramps + num_splits + num_destinations,
                len(time_array) - 1,
            ),
            dtype=np.float64,
        )

        # run the simulation and store the results
        for t in range(len(time_array) - 1):
            time = time_array[t]

            # get the ids of all components that contribute to the disturbance vector
            origin_ids = origin_queues_dict.keys()
            onramp_ids = onramp_queues_dict.keys()

            # evaluate the demand functions, turning rates and boundary conditions at the current time
            origin_demand_dict = {
                origin_id: origin_demands[origin_id](time) for origin_id in origin_ids
            }
            onramp_demand_dict = {
                onramp_id: onramp_demands[onramp_id](time) for onramp_id in onramp_ids
            }
            turning_rate_dict = {
                node_id: turning_rates[node_id](time)
                for node_id in turning_rates.keys()
            }
            boundary_condition_dict = {
                destination_id: destination_boundary_conditions[destination_id](time)
                for destination_id in destination_boundary_conditions.keys()
            }

            # combine the values into the disturbance vector for the state update
            d = self.network_dict_to_disturbance_vec(
                origin_demand_dict=origin_demand_dict,
                onramp_demand_dict=onramp_demand_dict,
                turning_rate_dict=turning_rate_dict,
                boundary_condition_dict=boundary_condition_dict,
            )

            # perform the state update
            x_next = system(state_history[:, t], d)

            # store the updated state and disturbance
            state_history[:, t + 1] = np.array(x_next).flatten()
            disturbance_history[:, t] = d

        # TODO: ! 7 - plotting of results, etc.

        return time_array, state_history, disturbance_history

    def plot(
        self,
        layout: str = "spring",
        pos: Optional[Mapping[str, tuple[float, float] | NDArray[np.float64]]] = None,
        figsize: tuple[int, int] = (10, 8),
        show: bool = True,
        save_path: str | None = None,
        flows: dict[str, float | NDArray[np.float64]] | None = None,
        densities: dict[str, float | NDArray[np.float64]] | None = None,
        speeds: dict[str, float | NDArray[np.float64]] | None = None,
        origin_queues: dict[str, float] | None = None,
        onramp_queues: dict[str, float] | None = None,
        offramp_queues: dict[str, float] | None = None,
    ):
        """Plot the network topology using the plotting helper.

        Args:
            layout: Layout algorithm for NetworkX ('spring','shell','circular',...)
            pos: Optional precomputed positions mapping node id -> (x, y)
            figsize: Figure size for Matplotlib
            show: Whether to call `plt.show()`
            save_path: Optional path to save the figure
        """

        # simplified plotting: draw static network topology only
        def _link_style(link):
            if isinstance(link, MotorwayLink):
                return "#222222", max(1.0, 0.6 * getattr(link, "lanes", 1)), "Motorway"
            if isinstance(link, Onramp):
                return "#2ca02c", 1.6, "Onramp"
            if isinstance(link, Offramp):
                return "#d62728", 1.6, "Offramp"
            if isinstance(link, Origin):
                return "#9467bd", 1.2, "Origin"
            if isinstance(link, Destination):
                return "#1f77b4", 1.2, "Destination"
            return "#888888", 1.0, "Other"

        G = nx.DiGraph()

        # add node objects
        for node in self.list_nodes():
            G.add_node(node.id, label=node.id, type="junction")

        def _ensure(node_id: str):
            if node_id not in G:
                G.add_node(node_id, label=node_id, type="external")

        # build edges from node.outgoing and represent origins/onramps as SRC:... nodes
        for node in self.list_nodes():
            for link in node.outgoing:
                src = node.id
                dst = getattr(link, "destination_node_id", None)
                if dst is None:
                    dst = f"DEST:{getattr(link, 'id', repr(link))}"
                    _ensure(dst)
                _ensure(src)

                color, width, ltype = _link_style(link)
                link_id = getattr(link, "id", "")
                G.add_edge(
                    src,
                    dst,
                    color=color,
                    width=width,
                    _link_obj=link,
                    link_id=link_id,
                    ltype=ltype,
                )

            for link in node.incoming:
                if isinstance(link, (Origin, Onramp)):
                    src = f"SRC:{getattr(link, 'id', repr(link))}"
                    _ensure(src)
                    dst = node.id
                    color, width, ltype = _link_style(link)
                    link_id = getattr(link, "id", "")
                    G.add_edge(
                        src,
                        dst,
                        color=color,
                        width=width,
                        _link_obj=link,
                        link_id=link_id,
                        ltype=ltype,
                    )

        # compute positions
        if pos is None:
            # attempt an orthogonal/cartesian layout along motorway chains
            pos = {}

            # build motorway adjacency (u -> list of (v, link))
            motor_adj: dict[str, list[tuple[str, object]]] = {}
            in_deg: dict[str, int] = {n.id: 0 for n in self.list_nodes()}
            for u, v, d in G.edges(data=True):
                if d.get("ltype") == "Motorway":
                    motor_adj.setdefault(u, []).append((v, d.get("_link_obj")))
                    in_deg[v] = in_deg.get(v, 0) + 1

            # find motorway sources (nodes with zero incoming motorway links)
            sources = [n for n, deg in in_deg.items() if deg == 0]

            # simple layout: traverse each motorway chain and place nodes along x axis
            visited = set()
            y_offset = 0

            def place_chain(src, y):
                stack = [(src, 0.0)]
                while stack:
                    node_id, x = stack.pop(0)
                    if node_id in pos:
                        # keep smallest x
                        pos[node_id] = (min(pos[node_id][0], x), pos[node_id][1])
                    else:
                        pos[node_id] = (x, y)

                    visited.add(node_id)
                    for v, link_obj in motor_adj.get(node_id, []):
                        length = (
                            getattr(link_obj, "length", 1.0)
                            if link_obj is not None
                            else 1.0
                        )
                        next_x = x + float(length)
                        if v not in visited:
                            stack.append((v, next_x))

            if sources:
                for s in sources:
                    place_chain(s, y_offset)
                    y_offset -= 1.0
            else:
                # fallback to spring layout if no motorway structure
                pos = nx.spring_layout(G, seed=42)

            # place externals (SRC:/DEST:) very close to junctions for compact layout
            for n in G.nodes():
                if n in pos:
                    continue
                s = str(n)
                placed = False
                # incoming source nodes (SRC:...) -> place just above their destination
                if s.startswith("SRC:"):
                    neighbors = list(G.successors(n)) if n in G else []
                    if neighbors and neighbors[0] in pos:
                        dst_pos = pos[neighbors[0]]
                        pos[n] = (dst_pos[0] - 0.08, dst_pos[1] + 0.25)
                        placed = True

                # destination placeholders (DEST:...) -> place just above their source
                if not placed and s.startswith("DEST:"):
                    preds = list(G.predecessors(n)) if n in G else []
                    if preds and preds[0] in pos:
                        src_pos = pos[preds[0]]
                        pos[n] = (src_pos[0] + 0.08, src_pos[1] + 0.25)
                        placed = True

                # fallback placement
                if not placed:
                    pos[n] = (0.0, y_offset)
                    y_offset -= 0.6

        fig, ax = plt.subplots(figsize=figsize)

        # draw nodes without IDs (IDs will be shown on hover if mplcursors available)
        junctions = [n for n, d in G.nodes(data=True) if d.get("type") == "junction"]
        externals = [n for n, d in G.nodes(data=True) if d.get("type") != "junction"]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=junctions,
            node_color="#ffffff",
            edgecolors="#111111",
            node_size=520,
            ax=ax,
        )

        # split externals into sources (SRC:), destinations (DEST:) and others
        externals_src = [n for n in externals if str(n).startswith("SRC:")]
        externals_dest = [n for n in externals if str(n).startswith("DEST:")]
        externals_other = [
            n for n in externals if n not in externals_src and n not in externals_dest
        ]

        nodes_src = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=externals_src,
            node_color="#e7d4f5",
            edgecolors="#9467bd",
            linewidths=2.5,
            node_size=300,
            node_shape="^",
            ax=ax,
        )
        nodes_dest = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=externals_dest,
            node_color="#b3d9ff",
            edgecolors="#1f77b4",
            linewidths=2.5,
            node_size=300,
            node_shape="s",
            ax=ax,
        )
        nodes_other = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=externals_other,
            node_color="#f0f0f0",
            edgecolors="#444444",
            linewidths=1.5,
            node_size=200,
            ax=ax,
        )

        # draw edges plainly and capture artist
        edges = list(G.edges(data=True))

        # group edges by type to draw with distinct styles
        motorway_edges = []
        onramp_edges = []
        offramp_edges = []
        origin_edges = []
        destination_edges = []
        other_edges = []

        for u, v, d in edges:
            ltype = d.get("ltype", "Other")
            if ltype == "Motorway":
                motorway_edges.append((u, v, d))
            elif ltype == "Onramp":
                onramp_edges.append((u, v, d))
            elif ltype == "Offramp":
                offramp_edges.append((u, v, d))
            elif ltype == "Origin":
                origin_edges.append((u, v, d))
            elif ltype == "Destination":
                destination_edges.append((u, v, d))
            else:
                other_edges.append((u, v, d))

        edge_artists = []

        def _draw_group(edge_list, style, base_width, alpha=1.0):
            if not edge_list:
                return None
            et = [(u, v) for u, v, _ in edge_list]
            colors = [d.get("color", "#333333") for _, _, d in edge_list]
            widths = [
                max(
                    0.8,
                    base_width
                    * (
                        float(getattr(d.get("_link_obj"), "lanes", 1))
                        if d.get("_link_obj")
                        else d.get("width", 1.0)
                    ),
                )
                for _, _, d in edge_list
            ]
            return (
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=et,
                    edge_color=colors,
                    style=style,
                    width=widths,
                    alpha=alpha,
                    arrowsize=18,
                    ax=ax,
                ),
                edge_list,
            )

        # draw edge groups and collect artists
        for result in [
            _draw_group(motorway_edges, "solid", 2.5),
            _draw_group(onramp_edges, "dashed", 2.0, 0.85),
            _draw_group(offramp_edges, "dashdot", 2.0, 0.85),
            _draw_group(origin_edges, "dotted", 1.8, 0.8),
            _draw_group(destination_edges, "dotted", 1.8, 0.8),
            _draw_group(other_edges, "solid", 1.0, 0.7),
        ]:
            if result:
                edge_artists.append(result)

        # create legend with visual style consistent with plot
        legend_elements = []
        type_order = ["Motorway", "Onramp", "Offramp", "Origin", "Destination", "Other"]

        for ltype in type_order:
            for _, _, d in edges:
                if d.get("ltype") == ltype:
                    color = d.get("color", "#333333")
                    width = d.get("width", 1.0)

                    # determine line style
                    if ltype == "Motorway":
                        style = "solid"
                        lw = 2.5
                    elif ltype == "Onramp":
                        style = "dashed"
                        lw = 2.0
                    elif ltype == "Offramp":
                        style = "dashdot"
                        lw = 2.0
                    elif ltype in ["Origin", "Destination"]:
                        style = "dotted"
                        lw = 1.8
                    else:
                        style = "solid"
                        lw = 1.0

                    legend_elements.append(
                        Line2D(
                            [0], [0], color=color, lw=lw, linestyle=style, label=ltype
                        )
                    )
                    break

        ax.legend(
            handles=legend_elements,
            title="Link Types",
            loc="upper right",
            framealpha=0.95,
            edgecolor="gray",
            fontsize=9,
        )

        ax.set_title("Traffic Network Topology", fontsize=14, fontweight="bold", pad=15)
        ax.set_axis_off()

        # add subtle grid for better readability
        ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5)
        ax.set_axis_off()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return ax
