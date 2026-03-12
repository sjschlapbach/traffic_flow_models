from __future__ import annotations

import math
import json
import casadi
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from typing import (
    Iterator,
    Callable,
    Tuple,
)


from traffic_flow_models.network import (
    Node,
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
)


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

    # ! Initialization and basic methods
    # region
    def __init__(self, nodes: list[Node] | None = None) -> None:
        """Initialize the Network object."""
        self._nodes: list[Node] = []

        if nodes is None:
            nodes = []

        for n in nodes:
            self.add_node(n)

    def __len__(self) -> int:
        """Return the number of nodes in the network.

        Returns:
            int: Number of nodes contained in the network.
        """
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over nodes in insertion order.

        Yields:
            Node: Next node in the network.
        """
        for n in self.list_nodes():
            yield n

    # endregion

    # ! Node management methods
    # region
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

    # endregion

    # ! Converstion / reshaping methods
    # region

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
            num_splits += len(node.outgoing)
            for link in node.outgoing:
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

                    # count destination connected to offramp for disturbance vector sizing
                    # flows are not tracked explicitly for offramp destinations
                    if link.destination is not None:
                        num_destinations += 1

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
        turning_rate_dict: dict[str, dict[str, float]],
        flow_boundary_condition_dict: dict[str, float],
        density_boundary_condition_dict: dict[str, float],
    ):
        """Pack origin demands and node turning rates into a vector.

        The disturbance vector contains all exogenous inputs required by the
        simulator: origin demands and per-node turning rates for outgoing
        links. The function iterates nodes in the network and concatenates
        the values in a deterministic order.

        Args:
            origin_demand_dict: Mapping origin id -> scalar demand (veh/h).
            turning_rate_dict: Mapping node id -> mapping outgoing link id -> turn rate.
            flow_boundary_condition_dict: Mapping destination id -> downstream flow (veh/h).
            density_boundary_condition_dict: Mapping destination id -> downstream density (veh/km/lane).

        Returns:
            Disturbance vector containing all exogenous inputs for simulation.

        Raises:
            ValueError: If required demand or turning-rate entries are missing
                or inconsistent with the network topology.
        """

        # "disturbance" variables = network inflows and turning rates
        # structure: [ origin_demands | turning_rates ]
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

            # set values for incoming origins
            for link in node.incoming:
                # initialize origins (queues only)
                if isinstance(link, Origin):
                    if link.id in origin_demand_dict:
                        d = np.concatenate((d, np.array([origin_demand_dict[link.id]])))
                    else:
                        raise ValueError(
                            f"Demand for origin {link.id} must be provided."
                        )

            # set values for outgoing links (destinations / destinations connected to offramps)
            for link in node.outgoing:
                if isinstance(link, Destination):
                    if link.id in flow_boundary_condition_dict:
                        d = np.concatenate(
                            (d, np.array([flow_boundary_condition_dict[link.id]]))
                        )
                    else:
                        raise ValueError(
                            f"Flow boundary condition for destination {link.id} must be provided."
                        )

                    if link.id in density_boundary_condition_dict:
                        d = np.concatenate(
                            (d, np.array([density_boundary_condition_dict[link.id]]))
                        )
                    else:
                        raise ValueError(
                            f"Density boundary condition for destination {link.id} must be provided."
                        )

                if isinstance(link, Offramp):
                    if link.destination is not None:
                        dest_id = link.destination.id
                        if dest_id in flow_boundary_condition_dict:
                            d = np.concatenate(
                                (d, np.array([flow_boundary_condition_dict[dest_id]]))
                            )
                        else:
                            raise ValueError(
                                f"Flow boundary condition for destination {dest_id} (connected to offramp {link.id}) must be provided."
                            )

                        if dest_id in density_boundary_condition_dict:
                            d = np.concatenate(
                                (
                                    d,
                                    np.array(
                                        [density_boundary_condition_dict[dest_id]]
                                    ),
                                )
                            )
                        else:
                            raise ValueError(
                                f"Density boundary condition for destination {dest_id} (connected to offramp {link.id}) must be provided."
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
    ) -> Tuple[
        dict[str, float | casadi.SX],
        dict[str, dict[str, float | casadi.SX]],
        dict[str, float | casadi.SX],
        dict[str, float | casadi.SX],
    ]:
        """Unpack a disturbance vector into structured disturbance dictionaries.

        Reverses the packing performed by `network_dict_to_disturbance_vec`.
        Accepts a NumPy 1-D array or a CasADi SX column vector and returns
        four dictionaries keyed by ids:
        - ``origin_demands``: mapping origin id -> scalar demand (veh/time)
        - ``turning_rates``: mapping node id -> (outgoing link id -> rate)
        - ``flow_boundary_conditions``: mapping destination id -> downstream flow
        - ``density_boundary_conditions``: mapping destination id -> downstream density

        The unpacking follows the node-ordering used in the network and will
        raise ValueError if the disturbance vector is too short or inconsistent
        with the network topology.

        Args:
            d: 1-D NumPy array or CasADi SX column vector containing the packed disturbances.

        Returns:
            Tuple of four dictionaries: ``(origin_demands, turning_rates, flow_boundary_conditions, density_boundary_conditions)``.

        Raises:
            ValueError: If the disturbance vector is too short for the network
                topology or cannot be parsed into the expected entries.
        """

        # initialize the structure dictionary containers for the disturbance vector
        origin_demands = dict[str, float | casadi.SX]()
        turning_rates = dict[str, dict[str, float | casadi.SX]]()
        flow_boundary_conditions = dict[str, float | casadi.SX]()
        density_boundary_conditions = dict[str, float | casadi.SX]()
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
                # initialize origins (queues only)
                if isinstance(link, Origin):

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
                            "Disturbance vector too short to extract all destination flow boundary conditions."
                        )

                    flow_boundary_conditions[link.id] = d[i_disturbance]
                    i_disturbance += 1

                    if i_disturbance + 1 > disturbance_size:
                        raise ValueError(
                            "Disturbance vector too short to extract all destination density boundary conditions."
                        )

                    density_boundary_conditions[link.id] = d[i_disturbance]
                    i_disturbance += 1

                if isinstance(link, Offramp):
                    if link.destination is not None:
                        if i_disturbance + 1 > disturbance_size:
                            raise ValueError(
                                "Disturbance vector too short to extract all destination flow boundary conditions."
                            )

                        flow_boundary_conditions[link.destination.id] = d[i_disturbance]
                        i_disturbance += 1

                        if i_disturbance + 1 > disturbance_size:
                            raise ValueError(
                                "Disturbance vector too short to extract all destination density boundary conditions."
                            )

                        density_boundary_conditions[link.destination.id] = d[
                            i_disturbance
                        ]
                        i_disturbance += 1
                    else:
                        raise ValueError(
                            f"Offramp {link.id} has no destination assigned."
                        )

        return (
            origin_demands,
            turning_rates,
            flow_boundary_conditions,
            density_boundary_conditions,
        )

    # endregion

    # ! Simulation and validation
    # region
    def validate(self) -> bool:
        """
        Validate network structure according to class requirements.

        Requirements validated:
            - Each network must have at least one origin link
            - Each network must have at least one destination
            - Each offramp needs to be connected to a destination
            - A node connected to an origin may only have one outgoing link (motorway link or onramp)
            - Each onramp needs to be connected to an origin through a node upstream
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

        # ensure that the network has at least one origin and one destination
        has_origin = False
        has_destination = False

        # collect all destination instances present in network (to check offramp targets)
        dests: set[Destination] = set()

        for node in self.list_nodes():
            for link in list(node.incoming) + list(node.outgoing):
                if isinstance(link, Origin):
                    has_origin = True
                if isinstance(link, Destination):
                    has_destination = True
                    dests.add(link)
                if isinstance(link, Offramp) and link.destination is not None:
                    has_destination = True
                    dests.add(link.destination)

        if not has_origin:
            raise ValueError("Network must contain at least one origin link.")

        if not has_destination:
            raise ValueError("Network must contain at least one destination.")

        # check that every node connected to an origin or onramp only has a onramp link
        # (for origins) or single motorway link (for origins or onramps) as an outgoing
        # link to ensure the correct computation of boundary constraints
        for node in self.list_nodes():
            incoming_origin = any(isinstance(link, Origin) for link in node.incoming)
            incoming_onramp = any(isinstance(link, Onramp) for link in node.incoming)

            # for origin nodes, verify that there is only an origin incoming link
            # and at most one outgoing link (motorway link or onramp)
            if incoming_origin:
                if len(node.incoming) != 1:
                    raise ValueError(
                        f"Node {node.id} has an origin incoming link but multiple incoming links."
                    )

                if len(node.outgoing) > 1:
                    raise ValueError(
                        f"Node {node.id} has an origin incoming link but multiple outgoing links."
                    )

                outgoing_link = node.outgoing[0]
                if not isinstance(outgoing_link, (MotorwayLink, Onramp)):
                    raise ValueError(
                        f"Node {node.id} has an origin incoming link but its outgoing link is not a motorway link or onramp."
                    )

            # for onramp nodes, verify that there is only one outgoing motorway link
            if incoming_onramp:
                if len(node.outgoing) != 1:
                    raise ValueError(
                        f"Node {node.id} has an onramp incoming link but multiple outgoing links."
                    )

                outgoing_link = node.outgoing[0]
                if not isinstance(outgoing_link, MotorwayLink):
                    raise ValueError(
                        f"Node {node.id} has an onramp incoming link but its outgoing link is not a motorway link."
                    )

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

    def _validate_state_history_numerical(
        self,
        flows: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        densities: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        speeds: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        origin_queues: dict[str, float] | dict[str, casadi.SX],
        onramp_queues: dict[str, float] | dict[str, casadi.SX],
        offramp_queues: dict[str, float] | dict[str, casadi.SX],
    ) -> None:
        """Validate that state-history dictionaries contain numerical values.

        This helper checks that the provided per-link and per-queue dictionaries
        contain numeric NumPy arrays (for flows, densities, speeds) or scalar
        numeric values (for queues). It raises a ValueError if any non-numerical
        entries are found.

        Args:
            flows: Mapping link id -> per-cell flow arrays or CasADi SX slices.
            densities: Mapping motorway link id -> per-cell density arrays or CasADi SX.
            speeds: Mapping motorway link id -> per-cell speed arrays or CasADi SX.
            origin_queues: Mapping origin id -> scalar queue values.
            onramp_queues: Mapping onramp id -> scalar queue values.
            offramp_queues: Mapping offramp id -> scalar queue values.

        Raises:
            ValueError: If any entry is not a numeric NumPy array or numeric scalar.
        """

        if (
            not all(
                isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
                for val in flows.values()
            )
            or not all(
                isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
                for val in densities.values()
            )
            or not all(
                isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
                for val in speeds.values()
            )
            or not all(
                isinstance(val, (float, np.floating)) for val in origin_queues.values()
            )
            or not all(
                isinstance(val, (float, np.floating)) for val in onramp_queues.values()
            )
            or not all(
                isinstance(val, (float, np.floating)) for val in offramp_queues.values()
            )
        ):
            raise ValueError("Non-numerical values found in state history.")

    def _validate_disturbance_history_numerical(
        self,
        origin_demands: dict[str, float] | dict[str, casadi.SX],
        turning_rates: dict[str, dict[str, float]] | dict[str, dict[str, casadi.SX]],
        flow_boundary_conditions: dict[str, float] | dict[str, casadi.SX],
        density_boundary_conditions: dict[str, float] | dict[str, casadi.SX],
    ) -> None:
        """Validate that disturbance-history dictionaries contain numerical values.

        This helper checks that the provided per-origin demands, per-node
        turning rate, and boundary condition dictionaries contain numeric scalar
        values. It raises a ValueError if any non-numerical entries are found.

        Args:
            origin_demands: Mapping origin id -> scalar demand value.
            turning_rates: Mapping node id -> mapping outgoing link id -> turn rate.
            flow_boundary_conditions: Mapping destination id -> downstream flow.
            density_boundary_conditions: Mapping destination id -> downstream density.

        Raises:
            ValueError: If any entry is not a numeric scalar.
        """

        if (
            not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in origin_demands.values()
            )
            or not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in flow_boundary_conditions.values()
            )
            or not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in density_boundary_conditions.values()
            )
        ):
            raise ValueError("Non-numerical values found in disturbance history.")

        # validate turning rates (nested dictionary)
        for node_id, node_rates in turning_rates.items():
            if not isinstance(node_rates, dict):
                raise ValueError(
                    f"Turning rates for node {node_id} must be a dictionary."
                )
            if not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in node_rates.values()
            ):
                raise ValueError(
                    f"Non-numerical turning rate values found for node {node_id}."
                )

    def _validate_initial_conditions_numerical(
        self,
        origin_demands: dict[str, Callable[[float], float]],
        turning_rates: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc: dict[str, Callable[[float], float]],
        destination_density_bc: dict[str, Callable[[float], float]],
        initial_flows: dict[str, float | NDArray[np.float64]] | None = None,
        initial_densities: dict[str, float | NDArray[np.float64]] | None = None,
        initial_speeds: dict[str, float | NDArray[np.float64]] | None = None,
    ):
        """Validate presence and basic consistency of initial-condition inputs.

        Ensures that for each node in the network the required callable demand
        and turning-rate functions are provided, and (if arrays of initial
        flows/densities/speeds are supplied) that entries exist for the
        respective links. Raises descriptive ValueError messages on missing
        or inconsistent inputs.

        Args:
            origin_demands: Mapping origin id -> callable(time) -> demand.
            turning_rates: Mapping node id -> callable(time) -> dict[outgoing->rate].
            destination_flow_bc: Mapping destination id -> callable(time) -> flow.
            destination_density_bc: Mapping destination id -> callable(time) -> density.
            initial_flows: Optional mapping link id -> scalar or per-cell array for initial flows.
            initial_densities: Optional mapping link id -> scalar or per-cell array for initial densities.
            initial_speeds: Optional mapping link id -> scalar or per-cell array for initial speeds.

        Raises:
            ValueError: If required functions or initial arrays are missing or inconsistent.
        """

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

            # validate that turning rates for each node are provided
            if node.id not in turning_rates and len(node.outgoing) > 1:
                raise ValueError(
                    f"Turning rate function for node {node.id} with multiple incoming and/or outgoing links not provided."
                )

            # validate that destination boundary conditions for each destination are provided
            for link in node.outgoing:
                if isinstance(link, Destination):
                    if link.id not in destination_flow_bc:
                        raise ValueError(
                            f"Destination flow boundary condition function for destination {link.id} not provided."
                        )

                    if link.id not in destination_density_bc:
                        raise ValueError(
                            f"Destination density boundary condition function for destination {link.id} not provided."
                        )

            # validate that initial flows are defined for all links if not None
            if initial_flows is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if link.id not in initial_flows:
                        raise ValueError(
                            f"Initial flow for link {link.id} not provided (required for origins, onramp, offramp, destinations, and motorway links)."
                        )

            # validate that initial densities are defined for all links if not None
            if initial_densities is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if (
                        not isinstance(link, Origin)
                        and not isinstance(link, Onramp)
                        and not isinstance(link, Destination)
                        and link.id not in initial_densities
                    ):
                        raise ValueError(
                            f"Initial density for link {link.id} not provided (required for motorway links)."
                        )

            # validate that initial speeds are defined for all links if not None
            if initial_speeds is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if (
                        not isinstance(link, Origin)
                        and not isinstance(link, Onramp)
                        and not isinstance(link, Destination)
                        and link.id not in initial_speeds
                    ):
                        raise ValueError(
                            f"Initial speed for link {link.id} not provided (required for motorway links)."
                        )

    # ! Network topology helpers
    # region
    def _compute_upcoming_lane_drop(self, link: MotorwayLink) -> int:
        """Compute lane drop between a motorway link and its downstream link.

        Checks the downstream node connected to `link` and, if the downstream
        successor is a motorway link or an offramp with fewer lanes, returns
        the number of lanes that are dropped. Returns 0 if no drop is detected.

        Args:
            link: The `MotorwayLink` to inspect.

        Returns:
            Number of lanes dropped (non-negative integer).

        Raises:
            ValueError: If `link.destination_node_id` is not set.
        """

        if link.destination_node_id is None or link.destination_node_id == "":
            raise ValueError(
                f"Motorway link {getattr(link,'id',repr(link))} has no destination_node_id set."
            )

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

        return upcoming_lane_drop

    # endregion

    # ! Evaluation and result visualizations
    # region
    def save_to_txt(self, filepath: str) -> None:
        """Save network structure to a text file for reference.

        Writes a human-readable representation of the network topology showing
        nodes, their IDs, connected links with IDs, and the connections between
        them. The structure is written in a way that starts from origins/onramps
        and follows the flow through the network.

        Args:
            filepath: Path where the text file should be saved.
        """
        with open(filepath, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("NETWORK STRUCTURE\n")
            f.write("=" * 80 + "\n\n")

            # list all nodes with their IDs
            f.write(f"Total Nodes: {len(self._nodes)}\n")
            f.write("-" * 80 + "\n\n")

            # iterate through all nodes and document their connections
            for node in self.list_nodes():
                f.write(f"NODE: {node.id}\n")
                f.write(f"  {'=' * 76}\n")

                # list incoming links
                if node.incoming:
                    f.write(f"  Incoming Links ({len(node.incoming)}):\n")
                    for link in node.incoming:
                        link_type = type(link).__name__
                        link_id = getattr(link, "id", "N/A")
                        origin_node = getattr(link, "origin_node_id", "N/A")
                        dest_node = getattr(link, "destination_node_id", "N/A")

                        f.write(f"    - {link_type} [ID: {link_id}]\n")
                        f.write(f"      Origin Node ID: {origin_node}\n")
                        f.write(f"      Destination Node ID: {dest_node}\n")

                        # add type-specific information
                        if isinstance(link, MotorwayLink):
                            f.write(
                                f"      Length: {link.length} km, Lanes: {link.lanes}, Cells: {len(link)}\n"
                            )
                        elif isinstance(link, (Onramp)):
                            f.write(
                                f"      Lanes: {link.lanes}, Capacity: {link.Qc} veh/h\n"
                            )
                else:
                    f.write("  Incoming Links: None\n")

                # list outgoing links
                if node.outgoing:
                    f.write(f"  Outgoing Links ({len(node.outgoing)}):\n")
                    for link in node.outgoing:
                        link_type = type(link).__name__
                        link_id = getattr(link, "id", "N/A")
                        origin_node = getattr(link, "origin_node_id", "N/A")
                        dest_node = getattr(link, "destination_node_id", "N/A")

                        f.write(f"    - {link_type} [ID: {link_id}]\n")
                        f.write(f"      Origin Node ID: {origin_node}\n")
                        f.write(f"      Destination Node ID: {dest_node}\n")

                        # add type-specific information
                        if isinstance(link, MotorwayLink):
                            f.write(
                                f"      Length: {link.length} km, Lanes: {link.lanes}, Cells: {len(link)}\n"
                            )
                        elif isinstance(link, Offramp):
                            f.write(
                                f"      Lanes: {link.lanes}, Capacity: {link.Qc} veh/h\n"
                            )
                            if link.destination is not None:
                                f.write(
                                    f"      Connected Destination: {link.destination.id}\n"
                                )
                        elif isinstance(link, Destination):
                            f.write("      (Network exit point)\n")
                else:
                    f.write("  Outgoing Links: None\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF NETWORK STRUCTURE\n")
            f.write("=" * 80 + "\n")

    def _serialize_link(
        self, link: MotorwayLink | Origin | Onramp | Offramp | Destination
    ) -> dict:
        """Serialize a link-like object to a JSON-compatible dictionary.

        This helper function converts a link object (MotorwayLink, Origin,
        Onramp, Offramp, Destination) into a dictionary format that can be
        to be stored in JSON format.

        Args:
            link: The link object to serialize.

        Returns:
            A dictionary containing the type and relevant attributes of the link.
        """

        if isinstance(link, MotorwayLink):
            return {
                "type": "MotorwayLink",
                "id": link.id,
                "length": link.length,
                "lanes": link.lanes,
                "lane_capacity": link.Qc_lane,
                "free_flow_speed": link.vf,
                "jam_density": link.rho_jam,
                "origin_node_id": link.origin_node_id,
                "destination_node_id": link.destination_node_id,
            }
        if isinstance(link, Origin):
            return {
                "type": "Origin",
                "id": link.id,
                "destination_node_id": link.destination_node_id,
            }
        if isinstance(link, Destination):
            return {
                "type": "Destination",
                "id": link.id,
                "origin_node_id": link.origin_node_id,
            }
        if isinstance(link, Onramp):
            return {
                "type": "Onramp",
                "id": link.id,
                "lanes": link.lanes,
                "lane_capacity": link.Qc_lane,
                "free_flow_speed": link.vf,
                "jam_density": link.rho_jam,
                "origin_node_id": link.origin_node_id,
                "destination_node_id": link.destination_node_id,
            }
        if isinstance(link, Offramp):
            return {
                "type": "Offramp",
                "id": link.id,
                "lanes": link.lanes,
                "lane_capacity": link.Qc_lane,
                "free_flow_speed": link.vf,
                "jam_density": link.rho_jam,
                "origin_node_id": link.origin_node_id,
                "destination_id": link.destination.id if link.destination else None,
            }

        raise TypeError(
            f"Unsupported link type for serialization: {type(link).__name__}"
        )

    def save_to_json(self, filepath: str) -> None:
        """Persist the network structure (nodes and links) to a JSON file.

        This method collects all nodes and links in the network, extracts their
        relevant attributes, and saves the structure in a JSON format that can be
        reloaded later using `load_from_json`.

        Args:
            filepath: Path where the JSON file should be saved.
        """

        # collect unique links across all nodes
        links_map: dict[str, MotorwayLink | Origin | Onramp | Offramp | Destination] = (
            {}
        )
        for node in self.list_nodes():
            for link in node.incoming + node.outgoing:
                if getattr(link, "id", None) in links_map:
                    continue
                links_map[link.id] = link

                # also collect offramp destinations (not in node links)
                if isinstance(link, Offramp) and link.destination is not None:
                    if link.destination.id not in links_map:
                        links_map[link.destination.id] = link.destination

        payload = {
            "nodes": [
                {
                    "id": node.id,
                    "incoming": [getattr(l, "id", None) for l in node.incoming],
                    "outgoing": [getattr(l, "id", None) for l in node.outgoing],
                }
                for node in self.list_nodes()
            ],
            "links": [self._serialize_link(link) for link in links_map.values()],
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "Network":
        """Load a network structure from a JSON file created by `save_to_json`.

        This method reads the JSON file, reconstructs the nodes and links with their
        relevant attributes, and re-establishes the connections between them to recreate
        the network structure.

        Args:
            filepath: Path to the JSON file containing the network structure.

        Returns:
            An instance of `Network` with the structure defined in the JSON file.
        """

        with open(filepath, "r") as f:
            data = json.load(f)

        links_data = data.get("links", [])
        nodes_data = data.get("nodes", [])

        links_by_id: dict[
            str, MotorwayLink | Origin | Onramp | Offramp | Destination
        ] = {}
        pending_offramp_dest: list[tuple[Offramp, str | None]] = []

        for entry in links_data:
            l_type = entry.get("type")
            if l_type == "MotorwayLink":
                link_obj = MotorwayLink(
                    length=entry["length"],
                    lanes=entry["lanes"],
                    lane_capacity=entry["lane_capacity"],
                    free_flow_speed=entry["free_flow_speed"],
                    jam_density=entry["jam_density"],
                    id=entry["id"],
                    origin_node_id=entry["origin_node_id"],
                    destination_node_id=entry["destination_node_id"],
                )
            elif l_type == "Origin":
                link_obj = Origin(
                    id=entry["id"],
                    destination_node_id=entry["destination_node_id"],
                )
            elif l_type == "Destination":
                link_obj = Destination(
                    id=entry["id"],
                    origin_node_id=entry["origin_node_id"],
                )
            elif l_type == "Onramp":
                link_obj = Onramp(
                    lanes=entry["lanes"],
                    lane_capacity=entry["lane_capacity"],
                    free_flow_speed=entry["free_flow_speed"],
                    jam_density=entry["jam_density"],
                    id=entry["id"],
                    origin_node_id=entry["origin_node_id"],
                    destination_node_id=entry["destination_node_id"],
                )
            elif l_type == "Offramp":
                link_obj = Offramp(
                    lanes=entry["lanes"],
                    lane_capacity=entry["lane_capacity"],
                    free_flow_speed=entry["free_flow_speed"],
                    jam_density=entry["jam_density"],
                    id=entry["id"],
                    destination=None,
                    origin_node_id=entry["origin_node_id"],
                )
                pending_offramp_dest.append((link_obj, entry.get("destination_id")))
            else:
                raise ValueError(f"Unknown link type in JSON: {l_type}")

            links_by_id[link_obj.id] = link_obj

        # resolve offramp destinations now that all links exist
        for offr, dest_id in pending_offramp_dest:
            if dest_id is not None:
                dest_link = links_by_id.get(dest_id)
                if isinstance(dest_link, Destination):
                    offr.destination = dest_link
                else:
                    raise ValueError(
                        f"Offramp {offr.id} references destination ID {dest_id} which is not a valid Destination link."
                    )
            else:
                raise ValueError(
                    f"Offramp {offr.id} does not have a destination_id specified in the JSON."
                )

        nodes: list[Node] = []
        for node_entry in nodes_data:
            incoming_ids = node_entry.get("incoming", [])
            outgoing_ids = node_entry.get("outgoing", [])
            incoming_links = [links_by_id[i] for i in incoming_ids if i in links_by_id]
            outgoing_links = [links_by_id[i] for i in outgoing_ids if i in links_by_id]
            node = Node(
                id=node_entry["id"], incoming=incoming_links, outgoing=outgoing_links
            )
            nodes.append(node)

        return cls(nodes=nodes)

    # endregion

    # ! Network visualization
    # region
    def plot(
        self,
        figsize: tuple[int, int] = (10, 8),
        show: bool = True,
        save_path: str | None = None,
    ):
        """Plot the network topology using the plotting helper.

        Args:
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

        # calculate scaling factor based on number of nodes
        num_nodes = len(self._nodes)
        scale_factor = max(0.3, 1.0 / math.sqrt(max(1, num_nodes / 4)))

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
                if isinstance(link, Origin):
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
        node_positions = {}
        for node in self.list_nodes():
            if hasattr(node, "position") and node.position is not None:
                node_positions[node.id] = np.array(node.position, dtype=np.float64)

        # if all nodes have positions, use them directly
        y_offset = 0
        if len(node_positions) == len(self._nodes):
            pos = node_positions
        else:
            # otherwise, use automatic layout
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
            if sources:
                for src in sources:
                    stack = [(src, 0.0)]
                    while stack:
                        node_id, x = stack.pop(0)
                        if node_id in pos:
                            # keep smallest x
                            pos[node_id] = np.array(
                                [
                                    min(float(pos[node_id][0]), float(x)),
                                    float(pos[node_id][1]),
                                ],
                                dtype=np.float64,
                            )
                        else:
                            pos[node_id] = np.array(
                                [float(x), float(y_offset)], dtype=np.float64
                            )

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
                    pos[n] = np.array(
                        [float(dst_pos[0]) - 0.08, float(dst_pos[1]) + 0.25],
                        dtype=np.float64,
                    )
                    placed = True

            # destination placeholders (DEST:...) -> place just above their source
            if not placed and s.startswith("DEST:"):
                preds = list(G.predecessors(n)) if n in G else []
                if preds and preds[0] in pos:
                    src_pos = pos[preds[0]]
                    pos[n] = np.array(
                        [float(src_pos[0]) + 0.08, float(src_pos[1]) + 0.25],
                        dtype=np.float64,
                    )
                    placed = True

            # fallback placement
            if not placed:
                pos[n] = np.array([0.0, float(y_offset)], dtype=np.float64)
                y_offset -= 0.6

        # avoid automatic display in interactive backends when `show` is False
        prev_interactive = plt.isinteractive()
        if not show and prev_interactive:
            plt.ioff()

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
            node_size=int(520 * scale_factor),
            ax=ax,
        )

        # split externals into sources (SRC:), destinations (DEST:) and others
        externals_src = [n for n in externals if str(n).startswith("SRC:")]
        externals_dest = [n for n in externals if str(n).startswith("DEST:")]
        externals_other = [
            n for n in externals if n not in externals_src and n not in externals_dest
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=externals_src,
            node_color="#e7d4f5",
            edgecolors="#9467bd",
            linewidths=2.5 * scale_factor,
            node_size=int(300 * scale_factor),
            node_shape="^",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=externals_dest,
            node_color="#b3d9ff",
            edgecolors="#1f77b4",
            linewidths=2.5 * scale_factor,
            node_size=int(300 * scale_factor),
            node_shape="s",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=externals_other,
            node_color="#f0f0f0",
            edgecolors="#444444",
            linewidths=1.5 * scale_factor,
            node_size=int(200 * scale_factor),
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
                    0.8 * scale_factor,
                    base_width
                    * scale_factor
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
                    arrowsize=int(18 * scale_factor),
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

    # endregion
