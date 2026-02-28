from __future__ import annotations

import os
import cv2
import math
import json
import warnings
import casadi
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
from numpy.typing import NDArray
from datetime import datetime
from matplotlib.lines import Line2D
from typing import (
    cast,
    Iterator,
    Callable,
    Tuple,
    Union,
    Any,
)


from traffic_flow_models.network import (
    Node,
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
)
from traffic_flow_models.model import CTM, METANET, METANETParams


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

    # visualization color constants (RGB tuples in range [0, 255])
    COLOR_BRIGHT_GREEN = (144, 238, 144)  # low density
    COLOR_DARK_GREEN = (0, 100, 0)  # critical density
    COLOR_ORANGE = (255, 165, 0)  # moderate congestion
    COLOR_DARK_RED = (139, 0, 0)  # jam density
    COLOR_CONGESTION_RED = (0.8, 0.0, 0.0)  # ramp queue congestion (normalized)

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
        onramp_demand_dict: dict[str, float],
        turning_rate_dict: dict[str, dict[str, float]],
        flow_boundary_condition_dict: dict[str, float],
        density_boundary_condition_dict: dict[str, float],
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
            flow_boundary_condition_dict: Mapping destination id -> downstream flow (veh/h).
            density_boundary_condition_dict: Mapping destination id -> downstream density (veh/km/lane).

        Returns:
            Disturbance vector containing all exogenous inputs for simulation.

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
        - ``onramp_demands``: mapping onramp id -> scalar demand (veh/time)
        - ``turning_rates``: mapping node id -> (outgoing link id -> rate)
        - ``flow_boundary_conditions``: mapping destination id -> downstream flow
        - ``density_boundary_conditions``: mapping destination id -> downstream density

        The unpacking follows the node-ordering used in the network and will
        raise ValueError if the disturbance vector is too short or inconsistent
        with the network topology.

        Args:
            d: 1-D NumPy array or CasADi SX column vector containing the packed disturbances.

        Returns:
            Tuple of five dictionaries: ``(origin_demands, onramp_demands, turning_rates, flow_boundary_conditions, density_boundary_conditions)``.

        Raises:
            ValueError: If the disturbance vector is too short for the network
                topology or cannot be parsed into the expected entries.
        """

        # initialize the structure dictionary containers for the disturbance vector
        origin_demands = dict[str, float | casadi.SX]()
        onramp_demands = dict[str, float | casadi.SX]()
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
            onramp_demands,
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
            - Each network must have at least one origin link or onramp
            - Each network must have at least one destination
            - Each offramp needs to be connected to a destination
            - A node connected to an origin may only have one outgoing link (motorway link)
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

        # check that every node connected to an origin or onramp only has a single motorway
        # link as an outgoing link to ensure the correct computation of boundary constraints
        for node in self.list_nodes():
            incoming_inflows = [
                link
                for link in node.incoming
                if (isinstance(link, Origin) or isinstance(link, Onramp))
            ]

            if len(incoming_inflows) > 0 and (
                len(node.outgoing) != 1
                or not isinstance(node.outgoing[0], MotorwayLink)
            ):
                raise ValueError(
                    f"Node {node.id} connected to an origin or onramp must have exactly one outgoing motorway link."
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
        onramp_demands: dict[str, float] | dict[str, casadi.SX],
        turning_rates: dict[str, dict[str, float]] | dict[str, dict[str, casadi.SX]],
        flow_boundary_conditions: dict[str, float] | dict[str, casadi.SX],
        density_boundary_conditions: dict[str, float] | dict[str, casadi.SX],
    ) -> None:
        """Validate that disturbance-history dictionaries contain numerical values.

        This helper checks that the provided per-origin, per-onramp, per-node
        turning rate, and boundary condition dictionaries contain numeric scalar
        values. It raises a ValueError if any non-numerical entries are found.

        Args:
            origin_demands: Mapping origin id -> scalar demand value.
            onramp_demands: Mapping onramp id -> scalar demand value.
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
                for val in onramp_demands.values()
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
        onramp_demands: dict[str, Callable[[float], float]],
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
            onramp_demands: Mapping onramp id -> callable(time) -> demand.
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

            # validate that onramp demands for each onramp are provided
            for link in node.incoming:
                if isinstance(link, Onramp):
                    if link.id not in onramp_demands:
                        raise ValueError(
                            f"Onramp demand function for onramp {link.id} not provided."
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
                            f"Initial density for link {link.id} not provided (required for onramp, offramp, motorway links)."
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
                            f"Initial speed for link {link.id} not provided (required for onramp, offramp, motorway links)."
                        )

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

    def _augment_network_initialization(
        self,
        initial_flows: dict[str, float | NDArray[np.float64]] | None,
        initial_densities: dict[str, float | NDArray[np.float64]] | None,
        initial_speeds: dict[str, float | NDArray[np.float64]] | None,
        initial_origin_queues: dict[str, float] | None,
        initial_onramp_queues: dict[str, float] | None,
        initial_offramp_queues: dict[str, float] | None,
        turning_rates: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc: dict[str, Callable[[float], float]],
        destination_density_bc: dict[str, Callable[[float], float]],
    ):
        """Prepare and augment initial per-link and per-queue dictionaries.

        This method fills defaults for missing initial states (flows, densities,
        speeds, queues) and ensures that turning-rate and destination boundary
        condition callables are available for all nodes/links. It returns a
        tuple with the initialized dictionaries used by the simulator.

        Args:
            initial_flows: Optional mapping link id -> scalar or per-cell array for initial flows.
            initial_densities: Optional mapping link id -> scalar or per-cell array for initial densities.
            initial_speeds: Optional mapping link id -> scalar or per-cell array for initial speeds.
            initial_origin_queues: Optional mapping origin id -> initial queue length.
            initial_onramp_queues: Optional mapping onramp id -> initial queue length.
            initial_offramp_queues: Optional mapping offramp id -> initial queue length.
            turning_rates: Mapping node id -> callable(time) -> turning-rate dict.
            destination_flow_bc: Mapping destination id -> callable(time) -> downstream flow.
            destination_density_bc: Mapping destination id -> callable(time) -> downstream density.

        Returns:
            Tuple containing:
            - link_flows_dict: mapping link id -> per-cell flows (np.ndarray)
            - link_densities_dict: mapping link id -> per-cell densities (np.ndarray)
            - link_speeds_dict: mapping link id -> per-cell speeds (np.ndarray)
            - origin_queues_dict: mapping origin id -> scalar queue
            - onramp_queues_dict: mapping onramp id -> scalar queue
            - offramp_queues_dict: mapping offramp id -> scalar queue
            - turning_rates_dict: mapping node id -> callable(time) -> dict[outgoing->rate]
            - destination_flow_bc_dict: mapping destination id -> callable(time) -> flow
            - destination_density_bc_dict: mapping destination id -> callable(time) -> density

        Raises:
            ValueError: If an offramp lacks a connected destination.
        """

        link_flows_dict: dict[str, NDArray[np.float64]] = {}
        link_densities_dict: dict[str, NDArray[np.float64]] = {}
        link_speeds_dict: dict[str, NDArray[np.float64]] = {}
        origin_queues_dict: dict[str, float] = {}
        onramp_queues_dict: dict[str, float] = {}
        offramp_queues_dict: dict[str, float] = {}
        turning_rates_dict: dict[str, Callable[[float], dict[str, float]]] = {}
        destination_flow_bc_dict: dict[str, Callable[[float], float]] = {}
        destination_density_bc_dict: dict[str, Callable[[float], float]] = {}

        for node in self.list_nodes():
            # split ratios should be defined for each node (add the ones that are missing for SISO nodes)
            if node.id not in turning_rates and len(node.outgoing) == 1:
                # capture the current outgoing link id in a default argument to avoid late-binding
                link_id = node.outgoing[0].id
                turning_rates_dict[node.id] = lambda _, link_id=link_id: {link_id: 1.0}
            else:
                turning_rates_dict[node.id] = turning_rates[node.id]

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

                            elif init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for link {link.id} (type: {type(link)}) has incorrect length. Using first value for origin / onramp flow."
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])

                            else:
                                link_flows_dict[link.id] = init_flow
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

                    if isinstance(link, Origin):
                        if (
                            initial_origin_queues is None
                            or link.id not in initial_origin_queues
                        ):
                            origin_queues_dict[link.id] = 0.0
                        else:
                            origin_queues_dict[link.id] = initial_origin_queues[link.id]

                    elif isinstance(link, Onramp):
                        if (
                            initial_onramp_queues is None
                            or link.id not in initial_onramp_queues
                        ):
                            onramp_queues_dict[link.id] = 0.0
                        else:
                            onramp_queues_dict[link.id] = initial_onramp_queues[link.id]

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

                            elif init_flow.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial flow array for motorway link {link.id} has incorrect length. Using first value for all cells instead."
                                )
                                link_flows_dict[link.id] = np.full(
                                    num_cells, init_flow[0]
                                )

                            else:
                                link_flows_dict[link.id] = init_flow
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
                                link_densities_dict[link.id] = init_density
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

                            elif init_speed.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial speed array for motorway link {link.id} has incorrect length. Using first value for all cells instead."
                                )
                                link_speeds_dict[link.id] = np.full(
                                    num_cells, init_speed[0]
                                )

                            else:
                                link_speeds_dict[link.id] = init_speed
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
                        if dest_id not in destination_flow_bc:
                            raise ValueError(
                                f"Destination flow boundary condition function for destination {dest_id} (connected to offramp {link.id}) not provided and cannot be inferred."
                            )
                        else:
                            destination_flow_bc_dict[dest_id] = destination_flow_bc[
                                dest_id
                            ]

                        if dest_id not in destination_density_bc:
                            warnings.warn(
                                f"Destination density boundary condition function for destination {dest_id} (connected to offramp {link.id}) not provided. Assuming downstream free flow conditions (zero density)."
                            )
                            # capture dest_id to avoid late-binding (if lambda ever uses it)
                            destination_density_bc_dict[dest_id] = (
                                lambda _, dest_id=dest_id: 0.0
                            )
                        else:
                            destination_density_bc_dict[dest_id] = (
                                destination_density_bc[dest_id]
                            )
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

                            elif init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for offramp {link.id} has incorrect length. Using first value instead."
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])

                            else:
                                link_flows_dict[link.id] = init_flow
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
                    if link.id not in destination_flow_bc:
                        raise ValueError(
                            f"Destination flow boundary condition function for destination {link.id} (connected to offramp {link.id}) not provided and cannot be inferred."
                        )
                    else:
                        destination_flow_bc_dict[link.id] = destination_flow_bc[link.id]

                    if link.id not in destination_density_bc:
                        warnings.warn(
                            f"Destination density boundary condition function for destination {link.id} (connected to offramp {link.id}) not provided. Assuming downstream free flow conditions (zero density)."
                        )
                        # capture link.id to avoid late-binding (if lambda ever uses it)
                        destination_density_bc_dict[link.id] = (
                            lambda _, link_id=link.id: 0.0
                        )
                    else:
                        destination_density_bc_dict[link.id] = destination_density_bc[
                            link.id
                        ]

                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for destination {link.id} is empty."
                                )

                            elif init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for destination {link.id} has incorrect length. Using first value instead."
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])

                            else:
                                link_flows_dict[link.id] = init_flow
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

        return (
            link_flows_dict,
            link_densities_dict,
            link_speeds_dict,
            origin_queues_dict,
            onramp_queues_dict,
            offramp_queues_dict,
            turning_rates_dict,
            destination_flow_bc_dict,
            destination_density_bc_dict,
        )

    def _run_simulation_loop(
        self,
        system: casadi.Function,
        duration: float,
        dt: float,
        x0: NDArray[np.float64],
        num_origins: int,
        num_onramps: int,
        num_splits: int,
        num_destinations: int,
        origin_queues_dict: dict[str, float],
        onramp_queues_dict: dict[str, float],
        turning_rates_dict: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc_dict: dict[str, Callable[[float], float]],
        destination_density_bc_dict: dict[str, Callable[[float], float]],
        origin_demands: dict[str, Callable[[float], float]],
        onramp_demands: dict[str, Callable[[float], float]],
        model: Union["CTM", "METANET"],
        model_params: Union[METANETParams, None],
    ):
        """Execute the discrete-time simulation loop for the network.

        Advances the system state using the provided CasADi `system` update
        function for the requested duration and time-step. It evaluates the
        callable disturbance inputs (origin/onramp demands, turning rates and
        boundary conditions) at each time step, forms the disturbance vector,
        calls the model update, and stores state and disturbance histories.

        Args:
            system: CasADi function implementing the network state update.
            duration: Total simulation time (same units as `dt`).
            dt: Time-step for integration.
            x0: Initial packed state vector (NumPy array).
            num_origins: Number of origin queue entries in disturbance vector.
            num_onramps: Number of onramp queue entries in disturbance vector.
            num_splits: Number of turning-rate scalars per time step.
            num_destinations: Number of destination boundary condition entries.
            origin_queues_dict: Mapping origin id -> initial queue (used to order disturbances).
            onramp_queues_dict: Mapping onramp id -> initial queue (used to order disturbances).
            turning_rates_dict: Mapping node id -> callable(time) -> turning-rate dict.
            destination_flow_bc_dict: Mapping destination id -> callable(time) -> flow.
            destination_density_bc_dict: Mapping destination id -> callable(time) -> density.
            origin_demands: Mapping origin id -> callable(time) -> demand.
            onramp_demands: Mapping onramp id -> callable(time) -> demand.
            turning_rates: Mapping node id -> callable(time) -> turning-rate dict (user-provided).
            model: Model instance (provides parameter conversion helper).
            model_params: Parameters passed to the model when building `params`.

        Returns:
            Tuple `(time_array, state_history, disturbance_history)` where
            - `time_array` is a 1-D NumPy array of time points,
            - `state_history` is a 2-D NumPy array of packed states over time,
            - `disturbance_history` is a 2-D NumPy array of packed disturbances over time.
        """

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
                num_origins + num_onramps + num_splits + 2 * num_destinations,
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
                node_id: turning_rates_dict[node_id](time)
                for node_id in turning_rates_dict.keys()
            }
            flow_boundary_condition_dict = {
                destination_id: destination_flow_bc_dict[destination_id](time)
                for destination_id in destination_flow_bc_dict.keys()
            }
            density_boundary_condition_dict = {
                destination_id: destination_density_bc_dict[destination_id](time)
                for destination_id in destination_density_bc_dict.keys()
            }

            # combine the values into the disturbance vector for the state update
            d = self.network_dict_to_disturbance_vec(
                origin_demand_dict=origin_demand_dict,
                onramp_demand_dict=onramp_demand_dict,
                turning_rate_dict=turning_rate_dict,
                flow_boundary_condition_dict=flow_boundary_condition_dict,
                density_boundary_condition_dict=density_boundary_condition_dict,
            )

            # obtain the model parameters in vector form for the model update
            if model_params is not None and isinstance(model, METANET):
                params = model.model_params_to_vec(
                    network=self, model_params=model_params
                )
            elif model_params is None and isinstance(model, METANET):
                raise ValueError(
                    "METANET model requires model_params to be provided for simulation."
                )
            elif model_params is not None and isinstance(model, CTM):
                raise ValueError(
                    "CTM model does not support model_params to be provided for simulation."
                )
            else:
                params = np.array(
                    [], dtype=np.float64
                )  # empty array for models that don't require parameters

            # perform the state update
            x_next = system(params, state_history[:, t], d)

            # store the updated state and disturbance
            state_history[:, t + 1] = np.array(x_next).flatten()
            disturbance_history[:, t] = d

        return time_array, state_history, disturbance_history

    def simulate(
        self,
        duration: float,
        dt: float,
        model: Union["CTM", "METANET"],
        origin_demands: dict[
            str, Callable[[float], float]
        ],  # for each origin id, provide a callable function returning the demand at time t
        onramp_demands: dict[
            str, Callable[[float], float]
        ],  # for each onramp id, provide a callable function returning the demand at time t
        turning_rates: dict[
            str, Callable[[float], dict[str, float]]
        ],  # for each node id, provide a callable function returning a dict mapping outgoing link ids to split ratios at time t
        destination_flow_bc: dict[
            str, Callable[[float], float]
        ],  # for each destination id, provide a callable function returning the downstream flow at time t (for CTM)
        destination_density_bc: dict[
            str, Callable[[float], float]
        ],  # for each destination id, provide a callable function returning the downstream flow at time t (for METANET)
        model_params: Union[
            METANETParams, None
        ] = None,  # model parameters for models that require this -> CTM is fully defined by link parameters
        initial_flows: (
            dict[str, NDArray[np.float64] | float] | None
        ) = None,  # for each link id, provide either a float (uniform initial flow) or an array of floats (per-cell initial flows; default: 0)
        initial_densities: (
            dict[str, NDArray[np.float64] | float] | None
        ) = None,  # for each link id, provide either a float (uniform initial density) or an array of floats (per-cell initial densities; default: 0)
        initial_speeds: (
            dict[str, NDArray[np.float64] | float] | None
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
        plot_results: bool = True,
        show_plots: bool = False,
        results_dir: (
            str | None
        ) = None,  # directory for saving results; if None and plot_results=True, uses timestamped folder in results/
    ):
        """Simulate the network over a time horizon using the provided model.

        Runs a forward simulation using the provided traffic `model` (which
        must expose `network_update_function`) and the per-component callable
        inputs for demands, turning rates and boundary conditions. The
        routine will discretize motorway links, initialize state and
        disturbance vectors, perform time-stepping to update the state, and
        optionally plot and save results.

        Args:
            duration: Total simulation time (same units as demand functions, e.g. hours).
            dt: Simulation time step (same units as `duration`).
            model: Traffic model instance providing `network_update_function`.
            origin_demands: Mapping origin id -> callable(time) -> demand (veh/h).
            onramp_demands: Mapping onramp id -> callable(time) -> demand (veh/h).
            turning_rates: Mapping node id -> callable(time) -> dict[outgoing_link_id -> split rate].
            destination_flow_bc: Mapping destination id -> callable(time) -> downstream flow (veh/h/lane) (required for CTM).
            destination_density_bc: Mapping destination id -> callable(time) -> downstream density (veh/km/lane) (required for METANET).
            initial_flows: Optional mapping link id -> scalar or per-cell array for initial flows (default: zeros).
            initial_densities: Optional mapping link id -> scalar or per-cell array for initial densities (default: zeros for mainline links).
            initial_speeds: Optional mapping link id -> scalar or per-cell array for initial speeds (default: free-flow speed for motorway links).
            initial_origin_queues: Optional mapping origin id -> initial queue length (veh).
            initial_onramp_queues: Optional mapping onramp id -> initial queue length (veh).
            initial_offramp_queues: Optional mapping offramp id -> initial queue length (veh).
            preferred_cell_size: Preferred link segmentation size (km) used when partitioning motorway links.
            plot_results: If True, generate plots and save results to `results_dir`.
            show_plots: If True, display plots interactively.
            results_dir: Directory for saving results; if None a timestamped folder under `results/` is used when `plot_results` is True.

        Returns:
            tuple: `(time_array, state_history, disturbance_history)` where
                - `time_array` is a 1-D NumPy array of time points,
                - `state_history` is a 2-D NumPy array of packed states over time (state_size x timesteps),
                - `disturbance_history` is a 2-D NumPy array of packed disturbances over time (disturbance_size x timesteps-1).

        Raises:
            ValueError: If required inputs are missing or inconsistent with the network topology.
        """
        # ! 1 - validate all inputs as required
        # validate that all required model parameters are provided
        if isinstance(model, METANET) and model_params is not None:
            model.validate_model_params(model_params=model_params)

        # validate that all required initial conditions are provided
        self._validate_initial_conditions_numerical(
            origin_demands=origin_demands,
            onramp_demands=onramp_demands,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
            initial_flows=initial_flows,
            initial_densities=initial_densities,
            initial_speeds=initial_speeds,
        )

        # ! 2 - discretize mainline motorway links according to preferred cell size and CFL condition -> create cells and link correctly
        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    # if the node at the end of the current link represents a lane drop, set it accordingly
                    upcoming_lane_drop = self._compute_upcoming_lane_drop(link)

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
        (
            link_flows_dict,
            link_densities_dict,
            link_speeds_dict,
            origin_queues_dict,
            onramp_queues_dict,
            offramp_queues_dict,
            turning_rates_dict,
            destination_flow_bc_dict,
            destination_density_bc_dict,
        ) = self._augment_network_initialization(
            initial_flows=initial_flows,
            initial_densities=initial_densities,
            initial_speeds=initial_speeds,
            initial_origin_queues=initial_origin_queues,
            initial_onramp_queues=initial_onramp_queues,
            initial_offramp_queues=initial_offramp_queues,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
        )

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
            dt=dt,
        )

        # ! 6 - run the simulation loop, update all link states, and track outputs
        time_array, state_history, disturbance_history = self._run_simulation_loop(
            system=system,
            duration=duration,
            dt=dt,
            x0=cast(NDArray[np.float64], x0),
            num_origins=num_origins,
            num_onramps=num_onramps,
            num_splits=num_splits,
            num_destinations=num_destinations,
            origin_queues_dict=origin_queues_dict,
            onramp_queues_dict=onramp_queues_dict,
            turning_rates_dict=turning_rates_dict,
            destination_flow_bc_dict=destination_flow_bc_dict,
            destination_density_bc_dict=destination_density_bc_dict,
            origin_demands=origin_demands,
            onramp_demands=onramp_demands,
            model=model,
            model_params=model_params,
        )

        # ! 7 - plotting of simulation results, network structure and saving to results directory
        if plot_results:
            # create timestamped results directory if not specified
            if results_dir is None:
                timestamp = datetime.now().strftime(
                    "simulation_results_%Y-%m-%d_%H%M%S"
                )
                results_dir = f"results/{timestamp}"

            os.makedirs(results_dir, exist_ok=True)
            print(f"Saving simulation results to {results_dir}")

            # save network topology plot
            topology_path = os.path.join(results_dir, "network_topology.png")
            self.plot(show=show_plots, save_path=topology_path)
            print(f"  Network topology saved to {topology_path}")

            # save simulation results as JSON file
            results_path = os.path.join(results_dir, "simulation_results.json")
            self.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=results_path,
                model=model,
                dt=dt,
                duration=duration,
                preferred_cell_size=preferred_cell_size,
                model_params=model_params,
            )
            print(f"  Simulation results saved to {results_path}")

            # save network structure as text file
            structure_path = os.path.join(results_dir, "network_structure.txt")
            self.save_to_txt(structure_path)
            structure_path_json = os.path.join(results_dir, "network_structure.json")
            self.save_to_json(structure_path_json)
            print(f"  Network structure saved to {structure_path}")

            # plot simulation results
            self.plot_simulation_results(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                save_dir=results_dir,
            )

        return time_array, state_history, disturbance_history

    # endregion

    # ! Evaluation and result visualizations
    # region
    def save_simulation_results_json(
        self,
        time_array: NDArray[np.float64],
        state_history: NDArray[np.float64],
        disturbance_history: NDArray[np.float64],
        filepath: str,
        model: Union["CTM", "METANET"],
        dt: float,
        duration: float,
        preferred_cell_size: float,
        model_params: Union["METANETParams", None] = None,
    ) -> None:
        """Save simulation results to a JSON file with comprehensive metadata.

        Writes the time array, state history and disturbance history to a
        JSON file for later reference. State and disturbance histories are
        split into link/node specific time series using the network unpacking
        helpers. Additionally saves simulation metadata including model type,
        simulation parameters, critical densities for each link, and model
        parameters (if applicable) for full reproducibility.

        Args:
            time_array: 1-D array of time points.
            state_history: 2-D array of state vectors over time.
            disturbance_history: 2-D array of disturbances over time.
            filepath: Path where the JSON file should be saved.
            model: The traffic flow model used for simulation (CTM or METANET).
            dt: Simulation time step.
            duration: Total simulation duration.
            preferred_cell_size: Preferred cell size used for link discretization.
            model_params: Model parameters for METANET (None for CTM).
        """

        # prepare containers for per-link / per-node time series
        flows_time: dict[str, list] = {}
        densities_time: dict[str, list] = {}
        speeds_time: dict[str, list] = {}
        origin_queues_time: dict[str, list] = {}
        onramp_queues_time: dict[str, list] = {}
        offramp_queues_time: dict[str, list] = {}

        origin_demands_time: dict[str, list] = {}
        onramp_demands_time: dict[str, list] = {}
        turning_rates_time: dict[str, dict[str, list]] = {}
        flow_boundary_conditions_time: dict[str, list] = {}
        density_boundary_conditions_time: dict[str, list] = {}

        num_timesteps = state_history.shape[1]

        # split state history into per-link/node time series
        for t in range(num_timesteps):
            flows_t, densities_t, speeds_t, origin_q_t, onramp_q_t, offramp_q_t = (
                self.state_vec_to_network_dict(state_history[:, t])
            )

            # initialize containers on first timestep
            if t == 0:
                for k in flows_t.keys():
                    flows_time[k] = []
                for k in densities_t.keys():
                    densities_time[k] = []
                for k in speeds_t.keys():
                    speeds_time[k] = []
                for k in origin_q_t.keys():
                    origin_queues_time[k] = []
                for k in onramp_q_t.keys():
                    onramp_queues_time[k] = []
                for k in offramp_q_t.keys():
                    offramp_queues_time[k] = []

            # append values (convert numpy -> native Python types)
            for k, v in flows_t.items():
                flows_time[k].append(np.asarray(v).tolist())
            for k, v in densities_t.items():
                densities_time[k].append(np.asarray(v).tolist())
            for k, v in speeds_t.items():
                speeds_time[k].append(np.asarray(v).tolist())

            for k, v in origin_q_t.items():
                origin_queues_time[k].append(float(np.asarray(v).tolist()))
            for k, v in onramp_q_t.items():
                onramp_queues_time[k].append(float(np.asarray(v).tolist()))
            for k, v in offramp_q_t.items():
                offramp_queues_time[k].append(float(np.asarray(v).tolist()))

        # split disturbance history into per-component time series
        if disturbance_history.size > 0:
            num_dist_timesteps = disturbance_history.shape[1]
            for t in range(num_dist_timesteps):
                (
                    origin_d_t,
                    onramp_d_t,
                    turning_t,
                    boundary_flow_t,
                    boundary_density_t,
                ) = self.disturbance_vec_to_network_dict(disturbance_history[:, t])

                if t == 0:
                    for k in origin_d_t.keys():
                        origin_demands_time[k] = []
                    for k in onramp_d_t.keys():
                        onramp_demands_time[k] = []
                    for node_id, inner in turning_t.items():
                        turning_rates_time[node_id] = {lk: [] for lk in inner.keys()}
                    for k in boundary_flow_t.keys():
                        flow_boundary_conditions_time[k] = []
                    for k in boundary_density_t.keys():
                        density_boundary_conditions_time[k] = []

                for k, v in origin_d_t.items():
                    origin_demands_time[k].append(float(np.asarray(v).tolist()))
                for k, v in onramp_d_t.items():
                    onramp_demands_time[k].append(float(np.asarray(v).tolist()))

                for node_id, inner in turning_t.items():
                    for lk, rate in inner.items():
                        turning_rates_time.setdefault(node_id, {}).setdefault(
                            lk, []
                        ).append(float(np.asarray(rate).tolist()))

                for k, v in boundary_flow_t.items():
                    flow_boundary_conditions_time[k].append(
                        float(np.asarray(v).tolist())
                    )
                for k, v in boundary_density_t.items():
                    density_boundary_conditions_time[k].append(
                        float(np.asarray(v).tolist())
                    )

        critical_densities: dict[str, float] = {}
        link_properties: dict[str, dict] = {}

        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if isinstance(model, CTM):
                        rho_crit = model.critical_density(
                            lane_capacity=link.Qc_lane,
                            free_flow_speed=link.vf,
                        )
                    elif isinstance(model, METANET):
                        if model_params is None:
                            raise ValueError(
                                "model_params required for METANET critical density calculation"
                            )
                        rho_crit = model.critical_density(
                            params=model_params,
                            link_id=link.id,
                            lane_capacity=link.Qc_lane,
                            free_flow_speed=link.vf,
                        )
                    else:
                        raise ValueError(f"Unknown model type: {type(model).__name__}")

                    critical_densities[link.id] = rho_crit

                    # extract cell lengths array (cells may have different lengths)
                    cell_lengths = [float(cell.length) for cell in link]

                    link_properties[link.id] = {
                        "length": link.length,
                        "lanes": link.lanes,
                        "lane_capacity": link.Qc_lane,
                        "free_flow_speed": link.vf,
                        "jam_density": link.rho_jam,
                        "num_cells": len(link),
                        "cell_lengths": cell_lengths,
                    }

        # build metadata section
        metadata = {
            "model_type": type(model).__name__,
            "simulation_parameters": {
                "dt": dt,
                "duration": duration,
                "preferred_cell_size": preferred_cell_size,
            },
            "link_properties": link_properties,
            "critical_densities": critical_densities,
        }

        # add model parameters if METANET
        if isinstance(model, METANET) and model_params is not None:
            # convert model_params to serializable format
            # handle alpha separately as it can be float or dict[str, float]
            metadata["model_parameters"] = {
                "tau": model_params["tau"],
                "nu": model_params["nu"],
                "kappa": model_params["kappa"],
                "delta": model_params["delta"],
                "phi": model_params["phi"],
                "alpha": (
                    {
                        link_id: alpha_val
                        for link_id, alpha_val in model_params["alpha"].items()
                    }
                    if isinstance(model_params["alpha"], dict)
                    else float(model_params["alpha"])
                ),
            }

        # assemble output structure
        out = {
            "metadata": metadata,
            "time_array": np.asarray(time_array).tolist(),
            "state_time_series": {
                "flows": flows_time,
                "densities": densities_time,
                "speeds": speeds_time,
                "origin_queues": origin_queues_time,
                "onramp_queues": onramp_queues_time,
                "offramp_queues": offramp_queues_time,
            },
            "disturbance_time_series": {
                "origin_demands": origin_demands_time,
                "onramp_demands": onramp_demands_time,
                "turning_rates": turning_rates_time,
                "flow_boundary_conditions": flow_boundary_conditions_time,
                "density_boundary_conditions": density_boundary_conditions_time,
            },
        }

        # write JSON to file
        with open(filepath, "w") as f:
            json.dump(out, f, indent=2)

    @classmethod
    def load_simulation_results_json(
        cls,
        filepath: str,
        network: "Network",
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        dict | None,
    ]:
        """Load simulation results from a JSON file with validation.

        Reads a JSON file created by `save_simulation_results_json` and validates
        that all required fields are present and contain valid numerical data.
        Returns the reconstructed time array, state history, disturbance history,
        and metadata (if present in the file).

        Args:
            filepath: Path to the JSON file containing simulation results.
            network: Network instance to use for validating structure against saved data.

        Returns:
            Tuple of (time_array, state_history, disturbance_history, metadata) where:
            - time_array: 1-D NumPy array of time points
            - state_history: 2-D NumPy array of state vectors over time
            - disturbance_history: 2-D NumPy array of disturbances over time
            - metadata: Dictionary containing simulation metadata (model type, parameters,
              critical densities, link properties) or None if not present in file

        Raises:
            ValueError: If required fields are missing or data validation fails.
            FileNotFoundError: If the specified file does not exist.
        """
        # load JSON data from file
        with open(filepath, "r") as f:
            data = json.load(f)

        # extract metadata if present (optional for backward compatibility)
        metadata = data.get("metadata", None)

        # validate top-level structure
        required_fields = ["time_array", "state_time_series", "disturbance_time_series"]
        for field in required_fields:
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}' in simulation results file."
                )

        # validate state_time_series structure
        state_required = [
            "flows",
            "densities",
            "speeds",
            "origin_queues",
            "onramp_queues",
            "offramp_queues",
        ]
        for field in state_required:
            if field not in data["state_time_series"]:
                raise ValueError(
                    f"Missing required field '{field}' in state_time_series."
                )

        # validate disturbance_time_series structure
        disturbance_required = [
            "origin_demands",
            "onramp_demands",
            "turning_rates",
            "flow_boundary_conditions",
            "density_boundary_conditions",
        ]
        for field in disturbance_required:
            if field not in data["disturbance_time_series"]:
                raise ValueError(
                    f"Missing required field '{field}' in disturbance_time_series."
                )

        # convert time_array to NumPy array
        time_array = np.array(data["time_array"], dtype=np.float64)
        if time_array.ndim != 1:
            raise ValueError("time_array must be a 1-D array.")

        num_timesteps = len(time_array)

        # reconstruct state dictionaries for validation and conversion
        state_series = data["state_time_series"]

        # validate that all link IDs in saved data match network structure
        for node in network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Origin):
                    if link.id not in state_series["origin_queues"]:
                        raise ValueError(
                            f"Origin queue data for '{link.id}' not found in saved results."
                        )
                elif isinstance(link, Onramp):
                    if link.id not in state_series["flows"]:
                        raise ValueError(
                            f"Flow data for onramp '{link.id}' not found in saved results."
                        )
                    if link.id not in state_series["onramp_queues"]:
                        raise ValueError(
                            f"Onramp queue data for '{link.id}' not found in saved results."
                        )

            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if link.id not in state_series["flows"]:
                        raise ValueError(
                            f"Flow data for motorway link '{link.id}' not found in saved results."
                        )
                    if link.id not in state_series["densities"]:
                        raise ValueError(
                            f"Density data for motorway link '{link.id}' not found in saved results."
                        )
                    if link.id not in state_series["speeds"]:
                        raise ValueError(
                            f"Speed data for motorway link '{link.id}' not found in saved results."
                        )
                elif isinstance(link, Offramp):
                    if link.id not in state_series["flows"]:
                        raise ValueError(
                            f"Flow data for offramp '{link.id}' not found in saved results."
                        )
                    if link.id not in state_series["offramp_queues"]:
                        raise ValueError(
                            f"Offramp queue data for '{link.id}' not found in saved results."
                        )
                elif isinstance(link, Destination):
                    if link.id not in state_series["flows"]:
                        raise ValueError(
                            f"Flow data for destination '{link.id}' not found in saved results."
                        )

        # reconstruct state_history by iterating through timesteps
        # first, get state size from network
        state_dicts = []
        for t in range(num_timesteps):
            flows_t = {
                str(id): np.array(v[t], dtype=np.float64)
                for id, v in state_series["flows"].items()
            }
            densities_t = {
                str(id): np.array(v[t], dtype=np.float64)
                for id, v in state_series["densities"].items()
            }
            speeds_t = {
                str(id): np.array(v[t], dtype=np.float64)
                for id, v in state_series["speeds"].items()
            }
            origin_queues_t = {
                str(id): float(v[t]) for id, v in state_series["origin_queues"].items()
            }
            onramp_queues_t = {
                str(id): float(v[t]) for id, v in state_series["onramp_queues"].items()
            }
            offramp_queues_t = {
                str(id): float(v[t]) for id, v in state_series["offramp_queues"].items()
            }

            # validate numerical data using existing validation (for first timestep)
            if t == 0:
                network._validate_state_history_numerical(
                    flows=flows_t,
                    densities=densities_t,
                    speeds=speeds_t,
                    origin_queues=origin_queues_t,
                    onramp_queues=onramp_queues_t,
                    offramp_queues=offramp_queues_t,
                )

            state_dicts.append(
                (
                    flows_t,
                    densities_t,
                    speeds_t,
                    origin_queues_t,
                    onramp_queues_t,
                    offramp_queues_t,
                )
            )

        # pack into state vectors
        state_vecs = []
        for (
            flows_t,
            densities_t,
            speeds_t,
            origin_q_t,
            onramp_q_t,
            offramp_q_t,
        ) in state_dicts:
            x_t, *_ = network.network_dict_to_state_vec(
                flow_dict=flows_t,
                density_dict=densities_t,
                speed_dict=speeds_t,
                origin_queue_dict=origin_q_t,
                onramp_queue_dict=onramp_q_t,
                offramp_queue_dict=offramp_q_t,
            )
            state_vecs.append(x_t)

        state_history = np.column_stack(state_vecs)

        # reconstruct disturbance_history similarly
        disturbance_series = data["disturbance_time_series"]
        num_dist_timesteps = (
            len(next(iter(disturbance_series["origin_demands"].values())))
            if disturbance_series["origin_demands"]
            else 0
        )

        disturbance_vecs = []
        for t in range(num_dist_timesteps):
            origin_demands_t = {
                k: float(v[t]) for k, v in disturbance_series["origin_demands"].items()
            }
            onramp_demands_t = {
                k: float(v[t]) for k, v in disturbance_series["onramp_demands"].items()
            }
            turning_rates_t = {
                node_id: {lk: float(rates[t]) for lk, rates in inner.items()}
                for node_id, inner in disturbance_series["turning_rates"].items()
            }
            flow_bc_t = {
                k: float(v[t])
                for k, v in disturbance_series["flow_boundary_conditions"].items()
            }
            density_bc_t = {
                k: float(v[t])
                for k, v in disturbance_series["density_boundary_conditions"].items()
            }

            # validate numerical data using existing validation (for first timestep)
            if t == 0:
                network._validate_disturbance_history_numerical(
                    origin_demands=cast(dict[str, float], origin_demands_t),
                    onramp_demands=cast(dict[str, float], onramp_demands_t),
                    turning_rates=cast(dict[str, dict[str, float]], turning_rates_t),
                    flow_boundary_conditions=cast(dict[str, float], flow_bc_t),
                    density_boundary_conditions=cast(dict[str, float], density_bc_t),
                )

            d_t = network.network_dict_to_disturbance_vec(
                origin_demand_dict=origin_demands_t,
                onramp_demand_dict=onramp_demands_t,
                turning_rate_dict=turning_rates_t,
                flow_boundary_condition_dict=flow_bc_t,
                density_boundary_condition_dict=density_bc_t,
            )
            disturbance_vecs.append(d_t)

        disturbance_history = (
            np.column_stack(disturbance_vecs) if disturbance_vecs else np.array([])
        )

        return time_array, state_history, disturbance_history, metadata

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

    def compute_performance_metrics(
        self,
        states: NDArray[np.float64],
        dt: float,
        timesteps: int,
    ) -> Tuple[float, float, float]:
        """Compute a set of performance metrics based on the provided simulation results

        Args:
            flow: 2-D array shape (num_cells, time_steps) in veh/h.
            density: 2-D array shape (num_cells, time_steps) in veh/km/lane.
            speed: 2-D array shape (num_cells, time_steps) in km/h.
            input_queue: 1-D array of input queue lengths over time (veh).
            onramp_queues: 2-D array shape (num_cells, time_steps) of onramp queues (veh).
            dt: Time step used in the simulation (hours).

        Returns:
            (VKT, VHT, overall_avg_speed) floats: vehicle-kilometres travelled,
                vehicle-hours travelled, and overall average speed.
        """

        # ! Part 1: Calculate VKT and VHT
        VKT: float = 0.0
        VHT: float = 0.0

        for t in range(timesteps - 1):
            # extract the different states at time t from the state vector
            flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
                self.state_vec_to_network_dict(states[:, t])
            )

            # verify that the extracted values are numerical arrays
            self._validate_state_history_numerical(
                flows=cast(dict[str, NDArray[np.float64]], flows),
                densities=cast(dict[str, NDArray[np.float64]], densities),
                speeds=cast(dict[str, NDArray[np.float64]], speeds),
                origin_queues=cast(dict[str, float], origin_queues),
                onramp_queues=cast(dict[str, float], onramp_queues),
                offramp_queues=cast(dict[str, float], offramp_queues),
            )

            # typecast to np.ndarray to ensure type safety
            flows = {k: np.asarray(v) for k, v in flows.items()}
            densities = {k: np.asarray(v) for k, v in densities.items()}
            speeds = {k: np.asarray(v) for k, v in speeds.items()}
            origin_queues = {k: float(v) for k, v in origin_queues.items()}
            onramp_queues = {k: float(v) for k, v in onramp_queues.items()}
            offramp_queues = {k: float(v) for k, v in offramp_queues.items()}

            # add the time vehicles spent in the origin, onramp, and offramp queues (veh * hours)
            VHT += dt * sum(origin_queues.values())
            VHT += dt * sum(onramp_queues.values())
            VHT += dt * sum(offramp_queues.values())

            # iterate over all outgoing motorway links and accumulate VKT and VHT
            for node in self.list_nodes():
                for link in node.outgoing:
                    if isinstance(link, MotorwayLink):
                        for idx, cell in link.enumerate_cells():
                            if cell is None:
                                raise ValueError(
                                    f"Cell {idx} not found in motorway link {link.id}."
                                )

                            # add VKT: distance * vehicles that passed (flow is veh/h)
                            VKT += cell.length * dt * flows[link.id][idx]

                            # add VHT for vehicles on the mainline segment (density is veh/km/lane)
                            VHT += (
                                cell.length * dt * densities[link.id][idx] * link.lanes
                            )

        # ! Part 2: Calculate vehicle-weighted average speed
        overall_avg_speed: float = VKT / VHT if VHT > 0 else 0.0

        return VKT, VHT, float(overall_avg_speed)

    def plot_simulation_results(
        self,
        time_array: NDArray[np.float64],
        state_history: NDArray[np.float64],
        disturbance_history: NDArray[np.float64],
        save_dir: str = "results",
    ) -> None:
        """Plot comprehensive simulation results for the network.

        Creates multiple figures showing density, flow, speed for all mainline
        links, demand/flow/queue plots for origins and onramps, flow plots for
        offramps and destinations, 3D surface plots for each motorway link, and
        summary plots per node showing all inflows and outflows.

        Args:
            time_array: 1-D array of time points (hours).
            state_history: 2-D array of state vectors over time, shape (state_size, timesteps).
            disturbance_history: 2-D array of disturbances over time, shape (disturbance_size, timesteps-1).
            save_dir: Directory where plots should be saved (default: "results").
        """
        # create results directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # convert time to seconds for plotting
        time_seconds = time_array * 3600.0
        num_timesteps = len(time_array)
        print(f"Generating simulation result plots in {save_dir}...")

        # build dictionaries mapping link_id -> array of values over time
        flows_over_time: dict[str, np.ndarray] = {}
        densities_over_time: dict[str, np.ndarray] = {}
        speeds_over_time: dict[str, np.ndarray] = {}
        origin_queues_over_time: dict[str, np.ndarray] = {}
        onramp_queues_over_time: dict[str, np.ndarray] = {}
        offramp_queues_over_time: dict[str, np.ndarray] = {}

        for t in range(num_timesteps):
            (
                flows_t,
                densities_t,
                speeds_t,
                origin_queues_t,
                onramp_queues_t,
                offramp_queues_t,
            ) = self.state_vec_to_network_dict(state_history[:, t])

            # make sure that the history values are numerical
            self._validate_state_history_numerical(
                flows=cast(dict[str, NDArray[np.float64]], flows_t),
                densities=cast(dict[str, NDArray[np.float64]], densities_t),
                speeds=cast(dict[str, NDArray[np.float64]], speeds_t),
                origin_queues=cast(dict[str, float], origin_queues_t),
                onramp_queues=cast(dict[str, float], onramp_queues_t),
                offramp_queues=cast(dict[str, float], offramp_queues_t),
            )

            # typecast to np.ndarray to ensure type safety
            flows_t = {k: np.asarray(v) for k, v in flows_t.items()}
            densities_t = {k: np.asarray(v) for k, v in densities_t.items()}
            speeds_t = {k: np.asarray(v) for k, v in speeds_t.items()}
            origin_queues_t = {k: np.asarray(v) for k, v in origin_queues_t.items()}
            onramp_queues_t = {k: np.asarray(v) for k, v in onramp_queues_t.items()}
            offramp_queues_t = {k: np.asarray(v) for k, v in offramp_queues_t.items()}

            # initialize dictionaries on first iteration
            if t == 0:
                for link_id in flows_t.keys():
                    flows_over_time[link_id] = np.zeros(
                        (len(flows_t[link_id]), num_timesteps)
                    )

                for link_id in densities_t.keys():
                    densities_over_time[link_id] = np.zeros(
                        (len(densities_t[link_id]), num_timesteps)
                    )

                for link_id in speeds_t.keys():
                    speeds_over_time[link_id] = np.zeros(
                        (len(speeds_t[link_id]), num_timesteps)
                    )

                for origin_id in origin_queues_t.keys():
                    origin_queues_over_time[origin_id] = np.zeros(num_timesteps)

                for onramp_id in onramp_queues_t.keys():
                    onramp_queues_over_time[onramp_id] = np.zeros(num_timesteps)

                for offramp_id in offramp_queues_t.keys():
                    offramp_queues_over_time[offramp_id] = np.zeros(num_timesteps)

            # store values for the current timestep
            for link_id, val in flows_t.items():
                flows_over_time[link_id][:, t] = val

            for link_id, val in densities_t.items():
                densities_over_time[link_id][:, t] = val

            for link_id, val in speeds_t.items():
                speeds_over_time[link_id][:, t] = val

            for origin_id, val in origin_queues_t.items():
                origin_queues_over_time[origin_id][t] = float(val)

            for onramp_id, val in onramp_queues_t.items():
                onramp_queues_over_time[onramp_id][t] = float(val)

            for offramp_id, val in offramp_queues_t.items():
                offramp_queues_over_time[offramp_id][t] = float(val)

        # extract demands from disturbance_history
        origin_demands_over_time = {}
        onramp_demands_over_time = {}

        for t in range(num_timesteps - 1):
            origin_demands_t, onramp_demands_t, _, _, _ = (
                self.disturbance_vec_to_network_dict(disturbance_history[:, t])
            )

            if t == 0:
                for origin_id in origin_demands_t.keys():
                    origin_demands_over_time[origin_id] = np.zeros(num_timesteps - 1)
                for onramp_id in onramp_demands_t.keys():
                    onramp_demands_over_time[onramp_id] = np.zeros(num_timesteps - 1)

            for origin_id, val in origin_demands_t.items():
                origin_demands_over_time[origin_id][t] = float(val)

            for onramp_id, val in onramp_demands_t.items():
                onramp_demands_over_time[onramp_id][t] = float(val)

        # ===== PART 1: Per-Link Plots for MotorwayLinks =====
        print("  Creating per-link density/flow/speed plots...")
        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    self._plot_motorway_link_results(
                        link=link,
                        time_seconds=time_seconds,
                        densities=densities_over_time[link.id],
                        flows=flows_over_time[link.id],
                        speeds=speeds_over_time[link.id],
                        save_dir=save_dir,
                    )

        # ===== PART 2: Per-Node Inflow Plots (Origins & Onramps) =====
        print("  Creating per-node inflow plots (origins & onramps)...")
        for node in self.list_nodes():
            inflow_components = [
                link for link in node.incoming if isinstance(link, (Origin, Onramp))
            ]
            if inflow_components:
                self._plot_node_inflows(
                    node=node,
                    inflow_components=inflow_components,
                    time_seconds=time_seconds,
                    flows_over_time=flows_over_time,
                    origin_queues_over_time=origin_queues_over_time,
                    onramp_queues_over_time=onramp_queues_over_time,
                    origin_demands_over_time=origin_demands_over_time,
                    onramp_demands_over_time=onramp_demands_over_time,
                    save_dir=save_dir,
                )

        # ===== PART 3: Per-Node Outflow Plots (Offramps & Destinations) =====
        print("  Creating per-node outflow plots (offramps & destinations)...")
        for node in self.list_nodes():
            outflow_components = [
                link
                for link in node.outgoing
                if isinstance(link, (Offramp, Destination))
            ]
            if outflow_components:
                self._plot_node_outflows(
                    node=node,
                    outflow_components=outflow_components,
                    time_seconds=time_seconds,
                    flows_over_time=flows_over_time,
                    offramp_queues_over_time=offramp_queues_over_time,
                    save_dir=save_dir,
                )

        # ===== PART 4: 3D Surface Plots for each MotorwayLink =====
        print("  Creating 3D surface plots for motorway links...")
        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink) and len(link) > 1:
                    # skip single-cell links for 3D plots (no spatial dimension to visualize)
                    self._plot_motorway_link_3d(
                        link=link,
                        time_seconds=time_seconds,
                        densities=densities_over_time[link.id],
                        flows=flows_over_time[link.id],
                        speeds=speeds_over_time[link.id],
                        save_dir=save_dir,
                    )

        # ===== PART 5: Per-Node Summary Plots (All Inflows & Outflows) =====
        print("  Creating per-node summary plots...")
        for node in self.list_nodes():
            self._plot_node_summary(
                node=node,
                time_seconds=time_seconds,
                flows_over_time=flows_over_time,
                save_dir=save_dir,
            )

        print(f"All plots saved to {save_dir}")

    def _plot_motorway_link_results(
        self,
        link: MotorwayLink,
        time_seconds: NDArray[np.float64],
        densities: NDArray[np.float64],
        flows: NDArray[np.float64],
        speeds: NDArray[np.float64],
        save_dir: str,
    ) -> None:
        """Create density, flow, and speed plots for a motorway link.

        Args:
            link: The MotorwayLink to plot.
            time_seconds: 1-D array of time points in seconds.
            densities: 2-D array of densities over time (cells x time).
            flows: 2-D array of flows over time (cells x time).
            speeds: 2-D array of speeds over time (cells x time).
            save_dir: Directory where plots should be saved.
        """
        num_cells = len(link)
        ncols = 3
        nrows = math.ceil(num_cells / ncols)
        actual_duration = time_seconds[-1]

        # calculate max values for y-axis scaling
        max_density = max(np.max(densities) * 1.1, link.rho_jam * 1.1)
        max_flow = max(np.max(flows[:, :-1]) * 1.1, link.Qc * 1.1)
        max_speed = max(np.max(speeds[:, :-1]) * 1.1, link.vf * 1.1)

        # figure 1: Density
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig1.suptitle(
            f"Vehicle Density - Link {link.id}", fontsize=14, fontweight="bold"
        )

        # properly handle axes array structure
        if nrows == 1 and ncols == 1:
            axes1 = [axes1]
        elif nrows == 1 or ncols == 1:
            axes1 = axes1.flatten()
        else:
            axes1 = axes1.flatten()

        for i, _ in link.enumerate_cells():
            axes1[i].plot(time_seconds, densities[i, :], linewidth=1.5)
            axes1[i].axhline(link.rho_jam, color="red", linestyle="--", linewidth=1)
            axes1[i].set_ylim([0, max(link.rho_jam * 1.1, max_density)])
            axes1[i].set_xlim([0, actual_duration])
            axes1[i].set_xlabel("time (s)")
            axes1[i].set_ylabel("density (veh/km/lane)")
            axes1[i].grid(True)
            axes1[i].set_title(f"Cell {i + 1}")

        for ax in axes1[num_cells:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_density.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig1)

        # figure 2: Flow
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig2.suptitle(f"Vehicle Flow - Link {link.id}", fontsize=14, fontweight="bold")

        # properly handle axes array structure
        if nrows == 1 and ncols == 1:
            axes2 = [axes2]
        elif nrows == 1 or ncols == 1:
            axes2 = axes2.flatten()
        else:
            axes2 = axes2.flatten()

        for i, _ in link.enumerate_cells():
            axes2[i].plot(
                time_seconds[:-1], flows[i, :-1], linewidth=1.5, label="Cell outflow"
            )
            axes2[i].axhline(link.Qc, color="red", linestyle="--", linewidth=1)
            axes2[i].set_ylim([0, max(link.Qc * 1.1, max_flow)])
            axes2[i].set_xlim([0, actual_duration])
            axes2[i].set_xlabel("time (s)")
            axes2[i].set_ylabel("flow (veh/h)")
            axes2[i].grid(True)
            axes2[i].set_title(f"Cell {i + 1}")

        for ax in axes2[num_cells:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_flow.png"), dpi=200, bbox_inches="tight"
        )
        plt.close(fig2)

        # figure 3: Speed
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig3.suptitle(f"Vehicle Speed - Link {link.id}", fontsize=14, fontweight="bold")

        # properly handle axes array structure
        if nrows == 1 and ncols == 1:
            axes3 = [axes3]
        elif nrows == 1 or ncols == 1:
            axes3 = axes3.flatten()
        else:
            axes3 = axes3.flatten()

        for i, _ in link.enumerate_cells():
            vf_cell = link.vf
            axes3[i].plot(time_seconds[:-1], speeds[i, :-1], linewidth=1.5)
            axes3[i].axhline(vf_cell, color="red", linestyle="--", linewidth=1)
            axes3[i].set_ylim([0, max(vf_cell * 1.1, max_speed)])
            axes3[i].set_xlim([0, actual_duration])
            axes3[i].set_xlabel("time (s)")
            axes3[i].set_ylabel("speed (km/h)")
            axes3[i].grid(True)
            axes3[i].set_title(f"Cell {i + 1}")

        for ax in axes3[num_cells:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_speed.png"), dpi=200, bbox_inches="tight"
        )
        plt.close(fig3)

    def _plot_node_inflows(
        self,
        node: Node,
        inflow_components: list,
        time_seconds: NDArray[np.float64],
        flows_over_time: dict,
        origin_queues_over_time: dict,
        onramp_queues_over_time: dict,
        origin_demands_over_time: dict,
        onramp_demands_over_time: dict,
        save_dir: str,
    ) -> None:
        """Create inflow plots (demand+flow and queue) for origins and onramps at a node."""
        num_inflows = len(inflow_components)
        ncols = 2  # demand+flow, queue
        nrows = num_inflows

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        fig.suptitle(f"Inflows at Node {node.id}", fontsize=14, fontweight="bold")

        # normalize axes indexing to 2D
        if num_inflows == 1:
            axes = np.array([[axes[0], axes[1]]])
        else:
            axes = np.array(axes).reshape(nrows, ncols)

        actual_duration = time_seconds[-1]

        for row_idx, link in enumerate(inflow_components):
            link_id = link.id
            is_origin = isinstance(link, Origin)

            # get demand and flow data
            if is_origin:
                demand = origin_demands_over_time.get(
                    link_id, np.zeros(len(time_seconds) - 1)
                )
                queue = origin_queues_over_time.get(
                    link_id, np.zeros(len(time_seconds))
                )
            else:
                demand = onramp_demands_over_time.get(
                    link_id, np.zeros(len(time_seconds) - 1)
                )
                queue = onramp_queues_over_time.get(
                    link_id, np.zeros(len(time_seconds))
                )

            flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))
            if len(flow.shape) > 1:
                flow = flow[0, :]

            # calculate max values for scaling
            max_demand = np.max(demand) * 1.1 if np.max(demand) > 0 else 2500
            max_flow = np.max(flow[:-1]) * 1.1 if np.max(flow[:-1]) > 0 else 2500
            max_queue = np.max(queue[:-1]) * 1.1 if np.max(queue[:-1]) > 0 else 100
            combined_max = max(max_demand, max_flow)

            # plot demand and flow
            axes[row_idx, 0].plot(
                time_seconds[:-1], demand, linewidth=1.5, label="Demand"
            )
            axes[row_idx, 0].plot(
                time_seconds[:-1], flow[:-1], linewidth=1.5, label="Flow"
            )
            axes[row_idx, 0].grid(True)
            axes[row_idx, 0].set_xlim([0, actual_duration])
            axes[row_idx, 0].set_ylim([0, combined_max])
            axes[row_idx, 0].set_xlabel("time (s)")
            axes[row_idx, 0].set_ylabel("veh/h")
            axes[row_idx, 0].set_title(
                f"{type(link).__name__} {link_id} - Demand & Flow"
            )
            axes[row_idx, 0].legend(fontsize="small", ncol=2, frameon=False)

            # plot queue
            axes[row_idx, 1].plot(
                time_seconds[:-1], queue[:-1], linewidth=1.5, color="tab:gray"
            )
            axes[row_idx, 1].grid(True)
            axes[row_idx, 1].set_xlim([0, actual_duration])
            axes[row_idx, 1].set_ylim([0, max_queue])
            axes[row_idx, 1].set_xlabel("time (s)")
            axes[row_idx, 1].set_ylabel("Queue (veh)")
            axes[row_idx, 1].set_title(f"{type(link).__name__} {link_id} - Queue")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"node_{node.id}_inflows.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_node_outflows(
        self,
        node: Node,
        outflow_components: list,
        time_seconds: NDArray[np.float64],
        flows_over_time: dict,
        offramp_queues_over_time: dict,
        save_dir: str,
    ) -> None:
        """Create outflow plots for offramps and destinations at a node.

        Args:
            node: The Node to plot.
            outflow_components: List of outgoing links (Offramp or Destination).
            time_seconds: 1-D array of time points in seconds.
            flows_over_time: Dictionary mapping link IDs to flow arrays over time.
            offramp_queues_over_time: Dictionary mapping offramp IDs to queue arrays over time.
            save_dir: Directory where plots should be saved.
        """
        num_outflows = len(outflow_components)

        # count offramps to determine if we need queue plots
        num_offramps = sum(
            1 for link in outflow_components if isinstance(link, Offramp)
        )
        ncols = 2 if num_offramps > 0 else 1  # flow, and queue if offramps exist
        nrows = num_outflows

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        fig.suptitle(f"Outflows at Node {node.id}", fontsize=14, fontweight="bold")

        # normalize axes indexing
        if num_outflows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif num_outflows == 1:
            axes = np.array([[axes[0], axes[1]]])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])
        else:
            axes = np.array(axes).reshape(nrows, ncols)

        actual_duration = time_seconds[-1]

        for row_idx, link in enumerate(outflow_components):
            link_id = link.id
            is_offramp = isinstance(link, Offramp)

            flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))
            if len(flow.shape) > 1:
                flow = flow[0, :]

            max_flow = np.max(flow[:-1]) * 1.1 if np.max(flow[:-1]) > 0 else 2500

            # plot flow
            axes[row_idx, 0].plot(time_seconds[:-1], flow[:-1], linewidth=1.5)
            axes[row_idx, 0].grid(True)
            axes[row_idx, 0].set_xlim([0, actual_duration])
            axes[row_idx, 0].set_ylim([0, max_flow])
            axes[row_idx, 0].set_xlabel("time (s)")
            axes[row_idx, 0].set_ylabel("flow (veh/h)")
            axes[row_idx, 0].set_title(f"{type(link).__name__} {link_id} - Flow")

            # plot queue if it's an offramp
            if is_offramp and ncols == 2:
                queue = offramp_queues_over_time.get(
                    link_id, np.zeros(len(time_seconds))
                )
                max_queue = np.max(queue[:-1]) * 1.1 if np.max(queue[:-1]) > 0 else 100

                axes[row_idx, 1].plot(
                    time_seconds[:-1], queue[:-1], linewidth=1.5, color="tab:gray"
                )
                axes[row_idx, 1].grid(True)
                axes[row_idx, 1].set_xlim([0, actual_duration])
                axes[row_idx, 1].set_ylim([0, max_queue])
                axes[row_idx, 1].set_xlabel("time (s)")
                axes[row_idx, 1].set_ylabel("Queue (veh)")
                axes[row_idx, 1].set_title(f"Offramp {link_id} - Queue")
            elif ncols == 2:
                # Hide queue subplot for destinations
                axes[row_idx, 1].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"node_{node.id}_outflows.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_motorway_link_3d(
        self,
        link: MotorwayLink,
        time_seconds: NDArray[np.float64],
        densities: NDArray[np.float64],
        flows: NDArray[np.float64],
        speeds: NDArray[np.float64],
        save_dir: str,
    ) -> None:
        """Create 3D surface plots for a motorway link."""
        num_cells = len(link)
        actual_duration = time_seconds[-1]

        # create meshgrids
        x_full, y_full = np.meshgrid(time_seconds, np.arange(1, num_cells + 1))
        x_truncated, y_truncated = np.meshgrid(
            time_seconds[:-1], np.arange(1, num_cells + 1)
        )

        # calculate max values
        max_rho_jam = link.rho_jam
        max_capacity = link.Qc
        max_vf = link.vf

        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(
            f"3D Visualization - Link {link.id}", fontsize=14, fontweight="bold"
        )

        # 3D density plot
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.plot_surface(
            x_full, y_full, densities, cmap="viridis", edgecolor="none", alpha=0.9
        )
        ax1.view_init(elev=30, azim=-37.5)
        ax1.set_xlabel("time (s)", rotation=30)
        ax1.set_ylabel("Cell", rotation=-37.5)
        ax1.set_zlabel("density (veh/km/lane)")
        ax1.set_xlim([0, actual_duration])
        ax1.set_ylim([1, num_cells])
        ax1.set_zlim([0, max_rho_jam * 1.1])

        # 3D flow plot
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.plot_surface(
            x_truncated,
            y_truncated,
            flows[:, :-1],
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )
        ax2.view_init(elev=30, azim=-37.5)
        ax2.set_xlabel("time (s)", rotation=30)
        ax2.set_ylabel("Cell", rotation=-37.5)
        ax2.set_zlabel("flow (veh/h)")
        ax2.set_xlim([0, actual_duration])
        ax2.set_ylim([1, num_cells])
        ax2.set_zlim([0, max_capacity * 1.1])

        # 3D speed plot
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot_surface(
            x_truncated,
            y_truncated,
            speeds[:, :-1],
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )
        ax3.view_init(elev=30, azim=-37.5)
        ax3.set_xlabel("time (s)", rotation=30)
        ax3.set_ylabel("Cell", rotation=-37.5)
        ax3.set_zlabel("speed (km/h)")
        ax3.set_xlim([0, actual_duration])
        ax3.set_ylim([1, num_cells])
        ax3.set_zlim([0, max_vf * 1.1])

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_3d_surfaces.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_node_summary(
        self,
        node: Node,
        time_seconds: NDArray[np.float64],
        flows_over_time: dict,
        save_dir: str,
    ) -> None:
        """Create a summary plot showing all inflows and outflows at a node.

        Args:
            node: The Node to plot.
            time_seconds: 1-D array of time points in seconds.
            flows_over_time: Dictionary mapping link IDs to flow arrays over time.
            save_dir: Directory where plots should be saved.
        """
        # collect all incoming and outgoing links
        incoming_links = node.incoming
        outgoing_links = node.outgoing

        # skip nodes with no meaningful flow to plot
        if not incoming_links and not outgoing_links:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Node {node.id} - Flow Summary", fontsize=14, fontweight="bold")
        actual_duration = time_seconds[-1]

        # plot incoming flows
        if incoming_links:
            max_inflow = 0
            for link in incoming_links:
                link_id = link.id
                flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))

                # for motorway links, take the flow from the last cell (outflow of the link)
                if isinstance(link, MotorwayLink):
                    flow_to_plot = flow[-1, :-1]
                elif len(flow.shape) > 1:
                    flow_to_plot = flow[0, :-1]
                else:
                    flow_to_plot = flow[:-1]

                axes[0].plot(
                    time_seconds[:-1],
                    flow_to_plot,
                    linewidth=1.5,
                    label=f"{type(link).__name__} {link_id}",
                )
                max_inflow = max(
                    max_inflow, np.max(flow_to_plot) if len(flow_to_plot) > 0 else 0
                )

            axes[0].grid(True)
            axes[0].set_xlim([0, actual_duration])
            axes[0].set_ylim([0, max_inflow * 1.1 if max_inflow > 0 else 2500])
            axes[0].set_xlabel("time (s)")
            axes[0].set_ylabel("flow (veh/h)")
            axes[0].set_title("Incoming Flows")
            axes[0].legend(fontsize="small", frameon=False)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No incoming links",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_axis_off()

        # plot outgoing flows
        if outgoing_links:
            max_outflow = 0
            for link in outgoing_links:
                link_id = link.id
                flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))

                # for motorway links, take the flow from the first cell (inflow to the link)
                if isinstance(link, MotorwayLink):
                    flow_to_plot = flow[0, :-1]
                elif len(flow.shape) > 1:
                    flow_to_plot = flow[0, :-1]
                else:
                    flow_to_plot = flow[:-1]

                axes[1].plot(
                    time_seconds[:-1],
                    flow_to_plot,
                    linewidth=1.5,
                    label=f"{type(link).__name__} {link_id}",
                )
                max_outflow = max(
                    max_outflow, np.max(flow_to_plot) if len(flow_to_plot) > 0 else 0
                )

            axes[1].grid(True)
            axes[1].set_xlim([0, actual_duration])
            axes[1].set_ylim([0, max_outflow * 1.1 if max_outflow > 0 else 2500])
            axes[1].set_xlabel("time (s)")
            axes[1].set_ylabel("flow (veh/h)")
            axes[1].set_title("Outgoing Flows")
            axes[1].legend(fontsize="small", frameon=False)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No outgoing links",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"node_{node.id}_summary.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

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

    @staticmethod
    def _density_to_color(
        rho: float, rho_crit: float, rho_jam: float
    ) -> tuple[int, int, int]:
        """Convert density to RGB color for visualization.

        Maps density values to a color gradient:
        - Low density (0 -> rho_crit): bright green -> dark green
        - High density (rho_crit -> rho_jam): dark green -> orange -> dark red

        Args:
            rho: Current density (veh/km/lane)
            rho_crit: Critical density (veh/km/lane)
            rho_jam: Jam density (veh/km/lane)

        Returns:
            RGB tuple with values in range [0, 255]
        """
        # use class-level color constants
        bright_green = Network.COLOR_BRIGHT_GREEN
        dark_green = Network.COLOR_DARK_GREEN
        orange = Network.COLOR_ORANGE
        dark_red = Network.COLOR_DARK_RED

        if rho < rho_crit:
            # interpolate from bright green to dark green
            ratio = rho / rho_crit if rho_crit > 0 else 0.0
            r = int(bright_green[0] + ratio * (dark_green[0] - bright_green[0]))
            g = int(bright_green[1] + ratio * (dark_green[1] - bright_green[1]))
            b = int(bright_green[2] + ratio * (dark_green[2] - bright_green[2]))
        else:
            # interpolate from dark green through orange to dark red
            # transition to orange quickly (first 10% of range) for congestion visibility
            excess = rho - rho_crit
            range_width = rho_jam - rho_crit
            ratio = min(excess / range_width if range_width > 0 else 1.0, 1.0)

            # two-stage interpolation: dark_green -> orange (quick) -> dark_red (gradual)
            if ratio < 0.1:
                # first 10%: dark_green to orange (rapid transition)
                sub_ratio = ratio / 0.1
                r = int(dark_green[0] + sub_ratio * (orange[0] - dark_green[0]))
                g = int(dark_green[1] + sub_ratio * (orange[1] - dark_green[1]))
                b = int(dark_green[2] + sub_ratio * (orange[2] - dark_green[2]))
            else:
                # remaining 90%: orange to dark_red (gradual transition)
                sub_ratio = (ratio - 0.1) / 0.90
                r = int(orange[0] + sub_ratio * (dark_red[0] - orange[0]))
                g = int(orange[1] + sub_ratio * (dark_red[1] - orange[1]))
                b = int(orange[2] + sub_ratio * (dark_red[2] - orange[2]))

        return (r, g, b)

    @staticmethod
    def _interpolate_frames(
        state_history: NDArray[np.float64],
        time_array: NDArray[np.float64],
        subsampling: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linearly interpolate frames for smoother visualization.

        Args:
            state_history: 2-D array (state_size x timesteps)
            time_array: 1-D array of time points
            subsampling: Number of intervals to split each time interval into
                        (subsampling=1 means no interpolation, subsampling=2 splits each interval in half,
                        subsampling=3 splits into thirds, etc.)

        Returns:
            Tuple of (interpolated_time_array, interpolated_state_history)
        """
        if subsampling == 1:
            return time_array, state_history
        elif subsampling < 1:
            raise ValueError(
                "Subsampling must be an integer greater than 1 for interpolation."
            )

        num_timesteps = state_history.shape[1]
        num_interpolated = (num_timesteps - 1) * subsampling + 1

        # create interpolated arrays
        interpolated_time = np.zeros(num_interpolated, dtype=np.float64)
        interpolated_state = np.zeros(
            (state_history.shape[0], num_interpolated), dtype=np.float64
        )

        # fill in interpolated values by splitting each interval into 'subsampling' parts
        for i in range(num_timesteps - 1):
            # original frame at start of interval
            start_idx = i * subsampling
            interpolated_time[start_idx] = time_array[i]
            interpolated_state[:, start_idx] = state_history[:, i]

            # create (subsampling - 1) interpolated frames within the interval
            for j in range(1, subsampling):
                interp_idx = start_idx + j
                alpha = j / subsampling
                interpolated_time[interp_idx] = (
                    time_array[i] * (1 - alpha) + time_array[i + 1] * alpha
                )
                interpolated_state[:, interp_idx] = (
                    state_history[:, i] * (1 - alpha) + state_history[:, i + 1] * alpha
                )

        # last frame at end of final interval
        interpolated_time[-1] = time_array[-1]
        interpolated_state[:, -1] = state_history[:, -1]

        return interpolated_time, interpolated_state

    def _draw_network_state_on_axes(
        self,
        ax: Axes,
        flows: dict,
        densities: dict,
        onramp_queues: dict,
        offramp_queues: dict,
        critical_densities: dict[str, float],
        link_properties: dict[str, dict],
        time_value: float,
        title: str,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> None:
        """Draw network state on a matplotlib axes.

        Helper method that draws nodes, motorway links (colored by density),
        onramps, offramps, and time annotation on the provided axes.

        Args:
            ax: Matplotlib axes to draw on
            flows: Dictionary mapping link IDs to flow arrays
            densities: Dictionary mapping link IDs to density arrays
            onramp_queues: Dictionary mapping onramp IDs to queue lengths
            offramp_queues: Dictionary mapping offramp IDs to queue lengths
            critical_densities: Dictionary mapping link IDs to critical densities
            link_properties: Dictionary mapping link IDs to property dictionaries
            time_value: Current simulation time for annotation
            title: Title for the subplot
            x_bounds: Tuple of (x_min, x_max) for axis limits
            y_bounds: Tuple of (y_min, y_max) for axis limits
        """
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        # draw nodes
        for node in self.list_nodes():
            if node.position is None:
                raise ValueError(
                    f"Node '{node.id}' lacks position information. "
                    "All nodes must have positions set for visualization."
                )
            x, y = node.position
            ax.plot(x, y, "ko", markersize=4, zorder=3)

        # draw motorway links with density coloring
        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if link.origin_node_id is None or link.destination_node_id is None:
                        raise ValueError(
                            f"Motorway link '{link.id}' is missing origin and/or destination node IDs."
                        )

                    upstream_node = self.get_node(link.origin_node_id)
                    downstream_node = self.get_node(link.destination_node_id)

                    if upstream_node is None or downstream_node is None:
                        raise ValueError(
                            f"Motorway link '{link.id}' references non-existent nodes: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    # compute maximum density for this link
                    max_rho = float(np.max(densities[link.id]))
                    rho_crit = critical_densities[link.id]
                    rho_jam = link_properties[link.id]["jam_density"]

                    # get color
                    r, g, b = self._density_to_color(max_rho, rho_crit, rho_jam)
                    color = (r / 255.0, g / 255.0, b / 255.0)

                    # draw link
                    if (
                        upstream_node.position is None
                        or downstream_node.position is None
                    ):
                        raise ValueError(
                            f"Motorway link '{link.id}' has nodes with missing position information: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    x1, y1 = upstream_node.position
                    x2, y2 = downstream_node.position
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=3,
                        solid_capstyle="round",
                        zorder=2,
                    )

        # draw onramps with congestion coloring
        for node in self.list_nodes():
            for link in node.incoming:
                if isinstance(link, Onramp):
                    if link.destination_node_id is None:
                        raise ValueError(
                            f"Onramp link '{link.id}' is missing destination node ID."
                        )

                    downstream_node = self.get_node(link.destination_node_id)
                    if downstream_node and downstream_node.position:
                        x, y = downstream_node.position

                        # determine color based on queue and flow/capacity ratio
                        queue = float(onramp_queues.get(link.id, 0.0))
                        if queue > 0:
                            color = Network.COLOR_CONGESTION_RED
                        else:
                            flow = float(flows.get(link.id, [0.0])[0])
                            capacity = link.Qc
                            ratio = min(flow / capacity if capacity > 0 else 0.0, 1.0)
                            r = 0.6 * (1 - ratio)
                            g = 1.0 - 0.5 * ratio
                            b = 0.6 * (1 - ratio)
                            color = (r, g, b)

                        # draw short line offset from node
                        ax.plot(
                            [x - 0.1, x],
                            [y + 0.1, y],
                            color=color,
                            linewidth=2,
                            linestyle="--",
                            zorder=1,
                        )

        # draw offramps with congestion coloring
        for node in self.list_nodes():
            for link in node.outgoing:
                if isinstance(link, Offramp):
                    if link.origin_node_id is None:
                        raise ValueError(
                            f"Offramp link '{link.id}' is missing origin node ID."
                        )

                    upstream_node = self.get_node(link.origin_node_id)
                    if upstream_node and upstream_node.position:
                        x, y = upstream_node.position

                        # determine color based on queue and flow/capacity ratio
                        queue = float(offramp_queues.get(link.id, 0.0))
                        if queue > 0:
                            color = Network.COLOR_CONGESTION_RED
                        else:
                            flow = float(flows.get(link.id, [0.0])[0])
                            capacity = link.Qc
                            ratio = min(flow / capacity if capacity > 0 else 0.0, 1.0)
                            r = 0.6 * (1 - ratio)
                            g = 1.0 - 0.5 * ratio
                            b = 0.6 * (1 - ratio)
                            color = (r, g, b)

                        # draw short line offset from node
                        ax.plot(
                            [x, x + 0.1],
                            [y, y + 0.1],
                            color=color,
                            linewidth=2,
                            linestyle="--",
                            zorder=1,
                        )

        # add time annotation
        ax.text(
            0.02,
            0.98,
            f"Time: {time_value:.2f} h",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # set axis limits and styling
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)

    @staticmethod
    def _fig_to_frame(fig: Figure, target_width: int, target_height: int) -> Any:
        """Convert matplotlib figure to OpenCV-compatible BGR frame.

        Helper method that converts a matplotlib figure to a numpy array
        in BGR format suitable for OpenCV video writing.

        Args:
            fig: Matplotlib figure to convert
            target_width: Target frame width in pixels
            target_height: Target frame height in pixels

        Returns:
            BGR image as numpy array (uint8)
        """
        # convert figure to numpy array using backend-agnostic method
        fig.canvas.draw()
        buf, (width, height) = fig.canvas.print_to_buffer()  # type: ignore
        # convert buffer to numpy array (RGBA format)
        img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        # extract RGB channels (drop alpha)
        img_rgb = img_rgba[:, :, :3]

        # resize to target dimensions if needed
        if (height, width) != (target_height, target_width):
            img_rgb = cv2.resize(img_rgb, (target_width, target_height))

        # convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return img_bgr

    def visualize_simulation(
        self,
        results_filepath: str,
        output_filepath: str,
        fps: int = 25,
        subsampling: int = 1,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 300,
    ) -> None:
        """Generate a video visualization of simulation results.

        Creates an AVI video showing the network topology with links colored
        by density. Color scheme transitions from bright green (low density)
        through dark green (critical density) to orange and dark red (jam density).

        Args:
            results_filepath: Path to simulation results JSON file
            output_filepath: Path for output video file (.avi)
            fps: Frames per second for output video
            subsampling: Number of intervals to split each time interval into for smoother animation
                        (1=no interpolation, 2=split intervals in half, 3=split into thirds, etc.)
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for rendering

        Raises:
            ValueError: If nodes lack position information or metadata is missing
            ImportError: If opencv-python is not installed
        """
        # load simulation results
        time_array, state_history, _, metadata = self.load_simulation_results_json(
            filepath=results_filepath, network=self
        )

        if metadata is None:
            raise ValueError(
                "Simulation results file lacks metadata. "
                "Only files with metadata can be visualized."
            )

        critical_densities = metadata["critical_densities"]
        link_properties = metadata["link_properties"]

        # apply frame interpolation if requested
        if subsampling > 1:
            time_array, state_history = self._interpolate_frames(
                state_history, time_array, subsampling
            )

        num_frames = state_history.shape[1]

        # compute plot bounds from node positions
        positions = []
        for node in self.list_nodes():
            if node.position is None:
                raise ValueError(
                    f"Node '{node.id}' lacks position information. "
                    "All nodes must have positions set for visualization."
                )
            positions.append(node.position)

        positions_arr = np.array(positions)
        x_min, y_min = positions_arr.min(axis=0)
        x_max, y_max = positions_arr.max(axis=0)

        # add padding for ramps and visual clarity
        x_padding = 0.15 * (x_max - x_min) if x_max > x_min else 1.0
        y_padding = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # initialize video writer
        frame_width = int(figsize[0] * dpi)
        frame_height = int(figsize[1] * dpi)
        video_writer = cv2.VideoWriter(
            filename=output_filepath,
            fourcc=cv2.VideoWriter.fourcc(*"MJPG"),
            fps=fps,
            frameSize=(frame_width, frame_height),
        )

        try:
            # generate frames
            for t in tqdm(range(num_frames)):
                # create figure
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

                # extract current state
                flows, densities, _, _, onramp_queues, offramp_queues = (
                    self.state_vec_to_network_dict(state_history[:, t])
                )

                # draw network state on axes using helper
                self._draw_network_state_on_axes(
                    ax=ax,
                    flows=flows,
                    densities=densities,
                    onramp_queues=onramp_queues,
                    offramp_queues=offramp_queues,
                    critical_densities=critical_densities,
                    link_properties=link_properties,
                    time_value=time_array[t],
                    title="Traffic Flow Simulation",
                    x_bounds=(x_min, x_max),
                    y_bounds=(y_min, y_max),
                )

                # convert figure to frame and write
                img_bgr = self._fig_to_frame(fig, frame_width, frame_height)
                video_writer.write(img_bgr)

                # close figure to free memory
                plt.close(fig)

        finally:
            # release video writer
            video_writer.release()

    def visualize_simulation_comparison(
        self,
        result_filepaths: list[str],
        labels: list[str],
        output_filepath: str,
        fps: int = 25,
        subsampling: int = 1,
        figsize: tuple[float, float] = (16, 12),
        dpi: int = 300,
        layout: tuple[int, int] | None = None,
    ) -> None:
        """Generate a comparison video showing multiple simulations side-by-side.

        Creates an AVI video with multiple subplots showing different simulation
        results synchronized in time. This is useful for comparing calibration
        results or different scenarios.

        Args:
            result_filepaths: List of paths to simulation results JSON files
            labels: List of labels for each simulation (same length as result_filepaths)
            output_filepath: Path for output video file (.avi)
            fps: Frames per second for output video
            subsampling: Number of intervals to split each time interval into for smoother animation
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for rendering
            layout: Optional tuple (nrows, ncols) for subplot layout. If None, auto-computed
                   to create an approximately square grid.

        Raises:
            ValueError: If inputs are invalid or simulations have different lengths
            ImportError: If opencv-python is not installed
        """
        if len(result_filepaths) != len(labels):
            raise ValueError(
                f"Number of result files ({len(result_filepaths)}) must match "
                f"number of labels ({len(labels)})"
            )

        if len(result_filepaths) == 0:
            raise ValueError("Must provide at least one result file")

        # load all simulation results
        simulations = []
        for filepath in result_filepaths:
            time_array, state_history, _, metadata = self.load_simulation_results_json(
                filepath=filepath, network=self
            )

            if metadata is None:
                raise ValueError(
                    f"Simulation results file '{filepath}' lacks metadata. "
                    "Only files with metadata can be visualized."
                )

            simulations.append(
                {
                    "time_array": time_array,
                    "state_history": state_history,
                    "metadata": metadata,
                }
            )

        # validate all simulations have same length
        num_timesteps = simulations[0]["time_array"].shape[0]
        for i, sim in enumerate(simulations[1:], start=1):
            if sim["time_array"].shape[0] != num_timesteps:
                raise ValueError(
                    f"Simulation {i+1} has {sim['time_array'].shape[0]} timesteps, "
                    f"but simulation 1 has {num_timesteps}. All simulations must have "
                    "the same number of timesteps for comparison."
                )

        # apply frame interpolation if requested
        for sim in simulations:
            if subsampling > 1:
                sim["time_array"], sim["state_history"] = self._interpolate_frames(
                    sim["state_history"], sim["time_array"], subsampling
                )

        num_frames = simulations[0]["state_history"].shape[1]

        # determine subplot layout
        n_sims = len(simulations)
        if layout is None:
            # auto-compute roughly square layout
            ncols = int(np.ceil(np.sqrt(n_sims)))
            nrows = int(np.ceil(n_sims / ncols))
        else:
            nrows, ncols = layout
            if nrows * ncols < n_sims:
                raise ValueError(
                    f"Layout {layout} has {nrows * ncols} subplots but "
                    f"{n_sims} simulations provided"
                )

        # compute plot bounds from node positions (shared across all subplots)
        positions = []
        for node in self.list_nodes():
            if node.position is None:
                raise ValueError(
                    f"Node '{node.id}' lacks position information. "
                    "All nodes must have positions set for visualization."
                )
            positions.append(node.position)

        positions_arr = np.array(positions)
        x_min, y_min = positions_arr.min(axis=0)
        x_max, y_max = positions_arr.max(axis=0)

        # add padding for ramps and visual clarity
        x_padding = 0.15 * (x_max - x_min) if x_max > x_min else 1.0
        y_padding = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # initialize video writer
        frame_width = int(figsize[0] * dpi)
        frame_height = int(figsize[1] * dpi)
        video_writer = cv2.VideoWriter(
            filename=output_filepath,
            fourcc=cv2.VideoWriter.fourcc(*"MJPG"),
            fps=fps,
            frameSize=(frame_width, frame_height),
        )

        try:
            # generate frames
            for t in tqdm(range(num_frames)):
                # create figure with subplots
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

                # ensure axes is always a flat array for consistent indexing
                if n_sims == 1:
                    axes = np.array([axes])
                else:
                    axes = np.array(axes).flatten()

                # draw each simulation on its subplot
                for i, (sim, label) in enumerate(zip(simulations, labels)):
                    ax = axes[i]

                    # extract current state
                    flows, densities, _, _, onramp_queues, offramp_queues = (
                        self.state_vec_to_network_dict(sim["state_history"][:, t])
                    )

                    # draw network state on axes using helper
                    self._draw_network_state_on_axes(
                        ax=ax,
                        flows=flows,
                        densities=densities,
                        onramp_queues=onramp_queues,
                        offramp_queues=offramp_queues,
                        critical_densities=sim["metadata"]["critical_densities"],
                        link_properties=sim["metadata"]["link_properties"],
                        time_value=sim["time_array"][t],
                        title=label,
                        x_bounds=(x_min, x_max),
                        y_bounds=(y_min, y_max),
                    )

                # hide unused subplots
                for i in range(n_sims, len(axes)):
                    axes[i].axis("off")

                plt.tight_layout()

                # convert figure to frame and write
                img_bgr = self._fig_to_frame(fig, frame_width, frame_height)
                video_writer.write(img_bgr)

                # close figure to free memory
                plt.close(fig)

        finally:
            # release video writer
            video_writer.release()

    # endregion
