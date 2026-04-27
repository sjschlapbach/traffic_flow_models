import casadi
from typing import TYPE_CHECKING, Tuple, Union

from .helpers import store_and_forward_update, update_queue, compute_node_outflows
from traffic_flow_models.network import (
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
)

if TYPE_CHECKING:
    from traffic_flow_models.network import Network, Node


class CTM:
    """
    Cell Transmission Model (CTM) implementation.

    Provides a first-order CTM update for motorway links together with
    store-and-forward handling for origins, onramps and offramps. The
    class exposes helpers for fundamental diagram quantities (critical
    density, backward wave speed) and node/link update routines used to
    assemble a full-network CasADi `Function` via
    `network_update_function`.

    Notes:
        - Motorway links are advanced using a first-order CTM-style
            density/flow update in `_update_motorway_link`.
        - Origins, onramps and offramps are represented with
            store-and-forward logic.
        - Split normalization, node-level supply checks and proportional
            flow reductions are handled in the node update logic.

    The implementation assumes network state/disturbance dictionaries
    are provided by the `Network` helpers and uses CasADi `SX` for the
    symbolic formulation.
    """

    def __init__(self):
        """Create an empty CTM model instance."""
        return

    # ! Model parameter calibration support (not applicable for CTM)
    # region
    def get_default_calibration_params(self):
        """CTM has no calibratable parameters.

        Raises:
            NotImplementedError: CTM does not support parameter calibration.
        """
        raise NotImplementedError(
            "CTM model does not have calibratable parameters. "
            "All model characteristics are determined by link properties "
            "(capacity, jam density, free-flow speed)."
        )

    def get_calibration_bounds(self, network, **kwargs):
        """CTM has no calibratable parameters.

        Raises:
            NotImplementedError: CTM does not support parameter calibration.
        """
        raise NotImplementedError("CTM model does not have calibratable parameters.")

    def prepare_calibration_params(self, params, network, **kwargs):
        """CTM has no calibratable parameters.

        Raises:
            NotImplementedError: CTM does not support parameter calibration.
        """
        raise NotImplementedError("CTM model does not have calibratable parameters.")

    def parse_calibration_params(self, param_vec, network, **kwargs):
        """CTM has no calibratable parameters.

        Raises:
            NotImplementedError: CTM does not support parameter calibration.
        """
        raise NotImplementedError("CTM model does not have calibratable parameters.")

    def prepare_system_params(self, param_vec, network, **kwargs):
        """CTM has no calibratable parameters.

        Raises:
            NotImplementedError: CTM does not support parameter calibration.
        """
        raise NotImplementedError("CTM model does not have calibratable parameters.")

    def get_calibration_param_names(
        self, network: "Network", model_options: dict | None = None
    ) -> list[str]:
        """CTM has no calibratable parameters.

        Raises:
            NotImplementedError: CTM does not support parameter calibration.
        """
        raise NotImplementedError(
            "CTM model does not have calibratable parameters or parameter names."
        )

    # endregion

    # ! Fundamental diagram helper functions
    # region
    def critical_density(
        self,
        lane_capacity: float,
        free_flow_speed: float,
    ) -> float:
        """
        Compute the critical density for a CTM cell.

        The critical density is the density at which flow is maximized and is
        computed as::
            rho_crit = lane_capacity / free_flow_speed

        This represents the transition point between free-flow and congested
        conditions in the fundamental diagram.

        Args:
            lane_capacity: Lane capacity (vehicles per time per lane).
            free_flow_speed: Free-flow speed (length per time).

        Returns:
            Critical density (vehicles per length per lane).
        """
        return lane_capacity / free_flow_speed

    def backward_wave_speed(
        self,
        capacity: float,
        lane_capacity: float,
        jam_density: float,
        free_flow_speed: float,
    ) -> float:
        """
        Compute the backward (congestion) wave speed for a CTM cell.

        The backward wave speed describes how congestion propagates upstream
        and is computed as::
            w = capacity / (jam_density - rho_crit)

        where rho_crit is the critical density. This is the slope of the
        congestion branch in the fundamental diagram.

        Args:
            capacity: Cell capacity (vehicles per time).
            lane_capacity: Capacity per lane (vehicles per time per lane).
            jam_density: Jam density (vehicles per length per lane).
            free_flow_speed: Free-flow speed (length per time).

        Returns:
            Backward wave speed (length per time).
        """

        rho_cr = self.critical_density(
            lane_capacity=lane_capacity, free_flow_speed=free_flow_speed
        )
        return capacity / (jam_density - rho_cr)

    # ! Network update helper functions
    # region
    def _get_node_outflow_link(
        self,
        network: "Network",
        splits: dict[str, dict[str, casadi.SX]],
        link: Union[MotorwayLink, Onramp, Offramp],
        flows: dict[str, casadi.SX],
    ) -> casadi.SX:
        """Compute the portion of a node's outflow routed into a motorway link.

        Using the provided `node` and its outgoing split ratios (`node_splits`),
        this helper sums the last-cell flows of all incoming links to the node
        and returns the fraction directed into `link` according to the split
        ratio for `link.id`.

        Args:
            network (Network): The network containing the node and links.
            splits (dict[str, dict[str, casadi.SX]]): Mapping from node id to
                mapping of outgoing link id to split ratio (CasADi SX).
            link (Union[MotorwayLink, Onramp, Offramp]): The downstream link
                receiving a portion of the node outflow.
            flows (dict[str, casadi.SX]): Mapping from link id to the
                current-step flow vector (CasADi SX) for that link.
            node_splits (dict[str, casadi.SX]): Mapping of outgoing link id
                to the split ratio used at `node` (CasADi SX).

        Returns:
            casadi.SX: Flow portion directed into `link` (vehicles / time).

        Raises:
            ValueError: If no split ratio is defined for `link.id` in
                `node_splits`.
        """
        if link.origin_node_id is None:
            raise ValueError(
                f"Motorway link {link.id} does not have a well-defined origin node."
            )

        # compute the normalized node splits for the upstream node
        upstream_node = network.get_node(link.origin_node_id)
        if upstream_node is None:
            raise ValueError(
                f"Origin node {link.origin_node_id} of motorway link {link.id} not found in network."
            )
        node_splits = self._compute_normalized_splits(
            node=upstream_node,
            node_splits=splits[link.origin_node_id],
        )

        # compute the outflow from the upstream node into this link
        inflow_sum: casadi.SX = casadi.sum(
            casadi.vertcat(*[flows[inc.id][-1] for inc in upstream_node.incoming])
        )
        node_split_link = node_splits[link.id]
        if node_split_link is None:
            raise ValueError(
                f"No split ratio defined for outgoing link {link.id} at node {upstream_node.id}"
            )
        node_outflow_link = (
            node_split_link * inflow_sum
        )  # = q_0(k) for this motorway link

        return node_outflow_link

    def _compute_next_density(
        self,
        density: casadi.SX,
        inflow: casadi.SX,
        outflow: casadi.SX,
        cell_length: float,
        link_lanes: int,
        dt: float,
    ) -> casadi.SX:
        """
        Compute the next-step density for a single cell using conservation.

        Applies the first-order CTM conservation update:

            rho_next = rho + dt / (cell_length * link_lanes) * (inflow - outflow)

        Args:
            density (casadi.SX): Current cell density (vehicles per length per lane).
            inflow (casadi.SX): Flow entering the cell (vehicles per time).
            outflow (casadi.SX): Flow leaving the cell (vehicles per time).
            cell_length (float): Length of the cell (same length units as densities).
            link_lanes (int): Number of lanes on the link.
            dt (float): Simulation timestep (time units consistent with flows).

        Returns:
            casadi.SX: The updated cell density (vehicles per length per lane).
        """
        next_density = density + dt / (cell_length * link_lanes) * (inflow - outflow)
        return next_density

    def _update_motorway_link(
        self,
        link: MotorwayLink,
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        upstream_node_outflow_link: casadi.SX,
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX, casadi.SX]:
        """Update densities, speeds and flows for a motorway link one step.

        Using a first-order CTM-style update, compute the next-step cell
        densities, speeds and internal cell-to-cell flows for the provided
        `link`. The update uses the provided `upstream_node_outflow_link` as
        the inflow into the first cell and the existing `flows` and
        `densities` dictionaries for internal exchanges. Supply-limited
        outflow from the last cell is intentionally left as `inf` here and
        handled at the node level.

        Args:
            link (MotorwayLink): The motorway link to update.
            flows (dict[str, casadi.SX]): Current-step flows for all links
                (mapping link id -> flow vector, CasADi SX).
            densities (dict[str, casadi.SX]): Current-step densities for all
                links (mapping link id -> density vector, CasADi SX).
            upstream_node_outflow_link (casadi.SX): Computed inflow from the
                upstream node into the first cell of `link` (CasADi SX).
            dt (float): Simulation timestep.

        Returns:
            Tuple[casadi.SX, casadi.SX, casadi.SX]: A tuple containing the
                next-step densities vector, speeds vector and flows vector
                for `link` (all CasADi SX).
        """
        link_flows = flows[link.id]
        link_densities = densities[link.id]

        next_densities_list = casadi.SX(len(link), 1)
        next_speeds_list = casadi.SX(len(link), 1)
        next_flows_list = casadi.SX(len(link), 1)

        for i, cell in link.enumerate_cells():
            # compute the new density in the cell based on the flows at the previous timestep
            # -> onramp and offramp flows do not need to be considered -> handled through nodes
            if i == 0:
                next_densities_list[i] = self._compute_next_density(
                    density=link_densities[i],
                    inflow=upstream_node_outflow_link,
                    outflow=link_flows[i],
                    cell_length=cell.length,
                    link_lanes=link.lanes,
                    dt=dt,
                )
            else:
                next_densities_list[i] = self._compute_next_density(
                    density=link_densities[i],
                    inflow=link_flows[i - 1],
                    outflow=link_flows[i],
                    cell_length=cell.length,
                    link_lanes=link.lanes,
                    dt=dt,
                )

        for j, cell in link.enumerate_cells():
            # compute the new flows based on the updated density (first-order model)
            q_demand = link.vf * next_densities_list[j] * link.lanes
            q_supply = (
                self.backward_wave_speed(
                    capacity=link.Qc,
                    lane_capacity=link.Qc_lane,
                    jam_density=link.rho_jam,
                    free_flow_speed=link.vf,
                )
                * (link.rho_jam - next_densities_list[j + 1])
                if j < len(link) - 1
                else casadi.inf  # no supply restriction for last cell at this point -> will be introduced in terms of proportional flow reduction
            )

            next_flows_list[j] = casadi.fmin(
                casadi.fmin(casadi.SX(link.Qc), q_demand),
                q_supply,
            )

            # compute the updated speed based on the updated density and flow
            next_speeds_list[j] = casadi.if_else(
                next_densities_list[j] > 0,
                next_flows_list[j] / (link.lanes * next_densities_list[j]),
                link.vf,
            )

        return next_densities_list, next_speeds_list, next_flows_list

    def _compute_normalized_splits(
        self, node: "Node", node_splits: dict[str, casadi.SX]
    ) -> dict[str, casadi.SX]:
        """Return normalized split ratios for a node's outgoing links.

        The function validates that split ratios are defined for all
        outgoing links and normalizes them so they sum to one. The returned
        mapping gives the normalized split for each outgoing link id as a
        CasADi `SX` expression.

        Args:
            node (Node): Node whose outgoing split ratios are to be
                normalized.
            node_splits (dict[str, casadi.SX]): Mapping of outgoing link id to
                (possibly unnormalized) split ratio (CasADi SX).

        Returns:
            dict[str, casadi.SX]: Mapping from outgoing link id to the
                normalized split ratio (CasADi SX).

        Raises:
            ValueError: If any outgoing split ratio for `node` is `None`.
        """
        if any([node_splits[out.id] is None for out in node.outgoing]):
            raise ValueError(
                f"Not all split ratios defined for outgoing links at node {node.id}."
            )

        splits_sum = casadi.sum(
            casadi.vertcat(*[node_splits[outgoing.id] for outgoing in node.outgoing])
        )
        normalized_node_splits: dict[str, casadi.SX] = {
            out.id: casadi.if_else(
                splits_sum > 0,
                node_splits[out.id] / splits_sum,
                casadi.SX(1 / len(node.outgoing)),
            )
            for out in node.outgoing
        }

        return normalized_node_splits

    def _compute_node_maximum_outflows(
        self,
        network: "Network",
        node: "Node",
        splits: dict[str, dict[str, casadi.SX]],
        densities: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        node_splits: dict[str, casadi.SX],
        flow_boundary_conditions: dict[str, casadi.SX],
        dt: float,
    ) -> casadi.SX:
        """Compute the maximum outflow the node can support given supplies.

        For each outgoing link, compute the supply-limited flow that the
        outgoing link can accept and convert it to a node-level limit using
        the normalized split ratios. The minimum across outgoing links
        determines the maximum node outflow (i.e., the most restrictive
        downstream supply). For destinations the provided flow boundary
        conditions are enforced; offramps are treated as store-and-forward
        links limited by their capacity; and motorway links use the CTM supply
        expression based on jam density and backward wave speed.

        Args:
            network (Network): The network containing the node and links.
            node (Node): Node for which the maximum supported outflow is
                computed.
            splits (dict[str, dict[str, casadi.SX]]): Mapping from node id to
                mapping of outgoing link id to split ratio (CasADi SX).
            densities (dict[str, casadi.SX]): Current-step densities for
                links (mapping link id -> density vector, CasADi SX).
            flows (dict[str, casadi.SX]): Current-step flows for links
                (mapping link id -> flow vector, CasADi SX).
            node_splits (dict[str, casadi.SX]): Normalized split
                ratios for the node's outgoing links (mapping link id ->
                CasADi SX).
            flow_boundary_conditions (dict[str, casadi.SX]): Mapping from
                destination id to flow boundary condition (vehicles / time)
            dt (float): Time step size.

        Returns:
            casadi.SX: The maximum supported outflow for the node (CasADi
                SX). This value is the upper bound on the sum of incoming
                flows the node can accept without causing spillback.

        Notes:
            The implementation currently uses the previous-step density of
            an outgoing motorway link's first cell when computing its
            supply; this may introduce a causality/consistency issue if
            next-step densities were required instead.
        """
        maximum_supported_node_outflow = casadi.SX(casadi.inf)
        for out in node.outgoing:
            if isinstance(out, Onramp):
                # flows from origins through nodes to onramps should not be supply-restricted
                # (onramps are assumed to have an infinite supply of space and a virtual queue)
                # -> network validation makes sure that the corresponding node only has one incoming
                #    link, which is the origin forwarding the demand function -> onramp demand
                continue

            elif isinstance(out, Destination):
                # destinations only limit the node outflow through the given flow boundary condition
                maximum_supported_node_outflow = casadi.fmin(
                    maximum_supported_node_outflow,
                    flow_boundary_conditions[out.id] / node_splits[out.id],
                )

            elif isinstance(out, Offramp):
                # destinations are modeled as store-and-forward links with a virtual queue
                # -> only the offramp capacity becomes a limiting factor for potential spillback
                # any congestion caused by downstream boundary conditions will only grow the off-ramp queue
                # (consistent with the definition for diverging flows in Daganzo, 1993)
                maximum_supported_node_outflow = casadi.fmin(
                    maximum_supported_node_outflow,
                    out.Qc / node_splits[out.id],
                )

            elif isinstance(out, MotorwayLink):
                # compute the outflow from the currently considered node into the considered
                # motorway link as a basis for the computation of the first cell next-step
                # density value (to be used as a supply restriction)
                node_outflow_link = self._get_node_outflow_link(
                    network=network,
                    splits=splits,
                    link=out,
                    flows=flows,
                )

                # compute the next-step density of the first cell of the outgoing link
                first_cell = out.get_cell(0)
                first_cell_next_density = self._compute_next_density(
                    density=densities[out.id][0],
                    inflow=node_outflow_link,
                    outflow=flows[out.id][0],
                    cell_length=first_cell.length,
                    link_lanes=out.lanes,
                    dt=dt,
                )

                # compute the outflow supply limit imposed through the outgoing motorway link
                maximum_supported_node_outflow = casadi.fmin(
                    maximum_supported_node_outflow,
                    (
                        self.backward_wave_speed(
                            capacity=out.Qc,
                            lane_capacity=out.Qc_lane,
                            jam_density=out.rho_jam,
                            free_flow_speed=out.vf,
                        )
                        * (out.rho_jam - first_cell_next_density)
                    )
                    / node_splits[out.id],
                )

        return maximum_supported_node_outflow

    def _compute_offramp_outflows(
        self,
        offramp: Offramp,
        node_outflow: casadi.SX,
        offramp_queues: dict[str, casadi.SX],
        density_boundary_condition: casadi.SX,
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX]:
        """Compute an offramp's outflow and update its store-and-forward queue.

        Offramps are modelled as store-and-forward links with finite
        capacity. This routine computes the offramp demand by combining the
        node outflow portion intended for the offramp (``node_outflow``) and
        the current virtual queue on the offramp. The actual outflow and the
        updated queue are obtained by calling ``store_and_forward_update``
        with the offramp's capacity, jam density, a computed backward wave
        speed, and the downstream (destination) boundary density.

        Args:
            offramp (Offramp): The offramp link to update.
            node_outflow (casadi.SX): Desired flow from the upstream node into
                the offramp (vehicles / time) as a CasADi expression.
            offramp_queues (dict[str, casadi.SX]): Current queue lengths on
                offramps indexed by link id (vehicles, CasADi SX).
            density_boundary_condition (casadi.SX): Downstream virtual density
                constraint of the connected destination (vehicles / length / lane)
            dt (float): Simulation timestep (time units consistent with flows).

        Returns:
            Tuple[casadi.SX, casadi.SX]: ``(next_outflow, next_queue)`` where
            ``next_outflow`` is the computed offramp outflow into the
            connected destination (vehicles / time) and ``next_queue`` is the
            updated queue length on the offramp (vehicles), both as CasADi SX
            expressions.

        Raises:
            ValueError: If the ``offramp`` does not have an associated
                ``destination`` link (required for the downstream density).
        """
        # update the offramp flow and queue based on the store-and-forward model
        next_outflow, next_queue = store_and_forward_update(
            capacity=offramp.Qc,
            jam_density=offramp.rho_jam,
            backward_wave_speed=self.backward_wave_speed(
                capacity=offramp.Qc,
                lane_capacity=offramp.Qc_lane,
                jam_density=offramp.rho_jam,
                free_flow_speed=offramp.vf,
            ),
            density=density_boundary_condition,
            demand=node_outflow,
            queue=offramp_queues[offramp.id],
            dt=dt,
        )

        return next_outflow, next_queue

    # endregion

    def network_update_function(
        self,
        network: "Network",
        num_flows: int,
        num_densities: int,
        num_speeds: int,
        num_origins: int,
        num_onramps: int,
        num_offramps: int,
        num_splits: int,
        num_destinations: int,
        dt: float,
    ) -> casadi.Function:
        """Build a CasADi function implementing one CTM network step.

        The returned CasADi `Function` (named ``ctm_network_step``) maps
        the symbolic model parameter vector, the current state vector ``x``
        and the disturbance vector ``d`` to the next-step state vector
        ``x_next`` according to the CTM dynamics combined with
        store-and-forward updates for origins, onramps and offramps.

        State and disturbance vector layouts follow
        `Network.state_vec_to_network_dict` and
        `Network.disturbance_vec_to_network_dict`. The disturbance vector
        contains origin demands, split ratios and boundary condition entries
        in the ordering expected by the network helpers.

        Args:
            network (Network): Network object containing links, nodes and
                helper methods to convert between vectors and dictionaries.
            num_flows (int): Length of the flow portion of the state vector.
            num_densities (int): Length of the density portion of the state
                vector.
            num_speeds (int): Length of the speed portion of the state vector.
            num_origins (int): Number of origin links (state/disturbance size).
            num_onramps (int): Number of onramp links (state/disturbance size).
            num_offramps (int): Number of offramp links (state/disturbance size).
            num_splits (int): Number of split-ratio disturbance entries.
            num_destinations (int): Number of boundary-condition disturbance entries.
            dt (float): Simulation timestep.

        Returns:
            casadi.Function: A CasADi function `f(params, x, d) -> x_next`
            implementing the network update for one timestep. The first
            argument to the function is the symbolic model parameter vector
            produced by `set_up_symbolic_model_params`.
        """

        # ! Set up variables for state update and cast types to be correct
        # set up state and disturbance vectors
        # state: flows, densities, speeds, origin, onramp
        # disturbances: origin_demands, offramp_split_ratios
        # CasADi type stubs are incorrect - sym() does accept string as first arg
        x = casadi.SX.sym(  # type: ignore
            "x",  # type: ignore
            num_flows
            + num_densities
            + num_speeds
            + num_origins
            + num_onramps
            + num_offramps,  # type: ignore
            1,  # type: ignore
        )
        d = casadi.SX.sym("d", num_origins + num_splits + 2 * num_destinations, 1)  # type: ignore

        # split up the state and disturbance vectors to obtain a dictionary for
        # efficient access of the relevant quantities during the state update
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            network.state_vec_to_network_dict(x=x)
        )
        (
            origin_demands,
            splits,
            flow_boundary_conditions,
            density_boundary_conditions,
        ) = network.disturbance_vec_to_network_dict(d=d)

        # typecast values of the dictionaries to casadi SX for symbolic computation
        flows = {k: casadi.SX(v) for k, v in flows.items()}
        densities = {k: casadi.SX(v) for k, v in densities.items()}
        speeds = {k: casadi.SX(v) for k, v in speeds.items()}
        origin_queues = {k: casadi.SX(v) for k, v in origin_queues.items()}
        onramp_queues = {k: casadi.SX(v) for k, v in onramp_queues.items()}
        offramp_queues = {k: casadi.SX(v) for k, v in offramp_queues.items()}
        origin_demands = {k: casadi.SX(v) for k, v in origin_demands.items()}
        splits = {
            k: {kk: casadi.SX(vv) for kk, vv in v.items()} for k, v in splits.items()
        }
        flow_boundary_conditions = {
            k: casadi.SX(v) for k, v in flow_boundary_conditions.items()
        }
        density_boundary_conditions = {
            k: casadi.SX(v) for k, v in density_boundary_conditions.items()
        }

        # initialize next-step state dictionaries
        next_flows: dict[str, casadi.SX] = {}
        next_densities: dict[str, casadi.SX] = {}
        next_speeds: dict[str, casadi.SX] = {}
        next_origin_queues: dict[str, casadi.SX] = {}
        next_onramp_queues: dict[str, casadi.SX] = {}
        next_offramp_queues: dict[str, casadi.SX] = {}

        # formulate the individual update equations for each node and update the overall system equation and the next step state
        # iterate through all nodes and update the corrresponding quantities of incoming and outgoing links
        for node in network.list_nodes():
            # ! 1) iterate through all incoming links (origins, onramps, and motorway links) and update the flows
            # if the desired flow at the downstream node is higher than the maximum outgoing flow of the node, reduce
            # the flows proportionally to their desired value to obtain the actual flow values for the density updates
            total_node_inflow = casadi.SX(0)
            for inc in node.incoming:
                if isinstance(inc, MotorwayLink):
                    # compute outflow of the upstream node directed into this motorway link
                    upstream_node_outflow_link = self._get_node_outflow_link(
                        network=network,
                        splits=splits,
                        link=inc,
                        flows=flows,
                    )

                    # step through the cells of the link and update the flows, densities and speeds accordingly
                    # for the last cell, currently ignore the downstream supply of space restriction
                    # -> will be handled separately after the loop once potential flow limits have been identified
                    next_densities_list, next_speeds_list, next_flows_list = (
                        self._update_motorway_link(
                            link=inc,
                            flows=flows,
                            densities=densities,
                            upstream_node_outflow_link=upstream_node_outflow_link,
                            dt=dt,
                        )
                    )

                    # add the computed next-step link outflow to the total node inflow
                    total_node_inflow += next_flows_list[-1]

                    # store the computed next-step densities, speeds and flows for the link
                    # in case of an overflow of the node, the last-cell values will be updated again
                    next_densities[inc.id] = next_densities_list
                    next_speeds[inc.id] = next_speeds_list
                    next_flows[inc.id] = next_flows_list

                elif isinstance(inc, Origin):
                    # since downstream supply of space restrictions at the node will be considered
                    # in the next step, the origin flow directly equals the demand + queue demand
                    next_flows[inc.id] = origin_demands[inc.id] + (
                        origin_queues[inc.id] / dt
                    )
                    total_node_inflow += next_flows[inc.id]

                elif isinstance(inc, Onramp):
                    # compute outflow of the upstream node directed into this motorway link
                    upstream_node_outflow_link = self._get_node_outflow_link(
                        network=network,
                        splits=splits,
                        link=inc,
                        flows=flows,
                    )

                    # if a controller is defined for the on-ramp, compute the maximum flow according to ramp
                    # metering (i.e. the controller output) and include it in the minimum
                    if inc.controller is not None:
                        r_k = inc.controller.compute_regulated_flow(
                            onramp_queues=onramp_queues,
                            flows=flows,
                            densities=densities,
                        )
                    else:
                        r_k = casadi.SX(casadi.inf)

                    # onramps are also modeled as store-and-forward links (as origins), but additionally
                    # have a finite capacity, which needs to be taken into account when computing the
                    # desired flow / flow on the onramp without considering downstream supply of space restrictions
                    # -> since onramps are indirectly connected to their demand through a node & origin, the
                    #    inflow from the upstream node (unconstrained) represents the demand
                    next_flows[inc.id] = casadi.fmin(
                        casadi.fmin(inc.Qc, r_k),
                        upstream_node_outflow_link + (onramp_queues[inc.id] / dt),
                    )
                    total_node_inflow += next_flows[inc.id]

                elif isinstance(inc, Offramp):
                    # compute outflow of the upstream node directed into this motorway link
                    upstream_node_outflow_link = self._get_node_outflow_link(
                        network=network,
                        splits=splits,
                        link=inc,
                        flows=flows,
                    )

                    # offramps are also modeled as store-and-forward links (as origins), but additionally
                    # have a finite capacity, which needs to be taken into account when computing the
                    # desired flow / flow on the offramp without considering downstream supply of space restrictions
                    next_flows[inc.id] = casadi.fmin(
                        inc.Qc,
                        upstream_node_outflow_link + (offramp_queues[inc.id] / dt),
                    )
                    total_node_inflow += next_flows[inc.id]

                else:
                    raise TypeError(f"Unknown incoming link type: {type(inc)}")

            # ! 2) Normalize the split ratios and compute the maximum outflow of the node
            # compute the maximum outflow of the link according to the supply of space equation for each outgoing link
            # we assume that the most congested outgoing link determines the maximum outflow of the node
            # -> since CTM focusses on accumulations, this would correspond to a spillback scenario across the node
            # (assuming that the split ratios are not affected by changing traffic conditions on individual links)
            normalized_node_splits = self._compute_normalized_splits(
                node=node, node_splits=splits[node.id]
            )
            maximum_supported_node_outflow = self._compute_node_maximum_outflows(
                network=network,
                node=node,
                splits=splits,
                densities=densities,
                flows=flows,
                node_splits=normalized_node_splits,
                flow_boundary_conditions=flow_boundary_conditions,
                dt=dt,
            )

            # ! 3) If the maximum outflow is larger than the currently computed sum of desired inflows,
            # if the maximum outflow is larger than the currently computed sum of desired inflows,
            # reduce the last cell flows (motorway link) and the overall flows on the onramp / origin proportionally
            total_capped_inflow = casadi.SX(0)
            reduction_factor = casadi.if_else(
                casadi.logic_and(
                    total_node_inflow > maximum_supported_node_outflow,
                    total_node_inflow > 1e-6,
                ),
                maximum_supported_node_outflow / total_node_inflow,
                1,
            )

            for inc in node.incoming:
                if (
                    isinstance(inc, Origin)
                    or isinstance(inc, Onramp)
                    or isinstance(inc, Offramp)
                ):
                    # if necessary, reduce the computed flows proportionally to their desired values
                    next_flows[inc.id] = next_flows[inc.id] * reduction_factor
                    total_capped_inflow += next_flows[inc.id]

                    # update the virtual queues on origins and onramps accordingly
                    # (based on the difference between desired and actual flow)
                    if isinstance(inc, Origin):
                        next_origin_queues[inc.id] = update_queue(
                            queue_length=origin_queues[inc.id],
                            demand=origin_demands[inc.id],
                            flow=next_flows[inc.id],
                            dt=dt,
                        )
                    elif isinstance(inc, Offramp):
                        # compute outflow of the upstream node directed into this motorway link
                        offramp_inflow = self._get_node_outflow_link(
                            network=network,
                            splits=splits,
                            link=inc,
                            flows=flows,
                        )

                        # update the virtual queue of the offramp based on the difference between
                        # the forwarded demand and the allowed flow
                        next_offramp_queues[inc.id] = update_queue(
                            queue_length=offramp_queues[inc.id],
                            demand=offramp_inflow,
                            flow=next_flows[inc.id],
                            dt=dt,
                        )
                    else:
                        # compute outflow of the upstream node directed into this motorway link
                        onramp_inflow = self._get_node_outflow_link(
                            network=network,
                            splits=splits,
                            link=inc,
                            flows=flows,
                        )

                        # update the virtual queue of the onramp based on the difference between
                        # the forwarded demand (through origin and node) and the allowed flow
                        next_onramp_queues[inc.id] = update_queue(
                            queue_length=onramp_queues[inc.id],
                            demand=onramp_inflow,
                            flow=next_flows[inc.id],
                            dt=dt,
                        )

                elif isinstance(inc, MotorwayLink):
                    # if necessary, reduce the last cell's flow proportionally to the desired value
                    next_flows[inc.id][-1] = next_flows[inc.id][-1] * reduction_factor
                    total_capped_inflow += next_flows[inc.id][-1]

                    # a modified cell flow also implies a modified speed (first-order model)
                    # (density is not affected, since it is updated explicitly based on previous-step quantities)
                    next_speeds[inc.id][-1] = casadi.if_else(
                        next_densities[inc.id][-1] > 1e-6,
                        next_flows[inc.id][-1]
                        / (inc.lanes * next_densities[inc.id][-1]),
                        inc.vf,
                    )

                else:
                    raise TypeError(f"Unknown incoming link type: {type(inc)}")

            # ! 4) Update the flows onto outgoing offramps and into destinations with corresponding queue updates where applicable
            for out in node.outgoing:
                if isinstance(out, Destination):
                    # destinations are assumed to consume all incoming flow
                    # (only impact the mainstream through the density boundary condition)
                    # since destinations are virtual sinks with no link length, they directly
                    # consume the next-step flow -> flow at destination at k+1 is equal to the outflow
                    # of the node at time k+1 multiplied by the corresponding split ratio
                    next_flows[out.id] = (
                        normalized_node_splits[out.id] * total_capped_inflow
                    )

                elif isinstance(out, Offramp):
                    # offramps are modeled as store-and-forward links with the mainline outflow given as a demand
                    # and a virtual queue that takes up excess demand if the offramp capacity is exceeded or
                    # congestion further reduces the correpsonding flows off the offramp
                    node_outflows = compute_node_outflows(
                        node=node, flows=flows, node_splits=normalized_node_splits
                    )

                    # fetch the node downstream of the offramp and the connected destination
                    # in order to identify the correct downstream density boundary condition
                    if out.destination_node_id is None:
                        raise ValueError(
                            f"Offramp {out.id} does not have a well-defined destination node."
                        )
                    offramp_downstream_node = network.get_node(out.destination_node_id)
                    if offramp_downstream_node is None:
                        raise ValueError(
                            f"Offramp {out.id} has invalid destination node id {out.destination_node_id}."
                        )
                    if len(offramp_downstream_node.outgoing) != 1 or not isinstance(
                        offramp_downstream_node.outgoing[0], Destination
                    ):
                        raise ValueError(
                            f"Offramp {out.id} is not connected to a single destination downstream."
                        )

                    destination = offramp_downstream_node.outgoing[0]
                    destination_density_bc = density_boundary_conditions[destination.id]

                    # offramp outflows are updated with the current step flows, since the offramp
                    # keeps track of its own queue and flow and has a physical length
                    next_outflow, next_queue = self._compute_offramp_outflows(
                        offramp=out,
                        node_outflow=node_outflows[out.id],
                        offramp_queues=offramp_queues,
                        density_boundary_condition=destination_density_bc,
                        dt=dt,
                    )

                    # set the offramp flow and the queue on the offramp (part of store-and-forward link)
                    # flow for the connected destination is not set => equal to the offramp flow
                    next_flows[out.id] = next_outflow
                    next_offramp_queues[out.id] = next_queue

                elif isinstance(out, MotorwayLink):
                    # motorway links are processed at nodes where they are incoming
                    pass

                elif isinstance(out, Onramp):
                    # onramps are processed at nodes where they are incoming
                    pass

                else:
                    raise TypeError(f"Unknown outgoing link type: {type(out)}")

        # combine the network dictionary values for the next step into a single state vector
        x_next, _, _, _, _, _, _, _, _ = network.network_dict_to_state_vec(
            flow_dict=next_flows,
            density_dict=next_densities,
            speed_dict=next_speeds,
            origin_queue_dict=next_origin_queues,
            onramp_queue_dict=next_onramp_queues,
            offramp_queue_dict=next_offramp_queues,
        )

        # wrap the state update in a nonlinear casadi function (with dummy parameters for CTM)
        sym_params = casadi.SX.sym("ctm_params", 0)  # type: ignore
        return casadi.Function("ctm_network_step", [sym_params, x, d], [x_next])
