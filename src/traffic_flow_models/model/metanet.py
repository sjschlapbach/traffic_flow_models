import numpy as np
import casadi
import warnings
from typing import Tuple


from traffic_flow_models.network.motorway_link import MotorwayLink
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.network import Network
from traffic_flow_models.network.node import Node
from traffic_flow_models.network.cell import Cell
from .helpers import (
    store_and_forward_update,
)


class METANET:
    def __init__(self, tau, nu, kappa, delta, phi, alpha):
        """Create a METANET model instance with given parameters.

        The METANET model is a second-order macroscopic traffic model that
        includes dynamics for both density and speed. This constructor
        stores the model parameters used in the speed update and flow
        computations.

        Args:
            tau (float): Relaxation time scale for speed dynamics (time).
            nu (float): Anticipation coefficient for downstream density
                gradient sensitivity.
            kappa (float): Small positive constant added to densities to
                avoid division-by-zero in speed/flow expressions.
            delta (float): Weighting coefficient for onramp influence on
                speed dynamics.
            phi (float): Coefficient for additional deceleration due to
                upcoming lane drops.
            alpha (float): Shape parameter used in the stationary velocity
                function (must be > 0).
        """

        self.tau = tau
        self.nu = nu
        self.kappa = kappa
        self.delta = delta
        self.phi = phi
        self.alpha = (
            alpha  # TODO: make this parameter link-specific -> relevant for fitting
        )

    def critical_density(self, lane_capacity: float, free_flow_speed: float) -> float:
        """
        Compute and return the critical density for a given lane capacity and free-flow speed.

        The METANET critical density is defined as
            rho_crit = lane_capacity / (free_flow_speed * exp(-1 / self.alpha))
        where self.alpha is the model shape parameter. Units: vehicles per length
        per lane. Requires self.alpha > 0.

        Args:
            lane_capacity: Lane capacity (vehicles per time).
            free_flow_speed: Free-flow speed (length per time).

        Returns:
            Critical density (vehicles per length per lane).
        """

        return lane_capacity / (free_flow_speed * np.exp(-1 / self.alpha))

    def backward_wave_speed(
        self,
        capacity: float,
        lane_capacity: float,
        jam_density: float,
        free_flow_speed: float,
    ) -> float:
        """
        Return the backward (congestion) wave speed for given fundamental parameters.

        The backward wave speed is computed as capacity / (jam_density - rho_crit)
        where rho_crit is the critical density computed from lane_capacity and
        free_flow_speed. This speed describes how congestion propagates upstream
        (length per time).

        Args:
            capacity: Cell capacity (vehicles per time).
            lane_capacity: Capacity per lane (vehicles per time).
            jam_density: Jam density (vehicles per length per lane).
            free_flow_speed: Free-flow speed (length per time).

        Returns:
            Backward wave speed (length per time).

        Raises:
            ValueError: If jam_density is less than or equal to the critical density.
        """

        rho_crit = self.critical_density(
            lane_capacity=lane_capacity, free_flow_speed=free_flow_speed
        )
        if jam_density <= rho_crit:
            raise ValueError(
                f"jam_density must be greater than the critical density to compute backward wave speed: {jam_density} <= {rho_crit}"
            )

        return capacity / (jam_density - rho_crit)

    def stationary_velocity(
        self, lane_capacity: float, free_flow_speed: float, density: float | casadi.SX
    ) -> float:
        """Compute the stationary (equilibrium) velocity for a cell.

        The stationary velocity is the speed that the traffic on the cell would
        adopt in the absence of dynamics, given the current density. METANET
        uses an exponential functional form parameterized by ``alpha`` and the
        cell's free-flow speed (fundamental diagram).

        Args:
            lane_capacity (float): Capacity per lane used to compute the
                critical density.
            free_flow_speed (float): Free-flow speed for the link.
            density (float | casadi.SX): The density at which to evaluate the
                stationary velocity (vehicles per length per lane).

        Returns:
            The stationary velocity (length per time unit).
        """

        exponent = (
            -1
            / self.alpha
            * (
                density
                / self.critical_density(
                    lane_capacity=lane_capacity, free_flow_speed=free_flow_speed
                )
            )
            ** self.alpha
        )

        return (
            free_flow_speed * casadi.exp(exponent)
            if isinstance(density, casadi.SX)
            else free_flow_speed * np.exp(exponent)
        )

    def _compute_virtual_downstream_density(
        self,
        node: Node,
        densities: dict[str, casadi.SX],
        boundary_conditions: dict[str, casadi.SX],
    ) -> casadi.SX:
        """Determine a node's virtual downstream density for METANET updates.

        The virtual downstream density is used when computing boundary and
        anticipation terms at nodes with multiple outgoing links. For each
        outgoing link the method selects an appropriate downstream density
        representation:
        - For a `MotorwayLink`: the density of its first cell is used.
        - For an `Offramp`: the connected destination's boundary condition
          is used (offramps do not carry internal density in the
          store-and-forward representation).
        - For a `Destination`: the provided boundary condition is used.

        If multiple motorway or destination densities are present they are
        combined using a weighted quadratic mean implemented as
        ``sum(d**2)/sum(d)`` (CasADi symbolic expression).

        Args:
            node (Node): Network node for which to compute the downstream
                density.
            densities (dict[str, casadi.SX]): Mapping link id -> vector of
                cell densities (CasADi SX) for motorway links.
            boundary_conditions (dict[str, casadi.SX]): Mapping of link or
                destination id to boundary density (CasADi SX).

        Returns:
            casadi.SX: Virtual downstream density to be used in downstream
            anticipation and supply computations.

        Raises:
            ValueError: If the node has no outgoing links or an offramp has
                no destination defined.
            TypeError: If an outgoing link has an unexpected type.
        """
        # determine the virtual downstream density of the node based on the outgoing links = q_m,N_m+1(k)
        node_downstream_density = None
        if len(node.outgoing) > 1:
            out_densities: list[casadi.SX] = []
            for out_link in node.outgoing:
                if isinstance(out_link, MotorwayLink):
                    # motorway link: use the density of the first cell as downstream density
                    out_densities.append(densities[out_link.id][0])
                elif isinstance(out_link, Offramp):
                    # offramp link: the store-and-forward model does not model density / speed on offramps
                    # -> directly use the boundary condition of the connected destination as downstream density
                    if out_link.destination is None:
                        raise ValueError(
                            f"Offramp {out_link.id} does not have a destination defined."
                        )

                    out_densities.append(boundary_conditions[out_link.destination.id])
                elif isinstance(out_link, Destination):
                    # destination link: density is provided as boundary condition
                    out_densities.append(boundary_conditions[out_link.id])
                else:
                    raise TypeError(f"Unknown outgoing link type {type(out_link)}")

            # combine the different downstream densities (e.g., weighted average)
            numer = casadi.vertcat(*[d**2 for d in out_densities])
            denom = casadi.vertcat(*out_densities)
            node_downstream_density = casadi.sum(numer) / casadi.sum(denom)

        elif len(node.outgoing) == 1:
            out_link = node.outgoing[0]

            if isinstance(out_link, MotorwayLink):
                # motorway link: use the density of the first cell as downstream density
                node_downstream_density = densities[out_link.id][0]
            elif isinstance(out_link, Offramp):
                # offramp link: the store-and-forward model does not model density / speed on offramps
                # -> directly use the boundary condition of the connected destination as downstream density
                if out_link.destination is None:
                    raise ValueError(
                        f"Offramp {out_link.id} does not have a destination defined."
                    )

                node_downstream_density = boundary_conditions[out_link.destination.id]
            elif isinstance(out_link, Destination):
                # destination link: density is provided as boundary condition
                node_downstream_density = boundary_conditions[out_link.id]
            else:
                raise TypeError(f"Unknown outgoing link type {type(out_link)}")
        else:
            raise ValueError(f"No outgoing links defined for node {node.id}")

        return node_downstream_density

    # TODO: split up this update into multiple helper functions to improve readability
    def network_update_function(
        self,
        network: Network,
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
        """Build a CasADi function implementing one METANET network step.

        The returned CasADi `Function` (named ``metanet_network_step``) maps
        the current state vector ``x`` and disturbance vector ``d`` to the
        next-step state vector ``x_next`` according to the METANET update
        rules combined with store-and-forward updates for origins, onramps
        and offramps.

        State and disturbance vector layouts are those used by
        `Network.state_vec_to_network_dict` and
        `Network.disturbance_vec_to_network_dict` and include flows,
        densities, speeds, origin/onramp queues, origin demands, onramp
        demands, split ratios and boundary conditions.

        Args:
            network (Network): Network object containing links, nodes and
                helper methods to convert between vectors and dictionaries.
            num_flows (int): Length of the flow portion of the state vector.
            num_densities (int): Length of the density portion of the state
                vector.
            num_speeds (int): Length of the speed portion of the state vector.
            num_origins (int): Number of origins.
            num_onramp (int): Number of onramps.
            num_offramp (int): Number of offramps.
            num_turning_rates (int): Number of turning rate disturbance entries.
            num_boundary_conditions (int): Number of boundary condition entries.
            inflows_jam_density (float): Jam density to use for origin
                inflows when modeling external inputs.
            inflows_free_flow_speed (float): Free-flow speed to use for
                origin inflows.
            dt (float): Simulation timestep.

        Returns:
            casadi.Function: A CasADi function `f(x,d) -> x_next` implementing
            the full network update for one timestep.
        """

        # set up state and disturbance vectors
        # state: flows, densities, speeds, origin, onramp
        # disturbances: origin_demands, onramp_demands, offramp_split_ratios
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
        d = casadi.SX.sym("d", num_origins + num_onramps + num_splits + num_destinations, 1)  # type: ignore

        # split up the state and disturbance vectors to obtain a dictionary for
        # efficient access of the relevant quantities during the state update
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            network.state_vec_to_network_dict(x=x)
        )
        origin_demands, onramp_demands, splits, boundary_conditions = (
            network.disturbance_vec_to_network_dict(d=d)
        )

        # typecast values of the dictionaries to casadi SX for symbolic computation
        flows = {k: casadi.SX(v) for k, v in flows.items()}
        densities = {k: casadi.SX(v) for k, v in densities.items()}
        speeds = {k: casadi.SX(v) for k, v in speeds.items()}
        origin_queues = {k: casadi.SX(v) for k, v in origin_queues.items()}
        onramp_queues = {k: casadi.SX(v) for k, v in onramp_queues.items()}
        offramp_queues = {k: casadi.SX(v) for k, v in offramp_queues.items()}
        origin_demands = {k: casadi.SX(v) for k, v in origin_demands.items()}
        onramp_demands = {k: casadi.SX(v) for k, v in onramp_demands.items()}
        splits = {
            k: {kk: casadi.SX(vv) for kk, vv in v.items()} for k, v in splits.items()
        }
        boundary_conditions = {k: casadi.SX(v) for k, v in boundary_conditions.items()}

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
            # ! 1) update the flows and queues for origins and onramps connected to this node
            node_downstream_density = self._compute_virtual_downstream_density(
                node=node,
                densities=densities,
                boundary_conditions=boundary_conditions,
            )
            for inc in node.incoming:
                if isinstance(inc, Origin):
                    next_inflow, next_queue = store_and_forward_update(
                        capacity=casadi.inf,
                        jam_density=casadi.inf,
                        backward_wave_speed=casadi.inf,
                        density=node_downstream_density,
                        demand=origin_demands[inc.id],
                        queue=origin_queues[inc.id],
                        dt=dt,
                    )

                    next_flows[inc.id] = next_inflow
                    next_origin_queues[inc.id] = next_queue

                elif isinstance(inc, Onramp):
                    # TODO: include possibility here for ramp metering controller (e.g. through ramp metering rate input)
                    next_inflow, next_queue = store_and_forward_update(
                        capacity=inc.Qc,
                        jam_density=inc.rho_jam,
                        backward_wave_speed=self.backward_wave_speed(
                            capacity=inc.Qc,
                            lane_capacity=inc.Qc_lane,
                            jam_density=inc.rho_jam,
                            free_flow_speed=inc.vf,
                        ),
                        density=node_downstream_density,
                        demand=onramp_demands[inc.id],
                        queue=onramp_queues[inc.id],
                        dt=dt,
                    )

                    next_flows[inc.id] = next_inflow
                    next_onramp_queues[inc.id] = next_queue

            # ! 2) compute the required boundary conditions based on the combined incoming / outgoing quantities
            # -> this includes the upstream speed, downstream density, etc. that are required by the model udpate
            # sum up the last cell flows of all incoming links, onramps and origins
            Qn = casadi.SX(0)
            for inc in node.incoming:
                if isinstance(inc, MotorwayLink):
                    # motorway link: use the last cell flow as upstream flow
                    Qn += flows[inc.id][-1]
                elif isinstance(inc, Origin):
                    # origin link: use the flow entering the origin (from state vector) - demand -> flow update separate
                    Qn += flows[inc.id][0]
                elif isinstance(inc, Onramp):
                    # onramp link: use the flow entering the onramp (from state vector) - demand -> flow update separate
                    Qn += flows[inc.id][0]
                else:
                    raise TypeError("Unknown incoming link type")

            # compute the node outflows based on the total available flow and the splits
            # node outflows = q_m,0(k) - dictionary with one value per outgoing edge
            node_outflows = {}
            node_splits = splits[node.id]

            if node_splits is None:
                raise ValueError(f"No split ratios defined for node {node.id}")

            for out in node.outgoing:
                out_split = node_splits[out.id]
                if out_split is None:
                    raise ValueError(
                        f"No split ratio defined for outgoing link {out.id} (type: {type(out)}) at node {node.id}"
                    )

                # re-normalize turning rates to make sure that they properly sum up to 1
                total_splits = casadi.sum(casadi.vertcat(*list(node_splits.values())))
                node_outflows[out.id] = Qn * out_split / casadi.fmax(total_splits, 1.0)

            # determine the virtual upstream speed of the node (for outgoing motorway links) = v_m,0(k)
            # since a speed parameter is required, only incoming motorway links are considered
            # if all incoming links are onramps / origins, assume free flow conditions upstream
            if all(not isinstance(inc, MotorwayLink) for inc in node.incoming):
                node_upstream_speed = min(
                    out.vf for out in node.outgoing if isinstance(out, MotorwayLink)
                )

            nom_terms = []
            denom_terms = []
            for inc in node.incoming:
                if isinstance(inc, MotorwayLink):
                    # motorway link: use the last cell speed and flow for upstream speed
                    nom_terms.append(speeds[inc.id][-1] * flows[inc.id][-1])
                    denom_terms.append(flows[inc.id][-1])

            node_upstream_speed = casadi.sum(casadi.vertcat(*nom_terms)) / casadi.sum(
                casadi.vertcat(*denom_terms)
            )

            for out in node.outgoing:
                # ! 3) update the offramp flows (& density/speed) for destinations connected to this node
                if isinstance(out, Destination):
                    # destinations are assumed to consume all incoming flow
                    # (only impact the mainstream through the density boundary condition)
                    next_flows[out.id] = node_outflows[out.id]
                elif isinstance(out, Offramp):
                    # offramps are modeled as store and forward links with finite capacity, meaning that the entire
                    # outflow is assumed to go into the offramp, where a queue might then form if the urban network
                    # boundary condition does not support the required outflow (spillbacks not considered /
                    # free flow boundary condition is applied to the relevant mainline highway segment!)
                    if out.destination is None:
                        raise ValueError(
                            f"Offramp {out.id} does not have a destination defined."
                        )

                    mainline_outflow = node_outflows[
                        out.id
                    ]  # desired offramp flow based on splits = flow onto offramp (queue on offramp itself)
                    offramp_demand = mainline_outflow + offramp_queues[out.id] / dt

                    # update the offramp flow and queue based on the store-and-forward model
                    next_outflow, next_queue = store_and_forward_update(
                        capacity=out.Qc,
                        jam_density=out.rho_jam,
                        backward_wave_speed=self.backward_wave_speed(
                            capacity=out.Qc,
                            lane_capacity=out.Qc_lane,
                            jam_density=out.rho_jam,
                            free_flow_speed=out.vf,
                        ),
                        density=boundary_conditions[out.destination.id],
                        demand=offramp_demand,
                        queue=offramp_queues[out.id],
                        dt=dt,
                    )

                    # set the offramp flow and the queue on the offramp (part of store-and-forward link)
                    next_flows[out.id] = next_outflow
                    next_offramp_queues[out.id] = next_queue

                    # the flow of the connected destination is equal to the offramp outflow
                    next_flows[out.destination.id] = next_outflow

                # ! 4) update the outgoing motorway links connected to this node (including all cells)
                elif isinstance(out, MotorwayLink):
                    if node_outflows[out.id] is None:
                        raise ValueError(
                            f"No outflow computed for outgoing motorway link {out.id} at node {node.id}"
                        )

                    link_flows = flows[out.id]
                    link_densities = densities[out.id]
                    link_speeds = speeds[out.id]

                    next_densities_list = casadi.SX(len(out), 1)
                    next_speeds_list = casadi.SX(len(out), 1)
                    next_flows_list = casadi.SX(len(out), 1)

                    for i, cell in out.enumerate_cells():
                        # compute updates for this cell and append to lists
                        d_next, s_next, f_next = self.cell_update(
                            link=out,
                            cell=cell,
                            upstream_flow=(
                                node_outflows[out.id]
                                if cell.upstream is None
                                else link_flows[i - 1]
                            ),
                            previous_flow=link_flows[i],
                            previous_density=link_densities[i],
                            downstream_density=(
                                link_densities[i + 1]
                                if cell.downstream is not None
                                else node_downstream_density
                            ),
                            upstream_speed=(
                                node_upstream_speed
                                if cell.upstream is None
                                else link_speeds[i - 1]
                            ),
                            previous_speed=link_speeds[i],
                            dt=dt,
                        )

                        next_densities_list[i] = d_next
                        next_speeds_list[i] = s_next
                        next_flows_list[i] = f_next

                    next_densities[out.id] = casadi.SX(next_densities_list)
                    next_speeds[out.id] = casadi.SX(next_speeds_list)
                    next_flows[out.id] = casadi.SX(next_flows_list)

                else:
                    raise TypeError(f"Unknown outgoing link type {type(out)}")

        # combine the network dictionary values for the next step into a single state vector
        x_next, _, _, _, _, _, _, _, _ = network.network_dict_to_state_vec(
            flow_dict=next_flows,
            density_dict=next_densities,
            speed_dict=next_speeds,
            origin_queue_dict=next_origin_queues,
            onramp_queue_dict=next_onramp_queues,
            offramp_queue_dict=next_offramp_queues,
        )

        # wrap the state update in a nonlinear casadi function
        return casadi.Function("metanet_network_step", [x, d], [x_next])

    def cell_update(
        self,
        link: MotorwayLink,
        cell: Cell,
        upstream_flow: casadi.SX,
        previous_flow: casadi.SX,
        previous_density: casadi.SX,
        downstream_density: casadi.SX,
        upstream_speed: casadi.SX,
        previous_speed: casadi.SX,
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX, casadi.SX]:
        """Compute one-step updates for density, speed and flow of a METANET cell.

        Implements the METANET discrete-time update for a homogeneous motorway
        cell. Density is updated by vehicle conservation using the provided
        upstream and previous outflow. Speed evolves according to METANET's
        second-order dynamics: relaxation toward the stationary velocity,
        convective coupling with upstream speed, anticipation of downstream
        density gradients, and additional deceleration for upcoming lane
        drops. The updated flow is computed from the updated density and
        speed.

        Args:
            link (MotorwayLink): Parent motorway link containing geometric
                and lane information.
            cell (Cell): Cell object with geometry and lane-drop info.
            upstream_flow (casadi.SX): Flow entering the cell from upstream
                (vehicles / time).
            previous_flow (casadi.SX): Flow leaving the cell at the previous
                time step (vehicles / time).
            previous_density (casadi.SX): Density at the previous time step
                (vehicles / length per lane).
            downstream_density (casadi.SX): Density in the downstream cell
                used for anticipation terms (vehicles / length per lane).
            upstream_speed (casadi.SX): Speed in the upstream cell used for
                convective coupling (length / time).
            previous_speed (casadi.SX): Speed at the previous time step in
                this cell (length / time).
            dt (float): Simulation timestep.

        Returns:
            Tuple[casadi.SX, casadi.SX, casadi.SX]: ``(density, speed, flow)``
            updated for one timestep where:
            - ``density``: Updated density (vehicles / length per lane).
            - ``speed``: Updated speed (length / time).
            - ``flow``: Updated outflow from the cell (vehicles / time).
        """

        # compute the new density based on the flows at the previous timestep
        # Note: off-ramps are modeled as splitting the outflow and do not
        # directly reduce the density update term (matches MATLAB METANET).
        density = previous_density + dt * (upstream_flow - previous_flow) / (
            cell.length * link.lanes
        )

        # compute the new speed based on the previous timestep
        speed = (
            previous_speed
            + dt
            / self.tau
            * (
                self.stationary_velocity(
                    lane_capacity=link.lane_capacity,
                    free_flow_speed=link.vf,
                    density=previous_density,
                )
                - previous_speed
            )
            + dt / cell.length * previous_speed * (upstream_speed - previous_speed)
            - (dt * self.nu)
            / (self.tau * cell.length)
            * (downstream_density - previous_density)
            / (previous_density + self.kappa)
        )

        # if a lane drop is coming up, add an additional term to the speed
        # update equation to account for the additional deceleration
        if cell.upcoming_lane_drop > 0:
            speed -= (
                dt
                * self.phi
                * cell.upcoming_lane_drop
                * previous_density
                * previous_speed**2
            ) / (
                cell.length
                * link.lanes
                * self.critical_density(
                    lane_capacity=link.lane_capacity,
                    free_flow_speed=link.vf,
                )
            )

        # ensure that the speed values remain non-negative
        speed = casadi.fmax(speed, 0)

        # compute the flow update of the cell based on the speed and density
        flow = density * speed * link.lanes

        return density, speed, flow
