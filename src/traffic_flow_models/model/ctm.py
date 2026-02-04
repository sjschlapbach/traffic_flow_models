import casadi
from typing import TYPE_CHECKING, Tuple

from .helpers import store_and_forward_update
from traffic_flow_models.network import (
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
)

if TYPE_CHECKING:
    from traffic_flow_models.network.network import Network


class CTM:
    """
    # TODO: add docstring
    """

    def __init__(self):
        """Create an empty CTM model instance."""
        return

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
    # def _compute_virtual_downstream_density(
    #     self,
    #     node: Node,
    #     densities: dict[str, casadi.SX],
    #     boundary_conditions: dict[str, casadi.SX],
    # ) -> Tuple[casadi.SX, Union[float, None], Union[float, None]]:
    #     """Determine a node's virtual downstream density for METANET updates.

    #     The virtual downstream density is used when computing boundary and
    #     anticipation terms at nodes with multiple outgoing links. For each
    #     outgoing link the method selects an appropriate downstream density
    #     representation:
    #     - For a `MotorwayLink`: the density of its first cell is used.
    #     - For an `Offramp`: the connected destination's boundary condition
    #       is used (offramps do not carry internal density in the
    #       store-and-forward representation).
    #     - For a `Destination`: the provided boundary condition is used.

    #     If multiple motorway or destination densities are present they are
    #     combined using a weighted quadratic mean implemented as
    #     ``sum(d**2)/sum(d)`` (CasADi symbolic expression).

    #     Args:
    #         node (Node): Network node for which to compute the downstream
    #             density.
    #         params (METANETSymbolicParams): METANET model parameters (CasADi SX).
    #         densities (dict[str, casadi.SX]): Mapping link id -> vector of
    #             cell densities (CasADi SX) for motorway links.
    #         boundary_conditions (dict[str, casadi.SX]): Mapping of link or
    #             destination id to boundary density (CasADi SX).

    #     Returns:
    #         Tuple[casadi.SX, Union[casadi.SX, None], Union[casadi.SX, None]]: A tuple containing:
    #             - The virtual downstream density of the node (CasADi SX).
    #             - The virtual downstream jam density of the node (CasADi SX or None).
    #             - The virtual downstream backward wave speed of the node (CasADi SX or None).

    #     Raises:
    #         ValueError: If the node has no outgoing links or an offramp has
    #             no destination defined.
    #         TypeError: If an outgoing link has an unexpected type.
    #     """
    #     # initialize variables
    #     node_downstream_density = None
    #     node_downstream_jam_density = None
    #     node_downstream_backward_wave_speed = None

    #     # determine the virtual downstream density of the node based on the outgoing links = q_m,N_m+1(k)
    #     if len(node.outgoing) > 1:
    #         out_densities: list[casadi.SX] = []

    #         for out_link in node.outgoing:
    #             if isinstance(out_link, MotorwayLink):
    #                 # motorway link: use the density of the first cell as downstream density
    #                 out_densities.append(densities[out_link.id][0])

    #             elif isinstance(out_link, Offramp):
    #                 # offramp link: the store-and-forward model does not model density / speed on offramps
    #                 # -> directly use the boundary condition of the connected destination as downstream density
    #                 if out_link.destination is None:
    #                     raise ValueError(
    #                         f"Offramp {out_link.id} does not have a destination defined."
    #                     )

    #                 out_densities.append(boundary_conditions[out_link.destination.id])

    #             elif isinstance(out_link, Destination):
    #                 # destination link: density is provided as boundary condition
    #                 out_densities.append(boundary_conditions[out_link.id])

    #             else:
    #                 raise TypeError(f"Unknown outgoing link type {type(out_link)}")

    #         # combine the different downstream densities (e.g., weighted average)
    #         numer = casadi.sum(casadi.vertcat(*[d**2 for d in out_densities]))
    #         denom = casadi.sum(casadi.vertcat(*out_densities))
    #         node_downstream_density = casadi.if_else(denom == 0, 0, numer / denom)

    #         # for multiple outgoing links, the virtual downstream jam density
    #         # and backward wave speed are not well-defined
    #         node_downstream_jam_density = None
    #         node_downstream_backward_wave_speed = None

    #     # single outgoing link: directly use its downstream density
    #     elif len(node.outgoing) == 1:
    #         out_link = node.outgoing[0]

    #         if isinstance(out_link, MotorwayLink):
    #             # motorway link: use the density of the first cell as downstream density
    #             node_downstream_density = densities[out_link.id][0]
    #             node_downstream_jam_density = out_link.rho_jam
    #             node_downstream_backward_wave_speed = self.backward_wave_speed(
    #                 params=params,
    #                 link_id=out_link.id,
    #                 capacity=out_link.lane_capacity * out_link.lanes,
    #                 lane_capacity=out_link.lane_capacity,
    #                 jam_density=out_link.rho_jam,
    #                 free_flow_speed=out_link.vf,
    #             )

    #         elif isinstance(out_link, Offramp):
    #             # offramp link: the store-and-forward model does not model density / speed on offramps
    #             # -> directly use the boundary condition of the connected destination as downstream density
    #             if out_link.destination is None:
    #                 raise ValueError(
    #                     f"Offramp {out_link.id} does not have a destination defined."
    #                 )

    #             node_downstream_density = boundary_conditions[out_link.destination.id]
    #             node_downstream_jam_density = None  # no downstream jam density defined -> handling on calling level required
    #             node_downstream_backward_wave_speed = None  # no downstream backward wave speed defined -> handling on calling level required
    #         elif isinstance(out_link, Destination):
    #             # destination link: density is provided as boundary condition
    #             node_downstream_density = boundary_conditions[out_link.id]
    #             node_downstream_jam_density = None  # no downstream jam density defined -> handling on calling level required
    #             node_downstream_backward_wave_speed = None  # no downstream backward wave speed defined -> handling on calling level required
    #         else:
    #             raise TypeError(f"Unknown outgoing link type {type(out_link)}")
    #     else:
    #         raise ValueError(f"No outgoing links defined for node {node.id}")

    #     return (
    #         node_downstream_density,
    #         node_downstream_jam_density,
    #         node_downstream_backward_wave_speed,
    #     )

    # def _compute_node_outflows_upstream_speed(
    #     self,
    #     node: Node,
    #     flows: dict[str, casadi.SX],
    #     speeds: dict[str, casadi.SX],
    #     splits: dict[str, dict[str, casadi.SX]],
    # ) -> Tuple[dict[str, casadi.SX], casadi.SX]:
    #     """Compute the node outflows and virtual upstream speed for METANET updates.

    #     The method computes the total available flow into the node by summing
    #     the last cell flows of all incoming motorway links as well as the
    #     flows from origins and onramps. Based on the total available flow and
    #     the provided split ratios, the method computes the outflows for each
    #     outgoing link. Additionally, the method computes the virtual upstream speed
    #     used in the speed update equations for outgoing motorway links.

    #     Args:
    #         node (Node): Network node for which to compute outflows and
    #             upstream speed.
    #         flows (dict[str, casadi.SX]): Mapping link id -> vector of cell
    #             flows (CasADi SX) for motorway links, origins, onramps and
    #             offramps.
    #         speeds (dict[str, casadi.SX]): Mapping link id -> vector of cell
    #             speeds (CasADi SX) for motorway links.
    #         splits (dict[str, dict[str, casadi.SX]]): Mapping of node id to
    #             mapping of outgoing link id to split ratio (CasADi SX).

    #     Returns:
    #         Tuple[dict[str, casadi.SX], casadi.SX]: A tuple containing:
    #             - A dictionary mapping outgoing link id to computed outflow
    #               (CasADi SX).
    #             - The virtual upstream speed (CasADi SX) used in speed updates.
    #     """
    #     Qn = casadi.SX(0)
    #     for inc in node.incoming:
    #         if isinstance(inc, MotorwayLink):
    #             # motorway link: use the last cell flow as upstream flow
    #             Qn += flows[inc.id][-1]
    #         elif isinstance(inc, Origin):
    #             # origin link: use the flow entering the origin (from state vector) - demand -> flow update separate
    #             Qn += flows[inc.id][0]
    #         elif isinstance(inc, Onramp):
    #             # onramp link: use the flow entering the onramp (from state vector) - demand -> flow update separate
    #             Qn += flows[inc.id][0]
    #         else:
    #             raise TypeError("Unknown incoming link type")

    #     # compute the node outflows based on the total available flow and the splits
    #     # node outflows = q_m,0(k) - dictionary with one value per outgoing edge
    #     node_outflows = {}
    #     node_splits = splits[node.id]

    #     if node_splits is None:
    #         raise ValueError(f"No split ratios defined for node {node.id}")

    #     for out in node.outgoing:
    #         out_split = node_splits[out.id]
    #         if out_split is None:
    #             raise ValueError(
    #                 f"No split ratio defined for outgoing link {out.id} (type: {type(out)}) at node {node.id}"
    #             )

    #         # re-normalize turning rates to make sure that they properly sum up to 1
    #         total_splits = casadi.sum(casadi.vertcat(*list(node_splits.values())))
    #         node_outflows[out.id] = Qn * out_split / casadi.fmax(total_splits, 1.0)

    #     # determine the virtual upstream speed of the node (for outgoing motorway links) = v_m,0(k)
    #     # since a speed parameter is required, only incoming motorway links are considered
    #     # if all incoming links are origins, assume free flow conditions upstream
    #     if all(not isinstance(inc, MotorwayLink) for inc in node.incoming):
    #         # if any onramps are connected to the node, use the minimum free-flow speed of those onramps
    #         if any(isinstance(inc, Onramp) for inc in node.incoming):
    #             node_upstream_speed = casadi.SX(
    #                 min(inc.vf for inc in node.incoming if isinstance(inc, Onramp))
    #             )

    #         # if only an origin is connected as an incoming link (and correspondingly only one outgoing
    #         # motorway link is allowed), choose the free-flow speed of the outgoing motorway link
    #         # for consistency (origin does not have free flow speed defined)
    #         else:
    #             if (
    #                 len(node.incoming) != 1
    #                 or not isinstance(node.incoming[0], Origin)
    #                 or len(node.outgoing) != 1
    #                 or not isinstance(node.outgoing[0], MotorwayLink)
    #             ):
    #                 raise ValueError(
    #                     "Encountered node without expected types of input links (more than one Origin / more than one outgoing link for origin-linked node)."
    #                 )

    #             node_upstream_speed = casadi.SX(node.outgoing[0].vf)

    #     else:
    #         # keep track of the minimum free-flow speed of incoming motorway links
    #         # -> in case upstream flow is zero, use this value as upstream speed
    #         min_vf: float = np.inf

    #         numer_terms = []
    #         denom_terms = []
    #         for inc in node.incoming:
    #             if isinstance(inc, MotorwayLink):
    #                 # motorway link: use the last cell speed and flow for upstream speed
    #                 numer_terms.append(speeds[inc.id][-1] * flows[inc.id][-1])
    #                 denom_terms.append(flows[inc.id][-1])
    #                 min_vf = min(min_vf, inc.vf)

    #         # catch the case where no values were measured -> should not happen
    #         if len(numer_terms) == 0 or len(denom_terms) == 0 or np.isinf(min_vf):
    #             raise ValueError(
    #                 f"No incoming motorway links with defined speeds/flows for node {node.id}."
    #             )

    #         numer_sum = casadi.sum(casadi.vertcat(*numer_terms))
    #         denom_sum = casadi.sum(casadi.vertcat(*denom_terms))
    #         node_upstream_speed = casadi.if_else(
    #             denom_sum == 0, min_vf, numer_sum / denom_sum
    #         )

    #     return node_outflows, node_upstream_speed

    # def _compute_offramp_outflows(
    #     self,
    #     params: METANETSymbolicParams,
    #     offramp: Offramp,
    #     node_outflows: dict[str, casadi.SX],
    #     offramp_queues: dict[str, casadi.SX],
    #     boundary_conditions: dict[str, casadi.SX],
    #     dt: float,
    # ) -> Tuple[casadi.SX, casadi.SX]:
    #     """Compute offramp outflow and update the offramp store-and-forward queue.

    #     Offramps are represented as store-and-forward links with finite
    #     capacity. This method computes the desired mainline outflow onto the
    #     offramp (from `node_outflows`) and combines it with the current
    #     offramp queue to form an offramp demand. The actual offramp outflow
    #     and the updated queue are computed via `store_and_forward_update`,
    #     which uses the offramp's capacity, jam density and the downstream
    #     (destination) boundary density.

    #     Args:
    #         offramp (Offramp): The offramp link for which to compute outflow.
    #         node_outflows (dict[str, casadi.SX]): Mapping of outgoing link id
    #             to the desired outflow at the node (CasADi SX).
    #         offramp_queues (dict[str, casadi.SX]): Current queue lengths on
    #             offramps (CasADi SX).
    #         boundary_conditions (dict[str, casadi.SX]): Mapping of destination
    #             id to boundary density (CasADi SX) used as downstream density.
    #         dt (float): Simulation timestep.

    #     Returns:
    #         Tuple[casadi.SX, casadi.SX]: Tuple `(next_outflow, next_queue)`
    #         where `next_outflow` is the offramp outflow (vehicles/time) into
    #         the connected destination, and `next_queue` is the updated queue
    #         length on the offramp (vehicles).

    #     Raises:
    #         ValueError: If the `offramp` does not have a `destination` defined.
    #     """
    #     if offramp.destination is None:
    #         raise ValueError(
    #             f"Offramp {offramp.id} does not have a destination defined."
    #         )

    #     mainline_outflow = node_outflows[
    #         offramp.id
    #     ]  # desired offramp flow based on splits = flow onto offramp (queue on offramp itself)
    #     offramp_demand = mainline_outflow + offramp_queues[offramp.id] / dt

    #     # update the offramp flow and queue based on the store-and-forward model
    #     next_outflow, next_queue = store_and_forward_update(
    #         capacity=offramp.Qc,
    #         jam_density=offramp.rho_jam,
    #         backward_wave_speed=self.backward_wave_speed(
    #             params=params,
    #             link_id=offramp.id,
    #             capacity=offramp.Qc,
    #             lane_capacity=offramp.Qc_lane,
    #             jam_density=offramp.rho_jam,
    #             free_flow_speed=offramp.vf,
    #         ),
    #         density=boundary_conditions[offramp.destination.id],
    #         demand=offramp_demand,
    #         queue=offramp_queues[offramp.id],
    #         dt=dt,
    #     )

    #     return next_outflow, next_queue

    # def _compute_motorway_link_outflows(
    #     self,
    #     params: METANETSymbolicParams,
    #     link: MotorwayLink,
    #     node: Node,
    #     node_outflows: dict[str, casadi.SX],
    #     node_downstream_density: casadi.SX,
    #     node_upstream_speed: casadi.SX,
    #     node_upstream_onramp_inflows: casadi.SX | None,
    #     flows: dict[str, casadi.SX],
    #     densities: dict[str, casadi.SX],
    #     speeds: dict[str, casadi.SX],
    #     dt: float,
    # ) -> Tuple[casadi.SX, casadi.SX, casadi.SX]:
    #     """Compute next-step densities, speeds and flows for a motorway link.

    #     This helper advances all cells on a `MotorwayLink` by one simulation
    #     timestep using the METANET `cell_update` routine. The method applies
    #     the node-level outflow as the upstream boundary condition for the
    #     first cell and uses `node_downstream_density` for the downstream
    #     boundary condition of the last cell. Per-cell upstream speeds are
    #     provided via `node_upstream_speed` for the first cell and the
    #     `speeds` vector for internal cells.

    #     Args:
    #         params (METANETSymbolicParams): Model parameters.
    #         link (MotorwayLink): Motorway link containing the cells to update.
    #         node (Node): The upstream node of the link (used for error
    #             messages and contextual checks).
    #         node_outflows (dict[str, casadi.SX]): Mapping of outgoing link id
    #             to the node-level outflow (CasADi SX).
    #         node_downstream_density (casadi.SX): Virtual downstream density
    #             at the node used as boundary for the last cell (CasADi SX).
    #         node_upstream_speed (casadi.SX): Virtual upstream speed at the
    #             node used as boundary for the first cell (CasADi SX).
    #         node_upstream_onramp_inflows (casadi.SX | None): Total inflow
    #             from onramps connected upstream of the node used to account
    #             for speed reduction terms caused by merging traffic.
    #         flows (dict[str, casadi.SX]): Current per-link flow vectors
    #             (CasADi SX).
    #         densities (dict[str, casadi.SX]): Current per-link density vectors
    #             (CasADi SX).
    #         speeds (dict[str, casadi.SX]): Current per-link speed vectors
    #             (CasADi SX).
    #         dt (float): Simulation timestep.

    #     Returns:
    #         Tuple[casadi.SX, casadi.SX, casadi.SX]: Three CasADi column vectors
    #         of length equal to the number of cells on `link` containing the
    #         virtual downstream link densities, speeds and outflows respectively.

    #     Raises:
    #         ValueError: If no node outflow has been computed for `link.id`.
    #     """

    #     if node_outflows[link.id] is None:
    #         raise ValueError(
    #             f"No outflow computed for outgoing motorway link {link.id} at node {node.id}"
    #         )

    #     link_flows = flows[link.id]
    #     link_densities = densities[link.id]
    #     link_speeds = speeds[link.id]

    #     next_densities_list = casadi.SX(len(link), 1)
    #     next_speeds_list = casadi.SX(len(link), 1)
    #     next_flows_list = casadi.SX(len(link), 1)

    #     for i, cell in link.enumerate_cells():
    #         # compute updates for this cell and append to lists
    #         d_next, s_next, f_next = self.cell_update(
    #             params=params,
    #             link=link,
    #             cell=cell,
    #             upstream_flow=(
    #                 link_flows[i - 1]
    #                 if cell.upstream is not None
    #                 else node_outflows[link.id]
    #             ),
    #             previous_flow=link_flows[i],
    #             previous_density=link_densities[i],
    #             downstream_density=(
    #                 link_densities[i + 1]
    #                 if cell.downstream is not None
    #                 else node_downstream_density
    #             ),
    #             upstream_speed=(
    #                 link_speeds[i - 1]
    #                 if cell.upstream is not None
    #                 else node_upstream_speed
    #             ),
    #             upstream_onramp_inflows=(
    #                 node_upstream_onramp_inflows if cell.upstream is None else None
    #             ),
    #             previous_speed=link_speeds[i],
    #             dt=dt,
    #         )

    #         next_densities_list[i] = d_next
    #         next_speeds_list[i] = s_next
    #         next_flows_list[i] = f_next

    #     return next_densities_list, next_speeds_list, next_flows_list

    # def cell_update(
    #     self,
    #     link: MotorwayLink,
    #     cell: Cell,
    #     upstream_flow: casadi.SX,
    #     previous_flow: casadi.SX,
    #     previous_density: casadi.SX,
    #     downstream_density: casadi.SX,
    #     upstream_speed: casadi.SX,
    #     upstream_onramp_inflows: casadi.SX | None,
    #     previous_speed: casadi.SX,
    #     dt: float,
    # ) -> Tuple[casadi.SX, casadi.SX, casadi.SX]:
    #     """Compute one-step updates for density, speed and flow of a METANET cell.

    #     Implements the METANET discrete-time update for a homogeneous motorway
    #     cell. Density is updated by vehicle conservation using the provided
    #     upstream and previous outflow. Speed evolves according to METANET's
    #     second-order dynamics: relaxation toward the stationary velocity,
    #     convective coupling with upstream speed, anticipation of downstream
    #     density gradients, and additional deceleration for upcoming lane
    #     drops. The updated flow is computed from the updated density and
    #     speed.

    #     Args:
    #         params (METANETSymbolicParams): Model parameters.
    #         link (MotorwayLink): Parent motorway link containing geometric
    #             and lane information.
    #         cell (Cell): Cell object with geometry and lane-drop info.
    #         upstream_flow (casadi.SX): Flow entering the cell from upstream
    #             (vehicles / time).
    #         previous_flow (casadi.SX): Flow leaving the cell at the previous
    #             time step (vehicles / time).
    #         previous_density (casadi.SX): Density at the previous time step
    #             (vehicles / length per lane).
    #         downstream_density (casadi.SX): Density in the downstream cell
    #             used for anticipation terms (vehicles / length per lane).
    #         upstream_speed (casadi.SX): Speed in the upstream cell used for
    #             convective coupling (length / time).
    #         upstream_onramp_inflows (casadi.SX | None): Total onramp inflow
    #             entering upstream of the cell used for speed reduction terms.
    #         previous_speed (casadi.SX): Speed at the previous time step in
    #             this cell (length / time).
    #         dt (float): Simulation timestep.

    #     Returns:
    #         Tuple[casadi.SX, casadi.SX, casadi.SX]: ``(density, speed, flow)``
    #         updated for one timestep where:
    #         - ``density``: Updated density (vehicles / length per lane).
    #         - ``speed``: Updated speed (length / time).
    #         - ``flow``: Updated outflow from the cell (vehicles / time).
    #     """

    #     # compute the new density based on the flows at the previous timestep
    #     # Note: off-ramps are modeled as splitting the outflow and do not
    #     # directly reduce the density update term (matches MATLAB METANET).
    #     density = previous_density + dt * (upstream_flow - previous_flow) / (
    #         cell.length * link.lanes
    #     )

    #     # compute the new speed based on the previous timestep
    #     speed = (
    #         previous_speed
    #         + dt
    #         / params["tau"]
    #         * (
    #             self.stationary_velocity(
    #                 params=params,
    #                 link_id=link.id,
    #                 lane_capacity=link.lane_capacity,
    #                 free_flow_speed=link.vf,
    #                 density=previous_density,
    #             )
    #             - previous_speed
    #         )
    #         + dt / cell.length * previous_speed * (upstream_speed - previous_speed)
    #         - (dt * params["nu"])
    #         / (params["tau"] * cell.length)
    #         * (downstream_density - previous_density)
    #         / (previous_density + params["kappa"])
    #     )

    #     # if the considered cell is the last downstream cell of a link into a node with onramp inflows, reduce the speed accordingly
    #     if upstream_onramp_inflows is not None:
    #         speed -= (
    #             (dt * params["delta"])
    #             / (cell.length * link.lanes)
    #             * (upstream_onramp_inflows * previous_speed)
    #             / (previous_density + params["kappa"])
    #         )

    #     # if a lane drop is coming up, add an additional term to the speed
    #     # update equation to account for the additional deceleration
    #     if cell.upcoming_lane_drop > 0:
    #         speed -= (
    #             dt
    #             * params["phi"]
    #             * cell.upcoming_lane_drop
    #             * previous_density
    #             * previous_speed**2
    #         ) / (
    #             cell.length
    #             * link.lanes
    #             * self.critical_density(
    #                 params=params,
    #                 link_id=link.id,
    #                 lane_capacity=link.lane_capacity,
    #                 free_flow_speed=link.vf,
    #             )
    #         )

    #     # ensure that the speed values remain non-negative
    #     speed = casadi.fmax(speed, 0)

    #     # compute the flow update of the cell based on the speed and density
    #     flow = density * speed * link.lanes

    #     return density, speed, flow

    # endregion

    def _compute_offramp_outflows(
        self,
        offramp: Offramp,
        mainline_outflow: casadi.SX,
        offramp_queues: dict[str, casadi.SX],
        boundary_conditions: dict[str, casadi.SX],
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX]:
        """Compute an offramp's outflow and update its store-and-forward queue.

        Offramps are modelled as store-and-forward links with finite
        capacity. This routine computes the offramp demand by combining the
        mainline portion intended for the offramp (``mainline_outflow``) and
        the current virtual queue on the offramp. The actual outflow and the
        updated queue are obtained by calling ``store_and_forward_update``
        with the offramp's capacity, jam density, a computed backward wave
        speed, and the downstream (destination) boundary density.

        Args:
            offramp (Offramp): The offramp link to update.
            mainline_outflow (casadi.SX): Desired flow from the mainline into
                the offramp (vehicles / time) as a CasADi expression.
            offramp_queues (dict[str, casadi.SX]): Current queue lengths on
                offramps indexed by link id (vehicles, CasADi SX).
            boundary_conditions (dict[str, casadi.SX]): Mapping from
                destination id to downstream density (vehicles / length / lane)
                used as the downstream boundary for the store-and-forward
                update (CasADi SX).
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
        if offramp.destination is None:
            raise ValueError(
                f"Offramp {offramp.id} does not have a destination defined."
            )

        # update the offramp flow and queue based on the store-and-forward model
        offramp_demand = mainline_outflow + offramp_queues[offramp.id] / dt
        next_outflow, next_queue = store_and_forward_update(
            capacity=offramp.Qc,
            jam_density=offramp.rho_jam,
            backward_wave_speed=self.backward_wave_speed(
                capacity=offramp.Qc,
                lane_capacity=offramp.Qc_lane,
                jam_density=offramp.rho_jam,
                free_flow_speed=offramp.vf,
            ),
            density=boundary_conditions[offramp.destination.id],
            demand=offramp_demand,
            queue=offramp_queues[offramp.id],
            dt=dt,
        )

        return next_outflow, next_queue

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
        """Build a CasADi function implementing one METANET network step.

        The returned CasADi `Function` (named ``metanet_network_step``) maps
        the symbolic model parameter vector, the current state vector ``x``
        and the disturbance vector ``d`` to the next-step state vector
        ``x_next`` according to the METANET dynamics combined with
        store-and-forward updates for origins, onramps and offramps.

        State and disturbance vector layouts follow
        `Network.state_vec_to_network_dict` and
        `Network.disturbance_vec_to_network_dict`. The disturbance vector
        contains origin demands, onramp demands, split ratios and boundary
        condition entries in the ordering expected by the network helpers.

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

        # TODO: extract duplicated / shared logic to helper functions where possible
        # formulate the individual update equations for each node and update the overall system equation and the next step state
        # iterate through all nodes and update the corrresponding quantities of incoming and outgoing links
        for node in network.list_nodes():
            # ! 1) iterate through all incoming links (origins, onramps, and motorway links) and update the flows
            # if the desired flow at the downstream node is higher than the maximum outgoing flow of the node, reduce
            # the flows proportionally to their desired value to obtain the actual flow values for the density updates
            total_node_inflow = casadi.SX(0)
            for inc in node.incoming:
                if isinstance(inc, MotorwayLink):
                    # TODO: extract this to a separate function
                    if inc.origin_node_id is None:
                        raise ValueError(
                            f"Motorway link {inc.id} does not have a well-defined origin node."
                        )

                    # identify the origin node of this link
                    upstream_node = network.get_node(id=inc.origin_node_id)
                    if upstream_node is None:
                        raise ValueError(
                            f"Origin node {inc.origin_node_id} of motorway link {inc.id} not found in network."
                        )

                    # compute the outflow from the upstream node into this link
                    upstream_inflow_sum: casadi.SX = casadi.sum(
                        casadi.vertcat(
                            *[flows[inc.id][-1] for inc in upstream_node.incoming]
                        )
                    )
                    upstream_node_split_link = splits[upstream_node.id][inc.id]
                    if upstream_node_split_link is None:
                        raise ValueError(
                            f"No split ratio defined for outgoing link {inc.id} at node {upstream_node.id}"
                        )
                    upstream_node_outflow_link = (
                        upstream_node_split_link * upstream_inflow_sum
                    )  # = q_0(k) for this motorway link

                    # TODO: extract this to a separate function
                    # step through the cells of the link and update the flows, densities and speeds accordingly
                    # for the last cell, currently ignore the downstream supply of space restriction
                    # -> will be handled separately after the loop once potential flow limits have been identified
                    link_flows = flows[inc.id]
                    link_densities = densities[inc.id]

                    next_densities_list = casadi.SX(len(inc), 1)
                    next_speeds_list = casadi.SX(len(inc), 1)
                    next_flows_list = casadi.SX(len(inc), 1)

                    for i, cell in inc.enumerate_cells():
                        # compute the new density in the cell based on the flows at the previous timestep
                        # -> onramp and offramp flows do not need to be considered anymore -> handled through nodes
                        if i == 0:
                            next_densities_list[i] = link_densities[i] + dt / (
                                cell.length * inc.lanes
                            ) * (upstream_node_outflow_link - link_flows[i])
                        else:
                            next_densities_list[i] = link_densities[i] + dt / (
                                cell.length * inc.lanes
                            ) * (link_flows[i - 1] - link_flows[i])

                    for j, cell in inc.enumerate_cells():
                        # compute the new flows based on the updated density (first-order model)
                        q_demand = inc.vf * next_densities_list[j] * inc.lanes
                        q_supply = (
                            self.backward_wave_speed(
                                capacity=inc.lane_capacity * inc.lanes,
                                lane_capacity=inc.lane_capacity,
                                jam_density=inc.rho_jam,
                                free_flow_speed=inc.vf,
                            )
                            * (inc.rho_jam - next_densities_list[j + 1])
                            if j < len(inc) - 1
                            else casadi.inf  # no supply restriction for last cell at this point -> will be introduced in terms of proportional flow reduction
                        )

                        next_flows_list[j] = casadi.fmin(
                            casadi.fmin(
                                casadi.SX(inc.lane_capacity * inc.lanes), q_demand
                            ),
                            q_supply,
                        )

                        # compute the updated speed based on the updated density and flow
                        next_speeds_list[j] = casadi.if_else(
                            next_densities_list[j] > 0,
                            next_flows_list[j] / (inc.lanes * next_densities_list[j]),
                            inc.vf,
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
                    # onramps are also modeled as store-and-forward links (as origins), but additionally
                    # have a finite capacity, which needs to be taken into account when computing the
                    # desired flow / flow on the onramp without considering downstream supply of space restrictions
                    next_flows[inc.id] = casadi.fmin(
                        inc.Qc,
                        onramp_demands[inc.id] + (onramp_queues[inc.id] / dt),
                    )
                    total_node_inflow += next_flows[inc.id]
                else:
                    raise TypeError(f"Unknown incoming link type: {type(inc)}")

            # TODO: extract this to a separate functions
            # ! 2) Normalize the split ratios and compute the maximum outflow of the node
            # compute the maximum outflow of the link according to the supply of space equation for each outgoing link
            # we assume that the most congested outgoing link determines the maximum outflow of the node
            # -> since CTM focusses on accumulations, this would correspond to a spillback scenario across the node
            # (assuming that the split ratios are not affected by changing traffic conditions on individual links)
            if any([splits[node.id][out.id] is None for out in node.outgoing]):
                raise ValueError(
                    f"Not all split ratios defined for outgoing links at node {node.id}."
                )

            normalized_node_splits: dict[str, casadi.SX] = {
                out.id: splits[node.id][out.id]
                / casadi.sum(
                    casadi.vertcat(
                        *[splits[node.id][outgoing.id] for outgoing in node.outgoing]
                    )
                )
                for out in node.outgoing
            }

            maximum_supported_node_outflow = casadi.SX(casadi.inf)
            for out in node.outgoing:
                if isinstance(out, Destination):
                    # destinations do not limit the outflow of the node
                    continue
                elif isinstance(out, Offramp):
                    # destinations are modeled as store-and-forward links with a virtual queue
                    # -> only the offramp capacity becomes a limiting factor for potential spillback
                    maximum_supported_node_outflow = casadi.fmin(
                        maximum_supported_node_outflow,
                        out.Qc / normalized_node_splits[out.id],
                    )
                elif isinstance(out, MotorwayLink):
                    # ! IMPORTANT TO FIX FOR CONSISTENT MODEL!
                    # TODO: investigate a potential causality / consistency issue where we would actually have to use
                    # the next-step density of the first cell of the outgoing link -> with the current computation
                    # approach / loop, we cannot be sure that this value has already been computed...
                    # (currently, the previous step density is used in its place)
                    maximum_supported_node_outflow = casadi.fmin(
                        maximum_supported_node_outflow,
                        (
                            self.backward_wave_speed(
                                capacity=out.lane_capacity * out.lanes,
                                lane_capacity=out.lane_capacity,
                                jam_density=out.rho_jam,
                                free_flow_speed=out.vf,
                            )
                            * (out.rho_jam - densities[out.id][0])
                        )
                        / normalized_node_splits[out.id],
                    )

            # TODO: extract this to a separate functions
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
                if isinstance(inc, Origin) or isinstance(inc, Onramp):
                    # if necessary, reduce the computed flows proportionally to their desired values
                    next_flows[inc.id] = next_flows[inc.id] * reduction_factor
                    total_capped_inflow += next_flows[inc.id]

                    # update the virtual queues on origins and onramps accordingly
                    # (based on the difference between desired and actual flow)
                    if isinstance(inc, Origin):
                        next_origin_queues[inc.id] = origin_queues[inc.id] + dt * (
                            origin_demands[inc.id] - next_flows[inc.id]
                        )
                    else:
                        next_onramp_queues[inc.id] = onramp_queues[inc.id] + dt * (
                            onramp_demands[inc.id] - next_flows[inc.id]
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
                    next_flows[out.id] = (
                        normalized_node_splits[out.id] * total_capped_inflow
                    )
                elif isinstance(out, Offramp):
                    # TODO: extract this to a separate function (check out _compute_offramp_outflows in METANET)
                    # offramps are modeled as store-and-forward links with the mainline outflow given as a demand
                    # and a virtual queue that takes up excess demand if the offramp capacity is exceeded or
                    # congestion further reduces the correpsonding flows off the offramp
                    next_outflow, next_queue = self._compute_offramp_outflows(
                        offramp=out,
                        mainline_outflow=normalized_node_splits[out.id]
                        * total_capped_inflow,
                        offramp_queues=offramp_queues,
                        boundary_conditions=boundary_conditions,
                        dt=dt,
                    )

                    # set the offramp flow and the queue on the offramp (part of store-and-forward link)
                    next_flows[out.id] = next_outflow
                    next_offramp_queues[out.id] = next_queue

                    # the flow of the connected destination is equal to the offramp outflow
                    if out.destination is not None:
                        next_flows[out.destination.id] = next_outflow
                    else:
                        raise ValueError(
                            f"Offramp {out.id} does not have a destination defined."
                        )
                elif isinstance(out, MotorwayLink):
                    # motorway links are processed at nodes where they are incoming
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
