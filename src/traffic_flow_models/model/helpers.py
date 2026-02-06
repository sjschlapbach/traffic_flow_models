import casadi
from typing import Tuple, Union

from traffic_flow_models.network import Node, Origin, Onramp, MotorwayLink


def store_and_forward_update(
    capacity: float,
    jam_density: float,
    backward_wave_speed: float,
    density: casadi.SX,
    demand: casadi.SX,
    queue: casadi.SX,
    dt: float,
) -> Tuple[casadi.SX, casadi.SX]:
    """Compute inflow and updated virtual input queue using the store-and-forward model.

    This function implements a store-and-forward input model suitable for
    CasADi symbolic variables. The inflow is the minimum of the segment
    capacity, the demand term (external demand plus queued vehicles distributed
    over the timestep), and the supply term given by the backward wave speed
    and jam density.

    Args:
        capacity: Maximum flow capacity of the segment (vehicles / time).
        jam_density: Jam density of the segment (vehicles / length).
        backward_wave_speed: Backward wave speed (length / time).
        density: Current density in the segment (CasADi SX, vehicles / length).
        demand: External demand entering the segment (CasADi SX, vehicles / time).
        queue: Current virtual input queue length (CasADi SX, vehicles).
        dt: Simulation timestep (time).

    Returns:
        A tuple ``(inflow, updated_queue)`` where:
        - ``inflow`` (CasADi SX): computed inflow entering the segment
          (vehicles / time) equal to
          ``min(capacity, demand + queue/dt, backward_wave_speed*(jam_density - density))``.
        - ``updated_queue`` (CasADi SX): new virtual queue after applying the
          inflow over the timestep, computed as ``queue + dt*(demand - inflow)``.
    """
    qin_demand: casadi.SX = demand + queue / dt
    qin_supply: casadi.SX = backward_wave_speed * (jam_density - density)
    inflow: casadi.SX = casadi.fmin(casadi.fmin(capacity, qin_demand), qin_supply)
    updated_queue: casadi.SX = update_queue(
        queue_length=queue, demand=demand, flow=inflow, dt=dt
    )
    return inflow, updated_queue


def update_queue(
    queue_length: casadi.SX,
    demand: casadi.SX,
    flow: casadi.SX,
    dt: float,
) -> casadi.SX:
    """
    Update the queue length based on demand and flow.

    Args:
        queue_length: Current queue length (vehicles).
        demand: Demand for vehicles (vehicles per time unit).
        flow: Actual flow of vehicles (vehicles per time unit).
        dt: Time step over which to integrate (time units consistent with flow units).

    Returns:
        Updated queue length (vehicles).
    """
    new_queue = queue_length + dt * (demand - flow)
    return new_queue


def compute_node_outflows(
    node: Node,
    flows: dict[str, casadi.SX],
    node_splits: Union[dict[str, casadi.SX], None],
) -> dict[str, casadi.SX]:
    """Compute node-level outflows from incoming flows and split ratios.

    This method sums the flows from all incoming links to obtain the
    total available upstream flow into `node` and then distributes that
    total to each outgoing link according to the split ratios provided
    in `node_splits` for the node.

    Args:
        node: Network node for which outgoing flows are computed.
        flows: Mapping link id -> CasADi SX vector of flows for that
            link (per-link layout depends on link type).
        node_splits: Mapping of outgoing link id to the split ratio
            used at `node` (CasADi SX).

    Returns:
        A dict mapping each outgoing link id to its computed outflow
        (CasADi SX).

    Raises:
        ValueError: If an outgoing link has no split defined.
            has no split defined.
        TypeError: If an incoming link has an unexpected type.
    """
    if node_splits is None:
        raise ValueError(f"No split ratios provided for node {node.id}")

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
    for out in node.outgoing:
        out_split = node_splits[out.id]
        if out_split is None:
            raise ValueError(
                f"No split ratio defined for outgoing link {out.id} (type: {type(out)}) at node {node.id}"
            )

        # re-normalize turning rates to make sure that they properly sum up to 1
        total_splits = casadi.sum(casadi.vertcat(*list(node_splits.values())))
        node_outflows[out.id] = (
            Qn * out_split / casadi.if_else(total_splits == 0, 1.0, total_splits)
        )

    return node_outflows
