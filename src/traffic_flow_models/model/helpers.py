import numpy as np
import casadi
from typing import Tuple
from numpy.typing import NDArray

from traffic_flow_models.network.cell import Cell
from traffic_flow_models.controller.alinea import AlineaController


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
