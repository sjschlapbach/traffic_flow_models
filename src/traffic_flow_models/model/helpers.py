from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from traffic_flow_models.network.cell import Cell
from traffic_flow_models.controller.alinea import AlineaController


def calculate_segment_input_flow(
    first_cell: Cell,
    backward_wave_speed: float,
    density: float,
    input_demand: float,
    input_queue: float,
    dt: float,
) -> Tuple[float, float]:
    """
    Calculate the input flow and updated virtual input queue for a highway segment.

    Args:
        first_cell: The first cell of the highway segment.
        backward_wave_speed: The backward wave speed of the first cell.
        density: Current density in the first cell (vehicles per length unit).
        input_demand: Demand for vehicles entering the segment (vehicles per time unit).
        input_queue: Current virtual input queue length (vehicles).
        dt: Time step over which to integrate (time units consistent with flow units).

    Returns:
        A tuple (input_flow, updated_input_queue) where input_flow is the
        calculated flow entering the segment and updated_input_queue is the
        new queue length after accounting for the flow.
    """
    qin_demand = input_demand + input_queue / dt
    qin_supply = backward_wave_speed * (first_cell.rho_jam - density)
    input_flow = min(first_cell.Qc, qin_demand, qin_supply)
    updated_input_queue = update_queue(
        queue_length=input_queue, demand=input_demand, flow=input_flow, dt=dt
    )
    return input_flow, updated_input_queue


def update_queue(queue_length: float, demand: float, flow: float, dt: float) -> float:
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


def __calculate_onramp_flow(
    cell: Cell,
    backward_wave_speed: float,
    density: float,
    onramp_demand: float,
    onramp_queue: int,
    r_controlled: float,
    dt: float,
) -> float:
    """
    Calculate the onramp flow based on demand, queue, and downstream conditions.

    Args:
        cell: The Cell instance to which the onramp is attached.
        backward_wave_speed: The backward wave speed of the cell.
        density: Current density in the cell (vehicles per length unit).
        onramp_demand: Demand for vehicles from the onramp (vehicles per time unit).
        onramp_queue: Current onramp queue length (vehicles).
        r_controlled: Controlled flow limit for the onramp (vehicles per time unit).

    Returns:
        Calculated onramp flow (vehicles per time unit).

    Raises:
        ValueError: If the cell does not have an onramp attached.
    """

    if cell.onramp is None:
        raise ValueError("Cell does not have an onramp attached.")

    r_demand = onramp_demand + onramp_queue / dt
    r_supply = backward_wave_speed * (cell.rho_jam - density)

    onramp_flow = min(cell.onramp.Qc, r_demand, r_supply, r_controlled)
    return onramp_flow


def calculate_regulated_onramp_flow(
    cell: Cell,
    cell_ix: int,
    backward_wave_speed: float,
    density: NDArray[np.float64],
    previous_onramp_flow: float,
    onramp_demand: float,
    onramp_queue: int,
    dt: float,
    controller: AlineaController | None = None,
) -> float:
    """
    Calculate the regulated according to demand and supply terms, as well as
    the ramp demand and queue and possible control actions.

    Args:
        cell: The Cell instance to which the onramp is attached.
        cell_ix: Index of the current cell in the network.
        backward_wave_speed: The backward wave speed of the cell.
        density: Current densities in the network (vehicles per length unit).
        previous_onramp_flow: Previous onramp flow (vehicles per time unit).
        onramp_demand: Demand for vehicles from the onramp (vehicles per time unit).
        onramp_queue: Current onramp queue length (vehicles).
        dt: Time step over which to integrate (time units consistent with flow units).
        controller: Optional AlineaController instance for regulating the onramp flow.

    Returns:
        Calculated onramp flow (vehicles per time unit).
    """

    if cell.onramp is None:
        raise ValueError("Cell does not have an onramp attached.")

    # initialize the regulated onramp flow to be infinite (i.e. not controlled)
    r_alinea = np.inf

    # if a controller is defined, compute the regulated onramp flow
    if controller is not None:
        r_alinea = controller.compute_regulated_flow(
            measured_densities=density, previous_flow=previous_onramp_flow
        )

    # compute the final onramp flow considering demand, supply and control
    regulated_onramp_flow = __calculate_onramp_flow(
        cell=cell,
        backward_wave_speed=backward_wave_speed,
        density=density[cell_ix],
        onramp_demand=onramp_demand,
        onramp_queue=onramp_queue,
        r_controlled=r_alinea,
        dt=dt,
    )

    return regulated_onramp_flow
