from traffic_flow_models import Network, Cell, Onramp, Offramp, AlineaController
import numpy as np
from typing import Callable
from numpy.typing import NDArray

from .demand import demand


def mainline_demand_a(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 4000)


def mainline_demand_b(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 4000)


def mainline_demand_c(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 1500)


def onramp_demand_a(time: float, network_length: int) -> NDArray[np.float64]:
    ramp_demands = np.zeros(network_length)
    ramp_demands[2] = demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 2000)
    return ramp_demands


def onramp_demand_b(time: float, network_length: int) -> NDArray[np.float64]:
    ramp_demands = np.zeros(network_length)
    ramp_demands[2] = demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 2500)
    return ramp_demands


def onramp_demand_c(time: float, network_length: int) -> NDArray[np.float64]:
    ramp_demands = np.zeros(network_length)
    ramp_demands[2] = demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 1500)
    return ramp_demands


def setup_network_ab(
    get_critical_density: Callable[[Cell], float],
    ramp_control: bool = False,
    alinea_gain: float = 5.0,
    alinea_setpoint: float | None = None,
) -> Network:
    """Create a simple network with a single onramp in the middle."""

    network = Network()
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    onramp_cell = network.add_cell(
        length=0.5,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    onramp_cell.onramp = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=(
            AlineaController(
                gain=alinea_gain,
                # set the ALINEA setpoint to the critical density of the cell (fallback if no static setpoint is provided)
                setpoint=(
                    alinea_setpoint
                    if alinea_setpoint is not None
                    else get_critical_density(onramp_cell)
                ),
                measurement_cell=3,
            )
            if ramp_control is True
            else None
        ),
    )

    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    return network


def setup_network_c(
    get_critical_density: Callable[[Cell], float],
    ramp_control: bool = False,
    alinea_gain: float = 5.0,
    alinea_setpoint: float | None = None,
) -> Network:
    """
    Create a simple network with a single onramp in the middle and a
    bottleneck with lane drop downstream.

    The bottleneck is created by reducing the number of lanes in the
    downstream cell, which reduces its capacity and creates congestion
    that propagates upstream and interacts with the onramp / virtual input queue.
    """

    network = Network()
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    onramp_cell = network.add_cell(
        length=0.5,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    onramp_cell.onramp = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=(
            AlineaController(
                gain=alinea_gain,
                # set the ALINEA setpoint to the critical density of the cell (fallback if no static setpoint is provided)
                setpoint=(
                    alinea_setpoint
                    if alinea_setpoint is not None
                    else get_critical_density(onramp_cell)
                ),
                measurement_cell=3,
            )
            if ramp_control is True
            else None
        ),
    )

    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    return network


def mainline_demand_d(time: float) -> float:
    # stronger upstream demand that produces a bottleneck downstream
    return demand(time, 300 / 3600, 1800 / 3600, 3600 / 3600, 3500)


def onramp_demand_d(time: float, network_length: int) -> NDArray[np.float64]:
    # single onramp feeding into the middle of the network with a peak
    ramp_demands = np.zeros(network_length)
    # attach to the third cell (index 2) to interact with a downstream offramp
    ramp_demands[2] = demand(time, 300 / 3600, 1500 / 3600, 3600 / 3600, 2000)
    return ramp_demands


def setup_network_d(
    get_critical_density: Callable[[Cell], float],
    ramp_control: bool = False,
    alinea_gain: float = 5.0,
    alinea_setpoint: float | None = None,
) -> Network:
    """Create a network with a mid-network onramp and a downstream offramp.

    The layout is designed so the onramp merges upstream of an offramp
    which takes a non-negligible split of the mainline flow. This makes
    the effects of on-/off-ramps (local queues, flow reductions and
    recovery downstream) clearly visible in the results and plots.
    """

    network = Network()
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    # cell with onramp attached (third cell)
    onramp_cell = network.add_cell(
        length=0.5,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    onramp_cell.onramp = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=(
            AlineaController(
                gain=alinea_gain,
                # set the ALINEA setpoint to the critical density of the cell (fallback if no static setpoint is provided)
                setpoint=(
                    alinea_setpoint
                    if alinea_setpoint is not None
                    else get_critical_density(onramp_cell)
                ),
                measurement_cell=3,
            )
            if ramp_control is True
            else None
        ),
    )

    # downstream cells - attach an offramp to the 5th cell to create a split
    network.add_cell(
        length=0.5, lanes=2, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5,
        lanes=2,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        offramp=Offramp(
            lanes=1,
            split_ratio=0.2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        ),
    )
    network.add_cell(
        length=0.5, lanes=2, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    return network
