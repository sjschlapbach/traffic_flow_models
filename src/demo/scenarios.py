from traffic_flow_models import Network, Onramp, AlineaController, CTM
import numpy as np
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
    ramp_control: bool = False, alinea_gain: float = 5.0, alinea_setpoint: float = 20.0
) -> Network:
    network = Network()
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        onramp=Onramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            controller=(
                AlineaController(
                    gain=alinea_gain, setpoint=alinea_setpoint, measurement_cell=3
                )
                if ramp_control is True
                else None
            ),
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
    ramp_control: bool = False, alinea_gain: float = 5.0, alinea_setpoint: float = 20.0
) -> Network:
    network = Network()
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    network.add_cell(
        length=0.5,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        onramp=Onramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            controller=(
                AlineaController(
                    gain=alinea_gain, setpoint=alinea_setpoint, measurement_cell=3
                )
                if ramp_control is True
                else None
            ),
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
