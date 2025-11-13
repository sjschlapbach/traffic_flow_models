from traffic_flow_models import Network, Onramp, AlineaController, CTM
import numpy as np
from numpy.typing import NDArray

from .demand import demand


def mainline_demand_a(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 4000)


def onramp_demand_a(time: float, network_length: int) -> NDArray[np.float64]:
    ramp_demands = np.zeros(network_length)
    ramp_demands[2] = demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 2000)
    return ramp_demands


def setup_network_a(ramp_control: bool = False) -> Network:
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
                AlineaController(gain=10.0, setpoint=20.0, measurement_cell=3)
                if ramp_control is True
                else None
            ),
        ),
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
    )
    network.add_cell(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    return network
