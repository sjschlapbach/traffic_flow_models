"""Demo script: build a small network and plot it using the network.plot utilities.

Usage: run this module as a script from the repository root:

    python -m src.demo.plot_network

"""

import os

from traffic_flow_models.network import Network


def build_sample_network() -> Network:
    net = Network()

    # add mainline cells with varying lanes
    net.add_cell(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=150
    )
    net.add_cell(
        length=1.2, lanes=2, lane_capacity=2000, free_flow_speed=90, jam_density=150
    )
    net.add_cell(
        length=0.8, lanes=2, lane_capacity=2000, free_flow_speed=80, jam_density=150
    )
    net.add_cell(
        length=1.5, lanes=4, lane_capacity=2000, free_flow_speed=110, jam_density=150
    )

    # attach on- and offramps to some cell
    net.add_onramp(0, lanes=1, lane_capacity=1000, free_flow_speed=70, jam_density=150)
    net.add_onramp(2, lanes=2, lane_capacity=1000, free_flow_speed=70, jam_density=150)
    net.add_offramp(
        2,
        lanes=1,
        lane_capacity=1000,
        free_flow_speed=70,
        jam_density=150,
        split_ratio=0.15,
    )

    return net


if __name__ == "__main__":
    network = build_sample_network()
    out_dir = os.path.join("src/demo/results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_network.png")
    network.plot(show=True, save_path=out_path)
