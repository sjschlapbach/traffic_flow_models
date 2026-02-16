"""Demo script: build a motorway link and plot it using the plotting utilities.

Usage: run this module as a script from the repository root:

    python -m src.demo.plot_demo_network

"""

import os
import argparse
from traffic_flow_models import (
    MotorwayLink,
    Onramp,
    Offramp,
    Destination,
    Origin,
    Node,
    Network,
)


def build_and_plot_network():
    # create a compact network containing motorway, onramp and offramp
    m1 = MotorwayLink(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m2 = MotorwayLink(
        length=1.0, lanes=2, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m3 = MotorwayLink(
        length=0.5, lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    origin = Origin()
    destination_main = Destination(id=None)
    destination_off = Destination(id=None)

    onr = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )

    offr = Offramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        destination=destination_off,
    )

    n0 = Node(incoming=[origin], outgoing=[m1])
    n0.position = (0.0, 0.0)

    n1 = Node(incoming=[m1, onr], outgoing=[m2])
    n1.position = (1.0, 0.0)

    n2 = Node(incoming=[m2], outgoing=[m3, offr])
    n2.position = (2.0, 0.0)

    n3 = Node(incoming=[m3], outgoing=[destination_main])
    n3.position = (2.5, 0.0)

    net = Network(nodes=[n0, n1, n2, n3])

    return net


if __name__ == "__main__":
    # check if plotting is disabled through command line argument (CI environment)
    parser = argparse.ArgumentParser(description="Plot Demo Network")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting for CI/automated runs",
    )
    args = parser.parse_args()
    plot_enabled = not args.no_plot

    net = build_and_plot_network()
    out_dir = os.path.join("src/demo/results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_network.png")
    net.plot(show=plot_enabled, save_path=out_path)
