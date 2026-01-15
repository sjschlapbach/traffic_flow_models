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


def build_simple_network():
    """Build a simple linear network with motorway, onramp and offramp."""
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
    n1 = Node(incoming=[m1, onr], outgoing=[m2])
    n2 = Node(incoming=[m2], outgoing=[m3, offr])
    n3 = Node(incoming=[m3], outgoing=[destination_main])

    net = Network(nodes=[n0, n1, n2, n3])

    return net


def build_circular_network():
    """Build a circular network with a loop formed by link2, link3, and link4."""
    # Create motorway links
    link1 = MotorwayLink(
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        id="link1",
    )
    link2 = MotorwayLink(
        length=1.2,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        id="link2",
    )
    link3 = MotorwayLink(
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        id="link3",
    )
    link4 = MotorwayLink(
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        id="link4",
    )
    link5 = MotorwayLink(
        length=0.8,
        lanes=2,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        id="link5",
    )

    # Create origin and destinations
    origin = Origin(id="main_origin")
    destination_main = Destination(id="main_destination")
    destination_off1 = Destination(id="offramp_dest_1")
    destination_off2 = Destination(id="offramp_dest_2")

    # Create onramp and offramps
    onramp1 = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
        id="onramp_1",
    )

    offramp1 = Offramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        destination=destination_off1,
        id="offramp_1",
    )

    offramp2 = Offramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        destination=destination_off2,
        id="offramp_2",
    )

    # Build network structure:
    # Origin -> link1 -> Junction (splits to link2 and link5)
    # link2 -> Junction -> link3 (in loop)
    # link3 -> Junction (with offramp1) -> link4 (in loop)
    # link4 -> Junction (with onramp1) -> back to link2 (completing loop)
    # link5 -> Junction -> Destination

    n0 = Node(incoming=[origin], outgoing=[link1])
    n1 = Node(incoming=[link1], outgoing=[link2, link5])  # Split to loop and exit
    n2 = Node(incoming=[link2], outgoing=[link3])  # In loop
    n3 = Node(incoming=[link3], outgoing=[link4, offramp1])  # In loop with offramp
    n4 = Node(incoming=[link4, onramp1], outgoing=[link2])  # Complete loop with onramp
    n5 = Node(incoming=[link5], outgoing=[destination_main])  # Exit path

    net = Network(nodes=[n0, n1, n2, n3, n4, n5])

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

    out_dir = os.path.join("src/demo/results")
    os.makedirs(out_dir, exist_ok=True)

    # Plot simple linear network
    print("Plotting simple linear network...")
    net_simple = build_simple_network()
    out_path_simple = os.path.join(out_dir, "demo_network_simple.png")
    net_simple.plot(show=plot_enabled, save_path=out_path_simple, figsize=(24, 16))
    print(f"  Saved to {out_path_simple}")

    # Plot circular network with loop
    print("Plotting circular network with loop...")
    net_circular = build_circular_network()
    out_path_circular = os.path.join(out_dir, "demo_network_circular.png")
    net_circular.plot(show=plot_enabled, save_path=out_path_circular, figsize=(28, 20))
    print(f"  Saved to {out_path_circular}")

    print("\nBoth networks plotted successfully!")
