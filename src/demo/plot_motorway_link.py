"""Demo script: build a motorway link and plot it using the plotting utilities.

Usage: run this module as a script from the repository root:

    python -m src.demo.plot_motorway_link

"""

import os
import argparse
from traffic_flow_models.network import MotorwayLink


def build_sample_motorway_link() -> MotorwayLink:
    link = MotorwayLink()

    # add mainline cells with varying lanes
    link.add_cell(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=150
    )
    link.add_cell(
        length=1.2, lanes=2, lane_capacity=2000, free_flow_speed=90, jam_density=150
    )
    link.add_cell(
        length=0.8, lanes=2, lane_capacity=2000, free_flow_speed=80, jam_density=150
    )
    link.add_cell(
        length=1.5, lanes=4, lane_capacity=2000, free_flow_speed=110, jam_density=150
    )

    # attach on- and offramps to some cell
    link.add_onramp(0, lanes=1, lane_capacity=1000, free_flow_speed=70, jam_density=150)
    link.add_onramp(2, lanes=2, lane_capacity=1000, free_flow_speed=70, jam_density=150)
    link.add_offramp(
        2,
        lanes=1,
        lane_capacity=1000,
        free_flow_speed=70,
        jam_density=150,
        split_ratio=0.15,
    )

    return link


if __name__ == "__main__":
    # check if plotting is disabled through command line argument (CI environment)
    parser = argparse.ArgumentParser(description="CTM Simulation Demo")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting for CI/automated runs",
    )
    args = parser.parse_args()
    plot_enabled = not args.no_plot

    link = build_sample_motorway_link()
    out_dir = os.path.join("src/demo/results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_network.png")
    link.plot(show=plot_enabled, save_path=out_path)
