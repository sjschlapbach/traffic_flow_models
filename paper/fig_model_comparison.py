"""Fig. 3: CTM and METANET on scenario C without control.

(a), (b) space-time density maps on the true cell boundaries, one colour scale;
(c) flow in the lane-drop link with its capacity; (d) on-ramp demand and on-ramp flow;
(e) on-ramp queue. Run with ``python -m paper.fig_model_comparison``.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from matplotlib.lines import Line2D

from traffic_flow_models import MotorwayLink

from paper import style
from paper.outputs import (
    density_grid,
    lane_drop,
    positions,
    results,
    save_figure,
    series,
    time_minutes,
)
from paper.runs import ensure_runs

RERUN_SIMULATION: bool = False
RUN_IDS = {"CTM": "sim_c_none_ctm", "METANET": "sim_c_none_metanet"}
FIGURE_NAME = "fig_model_comparison"
FIGURE_HEIGHT_MM = 120
ONRAMP_ID = "onramp"
ONRAMP_ORIGIN_ID = "origin_onr"


def main() -> None:
    """Build Fig. 3 from the stored runs."""
    ensure_runs(list(RUN_IDS.values()), rerun=RERUN_SIMULATION)
    _, bottleneck = lane_drop(RUN_IDS["CTM"])
    with plt.rc_context(style.RC_PARAMS):
        fig = plt.figure(
            figsize=(style.TEXT_WIDTH_IN, FIGURE_HEIGHT_MM / style.MM_PER_INCH)
        )
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1.0])
        maps = top.subplots(1, 2, sharex=True, sharey=True)
        panels = bottom.subplots(1, 3, sharex=True)
        _draw_maps(top, list(maps), bottleneck)
        handles = _draw_series(list(panels), bottleneck)
        for ax, letter, name in zip([*maps, *panels], "abcde", [*RUN_IDS, "", "", ""]):
            style.panel_label(ax, letter, name)
        fig.legend(handles=handles, loc="outside lower center", ncols=3)
        save_figure(fig, FIGURE_NAME)
        plt.close(fig)


def _draw_maps(top: SubFigure, maps: list[Axes], bottleneck: MotorwayLink) -> None:
    """Draw the density maps of both models with one colour scale.

    Args:
        top: Subfigure holding the maps (for the colour bar).
        maps: Axes of panels (a) and (b).
        bottleneck: Lane-drop link.
    """
    position = positions(RUN_IDS["CTM"])
    span = (position[bottleneck.id], position[bottleneck.id] + bottleneck.length)
    grids = [density_grid(rid) for rid in RUN_IDS.values()]
    vmax = max(float(grid[2].max()) for grid in grids)
    meshes = [style.density_map(ax, grid, vmax) for ax, grid in zip(maps, grids)]
    for ax in maps:
        ax.axhline(
            position[ONRAMP_ID],
            color=style.MAP_MARKER_COLOUR,
            linewidth=0.6,
            linestyle="--",
        )
        for edge in span:
            ax.axhline(
                edge, color=style.MAP_MARKER_COLOUR, linewidth=0.6, linestyle=":"
            )
    maps[0].set_ylabel(style.POSITION_LABEL)
    style.margin_label(maps[1], position[ONRAMP_ID], "on-ramp")
    style.margin_label(maps[1], sum(span) / 2, "lane drop")
    # both maps share the colour scale
    top.colorbar(meshes[0], ax=maps, label=style.DENSITY_LABEL)


def _draw_series(panels: list[Axes], bottleneck: MotorwayLink) -> list[Line2D]:
    """Draw bottleneck flow, on-ramp demand and flow, and on-ramp queue.

    Args:
        panels: Axes of panels (c), (d) and (e).
        bottleneck: Lane-drop link.

    Returns:
        Legend handles: one line per model and the on-ramp demand.
    """
    props = results(RUN_IDS["CTM"])["metadata"]["link_properties"][bottleneck.id]
    handles: list[Line2D] = []
    for model, rid in RUN_IDS.items():
        line = style.MODEL_STYLES[model]
        t = time_minutes(rid)
        outflow = series(rid, "flows", bottleneck.id, props["num_cells"] - 1)
        handles += panels[0].plot(t, outflow, label=model, **line)
        panels[1].plot(t, series(rid, "flows", ONRAMP_ID), **line)
        panels[2].plot(t, series(rid, "onramp_queues", ONRAMP_ID), **line)
    demand = series(RUN_IDS["CTM"], "origin_demands", ONRAMP_ORIGIN_ID)
    handles += panels[1].plot(
        time_minutes(RUN_IDS["CTM"], len(demand)),
        demand,
        label="on-ramp demand",
        **style.DEMAND_STYLE,
    )
    style.reference_line(panels[0], bottleneck.Qc, r"$Q_c^m$")
    for ax, label in zip(
        panels, [style.FLOW_LABEL, style.FLOW_LABEL, style.QUEUE_LABEL]
    ):
        style.time_axis(ax)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0.0)
    return handles


if __name__ == "__main__":
    main()
