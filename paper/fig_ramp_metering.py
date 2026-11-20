"""Fig. 4: ramp metering on scenario E (three on-ramps) under METANET.

Top row: space-time density maps without control, with ALINEA at every on-ramp, with
METALINE and with ALINEA plus a maximum queue N_m^max, on one colour scale. Bottom
row: queue at the mainline origin and at each on-ramp, one line per variant, with
N_m^max as reference. Run with ``python -m paper.fig_ramp_metering``.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from matplotlib.lines import Line2D

from paper import style
from paper.outputs import density_grid, positions, save_figure, series, time_minutes
from paper.runs import MAX_QUEUE_VEH, RUNS, build_network, ensure_runs

RERUN_SIMULATION: bool = False
VARIANTS = {
    "no control": "sim_e_none_metanet",
    "ALINEA": "sim_e1_alinea_metanet",
    "METALINE": "sim_e2_metaline_metanet",
    r"ALINEA, $N_m^{\max}$": "sim_e1_alinea_nmax_metanet",
}
FIGURE_NAME = "fig_ramp_metering"
FIGURE_HEIGHT_MM = 105
ORIGIN_ID = "origin"  # mainline origin of scenario E


def main() -> None:
    """Build Fig. 4 from the stored runs."""
    ensure_runs(list(VARIANTS.values()), rerun=RERUN_SIMULATION)
    onramp_ids: list[str] = build_network(RUNS[VARIANTS["no control"]])[1]["onramp_ids"]
    with plt.rc_context(style.RC_PARAMS):
        fig = plt.figure(
            figsize=(style.TEXT_WIDTH_IN, FIGURE_HEIGHT_MM / style.MM_PER_INCH)
        )
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.1, 1.0])
        maps = top.subplots(1, len(VARIANTS), sharex=True, sharey=True)
        queues = bottom.subplots(1, 1 + len(onramp_ids), sharex=True, sharey=True)
        _draw_maps(top, list(maps), onramp_ids)
        handles = _draw_queues(list(queues), onramp_ids)
        fig.legend(handles=handles, loc="outside lower center", ncols=len(VARIANTS))
        save_figure(fig, FIGURE_NAME)
        plt.close(fig)


def _draw_maps(top: SubFigure, maps: list[Axes], onramp_ids: list[str]) -> None:
    """Draw the density maps of all variants with one colour scale.

    Args:
        top: Subfigure holding the maps (for the colour bar).
        maps: One axes per variant.
        onramp_ids: On-ramp ids in driving order.
    """
    grids = [density_grid(rid) for rid in VARIANTS.values()]
    vmax = max(float(grid[2].max()) for grid in grids)
    meshes = [style.density_map(ax, grid, vmax) for ax, grid in zip(maps, grids)]
    position = positions(VARIANTS["no control"])
    for ax, letter, label in zip(maps, "abcd", VARIANTS):
        style.panel_label(ax, letter, label)
        for onramp_id in onramp_ids:
            ax.axhline(
                position[onramp_id],
                color=style.MAP_MARKER_COLOUR,
                linewidth=0.6,
                linestyle="--",
            )
    for number, onramp_id in enumerate(onramp_ids, start=1):
        style.margin_label(maps[-1], position[onramp_id], f"on-ramp {number}")
    maps[0].set_ylabel(style.POSITION_LABEL)
    # all maps share the colour scale
    top.colorbar(meshes[0], ax=maps, label=style.DENSITY_LABEL)


def _draw_queues(queues: list[Axes], onramp_ids: list[str]) -> list[Line2D]:
    """Draw the queue at the mainline origin and at each on-ramp for all variants.

    Args:
        queues: Axes for the origin and each on-ramp.
        onramp_ids: On-ramp ids in driving order.

    Returns:
        Legend handles, one per variant.
    """
    handles: list[Line2D] = []
    for label, rid in VARIANTS.items():
        t = time_minutes(rid)
        line = {**style.CONTROLLER_STYLES[label], "markevery": style.MARK_EVERY}
        handles += queues[0].plot(
            t, series(rid, "origin_queues", ORIGIN_ID), label=label, **line
        )
        for ax, onramp_id in zip(queues[1:], onramp_ids):
            ax.plot(t, series(rid, "onramp_queues", onramp_id), **line)
    names = ["origin"] + [f"on-ramp {i}" for i in range(1, len(onramp_ids) + 1)]
    for ax, letter, name in zip(queues, "efgh", names):
        style.panel_label(ax, letter, name)
        style.time_axis(ax)
    for ax in queues[1:]:
        style.reference_line(ax, MAX_QUEUE_VEH, None)
    style.margin_label(queues[-1], MAX_QUEUE_VEH, r"$N_m^{\max}$")
    queues[0].set_ylabel(style.QUEUE_LABEL)
    queues[0].set_ylim(bottom=0.0)
    return handles


if __name__ == "__main__":
    main()
