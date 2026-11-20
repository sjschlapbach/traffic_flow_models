"""Access to stored runs for the figures and tables, and their writers.

Structure (links, on-ramps, lane drop) comes from the network objects the scenario
builders return; time series come from the library results JSON.
"""

import csv
import functools

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from traffic_flow_models import MotorwayLink, Onramp
from traffic_flow_models.types import JSONDict

from paper.runs import PAPER_DIR, RUNS, build_network, load_json, mainline

FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"


@functools.cache
def results(run_id: str) -> JSONDict:
    """Return the stored results JSON of a simulation run (cached, read-only)."""
    return load_json(run_id, "simulation_results.json")


def series(
    run_id: str, quantity: str, element_id: str, cell: int = 0
) -> NDArray[np.float64]:
    """Return one stored time series of a simulation run, as stored.

    Args:
        run_id: Simulation run id.
        quantity: Key of ``state_time_series`` (``flows``, ``densities``, ``speeds``,
            ``onramp_queues``, ...) or of ``disturbance_time_series``
            (``origin_demands``).
        element_id: Link, origin or on-ramp id.
        cell: Cell index for per-cell quantities.

    Returns:
        The samples (K + 1 for states, K for disturbances).
    """
    data = results(run_id)
    group = (
        "disturbance_time_series"
        if quantity in data["disturbance_time_series"]
        else "state_time_series"
    )
    values = np.asarray(data[group][quantity][element_id], dtype=np.float64)
    return values[:, cell] if values.ndim == 2 else values


def time_minutes(run_id: str, samples: int | None = None) -> NDArray[np.float64]:
    """Return the stored time array of a run in minutes (the first ``samples``)."""
    time_h = np.asarray(results(run_id)["time_array"], dtype=np.float64)
    return 60.0 * time_h[:samples]


def lane_drop(run_id: str) -> tuple[MotorwayLink, MotorwayLink]:
    """Return the mainline link upstream of the lane drop and the lane-drop link."""
    links = mainline(*build_network(RUNS[run_id])[:2])
    return next(
        (upstream, link)
        for upstream, link in zip(links, links[1:])
        if link.lanes < upstream.lanes
    )


def positions(run_id: str) -> dict[str, float]:
    """Return the start of every mainline link and the merge of every on-ramp [km].

    Args:
        run_id: Run id.

    Returns:
        ``{link id: distance from the start of the mainline}``.
    """
    net, metadata, _ = build_network(RUNS[run_id])
    links = mainline(net, metadata)
    starts = np.cumsum([0.0] + [link.length for link in links[:-1]])
    position = {link.id: float(start) for link, start in zip(links, starts)}
    for node in net.list_nodes():
        for link in node.incoming:
            if isinstance(link, Onramp):
                position[link.id] = position[node.outgoing[0].id]
    return position


def density_grid(
    run_id: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the stored mainline densities on their true cell boundaries.

    Each stored sample k is drawn over [t_k - dt/2, t_k + dt/2].

    Args:
        run_id: Simulation run id.

    Returns:
        Time edges [min], cell edges [km] and densities (cells x samples).
    """
    link_ids = [link.id for link in mainline(*build_network(RUNS[run_id])[:2])]
    data = results(run_id)
    props = data["metadata"]["link_properties"]
    lengths = np.concatenate([props[lid]["cell_lengths"] for lid in link_ids])
    x_edges = np.concatenate([[0.0], np.cumsum(lengths)])
    density = np.hstack(
        [np.asarray(data["state_time_series"]["densities"][lid]) for lid in link_ids]
    )
    t = time_minutes(run_id)
    half_step = (t[1] - t[0]) / 2
    t_edges = np.concatenate([t - half_step, [t[-1] + half_step]])
    return t_edges, x_edges, density.T


def save_figure(fig: Figure, name: str) -> None:
    """Save a figure as vector PDF (no creation date) and PNG preview."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES_DIR / f"{name}.png")


def save_table(name: str, rows: list[list[object]], tex: str) -> None:
    """Save a table as CSV (first row the header, full precision) and TeX fragment."""
    TABLES_DIR.mkdir(exist_ok=True)
    with open(TABLES_DIR / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    (TABLES_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
