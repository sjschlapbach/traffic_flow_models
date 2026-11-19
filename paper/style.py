"""Figure style shared by all paper figures.

Figures are created at their final size (full text width of an A4 page with 1 inch
margins) with 9 pt text in Computer Modern, the font of the paper. The fonts ship with
matplotlib, so no LaTeX installation is needed. Thin dark-grey axes with inward ticks
and a light horizontal grid replace matplotlib's default look. Colours are
the Okabe-Ito colour-blind-safe palette with one fixed mapping for the whole paper;
every series also differs by line style or marker, so the figures work in greyscale.
"""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

MM_PER_INCH = 25.4
TEXT_WIDTH_IN = 159 / MM_PER_INCH

RC_PARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    # cmr10 has no unicode minus glyph
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 2.5,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    # time axes end at the first and last stored sample
    "axes.xmargin": 0.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "figure.constrained_layout.use": True,
    # embed TrueType fonts (no Type 3)
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}

# ! categorical colours (Okabe-Ito)
MODEL_STYLES: dict[str, dict[str, Any]] = {
    "CTM": {"color": "#0072B2", "linestyle": "-"},
    "METANET": {"color": "#D55E00", "linestyle": "--"},
}

# hue encodes the control law; markers separate the control laws in greyscale
CONTROLLER_STYLES: dict[str, dict[str, Any]] = {
    "no control": {"color": "#000000", "linestyle": "-"},
    "ALINEA": {"color": "#009E73", "linestyle": "-", "marker": "o"},
    "METALINE": {"color": "#CC79A7", "linestyle": "-", "marker": "^"},
    r"ALINEA, $N^\mathrm{max}$": {"color": "#E69F00", "linestyle": "-", "marker": "D"},
}
MARK_EVERY = 30  # one marker per 30 stored samples (5 min at dt = 10 s)

MEASUREMENT_STYLE: dict[str, Any] = {
    "color": "#999999",
    "linestyle": "none",
    "marker": ".",
    "markersize": 1.5,
}
REFERENCE_COLOUR = "#404040"  # reference lines (capacity, set point, N^max)
DEMAND_STYLE: dict[str, Any] = {"color": REFERENCE_COLOUR, "linewidth": 0.6}
DENSITY_COLORMAP = "viridis"
MAP_MARKER_COLOUR = "white"  # marked positions and labels inside density maps
CORRELATION_COLORMAP = "RdBu_r"

# ! axis labels (notation of the paper)
TIME_LABEL = "Time (min)"
POSITION_LABEL = "Position (km)"
FLOW_LABEL = r"Flow $q$ (veh/h)"
DENSITY_LABEL = r"Density $\rho$ (veh/km/lane)"
SPEED_LABEL = r"Speed $v$ (km/h)"
QUEUE_LABEL = r"Queue $N$ (veh)"

TIME_TICK_MIN = 30.0


def panel_label(ax: Axes, label: str, name: str = "") -> None:
    """Write a panel label such as ``(a)`` or ``(a) ALINEA`` above an axes.

    Args:
        ax: Axes of the panel.
        label: Panel letter without parentheses.
        name: Optional name of what the panel shows.
    """
    ax.text(
        0.0,
        1.02,
        f"({label}) {name}".rstrip(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )


def reference_line(ax: Axes, value: float, label: str | None) -> None:
    """Draw a horizontal reference line, labelled directly right of the axes.

    Args:
        ax: Axes to draw in.
        value: Position on the y axis.
        label: Direct label, or ``None`` for an unlabelled line.
    """
    ax.axhline(value, color=REFERENCE_COLOUR, linewidth=0.8, linestyle=":")
    if label is not None:
        margin_label(ax, value, label)


def margin_label(ax: Axes, value: float, label: str) -> None:
    """Write a direct label in the right margin of an axes, at a y value.

    Args:
        ax: Axes to label.
        value: Position on the y axis.
        label: Label text.
    """
    ax.annotate(
        label,
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(2, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=REFERENCE_COLOUR,
    )


def density_map(
    ax: Axes,
    grid: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    vmax: float,
) -> QuadMesh:
    """Draw a space-time density map on the true cell boundaries.

    Args:
        ax: Axes to draw in.
        grid: Time edges [min], cell edges [km] and densities (cells x samples).
        vmax: Upper limit of the colour scale, shared by compared maps.

    Returns:
        The mesh (for the colour bar).
    """
    t_edges, x_edges, density = grid
    mesh = ax.pcolormesh(
        t_edges,
        x_edges,
        density,
        cmap=DENSITY_COLORMAP,
        vmin=0.0,
        vmax=vmax,
        shading="flat",
        edgecolors="face",
        linewidth=0,
    )
    ax.grid(False)
    # cell boundaries as minor ticks
    ax.set_yticks(x_edges, minor=True)
    ax.set_ylim(x_edges[0], x_edges[-1])
    time_axis(ax)
    return mesh


def time_axis(ax: Axes) -> None:
    """Label a time axis [min] with ticks every ``TIME_TICK_MIN`` minutes."""
    ax.set_xlabel(TIME_LABEL)
    ax.xaxis.set_major_locator(MultipleLocator(TIME_TICK_MIN))
