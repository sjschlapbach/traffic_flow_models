"""Fig. 5: METANET calibration with noisy measurements.

(a) speed and (b) density in the last cell upstream of the lane drop: noisy
measurements, noise-free reference, simulation with the start values and with the
calibrated parameters; (c) parameter correlation matrix from the Jacobian at the
solution. One figure for scenario C (Fig. 5) and one of the same form for scenario C2
(C with ALINEA), which can replace it. Run with ``python -m paper.fig_calibration``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from paper import style
from paper.outputs import lane_drop, results, save_figure, series, time_minutes
from paper.runs import ensure_runs, load_array, load_json

RERUN_SIMULATION: bool = False
# runs of each figure: truth, calibration, simulation with start values and with the
# calibrated parameters
FIGURES = {
    "fig_calibration": {
        "truth": "sim_c_none_metanet",
        "calibration": "cal_c_noisy",
        "start": "sim_c_start_metanet",
        "calibrated": "sim_c_calibrated_metanet",
    },
    "fig_calibration_alinea": {
        "truth": "sim_c2_alinea_metanet",
        "calibration": "cal_c2_noisy",
        "start": "sim_c2_start_metanet",
        "calibrated": "sim_c2_calibrated_metanet",
    },
}
FIGURE_HEIGHT_MM = 62
PARAMETER_SYMBOLS = {
    "tau": r"$\tau$",
    "nu": r"$\nu$",
    "kappa": r"$\kappa$",
    "delta": r"$\delta$",
    "phi": r"$\phi$",
    "alpha": r"$\alpha$",
}
REFERENCE_LINEWIDTH = 0.6
ANNOTATION_SWITCH = 0.5  # |correlation| above which annotations are white


def correlation_matrix(run_id: str) -> NDArray[np.float64]:
    """Return the parameter correlation matrix of a calibration run.

    Same computation as ``Calibrator.analyze_parameter_correlation``, which prints
    and plots the matrix but does not return it: Cov = cost / max(1, m - n) (J^T J)^-1
    and Corr = Cov / (sigma sigma^T), with J the stored Jacobian at the solution.

    Args:
        run_id: Calibration run id.

    Returns:
        Correlation matrix.

    Raises:
        numpy.linalg.LinAlgError: If J^T J is singular (no other measure is used).
    """
    jac = load_array(run_id, "jacobian.npy")
    cost = load_json(run_id, "calibration.json")["cost"]
    n_residuals, n_params = jac.shape
    cov = cost / max(1, n_residuals - n_params) * np.linalg.inv(jac.T @ jac)
    std = np.sqrt(np.diag(cov))
    corr: NDArray[np.float64] = cov / np.outer(std, std)
    return corr


def main() -> None:
    """Build both calibration figures from the stored runs."""
    ensure_runs(
        [rid for runs in FIGURES.values() for rid in runs.values()],
        rerun=RERUN_SIMULATION,
    )
    for name, runs in FIGURES.items():
        _figure(name, runs)


def _figure(name: str, runs: dict[str, str]) -> None:
    """Draw and save one calibration figure.

    Args:
        name: File name without extension.
        runs: Run ids of the truth, the calibration, the simulation with the start
            values and the simulation with the calibrated parameters.
    """
    # last cell upstream of the lane drop
    upstream = lane_drop(runs["truth"])[0].id
    props = results(runs["truth"])["metadata"]["link_properties"][upstream]
    names: list[str] = load_json(runs["calibration"], "calibration.json")["param_names"]
    with plt.rc_context(style.RC_PARAMS):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(style.TEXT_WIDTH_IN, FIGURE_HEIGHT_MM / style.MM_PER_INCH),
            width_ratios=[1.0, 1.0, 1.35],
        )
        for ax, quantity, label in [
            (axes[0], "speeds", style.SPEED_LABEL),
            (axes[1], "densities", style.DENSITY_LABEL),
        ]:
            _draw_time_series(ax, runs, quantity, upstream, props["num_cells"] - 1)
            ax.set_ylabel(label)
        image = _correlation_panel(
            axes[2], correlation_matrix(runs["calibration"]), names
        )
        fig.colorbar(image, ax=axes[2], label="Correlation")
        for ax, letter in zip(axes, "abc"):
            style.panel_label(ax, letter)
        fig.legend(
            handles=axes[0].get_lines(),
            loc="outside lower center",
            ncols=4,
            markerscale=4,
        )
        save_figure(fig, name)
        plt.close(fig)


def _draw_time_series(
    ax: Axes, runs: dict[str, str], quantity: str, link_id: str, cell: int
) -> None:
    """Draw measurements, reference, start-value and calibrated simulation of a cell.

    Args:
        ax: Axes to draw in.
        runs: Run ids of the figure (see ``FIGURES``).
        quantity: ``"speeds"`` or ``"densities"``.
        link_id: Motorway link id.
        cell: Cell index.
    """
    measurements = load_json(runs["calibration"], "measurements.json")[
        "state_time_series"
    ]
    colour = style.MODEL_STYLES["METANET"]["color"]
    ax.plot(
        time_minutes(runs["truth"]),
        np.asarray(measurements[quantity][link_id])[:, cell],
        label="noisy measurements",
        **style.MEASUREMENT_STYLE,
    )
    ax.plot(
        time_minutes(runs["truth"]),
        series(runs["truth"], quantity, link_id, cell),
        color="black",
        linewidth=REFERENCE_LINEWIDTH,
        label="noise-free reference",
        # on top of the calibrated simulation, which it would hide otherwise
        zorder=3,
    )
    for key, label, linestyle in [
        ("start", "start values", ":"),
        ("calibrated", "calibrated", "-"),
    ]:
        ax.plot(
            time_minutes(runs[key]),
            series(runs[key], quantity, link_id, cell),
            color=colour,
            linestyle=linestyle,
            label=label,
        )
    style.time_axis(ax)
    ax.set_ylim(bottom=0.0)


def _correlation_panel(
    ax: Axes, corr: NDArray[np.float64], names: list[str]
) -> AxesImage:
    """Draw an annotated correlation matrix with a diverging map centred at 0."""
    image = ax.imshow(corr, cmap=style.CORRELATION_COLORMAP, vmin=-1.0, vmax=1.0)
    symbols = [PARAMETER_SYMBOLS[name] for name in names]
    ax.set_xticks(range(len(names)), symbols)
    ax.set_yticks(range(len(names)), symbols)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    for (i, j), value in np.ndenumerate(corr):
        ax.text(
            j,
            i,
            # math minus as on the colour bar; no leading zero so that neighbouring
            # negative values do not touch
            ("$-$" if value < 0 else "") + f"{abs(value):.2f}".replace("0.", "."),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > ANNOTATION_SWITCH else "black",
        )
    return image


if __name__ == "__main__":
    main()
