"""Table 3: METANET calibration results on scenarios C and C2 (C with ALINEA).

Per scenario and condition: estimate and relative error of each parameter against the
true value (tau in s), final cost, number of function evaluations and termination
status, all from ``calibration.json`` (``Calibrator.calibrate_model_params``). Run with
``python -m paper.table_calibration``.
"""

from paper.fig_calibration import PARAMETER_SYMBOLS
from paper.outputs import save_table
from paper.runs import ensure_runs, load_json

RERUN_SIMULATION: bool = False
TABLE_NAME = "table_calibration"
SECONDS_PER_HOUR = 3600.0
CALIBRATION_RUNS = {
    # paper names: scenario 1 is demo scenario C, "1 + ALINEA" its variant C2
    ("1", "exact"): "cal_c_exact",
    ("1", "noisy"): "cal_c_noisy",
    ("1", "noisy_reg"): "cal_c_noisy_reg",
    ("1 + ALINEA", "exact"): "cal_c2_exact",
    ("1 + ALINEA", "noisy"): "cal_c2_noisy",
    ("1 + ALINEA", "noisy_reg"): "cal_c2_noisy_reg",
}
CONDITION_LABELS = {
    "exact": "exact data",
    "noisy": "noisy data",
    "noisy_reg": r"noisy data, $\lambda = 0.1$",
}
UNITS = {
    "tau": "s",
    "nu": "km$^2$/h",
    "kappa": "veh/km/lane",
    "delta": "-",
    "phi": "-",
    "alpha": "-",
}


def main() -> None:
    """Build Table 3 from the stored runs."""
    ensure_runs(list(CALIBRATION_RUNS.values()), rerun=RERUN_SIMULATION)
    first = load_json(CALIBRATION_RUNS[("1", "exact")], "calibration.json")
    names: list[str] = first["param_names"]
    header: list[object] = [
        "scenario",
        "condition",
        "parameter",
        "unit",
        "true",
        "start",
        "estimate",
        "relative_error_percent",
        "cost",
        "nfev",
        "status",
        "message",
    ]
    rows = [header]
    tex = [
        r"\begin{tabular}{ll" + " r" * len(names) + " r r r}",
        r"\toprule",
        " & ".join(
            ["Scenario", "Data"]
            + [PARAMETER_SYMBOLS[name] for name in names]
            + ["Cost", r"$n_\mathrm{fev}$", "Status"]
        )
        + r" \\",
        " & ".join(["", ""] + [f"({UNITS[name]})" for name in names] + ["", "", ""])
        + r" \\",
        r"\midrule",
    ]
    for label, key in [("true", "true_params"), ("start", "initial_params")]:
        values = [f"{_display(name, first[key][name]):.4g}" for name in names]
        tex.append(rf"\multicolumn{{2}}{{l}}{{{label}}} & " + " & ".join(values))
        tex[-1] += r" & & & \\"
    tex.append(r"\midrule")

    for (scenario, condition), rid in CALIBRATION_RUNS.items():
        run_rows, line = _run_rows(scenario, condition, rid, names)
        rows += run_rows
        tex.append(line)
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    save_table(TABLE_NAME, rows, "\n".join(tex))


def _run_rows(
    scenario: str, condition: str, rid: str, names: list[str]
) -> tuple[list[list[object]], str]:
    """Return the CSV rows (one per parameter) and the TeX row of a calibration run."""
    cal = load_json(rid, "calibration.json")
    rows: list[list[object]] = []
    cells = []
    for name in names:
        true, start, estimate = (
            _display(name, cal[key][name])
            for key in ("true_params", "initial_params", "estimates")
        )
        error = 100.0 * abs(estimate - true) / true
        rows.append([scenario, condition, name, UNITS[name], true, start, estimate])
        rows[-1] += [error, cal["cost"], cal["nfev"], cal["status"], cal["message"]]
        cells.append(f"{estimate:.4g} ({error:.1f})")
    fit = [_tex_number(cal["cost"]), str(cal["nfev"]), str(cal["status"])]
    line = " & ".join([scenario, CONDITION_LABELS[condition], *cells, *fit]) + r" \\"
    return rows, line


def _display(name: str, value: float) -> float:
    """Return a parameter value in the unit of the table (tau in seconds)."""
    return value * SECONDS_PER_HOUR if name == "tau" else value


def _tex_number(value: float) -> str:
    """Format a number with three significant digits, powers of ten in math mode."""
    text = f"{value:.3g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


if __name__ == "__main__":
    main()
