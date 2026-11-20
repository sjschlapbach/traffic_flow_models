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
SCENARIO_LABELS = {"1": "Scenario 1", "1 + ALINEA": "Scenario 1 with ALINEA"}
CONDITION_LABELS = {
    "exact": "exact",
    "noisy": "noisy",
    "noisy_reg": r"noisy, $\lambda = 0.1$",
}
UNITS = {
    "tau": "s",
    "nu": "km$^2$/h",
    "kappa": "veh/km/lane",
    "delta": "-",
    "phi": "-",
    "alpha": "-",
}
# decimals per parameter in the TeX table, enough to show the true and start values exactly
DECIMALS = {"tau": 2, "nu": 2, "kappa": 2, "delta": 4, "phi": 2, "alpha": 3}


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
    columns = len(names) + 2
    tex = [
        r"\fontsize{10.5}{12.6}\selectfont",
        r"\begin{tabular}{@{}l" + " c" * (columns - 1) + "@{}}",
        r"\toprule",
        " & ".join(["", *(PARAMETER_SYMBOLS[name] for name in names), "Evaluations"])
        + r" \\",
        " & ".join(["Data", *(_unit(name) for name in names), ""]) + r" \\",
        r"\midrule",
    ]
    for label, key in [
        ("true values", "true_params"),
        ("start values", "initial_params"),
    ]:
        values = [_number(name, _display(name, first[key][name])) for name in names]
        tex.append(" & ".join([label, *values, ""]) + r" \\")

    for (scenario, condition), rid in CALIBRATION_RUNS.items():
        if condition == "exact":
            tex.append(r"\midrule")
            tex.append(
                rf"\multicolumn{{{columns}}}{{@{{}}l}}"
                rf"{{\textit{{{SCENARIO_LABELS[scenario]}}}}} \\"
            )
        run_rows, lines = _run_rows(scenario, condition, rid, names)
        rows += run_rows
        tex += lines
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    save_table(TABLE_NAME, rows, "\n".join(tex))


def _run_rows(
    scenario: str, condition: str, rid: str, names: list[str]
) -> tuple[list[list[object]], list[str]]:
    """Return the CSV rows (one per parameter) and the two TeX lines of a run.

    The first TeX line holds the estimates and the number of evaluations, the second
    the relative errors in %.
    """
    cal = load_json(rid, "calibration.json")
    rows: list[list[object]] = []
    estimates = []
    errors = []
    for name in names:
        true, start, estimate = (
            _display(name, cal[key][name])
            for key in ("true_params", "initial_params", "estimates")
        )
        error = 100.0 * abs(estimate - true) / true
        rows.append([scenario, condition, name, UNITS[name], true, start, estimate])
        rows[-1] += [error, cal["cost"], cal["nfev"], cal["status"], cal["message"]]
        estimates.append(_number(name, estimate))
        errors.append(rf"{{\footnotesize({error:.1f})}}")
    label = rf"\quad {CONDITION_LABELS[condition]}"
    lines = [
        " & ".join([label, *estimates, str(cal["nfev"])]) + r" \\",
        " & ".join(["", *errors, ""]) + r" \\[2pt]",
    ]
    return rows, lines


def _number(name: str, value: float) -> str:
    """Return a parameter value with the fixed number of decimals of its column."""
    return f"{value:.{DECIMALS[name]}f}"


def _unit(name: str) -> str:
    """Return the TeX unit cell of a parameter, empty if it is dimensionless."""
    return "" if UNITS[name] == "-" else f"({UNITS[name]})"


def _display(name: str, value: float) -> float:
    """Return a parameter value in the unit of the table (tau in seconds)."""
    return value * SECONDS_PER_HOUR if name == "tau" else value


if __name__ == "__main__":
    main()
