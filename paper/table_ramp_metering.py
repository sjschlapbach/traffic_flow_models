"""Table 2: ramp metering on scenario E (three on-ramps) under METANET.

Per variant: total time spent (VHT including queues) and its change against no
control, VHT on the motorway, time spent in queues, mean speed on the motorway (all
``Simulation.compute_metrics``), peak mainline density and maximum on-ramp and origin
queues (stored states). The ALINEA variants with a control interval T_c appear only
here. Run with ``python -m paper.table_ramp_metering``.
"""

import numpy as np

from traffic_flow_models import AlineaController

from paper.fig_ramp_metering import VARIANTS
from paper.outputs import results, save_table
from paper.runs import RUNS, build_network, ensure_runs, load_json, onramps

RERUN_SIMULATION: bool = False
TABLE_NAME = "table_ramp_metering"
# ALINEA with a control interval: demo gain, and gain retuned for the interval
CONTROL_INTERVAL_RUNS = {
    "sim_e1_alinea_tc_metanet": "",
    "sim_e1_alinea_tc_gain_metanet": " (retuned)",
}
RETUNED_RUN = "sim_e1_alinea_tc_gain_metanet"
BASELINE = "no control"

# (CSV column, TeX number format)
COLUMNS = [
    ("total_time_spent_veh_h", "{:.1f}"),
    ("tts_change_percent", "${:+.1f}$"),
    ("motorway_vht_veh_h", "{:.1f}"),
    ("queue_time_veh_h", "{:.1f}"),
    ("mean_speed_km_h", "{:.1f}"),
    ("max_density_veh_km_lane", "{:.1f}"),
    ("max_onramp_queue_veh", "{:.1f}"),
    ("max_origin_queue_veh", "{:.1f}"),
]
TEX_HEADER = [
    r"\fontsize{10.5}{12.6}\selectfont",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}l c c c c c c c c@{}}",
    r"\toprule",
    r" & \multicolumn{2}{c}{TTS} & VHT & Queue & $\bar v$ & $\max \rho$ & "
    r"\multicolumn{2}{c}{$\max N_m$ (veh)} \\",
    r"\cmidrule(lr){2-3}\cmidrule(l){8-9}",
    r"Variant & (veh\,h) & (\%) & (veh\,h) & (veh\,h) & (km/h) & (veh/km/lane) & "
    r"ramps & origin \\",
    r"\midrule",
]


def main() -> None:
    """Build Table 2 from the stored runs."""
    ensure_runs([*VARIANTS.values(), *CONTROL_INTERVAL_RUNS], rerun=RERUN_SIMULATION)
    rows_by_label = {
        **VARIANTS,
        **{_label(rid) + suffix: rid for rid, suffix in CONTROL_INTERVAL_RUNS.items()},
    }
    baseline = _values(VARIANTS[BASELINE])[0]
    values = {}
    for label, rid in rows_by_label.items():
        tts, *rest = _values(rid)
        values[label] = [tts, 100.0 * (tts - baseline) / baseline, *rest]
    rows: list[list[object]] = [["variant", "run", *[c[0] for c in COLUMNS]]]
    rows += [[label, rows_by_label[label], *row] for label, row in values.items()]

    # TeX order: the ALINEA variants as indented sub-rows of ALINEA, then METALINE
    tex_labels = {
        BASELINE: BASELINE,
        "ALINEA": "ALINEA",
        r"ALINEA, $N_m^{\max}$": r"\quad with $N_m^{\max}$",
        **{
            _label(rid)
            + suffix: rf"\quad with $T_c$, $K_I' = {_controller(rid)[0].gain:g}$"
            + (r"\textsuperscript{a}" if rid == RETUNED_RUN else "")
            for rid, suffix in CONTROL_INTERVAL_RUNS.items()
        },
        "METALINE": "METALINE",
    }
    tex = list(TEX_HEADER)
    for label, tex_label in tex_labels.items():
        cells = [c[1].format(value) for c, value in zip(COLUMNS, values[label])]
        if label == BASELINE:
            cells[1] = "--"
        tex.append(" & ".join([tex_label, *cells]) + r" \\")
    control_interval_s = _controller(RETUNED_RUN)[1]
    tex += [
        r"\bottomrule",
        rf"\multicolumn{{{len(COLUMNS) + 1}}}{{@{{}}l}}{{\footnotesize "
        rf"\textsuperscript{{a}}gain retuned for $T_c = {control_interval_s:g}$\,s}} \\",
        r"\end{tabular}",
        "",
    ]
    save_table(TABLE_NAME, rows, "\n".join(tex))


def _controller(rid: str) -> tuple[AlineaController, float]:
    """Return the ALINEA controller of a run and its control interval in s."""
    controller = onramps(*build_network(RUNS[rid])[:2])[0].controller
    if not isinstance(controller, AlineaController) or controller.control_dt is None:
        raise TypeError(f"run {rid} needs ALINEA with a control interval")
    return controller, controller.control_dt * 3600


def _label(rid: str) -> str:
    """Return the CSV row label of an ALINEA run with a control interval."""
    controller, control_interval_s = _controller(rid)
    return rf"ALINEA, $T_c = {control_interval_s:g}$ s, $K_I' = {controller.gain:g}$"


def _values(rid: str) -> list[float]:
    """Return TTS, VHT, queue time, mean speed, peak density and maximum queues."""
    metrics = load_json(rid, "metrics.json")
    states = results(rid)["state_time_series"]
    tts = metrics["with_queues"]["vht_veh_h"]
    vht = metrics["without_queues"]["vht_veh_h"]
    return [
        tts,
        vht,
        tts - vht,
        metrics["without_queues"]["mean_speed_km_h"],
        *(
            max(float(np.max(v)) for v in states[key].values())
            for key in ("densities", "onramp_queues", "origin_queues")
        ),
    ]


if __name__ == "__main__":
    main()
