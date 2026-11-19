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
CONTROL_INTERVAL_RUNS = ["sim_e1_alinea_tc_metanet", "sim_e1_alinea_tc_gain_metanet"]

# (CSV column, TeX header, unit, TeX number format)
COLUMNS = [
    ("total_time_spent_veh_h", "TTS", r"veh\,h", "{:.1f}"),
    ("tts_change_percent", r"$\Delta$TTS", r"\%", "{:+.1f}"),
    ("motorway_vht_veh_h", "VHT", r"veh\,h", "{:.1f}"),
    ("queue_time_veh_h", "queues", r"veh\,h", "{:.1f}"),
    ("mean_speed_km_h", r"$\bar v$", "km/h", "{:.1f}"),
    ("max_density_veh_km_lane", r"$\max \rho$", "veh/km/lane", "{:.1f}"),
    ("max_onramp_queue_veh", r"$\max N_\mathrm{r}$", "veh", "{:.1f}"),
    ("max_origin_queue_veh", r"$\max N_\mathrm{o}$", "veh", "{:.1f}"),
]


def main() -> None:
    """Build Table 2 from the stored runs."""
    ensure_runs(list(VARIANTS.values()) + CONTROL_INTERVAL_RUNS, rerun=RERUN_SIMULATION)
    rows_by_label = {**VARIANTS, **{_label(rid): rid for rid in CONTROL_INTERVAL_RUNS}}
    baseline = _values(VARIANTS["no control"])[0]
    values = {}
    for label, rid in rows_by_label.items():
        tts, *rest = _values(rid)
        values[label] = [tts, 100.0 * (tts - baseline) / baseline, *rest]

    tex = [
        r"\begin{tabular}{l" + " r" * len(COLUMNS) + "}",
        r"\toprule",
        " & ".join(["Variant"] + [c[1] for c in COLUMNS]) + r" \\",
        " & ".join([""] + [f"({c[2]})" for c in COLUMNS]) + r" \\",
        r"\midrule",
    ]
    for label, row in values.items():
        cells = [c[3].format(value) for c, value in zip(COLUMNS, row)]
        tex.append(" & ".join([label.replace(" s,", r"\,s,"), *cells]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    rows: list[list[object]] = [["variant", "run", *[c[0] for c in COLUMNS]]]
    rows += [[label, rows_by_label[label], *row] for label, row in values.items()]
    save_table(TABLE_NAME, rows, "\n".join(tex))


def _label(rid: str) -> str:
    """Return the row label of an ALINEA run with a control interval."""
    controller = onramps(*build_network(RUNS[rid])[:2])[0].controller
    if not isinstance(controller, AlineaController) or controller.control_dt is None:
        raise TypeError(f"run {rid} needs ALINEA with a control interval")
    return (
        rf"ALINEA, $T_\mathrm{{c}} = {controller.control_dt * 3600:g}$ s, "
        rf"$K = {controller.gain:g}$"
    )


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
