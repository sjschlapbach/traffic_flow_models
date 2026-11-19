"""Runs of the paper pipeline: registry, execution and storage.

A run is a simulation of a demo scenario (``src/demo/scenarios.py``), possibly with a
paper-only modification, or a METANET calibration on synthetic measurements. All
settings equal the demo scripts except the paper-only settings below (listed with the
other deviations in ``paper/README.md``). Results are stored in
``paper/results/<run_id>/`` and reused as long as the files that determine them are
unchanged; otherwise ``ensure_runs`` stops with the changed files.
"""

import dataclasses
import functools
import hashlib
import inspect
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from traffic_flow_models import (
    CTM,
    METANET,
    AlineaController,
    Calibrator,
    METANETParams,
    MotorwayLink,
    Network,
    Onramp,
    Simulation,
)
from traffic_flow_models.types import JSONDict, ModelOptions, TimeProfile
from demo.scenarios import (
    SCENARIO_DURATIONS,
    setup_network_c,
    setup_network_c2,
    setup_network_e,
    setup_network_e1,
    setup_network_e2,
)

PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
RESULTS_DIR = PAPER_DIR / "results"
# files that determine the results of every run
SOURCES = ["src/traffic_flow_models/**/*.py", "src/demo/scenarios.py", "paper/runs.py"]

# ! common settings of the demo scripts
DT_S = 10.0  # simulation time step [s]
DT_H = DT_S / 3600
PREFERRED_CELL_SIZE_KM = 0.5
DESTINATION_FLOW_BC_VEH_H = 6000.0
DESTINATION_DENSITY_BC_VEH_KM_LANE = 0.0

# ground truth METANET parameters of all demos (tau in hours)
TRUE_PARAMS: METANETParams = {
    "tau": 22.0 / 3600,
    "nu": 15.0,
    "kappa": 10.0,
    "delta": 1.4,
    "phi": 10.0,
    "alpha": 2.0,
}

# ! paper-only ALINEA extensions, applied to every on-ramp of scenario E1
CONTROL_INTERVAL_S = 60.0  # control interval T_c
# on-ramp length 0.5 km x 1 lane x jam density 180 veh/km/lane
MAX_QUEUE_VEH = 90.0

# ! calibration settings of the calibration demo
WINDOW_SIZE = 30
STRIDE = 15
MAX_NFEV = 200
MODEL_OPTIONS: ModelOptions = {"link_specific_alpha": False}  # one global alpha
REGULARIZATION_WEIGHT = 0.1
EXACT_DATA_SMOOTHING = 1  # noise-free data: windows start from the exact states
LIBRARY_DEFAULT_SMOOTHING: int = (
    inspect.signature(Calibrator.calibrate_model_params)
    .parameters["initial_state_smoothing"]
    .default
)

# measurement noise: std as a fraction of the mean of each quantity over all series,
# drawn in this order and clipped at zero (1.5 times the calibration demo's levels, so
# that the noise is clearly visible in the figures)
NOISE_SEED = 100
NOISE_STD_FRACTIONS: dict[str, float] = {
    "flows": 0.075,
    "densities": 0.15,
    "speeds": 0.075,
}

SCENARIO_BUILDERS: dict[
    str, Callable[[], tuple[Network, JSONDict, dict[str, TimeProfile]]]
] = {
    "C": setup_network_c,
    "C2": setup_network_c2,
    "E": setup_network_e,
    "E1": setup_network_e1,
    "E2": setup_network_e2,
}

type ModelName = Literal["CTM", "METANET"]
type Variant = Literal[
    "demo", "control_interval", "control_interval_scaled_gain", "max_queue"
]
type Condition = Literal["exact", "noisy", "noisy_reg"]

T = TypeVar("T")


class StaleResultsError(RuntimeError):
    """Stored results were produced by different code or upstream results."""


@dataclass(frozen=True)
class SimulationRun:
    """Simulation of a demo scenario.

    Attributes:
        run_id: Unique run id, also the name of the results folder.
        scenario: Key of ``SCENARIO_BUILDERS``.
        model: Traffic flow model.
        variant: ``"demo"`` (builder unchanged) or one of the paper-only ALINEA
            extensions: control interval with the demo gain, control interval with
            the gain scaled by T_c / dt, maximum queue on every on-ramp.
        parameters: METANET parameters: ``"true"`` (``TRUE_PARAMS``), ``"start"`` (the
            calibration start values) or the id of a calibration run whose estimates
            are used.
    """

    run_id: str
    scenario: str
    model: ModelName
    variant: Variant = "demo"
    parameters: str = "true"


@dataclass(frozen=True)
class CalibrationRun:
    """METANET calibration on synthetic measurements of a simulation run.

    Attributes:
        run_id: Unique run id, also the name of the results folder.
        truth: METANET simulation that provides the measurements.
        condition: ``"exact"`` (noise-free data, no smoothing, no regularisation),
            ``"noisy"`` (noisy data, default smoothing, no regularisation) or
            ``"noisy_reg"`` (noisy data, default smoothing, regularisation).
    """

    run_id: str
    truth: SimulationRun
    condition: Condition


type Run = SimulationRun | CalibrationRun

_TRUTH_C = SimulationRun("sim_c_none_metanet", "C", "METANET")
_TRUTH_C2 = SimulationRun("sim_c2_alinea_metanet", "C2", "METANET")
_RUN_LIST: list[Run] = [
    # S1: scenario C without control under both models
    SimulationRun("sim_c_none_ctm", "C", "CTM"),
    _TRUTH_C,
    # S2, S3: ramp metering on scenario E (three on-ramps), METANET
    SimulationRun("sim_e_none_metanet", "E", "METANET"),
    SimulationRun("sim_e1_alinea_metanet", "E1", "METANET"),
    SimulationRun("sim_e2_metaline_metanet", "E2", "METANET"),
    SimulationRun("sim_e1_alinea_nmax_metanet", "E1", "METANET", "max_queue"),
    SimulationRun("sim_e1_alinea_tc_metanet", "E1", "METANET", "control_interval"),
    SimulationRun(
        "sim_e1_alinea_tc_gain_metanet", "E1", "METANET", "control_interval_scaled_gain"
    ),
    # S4: calibration on scenario C and on scenario C2 (C with ALINEA)
    _TRUTH_C2,
    *[
        CalibrationRun(f"cal_{truth.scenario.lower()}_{condition}", truth, condition)
        for truth in (_TRUTH_C, _TRUTH_C2)
        for condition in ("exact", "noisy", "noisy_reg")
    ],
    SimulationRun("sim_c_start_metanet", "C", "METANET", parameters="start"),
    SimulationRun("sim_c_calibrated_metanet", "C", "METANET", parameters="cal_c_noisy"),
    SimulationRun("sim_c2_start_metanet", "C2", "METANET", parameters="start"),
    SimulationRun(
        "sim_c2_calibrated_metanet", "C2", "METANET", parameters="cal_c2_noisy"
    ),
]
RUNS: dict[str, Run] = {run.run_id: run for run in _RUN_LIST}


def ensure_runs(run_ids: list[str], rerun: bool) -> None:
    """Make sure the runs and the runs they depend on are stored and up to date.

    Args:
        run_ids: Runs needed by a script.
        rerun: Run and overwrite all of them (including dependencies) if True.

    Raises:
        StaleResultsError: If a stored run was produced by different sources or
            upstream results.
    """
    for run_id in _with_dependencies(run_ids):
        run, config_path = RUNS[run_id], RESULTS_DIR / run_id / "config.json"
        config = _config(run)
        if config_path.exists() and not rerun:
            stored = load_json(run_id, "config.json")
            changed = sorted(
                k for k in stored | config if stored.get(k) != config.get(k)
            )
            if changed:
                raise StaleResultsError(
                    f"stored run '{run_id}' is stale, changed: {', '.join(changed)}; "
                    "set RERUN_SIMULATION = True to recompute it"
                )
            continue
        print(f"[run] {run_id}", flush=True)
        shutil.rmtree(config_path.parent, ignore_errors=True)
        config_path.parent.mkdir(parents=True)
        if isinstance(run, CalibrationRun):
            calibrate(run, config_path.parent)
        else:
            simulate(run, config_path.parent)
        # written last, so an interrupted run counts as missing
        write_json(config_path, config)


def build_network(run: Run) -> tuple[Network, JSONDict, dict[str, TimeProfile]]:
    """Build the network of a run from its scenario builder.

    The paper-only variants replace every ALINEA controller of the scenario by one
    with the same measurement cell and set point plus a control interval (with the
    demo gain, or the gain scaled by T_c / dt so that the rate changes as fast per
    unit time as with an update every step), or set the maximum queue of every
    ALINEA-controlled on-ramp.

    Args:
        run: Run definition.

    Returns:
        ``(network, metadata, origin_demands)`` as returned by the builder.
    """
    sim_run = run if isinstance(run, SimulationRun) else run.truth
    net, metadata, origin_demands = SCENARIO_BUILDERS[sim_run.scenario]()
    if sim_run.variant == "demo":
        return net, metadata, origin_demands
    gain_factor = (
        CONTROL_INTERVAL_S / DT_S
        if sim_run.variant == "control_interval_scaled_gain"
        else 1.0
    )
    for onramp in onramps(net, metadata):
        demo_alinea = onramp.controller
        if not isinstance(demo_alinea, AlineaController):
            raise TypeError(f"run {sim_run.run_id} needs ALINEA-controlled on-ramps")
        if sim_run.variant == "max_queue":
            onramp.max_queue = MAX_QUEUE_VEH
            continue
        onramp.controller = AlineaController(
            onramp=onramp,
            measurement_link_id=demo_alinea.measurement_link_id,
            measurement_cell_idx=demo_alinea.measurement_cell,
            gain=demo_alinea.gain * gain_factor,
            density_setpoint=demo_alinea.density_setpoint,
            control_dt=CONTROL_INTERVAL_S / 3600,
        )
    return net, metadata, origin_demands


def onramps(net: Network, metadata: JSONDict) -> list[Onramp]:
    """Return the on-ramps of a scenario network in driving order."""
    links = [net.get_link(link_id) for link_id in metadata["onramp_ids"]]
    return [link for link in links if isinstance(link, Onramp)]


def mainline(net: Network, metadata: JSONDict) -> list[MotorwayLink]:
    """Return the motorway links of a scenario network in driving order."""
    links = [net.get_link(link_id) for link_id in metadata["motorway_ids"]]
    return [link for link in links if isinstance(link, MotorwayLink)]


def simulate(run: SimulationRun, run_dir: Path) -> None:
    """Simulate a run; store the library results and plots and the metrics.

    Args:
        run: Simulation run definition.
        run_dir: Output folder of this run.
    """
    sim, time, states = _run(run, run_dir)
    metrics: JSONDict = {}
    for name, ignore_queues in [("with_queues", False), ("without_queues", True)]:
        vkt, vht, speed = sim.compute_metrics(
            states=states, dt=DT_H, timesteps=len(time), ignore_queues=ignore_queues
        )
        metrics[name] = {"vkt_veh_km": vkt, "vht_veh_h": vht, "mean_speed_km_h": speed}
    write_json(run_dir / "metrics.json", metrics)


def calibrate(run: CalibrationRun, run_dir: Path) -> None:
    """Calibrate METANET on the measurements of the truth run and store the outcome.

    Args:
        run: Calibration run definition.
        run_dir: Output folder of this run.
    """
    truth_path = RESULTS_DIR / run.truth.run_id / "simulation_results.json"
    measurement_path = truth_path
    if run.condition != "exact":
        measurement_path = run_dir / "measurements.json"
        _write_noisy_measurements(truth_path, measurement_path)

    # simulating the truth partitions the links into cells, as the calibrator
    # requires (the calibration demo calibrates on its ground-truth network)
    net = _run(run.truth, None)[0].network
    model = METANET()
    initial_params = model.get_default_calibration_params()
    estimates, result, _ = Calibrator(network=net).calibrate_model_params(
        ground_truth_filepath=str(measurement_path),
        model=model,
        initial_params=initial_params,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        model_options=MODEL_OPTIONS,
        regularization_weight=(
            REGULARIZATION_WEIGHT if run.condition == "noisy_reg" else 0.0
        ),
        initial_state_smoothing=(
            EXACT_DATA_SMOOTHING
            if run.condition == "exact"
            else LIBRARY_DEFAULT_SMOOTHING
        ),
        max_nfev=MAX_NFEV,
        verbose=False,
        plot_correlation=True,
        plot_param_history=True,
        save_dir=str(run_dir),
    )
    np.save(run_dir / "jacobian.npy", result.jac)
    write_json(
        run_dir / "calibration.json",
        {
            "param_names": model.get_calibration_param_names(net, MODEL_OPTIONS),
            "true_params": TRUE_PARAMS,
            "initial_params": initial_params,
            "estimates": estimates,
            "cost": result.cost,
            "nfev": result.nfev,
            "status": result.status,
            "message": result.message,
        },
    )


def load_json(run_id: str, name: str) -> JSONDict:
    """Load a JSON file of a stored run."""
    with open(RESULTS_DIR / run_id / name, encoding="utf-8") as f:
        data: JSONDict = json.load(f)
    return data


def load_array(run_id: str, name: str) -> NDArray[np.float64]:
    """Load a ``.npy`` array of a stored run."""
    array: NDArray[np.float64] = np.load(RESULTS_DIR / run_id / name)
    return array


def write_json(path: Path, data: object) -> None:
    """Write JSON with sorted keys, so that identical content gives identical bytes."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _run(
    run: SimulationRun, run_dir: Path | None
) -> tuple[Simulation, NDArray[np.float64], NDArray[np.float64]]:
    """Build the network of a simulation run and simulate it with the demo settings.

    Args:
        run: Simulation run definition.
        run_dir: Output folder for the library's results JSON and plots; ``None``
            stores nothing.

    Returns:
        The simulation and ``(time, states)`` as returned by ``Simulation.run``.
    """
    net, metadata, origin_demands = build_network(run)
    sim = Simulation(
        network=net,
        model=CTM() if run.model == "CTM" else METANET(),
        model_params=_model_params(run),
    )
    time, states, _ = sim.run(
        duration=SCENARIO_DURATIONS[run.scenario],
        dt=DT_H,
        preferred_cell_size=PREFERRED_CELL_SIZE_KM,
        origin_demands=origin_demands,
        turning_rates={
            nid: functools.partial(_constant, split)
            for nid, split in metadata["splits"].items()
        },
        destination_flow_bc=_destination_bc(metadata, DESTINATION_FLOW_BC_VEH_H),
        destination_density_bc=_destination_bc(
            metadata, DESTINATION_DENSITY_BC_VEH_KM_LANE
        ),
        plot_results=run_dir is not None,
        show_plots=False,
        results_dir=None if run_dir is None else str(run_dir),
    )
    return sim, time, states


def _model_params(run: SimulationRun) -> METANETParams | None:
    """Return the METANET parameters of a simulation run (``None`` for CTM)."""
    if run.model == "CTM":
        return None
    if run.parameters == "true":
        return TRUE_PARAMS
    if run.parameters == "start":
        return METANET().get_default_calibration_params()
    estimates: METANETParams = load_json(run.parameters, "calibration.json")[
        "estimates"
    ]
    return estimates


def _config(run: Run) -> JSONDict:
    """Return what determines the results of a run.

    The run definition, the SHA-256 of every file in ``SOURCES`` and of the results
    of the runs it depends on.
    """
    files = [path for pattern in SOURCES for path in sorted(REPO_DIR.glob(pattern))]
    return {
        "run": dataclasses.asdict(run),
        **{f"source {p.relative_to(REPO_DIR).as_posix()}": _sha256(p) for p in files},
        **{f"results of {dep}": _sha256(_result_file(dep)) for dep in _deps(run)},
    }


def _deps(run: Run) -> list[str]:
    """Return the ids of the runs whose stored results a run needs."""
    if isinstance(run, CalibrationRun):
        return [run.truth.run_id]
    return [run.parameters] if run.parameters in RUNS else []


def _with_dependencies(run_ids: list[str]) -> list[str]:
    """Return the runs and all their dependencies, dependencies first, no repeats."""
    ordered: list[str] = []
    for run_id in run_ids:
        for dep in _with_dependencies(_deps(RUNS[run_id])) + [run_id]:
            if dep not in ordered:
                ordered.append(dep)
    return ordered


def _result_file(run_id: str) -> Path:
    """Return the main result file of a run (results JSON or calibration summary)."""
    if isinstance(RUNS[run_id], CalibrationRun):
        return RESULTS_DIR / run_id / "calibration.json"
    return RESULTS_DIR / run_id / "simulation_results.json"


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _constant(value: T, _time: float) -> T:
    """Return ``value`` at every time (time-invariant boundary profile)."""
    return value


def _destination_bc(
    metadata: JSONDict, value: float
) -> dict[str, Callable[[float], float]]:
    """Return a constant boundary condition for every destination."""
    return {
        did: functools.partial(_constant, value) for did in metadata["destination_ids"]
    }


def _write_noisy_measurements(truth_path: Path, measurement_path: Path) -> None:
    """Add Gaussian measurement noise to all flows, densities and speeds.

    Args:
        truth_path: Results JSON of the noise-free truth run.
        measurement_path: Output path of the noisy results JSON.
    """
    with open(truth_path, encoding="utf-8") as f:
        data: JSONDict = json.load(f)
    rng = np.random.default_rng(NOISE_SEED)
    for quantity, std_fraction in NOISE_STD_FRACTIONS.items():
        series = data["state_time_series"][quantity]
        mean = np.mean([np.mean(np.array(values)) for values in series.values()])
        for series_id, values in series.items():
            clean = np.array(values)
            noisy = clean + rng.normal(0, std_fraction * mean, clean.shape)
            series[series_id] = np.maximum(noisy, 0).tolist()
    write_json(measurement_path, data)
