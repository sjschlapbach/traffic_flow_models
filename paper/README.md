# Paper figures and tables

This folder generates every figure and the numbers behind every table of the results section of the STRC 2026 paper directly from the library. It is not part of the released package: it lies outside `src/`, so neither the wheel nor the sdist contains it.

The scripts use only the public API (`from traffic_flow_models import ...`) and the scenario builders in `src/demo/scenarios.py`, which are the single source for networks, demands and horizons. Paper-only variants modify the network a builder returns.

## Setup

Use the repository's `venv/` with the editable install (see the `suggested_commands` memory). The editable install puts `src/` on the path, so `from demo.scenarios import ...` resolves. No LaTeX installation is needed.

## Reproduce everything

From the repository root:

```bash
source venv/bin/activate
rm -rf paper/results paper/figures paper/tables   # optional: clean state
python -m paper.fig_model_comparison
python -m paper.fig_ramp_metering
python -m paper.table_ramp_metering
python -m paper.fig_calibration
python -m paper.table_calibration
```

From a clean state the five scripts take about one minute in total on an Apple-silicon laptop: about 2 s per simulation (including the library's diagnostic plots) and 2 to 5 s per calibration. Missing runs are computed on first use, so the order of the scripts does not matter. The workflow `.github/workflows/scripts_paper.yml` runs all five scripts from a clean state on pushes and pull requests that touch `src/` or `paper/`, reports whether the outputs are byte-identical to the committed ones (informational) and uploads them as artifact `paper-outputs`.

## Outputs

| Output                                              | Script                    | Runs                                                                                                                              |
| --------------------------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `figures/fig_model_comparison.pdf` (Fig. 3)         | `fig_model_comparison.py` | `sim_c_none_ctm`, `sim_c_none_metanet`                                                                                            |
| `figures/fig_ramp_metering.pdf` (Fig. 4)            | `fig_ramp_metering.py`    | `sim_e_none_metanet`, `sim_e1_alinea_metanet`, `sim_e2_metaline_metanet`, `sim_e1_alinea_nmax_metanet`                            |
| `tables/table_ramp_metering.csv` / `.tex` (Table 2) | `table_ramp_metering.py`  | the four runs of Fig. 4 and `sim_e1_alinea_tc_metanet`, `sim_e1_alinea_tc_gain_metanet`                                           |
| `figures/fig_calibration.pdf` (Fig. 5) | `fig_calibration.py` | `sim_c_none_metanet`, `cal_c_noisy`, `sim_c_start_metanet`, `sim_c_calibrated_metanet` |
| `figures/fig_calibration_alinea.pdf` (same form as Fig. 5, can replace it) | `fig_calibration.py` | `sim_c2_alinea_metanet`, `cal_c2_noisy`, `sim_c2_start_metanet`, `sim_c2_calibrated_metanet` |
| `tables/table_calibration.csv` / `.tex` (Table 3)   | `table_calibration.py`    | the six calibration runs                                                                                                          |

Every figure is a vector PDF with embedded fonts plus a PNG preview; every table is a CSV with full precision plus a booktabs `tabular` fragment for `\input`. Each script names the runs behind every panel and column.

## Scenario names in the paper

The paper numbers the networks instead of using the demo letters: Scenario 1 is demo scenario C (one on-ramp and a lane drop; model comparison and calibration, the calibration also on its ALINEA-controlled variant C2, written "1 + ALINEA" in Table 3), Scenario 2 is demo scenario E (three on-ramps; ramp metering). Run ids keep the demo letters because they name the builders in `src/demo/scenarios.py`.

## Run matrix

All settings equal the demo scripts: dt = 10 s, preferred cell size 0.5 km, destination flow boundary constraint 6000 veh/h, destination density boundary constraint 0, horizons from `SCENARIO_DURATIONS`, METANET tau = 22 s, nu = 15, kappa = 10, delta = 1.4, phi = 10, alpha = 2. The registry is `RUNS` in `runs.py`.

| Group | Run ids                                                                                   | Content                                                                                                                                                                                      |
| ----- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1    | `sim_c_none_{ctm,metanet}`                                                                | scenario C (one on-ramp, lane drop) without control, both models                                                                                                                             |
| S2 | `sim_e_none_metanet`, `sim_e1_alinea_metanet`, `sim_e2_metaline_metanet` | scenario E (three on-ramps; mainline peak 3500 veh/h, each on-ramp 1500 veh/h, 75 % of its capacity) without control, with ALINEA at every on-ramp (E1) and with METALINE (E2), METANET |
| S3 | `sim_e1_alinea_nmax_metanet`, `sim_e1_alinea_tc_metanet`, `sim_e1_alinea_tc_gain_metanet` | E1 with maximum queue $N_m^{\max}$ = `MAX_QUEUE_VEH` = 90 veh on every on-ramp; E1 with control interval $T_c$ = `CONTROL_INTERVAL_S` = 60 s, once with the demo gain $K_I'$ = 5 and once with $K_I'$ = 30 = 5 $T_c/\Delta t$, retuned for $T_c$ (paper-only; smaller gains such as 5 $\Delta t/T_c$ perform worse, see the briefing) |
| S4 | `sim_c2_alinea_metanet`, `cal_{c,c2}_{exact,noisy,noisy_reg}` | METANET calibration on scenario C and on scenario C2 (C with ALINEA); the lane drop of the C family is the only one in the demo scenarios where $\phi$ acts |
| S4 | `sim_c_start_metanet`, `sim_c_calibrated_metanet`, `sim_c2_start_metanet`, `sim_c2_calibrated_metanet` | scenarios C and C2 with the calibration start values and with the estimates of `cal_c_noisy` / `cal_c2_noisy` |

Calibration follows `src/demo/metanet_calibration.py`: start values `METANET().get_default_calibration_params()`, default bounds, window 30, stride 15, one global alpha, single start, `max_nfev` 200. Conditions: `exact` (noise-free data, `initial_state_smoothing=1`, lambda = 0), `noisy` (default smoothing, lambda = 0), `noisy_reg` (default smoothing, lambda = 0.1). The noise has the demo's form (Gaussian, clipped at 0) but 1.5 times its levels (a deviation from the demo chosen so that the noise is visible in the figures), i.e. std 7.5 % / 15 % / 7.5 % of the mean flow / density / speed, from `numpy.random.default_rng(100)`.

## Stored runs and `RERUN_SIMULATION`

Each script has `RERUN_SIMULATION: bool = False` at the top.

- `False`: stored runs are loaded. A missing run is computed and stored with a notice.
- `True`: all runs the script needs, including the runs they depend on, are computed again and overwritten.

A stored run is reused only while the files that determine it are unchanged: `config.json` holds the run definition and the SHA-256 of every file under `src/traffic_flow_models/`, of `src/demo/scenarios.py`, of `paper/runs.py` and of the results of the runs it depends on. If one of them changed, the script stops and names it; set `RERUN_SIMULATION = True` (or delete `results/`) to recompute. Any edit of `runs.py` therefore recomputes all runs (about one minute).

`results/<run_id>/` holds:

- `simulation_results.json` (the library results JSON) and `metrics.json` (`Simulation.compute_metrics` with and without queues);
- `link_plots/`, `node_plots/`, `network_topology.png`: the library's own plots of the full state, to check that the selected panels are representative;
- calibration runs: `calibration.json` (parameter names, true and start values, estimates, cost, `nfev`, status), `jacobian.npy` (Jacobian at the solution), `measurements.json` (noisy conditions) and the library's correlation and parameter-history plots;
- `config.json`, written last, so an interrupted run counts as missing.

## Figure style

`style.py` holds the one style of all figures: full text width 159 mm, created at final size, 9 pt Computer Modern (the font of the paper; the fonts ship with matplotlib), thin dark-grey axes with inward ticks, a light horizontal grid, time ticks every 30 min and panel labels that name what a panel shows. Okabe-Ito colours with one mapping for the whole paper: CTM `#0072B2`, METANET `#D55E00`; no control `#000000`, ALINEA `#009E73`, METALINE `#CC79A7`, ALINEA with $N_m^{\max}$ `#E69F00`; markers separate the control laws, so every series is distinguishable in greyscale. Density maps use `viridis`, the correlation matrix a diverging map with limits +-1. Time is in minutes, value axes start at 0, compared panels share their limits, and the stored samples are plotted without smoothing or resampling.

## Adding or swapping a run

Add a `SimulationRun` or `CalibrationRun` to `RUNS` in `runs.py` (a new paper-only variant also needs a branch in `build_network`), then refer to its id in the script of the figure or table. The script computes the run on first use.
