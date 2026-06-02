# Calibration

!!! warning "Experimental"

    Calibration tooling is experimental. The API may change in future releases.

Parameter calibration fits METANET model parameters to data from a reference
simulation (e.g. from SUMO microsimulation or a real-world reference run) using
**sliding-window nonlinear least-squares** optimisation via SciPy's Trust-Region
Reflective algorithm. Only METANET supports calibration; CTM has no calibratable
parameters (its characteristics are entirely determined by link geometry).

For a detailed algorithmic description, see
[Schlapbach et al., STRC 2026](../index.md#citation).

---

## Overview

The calibration workflow:

1. **Run a reference simulation** and save the results to a JSON file.
2. **Instantiate a `Calibrator`** with the same `Network` object.
3. **Call `calibrate_model_params()`** with the path to the saved JSON.
4. Optionally run **multi-start search** (Latin Hypercube Sampling) to escape local
   minima, and inspect **parameter correlations** to detect identifiability issues.

---

## Basic usage

```python
from traffic_flow_models import Calibrator, METANET, METANETParams

# Initial parameter guess
initial_params: METANETParams = {
    "tau": 10.0 / 3600,
    "nu": 20.0,
    "kappa": 20.0,
    "delta": 1.0,
    "phi": 1.0,
    "alpha": 1.0,
}

metanet = METANET()

# Create the calibrator (takes only the network)
calibrator = Calibrator(network=net)

# Calibrate — returns (params_dict, scipy_result, param_history_array)
calibrated_params, result, param_history = calibrator.calibrate_model_params(
    # ground truth file (can be generated from microsimulation through pipeline)
    ground_truth_filepath="results/ground_truth/simulation_results.json",
    model=metanet,
    initial_params=initial_params,
    window_size=30,       # time steps per sliding window
    stride=15,            # step between windows (default: window_size // 2)
    model_options={"link_specific_alpha": False},
    regularization_weight=0.0,
    max_nfev=200,
    verbose=True,
    save_dir="results/calibration",
    plot_correlation="parameter_correlation.png",
    plot_param_history="param_history.png",
    correlation_title="Parameter Correlation Analysis",
)

print(calibrated_params)          # METANETParams dict with calibrated values
print(f"Cost:    {result.cost:.4e}")
print(f"Success: {result.success}")
print(f"Evals:   {result.nfev}")
```

The calibration targets the six `METANETParams` fields: `tau`, `nu`, `kappa`,
`delta`, `phi`, and `alpha`. Fundamental-diagram parameters (`lane_capacity`,
`free_flow_speed`, `jam_density`) are **not** calibrated — they stay on the links.

### Return values

| Value               | Type                   | Description                                                                             |
| ------------------- | ---------------------- | --------------------------------------------------------------------------------------- |
| `calibrated_params` | `METANETParams`        | Calibrated parameter dict                                                               |
| `result`            | `scipy.OptimizeResult` | Full optimizer state (`result.cost`, `result.success`, `result.nfev`, `result.message`) |
| `param_history`     | `NDArray[float]`       | 2-D array `(n_evals × n_params)` — parameter vector at each function evaluation         |

---

## Custom bounds

Retrieve the model's default bounds and optionally override them:

```python
import numpy as np

lower, upper = METANET().get_calibration_bounds(network=net)

# Override specific bounds (order: tau, nu, kappa, delta, phi, alpha)
lower[0] = 5.0 / 3600  # tau ≥ 5 s
upper[1] = 100.0        # nu ≤ 100

calibrated_params, result, param_history = calibrator.calibrate_model_params(
    ground_truth_filepath="results/ground_truth/simulation_results.json",
    model=metanet,
    initial_params=initial_params,
    param_bounds=(lower, upper),
    window_size=30,
    stride=15,
    model_options={"link_specific_alpha": False},
)
```

---

## Multi-start search

When data are noisy or the loss surface is multimodal, use Latin Hypercube Sampling
to explore the parameter space from multiple starting points. The best result across
all runs is returned. Pass `plot_convergence` as a filename string to save a
diagnostic plot of cost, evaluations, and optimality across all LHS runs:

```python
calibrated_params, result, param_history = calibrator.calibrate_model_params(
    ground_truth_filepath="results/ground_truth/simulation_results.json",
    model=metanet,
    initial_params=initial_params,
    window_size=30,
    stride=15,
    model_options={"link_specific_alpha": False},
    regularization_weight=0.0,
    max_nfev=150,
    use_parameter_search=True,     # enable LHS multi-start
    n_samples=40,                  # number of LHS samples
    save_dir="results/calibration",
    plot_convergence="lhs_convergence.png",   # convergence across LHS runs
    plot_correlation="parameter_correlation.png",
    plot_param_history="param_history.png",
    convergence_title="Multi-Start Convergence",
    correlation_title="Parameter Correlation — LHS Best",
    verbose=True,
)
```

`plot_convergence` only produces output when `use_parameter_search=True`; it is
silently ignored in single-start mode. Both `plot_convergence` and
`plot_correlation` also accept `True` (uses a default filename) or `False`
(disabled).

---

## Correlation analysis

High off-diagonal correlations (> 0.8) indicate parameter pairs that cannot be
identified simultaneously. The analysis is triggered automatically during
`calibrate_model_params()` when `plot_correlation` is set. It can also be run
standalone on a result returned from a previous call:

```python
param_names = METANET().get_calibration_param_names(network=net)
# returns ['tau', 'nu', 'kappa', 'delta', 'phi', 'alpha']

calibrator.analyze_parameter_correlation(
    result=result,
    param_names=param_names,
    save_dir="results/calibration",
    filename="correlation_matrix.png",
    title="Parameter Correlation Analysis",
)
```

If a pair is strongly correlated, consider fixing one parameter to a physically
motivated value by setting its lower and upper bounds to the same value, then
re-running.

---

## Regularization and noisy data

For noisy real-world data, three strategies improve robustness:

1. **Disable link-specific alpha** (`link_specific_alpha: False`) to reduce
   the number of free parameters.
2. **Add Tikhonov regularization** (`regularization_weight`) to penalise
   deviation from `initial_params`.
3. **Use multi-start search** to avoid local minima introduced by noise.

```python
# Experiment: noisy data with Tikhonov regularization (no multi-start)
calibrated_params_reg, result_reg, param_history_reg = calibrator.calibrate_model_params(
    ground_truth_filepath="results/noisy/simulation_results.json",
    model=metanet,
    initial_params=initial_params,
    window_size=30,
    stride=15,
    model_options={"link_specific_alpha": False},
    regularization_weight=0.01,    # L2 ridge penalty
    use_parameter_search=False,
    save_dir="results/calibration",
    plot_correlation="noisy_reg_correlation.png",
    plot_param_history="noisy_reg_param_history.png",
    correlation_title="Parameter Correlation — Noisy (Regularized)",
)
```

---

## Link-specific alpha

Set `model_options={"link_specific_alpha": True}` to calibrate a separate `alpha`
for every motorway link. The returned `calibrated_params["alpha"]` is then a
`dict[str, float]` mapping link IDs to values rather than a scalar:

```python
calibrated_params_link, result_link, param_history_link = calibrator.calibrate_model_params(
    ground_truth_filepath="results/noisy/simulation_results.json",
    model=metanet,
    initial_params=initial_params,
    window_size=30,
    stride=15,
    model_options={"link_specific_alpha": True},
    regularization_weight=0.01,
    use_parameter_search=False,
    save_dir="results/calibration",
    plot_correlation="link_alpha_correlation.png",
    plot_param_history="link_alpha_param_history.png",
    correlation_title="Parameter Correlation — Link-Specific Alpha",
)

# calibrated_params_link["alpha"] is a dict when link_specific_alpha=True
alpha = calibrated_params_link["alpha"]
if isinstance(alpha, dict):
    for link_id, alpha_val in alpha.items():
        print(f"  Link {link_id}: alpha = {alpha_val:.4f}")
else:
    print(f"  Global alpha = {alpha:.4f}")
```

---

## Comparing parameter histories

After running multiple experiments you can overlay their parameter trajectories
to compare convergence across conditions:

```python
param_names = ["tau", "nu", "kappa", "delta", "phi", "alpha"]

calibrator.plot_parameter_convergence(
    param_history_exact=param_history_exact,     # from exact ground truth
    param_history_noreg=param_history_noreg,     # from noisy data, no regularization
    param_history_reg=param_history_reg,         # from noisy data, with regularization
    param_names=param_names,
    true_params=true_params,                     # METANETParams ground truth (for reference lines)
    save_dir="results/calibration",
)
```
