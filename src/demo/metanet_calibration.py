"""
METANET Model Parameter Calibration Demo

This demo script demonstrates the parameter calibration functionality for the
METANET macroscopic traffic flow model using realistic network scenarios. It
performs calibration experiments on two different network configurations (scenarios
A and C from metanet_simulation.py) with three data conditions each:

1. **Exact Calibration**: Ground truth data with no noise
2. **Noisy Calibration (no regularization)**: Realistic measurement noise
3. **Noisy Calibration (with regularization)**: Noisy data + Tikhonov regularization

Total: 6 calibration experiments (2 scenarios × 3 conditions)

The script demonstrates:
- model-agnostic calibration interface
- METANET-specific options (global vs. link-specific alpha)
- robustness strategies for noisy data (regularization)
- performance comparison across different network topologies
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from datetime import datetime
from numpy.typing import NDArray

from traffic_flow_models import METANET, METANETParams, Network
from demo.scenarios import (
    mainline_demand_a,
    mainline_demand_c,
    onramp_demand_a,
    onramp_demand_c,
    setup_network_ab,
    setup_network_c,
)


def calculate_parameter_errors(
    true_params: METANETParams,
    calibrated_params: METANETParams,
) -> Tuple[float, float, float, dict[str, float]]:
    """Calculate parameter errors between true and calibrated parameters.

    Args:
        true_params: True METANET parameters
        calibrated_params: Calibrated METANET parameters

    Returns:
        Tuple of (mean_error, min_error, max_error, individual_errors) where
        errors are relative errors in percent and individual_errors is a dict
        mapping parameter name to its relative error.
    """
    errors = {}

    # calculate errors for all parameters
    for param_name in ["tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]  # type: ignore
        calib_val = calibrated_params[param_name]  # type: ignore
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        errors[param_name] = rel_error

    # handle alpha (can be float or dict)
    true_alpha = true_params["alpha"]  # type: ignore
    calib_alpha = calibrated_params["alpha"]

    if isinstance(true_alpha, float) and isinstance(calib_alpha, float):
        rel_error = abs(calib_alpha - true_alpha) / (true_alpha + 1e-10) * 100
        errors["alpha"] = rel_error
    elif isinstance(calib_alpha, dict) and isinstance(true_alpha, (float, int)):
        # for link-specific alpha, average across all links
        link_errors = []
        true_alpha_float = float(true_alpha)
        for link_id, alpha_val in calib_alpha.items():
            rel_error = (
                abs(alpha_val - true_alpha_float) / (true_alpha_float + 1e-10) * 100
            )
            link_errors.append(rel_error)
        errors["alpha"] = float(np.mean(link_errors))

    # compute statistics
    error_values = list(errors.values())
    mean_error = float(np.mean(error_values))
    min_error = float(np.min(error_values))
    max_error = float(np.max(error_values))

    return mean_error, min_error, max_error, errors


def plot_calibration_comparison(
    network: Network,
    ground_truth_filepath: str,
    noisy_filepath: str,
    calibrated_params_exact: METANETParams,
    calibrated_params_noreg: METANETParams,
    calibrated_params_reg: METANETParams,
    dt: float,
    duration: float,
    preferred_cell_size: float,
    origin_demands: dict[str, Callable[[float], float]],
    onramp_demands: dict[str, Callable[[float], float]],
    turning_rates: dict[str, Callable[[float], dict[str, float]]],
    destination_flow_bc: dict[str, Callable[[float], float]],
    destination_density_bc: dict[str, Callable[[float], float]],
    save_dir: str,
) -> None:
    """Plot comparison of ground truth, noisy data, and calibrated predictions.

    Creates comparison plots for flow, density, and speed for each motorway link,
    showing ground truth, noisy measurements, exact calibration, prediction without
    regularization, and prediction with regularization.

    Args:
        network: Network instance
        ground_truth_filepath: Path to ground truth simulation results
        noisy_filepath: Path to noisy simulation results
        calibrated_params_exact: Calibrated parameters from exact (noiseless) data
        calibrated_params_noreg: Calibrated parameters without regularization
        calibrated_params_reg: Calibrated parameters with regularization
        dt: Simulation timestep
        duration: Simulation duration
        preferred_cell_size: Preferred cell size
        origin_demands: Origin demand functions
        onramp_demands: Onramp demand functions
        turning_rates: Turning rate functions
        destination_flow_bc: Destination flow boundary conditions
        destination_density_bc: Destination density boundary conditions
        save_dir: Directory to save plots
    """
    print("\n" + "-" * 80)
    print("Generating calibration comparison plots...")
    print("-" * 80)

    # load ground truth and noisy data
    time_array_gt, state_history_gt, _, _ = network.load_simulation_results_json(
        filepath=ground_truth_filepath, network=network
    )

    time_array_noisy, state_history_noisy, _, _ = network.load_simulation_results_json(
        filepath=noisy_filepath, network=network
    )

    # run simulations with exact calibration parameters
    print("  Running simulation with exact-calibration parameters...")
    metanet = METANET()
    _, state_history_exact, _ = network.simulate(
        duration=duration,
        dt=dt,
        model=metanet,
        model_params=calibrated_params_exact,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
        show_plots=False,
    )

    # run simulations with calibrated parameters (no regularization)
    print("  Running simulation with no-regularization parameters...")
    _, state_history_noreg, _ = network.simulate(
        duration=duration,
        dt=dt,
        model=metanet,
        model_params=calibrated_params_noreg,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
        show_plots=False,
    )

    # run simulations with calibrated parameters (with regularization)
    print("  Running simulation with regularization parameters...")
    _, state_history_reg, _ = network.simulate(
        duration=duration,
        dt=dt,
        model=metanet,
        model_params=calibrated_params_reg,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
        show_plots=False,
    )

    # extract states for each configuration
    time_seconds = time_array_gt * 3600  # convert hours to seconds

    # unpack states
    states_gt = [
        network.state_vec_to_network_dict(state_history_gt[:, t])
        for t in range(len(time_array_gt))
    ]
    states_noisy = [
        network.state_vec_to_network_dict(state_history_noisy[:, t])
        for t in range(len(time_array_noisy))
    ]
    states_exact = [
        network.state_vec_to_network_dict(state_history_exact[:, t])
        for t in range(len(time_array_gt))
    ]
    states_noreg = [
        network.state_vec_to_network_dict(state_history_noreg[:, t])
        for t in range(len(time_array_gt))
    ]
    states_reg = [
        network.state_vec_to_network_dict(state_history_reg[:, t])
        for t in range(len(time_array_gt))
    ]

    # collect all motorway links
    motorway_links = []
    for node in network.list_nodes():
        for link in node.outgoing:
            from traffic_flow_models.network import MotorwayLink

            if isinstance(link, MotorwayLink):
                motorway_links.append(link)

    if not motorway_links:
        print("  No motorway links found for plotting.")
        return

    # create plots for each quantity (flow, density, speed)
    for quantity in ["flow", "density", "speed"]:
        print(f"  Creating {quantity} comparison plots...")

        fig, axes = plt.subplots(
            len(motorway_links), 1, figsize=(12, 4 * len(motorway_links))
        )
        if len(motorway_links) == 1:
            axes = [axes]

        for idx, link in enumerate(motorway_links):
            ax = axes[idx]

            # extract data for this link (average across cells)
            if quantity == "flow":
                data_gt = np.array([np.mean(s[0][link.id]) for s in states_gt])
                data_noisy = np.array([np.mean(s[0][link.id]) for s in states_noisy])
                data_exact = np.array([np.mean(s[0][link.id]) for s in states_exact])
                data_noreg = np.array([np.mean(s[0][link.id]) for s in states_noreg])
                data_reg = np.array([np.mean(s[0][link.id]) for s in states_reg])
                ylabel = "Flow (veh/h/lane)"
            elif quantity == "density":
                data_gt = np.array([np.mean(s[1][link.id]) for s in states_gt])
                data_noisy = np.array([np.mean(s[1][link.id]) for s in states_noisy])
                data_exact = np.array([np.mean(s[1][link.id]) for s in states_exact])
                data_noreg = np.array([np.mean(s[1][link.id]) for s in states_noreg])
                data_reg = np.array([np.mean(s[1][link.id]) for s in states_reg])
                ylabel = "Density (veh/km/lane)"
            else:  # speed
                data_gt = np.array([np.mean(s[2][link.id]) for s in states_gt])
                data_noisy = np.array([np.mean(s[2][link.id]) for s in states_noisy])
                data_exact = np.array([np.mean(s[2][link.id]) for s in states_exact])
                data_noreg = np.array([np.mean(s[2][link.id]) for s in states_noreg])
                data_reg = np.array([np.mean(s[2][link.id]) for s in states_reg])
                ylabel = "Speed (km/h)"

            # plot all curves
            ax.plot(
                time_seconds,
                data_gt,
                "k-",
                linewidth=2,
                label="Ground Truth",
                alpha=0.8,
            )
            ax.plot(
                time_seconds,
                data_noisy,
                "gray",
                linewidth=1,
                label="Noisy Data",
                alpha=0.5,
            )
            ax.plot(
                time_seconds,
                data_exact,
                "g--",
                linewidth=1.5,
                label="Calibrated (exact data)",
                alpha=0.7,
            )
            ax.plot(
                time_seconds,
                data_noreg,
                "b--",
                linewidth=1.5,
                label="Calibrated (noisy, no reg)",
                alpha=0.7,
            )
            ax.plot(
                time_seconds,
                data_reg,
                "r-",
                linewidth=1.5,
                label="Calibrated (noisy, with reg)",
                alpha=0.7,
            )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Link {link.id}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # save figure
        plot_path = os.path.join(save_dir, f"{quantity}_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Saved to: {plot_path}")

    print("  Calibration comparison plots complete.")


def plot_parameter_convergence(
    param_history_exact: NDArray[np.float64],
    param_history_noreg: NDArray[np.float64],
    param_history_reg: NDArray[np.float64],
    param_names: list[str],
    true_params: METANETParams,
    save_dir: str,
) -> None:
    """Plot convergence of calibration parameters for multiple experiments.

    Creates a single figure with subplots showing how each parameter evolves
    during the optimization process for all three calibration cases:
    - Exact data (noiseless)
    - Noisy data without regularization
    - Noisy data with regularization

    Args:
        param_history_exact: 2-D array of parameter vectors for exact data
        param_history_noreg: 2-D array of parameter vectors for noisy data (no reg)
        param_history_reg: 2-D array of parameter vectors for noisy data (with reg)
        param_names: List of parameter names in order
        true_params: True METANET parameters for reference lines
        save_dir: Directory to save the plot
    """
    print("  Creating combined convergence plot...")

    num_params = len(param_names)

    # create subplot grid (2 columns)
    ncols = 2
    nrows = (num_params + 1) // 2

    _, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten() if num_params > 1 else [axes]

    for idx, param_name in enumerate(param_names):
        ax = axes[idx]

        # plot parameter evolution for all three cases
        iterations_exact = np.arange(param_history_exact.shape[0])
        iterations_noreg = np.arange(param_history_noreg.shape[0])
        iterations_reg = np.arange(param_history_reg.shape[0])

        param_values_exact = param_history_exact[:, idx]
        param_values_noreg = param_history_noreg[:, idx]
        param_values_reg = param_history_reg[:, idx]

        ax.plot(
            iterations_exact,
            param_values_exact,
            "g-",
            linewidth=2,
            alpha=0.7,
            label="Exact data",
        )
        ax.plot(
            iterations_noreg,
            param_values_noreg,
            "b--",
            linewidth=1.5,
            alpha=0.7,
            label="Noisy (no reg)",
        )
        ax.plot(
            iterations_reg,
            param_values_reg,
            "r-",
            linewidth=1.5,
            alpha=0.7,
            label="Noisy (with reg)",
        )

        # plot true value as horizontal line
        true_val = true_params[param_name]  # type: ignore
        if isinstance(true_val, (int, float)):
            ax.axhline(
                y=true_val,
                color="k",
                linestyle=":",
                linewidth=2,
                label="True value",
                alpha=0.6,
            )

        ax.set_xlabel("Function Evaluation", fontsize=10)
        ax.set_ylabel(param_name, fontsize=11, fontweight="bold")
        ax.set_title(f"Parameter: {param_name}", fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(num_params, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Parameter Convergence Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # save figure
    plot_path = os.path.join(save_dir, "convergence_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved to: {plot_path}")


def run_calibration_experiment(
    scenario_name: str,
    network: Network,
    metadata: dict,
    mainline_demand: Callable[[float], float],
    onramp_demand: Callable[[float], float],
    true_params: METANETParams,
    initial_params: METANETParams,
    dt: float,
    duration: float,
    preferred_cell_size: float,
    timestamp: str,
) -> None:
    """Run complete calibration experiment for one scenario.

    Args:
        scenario_name: Name of the scenario (e.g., "Scenario A")
        network: Network instance
        metadata: Network metadata dict
        mainline_demand: Mainline demand function
        onramp_demand: Onramp demand function
        true_params: True METANET parameters for ground truth generation
        initial_params: Initial guess for calibration
        dt: Simulation timestep
        duration: Simulation duration
        preferred_cell_size: Preferred cell size for discretization
        timestamp: Timestamp string for results directory
    """
    print("\n" + "=" * 80)
    print(f"{scenario_name}")
    print("=" * 80)
    print(f"Network: {len(network.list_nodes())} nodes")

    # build disturbance dictionaries expected by the new simulate signature
    origin_ids = metadata.get("origin_ids", [])
    onramp_ids = metadata.get("onramp_ids", [])
    destination_ids = metadata.get("destination_ids", [])
    splits = metadata.get("splits", {})

    origin_demands: dict[str, Callable[[float], float]] = {
        oid: mainline_demand for oid in origin_ids
    }
    onramp_demands: dict[str, Callable[[float], float]] = {
        rid: onramp_demand for rid in onramp_ids
    }
    destination_flow_bc: dict[str, Callable[[float], float]] = {
        did: (lambda _: 6000.0) for did in destination_ids
    }
    destination_density_bc: dict[str, Callable[[float], float]] = {
        did: (lambda _: 0.0) for did in destination_ids
    }

    # turning rates: create callables that return the provided split mapping (time-invariant here)
    turning_rates: dict[str, Callable[[float], dict[str, float]]] = {
        nid: (lambda _t, s=splits[nid]: s) for nid in splits.keys()
    }

    # ! 1) Generate ground truth data
    print("\n[1] Generating ground truth simulation data...")
    scenario_dir = (
        f"results/calibration_{timestamp}/{scenario_name.replace(' ', '_').lower()}"
    )
    ground_truth_dir = f"{scenario_dir}/ground_truth"
    os.makedirs(ground_truth_dir, exist_ok=True)

    metanet = METANET()
    time_array, _, _ = network.simulate(
        duration=duration,
        dt=dt,
        model=metanet,
        model_params=true_params,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=True,  # need True to save simulation_results.json for calibration
        show_plots=False,  # don't display plots interactively
        results_dir=ground_truth_dir,
    )

    ground_truth_filepath = os.path.join(ground_truth_dir, "simulation_results.json")
    print(f"    Duration: {duration} hours ({len(time_array)} timesteps)")
    print(f"    Saved to: {ground_truth_filepath}")

    # ! 2) Calibration Experiment 1: Exact data
    print("\n" + "-" * 80)
    print("[2] Calibration Experiment 1: Exact Ground Truth Data")
    print("-" * 80)

    calibrated_params_exact, result_exact, param_history_exact = (
        network.calibrate_model_params(
            ground_truth_filepath=ground_truth_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            model_options={"link_specific_alpha": False},  # use global alpha
            regularization_weight=0.0,
            verbose=True,
        )
    )

    # compare results
    print("\nComparison: True vs Calibrated (Exact Data)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]  # type: ignore
        calib_val = calibrated_params_exact[param_name]  # type: ignore
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        print(
            f"{param_name:<10} {true_val:<15.6f} {calib_val:<15.6f} {rel_error:<15.2f}%"
        )

    # handle alpha
    true_alpha = true_params["alpha"]  # type: ignore
    calib_alpha = calibrated_params_exact["alpha"]
    if isinstance(true_alpha, float) and isinstance(calib_alpha, float):
        rel_error = abs(calib_alpha - true_alpha) / (true_alpha + 1e-10) * 100
        print(
            f"{'alpha':<10} {true_alpha:<15.6f} {calib_alpha:<15.6f} {rel_error:<15.2f}%"
        )
    else:
        raise ValueError("Expected alpha values to be floats for this experiment")
    print("-" * 80)

    # ! 3) Generate noisy data
    print("\n" + "-" * 80)
    print("[3] Generating Noisy Measurement Data")
    print("-" * 80)

    with open(ground_truth_filepath, "r") as f:
        ground_truth_data = json.load(f)

    print("  Adding measurement noise:")
    print("    Flows:     Gaussian noise with std = 5% of mean (constant variance)")
    print("    Densities: Gaussian noise with std = 10% of mean (constant variance)")
    print("    Speeds:    Gaussian noise with std = 5% of mean (constant variance)")

    np.random.seed(42)  # for reproducibility
    noisy_data = ground_truth_data.copy()

    # add noise to flows (constant variance over time)
    mean_flow = np.mean(
        [
            np.mean(np.array(noisy_data["state_time_series"]["flows"][link_id]))
            for link_id in noisy_data["state_time_series"]["flows"].keys()
        ]
    )

    for link_id in noisy_data["state_time_series"]["flows"].keys():
        flows = np.array(noisy_data["state_time_series"]["flows"][link_id])
        noise_std = 0.05 * mean_flow  # constant std = 5% of mean
        noise = np.random.normal(0, noise_std, flows.shape)
        noisy_flows = flows + noise
        noisy_flows = np.maximum(noisy_flows, 0)
        noisy_data["state_time_series"]["flows"][link_id] = noisy_flows.tolist()

    # add noise to densities (constant variance over time)
    mean_density = np.mean(
        [
            np.mean(np.array(noisy_data["state_time_series"]["densities"][link_id]))
            for link_id in noisy_data["state_time_series"]["densities"].keys()
        ]
    )

    for link_id in noisy_data["state_time_series"]["densities"].keys():
        densities = np.array(noisy_data["state_time_series"]["densities"][link_id])
        noise_std = 0.10 * mean_density  # constant std = 10% of mean
        noise = np.random.normal(0, noise_std, densities.shape)
        noisy_densities = densities + noise
        noisy_densities = np.maximum(noisy_densities, 0)
        noisy_data["state_time_series"]["densities"][link_id] = noisy_densities.tolist()

    # add noise to speeds (constant variance over time)
    mean_speed = np.mean(
        [
            np.mean(np.array(noisy_data["state_time_series"]["speeds"][link_id]))
            for link_id in noisy_data["state_time_series"]["speeds"].keys()
        ]
    )

    for link_id in noisy_data["state_time_series"]["speeds"].keys():
        speeds = np.array(noisy_data["state_time_series"]["speeds"][link_id])
        noise_std = 0.05 * mean_speed  # constant std = 5% of mean
        noise = np.random.normal(0, noise_std, speeds.shape)
        noisy_speeds = speeds + noise
        noisy_speeds = np.maximum(noisy_speeds, 0)
        noisy_data["state_time_series"]["speeds"][link_id] = noisy_speeds.tolist()

    # save noisy data
    noisy_dir = f"{scenario_dir}/noisy_ground_truth"
    os.makedirs(noisy_dir, exist_ok=True)
    noisy_filepath = os.path.join(noisy_dir, "simulation_results.json")

    with open(noisy_filepath, "w") as f:
        json.dump(noisy_data, f, indent=2)

    print(f"  Saved to: {noisy_filepath}")

    # ! 4) Calibration Experiment 2: Noisy data without regularization
    print("\n" + "-" * 80)
    print("[4] Calibration Experiment 2: Noisy Data (No Regularization)")
    print("-" * 80)

    calibrated_params_noisy_noreg, result_noisy_noreg, param_history_noreg = (
        network.calibrate_model_params(
            ground_truth_filepath=noisy_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            model_options={"link_specific_alpha": False},
            regularization_weight=0.0,
            verbose=True,
        )
    )

    # compare results
    print("\nComparison: True vs Calibrated (Noisy Data - No Regularization)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]  # type: ignore
        calib_val = calibrated_params_noisy_noreg[param_name]  # type: ignore
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        print(
            f"{param_name:<10} {true_val:<15.6f} {calib_val:<15.6f} {rel_error:<15.2f}%"
        )

    # handle alpha
    calib_alpha = calibrated_params_noisy_noreg["alpha"]
    if isinstance(calib_alpha, float):
        rel_error = abs(calib_alpha - true_alpha) / (true_alpha + 1e-10) * 100
        print(
            f"{'alpha':<10} {true_alpha:<15.6f} {calib_alpha:<15.6f} {rel_error:<15.2f}%"
        )
    else:
        raise ValueError("Expected alpha value to be a float for this experiment")
    print("-" * 80)

    # ! 5) Calibration Experiment 3: Noisy data with regularization
    print("\n" + "-" * 80)
    print("[5] Calibration Experiment 3: Noisy Data (With Regularization)")
    print("-" * 80)

    calibrated_params_noisy_reg, result_noisy_reg, param_history_reg = (
        network.calibrate_model_params(
            ground_truth_filepath=noisy_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            model_options={"link_specific_alpha": False},
            regularization_weight=0.01,
            verbose=True,
        )
    )

    # compare results
    print("\nComparison: True vs Calibrated (Noisy Data - With Regularization)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]  # type: ignore
        calib_val = calibrated_params_noisy_reg[param_name]  # type: ignore
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        print(
            f"{param_name:<10} {true_val:<15.6f} {calib_val:<15.6f} {rel_error:<15.2f}%"
        )

    # handle alpha
    calib_alpha = calibrated_params_noisy_reg["alpha"]
    if isinstance(calib_alpha, float):
        rel_error = abs(calib_alpha - true_alpha) / (true_alpha + 1e-10) * 100
        print(
            f"{'alpha':<10} {true_alpha:<15.6f} {calib_alpha:<15.6f} {rel_error:<15.2f}%"
        )
    else:
        raise ValueError("Expected alpha value to be a float for this experiment")
    print("-" * 80)

    # ! 6) Calibration Experiment 4: Noisy data with link-specific alpha (optional, can be added if time permits)
    print("\n" + "-" * 80)
    print("[6] Calibration Experiment 4: Noisy Data (Link-Specific Alpha)")
    print("-" * 80)

    calibrated_params_noisy_link_alpha, result_noisy_link_alpha, param_history_link = (
        network.calibrate_model_params(
            ground_truth_filepath=noisy_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            model_options={"link_specific_alpha": True},
            regularization_weight=0.01,
            verbose=True,
        )
    )

    # compare results (for global parameters only, since alpha is now link-specific)
    print("\nComparison: True vs Calibrated (Noisy Data - Link-Specific Alpha)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]
        calib_val = calibrated_params_noisy_link_alpha[param_name]
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        print(
            f"{param_name:<10} {true_val:<15.6f} {calib_val:<15.6f} {rel_error:<15.2f}%"
        )

    # handle alpha (link-specific)
    calib_alpha = calibrated_params_noisy_link_alpha["alpha"]
    if isinstance(calib_alpha, dict):
        print("\nLink-specific alpha values:")
        for link_id, alpha_val in calib_alpha.items():
            rel_error = abs(alpha_val - true_alpha) / (true_alpha + 1e-10) * 100
            print(f"  Link {link_id}: {alpha_val:.6f} (rel. error: {rel_error:.2f}%)")
    else:
        raise ValueError("Expected alpha value to be a dict for this experiment")
    print("-" * 80)

    # ! 7) Calculate parameter errors for summary
    mean_error_exact, min_error_exact, max_error_exact, errors_exact = (
        calculate_parameter_errors(true_params, calibrated_params_exact)
    )

    mean_error_noreg, min_error_noreg, max_error_noreg, errors_noreg = (
        calculate_parameter_errors(true_params, calibrated_params_noisy_noreg)
    )

    mean_error_reg, min_error_reg, max_error_reg, errors_reg = (
        calculate_parameter_errors(true_params, calibrated_params_noisy_reg)
    )

    mean_error_link, min_error_link, max_error_link, errors_link = (
        calculate_parameter_errors(true_params, calibrated_params_noisy_link_alpha)
    )

    # ! 8) Generate calibration comparison plots
    plot_calibration_comparison(
        network=network,
        ground_truth_filepath=ground_truth_filepath,
        noisy_filepath=noisy_filepath,
        calibrated_params_exact=calibrated_params_exact,
        calibrated_params_noreg=calibrated_params_noisy_noreg,
        calibrated_params_reg=calibrated_params_noisy_reg,
        dt=dt,
        duration=duration,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        onramp_demands=onramp_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        save_dir=scenario_dir,
    )

    # ! 9) Generate parameter convergence comparison plot
    print("\n" + "-" * 80)
    print("Generating parameter convergence comparison plot...")
    print("-" * 80)

    # parameter names for METANET (in order)
    param_names = ["tau", "nu", "kappa", "delta", "phi", "alpha"]

    plot_parameter_convergence(
        param_history_exact=param_history_exact,
        param_history_noreg=param_history_noreg,
        param_history_reg=param_history_reg,
        param_names=param_names,
        true_params=true_params,
        save_dir=scenario_dir,
    )

    print("  Parameter convergence comparison complete.")

    # ! 10) Summary for this scenario
    print("\n" + "=" * 80)
    print(f"SUMMARY: {scenario_name}")
    print("=" * 80)

    print("\nCalibration Performance:")
    print("  Exact data:")
    print(f"    Final cost:        {result_exact.cost:.6e}")
    print(f"    Iterations:        {result_exact.nfev}")
    print(f"    Success:           {result_exact.success}")
    print(f"    Mean param error:  {mean_error_exact:.2f}%")
    print(f"    Error range:       [{min_error_exact:.2f}%, {max_error_exact:.2f}%]")

    print("\n  Noisy data - No regularization:")
    print(f"    Final cost:        {result_noisy_noreg.cost:.6e}")
    print(f"    Iterations:        {result_noisy_noreg.nfev}")
    print(f"    Success:           {result_noisy_noreg.success}")
    print(f"    Mean param error:  {mean_error_noreg:.2f}%")
    print(f"    Error range:       [{min_error_noreg:.2f}%, {max_error_noreg:.2f}%]")

    print("\n  Noisy data - With regularization (λ=0.01):")
    print(f"    Final cost:        {result_noisy_reg.cost:.6e}")
    print(f"    Iterations:        {result_noisy_reg.nfev}")
    print(f"    Success:           {result_noisy_reg.success}")
    print(f"    Mean param error:  {mean_error_reg:.2f}%")
    print(f"    Error range:       [{min_error_reg:.2f}%, {max_error_reg:.2f}%]")

    print("\n  Noisy data - Link-specific alpha (λ=0.01):")
    print(f"    Final cost:        {result_noisy_link_alpha.cost:.6e}")
    print(f"    Iterations:        {result_noisy_link_alpha.nfev}")
    print(f"    Success:           {result_noisy_link_alpha.success}")
    print(f"    Mean param error:  {mean_error_link:.2f}%")
    print(f"    Error range:       [{min_error_link:.2f}%, {max_error_link:.2f}%]")

    print(f"\nResults saved to: {scenario_dir}/")


def main():
    """Run the calibration demonstration."""
    print("=" * 80)
    print("METANET Model Parameter Calibration Demo")
    print("=" * 80)
    print("\nTesting scenarios A and C with exact and noisy measurements")
    print("Total experiments: 6 (2 scenarios × 3 conditions)")

    # common simulation parameters
    dt = 10.0 / 3600  # hours (10 seconds)
    duration = 5000.0 / 3600  # hours
    preferred_cell_size = 0.5  # km

    # ground truth METANET parameters (same for both scenarios)
    true_params: METANETParams = {
        "tau": 22.0 / 3600,  # hours
        "nu": 15.0,
        "kappa": 10.0,
        "delta": 1.4,
        "phi": 10.0,
        "alpha": 2.0,
    }

    # initial guess (slightly perturbed)
    initial_params: METANETParams = {
        "tau": 10.0 / 3600,
        "nu": 20.0,
        "kappa": 2.0,
        "delta": 1.0,
        "phi": 1.0,
        "alpha": 1.0,
    }

    print("\nTrue parameters:")
    print(f"  tau   = {true_params['tau']:.6f}")
    print(f"  nu    = {true_params['nu']:.2f}")
    print(f"  kappa = {true_params['kappa']:.2f}")
    print(f"  delta = {true_params['delta']:.2f}")
    print(f"  phi   = {true_params['phi']:.2f}")
    print(f"  alpha = {true_params['alpha']:.2f}")

    # create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ! Scenario A
    network_a, metadata_a = setup_network_ab()
    run_calibration_experiment(
        scenario_name="Scenario A",
        network=network_a,
        metadata=metadata_a,
        mainline_demand=mainline_demand_a,
        onramp_demand=onramp_demand_a,
        true_params=true_params,
        initial_params=initial_params,
        dt=dt,
        duration=duration,
        preferred_cell_size=preferred_cell_size,
        timestamp=timestamp,
    )

    # ! Scenario C (with lane drop/bottleneck)
    network_c, metadata_c = setup_network_c()
    run_calibration_experiment(
        scenario_name="Scenario C",
        network=network_c,
        metadata=metadata_c,
        mainline_demand=mainline_demand_c,
        onramp_demand=onramp_demand_c,
        true_params=true_params,
        initial_params=initial_params,
        dt=dt,
        duration=duration,
        preferred_cell_size=preferred_cell_size,
        timestamp=timestamp,
    )

    # overall summary
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: results/calibration_{timestamp}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
