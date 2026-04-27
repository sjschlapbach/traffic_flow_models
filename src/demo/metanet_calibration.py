"""
METANET Model Parameter Calibration Demo

This demo script demonstrates the parameter calibration functionality for the
METANET macroscopic traffic flow model using realistic network scenarios. It
performs calibration experiments on two different network configurations (scenarios
A and C from metanet_simulation.py) with four data conditions each:

1. **Exact Calibration**: Ground truth data with no noise
2. **Noisy Calibration (no regularization)**: Realistic measurement noise
3. **Noisy Calibration (with regularization)**: Noisy data + Tikhonov regularization
4. **Noisy Calibration (link-specific alpha)**: Link-specific alpha parameters with regularization

Total: 8 calibration experiments (2 scenarios × 4 conditions)

The script demonstrates:
- Model-agnostic calibration interface
- Multi-start parameter search using Latin Hypercube Sampling (optional)
- METANET-specific options (global vs. link-specific alpha)
- Robustness strategies for noisy data (Tikhonov regularization)
- Performance comparison across different network topologies
"""

import os
import json
import argparse
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from datetime import datetime

from traffic_flow_models import METANET, METANETParams, Network, Calibrator, Simulation
from demo.scenarios import (
    mainline_demand_a,
    mainline_demand_c,
    onramp_demand_a,
    onramp_demand_c,
    setup_network_a,
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
    for param_name in ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]
        calib_val = calibrated_params[param_name]
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        errors[param_name] = rel_error

    # handle alpha (can be float or dict)
    true_alpha = true_params["alpha"]
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
        turning_rates: Turning rate functions
        destination_flow_bc: Destination flow boundary conditions
        destination_density_bc: Destination density boundary conditions
        save_dir: Directory to save plots
    """
    print("\n" + "-" * 80)
    print("Generating calibration comparison plots...")
    print("-" * 80)

    # load ground truth and noisy data
    time_array_gt, state_history_gt, _, _ = Simulation.load_results(
        filepath=ground_truth_filepath, network=network
    )

    time_array_noisy, state_history_noisy, _, _ = Simulation.load_results(
        filepath=noisy_filepath, network=network
    )

    # run simulations with exact calibration parameters
    print("  Running simulation with exact-calibration parameters...")
    metanet = METANET()
    sim = Simulation(network, metanet, calibrated_params_exact)
    _, state_history_exact, _ = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
        show_plots=False,
    )

    # run simulations with calibrated parameters (no regularization)
    print("  Running simulation with no-regularization parameters...")
    sim = Simulation(network, metanet, calibrated_params_noreg)
    _, state_history_noreg, _ = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
        turning_rates=turning_rates,
        destination_flow_bc=destination_flow_bc,
        destination_density_bc=destination_density_bc,
        plot_results=False,
        show_plots=False,
    )

    # run simulations with calibrated parameters (with regularization)
    print("  Running simulation with regularization parameters...")
    sim = Simulation(network, metanet, calibrated_params_reg)
    _, state_history_reg, _ = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
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
    use_parameter_search: bool = False,
    generate_video: bool = False,
) -> None:
    """Run complete calibration experiment for one scenario.

    Performs 4 calibration experiments for the given scenario:
    1. Exact ground truth data (optional multi-start with 40 LHS samples)
    2. Noisy data without regularization (optional multi-start with 40 LHS samples)
    3. Noisy data with Tikhonov regularization (λ=0.01)
    4. Noisy data with link-specific alpha parameters (λ=0.01)

    Also generates comparison plots and convergence analysis.

    Args:
        scenario_name: Name of the scenario (e.g., "Scenario A")
        network: Network instance
        metadata: Network metadata dict
        mainline_demand: Mainline demand function
        onramp_demand: Onramp demand function
        true_params: True METANET parameters for ground truth generation
        initial_params: Initial guess for calibration (used in standard mode)
        dt: Simulation timestep (hours)
        duration: Simulation duration (hours)
        preferred_cell_size: Preferred cell size for discretization (km)
        timestamp: Timestamp string for results directory
        use_parameter_search: If True, use multi-start parameter search with Latin
            Hypercube Sampling (40 samples) for Experiments 1 and 2. If False, use
            single initial_params for all experiments.
        generate_video: If True, generate a comparison video of all simulations.
    """
    print("\n" + "=" * 80)
    print(f"{scenario_name}")
    print("=" * 80)
    print(f"Network: {len(network.list_nodes())} nodes")

    # build disturbance dictionaries expected by the new simulate signature
    destination_ids = metadata.get("destination_ids", [])
    splits = metadata.get("splits", {})

    origin_demands: dict[str, Callable[[float], float]] = {
        "origin": mainline_demand,
        "origin_onr": onramp_demand,
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
    sim = Simulation(network, metanet, true_params)
    time_array, _, _ = sim.run(
        duration=duration,
        dt=dt,
        preferred_cell_size=preferred_cell_size,
        origin_demands=origin_demands,
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

    # create calibrator for parameter estimation
    calibrator = Calibrator(network=network)

    # ! 2) Calibration Experiment 1: Exact data
    print("\n" + "-" * 80)
    if use_parameter_search:
        print("[2] Calibration Experiment 1: Exact Ground Truth Data (Multi-Start)")
    else:
        print("[2] Calibration Experiment 1: Exact Ground Truth Data")
    print("-" * 80)

    # run calibration (with or without parameter search based on flag)
    calibrated_params_exact, result_exact, param_history_exact = (
        calibrator.calibrate_model_params(
            verbose=True,
            ground_truth_filepath=ground_truth_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            stride=15,
            model_options={"link_specific_alpha": False},
            regularization_weight=0.0,
            max_nfev=150 if use_parameter_search else 200,
            use_parameter_search=use_parameter_search,
            n_samples=40 if use_parameter_search else 0,
            plot_convergence=(
                "exact_data_param_search_convergence.png"
                if use_parameter_search
                else False
            ),
            plot_correlation="exact_data_parameter_correlation.png",
            plot_param_history="exact_param_history.png",
            save_dir=scenario_dir,
            convergence_title="Multi-Start Parameter Search Convergence - Exact Data",
            correlation_title="Parameter Correlation Analysis - Exact Data",
        )
    )

    # compare results
    print("\nComparison: True vs Calibrated (Exact Data)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]
        calib_val = calibrated_params_exact[param_name]
        rel_error = abs(calib_val - true_val) / (true_val + 1e-10) * 100
        print(
            f"{param_name:<10} {true_val:<15.6f} {calib_val:<15.6f} {rel_error:<15.2f}%"
        )

    # handle alpha
    true_alpha = true_params["alpha"]
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
    if use_parameter_search:
        print(
            "[4] Calibration Experiment 2: Noisy Data (No Regularization, Multi-Start)"
        )
    else:
        print("[4] Calibration Experiment 2: Noisy Data (No Regularization)")
    print("-" * 80)

    # run calibration (with or without parameter search based on flag)
    calibrated_params_noisy_noreg, result_noisy_noreg, param_history_noreg = (
        calibrator.calibrate_model_params(
            verbose=True,
            ground_truth_filepath=noisy_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            stride=15,
            model_options={"link_specific_alpha": False},
            regularization_weight=0.0,
            max_nfev=150 if use_parameter_search else 200,
            use_parameter_search=use_parameter_search,
            n_samples=40 if use_parameter_search else 0,
            plot_convergence=(
                "noisy_noreg_param_search_convergence.png"
                if use_parameter_search
                else False
            ),
            plot_correlation="noisy_noreg_parameter_correlation.png",
            plot_param_history="noisy_noreg_param_history.png",
            save_dir=scenario_dir,
            convergence_title="Multi-Start Parameter Search Convergence - Noisy Data (No Regularization)",
            correlation_title="Parameter Correlation Analysis - Noisy Data (No Regularization)",
        )
    )

    # compare results
    print("\nComparison: True vs Calibrated (Noisy Data - No Regularization)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]
        calib_val = calibrated_params_noisy_noreg[param_name]
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
        calibrator.calibrate_model_params(
            verbose=True,
            ground_truth_filepath=noisy_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            stride=15,
            use_parameter_search=False,  # do not use parameter search for regularized case to isolate effect of regularization
            model_options={"link_specific_alpha": False},
            regularization_weight=0.01,
            plot_correlation="noisy_reg_parameter_correlation.png",
            plot_param_history="noisy_reg_param_history.png",
            save_dir=scenario_dir,
            correlation_title="Parameter Correlation Analysis - Noisy Data (With Regularization)",
        )
    )

    # compare results
    print("\nComparison: True vs Calibrated (Noisy Data - With Regularization)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi"]:
        true_val = true_params[param_name]
        calib_val = calibrated_params_noisy_reg[param_name]
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
        calibrator.calibrate_model_params(
            verbose=True,
            ground_truth_filepath=noisy_filepath,
            model=metanet,
            initial_params=initial_params,
            window_size=30,
            stride=15,
            use_parameter_search=False,  # do not use parameter search for link-specific alpha case due to complexity limitations
            model_options={"link_specific_alpha": True},
            regularization_weight=0.01,
            plot_correlation="link_specific_alpha_parameter_correlation.png",
            plot_param_history="link_specific_alpha_param_history.png",
            save_dir=scenario_dir,
            correlation_title="Parameter Correlation Analysis - Link-Specific Alpha",
        )
    )

    # compare results (for global parameters only, since alpha is now link-specific)
    print("\nComparison: True vs Calibrated (Noisy Data - Link-Specific Alpha)")
    print("-" * 80)
    print(f"{'Parameter':<10} {'True':<15} {'Calibrated':<15} {'Rel. Error':<15}")
    print("-" * 80)

    for param_name in ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi"]:
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
    param_names = ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi", "alpha"]

    calibrator.plot_parameter_convergence(
        param_history_exact=param_history_exact,
        param_history_noreg=param_history_noreg,
        param_history_reg=param_history_reg,
        param_names=param_names,
        true_params=true_params,
        save_dir=scenario_dir,
    )

    print("  Parameter convergence comparison complete.")

    # ! 10) Generate simulation comparison video
    if generate_video:
        print("\n" + "-" * 80)
        print("Generating simulation comparison video...")
        print("-" * 80)

        # create subdirectories for calibrated simulation results
        exact_sim_dir = f"{scenario_dir}/simulation_exact_data_calibration"
        noreg_sim_dir = f"{scenario_dir}/simulation_noisy_noreg_calibration"
        reg_sim_dir = f"{scenario_dir}/simulation_noisy_reg_calibration"
        os.makedirs(exact_sim_dir, exist_ok=True)
        os.makedirs(noreg_sim_dir, exist_ok=True)
        os.makedirs(reg_sim_dir, exist_ok=True)

        # run and save simulation with exact-calibration parameters
        print("  Saving simulation with exact-calibration parameters...")
        sim = Simulation(network, metanet, calibrated_params_exact)
        sim.run(
            duration=duration,
            dt=dt,
            preferred_cell_size=preferred_cell_size,
            origin_demands=origin_demands,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
            plot_results=True,
            show_plots=False,
            results_dir=exact_sim_dir,
        )

        # run and save simulation with no-regularization parameters
        print("  Saving simulation with no-regularization parameters...")
        sim = Simulation(network, metanet, calibrated_params_noisy_noreg)
        sim.run(
            duration=duration,
            dt=dt,
            preferred_cell_size=preferred_cell_size,
            origin_demands=origin_demands,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
            plot_results=True,
            show_plots=False,
            results_dir=noreg_sim_dir,
        )

        # run and save simulation with regularization parameters
        print("  Saving simulation with regularization parameters...")
        sim = Simulation(network, metanet, calibrated_params_noisy_reg)
        sim.run(
            duration=duration,
            dt=dt,
            preferred_cell_size=preferred_cell_size,
            origin_demands=origin_demands,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
            plot_results=True,
            show_plots=False,
            results_dir=reg_sim_dir,
        )

        # generate comparison video
        print("  Generating comparison video...")
        result_files = [
            ground_truth_filepath,
            os.path.join(exact_sim_dir, "simulation_results.json"),
            os.path.join(noreg_sim_dir, "simulation_results.json"),
            os.path.join(reg_sim_dir, "simulation_results.json"),
        ]

        labels = [
            "Ground Truth",
            "Exact Data",
            "Noisy (No Regularization)",
            "Noisy (With Regularization)",
        ]

        comparison_video_path = os.path.join(scenario_dir, "simulation_comparison.avi")
        sim.visualize_comparison(
            result_filepaths=result_files,
            labels=labels,
            output_filepath=comparison_video_path,
            fps=25,
            subsampling=2,
            figsize=(16, 12),
            dpi=150,
        )
        print(f"  Comparison video saved to: {comparison_video_path}")
    else:
        print("\n" + "-" * 80)
        print("Skipping video generation (use --generate-video to enable)")
        print("-" * 80)

    # ! 11) Summary for this scenario
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
    """Run the calibration demonstration.

    Executes 8 calibration experiments across 2 network scenarios (A and C):
    - Experiment 1: Exact ground truth data (optional multi-start)
    - Experiment 2: Noisy data without regularization (optional multi-start)
    - Experiment 3: Noisy data with Tikhonov regularization
    - Experiment 4: Noisy data with link-specific alpha parameters

    Use --parameter-search flag to enable multi-start parameter search with
    Latin Hypercube Sampling for Experiments 1 and 2.
    """
    # track existing simulation_results folders before starting
    existing_sim_dirs = set(glob.glob("results/simulation_results_*"))

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="METANET Model Parameter Calibration Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parameter-search",
        action="store_true",
        help="Use multi-start parameter search with Latin Hypercube Sampling for initialization (default: 40 samples per experiment)",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate comparison videos showing all four simulations side-by-side (can be time-consuming)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("METANET Model Parameter Calibration Demo")
    print("=" * 80)
    print("\nTesting scenarios A and C with exact and noisy measurements")
    print("Total experiments: 8 (2 scenarios × 4 conditions)")

    if args.parameter_search:
        print("\n*** MULTI-START PARAMETER SEARCH MODE ENABLED ***")
        print("  - Exact data: Multi-start parameter search (40 LHS samples)")
        print("  - Noisy data (no reg): Multi-start parameter search (40 LHS samples)")
        print("  - Noisy data (with reg): Standard (single initialization)")
        print("  - Link-specific alpha: Standard (single initialization)")
    else:
        print("\n*** STANDARD MODE (single initialization) ***")

    # common simulation parameters
    dt = 10.0 / 3600  # hours (10 seconds)
    duration = 5000.0 / 3600  # hours
    preferred_cell_size = 0.5  # km

    # ground truth METANET parameters (same for both scenarios)
    true_params: METANETParams = {
        "vf": 120.0,  # km/h
        "qc_lane": 2000.0,  # veh/h/lane
        "rho_jam": 150.0,  # veh/km/lane
        "tau": 22.0 / 3600,  # hours
        "nu": 15.0,
        "kappa": 10.0,
        "delta": 1.4,
        "phi": 10.0,
        "alpha": 2.0,
    }

    # initial guess (slightly perturbed)
    initial_params: METANETParams = {
        "vf": 100.0,  # km/h
        "qc_lane": 1500.0,  # veh/h/lane
        "rho_jam": 120.0,  # veh/km/lane
        "tau": 10.0 / 3600,
        "nu": 20.0,
        "kappa": 20.0,
        "delta": 1.0,
        "phi": 1.0,
        "alpha": 1.0,
    }

    print("\nTrue parameters:")
    print(f"  vf    = {true_params['vf']:.2f} km/h")
    print(f"  qc_lane = {true_params['qc_lane']:.2f} veh/h/lane")
    print(f"  rho_jam = {true_params['rho_jam']:.2f} veh/km/lane")
    print(f"  tau   = {true_params['tau']:.6f}")
    print(f"  nu    = {true_params['nu']:.2f}")
    print(f"  kappa = {true_params['kappa']:.2f}")
    print(f"  delta = {true_params['delta']:.2f}")
    print(f"  phi   = {true_params['phi']:.2f}")
    print(f"  alpha = {true_params['alpha']:.2f}")

    # create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ! Scenario A
    network_a, metadata_a, _ = setup_network_a()
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
        use_parameter_search=args.parameter_search,
        generate_video=args.generate_video,
    )

    # ! Scenario C (with lane drop/bottleneck)
    network_c, metadata_c, _ = setup_network_c()
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
        use_parameter_search=args.parameter_search,
        generate_video=args.generate_video,
    )

    # clean up simulation_results folders created during this calibration run
    print("\n" + "-" * 80)
    print("Cleaning up temporary simulation_results folders...")
    print("-" * 80)
    current_sim_dirs = set(glob.glob("results/simulation_results_*"))
    new_sim_dirs = current_sim_dirs - existing_sim_dirs

    if new_sim_dirs:
        print(f"  Found {len(new_sim_dirs)} new simulation_results folder(s) to remove")
        for sim_dir in new_sim_dirs:
            try:
                shutil.rmtree(sim_dir)
                print(f"    Deleted: {sim_dir}")
            except Exception as e:
                print(f"    Failed to delete {sim_dir}: {e}")
    else:
        print("  No new simulation_results folders were created during this run")

    # overall summary
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: results/calibration_{timestamp}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
