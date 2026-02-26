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
from typing import Callable
from datetime import datetime

from traffic_flow_models import METANET, METANETParams, Network
from demo.scenarios import (
    mainline_demand_a,
    mainline_demand_c,
    onramp_demand_a,
    onramp_demand_c,
    setup_network_ab,
    setup_network_c,
)


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

    calibrated_params_exact, result_exact = network.calibrate_model_params(
        ground_truth_filepath=ground_truth_filepath,
        model=metanet,
        initial_params=initial_params,
        window_size=30,
        model_options={"link_specific_alpha": False},  # use global alpha
        regularization_weight=0.0,
        verbose=True,
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
    print("    Flows:     ±5% (relative)")
    print("    Densities: ±10% (relative)")
    print("    Speeds:    ±5% (relative)")

    np.random.seed(42)  # for reproducibility
    noisy_data = ground_truth_data.copy()

    # add noise to flows
    for link_id in noisy_data["state_time_series"]["flows"].keys():
        flows = np.array(noisy_data["state_time_series"]["flows"][link_id])
        noise = np.random.normal(0, 0.05, flows.shape)
        noisy_flows = flows * (1 + noise)
        noisy_flows = np.maximum(noisy_flows, 0)
        noisy_data["state_time_series"]["flows"][link_id] = noisy_flows.tolist()

    # add noise to densities
    for link_id in noisy_data["state_time_series"]["densities"].keys():
        densities = np.array(noisy_data["state_time_series"]["densities"][link_id])
        noise = np.random.normal(0, 0.10, densities.shape)
        noisy_densities = densities * (1 + noise)
        noisy_densities = np.maximum(noisy_densities, 0)
        noisy_data["state_time_series"]["densities"][link_id] = noisy_densities.tolist()

    # add noise to speeds
    for link_id in noisy_data["state_time_series"]["speeds"].keys():
        speeds = np.array(noisy_data["state_time_series"]["speeds"][link_id])
        noise = np.random.normal(0, 0.05, speeds.shape)
        noisy_speeds = speeds * (1 + noise)
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

    calibrated_params_noisy_noreg, result_noisy_noreg = network.calibrate_model_params(
        ground_truth_filepath=noisy_filepath,
        model=metanet,
        initial_params=initial_params,
        window_size=30,
        model_options={"link_specific_alpha": False},
        regularization_weight=0.0,
        verbose=True,
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

    calibrated_params_noisy_reg, result_noisy_reg = network.calibrate_model_params(
        ground_truth_filepath=noisy_filepath,
        model=metanet,
        initial_params=initial_params,
        window_size=30,
        model_options={"link_specific_alpha": False},
        regularization_weight=0.01,
        verbose=True,
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

    calibrated_params_noisy_link_alpha, result_noisy_link_alpha = (
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

    # ! 7) Summary for this scenario
    print("\n" + "=" * 80)
    print(f"SUMMARY: {scenario_name}")
    print("=" * 80)

    print("\nCalibration Performance:")
    print(f"  Exact data:")
    print(f"    Final cost:    {result_exact.cost:.6e}")
    print(f"    Iterations:    {result_exact.nfev}")
    print(f"    Success:       {result_exact.success}")

    print(f"\n  Noisy data - No regularization:")
    print(f"    Final cost:    {result_noisy_noreg.cost:.6e}")
    print(f"    Iterations:    {result_noisy_noreg.nfev}")
    print(f"    Success:       {result_noisy_noreg.success}")

    print(f"\n  Noisy data - With regularization (λ=0.01):")
    print(f"    Final cost:    {result_noisy_reg.cost:.6e}")
    print(f"    Iterations:    {result_noisy_reg.nfev}")
    print(f"    Success:       {result_noisy_reg.success}")

    print(f"\n  Noisy data - Link-specific alpha (λ=0.01):")
    print(f"    Final cost:    {result_noisy_link_alpha.cost:.6e}")
    print(f"    Iterations:    {result_noisy_link_alpha.nfev}")
    print(f"    Success:       {result_noisy_link_alpha.success}")

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
