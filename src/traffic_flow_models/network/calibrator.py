"""
Calibrator class for traffic flow model parameter estimation.

This module contains helper methods for model calibration that have been extracted
from the Network class to improve maintainability and separation of concerns.
"""

from __future__ import annotations

import os
import math
import casadi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import qmc
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from scipy.optimize import OptimizeResult
from typing import TYPE_CHECKING, Tuple, Union, Callable
from traffic_flow_models.network import (
    Origin,
    Onramp,
    MotorwayLink,
    Offramp,
    Destination,
    Simulation,
)

if TYPE_CHECKING:
    from traffic_flow_models import Network, CTM, METANET, METANETParams


class Calibrator:
    """
    Calibrator class providing helper methods for traffic flow model parameter estimation.

    This class contains the calibration-related methods extracted from the Network class.
    It is instantiated with a reference to the network being calibrated.

    Attributes:
        network: Network instance to calibrate
    """

    def __init__(self, network: "Network"):
        """
        Initialize the Calibrator.

        Args:
            network: Network instance to calibrate
        """
        self.network = network

    def _infer_param_names(self, model, model_options: dict | None = None) -> list[str]:
        """Return ordered calibration parameter names by delegating to the model.

        This method calls `model.get_calibration_param_names(network, model_options)`
        and returns the result.

        Args:
            model: Model instance implementing `get_calibration_param_names`.
            model_options: Optional model-specific options forwarded to the model.

        Raises:
            NotImplementedError: If the model does not implement
                `get_calibration_param_names`.
        """
        if not hasattr(model, "get_calibration_param_names"):
            raise NotImplementedError(
                f"Model {type(model).__name__} must implement get_calibration_param_names"
            )

        names = model.get_calibration_param_names(
            network=self.network, model_options=model_options
        )
        return names

    def analyze_parameter_correlation(
        self,
        result: OptimizeResult,
        param_names: list[str],
        save_dir: str,
        filename: str = "parameter_correlation_analysis.png",
        title: str | None = None,
    ) -> None:
        """Analyze and visualize parameter correlations from calibration Jacobian.

        Computes and plots:
        - Parameter correlation matrix (normalized covariance)
        - Condition number of information matrix (identifiability measure)
        - Singular values of Jacobian (parameter ranking by sensitivity)

        Args:
            result: scipy OptimizeResult from least_squares calibration
            param_names: List of parameter names in order
            save_dir: Directory to save plots
            filename: Name of the output file (default: "parameter_correlation_analysis.png")
            title: Optional title for the plot (default: "Parameter Correlation Analysis")
        """

        print("\n" + "-" * 80)
        print("Parameter Correlation Analysis")
        print("-" * 80)

        # extract Jacobian at solution
        jac = result.jac
        if jac is None:
            print("  ⚠ WARNING: Jacobian not available in optimization result")
            print(
                "  This usually means the optimization terminated without computing the Jacobian."
            )
            print(f"  Optimization status: {result.status}")
            print(f"  Optimization message: {result.message}")
            print(f"  Success: {result.success}")
            print(f"  Number of function evaluations: {result.nfev}")
            print("  Correlation analysis cannot be performed.")
            return

        print(f"  Jacobian shape: {jac.shape}")
        print(f"  Optimization successful: {result.success}")

        # compute information matrix (Hessian approximation): H = J^T J
        info_matrix = jac.T @ jac

        # compute condition number
        cond_number = np.linalg.cond(info_matrix)
        print(f"  Condition number: {cond_number:.2e}")
        if cond_number > 1e6:
            print(
                "  ⚠ WARNING: Poor conditioning suggests parameter identifiability issues"
            )
        elif cond_number > 1e3:
            print(
                "  ⚠ Moderate conditioning - some parameters may be weakly identifiable"
            )
        else:
            print("  ✓ Good conditioning - parameters are well identifiable")

        # try to compute covariance matrix (approximate, using residual variance estimate)
        # Cov(θ) ≈ σ² (J^T J)^(-1) where σ² = RSS / (n - p)
        n_residuals = jac.shape[0]
        n_params = jac.shape[1]
        residual_variance = result.cost / max(1, n_residuals - n_params)

        try:
            cov_matrix = residual_variance * np.linalg.inv(info_matrix)
        except np.linalg.LinAlgError:
            print(
                "  ⚠ WARNING: Information matrix is singular - cannot compute covariance"
            )
            print("  This indicates severe parameter identifiability issues.")
            print("  Correlation analysis cannot be performed.")
            return

        # compute correlation matrix from covariance
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # print parameter uncertainties
        print("\n  Parameter uncertainties (std dev):")
        for i, name in enumerate(param_names):
            print(f"    {name}: ±{std_devs[i]:.6f}")

        # identify highly correlated pairs
        print("\n  Highly correlated parameter pairs (|corr| > 0.8):")
        found_high_corr = False
        for i in range(n_params):
            for j in range(i + 1, n_params):
                if abs(corr_matrix[i, j]) > 0.8:
                    print(
                        f"    {param_names[i]} - {param_names[j]}: {corr_matrix[i, j]:.3f}"
                    )
                    found_high_corr = True
        if not found_high_corr:
            print("    None found (good - parameters are relatively independent)")

        # compute singular values
        singular_values = np.linalg.svd(jac, compute_uv=False)
        print("\n  Singular values (sorted, measure of parameter sensitivity):")
        for i, sv in enumerate(singular_values[:n_params]):
            print(f"    σ_{i + 1}: {sv:.4e}")

        # plot correlation matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # correlation matrix heatmap
        ax = axes[0]
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_params))
        ax.set_yticks(range(n_params))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_yticklabels(param_names)
        ax.set_title("Parameter Correlation Matrix", fontweight="bold")

        # add correlation values as text
        for i in range(n_params):
            for j in range(n_params):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                    fontsize=9,
                )

        plt.colorbar(im, ax=ax, label="Correlation")

        # singular values bar plot
        ax = axes[1]
        ax.bar(range(1, n_params + 1), singular_values[:n_params])
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Magnitude")
        ax.set_title("Singular Values (Parameter Sensitivities)", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # add main title
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

        plt.tight_layout()

        # save figure
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n  Correlation analysis plot saved to: {plot_path}")

    def plot_parameter_search_convergence(
        self,
        results_list: list[dict],
        save_dir: str,
        filename: str = "parameter_search_convergence.png",
        title: str | None = None,
    ) -> None:
        """Plot convergence statistics across multi-start parameter search runs.

        Args:
            results_list: List of dictionaries with parameter search results
            save_dir: Directory to save plots
            filename: Name of the output file (default: "parameter_search_convergence.png")
            title: Optional title for the plot (default: "Multi-Start Parameter Search Convergence")
        """
        print("\n  Creating parameter search convergence plot...")

        successful_runs = [r for r in results_list if r["success"]]
        if not successful_runs:
            print("    No successful runs to plot")
            return

        n_runs = len(results_list)
        run_indices = list(range(1, n_runs + 1))

        # extract metrics
        costs = [r.get("cost", float("inf")) for r in results_list]
        nfevs = [r.get("nfev", 0) for r in results_list]
        optimality = [r.get("optimality", float("inf")) for r in results_list]
        success_markers = [1 if r.get("success", False) else 0 for r in results_list]

        # create figure with 3 subplots
        _, axes = plt.subplots(3, 1, figsize=(14, 10))

        # subplot 1: final cost
        ax = axes[0]
        colors = ["green" if s else "red" for s in success_markers]
        ax.scatter(run_indices, costs, c=colors, alpha=0.7, s=50)
        ax.set_ylabel("Final Cost", fontweight="bold")
        ax.set_yscale("log")
        plot_title = title if title else "Multi-Start Parameter Search Convergence"
        ax.set_title(plot_title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(
            [
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="g", markersize=8
                ),
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="r", markersize=8
                ),
            ],
            ["Success", "Failed"],
            loc="best",
        )

        # subplot 2: function evaluations
        ax = axes[1]
        ax.bar(run_indices, nfevs, color=colors, alpha=0.7)
        ax.set_ylabel("Function Evaluations", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # subplot 3: optimality
        ax = axes[2]
        ax.scatter(run_indices, optimality, c=colors, alpha=0.7, s=50)
        ax.set_ylabel("Optimality", fontweight="bold")
        ax.set_xlabel("Configuration Index", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # add configuration labels on x-axis (rotate for readability)
        for ax in axes:
            ax.set_xlim(0, n_runs + 1)
            if n_runs <= 20:  # only show labels if not too many
                config_labels = [
                    (
                        r.get("config_str", f"Config {i}")[:20] + "..."
                        if len(r.get("config_str", "")) > 20
                        else r.get("config_str", f"Config {i}")
                    )
                    for i, r in enumerate(results_list, 1)
                ]
                ax.set_xticks(
                    run_indices[:: max(1, n_runs // 10)]
                )  # show every Nth label
                ax.set_xticklabels(
                    config_labels[:: max(1, n_runs // 10)],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )

        plt.tight_layout()

        # save figure
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Saved to: {plot_path}")

    def plot_parameter_history(
        self,
        param_history: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        param_names: list[str],
        save_dir: str,
        filename: str = "parameter_convergence.png",
        title: str | None = None,
    ) -> None:
        """Plot parameter values across function evaluations with bounds.

        Creates a single figure overlaying all parameter value series (one line per
        parameter) and draws horizontal lines for the parameter bounds. Saves the
        figure to `save_dir/filename`.

        Args:
            param_history: 2-D array (n_evals x n_params) of parameter vectors.
            lower_bounds: 1-D array of lower bounds for each parameter.
            upper_bounds: 1-D array of upper bounds for each parameter.
            param_names: List of parameter names (length >= n_params or will be
                         generated as needed).
            save_dir: Directory to save the plot.
            filename: Output filename.
            title: Optional plot title.
        """
        ph = np.asarray(param_history)
        if ph.ndim != 2:
            raise ValueError("param_history must be a 2-D array (n_evals x n_params)")

        n_evals, n_params = ph.shape

        _, ax = plt.subplots(figsize=(12, 6))

        cmap = plt.get_cmap("tab20")
        for i in range(n_params):
            color = cmap(i % 20)
        if ph.ndim != 2:
            raise ValueError("param_history must be a 2-D array (n_evals x n_params)")

        n_evals, n_params = ph.shape

        # layout: use 2 columns by default for a compact overview
        ncols = 2
        nrows = math.ceil(n_params / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False)
        axes_list = axes.flatten()

        cmap = plt.get_cmap("tab20")
        x = np.arange(n_evals)

        for i in range(n_params):
            ax = axes_list[i]
            color = cmap(i % 20)
            name = param_names[i] if i < len(param_names) else f"p{i}"

            ax.plot(x, ph[:, i], color=color, linewidth=1.6)

            # bounds (drawn as thin gray lines for clarity)
            try:
                lb = float(lower_bounds[i])
            except Exception:
                lb = None
            try:
                ub = float(upper_bounds[i])
            except Exception:
                ub = None

            if lb is not None:
                ax.axhline(lb, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
            if ub is not None:
                ax.axhline(ub, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.25)

            # only show x-tick labels on bottom row
            row_idx = i // ncols
            if row_idx != nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Function Evaluation", fontsize=10)

        # hide any unused subplots
        for j in range(n_params, len(axes_list)):
            axes_list[j].set_visible(False)

        if title:
            plt.suptitle(title, fontsize=13, fontweight="bold")

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Saved parameter-history plot to: {plot_path}")

    def extract_measurable_states(
        self,
        state_history: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], list[Tuple[str, str, int]]]:
        """Extract measurable quantities (flows, densities, speeds) from state history.

        This helper method extracts only the measurable state components from the
        full state history, excluding queue states which are not properly observable in
        microsimulation. The extracted states include per-cell flows, densities,
        and speeds for all motorway links.

        Args:
            state_history: 2-D array of state vectors over time (state_dim x num_timesteps).

        Returns:
            Tuple containing:
            - measurable_states: 2-D array of measurable state components (measurable_dim x num_timesteps)
            - index_map: List of tuples (link_id, quantity_type, cell_index) describing
              the ordering of measurable states, where quantity_type is 'flow', 'density', or 'speed'.

        Raises:
            ValueError: If state_history dimensions are invalid.
        """

        if state_history.ndim != 2:
            raise ValueError("state_history must be a 2-D array")

        measurable_indices: list[int] = []
        index_map: list[Tuple[str, str, int]] = []
        current_idx = 0

        # extract measurable states node by node in the same order as state vector construction
        for node in self.network.list_nodes():
            # skip incoming origin/onramp flows and queues (not part of measurable set)
            for link in node.incoming:
                if isinstance(link, Origin):
                    current_idx += 2  # flow(1) + queue(1)
                elif isinstance(link, Onramp):
                    current_idx += 2  # flow(1) + queue(1)

            # extract motorway link flows, densities, speeds
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    num_cells = len(link)

                    # flows
                    for cell_idx in range(num_cells):
                        measurable_indices.append(current_idx + cell_idx)
                        index_map.append((link.id, "flow", cell_idx))
                    current_idx += num_cells

                    # densities
                    for cell_idx in range(num_cells):
                        measurable_indices.append(current_idx + cell_idx)
                        index_map.append((link.id, "density", cell_idx))
                    current_idx += num_cells

                    # speeds
                    for cell_idx in range(num_cells):
                        measurable_indices.append(current_idx + cell_idx)
                        index_map.append((link.id, "speed", cell_idx))
                    current_idx += num_cells

                elif isinstance(link, Offramp):
                    current_idx += 2  # flow(1) + queue(1)
                elif isinstance(link, Destination):
                    current_idx += 1  # flow(1)

        # extract measurable states from state_history
        measurable_states = state_history[measurable_indices, :]
        return measurable_states, index_map

    def compute_calibration_residuals(
        self,
        param_vec: NDArray[np.float64],
        model: Union["CTM", "METANET"],
        system: casadi.Function,
        ground_truth_states: NDArray[np.float64],
        ground_truth_disturbances: NDArray[np.float64],
        measurable_indices: NDArray[np.float64],
        window_indices: list[int],
        model_options: dict | None = None,
    ) -> NDArray[np.float64]:
        """Compute residuals between model predictions and ground truth over a time window.

        This method simulates the model forward in time using the provided parameters
        and compares the predicted measurable states (flows, densities, speeds) with
        the ground truth data. The residuals are computed only for the measurable
        quantities, excluding queue states.

        Args:
            param_vec: 1-D array of calibration parameters in model-specific format.
            model: Macroscopic traffic flow model.
            system: CasADi function implementing the network update step.
            ground_truth_states: 2-D array of full ground truth state vectors (state_dim x num_timesteps).
            ground_truth_disturbances: 2-D array of disturbances (disturbance_dim x num_timesteps-1).
            measurable_indices: 2-D array of measurable ground truth states (measurable_dim x num_timesteps).
            window_indices: List of timestep indices defining the calibration window.
            model_options: Dictionary of model-specific options passed to the model's
                          prepare_system_params method.

        Returns:
            1-D array of residuals (predicted - measured) for all measurable states
            across all timesteps in the window.
        """
        # convert calibration parameter vector to system parameter vector format
        # (model-specific conversion handled by the model)
        system_param_vec = model.prepare_system_params(
            param_vec=param_vec,
            network=self.network,
            model_options=model_options,
        )

        # simulate forward through the window using multi-step prediction
        # (start from ground truth only at window beginning, then propagate predictions)
        residuals_list = []

        # initialize with ground truth at window start
        x_current_predicted = ground_truth_states[:, window_indices[0]]

        for _, t_idx in enumerate(window_indices[:-1]):
            # get disturbance at current time
            d_current = ground_truth_disturbances[:, t_idx]

            # predict next state using model with expanded parameter vector
            x_next_predicted = system(system_param_vec, x_current_predicted, d_current)
            x_next_predicted = np.array(x_next_predicted).flatten()

            # extract measurable components from ground truth
            predicted_measurable = measurable_indices[:, t_idx + 1]

            # compute residuals for measurable states only
            residuals = []
            measurable_idx = 0
            current_idx = 0

            for node in self.network.list_nodes():
                # skip incoming origin/onramp flows and queues
                for link in node.incoming:
                    if isinstance(link, Origin):
                        current_idx += 2
                    elif isinstance(link, Onramp):
                        current_idx += 2

                # extract motorway link flows, densities, speeds
                for link in node.outgoing:
                    if isinstance(link, MotorwayLink):
                        num_cells = len(link)
                        # flows, densities, speeds
                        for _ in range(num_cells * 3):
                            residuals.append(
                                x_next_predicted[current_idx]
                                - predicted_measurable[measurable_idx]
                            )
                            current_idx += 1
                            measurable_idx += 1
                    elif isinstance(link, Offramp):
                        current_idx += 2
                    elif isinstance(link, Destination):
                        current_idx += 1

            residuals_list.extend(residuals)

            # use predicted state as current state for next iteration (multi-step prediction)
            x_current_predicted = x_next_predicted

        return np.array(residuals_list, dtype=np.float64)

    def calibrate_model_params(
        self,
        ground_truth_filepath: str,
        model: Union["CTM", "METANET"],
        initial_params: METANETParams | None = None,
        window_size: int = 50,
        stride: int | None = None,
        param_bounds: Tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        model_options: dict | None = None,
        regularization_weight: float = 0.0,
        max_nfev: int = 1000,
        verbose: bool = True,
        use_parameter_search: bool = False,
        n_samples: int | None = None,
        plot_convergence: bool | str = False,
        plot_correlation: bool | str = False,
        plot_param_history: bool | str = False,
        save_dir: str | None = None,
        convergence_title: str | None = None,
        correlation_title: str | None = None,
        # If True (default) use disturbance time series saved in the ground-truth
        # file. If False, the caller must provide callable disturbance inputs
        # via the arguments below (origin_demands_fn, turning_rates_fn,
        # flow_boundary_conditions_fn, density_boundary_conditions_fn).
        use_disturbance_from_file: bool = True,
        origin_demands_fn: dict[str, Callable[[float], float]] | None = None,
        turning_rates_fn: dict[str, Callable[[float], dict[str, float]]] | None = None,
        flow_boundary_conditions_fn: dict[str, Callable[[float], float]] | None = None,
        density_boundary_conditions_fn: (
            dict[str, Callable[[float], float]] | None
        ) = None,
    ) -> Tuple["METANETParams", OptimizeResult, NDArray[np.float64]]:
        """Calibrate model parameters using ground truth simulation data.

        This method performs parameter estimation by minimizing the prediction error
        between the model and provided ground truth data (e.g. microsimulation). It uses a
        sliding window approach to balance computational efficiency with capturing
        multi-step dynamics. Only measurable quantities (flows, densities, speeds)
        are used for calibration, excluding queue states.

        The optimization is performed using scipy's Trust Region Reflective algorithm
        (least_squares with method='trf'), which handles bounded nonlinear least squares
        problems robustly.

        The method is model-agnostic and uses the model's calibration interface methods:
        - get_default_calibration_params(): Get default initial parameters
        - get_calibration_bounds(): Get parameter bounds
        - prepare_calibration_params(): Convert params to optimization vector
        - parse_calibration_params(): Convert optimization vector to params

        Multi-Start Parameter Search Mode:
        -----------------------------------
        When use_parameter_search=True, the method uses Latin Hypercube Sampling (LHS)
        to generate diverse initial parameter configurations within the parameter bounds.
        This multi-start approach helps avoid local minima by exploring the parameter
        space efficiently. The best solution across all runs is returned.

        Performance Tips for Noisy Data:
        ---------------------------------
        1. METANET: Set model_options={'link_specific_alpha': False} to reduce DOF
        2. Increase window_size (e.g., 50-100) to capture more dynamics
        3. Add regularization (regularization_weight=0.01-0.1) to prevent overfitting
        4. Provide good initial_params based on domain knowledge or literature values
        5. Try multiple random initializations and pick best result
        6. Consider filtering/smoothing ground truth data before calibration

        Args:
            ground_truth_filepath: Path to JSON file containing ground truth simulation results
                (must be in the format produced by save_simulation_results_json).
            model: Macroscopic flow model instance. Must implement calibration interface.
            initial_params: Initial guess for model parameters. If None, uses model defaults.
                Ignored when use_parameter_search=True.
            window_size: Number of timesteps per calibration window. Larger windows capture
                more dynamics but increase computational cost. Default: 50.
            stride: Stride between consecutive calibration windows. Must be <= window_size to
                avoid ignoring data points. If None, defaults to window_size // 2 (50% overlap).
                Smaller stride values increase overlap and computational cost but may improve
                calibration accuracy. Default: None (window_size // 2).
            param_bounds: Optional tuple of (lower_bounds, upper_bounds) as numpy arrays.
                If None, uses model's default bounds.
            model_options: Dictionary of model-specific calibration options. For METANET:
                - 'link_specific_alpha' (bool): If True, calibrates separate alpha for each
                  link. If False (default), uses single global alpha. Setting to False
                  reduces degrees of freedom and often improves robustness with noisy data.
            regularization_weight: L2 regularization weight (lambda) for Ridge regression.
                Penalizes deviation from initial_params: cost = ||residuals||^2 + lambda*||p-p0||^2.
                Helps prevent overfitting to noisy data. Typical values: 0.0 (none) to 0.1 (strong).
                Default: 0.0.
            max_nfev: Maximum number of function evaluations for the optimizer. Default: 1000.
            verbose: If True, print calibration progress and results.
            use_parameter_search: If True, perform multi-start optimization using Latin Hypercube
                Sampling to generate initial parameter configurations. If False, use single
                initial_params. Default: False.
            n_samples: Number of Latin Hypercube samples when use_parameter_search=True. If None,
                defaults to min(20, 2^n_params) to balance exploration and computational cost.
                Default: None.
            plot_convergence: If True and use_parameter_search=True and save_dir is provided,
                generate convergence plot showing cost, evaluations, and optimality across
                parameter search runs. Can also be a string specifying the output filename.
                Default: False.
            plot_correlation: If True and save_dir is provided, generate parameter correlation
                analysis plot showing correlation matrix and singular values. Can also be a
                string specifying the output filename. Default: False.
            save_dir: Directory to save diagnostic plots. If None, no plots are saved.
                Default: None.
            convergence_title: Optional custom title for parameter search convergence plot.
                Default: None.
            correlation_title: Optional custom title for parameter correlation plot.
                Default: None.
            plot_param_history: If True and `save_dir` is provided, save a single
                parameter-history plot that shows the value of each optimization
                parameter across function evaluations and the parameter bounds.
                Can also be a string specifying the output filename. Default: False.

        Returns:
            Tuple containing:
            - calibrated_params: Dictionary of calibrated model parameters.
            - result: scipy OptimizeResult object with detailed optimization information
              (cost, iterations, termination status, etc.).
            - param_history: 2-D array of parameter vectors over optimization iterations
              (shape: num_evaluations x num_parameters).

        Raises:
            NotImplementedError: If model does not support parameter calibration.
            ValueError: If ground truth file is invalid or incompatible with network structure.
            FileNotFoundError: If ground_truth_filepath does not exist.
        """
        if model_options is None:
            model_options = {}

        # set default stride if not provided
        if stride is None:
            stride = window_size // 2

        # validate stride parameter
        if stride > window_size:
            raise ValueError(
                f"stride ({stride}) must be <= window_size ({window_size}) "
                "to avoid ignoring data points"
            )
        if stride < 1:
            raise ValueError(f"stride ({stride}) must be >= 1")

        # ! Multi-Start Parameter Search Mode: Use LHS to explore parameter space
        if use_parameter_search:
            if verbose:
                print("\n" + "=" * 80)
                print("MULTI-START PARAMETER SEARCH CALIBRATION")
                print("=" * 80)

            # set up parameter bounds first to determine grid size
            if param_bounds is None:
                try:
                    lower_bounds, upper_bounds = model.get_calibration_bounds(
                        network=self.network, model_options=model_options
                    )
                except (AttributeError, NotImplementedError):
                    raise NotImplementedError(
                        f"Model {type(model).__name__} does not support calibration bounds"
                    )
            else:
                lower_bounds, upper_bounds = param_bounds

            n_params = len(lower_bounds)

            # determine number of LHS samples
            if n_samples is None:
                # default: min(20, 2^n_params) to balance exploration and cost
                n_samples = min(20, 2**n_params)
            total_configs = n_samples

            if verbose:
                print(f"  Parameters: {n_params}")
                print(f"  Sampling method: Latin Hypercube Sampling (LHS)")
                print(f"  Number of samples: {total_configs}")
                print(f"  Max iterations per run: {max_nfev}")
                print()

            # generate parameter names using the model's authoritative method
            param_names = self._infer_param_names(
                model=model, model_options=model_options
            )

            # generate Latin Hypercube samples in [0, 1]^n_params
            rng = np.random.default_rng(seed=42)
            sampler = qmc.LatinHypercube(d=n_params, rng=rng)
            lhs_samples = sampler.random(n=n_samples)

            # scale samples to parameter bounds
            scaled_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)

            # track results across all parameter search runs
            results_list = []
            best_cost = float("inf")
            best_params = None
            best_result = None
            best_param_history = None

            # test each LHS sample
            for config_idx, initial_vec in enumerate(scaled_samples, start=1):
                # convert to parameter dict for the model
                initial_params_grid = model.parse_calibration_params(
                    param_vec=initial_vec,
                    network=self.network,
                    model_options=model_options,
                )

                # create config string showing parameter values
                config_str = ", ".join(
                    [
                        f"{param_names[i]}={initial_vec[i]:.3f}"
                        for i in range(min(3, n_params))  # show first 3 params
                    ]
                ) + ("..." if n_params > 3 else "")

                if verbose:
                    print(f"  [{config_idx}/{total_configs}] Testing: {config_str}")

                # run calibration with this initialization (recursive call with use_parameter_search=False)
                try:
                    calibrated_params_run, result_run, param_history_run = (
                        self.calibrate_model_params(
                            ground_truth_filepath=ground_truth_filepath,
                            model=model,
                            initial_params=initial_params_grid,
                            window_size=window_size,
                            stride=stride,
                            param_bounds=(lower_bounds, upper_bounds),
                            model_options=model_options,
                            regularization_weight=regularization_weight,
                            max_nfev=max_nfev,
                            verbose=False,  # suppress per-run output
                            use_parameter_search=False,  # disable recursive parameter search
                        )
                    )

                    # store results
                    results_list.append(
                        {
                            "config_str": config_str,
                            "initial_vec": initial_vec.copy(),
                            "cost": result_run.cost,
                            "success": result_run.success,
                            "nfev": result_run.nfev,
                            "optimality": result_run.optimality,
                            "final_params": result_run.x,
                        }
                    )

                    if verbose:
                        print(
                            f"       Cost: {result_run.cost:.4e} | "
                            f"Evals: {result_run.nfev} | "
                            f"Success: {result_run.success}"
                        )

                    # track best solution
                    if result_run.success and result_run.cost < best_cost:
                        best_cost = result_run.cost
                        best_params = calibrated_params_run
                        best_result = result_run
                        best_param_history = param_history_run
                        if verbose:
                            print(f"       *** New best solution! ***")

                except Exception as e:
                    if verbose:
                        print(f"       ERROR: {e}")
                    results_list.append(
                        {
                            "config_str": config_str,
                            "initial_vec": initial_vec.copy(),
                            "cost": float("inf"),
                            "success": False,
                            "nfev": 0,
                            "optimality": float("inf"),
                            "final_params": None,
                        }
                    )

            # compute statistics
            successful_runs = [r for r in results_list if r["success"]]
            costs = [r["cost"] for r in successful_runs]

            if verbose:
                print("\n" + "=" * 80)
                print("PARAMETER SEARCH SUMMARY")
                print("=" * 80)
                print(f"  Successful runs: {len(successful_runs)} / {total_configs}")
                if costs:
                    print(f"  Best cost:       {min(costs):.6e}")
                    print(f"  Worst cost:      {max(costs):.6e}")
                    print(f"  Mean cost:       {np.mean(costs):.6e}")
                    print(f"  Std cost:        {np.std(costs):.6e}")
                else:
                    print("  No successful runs!")

            # check if we found a solution
            if best_params is None or best_result is None or best_param_history is None:
                raise RuntimeError(
                    "Parameter search failed - no successful calibrations"
                )

            if verbose:
                print(
                    f"\n  Parameter search correlation plot requested: {bool(plot_correlation)}"
                )
                print(
                    f"  Parameter search save directory provided: {save_dir is not None}"
                )

            # plot convergence if requested
            if plot_convergence and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                # extract filename from plot_convergence if it's a string, otherwise use default
                convergence_filename = (
                    plot_convergence
                    if isinstance(plot_convergence, str)
                    else "parameter_search_convergence.png"
                )
                self.plot_parameter_search_convergence(
                    results_list=results_list,
                    save_dir=save_dir,
                    filename=convergence_filename,
                    title=convergence_title,
                )

            # plot correlation analysis if requested
            if plot_correlation and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                # extract filename from plot_correlation if it's a string, otherwise use default
                correlation_filename = (
                    plot_correlation
                    if isinstance(plot_correlation, str)
                    else "parameter_correlation_analysis.png"
                )
                self.analyze_parameter_correlation(
                    result=best_result,
                    param_names=param_names,
                    save_dir=save_dir,
                    filename=correlation_filename,
                    title=correlation_title,
                )

            # plot parameter history for the best run if requested
            if (
                plot_param_history
                and save_dir is not None
                and best_param_history is not None
            ):
                os.makedirs(save_dir, exist_ok=True)
                history_filename = (
                    plot_param_history
                    if isinstance(plot_param_history, str)
                    else "parameter_convergence_search_best.png"
                )
                self.plot_parameter_history(
                    param_history=best_param_history,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    param_names=param_names,
                    save_dir=save_dir,
                    filename=history_filename,
                    title=convergence_title,
                )

            return best_params, best_result, best_param_history

        # ! Standard Single-Run Calibration Mode
        if verbose:
            print("\n" + "=" * 80)
            print("PARAMETER CALIBRATION")
            print("=" * 80)

        # ! 1) Load ground truth data
        if verbose:
            print(f"Loading ground truth data from {ground_truth_filepath}")

        # load the simulation results from the corresponding file - depending on the flag
        # regarding the provided structure of disturbance components, either load the full
        # simulation data (state and disturbance histories) from the file, or only the state
        # history with disturbance quantities passed as separate functions to this method
        time_array, state_history, disturbance_history, _ = Simulation.load_results(
            filepath=ground_truth_filepath,
            network=self.network,
            load_mainline_only=(not use_disturbance_from_file),
        )

        # if the caller supplied callable disturbances, build the disturbance
        # history by sampling those callables on the simulation grid. This
        # replaces the file-based disturbance history when requested.
        if not use_disturbance_from_file:
            if (
                origin_demands_fn is None
                or turning_rates_fn is None
                or flow_boundary_conditions_fn is None
                or density_boundary_conditions_fn is None
            ):
                raise ValueError(
                    "When use_disturbance_from_file=False you must provide all required disturbance functions: origin_demands_fn, turning_rates_fn, flow_boundary_conditions_fn, density_boundary_conditions_fn"
                )

            # disturbance vectors are defined on the disturbance time grid
            # which corresponds to `time_array[:-1]` (one fewer than state timesteps)
            times = np.array(time_array, dtype=float)
            if times.size >= 2:
                sample_times = times[:-1]
            else:
                sample_times = times

            disturbance_vecs: list[NDArray[np.float64]] = []

            for t in sample_times:
                # build per-timestep disturbance dictionaries by sampling
                origin_dem_t: dict[str, float] = {}
                for oid, fn in origin_demands_fn.items():
                    origin_dem_t[oid] = float(fn(float(t)))

                # turning rates: prepare a per-node mapping; supply defaults
                # of 1.0 for single-outgoing-link nodes when not provided
                turning_rates_t: dict[str, dict[str, float]] = {}
                for node in self.network.list_nodes():
                    if turning_rates_fn and node.id in turning_rates_fn:
                        node_rates = turning_rates_fn[node.id](float(t))
                    else:
                        # default SISO nodes to 1.0 on their single outgoing link
                        if len(node.outgoing) == 1:
                            node_rates = {node.outgoing[0].id: 1.0}
                        else:
                            raise ValueError(
                                f"Turning-rate function for node {node.id} not provided and node has multiple outgoing links"
                            )

                    # ensure numeric types
                    turning_rates_t[node.id] = {
                        k: float(v) for k, v in node_rates.items()
                    }

                # boundary conditions: sample provided callables or fall back to zeros
                flow_bc_t: dict[str, float] = {}
                density_bc_t: dict[str, float] = {}
                if flow_boundary_conditions_fn is not None:
                    for did, fn in flow_boundary_conditions_fn.items():
                        flow_bc_t[did] = float(fn(float(t)))
                if density_boundary_conditions_fn is not None:
                    for did, fn in density_boundary_conditions_fn.items():
                        density_bc_t[did] = float(fn(float(t)))

                # pack into disturbance vector using network helper
                d_t = self.network.network_dict_to_disturbance_vec(
                    origin_demand_dict=origin_dem_t,
                    turning_rate_dict=turning_rates_t,
                    flow_boundary_condition_dict=flow_bc_t,
                    density_boundary_condition_dict=density_bc_t,
                )

                disturbance_vecs.append(d_t)

            disturbance_history = (
                np.column_stack(disturbance_vecs) if disturbance_vecs else np.array([])
            )

        num_timesteps = len(time_array)
        if verbose:
            print(
                f"  Loaded {num_timesteps} timesteps (duration: {time_array[-1]:.4f})"
            )

        # ! 2) Extract measurable states from ground truth
        measurable_states, index_map = self.extract_measurable_states(state_history)
        num_measurable = measurable_states.shape[0]

        if verbose:
            print(f"  Extracted {num_measurable} measurable state components")
            print(
                f"    (flows, densities, speeds for {sum(1 for _, qty, _ in index_map if qty == 'flow')} cells)"
            )

        # ! 3) Set up initial parameters using model interface
        if initial_params is None:
            initial_params = model.get_default_calibration_params()
            if verbose:
                print("  Using model default initial parameters")
        else:
            if verbose:
                print("  Using provided initial parameters")

        # convert to vector form using model method
        initial_param_vec = model.prepare_calibration_params(
            params=initial_params,
            network=self.network,
            model_options=model_options,
        )

        # set up parameter bounds using model method
        if param_bounds is None:
            lower_bounds, upper_bounds = model.get_calibration_bounds(
                network=self.network, model_options=model_options
            )
            if verbose:
                print("  Using model default parameter bounds")
        else:
            lower_bounds, upper_bounds = param_bounds
            if verbose:
                print("  Using provided parameter bounds")

        if verbose:
            print(f"  Parameter vector size: {len(initial_param_vec)}")

            # display model-specific options
            if model_options:
                print("  Model options:")
                for key, value in model_options.items():
                    print(f"    {key}: {value}")

            # display bounds if available
            if len(lower_bounds) > 0:
                param_nanes = self._infer_param_names(
                    model=model, model_options=model_options
                )
                print("  Parameter bounds:")
                for i, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
                    name = param_nanes[i] if i < len(lower_bounds) else f"param_{i}"
                    print(f"    {name}: [{lower:.2e}, {upper:.2f}]")

            if regularization_weight > 0:
                print(f"  Regularization: λ={regularization_weight}")

        # ! 4) Build CasADi system function
        if verbose:
            print("  Building CasADi system function...")

        # count components for system function
        num_flows = 0
        num_densities = 0
        num_speeds = 0
        num_origins = 0
        num_onramps = 0
        num_offramps = 0
        num_destinations = 0

        for node in self.network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Origin):
                    num_origins += 1
                    num_flows += 1
                elif isinstance(link, Onramp):
                    num_onramps += 1
                    num_flows += 1

            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    num_flows += len(link)
                    num_densities += len(link)
                    num_speeds += len(link)
                elif isinstance(link, Offramp):
                    num_offramps += 1
                    num_flows += 1
                elif isinstance(link, Destination):
                    num_destinations += 1
                    num_flows += 1

        # count splits
        num_splits = sum(len(node.outgoing) for node in self.network.list_nodes())

        # construct the system update function for the chosen model and network structure
        dt = float(time_array[1] - time_array[0]) if len(time_array) > 1 else 0.01
        system = model.network_update_function(
            network=self.network,
            num_flows=num_flows,
            num_densities=num_densities,
            num_speeds=num_speeds,
            num_origins=num_origins,
            num_onramps=num_onramps,
            num_offramps=num_offramps,
            num_splits=num_splits,
            num_destinations=num_destinations,
            dt=dt,
        )

        if verbose:
            print("  System function built successfully")

        # ! 6) Define residual function for optimization
        # track parameter history for convergence analysis
        param_history: list[NDArray[np.float64]] = []

        def residual_func(param_vec: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute residuals over all time windows with optional regularization."""
            # store parameter vector for convergence tracking
            param_history.append(param_vec.copy())

            # use overlapping sliding windows
            # ceiling-style calculation ensures all timesteps are covered
            num_windows = max(1, math.ceil((num_timesteps - window_size) / stride) + 1)
            all_residuals = []
            for w in range(num_windows):
                # define window indices
                start_idx = w * stride
                end_idx = min(start_idx + window_size + 1, num_timesteps)
                window_indices = list(range(start_idx, end_idx))

                if len(window_indices) < 2:
                    continue

                # compute residuals for this window
                window_residuals = self.compute_calibration_residuals(
                    param_vec=param_vec,
                    model=model,
                    system=system,
                    ground_truth_states=state_history,
                    ground_truth_disturbances=disturbance_history,
                    measurable_indices=measurable_states,
                    window_indices=window_indices,
                    model_options=model_options,
                )
                all_residuals.append(window_residuals)

            residuals = np.concatenate(all_residuals) if all_residuals else np.array([])

            # add Tikhonov regularization term if requested
            if regularization_weight > 0:
                # penalize deviation from initial parameters
                regularization_term = np.sqrt(regularization_weight) * (
                    param_vec - initial_param_vec
                )
                residuals = np.concatenate([residuals, regularization_term])

            return residuals

        # ! 6) Run optimization
        if verbose:
            print("\nStarting parameter calibration...")
            print(f"  Window size: {window_size} timesteps")
            print(f"  Number of windows: {max(1, (num_timesteps - 1) // window_size)}")

        result = least_squares(
            fun=residual_func,
            x0=initial_param_vec,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            verbose=2 if verbose else 0,
            max_nfev=max_nfev,
        )

        # ! 7) Extract calibrated parameters using model method
        calibrated_param_vec = result.x
        calibrated_params = model.parse_calibration_params(
            param_vec=calibrated_param_vec,
            network=self.network,
            model_options=model_options,
        )

        if verbose:
            print("\n" + "=" * 80)
            print("CALIBRATION RESULTS")
            print("=" * 80)
            print(f"  Status: {result.message}")
            print(f"  Success: {result.success}")
            print(f"  Function evaluations: {result.nfev}")
            print(f"  Final cost: {result.cost:.6e}")
            print(f"  Optimality: {result.optimality:.2e}")
            print("\nCalibrated parameters:")

            # print parameters (model-agnostic)
            for key, value in calibrated_params.items():
                if isinstance(value, dict):
                    print(f"  {key}: (link-specific)")
                    for link_id, param_val in value.items():
                        print(f"    {link_id}: {param_val:.6f}")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")

        # convert parameter history to array
        param_history_array = np.array(param_history)

        # plot correlation analysis if requested (standard mode)
        if verbose:
            print(f"\n  Correlation plot requested: {bool(plot_correlation)}")
            print(f"  Save directory provided: {save_dir is not None}")
            if plot_correlation and save_dir is not None:
                print(
                    f"  → Generating correlation plot: {plot_correlation if isinstance(plot_correlation, str) else 'parameter_correlation_analysis.png'}"
                )

        if plot_correlation and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            # generate parameter names using the model's authoritative method
            param_names = self._infer_param_names(
                model=model, model_options=model_options
            )

            # extract filename from plot_correlation if it's a string, otherwise use default
            correlation_filename = (
                plot_correlation
                if isinstance(plot_correlation, str)
                else "parameter_correlation_analysis.png"
            )
            self.analyze_parameter_correlation(
                result=result,
                param_names=param_names,
                save_dir=save_dir,
                filename=correlation_filename,
                title=correlation_title,
            )
        elif verbose:
            if not plot_correlation:
                print("  ℹ  Correlation plot not requested (plot_correlation=False)")
            if save_dir is None:
                print("  ℹ  No save directory provided (save_dir=None)")

        # optionally plot parameter history for this single-run calibration
        if plot_param_history and save_dir is not None:
            param_names = self._infer_param_names(
                model=model, model_options=model_options
            )

            history_filename = (
                plot_param_history
                if isinstance(plot_param_history, str)
                else "parameter_convergence.png"
            )
            os.makedirs(save_dir, exist_ok=True)
            self.plot_parameter_history(
                param_history=param_history_array,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                param_names=param_names,
                save_dir=save_dir,
                filename=history_filename,
                title=convergence_title,
            )

        return calibrated_params, result, param_history_array

    def plot_parameter_convergence(
        self,
        param_history_exact: NDArray[np.float64],
        param_history_noreg: NDArray[np.float64],
        param_history_reg: NDArray[np.float64],
        param_names: list[str],
        true_params: "METANETParams",
        save_dir: str,
        filename: str = "convergence_comparison.png",
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
            filename: Name of the output file (default: "convergence_comparison.png")
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
            true_val = true_params[param_name]
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
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Saved to: {plot_path}")
