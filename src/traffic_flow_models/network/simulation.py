"""Simulation class for traffic flow models.

This module contains the Simulation class that handles execution of traffic
simulations, result storage, and visualization. It was extracted from the
Network class to separate concerns between network topology/validation
and simulation execution.
"""

from __future__ import annotations

import os
import json
import cv2
import math
import warnings
import casadi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
from numpy.typing import NDArray
from datetime import datetime
from typing import Callable, Tuple, Union, Any, cast, TYPE_CHECKING

from traffic_flow_models.network import (
    Node,
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
)
from traffic_flow_models.model import CTM, METANET, METANETParams

if TYPE_CHECKING:
    from traffic_flow_models.network import Network


class Simulation:
    """
    Simulation class for executing and visualizing traffic flow simulations.

    This class handles the execution of traffic simulations using either the
    CTM or METANET models, computes performance metrics, stores results,
    and generates visualizations including plots and videos.

    The Simulation class takes a Network instance and a traffic model, then
    provides methods to:
    - Run simulations with specified demands and boundary conditions
    - Save and load simulation results
    - Compute performance metrics (VKT, VHT, average speed)
    - Generate static plots of simulation results
    - Create video visualizations of traffic flow dynamics
    """

    # visualization color constants (RGB tuples in range [0, 255])
    COLOR_BRIGHT_GREEN = (144, 238, 144)  # low density
    COLOR_DARK_GREEN = (0, 100, 0)  # critical density
    COLOR_ORANGE = (255, 165, 0)  # moderate congestion
    COLOR_DARK_RED = (139, 0, 0)  # jam density
    COLOR_CONGESTION_RED = (0.8, 0.0, 0.0)  # ramp queue congestion (normalized)

    def __init__(
        self,
        network: "Network",  # type: ignore  # Forward reference
        model: Union[CTM, METANET],
        model_params: Union[METANETParams, None] = None,
    ):
        """Initialize the Simulation instance.

        Args:
            network: Network instance containing the traffic network topology.
            model: Traffic flow model to use (CTM or METANET).
            model_params: Model parameters for METANET (required for METANET, must be None for CTM).

        Raises:
            ValueError: If model_params is provided for CTM or missing for METANET.
        """
        from traffic_flow_models.network import Network

        if not isinstance(network, Network):
            raise TypeError("network must be a Network instance")

        if not isinstance(model, (CTM, METANET)):
            raise TypeError("model must be a CTM or METANET instance")

        if isinstance(model, CTM) and model_params is not None:
            raise ValueError("CTM model does not accept model_params")

        if isinstance(model, METANET) and model_params is None:
            raise ValueError("METANET model requires model_params to be provided")

        if isinstance(model, METANET) and model_params is not None:
            model.validate_model_params(model_params=model_params)

        self.network = network
        self.model = model
        self.model_params = model_params

        # results stored after run() is called
        self._time_array: NDArray[np.float64] | None = None
        self._state_history: NDArray[np.float64] | None = None
        self._disturbance_history: NDArray[np.float64] | None = None
        self._last_dt: float | None = None
        self._last_duration: float | None = None
        self._last_preferred_cell_size: float | None = None

    # ! Validation helper methods
    # region
    def _validate_state_history_numerical(
        self,
        flows: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        densities: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        speeds: dict[str, NDArray[np.float64]] | dict[str, casadi.SX],
        origin_queues: dict[str, float] | dict[str, casadi.SX],
        onramp_queues: dict[str, float] | dict[str, casadi.SX],
        offramp_queues: dict[str, float] | dict[str, casadi.SX],
    ) -> None:
        """Validate that state-history dictionaries contain numerical values.

        This helper checks that the provided per-link and per-queue dictionaries
        contain numeric NumPy arrays (for flows, densities, speeds) or scalar
        numeric values (for queues). It raises a ValueError if any non-numerical
        entries are found.

        Args:
            flows: Mapping link id -> per-cell flow arrays or CasADi SX slices.
            densities: Mapping motorway link id -> per-cell density arrays or CasADi SX.
            speeds: Mapping motorway link id -> per-cell speed arrays or CasADi SX.
            origin_queues: Mapping origin id -> scalar queue values.
            onramp_queues: Mapping onramp id -> scalar queue values.
            offramp_queues: Mapping offramp id -> scalar queue values.

        Raises:
            ValueError: If any entry is not a numeric NumPy array or numeric scalar.
        """

        if (
            not all(
                isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
                for val in flows.values()
            )
            or not all(
                isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
                for val in densities.values()
            )
            or not all(
                isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating)
                for val in speeds.values()
            )
            or not all(
                isinstance(val, (float, np.floating)) for val in origin_queues.values()
            )
            or not all(
                isinstance(val, (float, np.floating)) for val in onramp_queues.values()
            )
            or not all(
                isinstance(val, (float, np.floating)) for val in offramp_queues.values()
            )
        ):
            raise ValueError("Non-numerical values found in state history.")

    def _validate_disturbance_history_numerical(
        self,
        origin_demands: dict[str, float] | dict[str, casadi.SX],
        turning_rates: dict[str, dict[str, float]] | dict[str, dict[str, casadi.SX]],
        flow_boundary_conditions: dict[str, float] | dict[str, casadi.SX],
        density_boundary_conditions: dict[str, float] | dict[str, casadi.SX],
    ) -> None:
        """Validate that disturbance-history dictionaries contain numerical values.

        This helper checks that the provided per-origin demand values, per-node
        turning rates, and boundary condition dictionaries contain numeric scalar
        values. It raises a ValueError if any non-numerical entries are found.

        Args:
            origin_demands: Mapping origin id -> scalar demand value.
            turning_rates: Mapping node id -> mapping outgoing link id -> turn rate.
            flow_boundary_conditions: Mapping destination id -> downstream flow.
            density_boundary_conditions: Mapping destination id -> downstream density.

        Raises:
            ValueError: If any entry is not a numeric scalar.
        """

        if (
            not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in origin_demands.values()
            )
            or not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in flow_boundary_conditions.values()
            )
            or not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in density_boundary_conditions.values()
            )
        ):
            raise ValueError("Non-numerical values found in disturbance history.")

        # validate turning rates (nested dictionary)
        for node_id, node_rates in turning_rates.items():
            if not isinstance(node_rates, dict):
                raise ValueError(
                    f"Turning rates for node {node_id} must be a dictionary."
                )
            if not all(
                isinstance(val, (float, np.floating, int, np.integer))
                for val in node_rates.values()
            ):
                raise ValueError(
                    f"Non-numerical turning rate values found for node {node_id}."
                )

    def _validate_initial_conditions_numerical(
        self,
        origin_demands: dict[str, Callable[[float], float]],
        turning_rates: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc: dict[str, Callable[[float], float]],
        destination_density_bc: dict[str, Callable[[float], float]],
        initial_flows: dict[str, float | NDArray[np.float64]] | None = None,
        initial_densities: dict[str, float | NDArray[np.float64]] | None = None,
        initial_speeds: dict[str, float | NDArray[np.float64]] | None = None,
    ):
        """Validate presence and basic consistency of initial-condition inputs.

        Ensures that for each node in the network the required callable demand
        and turning-rate functions are provided, and (if arrays of initial
        flows/densities/speeds are supplied) that entries exist for the
        respective links. Raises descriptive ValueError messages on missing
        or inconsistent inputs.

        Args:
            origin_demands: Mapping origin id -> callable(time) -> demand.
            turning_rates: Mapping node id -> callable(time) -> dict[outgoing->rate].
            destination_flow_bc: Mapping destination id -> callable(time) -> flow.
            destination_density_bc: Mapping destination id -> callable(time) -> density.
            initial_flows: Optional mapping link id -> scalar or per-cell array for initial flows.
            initial_densities: Optional mapping link id -> scalar or per-cell array for initial densities.
            initial_speeds: Optional mapping link id -> scalar or per-cell array for initial speeds.

        Raises:
            ValueError: If required functions or initial arrays are missing or inconsistent.
        """

        for node in self.network.list_nodes():
            # validate node structure
            node.validate()

            # validate that origin demands for each origin are provided
            for link in node.incoming:
                if isinstance(link, Origin):
                    if link.id not in origin_demands:
                        raise ValueError(
                            f"Origin demand function for origin {link.id} not provided."
                        )

            # validate that turning rates for each node are provided
            if node.id not in turning_rates and len(node.outgoing) > 1:
                raise ValueError(
                    f"Turning rate function for node {node.id} with multiple incoming and/or outgoing links not provided."
                )

            # validate that destination boundary conditions for each destination are provided
            for link in node.outgoing:
                if isinstance(link, Destination):
                    if link.id not in destination_flow_bc:
                        raise ValueError(
                            f"Destination flow boundary condition function for destination {link.id} not provided."
                        )

                    if link.id not in destination_density_bc:
                        raise ValueError(
                            f"Destination density boundary condition function for destination {link.id} not provided."
                        )

            # validate that initial flows are defined for all links if not None
            if initial_flows is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if link.id not in initial_flows:
                        raise ValueError(
                            f"Initial flow for link {link.id} not provided (required for origins, onramp, offramp, destinations, and motorway links)."
                        )

            # validate that initial densities are defined for all links if not None
            if initial_densities is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if (
                        not isinstance(link, Origin)
                        and not isinstance(link, Onramp)
                        and not isinstance(link, Destination)
                        and link.id not in initial_densities
                    ):
                        raise ValueError(
                            f"Initial density for link {link.id} not provided (required for motorway links)."
                        )

            # validate that initial speeds are defined for all links if not None
            if initial_speeds is not None:
                for link in list(node.incoming) + list(node.outgoing):
                    if (
                        not isinstance(link, Origin)
                        and not isinstance(link, Onramp)
                        and not isinstance(link, Destination)
                        and link.id not in initial_speeds
                    ):
                        raise ValueError(
                            f"Initial speed for link {link.id} not provided (required for motorway links)."
                        )

    # endregion

    # ! Initialization helpers
    # region
    def _augment_network_initialization(
        self,
        initial_flows: dict[str, float | NDArray[np.float64]] | None,
        initial_densities: dict[str, float | NDArray[np.float64]] | None,
        initial_speeds: dict[str, float | NDArray[np.float64]] | None,
        initial_origin_queues: dict[str, float] | None,
        initial_onramp_queues: dict[str, float] | None,
        initial_offramp_queues: dict[str, float] | None,
        turning_rates: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc: dict[str, Callable[[float], float]],
        destination_density_bc: dict[str, Callable[[float], float]],
    ):
        """Prepare and augment initial per-link and per-queue dictionaries.

        This method fills defaults for missing initial states (flows, densities,
        speeds, queues) and ensures that turning-rate and destination boundary
        condition callables are available for all nodes/links. It returns a
        tuple with the initialized dictionaries used by the simulator.

        Args:
            initial_flows: Optional mapping link id -> scalar or per-cell array for initial flows.
            initial_densities: Optional mapping link id -> scalar or per-cell array for initial densities.
            initial_speeds: Optional mapping link id -> scalar or per-cell array for initial speeds.
            initial_origin_queues: Optional mapping origin id -> initial queue length.
            initial_onramp_queues: Optional mapping onramp id -> initial queue length.
            initial_offramp_queues: Optional mapping offramp id -> initial queue length.
            turning_rates: Mapping node id -> callable(time) -> turning-rate dict.
            destination_flow_bc: Mapping destination id -> callable(time) -> downstream flow.
            destination_density_bc: Mapping destination id -> callable(time) -> downstream density.

        Returns:
            Tuple containing:
            - link_flows_dict: mapping link id -> per-cell flows (np.ndarray)
            - link_densities_dict: mapping link id -> per-cell densities (np.ndarray)
            - link_speeds_dict: mapping link id -> per-cell speeds (np.ndarray)
            - origin_queues_dict: mapping origin id -> scalar queue
            - onramp_queues_dict: mapping onramp id -> scalar queue
            - offramp_queues_dict: mapping offramp id -> scalar queue
            - turning_rates_dict: mapping node id -> callable(time) -> dict[outgoing->rate]
            - destination_flow_bc_dict: mapping destination id -> callable(time) -> flow
            - destination_density_bc_dict: mapping destination id -> callable(time) -> density

        Raises:
            ValueError: If an offramp lacks a connected destination.
        """

        link_flows_dict: dict[str, NDArray[np.float64]] = {}
        link_densities_dict: dict[str, NDArray[np.float64]] = {}
        link_speeds_dict: dict[str, NDArray[np.float64]] = {}
        origin_queues_dict: dict[str, float] = {}
        onramp_queues_dict: dict[str, float] = {}
        offramp_queues_dict: dict[str, float] = {}
        turning_rates_dict: dict[str, Callable[[float], dict[str, float]]] = {}
        destination_flow_bc_dict: dict[str, Callable[[float], float]] = {}
        destination_density_bc_dict: dict[str, Callable[[float], float]] = {}

        for node in self.network.list_nodes():
            # split ratios should be defined for each node (add the ones that are missing for SISO nodes)
            if node.id not in turning_rates and len(node.outgoing) == 1:
                # capture the current outgoing link id in a default argument to avoid late-binding
                link_id = node.outgoing[0].id
                turning_rates_dict[node.id] = lambda _, link_id=link_id: {link_id: 1.0}
            else:
                turning_rates_dict[node.id] = turning_rates[node.id]

            # initialize incoming links (only onramps or origins - no motorway links)
            for link in node.incoming:
                if isinstance(link, Origin) or isinstance(link, Onramp):
                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for link {link.id} (type: {type(link)}) is empty."
                                )

                            elif init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for link {link.id} (type: {type(link)}) has incorrect length. Using first value for origin / onramp flow.",
                                    stacklevel=2,
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])

                            else:
                                link_flows_dict[link.id] = init_flow
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

                    if isinstance(link, Origin):
                        if (
                            initial_origin_queues is None
                            or link.id not in initial_origin_queues
                        ):
                            origin_queues_dict[link.id] = 0.0
                        else:
                            origin_queues_dict[link.id] = initial_origin_queues[link.id]

                    elif isinstance(link, Onramp):
                        if (
                            initial_onramp_queues is None
                            or link.id not in initial_onramp_queues
                        ):
                            onramp_queues_dict[link.id] = 0.0
                        else:
                            onramp_queues_dict[link.id] = initial_onramp_queues[link.id]

            # initialize outgoing links (mainline links, offramps, and destinations)
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    num_cells = len(link)

                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for motorway link {link.id} is empty."
                                )

                            elif init_flow.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial flow array for motorway link {link.id} has incorrect length. Using first value for all cells instead.",
                                    stacklevel=2,
                                )
                                link_flows_dict[link.id] = np.full(
                                    num_cells, init_flow[0]
                                )

                            else:
                                link_flows_dict[link.id] = init_flow
                        else:
                            link_flows_dict[link.id] = np.full(num_cells, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(num_cells)

                    if initial_densities is not None and link.id in initial_densities:
                        init_density = initial_densities[link.id]
                        if isinstance(init_density, np.ndarray):
                            if init_density.shape[0] == 0:
                                raise ValueError(
                                    f"Initial density array for motorway link {link.id} is empty."
                                )

                            if init_density.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial density array for motorway link {link.id} has incorrect length. Using first value for all cells instead.",
                                    stacklevel=2,
                                )
                                link_densities_dict[link.id] = np.full(
                                    num_cells, init_density[0]
                                )

                            else:
                                link_densities_dict[link.id] = init_density
                        else:
                            link_densities_dict[link.id] = np.full(
                                num_cells, init_density
                            )
                    else:
                        link_densities_dict[link.id] = np.zeros(num_cells)

                    if initial_speeds is not None and link.id in initial_speeds:
                        init_speed = initial_speeds[link.id]
                        if isinstance(init_speed, np.ndarray):
                            if init_speed.shape[0] == 0:
                                raise ValueError(
                                    f"Initial speed array for motorway link {link.id} is empty."
                                )

                            elif init_speed.shape[0] != num_cells:
                                warnings.warn(
                                    f"Initial speed array for motorway link {link.id} has incorrect length. Using first value for all cells instead.",
                                    stacklevel=2,
                                )
                                link_speeds_dict[link.id] = np.full(
                                    num_cells, init_speed[0]
                                )

                            else:
                                link_speeds_dict[link.id] = init_speed
                        else:
                            link_speeds_dict[link.id] = np.full(num_cells, init_speed)
                    else:
                        link_speeds_dict[link.id] = np.full(num_cells, link.vf)

                # for offramps initialize offramp flows and queues
                if isinstance(link, Offramp):
                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for offramp {link.id} is empty."
                                )

                            elif init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for offramp {link.id} has incorrect length. Using first value instead.",
                                    stacklevel=2,
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])

                            else:
                                link_flows_dict[link.id] = init_flow
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

                    if (
                        initial_offramp_queues is not None
                        and link.id in initial_offramp_queues
                    ):
                        offramp_queues_dict[link.id] = initial_offramp_queues[link.id]
                    else:
                        offramp_queues_dict[link.id] = 0.0

                elif isinstance(link, Destination):
                    # for destinations with missing boundary conditions, assign a constant zero function (downstream in free-flow)
                    if link.id not in destination_flow_bc:
                        raise ValueError(
                            f"Destination flow boundary condition function for destination {link.id} (connected to offramp {link.id}) not provided and cannot be inferred."
                        )
                    else:
                        destination_flow_bc_dict[link.id] = destination_flow_bc[link.id]

                    if link.id not in destination_density_bc:
                        warnings.warn(
                            f"Destination density boundary condition function for destination {link.id} (connected to offramp {link.id}) not provided. Assuming downstream free flow conditions (zero density).",
                            stacklevel=2,
                        )
                        # capture link.id to avoid late-binding (if lambda ever uses it)
                        destination_density_bc_dict[link.id] = (
                            lambda _, link_id=link.id: 0.0
                        )
                    else:
                        destination_density_bc_dict[link.id] = destination_density_bc[
                            link.id
                        ]

                    if initial_flows is not None and link.id in initial_flows:
                        init_flow = initial_flows[link.id]
                        if isinstance(init_flow, np.ndarray):
                            if init_flow.shape[0] == 0:
                                raise ValueError(
                                    f"Initial flow array for destination {link.id} is empty."
                                )

                            elif init_flow.shape[0] != 1:
                                warnings.warn(
                                    f"Initial flow array for destination {link.id} has incorrect length. Using first value instead.",
                                    stacklevel=2,
                                )
                                link_flows_dict[link.id] = np.full(1, init_flow[0])

                            else:
                                link_flows_dict[link.id] = init_flow
                        else:
                            link_flows_dict[link.id] = np.full(1, init_flow)
                    else:
                        link_flows_dict[link.id] = np.zeros(1)

        return (
            link_flows_dict,
            link_densities_dict,
            link_speeds_dict,
            origin_queues_dict,
            onramp_queues_dict,
            offramp_queues_dict,
            turning_rates_dict,
            destination_flow_bc_dict,
            destination_density_bc_dict,
        )

    # endregion

    # ! Core simulation execution
    # region
    def _run_simulation_loop(
        self,
        system: casadi.Function,
        duration: float,
        dt: float,
        x0: NDArray[np.float64],
        num_origins: int,
        num_splits: int,
        num_destinations: int,
        origin_queues_dict: dict[str, float],
        turning_rates_dict: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc_dict: dict[str, Callable[[float], float]],
        destination_density_bc_dict: dict[str, Callable[[float], float]],
        origin_demands: dict[str, Callable[[float], float]],
    ):
        """Execute the discrete-time simulation loop for the network.

        Advances the system state using the provided CasADi `system` update
        function for the requested duration and time-step. It evaluates the
        callable disturbance inputs (origin demands, turning rates and
        boundary conditions) at each time step, forms the disturbance vector,
        calls the model update, and stores state and disturbance histories.

        Args:
            system: CasADi function implementing the network state update.
            duration: Total simulation time (same units as `dt`).
            dt: Time-step for integration.
            x0: Initial packed state vector (NumPy array).
            num_origins: Number of origin queue entries in disturbance vector.
            num_splits: Number of turning-rate scalars per time step.
            num_destinations: Number of destination boundary condition entries.
            origin_queues_dict: Mapping origin id -> initial queue (used to order disturbances).
            turning_rates_dict: Mapping node id -> callable(time) -> turning-rate dict.
            destination_flow_bc_dict: Mapping destination id -> callable(time) -> flow.
            destination_density_bc_dict: Mapping destination id -> callable(time) -> density.
            origin_demands: Mapping origin id -> callable(time) -> demand.

        Returns:
            Tuple `(time_array, state_history, disturbance_history)` where
            - `time_array` is a 1-D NumPy array of time points,
            - `state_history` is a 2-D NumPy array of packed states over time,
            - `disturbance_history` is a 2-D NumPy array of packed disturbances over time.
        """

        time_array: NDArray[np.float64] = np.arange(
            0, duration + dt, dt, dtype=np.float64
        )

        # initialize variables for state, input and disturbance tracking
        state_history: NDArray[np.float64] = np.zeros(
            (len(x0) if isinstance(x0, np.ndarray) else x0.size1(), len(time_array)),
            dtype=np.float64,
        )
        state_history[:, 0] = x0
        disturbance_history: NDArray[np.float64] = np.zeros(
            (
                num_origins + num_splits + 2 * num_destinations,
                len(time_array) - 1,
            ),
            dtype=np.float64,
        )

        # run the simulation and store the results
        for t in range(len(time_array) - 1):
            time = time_array[t]

            # get the ids of all components that contribute to the disturbance vector
            origin_ids = origin_queues_dict.keys()

            # evaluate the demand functions, turning rates and boundary conditions at the current time
            origin_demand_dict = {
                origin_id: origin_demands[origin_id](time) for origin_id in origin_ids
            }
            turning_rate_dict = {
                node_id: turning_rates_dict[node_id](time)
                for node_id in turning_rates_dict.keys()
            }
            flow_boundary_condition_dict = {
                destination_id: destination_flow_bc_dict[destination_id](time)
                for destination_id in destination_flow_bc_dict.keys()
            }
            density_boundary_condition_dict = {
                destination_id: destination_density_bc_dict[destination_id](time)
                for destination_id in destination_density_bc_dict.keys()
            }

            # combine the values into the disturbance vector for the state update
            d = self.network.network_dict_to_disturbance_vec(
                origin_demand_dict=origin_demand_dict,
                turning_rate_dict=turning_rate_dict,
                flow_boundary_condition_dict=flow_boundary_condition_dict,
                density_boundary_condition_dict=density_boundary_condition_dict,
            )

            # obtain the model parameters in vector form for the model update
            if self.model_params is not None and isinstance(self.model, METANET):
                params = self.model.model_params_to_vec(
                    network=self.network, model_params=self.model_params
                )
            elif self.model_params is None and isinstance(self.model, METANET):
                raise ValueError(
                    "METANET model requires model_params to be provided for simulation."
                )
            elif self.model_params is not None and isinstance(self.model, CTM):
                raise ValueError(
                    "CTM model does not support model_params to be provided for simulation."
                )
            else:
                params = np.array(
                    [], dtype=np.float64
                )  # empty array for models that don't require parameters

            # perform the state update
            x_next = system(params, state_history[:, t], d)

            # store the updated state and disturbance
            state_history[:, t + 1] = np.array(x_next).flatten()
            disturbance_history[:, t] = d

        return time_array, state_history, disturbance_history

    # endregion

    # ! Main simulation runner
    # region
    def run(
        self,
        duration: float,
        dt: float,
        origin_demands: dict[str, Callable[[float], float]],
        turning_rates: dict[str, Callable[[float], dict[str, float]]],
        destination_flow_bc: dict[str, Callable[[float], float]],
        destination_density_bc: dict[str, Callable[[float], float]],
        initial_flows: dict[str, NDArray[np.float64] | float] | None = None,
        initial_densities: dict[str, NDArray[np.float64] | float] | None = None,
        initial_speeds: dict[str, NDArray[np.float64] | float] | None = None,
        initial_origin_queues: dict[str, float] | None = None,
        initial_onramp_queues: dict[str, float] | None = None,
        initial_offramp_queues: dict[str, float] | None = None,
        preferred_cell_size: float = 0.5,
        plot_results: bool = True,
        show_plots: bool = False,
        results_dir: str | None = None,
    ):
        """Run the network simulation over a time horizon.

        Executes a forward simulation using the traffic model specified during
        Simulation instantiation and the per-component callable inputs for demands,
        turning rates and boundary conditions. The routine will discretize motorway
        links, initialize state and disturbance vectors, perform time-stepping to
        update the state, and optionally plot and save results.

        Args:
            duration: Total simulation time (same units as demand functions, e.g. hours).
            dt: Simulation time step (same units as `duration`).
            origin_demands: Mapping origin id -> callable(time) -> demand (veh/h).
            turning_rates: Mapping node id -> callable(time) -> dict[outgoing_link_id -> split rate].
            destination_flow_bc: Mapping destination id -> callable(time) -> downstream flow (veh/h/lane).
            destination_density_bc: Mapping destination id -> callable(time) -> downstream density (veh/km/lane).
            initial_flows: Optional mapping link id -> scalar or per-cell array for initial flows (default: zeros).
            initial_densities: Optional mapping link id -> scalar or per-cell array for initial densities (default: zeros).
            initial_speeds: Optional mapping link id -> scalar or per-cell array for initial speeds (default: free-flow speed).
            initial_origin_queues: Optional mapping origin id -> initial queue length (veh).
            initial_onramp_queues: Optional mapping onramp id -> initial queue length (veh).
            initial_offramp_queues: Optional mapping offramp id -> initial queue length (veh).
            preferred_cell_size: Preferred link segmentation size (km) used when partitioning motorway links.
            plot_results: If True, generate plots and save results to `results_dir`.
            show_plots: If True, display plots interactively.
            results_dir: Directory for saving results; if None a timestamped folder under `results/` is used when `plot_results` is True.

        Returns:
            tuple: `(time_array, state_history, disturbance_history)` where
                - `time_array` is a 1-D NumPy array of time points,
                - `state_history` is a 2-D NumPy array of packed states over time (state_size x timesteps),
                - `disturbance_history` is a 2-D NumPy array of packed disturbances over time (disturbance_size x timesteps-1).

        Raises:
            ValueError: If required inputs are missing or inconsistent with the network topology.
        """
        # ! 1 - validate all inputs as required
        self._validate_initial_conditions_numerical(
            origin_demands=origin_demands,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
            initial_flows=initial_flows,
            initial_densities=initial_densities,
            initial_speeds=initial_speeds,
        )

        # ! 2 - discretize mainline motorway links according to preferred cell size and CFL condition
        for node in self.network.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    upcoming_lane_drop = self.network._compute_upcoming_lane_drop(link)
                    link.partition_link(
                        preferred_cell_size=preferred_cell_size,
                        dt=dt,
                        upcoming_lane_drop=upcoming_lane_drop,
                    )

        # ! 3 - augment the node and link states
        (
            link_flows_dict,
            link_densities_dict,
            link_speeds_dict,
            origin_queues_dict,
            onramp_queues_dict,
            offramp_queues_dict,
            turning_rates_dict,
            destination_flow_bc_dict,
            destination_density_bc_dict,
        ) = self._augment_network_initialization(
            initial_flows=initial_flows,
            initial_densities=initial_densities,
            initial_speeds=initial_speeds,
            initial_origin_queues=initial_origin_queues,
            initial_onramp_queues=initial_onramp_queues,
            initial_offramp_queues=initial_offramp_queues,
            turning_rates=turning_rates,
            destination_flow_bc=destination_flow_bc,
            destination_density_bc=destination_density_bc,
        )

        # combine state from separate arrays into single state vector
        (
            x0,
            num_flows,
            num_densities,
            num_speeds,
            num_origins,
            num_onramps,
            num_offramps,
            num_splits,
            num_destinations,
        ) = self.network.network_dict_to_state_vec(
            flow_dict=link_flows_dict,
            density_dict=link_densities_dict,
            speed_dict=link_speeds_dict,
            origin_queue_dict=origin_queues_dict,
            onramp_queue_dict=onramp_queues_dict,
            offramp_queue_dict=offramp_queues_dict,
        )

        # ! 4 - generate the model update equations
        system: casadi.Function = self.model.network_update_function(
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

        # ! 5 - run the simulation loop
        time_array, state_history, disturbance_history = self._run_simulation_loop(
            system=system,
            duration=duration,
            dt=dt,
            x0=cast(NDArray[np.float64], x0),
            num_origins=num_origins,
            num_splits=num_splits,
            num_destinations=num_destinations,
            origin_queues_dict=origin_queues_dict,
            turning_rates_dict=turning_rates_dict,
            destination_flow_bc_dict=destination_flow_bc_dict,
            destination_density_bc_dict=destination_density_bc_dict,
            origin_demands=origin_demands,
        )

        # ! 6 - plotting of simulation results and saving to results directory
        if plot_results:
            if results_dir is None:
                timestamp = datetime.now().strftime(
                    "simulation_results_%Y-%m-%d_%H%M%S"
                )
                results_dir = f"results/{timestamp}"

            os.makedirs(results_dir, exist_ok=True)
            print(f"Saving simulation results to {results_dir}")

            # save network topology plot
            topology_path = os.path.join(results_dir, "network_topology.png")
            self.network.plot(show=show_plots, save_path=topology_path)
            print(f"  Network topology saved to {topology_path}")

            # save simulation results as JSON file
            results_path = os.path.join(results_dir, "simulation_results.json")
            self.save_results(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=results_path,
                dt=dt,
                duration=duration,
                preferred_cell_size=preferred_cell_size,
                model_params=self.model_params,
            )
            print(f"  Simulation results saved to {results_path}")

            # save network structure as text file
            structure_path = os.path.join(results_dir, "network_structure.txt")
            self.network.save_to_txt(structure_path)
            structure_path_json = os.path.join(results_dir, "network_structure.json")
            self.network.save_to_json(structure_path_json)
            print(f"  Network structure saved to {structure_path}")

            # plot simulation results
            self.plot_results(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                save_dir=results_dir,
            )

        # store results on the instance for later use (e.g., save_results without args)
        self._time_array = time_array
        self._state_history = state_history
        self._disturbance_history = disturbance_history
        self._last_dt = dt
        self._last_duration = duration
        self._last_preferred_cell_size = preferred_cell_size

        return time_array, state_history, disturbance_history

    # endregion

    # ! Result persistence and metrics
    # region
    def save_results(
        self,
        filepath: str,
        time_array: NDArray[np.float64] | None = None,
        state_history: NDArray[np.float64] | None = None,
        disturbance_history: NDArray[np.float64] | None = None,
        dt: float | None = None,
        duration: float | None = None,
        preferred_cell_size: float | None = None,
        model_params: Union["METANETParams", None] = None,
    ) -> None:
        """Save simulation results to a JSON file with comprehensive metadata.

        Writes the time array, state history and disturbance history to a
        JSON file for later reference. State and disturbance histories are
        split into link/node specific time series using the network unpacking
        helpers. Additionally saves simulation metadata including model type,
        simulation parameters, critical densities for each link, and model
        parameters (if applicable) for full reproducibility.

        When called after ``run()``, all array arguments can be omitted and
        the results stored on the instance are used automatically.

        Args:
            filepath: Path where the JSON file should be saved.
            time_array: 1-D array of time points (uses stored results when None).
            state_history: 2-D array of state vectors over time (uses stored results when None).
            disturbance_history: 2-D array of disturbances over time (uses stored results when None).
            dt: Simulation time step (uses value from last run() when None).
            duration: Total simulation duration (uses value from last run() when None).
            preferred_cell_size: Preferred cell size (uses value from last run() when None).
            model_params: Model parameters for METANET (None for CTM).
        """
        # Fall back to stored instance results when arguments are not supplied
        if time_array is None:
            if self._time_array is None:
                raise ValueError(
                    "No time_array provided and no results stored. Call run() first."
                )
            time_array = self._time_array
        if state_history is None:
            if self._state_history is None:
                raise ValueError(
                    "No state_history provided and no results stored. Call run() first."
                )
            state_history = self._state_history
        if disturbance_history is None:
            if self._disturbance_history is None:
                raise ValueError(
                    "No disturbance_history provided and no results stored. Call run() first."
                )
            disturbance_history = self._disturbance_history
        if dt is None:
            if self._last_dt is None:
                raise ValueError(
                    "No dt provided and no results stored. Call run() first."
                )
            dt = self._last_dt
        if duration is None:
            if self._last_duration is None:
                raise ValueError(
                    "No duration provided and no results stored. Call run() first."
                )
            duration = self._last_duration
        if preferred_cell_size is None:
            if self._last_preferred_cell_size is None:
                raise ValueError(
                    "No preferred_cell_size provided and no results stored. Call run() first."
                )
            preferred_cell_size = self._last_preferred_cell_size

        # prepare containers for per-link / per-node time series
        flows_time: dict[str, list] = {}
        densities_time: dict[str, list] = {}
        speeds_time: dict[str, list] = {}
        origin_queues_time: dict[str, list] = {}
        onramp_queues_time: dict[str, list] = {}
        offramp_queues_time: dict[str, list] = {}

        origin_demands_time: dict[str, list] = {}
        turning_rates_time: dict[str, dict[str, list]] = {}
        flow_boundary_conditions_time: dict[str, list] = {}
        density_boundary_conditions_time: dict[str, list] = {}

        num_timesteps = state_history.shape[1]

        # split state history into per-link/node time series
        for t in range(num_timesteps):
            flows_t, densities_t, speeds_t, origin_q_t, onramp_q_t, offramp_q_t = (
                self.network.state_vec_to_network_dict(state_history[:, t])
            )

            # initialize containers on first timestep
            if t == 0:
                for k in flows_t.keys():
                    flows_time[k] = []
                for k in densities_t.keys():
                    densities_time[k] = []
                for k in speeds_t.keys():
                    speeds_time[k] = []
                for k in origin_q_t.keys():
                    origin_queues_time[k] = []
                for k in onramp_q_t.keys():
                    onramp_queues_time[k] = []
                for k in offramp_q_t.keys():
                    offramp_queues_time[k] = []

            # append values (convert numpy -> native Python types)
            for k, v in flows_t.items():
                flows_time[k].append(np.asarray(v).tolist())
            for k, v in densities_t.items():
                densities_time[k].append(np.asarray(v).tolist())
            for k, v in speeds_t.items():
                speeds_time[k].append(np.asarray(v).tolist())

            for k, v in origin_q_t.items():
                origin_queues_time[k].append(float(np.asarray(v).tolist()))
            for k, v in onramp_q_t.items():
                onramp_queues_time[k].append(float(np.asarray(v).tolist()))
            for k, v in offramp_q_t.items():
                offramp_queues_time[k].append(float(np.asarray(v).tolist()))

        # split disturbance history into per-component time series
        if disturbance_history.size > 0:
            num_dist_timesteps = disturbance_history.shape[1]
            for t in range(num_dist_timesteps):
                (
                    origin_d_t,
                    turning_t,
                    boundary_flow_t,
                    boundary_density_t,
                ) = self.network.disturbance_vec_to_network_dict(
                    disturbance_history[:, t]
                )

                if t == 0:
                    for k in origin_d_t.keys():
                        origin_demands_time[k] = []
                    for node_id, inner in turning_t.items():
                        turning_rates_time[node_id] = {lk: [] for lk in inner.keys()}
                    for k in boundary_flow_t.keys():
                        flow_boundary_conditions_time[k] = []
                    for k in boundary_density_t.keys():
                        density_boundary_conditions_time[k] = []

                for k, v in origin_d_t.items():
                    origin_demands_time[k].append(float(np.asarray(v).tolist()))

                for node_id, inner in turning_t.items():
                    for lk, rate in inner.items():
                        turning_rates_time.setdefault(node_id, {}).setdefault(
                            lk, []
                        ).append(float(np.asarray(rate).tolist()))

                for k, v in boundary_flow_t.items():
                    flow_boundary_conditions_time[k].append(
                        float(np.asarray(v).tolist())
                    )
                for k, v in boundary_density_t.items():
                    density_boundary_conditions_time[k].append(
                        float(np.asarray(v).tolist())
                    )

        critical_densities: dict[str, float] = {}
        link_properties: dict[str, dict] = {}

        for node in self.network.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if isinstance(self.model, CTM):
                        rho_crit = self.model.critical_density(
                            lane_capacity=link.Qc_lane,
                            free_flow_speed=link.vf,
                        )
                    elif isinstance(self.model, METANET):
                        if model_params is None:
                            raise ValueError(
                                "model_params required for METANET critical density calculation"
                            )
                        rho_crit = self.model.critical_density(
                            params=model_params,
                            link_id=link.id,
                            lane_capacity=link.Qc_lane,
                            free_flow_speed=link.vf,
                        )
                    else:
                        raise ValueError(
                            f"Unknown model type: {type(self.model).__name__}"
                        )

                    critical_densities[link.id] = rho_crit

                    # extract cell lengths array (cells may have different lengths)
                    cell_lengths = [float(cell.length) for cell in link]

                    link_properties[link.id] = {
                        "length": link.length,
                        "lanes": link.lanes,
                        "lane_capacity": link.Qc_lane,
                        "free_flow_speed": link.vf,
                        "jam_density": link.rho_jam,
                        "num_cells": len(link),
                        "cell_lengths": cell_lengths,
                    }

        # build metadata section
        metadata = {
            "model_type": type(self.model).__name__,
            "simulation_parameters": {
                "dt": dt,
                "duration": duration,
                "preferred_cell_size": preferred_cell_size,
            },
            "link_properties": link_properties,
            "critical_densities": critical_densities,
        }

        # add model parameters if METANET
        if isinstance(self.model, METANET) and model_params is not None:
            # convert model_params to serializable format
            # handle alpha separately as it can be float or dict[str, float]
            metadata["model_parameters"] = {
                "tau": model_params["tau"],
                "nu": model_params["nu"],
                "kappa": model_params["kappa"],
                "delta": model_params["delta"],
                "phi": model_params["phi"],
                "alpha": (
                    {
                        link_id: alpha_val
                        for link_id, alpha_val in model_params["alpha"].items()
                    }
                    if isinstance(model_params["alpha"], dict)
                    else float(model_params["alpha"])
                ),
            }

        # assemble output structure
        out = {
            "metadata": metadata,
            "time_array": np.asarray(time_array).tolist(),
            "state_time_series": {
                "flows": flows_time,
                "densities": densities_time,
                "speeds": speeds_time,
                "origin_queues": origin_queues_time,
                "onramp_queues": onramp_queues_time,
                "offramp_queues": offramp_queues_time,
            },
            "disturbance_time_series": {
                "origin_demands": origin_demands_time,
                "turning_rates": turning_rates_time,
                "flow_boundary_conditions": flow_boundary_conditions_time,
                "density_boundary_conditions": density_boundary_conditions_time,
            },
        }

        # write JSON to file
        with open(filepath, "w") as f:
            json.dump(out, f, indent=2)

    @classmethod
    def load_results(
        cls,
        filepath: str,
        network: "Network",
        load_mainline_only: bool = False,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        dict | None,
    ]:
        """Load simulation results from a JSON file with validation.

        Reads a JSON file created by `save_results` and validates that all required
        fields are present and contain valid numerical data. When
        ``load_mainline_only`` is True, the function expects the file to contain
        only mainline `flows`, `densities`, and `speeds` together with boundary
        conditions in `disturbance_time_series` (flow/density). In that mode
        missing onramp/offramp/origin flows and queue values as well as missing
        disturbance entries (origin demands / turning rates) are filled with
        sensible defaults so the returned packed vectors remain consistent with
        the provided `network`.

        Args:
            filepath: Path to the JSON file containing simulation results.
            network: Network instance to use for validating structure against saved data.
            load_mainline_only: If True, only require and load mainline quantities
                (flows/densities/speeds and boundary conditions). Other
                quantities (queues, demands, turning rates) will be filled with
                default values when absent.

        Returns:
            Tuple of (time_array, state_history, disturbance_history, metadata).

        Raises:
            ValueError: If required fields are missing or data validation fails.
            FileNotFoundError: If the specified file does not exist.
        """
        # load JSON data from file
        with open(filepath, "r") as f:
            data = json.load(f)

        # extract metadata if present (optional for backward compatibility)
        metadata = data.get("metadata", None)

        # validate top-level structure
        required_fields = ["time_array", "state_time_series", "disturbance_time_series"]
        for field in required_fields:
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}' in simulation results file."
                )

        # depending on the load mode, only a subset of state/disturbance fields
        # are required. When mainline-only mode is selected we only require the
        # mainline fields and the boundary-condition components.
        if load_mainline_only:
            state_required = ["flows", "densities", "speeds"]
            disturbance_required = [
                "flow_boundary_conditions",
                "density_boundary_conditions",
            ]
        else:
            state_required = [
                "flows",
                "densities",
                "speeds",
                "origin_queues",
                "onramp_queues",
                "offramp_queues",
            ]
            disturbance_required = [
                "origin_demands",
                "turning_rates",
                "flow_boundary_conditions",
                "density_boundary_conditions",
            ]

        for field in state_required:
            if field not in data["state_time_series"]:
                raise ValueError(
                    f"Missing required field '{field}' in state_time_series."
                )

        for field in disturbance_required:
            if field not in data["disturbance_time_series"]:
                raise ValueError(
                    f"Missing required field '{field}' in disturbance_time_series."
                )

        # convert time_array to NumPy array
        time_array = np.array(data["time_array"], dtype=np.float64)
        if time_array.ndim != 1:
            raise ValueError("time_array must be a 1-D array.")

        num_timesteps = len(time_array)

        # reconstruct state dictionaries for validation and conversion
        state_series = data["state_time_series"]

        # validate that required per-link entries exist for the given network
        for node in network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Origin):
                    if not load_mainline_only and link.id not in state_series.get(
                        "origin_queues", {}
                    ):
                        raise ValueError(
                            f"Origin queue data for '{link.id}' not found in saved results."
                        )
                elif isinstance(link, Onramp):
                    if not load_mainline_only and link.id not in state_series.get(
                        "flows", {}
                    ):
                        raise ValueError(
                            f"Flow data for onramp '{link.id}' not found in saved results."
                        )
                    if not load_mainline_only and link.id not in state_series.get(
                        "onramp_queues", {}
                    ):
                        raise ValueError(
                            f"Onramp queue data for '{link.id}' not found in saved results."
                        )

            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if link.id not in state_series.get("flows", {}):
                        raise ValueError(
                            f"Flow data for motorway link '{link.id}' not found in saved results."
                        )
                    if link.id not in state_series.get("densities", {}):
                        raise ValueError(
                            f"Density data for motorway link '{link.id}' not found in saved results."
                        )
                    if link.id not in state_series.get("speeds", {}):
                        raise ValueError(
                            f"Speed data for motorway link '{link.id}' not found in saved results."
                        )
                elif isinstance(link, Offramp):
                    if not load_mainline_only and link.id not in state_series.get(
                        "flows", {}
                    ):
                        raise ValueError(
                            f"Flow data for offramp '{link.id}' not found in saved results."
                        )
                    if not load_mainline_only and link.id not in state_series.get(
                        "offramp_queues", {}
                    ):
                        raise ValueError(
                            f"Offramp queue data for '{link.id}' not found in saved results."
                        )
                elif isinstance(link, Destination):
                    if not load_mainline_only and link.id not in state_series.get(
                        "flows", {}
                    ):
                        raise ValueError(
                            f"Flow data for destination '{link.id}' not found in saved results."
                        )

        # Reconstruct state history timestep-by-timestep. We iterate over the
        # network topology so missing non-mainline fields can be populated with
        # defaults when ``load_mainline_only`` is True.
        state_dicts = []
        for t in range(num_timesteps):
            flows_t: dict[str, NDArray[np.float64]] = {}
            densities_t: dict[str, NDArray[np.float64]] = {}
            speeds_t: dict[str, NDArray[np.float64]] = {}
            origin_queues_t: dict[str, float] = {}
            onramp_queues_t: dict[str, float] = {}
            offramp_queues_t: dict[str, float] = {}

            for node in network.list_nodes():
                # incoming links (origins / onramps)
                for link in node.incoming:
                    # flow for incoming links: present in file for full mode,
                    # absent for mainline-only files -> default to [0.0]
                    if link.id in state_series.get("flows", {}):
                        flows_t[str(link.id)] = np.array(
                            state_series["flows"][link.id][t], dtype=np.float64
                        )
                    else:
                        flows_t[str(link.id)] = np.array([0.0], dtype=np.float64)

                    # queues: origins and onramps
                    if isinstance(link, Origin):
                        if link.id in state_series.get("origin_queues", {}):
                            origin_queues_t[str(link.id)] = float(
                                state_series["origin_queues"][link.id][t]
                            )
                        else:
                            if load_mainline_only:
                                origin_queues_t[str(link.id)] = 0.0
                            else:
                                raise ValueError(
                                    f"Origin queue data for '{link.id}' not found in saved results."
                                )

                    if isinstance(link, Onramp):
                        if link.id in state_series.get("onramp_queues", {}):
                            onramp_queues_t[str(link.id)] = float(
                                state_series["onramp_queues"][link.id][t]
                            )
                        else:
                            if load_mainline_only:
                                onramp_queues_t[str(link.id)] = 0.0
                            else:
                                raise ValueError(
                                    f"Onramp queue data for '{link.id}' not found in saved results."
                                )

                # outgoing links
                for link in node.outgoing:
                    if isinstance(link, MotorwayLink):
                        if link.id in state_series.get("flows", {}):
                            flows_t[str(link.id)] = np.array(
                                state_series["flows"][link.id][t], dtype=np.float64
                            )
                        else:
                            raise ValueError(
                                f"Flow data for motorway link '{link.id}' not found in saved results."
                            )

                        if link.id in state_series.get("densities", {}):
                            densities_t[str(link.id)] = np.array(
                                state_series["densities"][link.id][t], dtype=np.float64
                            )
                        else:
                            raise ValueError(
                                f"Density data for motorway link '{link.id}' not found in saved results."
                            )

                        if link.id in state_series.get("speeds", {}):
                            speeds_t[str(link.id)] = np.array(
                                state_series["speeds"][link.id][t], dtype=np.float64
                            )
                        else:
                            raise ValueError(
                                f"Speed data for motorway link '{link.id}' not found in saved results."
                            )

                    elif isinstance(link, Offramp):
                        if link.id in state_series.get("flows", {}):
                            flows_t[str(link.id)] = np.array(
                                state_series["flows"][link.id][t], dtype=np.float64
                            )
                        else:
                            # absent in mainline-only files -> default to zero
                            if load_mainline_only:
                                flows_t[str(link.id)] = np.array(
                                    [0.0], dtype=np.float64
                                )
                            else:
                                raise ValueError(
                                    f"Flow data for offramp '{link.id}' not found in saved results."
                                )

                        if link.id in state_series.get("offramp_queues", {}):
                            offramp_queues_t[str(link.id)] = float(
                                state_series["offramp_queues"][link.id][t]
                            )
                        else:
                            if load_mainline_only:
                                offramp_queues_t[str(link.id)] = 0.0
                            else:
                                raise ValueError(
                                    f"Offramp queue data for '{link.id}' not found in saved results."
                                )

                    elif isinstance(link, Destination):
                        if link.id in state_series.get("flows", {}):
                            flows_t[str(link.id)] = np.array(
                                state_series["flows"][link.id][t], dtype=np.float64
                            )
                        else:
                            if load_mainline_only:
                                flows_t[str(link.id)] = np.array(
                                    [0.0], dtype=np.float64
                                )
                            else:
                                raise ValueError(
                                    f"Flow data for destination '{link.id}' not found in saved results."
                                )

            # validate numerical data using existing validation (for first timestep)
            if t == 0:
                network._validate_state_history_numerical(
                    flows=flows_t,
                    densities=densities_t,
                    speeds=speeds_t,
                    origin_queues=origin_queues_t,
                    onramp_queues=onramp_queues_t,
                    offramp_queues=offramp_queues_t,
                )

            state_dicts.append(
                (
                    flows_t,
                    densities_t,
                    speeds_t,
                    origin_queues_t,
                    onramp_queues_t,
                    offramp_queues_t,
                )
            )

        # pack into state vectors
        state_vecs = []
        for (
            flows_t,
            densities_t,
            speeds_t,
            origin_q_t,
            onramp_q_t,
            offramp_q_t,
        ) in state_dicts:
            x_t, *_ = network.network_dict_to_state_vec(
                flow_dict=flows_t,
                density_dict=densities_t,
                speed_dict=speeds_t,
                origin_queue_dict=origin_q_t,
                onramp_queue_dict=onramp_q_t,
                offramp_queue_dict=offramp_q_t,
            )
            state_vecs.append(x_t)

        state_history = np.column_stack(state_vecs)

        # Reconstruct disturbance history
        # disturbance series timeline should align with state timeline:
        # disturbances represent inputs between state timesteps, therefore
        # their length is expected to be `len(time_array) - 1`.
        disturbance_series = data["disturbance_time_series"]
        num_dist_timesteps = max(0, num_timesteps - 1)

        disturbance_vecs = []
        for t in range(num_dist_timesteps):
            # origin demands
            origin_demands_t: dict[str, float] = {}
            if disturbance_series.get("origin_demands"):
                for k, v in disturbance_series["origin_demands"].items():
                    origin_demands_t[k] = float(v[t])
            else:
                # default zero demands when not present (mainline-only files)
                for node in network.list_nodes():
                    for link in node.incoming:
                        if isinstance(link, Origin):
                            origin_demands_t[link.id] = 0.0

            # turning rates
            turning_rates_t: dict[str, dict[str, float]] = {}
            if disturbance_series.get("turning_rates"):
                for node_id, inner in disturbance_series["turning_rates"].items():
                    turning_rates_t[node_id] = {
                        lk: float(vals[t]) for lk, vals in inner.items()
                    }
            else:
                # default: equal split across outgoing links for each node
                for node in network.list_nodes():
                    outgoing = list(node.outgoing)
                    if len(outgoing) == 0:
                        turning_rates_t[node.id] = {}
                    else:
                        share = 1.0 / len(outgoing)
                        turning_rates_t[node.id] = {
                            lk.id: float(share) for lk in outgoing
                        }

            # boundary conditions (flow / density)
            flow_bc_t: dict[str, float] = {}
            if disturbance_series.get("flow_boundary_conditions"):
                for k, v in disturbance_series["flow_boundary_conditions"].items():
                    flow_bc_t[k] = float(v[t])
            else:
                # fallback zeros for missing boundary conditions
                for node in network.list_nodes():
                    for link in node.outgoing:
                        if isinstance(link, Destination):
                            flow_bc_t[link.id] = 0.0

            density_bc_t: dict[str, float] = {}
            if disturbance_series.get("density_boundary_conditions"):
                for k, v in disturbance_series["density_boundary_conditions"].items():
                    density_bc_t[k] = float(v[t])
            else:
                for node in network.list_nodes():
                    for link in node.outgoing:
                        if isinstance(link, Destination):
                            density_bc_t[link.id] = 0.0

            # validate numerical data using existing validation (for first timestep)
            if t == 0:
                network._validate_disturbance_history_numerical(
                    origin_demands=cast(dict[str, float], origin_demands_t),
                    turning_rates=cast(dict[str, dict[str, float]], turning_rates_t),
                    flow_boundary_conditions=cast(dict[str, float], flow_bc_t),
                    density_boundary_conditions=cast(dict[str, float], density_bc_t),
                )

            d_t = network.network_dict_to_disturbance_vec(
                origin_demand_dict=origin_demands_t,
                turning_rate_dict=turning_rates_t,
                flow_boundary_condition_dict=flow_bc_t,
                density_boundary_condition_dict=density_bc_t,
            )
            disturbance_vecs.append(d_t)

        disturbance_history = (
            np.column_stack(disturbance_vecs) if disturbance_vecs else np.array([])
        )

        return time_array, state_history, disturbance_history, metadata

    def compute_metrics(
        self,
        states: NDArray[np.float64],
        dt: float,
        timesteps: int,
    ) -> Tuple[float, float, float]:
        """Compute a set of performance metrics based on the provided simulation results

        Args:
            states: 2-D array shape (state_size, timesteps) containing state vectors over time.
            dt: Time step used in the simulation (hours).
            timesteps: Number of timesteps in the simulation.

        Returns:
            (VKT, VHT, overall_avg_speed) floats: vehicle-kilometres travelled,
                vehicle-hours travelled, and overall average speed.
        """

        # ! Part 1: Calculate VKT and VHT
        VKT: float = 0.0
        VHT: float = 0.0

        for t in range(timesteps - 1):
            # extract the different states at time t from the state vector
            flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
                self.network.state_vec_to_network_dict(states[:, t])
            )

            # verify that the extracted values are numerical arrays
            self._validate_state_history_numerical(
                flows=cast(dict[str, NDArray[np.float64]], flows),
                densities=cast(dict[str, NDArray[np.float64]], densities),
                speeds=cast(dict[str, NDArray[np.float64]], speeds),
                origin_queues=cast(dict[str, float], origin_queues),
                onramp_queues=cast(dict[str, float], onramp_queues),
                offramp_queues=cast(dict[str, float], offramp_queues),
            )

            # typecast to np.ndarray to ensure type safety
            flows = {k: np.asarray(v) for k, v in flows.items()}
            densities = {k: np.asarray(v) for k, v in densities.items()}
            speeds = {k: np.asarray(v) for k, v in speeds.items()}
            origin_queues = {k: float(v) for k, v in origin_queues.items()}
            onramp_queues = {k: float(v) for k, v in onramp_queues.items()}
            offramp_queues = {k: float(v) for k, v in offramp_queues.items()}

            # add the time vehicles spent in the origin, onramp, and offramp queues (veh * hours)
            VHT += dt * sum(origin_queues.values())
            VHT += dt * sum(onramp_queues.values())
            VHT += dt * sum(offramp_queues.values())

            # iterate over all outgoing motorway links and accumulate VKT and VHT
            for node in self.network.list_nodes():
                for link in node.outgoing:
                    if isinstance(link, MotorwayLink):
                        for idx, cell in link.enumerate_cells():
                            if cell is None:
                                raise ValueError(
                                    f"Cell {idx} not found in motorway link {link.id}."
                                )

                            # add VKT: distance * vehicles that passed (flow is veh/h)
                            VKT += cell.length * dt * flows[link.id][idx]

                            # add VHT for vehicles on the mainline segment (density is veh/km/lane)
                            VHT += (
                                cell.length * dt * densities[link.id][idx] * link.lanes
                            )

        # ! Part 2: Calculate vehicle-weighted average speed
        overall_avg_speed: float = VKT / VHT if VHT > 0 else 0.0

        return VKT, VHT, float(overall_avg_speed)

    # endregion

    # ! Plotting and visualization
    # region
    def plot_results(
        self,
        time_array: NDArray[np.float64],
        state_history: NDArray[np.float64],
        disturbance_history: NDArray[np.float64],
        save_dir: str = "results",
    ) -> None:
        """Plot comprehensive simulation results for the network.

        Creates multiple figures showing density, flow, speed for all mainline
        links, demand/flow/queue plots for origins and onramps, flow plots for
        offramps and destinations, 3D surface plots for each motorway link, and
        summary plots per node showing all inflows and outflows.

        Args:
            time_array: 1-D array of time points (hours).
            state_history: 2-D array of state vectors over time, shape (state_size, timesteps).
            disturbance_history: 2-D array of disturbances over time, shape (disturbance_size, timesteps-1).
            save_dir: Directory where plots should be saved (default: "results").
        """
        # create results directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # convert time to seconds for plotting
        time_seconds = time_array * 3600.0
        num_timesteps = len(time_array)
        print(f"Generating simulation result plots in {save_dir}...")

        # build dictionaries mapping link_id -> array of values over time
        flows_over_time: dict[str, np.ndarray] = {}
        densities_over_time: dict[str, np.ndarray] = {}
        speeds_over_time: dict[str, np.ndarray] = {}
        origin_queues_over_time: dict[str, np.ndarray] = {}
        onramp_queues_over_time: dict[str, np.ndarray] = {}
        offramp_queues_over_time: dict[str, np.ndarray] = {}

        for t in range(num_timesteps):
            (
                flows_t,
                densities_t,
                speeds_t,
                origin_queues_t,
                onramp_queues_t,
                offramp_queues_t,
            ) = self.network.state_vec_to_network_dict(state_history[:, t])

            # make sure that the history values are numerical
            self._validate_state_history_numerical(
                flows=cast(dict[str, NDArray[np.float64]], flows_t),
                densities=cast(dict[str, NDArray[np.float64]], densities_t),
                speeds=cast(dict[str, NDArray[np.float64]], speeds_t),
                origin_queues=cast(dict[str, float], origin_queues_t),
                onramp_queues=cast(dict[str, float], onramp_queues_t),
                offramp_queues=cast(dict[str, float], offramp_queues_t),
            )

            # typecast to np.ndarray to ensure type safety
            flows_t = {k: np.asarray(v) for k, v in flows_t.items()}
            densities_t = {k: np.asarray(v) for k, v in densities_t.items()}
            speeds_t = {k: np.asarray(v) for k, v in speeds_t.items()}
            origin_queues_t = {k: np.asarray(v) for k, v in origin_queues_t.items()}
            onramp_queues_t = {k: np.asarray(v) for k, v in onramp_queues_t.items()}
            offramp_queues_t = {k: np.asarray(v) for k, v in offramp_queues_t.items()}

            # initialize dictionaries on first iteration
            if t == 0:
                for link_id in flows_t.keys():
                    flows_over_time[link_id] = np.zeros(
                        (len(flows_t[link_id]), num_timesteps)
                    )

                for link_id in densities_t.keys():
                    densities_over_time[link_id] = np.zeros(
                        (len(densities_t[link_id]), num_timesteps)
                    )

                for link_id in speeds_t.keys():
                    speeds_over_time[link_id] = np.zeros(
                        (len(speeds_t[link_id]), num_timesteps)
                    )

                for origin_id in origin_queues_t.keys():
                    origin_queues_over_time[origin_id] = np.zeros(num_timesteps)

                for onramp_id in onramp_queues_t.keys():
                    onramp_queues_over_time[onramp_id] = np.zeros(num_timesteps)

                for offramp_id in offramp_queues_t.keys():
                    offramp_queues_over_time[offramp_id] = np.zeros(num_timesteps)

            # store values for the current timestep
            for link_id, val in flows_t.items():
                flows_over_time[link_id][:, t] = val

            for link_id, val in densities_t.items():
                densities_over_time[link_id][:, t] = val

            for link_id, val in speeds_t.items():
                speeds_over_time[link_id][:, t] = val

            for origin_id, val in origin_queues_t.items():
                origin_queues_over_time[origin_id][t] = float(val)

            for onramp_id, val in onramp_queues_t.items():
                onramp_queues_over_time[onramp_id][t] = float(val)

            for offramp_id, val in offramp_queues_t.items():
                offramp_queues_over_time[offramp_id][t] = float(val)

        # extract relevant disturbance time series (turning rates)
        origin_demands_over_time: dict[str, np.ndarray] = {}

        # prepare node outflow containers (one entry per outgoing link)
        node_outflows_over_time: dict[str, dict[str, np.ndarray]] = {}
        for node in self.network.list_nodes():
            node_outflows_over_time[node.id] = {}
            for out in node.outgoing:
                node_outflows_over_time[node.id][out.id] = np.zeros(num_timesteps)

        if disturbance_history.size == 0:
            raise ValueError(
                "Disturbance history is empty. Cannot reconstruct turning rates and origin demands for plotting."
            )

        for t in range(num_timesteps - 1):
            (origin_demands_t, turning_rates_t, _, _) = (
                self.network.disturbance_vec_to_network_dict(disturbance_history[:, t])
            )

            # initialize origin array on first disturbance timestep
            if t == 0:
                for origin_id in origin_demands_t.keys():
                    origin_demands_over_time[origin_id] = np.zeros(num_timesteps - 1)

            # store origin demand values
            for origin_id, val in origin_demands_t.items():
                origin_demands_over_time[origin_id][t] = float(val)

            # compute node-level outflows by multiplying total upstream flow by turning rates
            for node in self.network.list_nodes():
                # compute total available upstream flow Qn at time t (sum of upstream link outflows)
                Qn = 0.0
                for inc in node.incoming:
                    inc_id = inc.id
                    flow_arr = flows_over_time.get(inc_id, np.zeros((1, num_timesteps)))
                    if isinstance(inc, MotorwayLink):
                        Qn += float(flow_arr[-1, t])
                    else:
                        Qn += float(flow_arr[0, t])

                # get turning rates for this node at time t
                node_rates = turning_rates_t.get(node.id, None)

                # If turning rates are present and non-zero, distribute Qn by the rates.
                # Otherwise fall back to the actual link inflows recorded in `flows_over_time`
                if node_rates is not None:
                    # normalize provided rates and distribute
                    for out in node.outgoing:
                        rate = float(node_rates.get(out.id, 0.0))
                        node_outflows_over_time[node.id][out.id][t] = Qn * (
                            rate / sum(float(v) for v in node_rates.values())
                        )
                else:
                    warnings.warn(
                        f"Turning rates for node '{node.id}' at time {t} not found. Falling back to recorded link inflows for outflow estimation.",
                        stacklevel=2,
                    )

                    # fallback: use recorded link inflow (first cell for outgoing links)
                    for out in node.outgoing:
                        node_outflows_over_time[node.id][out.id][t] = float(
                            flows_over_time[out.id][0, t]
                        )

        # copy last disturbance values forward to the final state timestep
        for _, outs in node_outflows_over_time.items():
            for _, arr in outs.items():
                if num_timesteps > 1:
                    arr[-1] = arr[-2]

        # extract demands for onramps as equal to the origins they are fed by
        # delayed by one timestep due to the store-and-forward nature of the origin
        onramp_demands_over_time: dict[str, np.ndarray] = {}
        for node in self.network.list_nodes():
            for link in node.outgoing:
                if isinstance(link, Onramp):
                    # assume single origin feeding the onramp (system structure requirement)
                    if not (
                        len(node.incoming) == 1 and isinstance(node.incoming[0], Origin)
                    ):
                        raise ValueError(
                            f"Onramp {link.id} must be fed by exactly one origin. Found: {[type(lk).__name__ for lk in node.incoming]}"
                        )

                    origin_id = node.incoming[0].id
                    onramp_demands_over_time[link.id] = np.zeros(num_timesteps - 1)
                    for t in range(1, num_timesteps - 1):
                        onramp_demands_over_time[link.id][t] = flows_over_time[
                            origin_id
                        ][0, t - 1]

        # ===== PART 1: Per-Link Plots for MotorwayLinks =====
        print("  Creating per-link density/flow/speed plots...")
        for node in self.network.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    link_inflow = None
                    if node_outflows_over_time and node.id in node_outflows_over_time:
                        link_inflow = node_outflows_over_time[node.id].get(link.id)

                    self._plot_motorway_link_results(
                        link=link,
                        time_seconds=time_seconds,
                        densities=densities_over_time[link.id],
                        flows=flows_over_time[link.id],
                        speeds=speeds_over_time[link.id],
                        save_dir=save_dir,
                        link_inflow=link_inflow,
                    )

        # ===== PART 2: Per-Node Inflow Plots (Origins) =====
        print("  Creating per-node inflow plots (origins)...")
        for node in self.network.list_nodes():
            inflow_components = [
                link for link in node.incoming if isinstance(link, (Origin, Onramp))
            ]
            if inflow_components:
                self._plot_node_inflows(
                    node=node,
                    inflow_components=inflow_components,
                    time_seconds=time_seconds,
                    flows_over_time=flows_over_time,
                    origin_queues_over_time=origin_queues_over_time,
                    onramp_queues_over_time=onramp_queues_over_time,
                    origin_demands_over_time=origin_demands_over_time,
                    onramp_demands_over_time=onramp_demands_over_time,
                    save_dir=save_dir,
                )

        # ===== PART 3: Per-Node Outflow Plots (Offramps & Destinations) =====
        print("  Creating per-node outflow plots (offramps & destinations)...")
        for node in self.network.list_nodes():
            outflow_components = [
                link
                for link in node.outgoing
                if isinstance(link, (Offramp, Destination))
            ]
            if outflow_components:
                self._plot_node_outflows(
                    node=node,
                    outflow_components=outflow_components,
                    time_seconds=time_seconds,
                    flows_over_time=flows_over_time,
                    offramp_queues_over_time=offramp_queues_over_time,
                    save_dir=save_dir,
                )

        # ===== PART 4: 3D Surface Plots for each MotorwayLink =====
        print("  Creating 3D surface plots for motorway links...")
        for node in self.network.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink) and len(link) > 1:
                    # skip single-cell links for 3D plots (no spatial dimension to visualize)
                    self._plot_motorway_link_3d(
                        link=link,
                        time_seconds=time_seconds,
                        densities=densities_over_time[link.id],
                        flows=flows_over_time[link.id],
                        speeds=speeds_over_time[link.id],
                        save_dir=save_dir,
                    )

        # ===== PART 5: Per-Node Summary Plots (All Inflows & Outflows) =====
        print("  Creating per-node summary plots...")
        for node in self.network.list_nodes():
            self._plot_node_summary(
                node=node,
                time_seconds=time_seconds,
                flows_over_time=flows_over_time,
                node_outflows_over_time=node_outflows_over_time,
                save_dir=save_dir,
            )

        print(f"All plots saved to {save_dir}")

    def _plot_motorway_link_results(
        self,
        link: MotorwayLink,
        time_seconds: NDArray[np.float64],
        densities: NDArray[np.float64],
        flows: NDArray[np.float64],
        speeds: NDArray[np.float64],
        save_dir: str,
        link_inflow: NDArray[np.float64] | None = None,
    ) -> None:
        """Create density, flow, and speed plots for a motorway link.

        Args:
            link: The MotorwayLink to plot.
            time_seconds: 1-D array of time points in seconds.
            densities: 2-D array of densities over time (cells x time).
            flows: 2-D array of flows over time (cells x time).
            speeds: 2-D array of speeds over time (cells x time).
            save_dir: Directory where plots should be saved.
        """
        num_cells = len(link)
        ncols = 3

        # include one extra subplot slot for the link inflow so it sits inline
        n_cells_needed = num_cells + 1
        nrows = math.ceil(n_cells_needed / ncols)
        actual_duration = float(time_seconds[-1])

        # calculate max values for y-axis scaling
        max_density = max(np.max(densities) * 1.1, link.rho_jam * 1.1)
        max_flow = max(np.max(flows[:, :-1]) * 1.1, link.Qc * 1.1)
        max_speed = max(np.max(speeds[:, :-1]) * 1.1, link.vf * 1.1)

        # figure 1: Density
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig1.suptitle(
            f"Vehicle Density - Link {link.id}", fontsize=14, fontweight="bold"
        )

        # properly handle axes array structure
        if nrows == 1 and ncols == 1:
            axes1 = [axes1]
        elif nrows == 1 or ncols == 1:
            axes1 = axes1.flatten()
        else:
            axes1 = axes1.flatten()

        for i, _ in link.enumerate_cells():
            axes1[i].plot(time_seconds, densities[i, :], linewidth=1.5)
            axes1[i].axhline(link.rho_jam, color="red", linestyle="--", linewidth=1)
            axes1[i].set_ylim([0, max(link.rho_jam * 1.1, max_density)])
            axes1[i].set_xlim([0, actual_duration])
            axes1[i].set_xlabel("time (s)")
            axes1[i].set_ylabel("density (veh/km/lane)")
            axes1[i].grid(True)
            axes1[i].set_title(f"Cell {i + 1}")

        for ax in axes1[num_cells:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_density.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig1)

        # figure 2: Flow (with an additional inflow subplot shown inline)
        # create a gridspec large enough to hold the inflow plus all cell plots
        fig2 = plt.figure(figsize=(4 * ncols, 3 * nrows))
        fig2.suptitle(f"Vehicle Flow - Link {link.id}", fontsize=14, fontweight="bold")
        gs = fig2.add_gridspec(nrows, ncols)

        # first cell: inflow axis (inline with the cell subplots)
        ax_inflow = fig2.add_subplot(gs[0, 0])
        if link_inflow is not None and link_inflow.size > 0:
            ax_inflow.plot(
                time_seconds[:-1],
                link_inflow[:-1],
                color="tab:orange",
                linewidth=2,
                label="Link inflow (node outflow)",
            )

            # draw capacity of first cell on inflow subplot
            ax_inflow.axhline(link.Qc, color="red", linestyle="--", linewidth=1)
            ax_inflow.set_xlim(0, actual_duration)
            ax_inflow.set_xlabel("time (s)")
            ax_inflow.set_ylabel("flow (veh/h)")
            ax_inflow.grid(True)
            ax_inflow.set_title("Link inflow (node outflow)")
            ax_inflow.legend(fontsize="small", frameon=False)
        else:
            ax_inflow.set_visible(False)

        # remaining grid positions are used for per-cell flow axes (skip [0,0])
        axes2 = []
        positions = [
            (r, c)
            for r in range(nrows)
            for c in range(ncols)
            if not (r == 0 and c == 0)
        ]
        for r, c in positions:
            # make all cell axes share y-axis with the inflow axis so they align automatically
            axes2.append(fig2.add_subplot(gs[r, c], sharey=ax_inflow))

        for i, _ in link.enumerate_cells():
            ax = axes2[i]
            ax.plot(
                time_seconds[:-1], flows[i, :-1], linewidth=1.5, label="Cell outflow"
            )
            ax.axhline(link.Qc, color="red", linestyle="--", linewidth=1)

            # rely on shared y-axis and autoscale to align y-limits across subplots
            ax.set_xlim([0, actual_duration])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("flow (veh/h)")
            ax.grid(True)
            ax.set_title(f"Cell {i + 1}")

        for ax in axes2[num_cells:]:
            ax.set_visible(False)

        # trigger autoscale across shared y-axes so all subplots use the same y-range
        axes_for_autoscale = [ax_inflow] + [axes2[i] for i, _ in link.enumerate_cells()]
        for a in axes_for_autoscale:
            a.relim()
            a.autoscale_view(scalex=False, scaley=True)

        plt.tight_layout()
        fig2.savefig(
            os.path.join(save_dir, f"{link.id}_flow.png"), dpi=200, bbox_inches="tight"
        )
        plt.close(fig2)

        # figure 3: Speed
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig3.suptitle(f"Vehicle Speed - Link {link.id}", fontsize=14, fontweight="bold")

        # properly handle axes array structure
        if nrows == 1 and ncols == 1:
            axes3 = [axes3]
        elif nrows == 1 or ncols == 1:
            axes3 = axes3.flatten()
        else:
            axes3 = axes3.flatten()

        for i, _ in link.enumerate_cells():
            vf_cell = link.vf
            axes3[i].plot(time_seconds[:-1], speeds[i, :-1], linewidth=1.5)
            axes3[i].axhline(vf_cell, color="red", linestyle="--", linewidth=1)
            axes3[i].set_ylim([0, max(vf_cell * 1.1, max_speed)])
            axes3[i].set_xlim([0, actual_duration])
            axes3[i].set_xlabel("time (s)")
            axes3[i].set_ylabel("speed (km/h)")
            axes3[i].grid(True)
            axes3[i].set_title(f"Cell {i + 1}")

        for ax in axes3[num_cells:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_speed.png"), dpi=200, bbox_inches="tight"
        )
        plt.close(fig3)

    def _plot_node_inflows(
        self,
        node: Node,
        inflow_components: list,
        time_seconds: NDArray[np.float64],
        flows_over_time: dict,
        origin_queues_over_time: dict,
        onramp_queues_over_time: dict,
        origin_demands_over_time: dict,
        onramp_demands_over_time: dict,
        save_dir: str,
    ) -> None:
        """Create inflow plots (demand+flow and queue) for origins at a node."""
        num_inflows = len(inflow_components)
        ncols = 2  # demand+flow, queue
        nrows = num_inflows

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        fig.suptitle(f"Inflows at Node {node.id}", fontsize=14, fontweight="bold")

        # normalize axes indexing to 2D
        if num_inflows == 1:
            axes = np.array([[axes[0], axes[1]]])
        else:
            axes = np.array(axes).reshape(nrows, ncols)

        actual_duration = time_seconds[-1]

        for row_idx, link in enumerate(inflow_components):
            link_id = link.id
            is_origin = isinstance(link, Origin)

            # get demand and flow data
            if is_origin:
                demand = origin_demands_over_time.get(
                    link_id, np.zeros(len(time_seconds) - 1)
                )
                queue = origin_queues_over_time.get(
                    link_id, np.zeros(len(time_seconds))
                )
            else:
                demand = onramp_demands_over_time.get(
                    link_id, np.zeros(len(time_seconds) - 1)
                )
                queue = onramp_queues_over_time.get(
                    link_id, np.zeros(len(time_seconds))
                )

            flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))
            if len(flow.shape) > 1:
                flow = flow[0, :]

            # calculate max values for scaling
            max_demand = np.max(demand) * 1.1 if np.max(demand) > 0 else 2500
            max_flow = np.max(flow[:-1]) * 1.1 if np.max(flow[:-1]) > 0 else 2500
            max_queue = np.max(queue[:-1]) * 1.1 if np.max(queue[:-1]) > 0 else 100
            combined_max = max(max_demand, max_flow)

            # plot demand and flow
            axes[row_idx, 0].plot(
                time_seconds[:-1], demand, linewidth=1.5, label="Demand"
            )
            axes[row_idx, 0].plot(
                time_seconds[:-1], flow[:-1], linewidth=1.5, label="Flow"
            )
            axes[row_idx, 0].grid(True)
            axes[row_idx, 0].set_xlim([0, actual_duration])
            axes[row_idx, 0].set_ylim([0, combined_max])
            axes[row_idx, 0].set_xlabel("time (s)")
            axes[row_idx, 0].set_ylabel("veh/h")
            axes[row_idx, 0].set_title(
                f"{type(link).__name__} {link_id} - Demand & Flow"
            )
            axes[row_idx, 0].legend(fontsize="small", ncol=2, frameon=False)

            # plot queue
            axes[row_idx, 1].plot(
                time_seconds[:-1], queue[:-1], linewidth=1.5, color="tab:gray"
            )
            axes[row_idx, 1].grid(True)
            axes[row_idx, 1].set_xlim([0, actual_duration])
            axes[row_idx, 1].set_ylim([0, max_queue])
            axes[row_idx, 1].set_xlabel("time (s)")
            axes[row_idx, 1].set_ylabel("Queue (veh)")
            axes[row_idx, 1].set_title(f"{type(link).__name__} {link_id} - Queue")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"node_{node.id}_inflows.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_node_outflows(
        self,
        node: Node,
        outflow_components: list,
        time_seconds: NDArray[np.float64],
        flows_over_time: dict,
        offramp_queues_over_time: dict,
        save_dir: str,
    ) -> None:
        """Create outflow plots for offramps and destinations at a node."""
        num_outflows = len(outflow_components)

        # count offramps to determine if we need queue plots
        num_offramps = sum(
            1 for link in outflow_components if isinstance(link, Offramp)
        )
        ncols = 2 if num_offramps > 0 else 1  # flow, and queue if offramps exist
        nrows = num_outflows

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        fig.suptitle(f"Outflows at Node {node.id}", fontsize=14, fontweight="bold")

        # normalize axes indexing
        if num_outflows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif num_outflows == 1:
            axes = np.array([[axes[0], axes[1]]])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])
        else:
            axes = np.array(axes).reshape(nrows, ncols)

        actual_duration = time_seconds[-1]

        for row_idx, link in enumerate(outflow_components):
            link_id = link.id
            is_offramp = isinstance(link, Offramp)

            flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))
            if len(flow.shape) > 1:
                flow = flow[0, :]

            max_flow = np.max(flow[:-1]) * 1.1 if np.max(flow[:-1]) > 0 else 2500

            # plot flow
            axes[row_idx, 0].plot(time_seconds[:-1], flow[:-1], linewidth=1.5)
            axes[row_idx, 0].grid(True)
            axes[row_idx, 0].set_xlim([0, actual_duration])
            axes[row_idx, 0].set_ylim([0, max_flow])
            axes[row_idx, 0].set_xlabel("time (s)")
            axes[row_idx, 0].set_ylabel("flow (veh/h)")
            axes[row_idx, 0].set_title(f"{type(link).__name__} {link_id} - Flow")

            # plot queue if it's an offramp
            if is_offramp and ncols == 2:
                queue = offramp_queues_over_time.get(
                    link_id, np.zeros(len(time_seconds))
                )
                max_queue = np.max(queue[:-1]) * 1.1 if np.max(queue[:-1]) > 0 else 100

                axes[row_idx, 1].plot(
                    time_seconds[:-1], queue[:-1], linewidth=1.5, color="tab:gray"
                )
                axes[row_idx, 1].grid(True)
                axes[row_idx, 1].set_xlim([0, actual_duration])
                axes[row_idx, 1].set_ylim([0, max_queue])
                axes[row_idx, 1].set_xlabel("time (s)")
                axes[row_idx, 1].set_ylabel("Queue (veh)")
                axes[row_idx, 1].set_title(f"Offramp {link_id} - Queue")
            elif ncols == 2:
                # Hide queue subplot for destinations
                axes[row_idx, 1].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"node_{node.id}_outflows.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_motorway_link_3d(
        self,
        link: MotorwayLink,
        time_seconds: NDArray[np.float64],
        densities: NDArray[np.float64],
        flows: NDArray[np.float64],
        speeds: NDArray[np.float64],
        save_dir: str,
    ) -> None:
        """Create 3D surface plots for a motorway link."""
        num_cells = len(link)
        actual_duration = time_seconds[-1]

        # create meshgrids
        x_full, y_full = np.meshgrid(time_seconds, np.arange(1, num_cells + 1))
        x_truncated, y_truncated = np.meshgrid(
            time_seconds[:-1], np.arange(1, num_cells + 1)
        )

        # calculate max values
        max_rho_jam = link.rho_jam
        max_capacity = link.Qc
        max_vf = link.vf

        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(
            f"3D Visualization - Link {link.id}", fontsize=14, fontweight="bold"
        )

        # 3D density plot
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.plot_surface(
            x_full, y_full, densities, cmap="viridis", edgecolor="none", alpha=0.9
        )
        ax1.view_init(elev=30, azim=-37.5)
        ax1.set_xlabel("time (s)", rotation=30)
        ax1.set_ylabel("Cell", rotation=-37.5)
        ax1.set_zlabel("density (veh/km/lane)")
        ax1.set_xlim([0, actual_duration])
        ax1.set_ylim([1, num_cells])
        ax1.set_zlim([0, max_rho_jam * 1.1])

        # 3D flow plot
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.plot_surface(
            x_truncated,
            y_truncated,
            flows[:, :-1],
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )
        ax2.view_init(elev=30, azim=-37.5)
        ax2.set_xlabel("time (s)", rotation=30)
        ax2.set_ylabel("Cell", rotation=-37.5)
        ax2.set_zlabel("flow (veh/h)")
        ax2.set_xlim([0, actual_duration])
        ax2.set_ylim([1, num_cells])
        ax2.set_zlim([0, max_capacity * 1.1])

        # 3D speed plot
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot_surface(
            x_truncated,
            y_truncated,
            speeds[:, :-1],
            cmap="viridis",
            edgecolor="none",
            alpha=0.9,
        )
        ax3.view_init(elev=30, azim=-37.5)
        ax3.set_xlabel("time (s)", rotation=30)
        ax3.set_ylabel("Cell", rotation=-37.5)
        ax3.set_zlabel("speed (km/h)")
        ax3.set_xlim([0, actual_duration])
        ax3.set_ylim([1, num_cells])
        ax3.set_zlim([0, max_vf * 1.1])

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{link.id}_3d_surfaces.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_node_summary(
        self,
        node: Node,
        time_seconds: NDArray[np.float64],
        flows_over_time: dict,
        node_outflows_over_time: dict,
        save_dir: str = "results",
    ) -> None:
        """Create a summary plot showing all inflows and outflows at a node."""
        # collect all incoming and outgoing links
        incoming_links = node.incoming
        outgoing_links = node.outgoing

        # skip nodes with no meaningful flow to plot
        if not incoming_links and not outgoing_links:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Node {node.id} - Flow Summary", fontsize=14, fontweight="bold")
        actual_duration = time_seconds[-1]

        # plot incoming flows and compute total incoming (accumulate to avoid extra copies)
        incoming_total = np.zeros(len(time_seconds) - 1)
        max_inflow = 0
        if incoming_links:
            for link in incoming_links:
                link_id = link.id
                flow = flows_over_time.get(link_id, np.zeros((1, len(time_seconds))))

                # for motorway links, take the flow from the last cell (outflow of the link)
                if isinstance(link, MotorwayLink):
                    flow_to_plot = flow[-1, :-1]
                elif len(flow.shape) > 1:
                    flow_to_plot = flow[0, :-1]
                else:
                    flow_to_plot = flow[:-1]

                flow_arr = np.asarray(flow_to_plot)
                incoming_total += flow_arr

                axes[0].plot(
                    time_seconds[:-1],
                    flow_arr,
                    linewidth=1.0,
                    label=f"{type(link).__name__} {link_id}",
                )
                max_inflow = max(
                    max_inflow, np.max(flow_arr) if flow_arr.size > 0 else 0
                )

            axes[0].grid(True)
            axes[0].set_xlim([0, actual_duration])
            axes[0].set_xlabel("time (s)")
            axes[0].set_ylabel("flow (veh/h)")
            axes[0].set_title("Incoming Flows (last segment of incoming links)")

            # total incoming line
            axes[0].plot(
                time_seconds[:-1],
                incoming_total,
                color="k",
                linewidth=2.2,
                linestyle=(0, (5, 2)),
                label="Total incoming",
            )
            axes[0].legend(fontsize="small", frameon=False)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No incoming links",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_axis_off()

        # plot outgoing flows (use node-level outflows when available) and compute total outgoing
        outgoing_total = np.zeros(len(time_seconds) - 1)
        max_outflow = 0
        if outgoing_links:
            for link in outgoing_links:
                flow_to_plot = node_outflows_over_time[node.id][link.id][:-1]
                label_suffix = "node outflow"

                flow_arr = np.asarray(flow_to_plot)
                outgoing_total += flow_arr

                axes[1].plot(
                    time_seconds[:-1],
                    flow_arr,
                    linewidth=1.0,
                    label=f"{type(link).__name__} {link.id} ({label_suffix})",
                )
                max_outflow = max(
                    max_outflow, np.max(flow_arr) if flow_arr.size > 0 else 0
                )

            axes[1].grid(True)
            axes[1].set_xlim([0, actual_duration])
            axes[1].set_xlabel("time (s)")
            axes[1].set_ylabel("flow (veh/h)")
            axes[1].set_title("Outgoing Flows (node outflows)")

            # total outgoing line
            axes[1].plot(
                time_seconds[:-1],
                outgoing_total,
                color="k",
                linewidth=2.2,
                linestyle=(0, (5, 2)),
                label="Total outgoing",
            )

            axes[1].legend(fontsize="small", frameon=False)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No outgoing links",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()

        # ensure both subplots use the same y-axis scale for easy comparison by
        # sharing autoscaling rather than computing explicit limits
        axes[0].relim()
        axes[1].relim()
        axes[0].autoscale_view(scalex=False, scaley=True)
        axes[1].autoscale_view(scalex=False, scaley=True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"node_{node.id}_summary.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    def visualize(
        self,
        results_filepath: str,
        output_filepath: str,
        fps: int = 25,
        subsampling: int = 1,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 300,
    ) -> None:
        """Generate a video visualization of simulation results.

        Creates an AVI video showing the network topology with links colored
        by density. Color scheme transitions from bright green (low density)
        through dark green (critical density) to orange and dark red (jam density).

        Args:
            results_filepath: Path to simulation results JSON file
            output_filepath: Path for output video file (.avi)
            fps: Frames per second for output video
            subsampling: Number of intervals to split each time interval into for smoother animation
                        (1=no interpolation, 2=split intervals in half, 3=split into thirds, etc.)
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for rendering

        Raises:
            ValueError: If nodes lack position information or metadata is missing
            ImportError: If opencv-python is not installed
        """
        # load simulation results
        time_array, state_history, _, metadata = self.load_results(
            filepath=results_filepath, network=self.network
        )

        if metadata is None:
            raise ValueError(
                "Simulation results file lacks metadata. "
                "Only files with metadata can be visualized."
            )

        critical_densities = metadata["critical_densities"]
        link_properties = metadata["link_properties"]

        # apply frame interpolation if requested
        if subsampling > 1:
            time_array, state_history = self._interpolate_frames(
                state_history, time_array, subsampling
            )

        num_frames = state_history.shape[1]

        # compute plot bounds from node positions
        positions = []
        for node in self.network.list_nodes():
            if node.position is None:
                raise ValueError(
                    f"Node '{node.id}' lacks position information. "
                    "All nodes must have positions set for visualization."
                )
            positions.append(node.position)

        positions_arr = np.array(positions)
        x_min, y_min = positions_arr.min(axis=0)
        x_max, y_max = positions_arr.max(axis=0)

        # add padding for ramps and visual clarity
        x_padding = 0.15 * (x_max - x_min) if x_max > x_min else 1.0
        y_padding = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # initialize video writer
        frame_width = int(figsize[0] * dpi)
        frame_height = int(figsize[1] * dpi)
        video_writer = cv2.VideoWriter(
            filename=output_filepath,
            fourcc=cv2.VideoWriter.fourcc(*"MJPG"),
            fps=fps,
            frameSize=(frame_width, frame_height),
        )

        try:
            # generate frames
            for t in tqdm(range(num_frames)):
                # create figure
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

                # extract current state
                flows, densities, _, _, onramp_queues, offramp_queues = (
                    self.network.state_vec_to_network_dict(state_history[:, t])
                )

                # draw network state on axes using helper
                self._draw_network_state_on_axes(
                    ax=ax,
                    flows=flows,
                    densities=densities,
                    onramp_queues=onramp_queues,
                    offramp_queues=offramp_queues,
                    critical_densities=critical_densities,
                    link_properties=link_properties,
                    time_value=time_array[t],
                    title="Traffic Flow Simulation",
                    x_bounds=(x_min, x_max),
                    y_bounds=(y_min, y_max),
                )

                # convert figure to frame and write
                img_bgr = self._fig_to_frame(fig, frame_width, frame_height)
                video_writer.write(img_bgr)

                # close figure to free memory
                plt.close(fig)

        finally:
            # release video writer
            video_writer.release()

    def visualize_comparison(
        self,
        result_filepaths: list[str],
        labels: list[str],
        output_filepath: str,
        fps: int = 25,
        subsampling: int = 1,
        figsize: tuple[float, float] = (16, 12),
        dpi: int = 300,
        layout: tuple[int, int] | None = None,
    ) -> None:
        """Generate a comparison video showing multiple simulations side-by-side.

        Creates an AVI video with multiple subplots showing different simulation
        results synchronized in time. This is useful for comparing calibration
        results or different scenarios.

        Args:
            result_filepaths: List of paths to simulation results JSON files
            labels: List of labels for each simulation (same length as result_filepaths)
            output_filepath: Path for output video file (.avi)
            fps: Frames per second for output video
            subsampling: Number of intervals to split each time interval into for smoother animation
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for rendering
            layout: Optional tuple (nrows, ncols) for subplot layout. If None, auto-computed
                   to create an approximately square grid.

        Raises:
            ValueError: If inputs are invalid or simulations have different lengths
            ImportError: If opencv-python is not installed
        """
        if len(result_filepaths) != len(labels):
            raise ValueError(
                f"Number of result files ({len(result_filepaths)}) must match "
                f"number of labels ({len(labels)})"
            )

        if len(result_filepaths) == 0:
            raise ValueError("Must provide at least one result file")

        # load all simulation results
        simulations = []
        for filepath in result_filepaths:
            time_array, state_history, _, metadata = self.load_results(
                filepath=filepath, network=self.network
            )

            if metadata is None:
                raise ValueError(
                    f"Simulation results file '{filepath}' lacks metadata. "
                    "Only files with metadata can be visualized."
                )

            simulations.append(
                {
                    "time_array": time_array,
                    "state_history": state_history,
                    "metadata": metadata,
                }
            )

        n_sims = len(simulations)
        if n_sims == 1:
            raise ValueError(
                "Only one simulation provided. Simulation comparison plotting requires results from at least 2 simulation runs."
            )

        # validate all simulations have same length and aligned timestamps
        num_timesteps = simulations[0]["time_array"].shape[0]
        reference_time_array = simulations[0]["time_array"]

        for i, sim in enumerate(simulations[1:], start=1):
            if sim["time_array"].shape[0] != num_timesteps:
                raise ValueError(
                    f"Simulation {i+1} has {sim['time_array'].shape[0]} timesteps, "
                    f"but simulation 1 has {num_timesteps}. All simulations must have "
                    "the same number of timesteps for comparison."
                )

            # verify that time arrays are actually aligned (not just same length)
            if not np.allclose(
                sim["time_array"], reference_time_array, rtol=1e-9, atol=1e-12
            ):
                raise ValueError(
                    f"Simulation {i+1} has misaligned timestamps compared to simulation 1. "
                    f"All simulations must have identical time arrays for synchronized comparison. "
                    f"If comparing simulations with different time grids, resampling is required."
                )

        # apply frame interpolation if requested
        for sim in simulations:
            if subsampling > 1:
                sim["time_array"], sim["state_history"] = self._interpolate_frames(
                    sim["state_history"], sim["time_array"], subsampling
                )

        num_frames = simulations[0]["state_history"].shape[1]

        # determine subplot layout
        if layout is None:
            # auto-compute roughly square layout
            ncols = int(np.ceil(np.sqrt(n_sims)))
            nrows = int(np.ceil(n_sims / ncols))
        else:
            nrows, ncols = layout
            if nrows * ncols < n_sims:
                raise ValueError(
                    f"Layout {layout} has {nrows * ncols} subplots but "
                    f"{n_sims} simulations provided"
                )

        # compute plot bounds from node positions (shared across all subplots)
        positions = []
        for node in self.network.list_nodes():
            if node.position is None:
                raise ValueError(
                    f"Node '{node.id}' lacks position information. "
                    "All nodes must have positions set for visualization."
                )
            positions.append(node.position)

        positions_arr = np.array(positions)
        x_min, y_min = positions_arr.min(axis=0)
        x_max, y_max = positions_arr.max(axis=0)

        # add padding for ramps and visual clarity
        x_padding = 0.15 * (x_max - x_min) if x_max > x_min else 1.0
        y_padding = 0.15 * (y_max - y_min) if y_max > y_min else 1.0

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # initialize video writer
        frame_width = int(figsize[0] * dpi)
        frame_height = int(figsize[1] * dpi)
        video_writer = cv2.VideoWriter(
            filename=output_filepath,
            fourcc=cv2.VideoWriter.fourcc(*"MJPG"),
            fps=fps,
            frameSize=(frame_width, frame_height),
        )

        try:
            # generate frames
            for t in tqdm(range(num_frames)):
                # create figure with subplots
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

                # flatten array for consistent indexing
                axes = np.array(axes).flatten()

                # draw each simulation on its subplot
                for i, (sim, label) in enumerate(zip(simulations, labels)):
                    ax = axes[i]

                    # extract current state
                    flows, densities, _, _, onramp_queues, offramp_queues = (
                        self.network.state_vec_to_network_dict(
                            sim["state_history"][:, t]
                        )
                    )

                    # draw network state on axes using helper
                    self._draw_network_state_on_axes(
                        ax=ax,
                        flows=flows,
                        densities=densities,
                        onramp_queues=onramp_queues,
                        offramp_queues=offramp_queues,
                        critical_densities=sim["metadata"]["critical_densities"],
                        link_properties=sim["metadata"]["link_properties"],
                        time_value=sim["time_array"][t],
                        title=label,
                        x_bounds=(x_min, x_max),
                        y_bounds=(y_min, y_max),
                    )

                # hide unused subplots
                for i in range(n_sims, len(axes)):
                    axes[i].axis("off")

                plt.tight_layout()

                # convert figure to frame and write
                img_bgr = self._fig_to_frame(fig, frame_width, frame_height)
                video_writer.write(img_bgr)

                # close figure to free memory
                plt.close(fig)

        finally:
            # release video writer
            video_writer.release()

    @staticmethod
    def _density_to_color(
        rho: float, rho_crit: float, rho_jam: float
    ) -> tuple[int, int, int]:
        """Convert density to RGB color for visualization.

        Maps density values to a color gradient:
        - Low density (0 -> rho_crit): bright green -> dark green
        - High density (rho_crit -> rho_jam): dark green -> orange -> dark red

        Args:
            rho: Current density (veh/km/lane)
            rho_crit: Critical density (veh/km/lane)
            rho_jam: Jam density (veh/km/lane)

        Returns:
            RGB tuple with values in range [0, 255]
        """
        # use class-level color constants
        bright_green = Simulation.COLOR_BRIGHT_GREEN
        dark_green = Simulation.COLOR_DARK_GREEN
        orange = Simulation.COLOR_ORANGE
        dark_red = Simulation.COLOR_DARK_RED

        if rho < rho_crit:
            # interpolate from bright green to dark green
            ratio = rho / rho_crit if rho_crit > 0 else 0.0
            r = int(bright_green[0] + ratio * (dark_green[0] - bright_green[0]))
            g = int(bright_green[1] + ratio * (dark_green[1] - bright_green[1]))
            b = int(bright_green[2] + ratio * (dark_green[2] - bright_green[2]))
        else:
            # interpolate from dark green through orange to dark red
            # transition to orange quickly (first 10% of range) for congestion visibility
            excess = rho - rho_crit
            range_width = rho_jam - rho_crit
            ratio = min(excess / range_width if range_width > 0 else 1.0, 1.0)

            # two-stage interpolation: dark_green -> orange (quick) -> dark_red (gradual)
            if ratio < 0.1:
                # first 10%: dark_green to orange (rapid transition)
                sub_ratio = ratio / 0.1
                r = int(dark_green[0] + sub_ratio * (orange[0] - dark_green[0]))
                g = int(dark_green[1] + sub_ratio * (orange[1] - dark_green[1]))
                b = int(dark_green[2] + sub_ratio * (orange[2] - dark_green[2]))
            else:
                # remaining 90%: orange to dark_red (gradual transition)
                sub_ratio = (ratio - 0.1) / 0.90
                r = int(orange[0] + sub_ratio * (dark_red[0] - orange[0]))
                g = int(orange[1] + sub_ratio * (dark_red[1] - orange[1]))
                b = int(orange[2] + sub_ratio * (dark_red[2] - orange[2]))

        return (r, g, b)

    @staticmethod
    def _interpolate_frames(
        state_history: NDArray[np.float64],
        time_array: NDArray[np.float64],
        subsampling: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linearly interpolate frames for smoother visualization.

        Args:
            state_history: 2-D array (state_size x timesteps)
            time_array: 1-D array of time points
            subsampling: Number of intervals to split each time interval into
                        (subsampling=1 means no interpolation, subsampling=2 splits each interval in half,
                        subsampling=3 splits into thirds, etc.)

        Returns:
            Tuple of (interpolated_time_array, interpolated_state_history)
        """
        if subsampling == 1:
            return time_array, state_history
        elif subsampling < 1:
            raise ValueError(
                "Subsampling must be an integer greater than 1 for interpolation."
            )

        num_timesteps = state_history.shape[1]
        num_interpolated = (num_timesteps - 1) * subsampling + 1

        # create interpolated arrays
        interpolated_time = np.zeros(num_interpolated, dtype=np.float64)
        interpolated_state = np.zeros(
            (state_history.shape[0], num_interpolated), dtype=np.float64
        )

        # fill in interpolated values by splitting each interval into 'subsampling' parts
        for i in range(num_timesteps - 1):
            # original frame at start of interval
            start_idx = i * subsampling
            interpolated_time[start_idx] = time_array[i]
            interpolated_state[:, start_idx] = state_history[:, i]

            # create (subsampling - 1) interpolated frames within the interval
            for j in range(1, subsampling):
                interp_idx = start_idx + j
                alpha = j / subsampling
                interpolated_time[interp_idx] = (
                    time_array[i] * (1 - alpha) + time_array[i + 1] * alpha
                )
                interpolated_state[:, interp_idx] = (
                    state_history[:, i] * (1 - alpha) + state_history[:, i + 1] * alpha
                )

        # last frame at end of final interval
        interpolated_time[-1] = time_array[-1]
        interpolated_state[:, -1] = state_history[:, -1]

        return interpolated_time, interpolated_state

    def _draw_network_state_on_axes(
        self,
        ax: Axes,
        flows: dict,
        densities: dict,
        onramp_queues: dict,
        offramp_queues: dict,
        critical_densities: dict[str, float],
        link_properties: dict[str, dict],
        time_value: float,
        title: str,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> None:
        """Draw network state on a matplotlib axes.

        Helper method that draws nodes, motorway links (colored by density),
        onramps, offramps, and time annotation on the provided axes.

        Args:
            ax: Matplotlib axes to draw on
            flows: Dictionary mapping link IDs to flow arrays
            densities: Dictionary mapping link IDs to density arrays
            onramp_queues: Dictionary mapping onramp IDs to queue lengths
            offramp_queues: Dictionary mapping offramp IDs to queue lengths
            critical_densities: Dictionary mapping link IDs to critical densities
            link_properties: Dictionary mapping link IDs to property dictionaries
            time_value: Current simulation time for annotation
            title: Title for the subplot
            x_bounds: Tuple of (x_min, x_max) for axis limits
            y_bounds: Tuple of (y_min, y_max) for axis limits
        """
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        # draw nodes
        for node in self.network.list_nodes():
            if node.position is None:
                raise ValueError(
                    f"Node '{node.id}' lacks position information. "
                    "All nodes must have positions set for visualization."
                )
            x, y = node.position
            ax.plot(x, y, "ko", markersize=4, zorder=3)

        # draw motorway links with density coloring and on- and off-ramps with congestion coloring
        for node in self.network.list_nodes():
            for link in node.outgoing:
                if isinstance(link, MotorwayLink):
                    if link.origin_node_id is None or link.destination_node_id is None:
                        raise ValueError(
                            f"Motorway link '{link.id}' is missing origin and/or destination node IDs."
                        )

                    upstream_node = self.network.get_node(link.origin_node_id)
                    downstream_node = self.network.get_node(link.destination_node_id)

                    if upstream_node is None or downstream_node is None:
                        raise ValueError(
                            f"Motorway link '{link.id}' references non-existent nodes: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    # compute maximum density for this link
                    max_rho = float(np.max(densities[link.id]))
                    rho_crit = critical_densities[link.id]
                    rho_jam = link_properties[link.id]["jam_density"]

                    # get color
                    r, g, b = self._density_to_color(max_rho, rho_crit, rho_jam)
                    color = (r / 255.0, g / 255.0, b / 255.0)

                    # draw link
                    if (
                        upstream_node.position is None
                        or downstream_node.position is None
                    ):
                        raise ValueError(
                            f"Motorway link '{link.id}' has nodes with missing position information: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    x1, y1 = upstream_node.position
                    x2, y2 = downstream_node.position
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=3,
                        solid_capstyle="round",
                        zorder=2,
                    )

                elif isinstance(link, Onramp):
                    # require both origin and destination node IDs for spatial plotting
                    if link.origin_node_id is None or link.destination_node_id is None:
                        raise ValueError(
                            f"Onramp link '{link.id}' is missing origin and/or destination node IDs."
                        )

                    upstream_node = self.network.get_node(link.origin_node_id)
                    downstream_node = self.network.get_node(link.destination_node_id)

                    if upstream_node is None or downstream_node is None:
                        raise ValueError(
                            f"Onramp link '{link.id}' references non-existent nodes: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    if (
                        upstream_node.position is None
                        or downstream_node.position is None
                    ):
                        raise ValueError(
                            f"Onramp link '{link.id}' has nodes with missing position information: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    x1, y1 = upstream_node.position
                    x2, y2 = downstream_node.position

                    # determine color based on queue and flow/capacity ratio (unchanged)
                    queue = float(onramp_queues.get(link.id, 0.0))
                    if queue > 0:
                        color = Simulation.COLOR_CONGESTION_RED
                    else:
                        flow = float(flows.get(link.id, [0.0])[0])
                        capacity = link.Qc
                        ratio = min(flow / capacity if capacity > 0 else 0.0, 1.0)
                        r = 0.6 * (1 - ratio)
                        g = 1.0 - 0.5 * ratio
                        b = 0.6 * (1 - ratio)
                        color = (r, g, b)

                    # draw full line between origin and destination nodes
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=3,
                        solid_capstyle="round",
                        zorder=2,
                    )

                elif isinstance(link, Offramp):
                    # require both origin and destination node IDs for spatial plotting
                    if link.origin_node_id is None or link.destination_node_id is None:
                        raise ValueError(
                            f"Offramp link '{link.id}' is missing origin and/or destination node IDs."
                        )

                    upstream_node = self.network.get_node(link.origin_node_id)
                    downstream_node = self.network.get_node(link.destination_node_id)

                    if upstream_node is None or downstream_node is None:
                        raise ValueError(
                            f"Offramp link '{link.id}' references non-existent nodes: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    if (
                        upstream_node.position is None
                        or downstream_node.position is None
                    ):
                        raise ValueError(
                            f"Offramp link '{link.id}' has nodes with missing position information: "
                            f"origin '{link.origin_node_id}', destination '{link.destination_node_id}'."
                        )

                    x1, y1 = upstream_node.position
                    x2, y2 = downstream_node.position

                    # determine color based on queue and flow/capacity ratio (unchanged)
                    queue = float(offramp_queues.get(link.id, 0.0))
                    if queue > 0:
                        color = Simulation.COLOR_CONGESTION_RED
                    else:
                        flow = float(flows.get(link.id, [0.0])[0])
                        capacity = link.Qc
                        ratio = min(flow / capacity if capacity > 0 else 0.0, 1.0)
                        r = 0.6 * (1 - ratio)
                        g = 1.0 - 0.5 * ratio
                        b = 0.6 * (1 - ratio)
                        color = (r, g, b)

                    # draw full line between origin and destination nodes
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=3,
                        solid_capstyle="round",
                        zorder=2,
                    )

        # add time annotation
        ax.text(
            0.02,
            0.98,
            f"Time: {time_value:.2f} h",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # set axis limits and styling
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)

    @staticmethod
    def _fig_to_frame(fig: Figure, target_width: int, target_height: int) -> Any:
        """Convert matplotlib figure to OpenCV-compatible BGR frame.

        Helper method that converts a matplotlib figure to a numpy array
        in BGR format suitable for OpenCV video writing.

        Args:
            fig: Matplotlib figure to convert
            target_width: Target frame width in pixels
            target_height: Target frame height in pixels

        Returns:
            BGR image as numpy array (uint8)
        """
        # convert figure to numpy array using backend-agnostic method
        fig.canvas.draw()
        buf, (width, height) = fig.canvas.print_to_buffer()  # type: ignore
        # convert buffer to numpy array (RGBA format)
        img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        # extract RGB channels (drop alpha)
        img_rgb = img_rgba[:, :, :3]

        # resize to target dimensions if needed
        if (height, width) != (target_height, target_width):
            img_rgb = cv2.resize(img_rgb, (target_width, target_height))

        # convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return img_bgr

    # endregion
