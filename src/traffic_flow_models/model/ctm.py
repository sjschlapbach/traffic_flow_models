from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from traffic_flow_models.network.network import Network, Cell
from .helpers import (
    calculate_segment_input_flow,
    calculate_regulated_onramp_flow,
    cap_cell_flows,
    update_queue,
    calculate_cell_flow,
)


class CTM:
    """Cell-Transmission Model (CTM) model implementation.

    This class provides a set of functions to use the CTM model either on the
    network structure in this repository or as simple step functions to a custom
    user-defined structure outside of this package.

    Please note that the CTM model is a first-order macrsocopic traffic flow model
    that only updates the density directly. The second degree of freedom in the
    relation between flow, density and speed is set through a proportional reduction
    of the cell inflows in case they exceed the cell's capacity.

    Methods:
        cell_update: Compute next-step density and a speed estimate for a cell.
    """

    def __init__(self) -> None:
        """Create a CTM model instance."""
        return

    # TODO: extend tests to include this function
    def critical_density(self, cell: Cell) -> float:
        """Return the critical density for a given cell.

        Args:
            cell: The `Cell` instance for which to compute the critical density.

        Returns:
            The critical density (vehicles per kilometer per lane).
        """
        return cell.Qc_lane / cell.vf

    # TODO: extend tests to include this function
    def backward_wave_speed(self, cell: Cell) -> float:
        """Return the backward wave speed for a given cell.

        Args:
            cell: The `Cell` instance for which to compute the backward wave speed.

        Returns:
            The backward wave speed computed based on the critical density.
        """
        rho_cr = self.critical_density(cell=cell)
        return cell.Qc / (cell.rho_jam - rho_cr)

    def cell_update(
        self,
        cell_lanes: int,
        cell_length: float,
        density: float,
        upstream_flow: float,
        cell_flow: float,
        onramp_flow: float,
        offramp_flow: float,
        dt: float,
    ) -> Tuple[float, float]:
        """Compute the CTM density and speed update for a single cell.

        The method implements the discrete CTM conservation update for a
        homogeneous cell and returns a tuple ``(next_density, speed)`` where
        ``next_density`` is the density after time step ``dt`` and ``speed`` is
        a simple flow-derived speed estimate.

        It is assumed that potential flow reductions that are required to
        respect physical cell capacity constraints are applied before calling
        this function.

        Args:
            cell_lanes: Number of lanes in the cell.
            cell_length: Cell length (same units as density denominator,
                typically kilometers).
            density: Current density (vehicles per length per lane).
            upstream_flow: Flow entering the cell from upstream (vehicles per
                time unit).
            cell_flow: Flow leaving the cell (vehicles per time unit).
            onramp_flow: Flow entering from an onramp attached to the cell.
            offramp_flow: Flow leaving via an offramp attached to the cell.
            dt: Time step over which to integrate (time units consistent with
                the flow units).

        Returns:
            A tuple (next_density, speed) where next_density is the updated
            density and speed is calculated through the fundamental diagram
            relation between the cell flow and the cell density (triangular
            fundamental diagram).
        """
        # update the density based on the CTM conservation equation
        next_density = density + dt * (
            upstream_flow + onramp_flow - offramp_flow - cell_flow
        ) / (cell_length * cell_lanes)

        # update the speed only based on the computed flow and density (first
        # order model -> no explicit speed model updates)
        speed = cell_flow / (cell_lanes * density) if cell_flow > 0 else 0.0

        return next_density, speed

    def step(
        self,
        network: Network,
        density: NDArray[np.float64],
        speed: NDArray[
            np.float64
        ],  # unused - required for input argument consistency across models
        flow: NDArray[
            np.float64
        ],  # unused - required for input argument consistency across models
        mainline_demand: float,
        input_queue: int,
        onramp_demand: NDArray[np.float64],
        onramp_queue: NDArray[np.float64],
        onramp_flow: NDArray[np.float64],
        dt: float,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Advance the CTM model by a single time step for the whole network.

        This method computes the mainline cell flows, onramp flows and the
        resulting next-step densities and speeds for every cell in the
        provided `network` using the Cell-Transmission Model (CTM).

        Args:
            network: The `Network` object describing the cells, their
                fundamental diagram parameters and ramp connectivity.
            density: 1-D array of current densities for each cell
                (vehicles per length per lane), shape `(num_cells,)`.
            mainline_demand: Demand (flow) entering the first cell of the
                segment (vehicles per time unit).
            input_queue: Integer queue length at the segment input (vehicles).
            onramp_demand: 1-D array with onramp demands for each cell
                (vehicles per time unit), shape `(num_cells,)`.
            onramp_queue: 1-D integer array with current onramp queue lengths
                for each cell (vehicles), shape `(num_cells,)`.
            previous_onramp_flow: 1-D array with the previous time-step onramp
                flows for each cell (vehicles per time unit), shape
                `(num_cells,)`.
            dt: Time step length (same time units as flows).
            controller: Optional `AlineaController` used to compute regulated
                onramp flows. If `None`, onramps (if present) are left
                unregulated and rely on capacity capping only.

        Returns:
        - Tuple with the following values:
            - flow: ndarray, mainline outflow from each cell (vehicles per
              time unit), shape `(num_cells,)`.
            - density: ndarray, densities after advancing one time step
              (vehicles per length per lane), shape `(num_cells,)`.
            - speed: ndarray, simple flow-derived speed estimate for each
              cell (length per time unit), shape `(num_cells,)`.
            - input_flow: float, flow that actually entered the first cell
              during this step (vehicles per time unit).
            - next_input_queue: float, updated queue length at the segment
              input (vehicles) after this time step.
            - onramp_flow: ndarray, onramp flows applied to each cell
              (vehicles per time unit), shape `(num_cells,)`.
            - offramp_flow: ndarray, offramp flows applied to each cell
              (vehicles per time unit), shape `(num_cells,)`.
            - next_onramp_queue: ndarray, updated onramp queue lengths for
              each cell (vehicles), shape `(num_cells,)`.

        Notes:
            - Units must be consistent across `density`, `flow` and `dt`.
            - The function enforces physical capacity constraints by
              capping combined mainline/onramp flows where required using
              the second available degree of freedom in the CTM model
              that is not used by the update questions directly. The
              optional `controller` can be used to regulate the onramp flows.
        """

        # initialize model quantities for current iteration
        num_cells = len(network.cells)
        next_flow = np.zeros(num_cells)
        next_speed = np.zeros(num_cells)
        next_density = np.zeros(num_cells)
        next_onramp_flow = np.zeros(num_cells)
        next_onramp_queue = np.zeros(num_cells)
        next_offramp_flow = np.zeros(num_cells)

        # compute the input flow and queue simulating congestion at
        # the beginning of the currently considered highway segment
        next_input_flow, next_input_queue = calculate_segment_input_flow(
            first_cell=network.cells[0],
            backward_wave_speed=self.backward_wave_speed(cell=network.cells[0]),
            density=density[0],
            input_demand=mainline_demand,
            input_queue=input_queue,
            dt=dt,
        )

        # if a ramp controller is defined and an onramp is present in the
        # first cell, compute the regulated onramp flow
        if network.cells[0].onramp is not None:
            next_onramp_flow[0] = calculate_regulated_onramp_flow(
                cell=network.cells[0],
                cell_ix=0,
                backward_wave_speed=self.backward_wave_speed(cell=network.cells[0]),
                density=density,
                previous_onramp_flow=onramp_flow[0],
                onramp_demand=onramp_demand[0],
                onramp_queue=onramp_queue[0],
                controller=network.cells[0].onramp.controller,
                dt=dt,
            )

            # to ensure that the flows in the CTM model remains physically feasible
            # in cells with an onramp, we need to cap them at the maximum capacity
            # -> in case of violations, both flows are reduced proportionally
            next_input_flow, next_onramp_flow[0] = cap_cell_flows(
                cell=network.cells[0],
                backward_wave_speed=self.backward_wave_speed(cell=network.cells[0]),
                density=density[0],
                current_flow=next_input_flow,
                onramp_flow=next_onramp_flow[0],
            )

            # update the queue length on the onramp
            next_onramp_queue[0] = update_queue(
                queue_length=onramp_queue[0],
                demand=onramp_demand[0],
                flow=next_onramp_flow[0],
                dt=dt,
            )

            # update the queue length at the input of the segment
            next_input_queue = update_queue(
                queue_length=input_queue,
                demand=mainline_demand,
                flow=next_input_flow,
                dt=dt,
            )

        # iterate over all intermediate cells and update the onramp and
        # cell flows according to the physically possible values
        for i in range(num_cells):
            # compute the cell flow based on the downstream density
            if i == num_cells - 1:
                next_flow[i] = calculate_cell_flow(
                    cell=network.cells[i],
                    backward_wave_speed=self.backward_wave_speed(cell=network.cells[i]),
                    density=density[i],
                    downstream_density=self.critical_density(cell=network.cells[i]),
                )
            else:
                next_flow[i] = calculate_cell_flow(
                    cell=network.cells[i],
                    backward_wave_speed=self.backward_wave_speed(cell=network.cells[i]),
                    density=density[i],
                    downstream_density=density[i + 1],
                )

            # LOOKAHEAD: verify that the computed flow can be sustained
            # to ensure that the flows in the CTM model remains physically feasible
            # in cells with an onramp, we need to cap them at the maximum capacity
            # -> in case of violations, both flows are reduced proportionally
            if i < num_cells - 1 and network.cells[i + 1].onramp is not None:
                next_onramp_flow[i + 1] = calculate_regulated_onramp_flow(
                    cell=network.cells[i + 1],
                    cell_ix=i + 1,
                    backward_wave_speed=self.backward_wave_speed(
                        cell=network.cells[i + 1]
                    ),
                    density=density,
                    previous_onramp_flow=onramp_flow[i + 1],
                    onramp_demand=onramp_demand[i + 1],
                    onramp_queue=onramp_queue[i + 1],
                    controller=network.cells[
                        i + 1
                    ].onramp.controller,  # pyright: ignore[reportOptionalMemberAccess]
                    dt=dt,
                )

                next_flow[i], next_onramp_flow[i + 1] = cap_cell_flows(
                    cell=network.cells[i + 1],
                    backward_wave_speed=self.backward_wave_speed(
                        cell=network.cells[i + 1]
                    ),
                    density=density[i + 1],
                    current_flow=next_flow[i],
                    onramp_flow=next_onramp_flow[i + 1],
                )

                # update the queue length on the onramp
                next_onramp_queue[i + 1] = update_queue(
                    queue_length=onramp_queue[i + 1],
                    demand=onramp_demand[i + 1],
                    flow=next_onramp_flow[i + 1],
                    dt=dt,
                )

            # if an offramp is present in the current cell, split the flow accordingly
            current_cell = network.cells[i]
            if current_cell.offramp is not None:
                next_offramp_flow[i] = current_cell.offramp.split_ratio * next_flow[i]
                next_flow[i] = (1 - current_cell.offramp.split_ratio) * next_flow[i]

            # compute the flow, density and speed updates for the current cell
            if i > 0:
                next_density[i], next_speed[i] = self.cell_update(
                    cell_lanes=current_cell.lanes,
                    cell_length=current_cell.length,
                    density=density[i],
                    upstream_flow=next_flow[i - 1],
                    cell_flow=next_flow[i],
                    onramp_flow=onramp_flow[i],
                    offramp_flow=next_offramp_flow[i],
                    dt=dt,
                )
            else:
                next_density[i], next_speed[i] = self.cell_update(
                    cell_lanes=current_cell.lanes,
                    cell_length=current_cell.length,
                    density=density[i],
                    upstream_flow=next_input_flow,
                    cell_flow=next_flow[i],
                    onramp_flow=onramp_flow[i],
                    offramp_flow=next_offramp_flow[i],
                    dt=dt,
                )

        # return all updated quantities for the network
        return (
            next_flow,
            next_density,
            next_speed,
            next_input_flow,
            next_input_queue,
            next_onramp_flow,
            next_offramp_flow,
            next_onramp_queue,
        )
