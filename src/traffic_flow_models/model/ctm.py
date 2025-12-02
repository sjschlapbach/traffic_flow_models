from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from traffic_flow_models.network.network import Network, Cell
from .helpers import (
    calculate_segment_input_flow,
    calculate_regulated_onramp_flow,
    update_queue,
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
        critical_density: Compute the critical density for a cell.
        backward_wave_speed: Compute the backward wave speed for a cell.
        raw_updated_cell_flow: Compute the desired flow for a cell based on demand and supply.
        cap_cell_flows: Cap the combined mainline and onramp flows to not exceed cell capacity.
        step: Perform a single time step update of the CTM model over the entire network.
    """

    def __init__(self) -> None:
        """Create a CTM model instance."""
        return

    def critical_density(self, cell: Cell) -> float:
        """Return the critical density for a given cell.

        Args:
            cell: The `Cell` instance for which to compute the critical density.

        Returns:
            The critical density (vehicles per kilometer per lane).
        """
        return cell.Qc_lane / cell.vf

    def backward_wave_speed(self, cell: Cell) -> float:
        """Return the backward wave speed for a given cell.

        Args:
            cell: The `Cell` instance for which to compute the backward wave speed.

        Returns:
            The backward wave speed computed based on the critical density.
        """
        rho_cr = self.critical_density(cell=cell)
        return cell.Qc / (cell.rho_jam - rho_cr)

    def raw_updated_cell_flow(
        self,
        cell: Cell,
        backward_wave_speed: float,
        density: float,
        downstream_density: float,
        downstream_jam_density: float,
    ) -> float:
        """
        Compute the flow for a cell based on its demand for flow and the downstream supply.

        Args:
            cell: The Cell instance for which to compute the flow.
            backward_wave_speed: The backward wave speed of the cell.
            density: The density in the cell (vehicles per length unit).
            downstream_density: The density in the downstream cell (vehicles per length unit).
            downstream_jam_density: The jam density in the downstream cell (vehicles per length unit).

        Returns:
            The computed cell flow (vehicles per time unit). This does not take into account
            potential capacity constraints due to combined onramp and mainline flows.
        """
        q_demand = cell.vf * density * cell.lanes
        q_supply = backward_wave_speed * (downstream_jam_density - downstream_density)

        if cell.offramp is not None:
            q_supply *= 1 + cell.offramp.split_ratio

        return min(cell.Qc, q_demand, q_supply)

    def cap_cell_flows(
        self,
        cell: Cell,
        backward_wave_speed: float,
        density: float,
        current_flow: float,
        onramp_flow: float,
    ) -> Tuple[np.float64, np.float64]:
        """
        Verify that the cell desired cell flow and the onramp flow combined do
        not exceed the capacity of the cell (for CTM). If they do, scale them
        down proportionally.

        Args:
            cell: The Cell instance being evaluated.
            backward_wave_speed: The backward wave speed of the cell.
            density: Current density in the cell (vehicles per length unit).
            current_flow: Desired flow in the cell (vehicles per time unit).
            onramp_flow: Desired onramp flow into the cell (vehicles per time unit).

        Returns:
            A tuple (capped_cell_flow, capped_onramp_flow) where both flows are
            scaled down if necessary to not exceed the cell capacity.
        """

        supply_threshold = min(backward_wave_speed * (cell.rho_jam - density), cell.Qc)
        total_flow = current_flow + onramp_flow

        if total_flow > supply_threshold:
            scaling_factor = supply_threshold / total_flow
            reduced_mainline_flow = current_flow * scaling_factor
            reduced_onramp_flow = onramp_flow * scaling_factor
            return np.float64(reduced_mainline_flow), np.float64(reduced_onramp_flow)

        return np.float64(current_flow), np.float64(onramp_flow)

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
        input_flow: float,
        onramp_demand: NDArray[np.float64],
        onramp_queue: NDArray[np.float64],
        onramp_flow: NDArray[np.float64],
        offramp_flow: NDArray[np.float64],
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
            input_queue: Float queue length at the segment input (vehicles).
            input_flow: Flow that entered the first cell during the previous
                time step (vehicles per time unit).
            onramp_demand: 1-D array with onramp demands for each cell
                (vehicles per time unit), shape `(num_cells,)`.
            onramp_queue: 1-D integer array with current onramp queue lengths
                for each cell (vehicles), shape `(num_cells,)`.
            onramp_flow: 1-D array with the previous time-step onramp flows for
                each cell (vehicles per time unit), shape `(num_cells,)`.
            offramp_flow: 1-D array with the previous time-step offramp flows for
                each cell (vehicles per time unit), shape `(num_cells,)`.
            previous_onramp_flow: 1-D array with the previous time-step onramp
                flows for each cell (vehicles per time unit), shape
                `(num_cells,)`.
            dt: Time step length (same time units as flows).

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
        next_input_queue = 0.0
        next_input_flow = 0.0
        next_onramp_flow = np.zeros(num_cells)
        next_onramp_queue = np.zeros(num_cells)
        next_offramp_flow = np.zeros(num_cells)

        # update the densities of all cells based on the previous flows
        for i in range(num_cells):
            cell = network.cells[i]
            next_density[i] = density[i] + dt * (
                (flow[i - 1] if i > 0 else input_flow)
                + onramp_flow[i]
                - flow[i]
                - offramp_flow[i]
            ) / (cell.length * cell.lanes)

        # FIRST CELL
        # check if the mainline demand (including queue dissipation demand)
        # and the potential onramp demand in the first cell (including possible
        # queue dissipation demand) can be satisfied given the current density state
        first_cell = network.cells[0]
        next_input_flow, next_input_queue = calculate_segment_input_flow(
            first_cell=first_cell,
            backward_wave_speed=self.backward_wave_speed(cell=first_cell),
            density=density[0],
            input_demand=mainline_demand,
            input_queue=input_queue,
            dt=dt,
        )
        next_onramp_flow[0] = (
            calculate_regulated_onramp_flow(
                cell=first_cell,
                cell_ix=0,
                backward_wave_speed=self.backward_wave_speed(cell=first_cell),
                density=density,
                previous_onramp_flow=onramp_flow[0],
                onramp_demand=onramp_demand[0],
                onramp_queue=onramp_queue[0],
                controller=first_cell.onramp.controller,
                dt=dt,
            )
            if first_cell.onramp is not None
            else 0.0
        )

        # check if the combine inflows into the first cell exceed its supply
        # -> if they do, proportionally scale them down to fit within the supply
        # (and add surplus vehicles to the respective virtual queues)
        next_input_flow, next_onramp_flow[0] = self.cap_cell_flows(
            cell=network.cells[0],
            backward_wave_speed=self.backward_wave_speed(cell=network.cells[0]),
            density=next_density[0],
            current_flow=next_input_flow,
            onramp_flow=next_onramp_flow[0],
        )

        # update the mainline input queue and first cell onramp queue based on the flows
        next_input_queue = update_queue(
            queue_length=input_queue,
            demand=mainline_demand,
            flow=next_input_flow,
            dt=dt,
        )
        next_onramp_queue[0] = update_queue(
            queue_length=onramp_queue[0],
            demand=onramp_demand[0],
            flow=next_onramp_flow[0],
            dt=dt,
        )

        # CELL UPDATES (flows, densities, speeds, on- and offramp flows; entire network)
        for i in range(num_cells):
            # update the desired cell outflow
            # (assume critical density / free flow downstream for the last cell)
            if i == num_cells - 1:
                next_flow[i] = self.raw_updated_cell_flow(
                    cell=network.cells[i],
                    backward_wave_speed=self.backward_wave_speed(cell=network.cells[i]),
                    density=next_density[i],
                    downstream_density=self.critical_density(
                        cell=network.cells[i]
                    ),  # assume critical density downstream
                    downstream_jam_density=network.cells[i].rho_jam,
                )
            else:
                next_flow[i] = self.raw_updated_cell_flow(
                    cell=network.cells[i],
                    backward_wave_speed=self.backward_wave_speed(
                        cell=network.cells[i + 1]
                    ),
                    density=next_density[i],
                    downstream_density=next_density[i + 1],
                    downstream_jam_density=network.cells[i + 1].rho_jam,
                )

            # if downstream cell has an onramp, verify supply limits are respected
            # -> in case of violation -> reduce the cell outflow and next cell ramp inflow
            # -> also update the onramp queue accordingly
            if i < num_cells - 1 and network.cells[i + 1].onramp is not None:
                downstream_cell = network.cells[i + 1]
                next_onramp_flow[i + 1] = (
                    calculate_regulated_onramp_flow(
                        cell=downstream_cell,
                        cell_ix=i,
                        backward_wave_speed=self.backward_wave_speed(
                            cell=downstream_cell
                        ),
                        density=density,
                        previous_onramp_flow=onramp_flow[i + 1],
                        onramp_demand=onramp_demand[i + 1],
                        onramp_queue=onramp_queue[i + 1],
                        controller=downstream_cell.onramp.controller,
                        dt=dt,
                    )
                    if downstream_cell.onramp is not None
                    else 0.0
                )

                next_flow[i], next_onramp_flow[i + 1] = self.cap_cell_flows(
                    cell=network.cells[i + 1],
                    backward_wave_speed=self.backward_wave_speed(
                        cell=network.cells[i + 1]
                    ),
                    density=next_density[i + 1],
                    current_flow=next_flow[i],
                    onramp_flow=next_onramp_flow[i + 1],
                )

                # update the onramp flow with the difference between the demand for flow
                # and the actual flow supported by the onramp
                next_onramp_queue[i + 1] = update_queue(
                    queue_length=onramp_queue[i + 1],
                    demand=onramp_demand[i + 1],
                    flow=next_onramp_flow[i + 1],
                    dt=dt,
                )

            # update the speed for the current cell based on density and flow
            next_speed[i] = (
                (next_flow[i] + next_offramp_flow[i])
                / (network.cells[i].lanes * next_density[i])
                if next_density[i] > 0
                else network.cells[i].vf
            )

            # in case the current cell has an offramp, update the offramp flow accordingly
            if network.cells[i].offramp is not None:
                split = network.cells[i].offramp.split_ratio  # type: ignore
                next_offramp_flow[i] = split / (1 - split) * next_flow[i]

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
