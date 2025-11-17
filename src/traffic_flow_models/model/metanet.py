from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from traffic_flow_models.network.network import Network, Cell
from .helpers import (
    calculate_segment_input_flow,
    calculate_regulated_onramp_flow,
    update_queue,
)


# TODO: add tests for the METANET model
# TODO: add docstrings to all methods
class METANET:
    def __init__(self, tau, nu, kappa, delta, phi, alpha):
        self.tau = tau
        self.nu = nu
        self.kappa = kappa
        self.delta = delta
        self.phi = phi
        self.alpha = alpha

    def critical_density(self, cell: Cell) -> float:
        return cell.Qc_lane / (cell.vf * np.exp(-1 / self.alpha))

    def backward_wave_speed(self, cell: Cell) -> float:
        return cell.Qc / (cell.rho_jam - self.critical_density(cell=cell))

    def stationary_velocity(self, cell: Cell, density: float) -> float:
        return cell.vf * np.exp(
            -1 / self.alpha * (density / self.critical_density(cell=cell)) ** self.alpha
        )

    def cell_update(
        self,
        cell,
        upstream_flow,
        previous_flow,
        onramp_flow,
        offramp_flow,
        previous_density,
        downstream_density,
        upstream_speed,
        previous_speed,
        dt,
    ) -> Tuple[float, float, float]:
        # compute the new density based on the flows at the previous timestep
        # Note: off-ramps are modeled as splitting the outflow and do not
        # directly reduce the density update term (matches MATLAB METANET).
        density = previous_density + dt * (
            upstream_flow + onramp_flow - offramp_flow - previous_flow
        ) / (cell.length * cell.lanes)

        # compute the new speed based on the previous timestep
        speed = (
            previous_speed
            + dt
            / self.tau
            * (self.stationary_velocity(cell, previous_density) - previous_speed)
            + dt / cell.length * previous_speed * (upstream_speed - previous_speed)
            - (dt * self.nu)
            / (self.tau * cell.length)
            * (downstream_density - previous_density)
            / (previous_density + self.kappa)
            - (dt * self.delta)
            / (cell.length * cell.lanes)
            * (onramp_flow * previous_speed)
            / (previous_density + self.kappa)
        )

        # if a lane drop is coming up, add an additional term to the speed
        # update equation to account for the additional deceleration
        if cell.upcoming_lane_drop > 0:
            speed -= (
                dt
                * self.phi
                * cell.upcoming_lane_drop
                * previous_density
                * previous_speed**2
            ) / (cell.length * cell.lanes * self.critical_density(cell))

        # ensure that the speed values remain non-negative
        speed = max(speed, 0)

        # compute the flow update of the cell based on the speed and density
        flow = density * speed * cell.lanes

        return density, speed, flow

    def step(
        self,
        network: Network,
        density: NDArray[np.float64],
        speed: NDArray[np.float64],
        flow: NDArray[np.float64],
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
            # if an offramp is present in the current cell, split the flow accordingly
            # offramps are assumed to be at the end of a cell, therefore only affecting
            # the outflow of the cell (and not affecting the density or speed updates)
            current_cell = network.cells[i]
            if current_cell.offramp is not None:
                next_offramp_flow[i] = current_cell.offramp.split_ratio * flow[i]
                flow[i] = (1 - current_cell.offramp.split_ratio) * flow[i]

            if i == 0:
                # in the first cell, we assume v_0 = v_1 to eliminate the
                # corresponding term from the speed update equation
                next_density[i], next_speed[i], next_flow[i] = self.cell_update(
                    cell=current_cell,
                    upstream_flow=next_input_flow,
                    previous_flow=flow[i],
                    onramp_flow=next_onramp_flow[i],
                    offramp_flow=next_offramp_flow[i],
                    previous_density=density[i],
                    downstream_density=density[i + 1],
                    upstream_speed=speed[i],
                    previous_speed=speed[i],
                    dt=dt,
                )

            elif i == num_cells - 1:
                # in the last cell, we assume rho_{n+1} = rho_n
                next_density[i], next_speed[i], next_flow[i] = self.cell_update(
                    cell=current_cell,
                    upstream_flow=flow[i - 1],
                    previous_flow=flow[i],
                    onramp_flow=next_onramp_flow[i],
                    offramp_flow=next_offramp_flow[i],
                    previous_density=density[i],
                    downstream_density=density[i],
                    upstream_speed=speed[i - 1],
                    previous_speed=speed[i],
                    dt=dt,
                )

            else:
                next_density[i], next_speed[i], next_flow[i] = self.cell_update(
                    cell=current_cell,
                    upstream_flow=flow[i - 1],
                    previous_flow=flow[i],
                    onramp_flow=next_onramp_flow[i],
                    offramp_flow=next_offramp_flow[i],
                    previous_density=density[i],
                    downstream_density=density[i + 1],
                    upstream_speed=speed[i - 1],
                    previous_speed=speed[i],
                    dt=dt,
                )

            # LOOKAHEAD: verify that the combined flow from the mainline and the onramp can
            # be accommodated by the downstream cell's supply and extend the virtual queue
            # on the onramp otherwise (if applicable)
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

                # update the queue length on the onramp
                next_onramp_queue[i + 1] = update_queue(
                    queue_length=onramp_queue[i + 1],
                    demand=onramp_demand[i + 1],
                    flow=next_onramp_flow[i + 1],
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
