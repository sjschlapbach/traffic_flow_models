from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from traffic_flow_models.network.network import Network, Cell
from .helpers import (
    calculate_segment_input_flow,
    calculate_regulated_onramp_flow,
    update_queue,
)


class METANET:
    def __init__(self, tau, nu, kappa, delta, phi, alpha):
        """Create a METANET model instance with given parameters.

        The METANET model is a second-order macroscopic traffic model that
        includes dynamics for both density and speed. This constructor stores
        the model parameters used in the speed update and flow computations.

        Args:
            tau: Relaxation time scale for speed dynamics (time units).
            nu: Anticipation coefficient controlling sensitivity to downstream
                density gradients.
            kappa: Small positive constant added to densities to avoid
                division-by-zero in speed/flow formulas.
            delta: Weighting coefficient for onramp influence on speed.
            phi: Coefficient for additional deceleration due to upcoming
                 lane drops.
            alpha: Shape parameter used in the stationary velocity function.
        """

        self.tau = tau
        self.nu = nu
        self.kappa = kappa
        self.delta = delta
        self.phi = phi
        self.alpha = alpha

    def critical_density(self, cell: Cell) -> float:
        """Return the critical density for a given cell.

        The METANET implementation uses an adjusted definition of the
        critical density that depends on the model parameter ``alpha`` and the
        cell's free-flow speed. The returned value has units of vehicles per
        length per lane.

        Args:
            cell: The `Cell` instance for which to compute the critical
                density.

        Returns:
            The critical density (vehicles per length per lane).
        """

        return cell.Qc_lane / (cell.vf * np.exp(-1 / self.alpha))

    def backward_wave_speed(self, cell: Cell) -> float:
        """Return the backward (congestion) wave speed for a given cell.

        The backward wave speed is computed from the cell capacity and the
        difference between jam density and critical density. This value is
        typically used to compute how congestion propagates upstream.

        Args:
            cell: The `Cell` instance for which to compute the backward wave
                speed.

        Returns:
            The backward wave speed (length per time).
        """

        return cell.Qc / (cell.rho_jam - self.critical_density(cell=cell))

    def stationary_velocity(self, cell: Cell, density: float) -> float:
        """Compute the stationary (equilibrium) velocity for a cell.

        The stationary velocity is the speed that the traffic on the cell would
        adopt in the absence of dynamics, given the current density. METANET
        uses an exponential functional form parameterized by ``alpha`` and the
        cell's free-flow speed (fundamental diagram).

        Args:
            cell: The `Cell` instance providing the free-flow speed and
                critical density information.
            density: The density at which to evaluate the stationary
                velocity (vehicles per length per lane).

        Returns:
            The stationary velocity (length per time unit).
        """

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
        """Compute density, speed and flow updates for a single METANET cell.

        Implements the METANET discrete-time update for a homogeneous cell.
        The method updates the density using conservation of vehicles and
        updates the speed using the METANET second-order dynamics which
        include relaxation towards a stationary velocity, convection from
        upstream speed differences, anticipation of downstream density
        gradients, and onramp-induced effects. An extra deceleration term is
        added when a lane drop is present in the subsequent downstream cell.

        Note: Off-ramps in this implementation split the flow inside the cell
        into a mainline outflow component (q) and an offramp flow component.
        The computation of an offramp flow through the split ratio always needs
        to be based on the total cell flow (not only the mainline cell outflow)

        Args:
            cell: The `Cell` instance describing geometry and lane-drop info.
            upstream_flow: Flow entering the cell from upstream (vehicles per
                time unit).
            previous_flow: Flow leaving the cell at the previous time step
                (vehicles per time unit).
            onramp_flow: Flow entering from an onramp attached to the cell
                (vehicles per time unit).
            offramp_flow: Flow leaving via an offramp attached to the cell
                (vehicles per time unit).
            previous_density: Density at the previous time step
                (vehicles per length per lane).
            downstream_density: Density in the downstream cell used for
                anticipation terms (vehicles per length per lane).
            upstream_speed: Speed in the upstream cell used for convective
                coupling (length per time).
            previous_speed: Speed at the previous time step in this cell
                (length per time).
            dt: Time step length (same time units as flows).

        Returns:
            A tuple ``(density, speed, flow)`` where:
            - density: Updated density after one time step (vehicles per
              length per lane).
            - speed: Updated speed after one time step (length per time).
            - flow: Updated flow leaving the cell (vehicles per time).
        """

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
        input_queue: float,
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
        """Advance the METANET model by a single time step for the whole network.

        This method computes the mainline cell flows, onramp flows and the
        resulting next-step densities and speeds for every cell in the provided
        `network` using the METANET discretization implemented in
        :meth:`cell_update`.

        Args:
            network: The `Network` object describing the cells, their
                fundamental diagram parameters and ramp connectivity.
            density: 1-D array of current densities for each cell
                (vehicles per length per lane), shape `(num_cells,)`.
            speed: 1-D array of current speeds for each cell (length per time),
                shape `(num_cells,)`.
            flow: 1-D array of current outflows for each cell (vehicles per
                time), shape `(num_cells,)`.
            mainline_demand: Demand (flow) entering the first cell of the
                segment (vehicles per time unit).
            input_queue: Float queue length at the segment input (vehicles).
            input_flow: Flow that entered the first cell during the previous
            onramp_demand: 1-D array with onramp demands for each cell
                (vehicles per time unit), shape `(num_cells,)`.
            onramp_queue: 1-D integer array with current onramp queue lengths
                for each cell (vehicles), shape `(num_cells,)`.
            onramp_flow: 1-D array with the previous time-step onramp flows for
                each cell (vehicles per time unit), shape `(num_cells,)`.
            offramp_flow: 1-D array with the previous time-step offramp flows for
                each cell (vehicles per time unit), shape `(num_cells,)`.
            dt: Time step length (same time units as flows).

        Returns:
            A tuple with the following values:
            - flow: ndarray, mainline outflow from each cell (vehicles per
              time unit), shape `(num_cells,)`.
            - density: ndarray, densities after advancing one time step
              (vehicles per length per lane), shape `(num_cells,)`.
            - speed: ndarray, speeds after advancing one time step (length
              per time unit), shape `(num_cells,)`.
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
            - The function enforces physical capacity constraints by relying
              on regulated onramp flows (via controllers) and using
              look-ahead checks for downstream onramps.
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
        first_cell = network.cells[0]
        next_input_flow, next_input_queue = calculate_segment_input_flow(
            first_cell=first_cell,
            backward_wave_speed=self.backward_wave_speed(cell=first_cell),
            density=density[0],
            input_demand=mainline_demand,
            input_queue=input_queue,
            dt=dt,
        )

        # if a ramp controller is defined and an onramp is present in the
        # first cell, compute the regulated onramp flow
        if first_cell.onramp is not None:
            next_onramp_flow[0] = calculate_regulated_onramp_flow(
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
            # if an offramp is present in the current cell, determine the
            # offramp flow based on the total cell outflow at the previous
            # time step and the split ratio. Note that this does not correspond
            # to the previous offramp flow, since the mainline cell outflow is
            # updated in between through the cell update function
            current_cell = network.cells[i]

            # in the first cell, we assume v_0 = v_1 to eliminate the
            # corresponding term from the speed update equation
            # in the last cell, we assume rho_{n+1} = rho_n
            next_density[i], next_speed[i], next_flow[i] = self.cell_update(
                cell=current_cell,
                upstream_flow=input_flow if i == 0 else flow[i - 1],
                previous_flow=flow[i],
                onramp_flow=onramp_flow[i],
                offramp_flow=offramp_flow[i],
                previous_density=density[i],
                downstream_density=density[i + 1] if i < num_cells - 1 else density[i],
                upstream_speed=speed[i] if i == 0 else speed[i - 1],
                previous_speed=speed[i],
                dt=dt,
            )

            # in case the current cell has an offramp, update the offramp flow accordingly
            if network.cells[i].offramp is not None:
                split = network.cells[i].offramp.split_ratio  # type: ignore
                next_offramp_flow[i] = split / (1 - split) * next_flow[i]

            # LOOKAHEAD: if the next downstream cell has an onramp, compute the regulated
            # onramp flow and update the corresponding queue (if not the entire flow can be served)
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
