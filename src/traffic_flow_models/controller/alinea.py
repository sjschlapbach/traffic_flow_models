import casadi
from typing import TYPE_CHECKING

from traffic_flow_models.controller.queue_control import apply_queue_control

if TYPE_CHECKING:
    from traffic_flow_models.network.onramp import Onramp


class AlineaController:
    def __init__(
        self,
        onramp: "Onramp",
        measurement_link_id: str,
        measurement_cell_idx: int,
        gain: float,
        density_setpoint: float,
        control_dt: float | None = None,
    ) -> None:
        """Create an ALINEA controller instance.

        Args:
            onramp: Onramp object to which the controller is attached.
            measurement_link_id: ID of the link where the density measurement is taken for feedback
            measurement_cell_idx: Index of the cell on the measurement link where the density is measured
            gain: ALINEA controller gain parameter (positive scalar).
            density_setpoint: Desired downstream density setpoint (vehicles per length per lane).
            control_dt: Optional control interval (same time unit as the simulation `dt`, an
                integer multiple of it); the metering rate is held between updates. None
                (default) updates it at every simulation step.
        """

        if density_setpoint < 0.0:
            raise ValueError("Setpoint density must be non-negative.")

        if gain <= 0.0:
            raise ValueError("Gain must be positive.")

        if measurement_cell_idx < 0:
            raise ValueError("Measurement cell index must be non-negative.")

        self.onramp = onramp
        self.measurement_link_id: str = measurement_link_id
        self.measurement_cell: int = measurement_cell_idx

        self.gain: float = gain
        self.density_setpoint: float = density_setpoint
        self.control_dt: float | None = control_dt

        # queue-control inputs, set by `set_queue_control` (see `Onramp.max_queue`)
        self._rate_update_dt: float | None = None
        self._origin_id: str | None = None

    def set_queue_control(self, rate_update_dt: float, origin_id: str) -> None:
        """Set the inputs of the queue-control law (called automatically by the library).

        The models call this for onramps with a `max_queue` when they build the step.

        Args:
            rate_update_dt: Update interval of the controller (`control_dt`, or the
                simulation `dt` if it is None).
            origin_id: ID of the origin feeding the onramp (its flow is the ramp arrival flow).
        """
        self._rate_update_dt = rate_update_dt
        self._origin_id = origin_id

    def compute_regulated_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
    ) -> casadi.SX:
        """Compute the regulated onramp flow using the ALINEA feedback law.

        Args:
            onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).

        Returns:
            The regulated onramp flow (vehicles per time unit).
        """
        # extract the relevant density and flow values for the feedback law
        measured_density = densities[self.measurement_link_id][self.measurement_cell]
        previous_flow = flows[self.onramp.id][
            0
        ]  # on-ramps only store a single flow value

        if measured_density is None or previous_flow is None:
            raise ValueError(
                f"Missing flow or density information for controller on onramp {self.onramp.id}"
            )

        flow_adjustment = self.gain * (self.density_setpoint - measured_density)
        regulated_flow = previous_flow + flow_adjustment
        return apply_queue_control(
            metering_rate=casadi.fmax(regulated_flow, casadi.SX(0.0)),  # non-negative
            onramp=self.onramp,
            onramp_queues=onramp_queues,
            flows=flows,
            rate_update_dt=self._rate_update_dt,
            origin_id=self._origin_id,
        )
