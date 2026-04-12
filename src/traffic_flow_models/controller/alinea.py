import casadi
from typing import TYPE_CHECKING

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
    ) -> None:
        """Create an ALINEA controller instance.

        Args:
            onramp: Onramp object to which the controller is attached.
            measurement_link_id: ID of the link where the density measurement is taken for feedback
            measurement_cell_idx: Index of the cell on the measurement link where the density is measured
            gain: ALINEA controller gain parameter (positive scalar).
            density_setpoint: Desired downstream density setpoint (vehicles per length per lane).
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

    # TODO: ALSO IMPLEMENT OVERRIDE CASE AT 0.9 * max queue length or something like this to prevent spillback if possible
    def compute_regulated_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        dt: float,
    ) -> casadi.SX:
        """Compute the regulated onramp flow using the ALINEA feedback law.

        Args:
            onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).
            dt: Simulation time step size (placeholder for other controllers).

        Returns:
            The regulated onramp flow (vehicles per time unit).
        """
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
        return casadi.fmax(regulated_flow, casadi.SX(0.0))  # ensure non-negative flow
