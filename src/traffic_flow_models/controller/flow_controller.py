import casadi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traffic_flow_models.network.onramp import Onramp


class FlowController:
    def __init__(self, onramp: "Onramp", flow: float) -> None:
        """Create a fixed-flow ramp metering controller instance.

        Args:
            onramp: Onramp object to which the controller is attached.
            flow: value of the ramp metering flow that should be applied as the limit
        """
        if flow < 0.0:
            raise ValueError("Flow must be non-negative.")

        self.onramp = onramp
        self.flow: casadi.SX = casadi.SX(flow)

    def compute_regulated_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        dt: float,
    ) -> casadi.SX:
        """Return the fixed regulated onramp flow.

        Args:
            onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).
            dt: Simulation time step size (placeholder for other controllers).

        Returns:
            The regulated onramp flow (vehicles per time unit).
        """
        return self.flow
