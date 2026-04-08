import casadi


class FlowController:
    def __init__(self, onramp_id: str, flow: float) -> None:
        """Create a fixed-flow ramp metering controller instance.

        Args:
            onramp_id: ID of the on-ramp to which the controller is attached
            flow: value of the ramp metering flow that should be applied as the limit
        """
        if flow < 0.0:
            raise ValueError("Flow must be non-negative.")

        self.onramp_id: str = onramp_id
        self.flow: casadi.SX = casadi.SX(flow)

    def compute_regulated_flow(
        self, flows: dict[str, casadi.SX], densities: dict[str, casadi.SX]
    ) -> casadi.SX:
        """Compute the regulated onramp flow using the ALINEA feedback law.

        Args:
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).

        Returns:
            The regulated onramp flow (vehicles per time unit).
        """
        return self.flow
