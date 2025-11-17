import numpy as np
from numpy.typing import NDArray


class AlineaController:
    def __init__(self, gain: float, setpoint: float, measurement_cell: int) -> None:
        """Create an ALINEA controller instance.

        Args:
            gain: ALINEA controller gain parameter (typically between 70 and
                150).
            setpoint: Desired downstream density setpoint (vehicles per length
                per lane).
        """

        if setpoint < 0.0:
            raise ValueError("Setpoint density must be non-negative.")

        if measurement_cell < 0:
            raise ValueError("Measurement cell index must be non-negative.")

        self.gain: float = gain
        self.setpoint: float = setpoint
        self.measurement_cell: int = measurement_cell

    def compute_regulated_flow(
        self, measured_densities: NDArray[np.float64], previous_flow: float
    ) -> float:
        """Compute the regulated onramp flow using the ALINEA feedback law.

        Args:
            measured_density: Measured density downstream of the onramp
                (vehicles per length per lane).
            previous_flow: Previous onramp flow (vehicles per time unit).

        Returns:
            The regulated onramp flow (vehicles per time unit).
        """
        flow_adjustment = self.gain * (
            self.setpoint - measured_densities[self.measurement_cell]
        )
        regulated_flow = previous_flow + flow_adjustment
        return max(regulated_flow, 0.0)  # ensure non-negative flow
