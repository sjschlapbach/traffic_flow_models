import casadi
import numpy as np
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from traffic_flow_models.network.onramp import Onramp


class MetalineController:
    def __init__(
        self,
        onramp: "Onramp",
        measurement_cells: list[tuple[str, int]],
        gain_matrix: dict[str, NDArray[np.float64]],
        density_setpoints: list[tuple[str, int, float]],
    ) -> None:
        """Create an Metaline controller instance.

        Args:
            onramp: The on-ramp for which to control flow.
            measurement_cells: A list of tuples (link_id, cell_idx) indicating the cells to measure.
            gain_matrix: A dictionary mapping link IDs to their corresponding gain matrices.
            density_setpoints: A list of tuples (link_id, cell_idx, setpoint) indicating the desired densities.
        """
        # verify that density setpoins are specified for each measurement cell
        for link_id, cell_idx in measurement_cells:
            if not any(
                link_id == setpoint[0] and cell_idx == setpoint[1]
                for setpoint in density_setpoints
            ):
                raise ValueError(
                    f"Density setpoint not specified for measurement cell ({link_id}, {cell_idx})."
                )

        # verify that the density setpoint is non-negative
        for setpoint in density_setpoints:
            if setpoint[2] < 0.0:
                raise ValueError(
                    f"Density setpoint for cell ({setpoint[0]}, {setpoint[1]}) must be non-negative."
                )

        # verify that one of the lines in the gain_matrix corresponds to on-ramp under consideration
        if onramp.id not in gain_matrix.keys():
            raise ValueError(
                f"Gain matrix must contain a line corresponding to the on-ramp link ID {onramp.id}."
            )

        # verify that the number of measurement cells is consistent with the gain matrix dimension
        if len(measurement_cells) != gain_matrix[onramp.id].shape[1]:
            raise ValueError(
                f"Number of measurement cells ({len(measurement_cells)}) must be consistent with the gain matrix dimension ({gain_matrix[onramp.id].shape[1]})."
            )

        # store the controller parameters
        self.onramp = onramp
        self.measurement_cells: list[tuple[str, int]] = measurement_cells
        self.gain_matrix: dict[str, NDArray[np.float64]] = gain_matrix
        self.density_setpoints: list[tuple[str, int, float]] = density_setpoints

    def compute_regulated_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
    ) -> casadi.SX:
        """
        Compute the regulated onramp flow using the Metaline feedback law.

        Args:
            onramp_queues: A dictionary mapping onramp IDs to their respective queue states.
            flows: A dictionary mapping link IDs to their respective flow states.
            densities: A dictionary mapping link IDs to their respective density states.

        Returns:
            The regulated onramp flow.
        """

        # get the relevant densities from the measurement cells
        density_errors: list[casadi.SX] = []
        for link_id, cell_idx in self.measurement_cells:
            current_density: casadi.SX = densities[link_id][cell_idx]
            density_setpoint: float = next(
                setpoint[2]
                for setpoint in self.density_setpoints
                if setpoint[0] == link_id and setpoint[1] == cell_idx
            )
            density_errors.append(current_density - density_setpoint)

        # get the previous flow of the considered onramp
        # on-ramps only store a single flow value
        previous_flow: casadi.SX = flows[self.onramp.id][0]

        # compute the regulated flow using the gain matrix and the density errors
        gain_matrix_onramp = self.gain_matrix[self.onramp.id]
        regulated_flow = previous_flow - casadi.mtimes(
            gain_matrix_onramp, casadi.vertcat(*density_errors)
        )

        return regulated_flow
