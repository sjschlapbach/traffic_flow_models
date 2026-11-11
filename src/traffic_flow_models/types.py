"""Shared type aliases used across the traffic_flow_models package."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import casadi
import numpy as np
from numpy.typing import NDArray

# the `type` aliases are evaluated lazily, so these imports are only needed for type checking
if TYPE_CHECKING:
    from traffic_flow_models.controller import (
        AlineaController,
        CustomController,
        FlowController,
        MetalineController,
    )
    from traffic_flow_models.network import (
        Destination,
        MotorwayLink,
        Offramp,
        Onramp,
        Origin,
    )

# any link that can be attached to a node
type Link = MotorwayLink | Onramp | Offramp | Origin | Destination

# any on-ramp metering controller
type RampController = (
    FlowController | AlineaController | MetalineController | CustomController
)

# packed state or disturbance vector: numeric in simulation, symbolic in model graphs
type PackedVector = NDArray[np.float64] | casadi.SX

# simulation time -> value, e.g. an origin demand [veh/h] or a boundary condition
type TimeProfile = Callable[[float], float]

# simulation time -> split rate per outgoing link id of a node
type TurningRateProfile = Callable[[float], dict[str, float]]

# time series per link/origin/onramp id (per-cell links: shape cells x timesteps)
type TimeSeriesById = dict[str, NDArray[np.float64]]

# parsed JSON of a results file, whose content is only known at runtime
type JSONDict = dict[str, Any]


class ModelOptions(TypedDict, total=False):
    """Model-specific calibration options (METANET: ``link_specific_alpha``)."""

    link_specific_alpha: bool


class ParameterSearchRun(TypedDict):
    """Outcome of one multi-start run of the calibration parameter search."""

    config_str: str
    initial_vec: NDArray[np.float64]
    cost: float
    success: bool
    nfev: int
    optimality: float
    final_params: NDArray[np.float64] | None
