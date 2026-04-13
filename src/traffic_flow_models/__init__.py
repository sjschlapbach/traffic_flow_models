"""Top-level package for traffic_flow_models.

Expose commonly used symbols at package level so tests can do
`from traffic_flow_models import Cell`.
"""

# re-export network components
from .network.cell import Cell
from .network.origin import Origin
from .network.destination import Destination
from .network.onramp import Onramp
from .network.offramp import Offramp
from .network.motorway_link import MotorwayLink
from .network.node import Node
from .network.network import Network
from .network.simulation import Simulation
from .network.calibrator import Calibrator

# re-export model components
from .model.ctm import CTM
from .model.metanet import METANET, METANETParams, METANETSymbolicParams

# re-export controller components
from .controller.flow_controller import FlowController
from .controller.alinea import AlineaController
from .controller.metaline import MetalineController
from .controller.custom_controller import CustomController

# re-export simulation and pipeline components
from .simulator.sumo_simulation import SUMOSimulation
from .simulator.sumo_pipeline import SUMOPipeline

# re-export arbitrator components
from .arbitrator import (
    NetworkArbitrator,
    LoopDetectorGenerator,
    DemandAggregator,
    TurningRateAggregator,
    BackboneStateAggregator,
)

__all__ = [
    "Cell",
    "Origin",
    "Destination",
    "Onramp",
    "Offramp",
    "MotorwayLink",
    "Node",
    "Network",
    "Simulation",
    "Calibrator",
    "CTM",
    "METANET",
    "METANETParams",
    "METANETSymbolicParams",
    "FlowController",
    "CustomController",
    "AlineaController",
    "MetalineController",
    "SUMOSimulation",
    "SUMOPipeline",
    "NetworkArbitrator",
    "LoopDetectorGenerator",
    "DemandAggregator",
    "TurningRateAggregator",
    "BackboneStateAggregator",
]
