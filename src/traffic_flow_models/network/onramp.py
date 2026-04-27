import uuid
from typing import Union, Literal

from traffic_flow_models.controller import (
    FlowController,
    AlineaController,
    MetalineController,
    CustomController,
)


class Onramp:
    """A simple container for on-ramp physical parameters.

    Attributes:
        id: Identifier for the origin link (for demand assignment).
        lanes: Number of lanes on the onramp.
        Qc_lane: Capacity per lane in vehicles per hour.
        Qc: Total onramp capacity in vehicles per hour.
        vf: Free-flow speed in km/h.
        rho_jam: Jam density in vehicles per km per lane.
    """

    def __init__(
        self,
        length: float,
        lanes: int,
        id: str | None = None,
        controller: (
            Union[
                FlowController, AlineaController, MetalineController, CustomController
            ]
            | None
        ) = None,
        origin_node_id: str | None = None,
        destination_node_id: str | None = None,
    ) -> None:
        """Initialize the Onramp parameters.

        Args:
            id: Identifier for the origin link (for demand assignment; optional).
            lanes: Number of lanes on the onramp.
            id: Optional identifier for the onramp link. If not provided,
                a unique ID is generated automatically.
            controller: Optional ramp metering controller.
            origin_node_id: Optional ID of the upstream node.
            destination_node_id: Optional ID of the downstream node.
        """

        if lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        if length <= 0:
            raise ValueError("Onramp length must be positive.")

        self.id: str = (
            id if id is not None else str(uuid.uuid4())
        )  # identifier for the origin link
        self.length: float = (
            length  # length of the onramp link (same units as motorway links)
        )
        self.lanes: int = lanes  # number of lanes

        self.controller = controller  # optional ramp metering controller
        self.control_status: Literal["unset", "hero_master", "hero_slave"] = (
            "unset"  # control status for use in coordinated ramp metering
        )

        # neighbor onramps for coordinated ramp metering (populated by Network helper)
        self.upstream_onramps: list[Onramp] = []
        self.downstream_onramps: list[Onramp] = []

        self.origin_node_id: str | None = origin_node_id
        self.destination_node_id: str | None = destination_node_id
