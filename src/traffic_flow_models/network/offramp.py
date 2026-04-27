import uuid

from traffic_flow_models.network.destination import Destination


class Offramp:
    """A simple container for off-ramp physical parameters.

    Attributes:
        lanes: Number of lanes on the offramp.
        destination: Optional linked `Destination` instance.
    """

    def __init__(
        self,
        lanes: int,
        id: str | None = None,
        origin_node_id: str | None = None,
        destination_node_id: str | None = None,
    ) -> None:
        """Initialize the Offramp parameters.

        Args:
            lanes: Number of lanes on the offramp.
            id: Optional identifier for the offramp link. If not provided, 
                a unique ID is generated automatically.
            origin_node_id: Optional identifier for the origin node
                to which this offramp is connected.
            destination_node_id: Optional identifier for the destination node
                to which this offramp is connected.
        """

        if lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        self.id: str = (
            id if id is not None else str(uuid.uuid4())
        )  # identifier for the offramp link
        self.lanes: int = lanes  # number of lanes

        self.origin_node_id: str | None = origin_node_id
        self.destination_node_id: str | None = destination_node_id
