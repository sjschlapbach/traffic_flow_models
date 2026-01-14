import uuid


class Origin:
    """A simple container for origin link id.

    Since the origin is a virtual link without physical attributes / flow
    restrictions / etc., this class only contains an identifier. Demand values
    for the origin link are provided during simulation.

    Attributes:
        id: Identifier for the origin link (for demand assignment).
        destination_node_id: Identifier of the `Node` to which this origin is connected.
    """

    def __init__(
        self,
        id: str | None = None,
        destination_node_id: str | None = None,
    ) -> None:
        """Initialize the Origin parameters.

        Args:
            id: Identifier for the origin link (for demand assignment).
            destination_node_id: Identifier of the `Node` to which this origin is connected.
        """

        self.id: str = (
            id if id is not None else str(uuid.uuid4())
        )  # identifier for the origin link
        self.destination_node_id: str | None = destination_node_id
