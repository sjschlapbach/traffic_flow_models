import uuid


class Destination:
    """A simple container for destination physical parameters.

    Attributes:
        id: Identifier for the destination link (for potential
            downstream density assignment that may limit outflow)
    """

    def __init__(
        self, id: str | None = None, origin_node_id: str | None = None
    ) -> None:
        """Initialize the Destination parameters.

        Args:
            id: Identifier for the destination link (for potential
                downstream density assignment that may limit outflow)
            origin_node_id: Optional identifier for the origin node
                to which this destination is connected.
        """

        self.id: str = (
            id if id is not None else str(uuid.uuid4())
        )  # identifier for the destination link
        self.origin_node_id: str | None = origin_node_id
