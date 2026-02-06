from typing import Optional


class Cell:
    """Represents a single highway mainline cell.

    This class stores the minimal physical and topological attributes owned
    by a cell. Higher-level link properties (e.g. lane capacity, free-flow
    speed, jam density) are stored on the parent link objects (e.g.
    `MotorwayLink`) or are provided by the model when needed. The
    `Network` container is responsible for assembling cells into links and
    connecting onramps/offramps.

    Attributes:
        length (float): Cell length in kilometers (must be positive).
        upcoming_lane_drop (int): Number of lanes dropping downstream of
            this cell (0 if no lane drop).
        upstream (Optional[Cell]): Reference to the upstream cell in the
            linked list (set by the network builder).
        downstream (Optional[Cell]): Reference to the downstream cell in
            the linked list (set by the network builder).
    """

    # type annotations for static tools: instances will have these attributes
    upstream: Optional["Cell"]
    downstream: Optional["Cell"]

    def __init__(
        self,
        length: float,
        upcoming_lane_drop: int = 0,
    ) -> None:
        """Create a new Cell with physical parameters.

        Please note that the critical density and backward wave speed are
        not stored in the cell explicitly, since their values depend on the
        fundamental diagram shape used in the model. They are computed through
        member functions of the corresponding model classes.

        Args:
            length: Cell length in kilometers.
            upcoming_lane_drop: Number of lanes dropping downstream of this cell.

        Raises:
            ValueError: If any of the physical parameters are non-positive.
        """

        if length <= 0:
            raise ValueError("Cell length must be positive.")

        self.length: float = length  # in kilometers

        # bidirectional linked list pointers for network topology
        self.upstream: Optional["Cell"] = None
        self.downstream: Optional["Cell"] = None

        # store if there is a lane drop coming up between this and the next downstream cell
        # if set, the number of dropped lanes is stored
        self.upcoming_lane_drop: int = upcoming_lane_drop
