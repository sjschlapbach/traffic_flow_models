import uuid
from typing import Optional

from .cell import Cell


class MotorwayLink:
    """A motorway link composed of connected `Cell` instances.

    The MotorwayLink class stores mainline `Cell` instances as a bidirectional
    linked list arranged from upstream to downstream. It provides convenience
    methods to add cells and attach or detach on-/off-ramps. The MotorwayLink does
    not perform simulation — it only manages the topology and basic
    validation when linking objects together. Link-level physical
    parameters (e.g. lane capacity, free-flow speed, jam density) are
    stored on the link; per-cell geometry is stored on the contained
    `Cell` objects.

    Attributes:
        length (float): Total link length in kilometers.
        lanes (int): Number of lanes on the motorway link.
        lane_capacity (float): Capacity per lane in vehicles per time.
        vf (float): Free-flow speed for the link (length per time).
        rho_jam (float): Jam density (vehicles per length per lane).
        id (str): Unique identifier for the link.
        origin_node_id (Optional[str]): Optional ID of the upstream node.
        destination_node_id (Optional[str]): Optional ID of the downstream node.
        _head (Optional[Cell]): Reference to the first (upstream) cell.
        _tail (Optional[Cell]): Reference to the last (downstream) cell.
        _cell_count (int): Number of cells in the motorway.
    """

    def __init__(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        id: str | None = None,
        origin_node_id: str | None = None,
        destination_node_id: str | None = None,
    ) -> None:
        """
        Initialize an empty motorway link.

        The created motorway link contains no cells initially. Cells are created
        automatically during the partitioning of the network according to the
        model requirements, the CFL condition and the desired cell size specificed
        by the user.
        """

        if length <= 0:
            raise ValueError("Link length must be positive.")

        if lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        if lane_capacity <= 0:
            raise ValueError("Lane capacity must be positive.")

        if free_flow_speed <= 0:
            raise ValueError("Free-flow speed must be positive.")

        if jam_density <= 0:
            raise ValueError("Jam density must be positive.")

        # set link parameters
        self.length: float = length  # in kilometers
        self.lanes: int = lanes  # number of lanes
        self.lane_capacity: float = lane_capacity  # in vehicles per hour per lane
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane

        # identifier
        self.id: str = id if id is not None else str(uuid.uuid4())

        # optional start / end node identifiers to be set during connection
        self.origin_node_id: str | None = origin_node_id
        self.destination_node_id: str | None = destination_node_id

        # bidirectional linked list structure
        self._head: Optional[Cell] = None
        self._tail: Optional[Cell] = None
        self._cell_count: int = 0

    def add_cell(
        self,
        length: float,
        upcoming_lane_drop: int = 0,
    ) -> Cell:
        """Create and append a new `Cell` to the motorway link.

        Constructs a `Cell` with the supplied geometric parameters and
        appends it to the downstream end of the link. If the link already
        contains cells, upstream/downstream pointers are updated so the new
        cell is connected in the bidirectional linked list.

        Args:
            length (float): Cell length in kilometers (must be positive).
            upcoming_lane_drop (int): Number of lanes dropping downstream
                of this cell (default: 0).

        Returns:
            Cell: The newly created `Cell` instance.

        Raises:
            ValueError: If `length` is non-positive.
        """

        # check if the motorway link already contains an upstream cell and if there is a lane drop
        new_cell = Cell(length=length, upcoming_lane_drop=upcoming_lane_drop)

        # link the new cell into the bidirectional linked list
        if self._head is None:
            # first cell in motorway link
            self._head = new_cell
            self._tail = new_cell
        elif self._tail is None:
            # should not happen - if head is defined, tail should be too
            raise RuntimeError("Motorway link linked list is in an invalid state")
        else:
            # append to end of list
            self._tail.downstream = new_cell
            new_cell.upstream = self._tail
            self._tail = new_cell

        self._cell_count += 1

        return new_cell

    def partition_link(
        self, preferred_cell_size: float, dt: float, upcoming_lane_drop: int = 0
    ) -> None:
        """Partition the motorway link into a sequence of `Cell` objects.

        Cells are created to approximate `preferred_cell_size` while ensuring
        the Courant-Friedrichs-Lewy (CFL) condition is satisfied for the
        link's free-flow speed given the timestep `dt`. The last cell may be
        slightly longer to match the total link length. `dt` is expected in
        the same time units as `vf` (e.g. hours if `vf` is km/h).

        Args:
            preferred_cell_size (float): Preferred cell length in kilometers.
            dt (float): Simulation timestep (same time units as `vf`).
            upcoming_lane_drop (int): Number of lanes dropping downstream of
                the last cell (default: 0).
        """

        if preferred_cell_size <= 0:
            raise ValueError("Preferred cell size must be positive.")

        # clear existing cells
        self._head = None
        self._tail = None
        self._cell_count = 0

        # determine maximum allowable cell size from CFL condition
        min_cell_length = self.vf * dt
        valid_cell_size = max(preferred_cell_size, min_cell_length + 0.001)
        num_cells = int(self.length // valid_cell_size)

        if num_cells == 0:
            raise ValueError(
                f"Motorway link too short to fit a single segment. Segment length: {self.length} km, minimum cell size: {valid_cell_size} km"
            )

        # compute the homogeneous cell size (as close as possible to the preferred one; rounded to meters)
        balanced_cell_size = round(self.length / num_cells, 3)

        # add cells of the preferred length (or as close as possible to satisfy CFL condition)
        for _ in range(num_cells - 1):
            self.add_cell(length=balanced_cell_size)

        # add the last cell with the optional lane drop parameter to the network
        # the length of the last cell is determined by the remaining length of the link
        last_cell_size = self.length - balanced_cell_size * (num_cells - 1)
        self.add_cell(length=last_cell_size, upcoming_lane_drop=upcoming_lane_drop)

    def __len__(self) -> int:
        """Return the number of cells in the motorway link."""
        return self._cell_count

    def __iter__(self):
        """Iterate over cells from upstream to downstream."""
        current = self._head
        while current is not None:
            yield current
            current = current.downstream

    def first_cell(self) -> Optional[Cell]:
        """Return the first (most upstream) cell, or None if motorway link is empty."""
        return self._head

    def last_cell(self) -> Optional[Cell]:
        """Return the last (most downstream) cell, or None if motorway link is empty."""
        return self._tail

    def enumerate_cells(self):
        """Iterate over (index, cell) tuples from upstream to downstream."""
        for i, cell in enumerate(self):
            yield i, cell

    def get_cell(self, index: int) -> Cell:
        """Get cell at specified index by traversing linked list.

        Args:
            index: Zero-based index of cell (0 = first/upstream cell).

        Returns:
            Cell at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= self._cell_count:
            raise IndexError(f"Cell index {index} out of range [0, {self._cell_count})")

        current = self._head
        for _ in range(index):
            if current is None:
                raise IndexError(f"Cell index {index} out of range")
            current = current.downstream

        if current is None:
            raise IndexError(f"Cell index {index} out of range")

        return current
