from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .onramp import Onramp  # pragma: no cover - typing only
    from .offramp import Offramp  # pragma: no cover - typing only


class Cell:
    """Represents a single highway mainline cell.

    A Cell stores the physical parameters required for simple traffic
    modelling (length, lanes, capacity, speeds, densities) and optional
    references to neighboring cells and a single onramp/offramp. The class
    intentionally contains only data; network topology and validation are
    handled by the `Network` container.

    Attributes:
        length: Cell length in kilometers.
        lanes: Number of lanes on the cell.
        lane_capacity: Capacity per lane in vehicles per hour.
        free_flow_speed: Free-flow speed in km/h.
        jam_density: Jam density in vehicles per km per lane.
        downstream_cell: Optional reference to the downstream Cell.
        upstream_cell: Optional reference to the upstream Cell.
        onramp: Optional attached `Onramp` instance.
        offramp: Optional attached `Offramp` instance.
    """

    # type annotations for static tools: instances will have these attributes
    onramp: Optional["Onramp"]
    offramp: Optional["Offramp"]

    # simple downstream/upstream references
    downstream_cell: Optional["Cell"]
    upstream_cell: Optional["Cell"]

    def __init__(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        downstream_cell: Optional["Cell"] = None,
        onramp: Optional["Onramp"] = None,
        offramp: Optional["Offramp"] = None,
    ) -> None:
        """Create a new Cell with physical parameters.

        Args:
            length: Cell length in kilometers.
            lanes: Number of lanes on the cell.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
            downstream_cell: Optional downstream cell reference.
            onramp: Optional `Onramp` instance to attach to this cell.
            offramp: Optional `Offramp` instance to attach to this cell.
        """
        self.length: float = length  # in kilometers
        self.lanes: int = lanes  # number of lanes
        self.Qc_lane: float = lane_capacity  # in vehicles per hour per lane
        self.Qc: float = lane_capacity * lanes  # total cell capacity
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane
        self.rho_cr: float = self.Qc_lane / self.vf  # critical density
        self.w: float = self.Qc / (self.rho_jam - self.rho_cr)  # backwards wave speed

        # downstream/upstream references set through the network / user
        self.downstream_cell: Optional[Cell] = downstream_cell
        self.upstream_cell: Optional[Cell] = None

        # at most one ramp of each type may attach to a cell (optional)
        # allow passing ramps via constructor for convenience
        self.onramp = onramp
        self.offramp = offramp
