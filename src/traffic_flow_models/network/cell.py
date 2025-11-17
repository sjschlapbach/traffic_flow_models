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
        onramp: Optional attached `Onramp` instance.
        offramp: Optional attached `Offramp` instance.
    """

    # type annotations for static tools: instances will have these attributes
    onramp: Optional["Onramp"]
    offramp: Optional["Offramp"]

    def __init__(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        onramp: Optional["Onramp"] = None,
        offramp: Optional["Offramp"] = None,
    ) -> None:
        """Create a new Cell with physical parameters.

        Please note that the critical density and backward wave speed are
        not stored in the cell explicitly, since their values depend on the
        fundamental diagram shape used in the model. They are computed through
        member functions of the corresponding model classes.

        Args:
            length: Cell length in kilometers.
            lanes: Number of lanes on the cell.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
            onramp: Optional `Onramp` instance to attach to this cell.
            offramp: Optional `Offramp` instance to attach to this cell.
        """

        if length <= 0:
            raise ValueError("Cell length must be positive.")

        if lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        if lane_capacity <= 0:
            raise ValueError("Lane capacity must be positive.")

        if free_flow_speed <= 0:
            raise ValueError("Free-flow speed must be positive.")

        if jam_density <= 0:
            raise ValueError("Jam density must be positive.")

        self.length: float = length  # in kilometers
        self.lanes: int = lanes  # number of lanes
        self.Qc_lane: float = lane_capacity  # in vehicles per hour per lane
        self.Qc: float = lane_capacity * lanes  # total cell capacity
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane

        # store if there is a lane drop coming up between this and the next downstream cell
        # if set, the number of dropped lanes is stored
        self.upcoming_lane_drop: int = 0

        # at most one ramp of each type may attach to a cell (optional)
        # allow passing ramps via constructor for convenience
        self.onramp = onramp
        self.offramp = offramp
