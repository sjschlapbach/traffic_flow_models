class Offramp:
    """A simple container for off-ramp physical parameters.

    Attributes:
        lanes: Number of lanes on the offramp.
        lane_capacity: Capacity per lane in vehicles per hour.
        free_flow_speed: Free-flow speed in km/h.
        jam_density: Jam density in vehicles per km per lane.
    """

    def __init__(
        self,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
    ) -> None:
        """Initialize the Offramp parameters.

        Args:
            lanes: Number of lanes on the offramp.
            lane_capacity: Vehicles per hour per lane capacity.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
        """
        self.lanes: int = lanes  # number of lanes
        self.lane_capacity: float = lane_capacity  # vehicles per hour per lane
        self.free_flow_speed: float = free_flow_speed  # kilometers per hour
        self.jam_density: float = jam_density  # vehicles per kilometer per lane
