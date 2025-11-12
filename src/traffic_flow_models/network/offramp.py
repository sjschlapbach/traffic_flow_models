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
        split_ratio: float,
    ) -> None:
        """Initialize the Offramp parameters.

        Args:
            lanes: Number of lanes on the offramp.
            lane_capacity: Vehicles per hour per lane capacity.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
        """
        self.lanes: int = lanes  # number of lanes
        self.Qc_lane: float = lane_capacity  # in vehicles per hour per lane
        self.Qc: float = lane_capacity * lanes  # total cell capacity
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane
        self.rho_cr: float = self.Qc_lane / self.vf  # critical density
        self.w: float = self.Qc_lane / (
            self.rho_jam - self.rho_cr
        )  # backwards wave speed
        self.split_ratio: float = split_ratio  # portion of mainline flow exiting
