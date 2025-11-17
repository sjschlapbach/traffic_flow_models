from traffic_flow_models.controller.alinea import AlineaController


class Onramp:
    """A simple container for on-ramp physical parameters.

    Attributes:
        lanes: Number of lanes on the onramp.
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
        controller: AlineaController | None = None,
    ) -> None:
        """Initialize the Onramp parameters.

        Args:
            lanes: Number of lanes on the onramp.
            lane_capacity: Vehicles per hour per lane capacity.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
        """

        if lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        if lane_capacity <= 0:
            raise ValueError("Lane capacity must be positive.")

        if free_flow_speed <= 0:
            raise ValueError("Free-flow speed must be positive.")

        if jam_density <= 0:
            raise ValueError("Jam density must be positive.")

        self.lanes: int = lanes  # number of lanes
        self.Qc_lane: float = lane_capacity  # in vehicles per hour per lane
        self.Qc: float = lane_capacity * lanes  # total cell capacity
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane
        self.controller = controller  # optional ramp metering controller
