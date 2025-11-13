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
        self.lanes: int = lanes  # number of lanes
        self.Qc_lane: float = lane_capacity  # in vehicles per hour per lane
        self.Qc: float = lane_capacity * lanes  # total cell capacity
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane
        self.rho_cr: float = self.Qc_lane / self.vf  # critical density
        self.w: float = self.Qc_lane / (
            self.rho_jam - self.rho_cr
        )  # backwards wave speed
        self.controller = controller  # optional ramp metering controller
