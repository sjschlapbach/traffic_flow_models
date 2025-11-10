class Onramp:
    def __init__(
        self,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
    ) -> None:
        self.lanes: int = lanes  # number of lanes
        self.lane_capacity: float = lane_capacity  # in vehicles per hour per lane
        self.free_flow_speed: float = free_flow_speed  # in kilometers per hour
        self.jam_density: float = jam_density  # in vehicles per kilometer per lane
