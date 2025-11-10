class Offramp:
    def __init__(
        self,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
    ) -> None:
        self.lanes: int = lanes  # number of lanes
        self.lane_capacity: float = lane_capacity  # vehicles per hour per lane
        self.free_flow_speed: float = free_flow_speed  # kilometers per hour
        self.jam_density: float = jam_density  # vehicles per kilometer per lane
