from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .onramp import Onramp  # pragma: no cover - typing only
    from .offramp import Offramp  # pragma: no cover - typing only


class Link:
    # type annotations for static tools: instances will have these attributes
    onramp: Optional["Onramp"]
    offramp: Optional["Offramp"]

    # simple downstream/upstream references
    downstream_link: Optional["Link"]
    upstream_link: Optional["Link"]

    def __init__(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        downstream_link: Optional["Link"] = None,
        onramp: Optional["Onramp"] = None,
        offramp: Optional["Offramp"] = None,
    ) -> None:
        self.length: float = length  # in kilometers
        self.lanes: int = lanes  # number of lanes
        self.lane_capacity: float = lane_capacity  # in vehicles per hour per lane
        self.free_flow_speed: float = free_flow_speed  # in kilometers per hour
        self.jam_density: float = jam_density  # in vehicles per kilometer per lane

        # downstream/upstream references set through the network / user
        self.downstream_link: Optional[Link] = downstream_link
        self.upstream_link: Optional[Link] = None

        # at most one ramp of each type may attach to a link (optional)
        # allow passing ramps via constructor for convenience
        self.onramp = onramp
        self.offramp = offramp
