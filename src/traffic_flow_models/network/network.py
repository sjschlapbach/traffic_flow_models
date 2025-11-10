from typing import List, Optional

from .link import Link
from .onramp import Onramp
from .offramp import Offramp


class Network:
    def __init__(self) -> None:
        # ordered list of mainline links (upstream -> downstream)
        self.links: List[Link] = []

    def add_link(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        onramp: Optional[Onramp] = None,
        offramp: Optional[Offramp] = None,
    ) -> Link:
        new_link = Link(
            length=length,
            lanes=lanes,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
        )

        # chain from previous downstream reference: set downstream and upstream
        # pointers so both sides of the connection are known.
        if len(self.links) > 0:
            prev = self.links[-1]
            prev.downstream_link = new_link
            new_link.upstream_link = prev
        self.links.append(new_link)

        # attach provided ramp objects directly (do not attempt to construct
        # ramps from dictionaries). Validate types for helpful errors.
        if onramp is not None:
            if not isinstance(onramp, Onramp):
                raise TypeError("onramp must be an Onramp instance")
            new_link.onramp = onramp

        if offramp is not None:
            if not isinstance(offramp, Offramp):
                raise TypeError("offramp must be an Offramp instance")
            new_link.offramp = offramp

        return new_link

    def add_onramp(
        self,
        link_index: int,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
    ) -> Onramp:
        link = self.links[link_index]
        if link.onramp is not None:
            raise ValueError("Link already has an onramp attached")

        ramp = Onramp(
            lanes=lanes,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
        )
        link.onramp = ramp
        return ramp

    def add_offramp(
        self,
        link_index: int,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
    ) -> Offramp:
        link = self.links[link_index]
        if link.offramp is not None:
            raise ValueError("Link already has an offramp attached")
        ramp = Offramp(
            lanes=lanes,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
        )
        link.offramp = ramp
        return ramp

    def get_onramp(self, link_index: int) -> Optional[Onramp]:
        return self.links[link_index].onramp

    def get_offramp(self, link_index: int) -> Optional[Offramp]:
        return self.links[link_index].offramp

    def remove_onramp(self, link_index: int) -> None:
        link = self.links[link_index]
        link.onramp = None

    def remove_offramp(self, link_index: int) -> None:
        link = self.links[link_index]
        link.offramp = None
