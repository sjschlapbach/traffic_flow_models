from typing import List, Optional

from .link import Link
from .onramp import Onramp
from .offramp import Offramp


class Network:
    """A simple ordered mainline network container.

    The Network class stores an ordered list of mainline `Link` instances
    arranged from upstream (index 0) to downstream (last index). It
    provides convenience methods to add links and attach or detach on-/off-
    ramps. The Network does not perform simulation — it only manages the
    topology and basic validation when linking objects together.

    Attributes:
        links: Ordered list of mainline `Link` objects (upstream ->
            downstream).
    """

    def __init__(self) -> None:
        """Initialize an empty Network.

        The created network contains an empty `links` list. Links can be
        added with `add_link` which takes physical parameters and optionally
        attaches existing ramp objects.
        """
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
        """Create a new mainline link and append it to the network.

        This method constructs a `Link` instance using the provided
        physical parameters and appends it to the end of the network. If a
        previous link exists it will set the upstream/downstream references
        so the two links are connected. Optional `Onramp`/`Offramp`
        instances can be attached directly; their types are validated.

        Args:
            length: Link length in kilometers.
            lanes: Number of lanes on the link.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
            onramp: Optional existing `Onramp` instance to attach.
            offramp: Optional existing `Offramp` instance to attach.

        Returns:
            The newly created `Link` instance.

        Raises:
            TypeError: If provided `onramp`/`offramp` are not of the expected
                types.
        """

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
        """Attach a new `Onramp` to a link by index.

        Args:
            link_index: Index of the link in `self.links` to attach the ramp
                to.
            lanes: Number of lanes on the onramp.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h for the onramp.
            jam_density: Jam density in vehicles per km per lane for the
                onramp.

        Returns:
            The created `Onramp` instance.

        Raises:
            ValueError: If the target link already has an onramp attached.
        """

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
        """Attach a new `Offramp` to a link by index.

        Args:
            link_index: Index of the link in `self.links` to attach the ramp
                to.
            lanes: Number of lanes on the offramp.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h for the offramp.
            jam_density: Jam density in vehicles per km per lane for the
                offramp.

        Returns:
            The created `Offramp` instance.

        Raises:
            ValueError: If the target link already has an offramp attached.
        """

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
        """Return the `Onramp` attached to the link at `link_index`.

        Returns None if no onramp is attached.
        """

        return self.links[link_index].onramp

    def get_offramp(self, link_index: int) -> Optional[Offramp]:
        """Return the `Offramp` attached to the link at `link_index`.

        Returns None if no offramp is attached.
        """

        return self.links[link_index].offramp

    def remove_onramp(self, link_index: int) -> None:
        """Detach and remove the onramp from the link at `link_index`.

        After calling this the link's `onramp` attribute will be set to
        `None`.
        """

        link = self.links[link_index]
        link.onramp = None

    def remove_offramp(self, link_index: int) -> None:
        """Detach and remove the offramp from the link at `link_index`.

        After calling this the link's `offramp` attribute will be set to
        `None`.
        """

        link = self.links[link_index]
        link.offramp = None
