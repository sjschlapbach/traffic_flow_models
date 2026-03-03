import uuid
import warnings
from typing import Iterable, List, Tuple, Any

from traffic_flow_models.network.motorway_link import MotorwayLink
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.destination import Destination


class Node:
    """
    Base class for network nodes to create connected networks.

    Networks compatible with the more complex second-order traffic model formulations
    often require nodes to connect different types of links. Correspondingly, this
    node class serves as a connector between different links, accepting multiple
    incoming and outgoing links that can be motorway links, onramps, offramps,
    origins and destinations.

    Attributes:
        id: Unique identifier for the node.
        incoming: List of incoming links connected to the node.
        outgoing: List of outgoing links connected to the node.
    """

    def __init__(
        self,
        id: str | None = None,
        incoming: Iterable[Any] | None = None,
        outgoing: Iterable[Any] | None = None,
    ) -> None:
        """Initialize the Node object.

        Args:
            id: Unique identifier for the node (optional).
            incoming: Optional iterable of incoming link objects to attach.
            outgoing: Optional iterable of outgoing link objects to attach.
        """
        self.id: str = id if id is not None else str(uuid.uuid4())  # node identifier
        self.incoming: List[Any] = []  # list of incoming links
        self.outgoing: List[Any] = []  # list of outgoing links

        # optional position coordinates (mainly for illustration purposes)
        self.position: Tuple[float, float] | None = None

        # attach any provided links using the public helpers to ensure validation
        if incoming is not None:
            self.add_incoming_multiple(incoming)

        if outgoing is not None:
            self.add_outgoing_multiple(outgoing)

    def _allowed_incoming_types(self):
        """
        Return a tuple of allowed types for incoming links.
        """
        return (MotorwayLink, Onramp, Origin)

    def _allowed_outgoing_types(self):
        """
        Return a tuple of allowed types for outgoing links.
        """
        return (MotorwayLink, Offramp, Destination)

    def _validate_link_type(self, link: Any, allowed_types: tuple) -> None:
        """
        Validate that a link is of an allowed type.

        Args:
            link: The link object to validate.
            allowed_types: A tuple of allowed types for the link.

        Raises:
            TypeError: If the link is not of an allowed type.
        """
        if not isinstance(link, allowed_types):
            allowed_names = ", ".join([t.__name__ for t in allowed_types])
            raise TypeError(
                f"Link of type {type(link).__name__} is not allowed here. Allowed: {allowed_names}"
            )

    def validate(self) -> None:
        """Validate the node configuration.

        Ensures that the node has at least one incoming and one outgoing link and that
        all attached links are of allowed types.

        Raises:
            ValueError: If the node has no incoming or outgoing links.
            TypeError: If any attached link is of an invalid type.
        """

        if len(self.incoming) == 0:
            raise ValueError("Node must have at least one incoming link.")

        if len(self.outgoing) == 0:
            raise ValueError("Node must have at least one outgoing link.")

        # validate types of attached links
        for link in self.incoming:
            self._validate_link_type(link, self._allowed_incoming_types())

        for link in self.outgoing:
            self._validate_link_type(link, self._allowed_outgoing_types())

    def set_position(self, x: float, y: float) -> None:
        """
        Set the position coordinates of the node.

        Args:
            x: X-coordinate of the node's position.
            y: Y-coordinate of the node's position.
        """
        self.position = (x, y)

    def _set_origin_node_to_self(self, link: Any) -> None:
        """
        Set the origin_node_id attribute of a link to this node's id.

        Args:
            link: The link object whose origin_node_id to set.
        """
        if hasattr(link, "origin_node_id"):
            link.origin_node_id = self.id
        else:
            raise TypeError(
                f"Link of type {type(link).__name__} has no 'origin_node_id' attribute to set."
            )

    def _set_destination_node_to_self(self, link: Any) -> None:
        """
        Set the destination_node_id attribute of a link to this node's id.

        Args:
            link: The link object whose destination_node_id to set.
        """
        if hasattr(link, "destination_node_id"):
            link.destination_node_id = self.id
        else:
            raise TypeError(
                f"Link of type {type(link).__name__} has no 'destination_node_id' attribute to set."
            )

    def _clear_origin_node_id(self, link: Any) -> None:
        """
        Clear the origin_node_id attribute of a link if it exists.

        Args:
            link: The link object whose origin_node_id to clear.
        """
        if hasattr(link, "origin_node_id"):
            link.origin_node_id = None
        else:
            raise TypeError(
                f"Link of type {type(link).__name__} has no 'origin_node_id' attribute to clear."
            )

    def _clear_destination_node_id(self, link: Any) -> None:
        """
        Clear the destination_node_id attribute of a link if it exists.

        Args:
            link: The link object whose destination_node_id to clear.
        """
        if hasattr(link, "destination_node_id"):
            link.destination_node_id = None
        else:
            raise TypeError(
                f"Link of type {type(link).__name__} has no 'destination_node_id' attribute to clear."
            )

    def add_incoming(self, link: Any) -> None:
        """
        Add a single incoming link after validating its type.

        Args:
            link: The link object to add as incoming.
        """
        self._validate_link_type(link, self._allowed_incoming_types())
        if link in self.incoming:
            warnings.warn("Link already present in incoming links.", stacklevel=2)
            return

        # add the link to the incoming list
        self.incoming.append(link)

        # update the destination of the link
        self._set_destination_node_to_self(link)

    def add_outgoing(self, link: Any) -> None:
        """
        Add a single outgoing link after validating its type.

        Args:
            link: The link object to add as outgoing.
        """
        self._validate_link_type(link, self._allowed_outgoing_types())
        if link in self.outgoing:
            warnings.warn("Link already present in outgoing links.", stacklevel=2)
            return

        # add the link to the outgoing list
        self.outgoing.append(link)

        # update the origin of the link
        self._set_origin_node_to_self(link)

    def add_incoming_multiple(self, links: Iterable[Any]) -> None:
        """
        Add multiple incoming links from an iterable.

        Args:
            links: An iterable of link objects to add as incoming.
        """
        for l in links:
            self.add_incoming(l)

    def add_outgoing_multiple(self, links: Iterable[Any]) -> None:
        """
        Add multiple outgoing links from an iterable.

        Args:
            links: An iterable of link objects to add as outgoing.
        """
        for l in links:
            self.add_outgoing(l)

    def remove_incoming_by_id(self, id: str) -> None:
        """Remove the first incoming link matching the given `id`.

        Args:
            id: The identifier of the incoming link to remove.
        """
        for link in list(self.incoming):
            if getattr(link, "id", None) == id:
                # clear attached node id on the removed link when possible
                self._clear_destination_node_id(link)

                # remove the link from the incoming list
                self.incoming.remove(link)
                return

        warnings.warn("No incoming link with id found to remove.", stacklevel=2)

    def remove_outgoing_by_id(self, id: str) -> None:
        """Remove the first outgoing link matching the given `id`.

        Args:
            id: The identifier of the outgoing link to remove.
        """
        for link in list(self.outgoing):
            if getattr(link, "id", None) == id:
                # clear attached node id on the removed link when possible
                self._clear_origin_node_id(link)

                # remove the link from the outgoing list
                self.outgoing.remove(link)
                return

        warnings.warn("No outgoing link with id found to remove.", stacklevel=2)

    def set_incoming(self, links: Iterable[Any]) -> None:
        """
        Replace all incoming links with the provided iterable after validation.

        Args:
            links: An iterable of link objects to set as incoming.
        """
        new_list: List[Any] = []
        for l in links:
            self._validate_link_type(l, self._allowed_incoming_types())
            if l not in new_list:
                new_list.append(l)
            else:
                warnings.warn(
                    "Duplicate link found in incoming links; ignoring.", stacklevel=2
                )

        old_incoming = list(self.incoming)
        self.incoming = new_list

        # clear destination_node_id on any previously attached links that were removed
        for old in old_incoming:
            self._clear_destination_node_id(old)

        # set destination_node_id on newly attached incoming links
        for l in self.incoming:
            self._set_destination_node_to_self(l)

    def set_outgoing(self, links: Iterable[Any]) -> None:
        """
        Replace all outgoing links with the provided iterable after validation.

        Args:
            links: An iterable of link objects to set as outgoing.
        """
        new_list: List[Any] = []
        for l in links:
            self._validate_link_type(l, self._allowed_outgoing_types())
            if l not in new_list:
                new_list.append(l)
            else:
                warnings.warn(
                    "Duplicate link found in outgoing links; ignoring.", stacklevel=2
                )

        old_outgoing = list(self.outgoing)
        self.outgoing = new_list

        # clear origin_node_id on any previously attached links that were removed
        for old in old_outgoing:
            self._clear_origin_node_id(old)

        # set origin_node_id on newly attached outgoing links
        for l in self.outgoing:
            self._set_origin_node_to_self(l)
