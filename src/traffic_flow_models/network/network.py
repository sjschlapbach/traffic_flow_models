from typing import Iterator
from collections import deque

from traffic_flow_models.network.node import Node
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.offramp import Offramp


class Network:
    """
    Network class representing the traffic network.

    The network class is the main containeder for all network components,
    keeping track of all nodes included in the network alongside their connected
    links. Additionally, it is responsible for validating the structure of a network
    before simulation and the initialization of all necessary parameters (including
    demands, split ratios, densities downstream of destinations, etc.).

    Requirements:
        - Each network must have at least one origin link or onramp
        - Each network must have at least one destination
        - Each offramp needs to be connected to a destination
        - Each node needs to have at least one incoming and one outgoing link
          (including origins and destinations beyond regular links)
        - All nodes in the network must be connected through links (no unconnected components)
    """

    def __init__(self) -> None:
        """Initialize the Network object."""
        self._nodes: list[Node] = []

    def add_node(self, node: Node) -> None:
        """
        Add a `Node` instance to the network.

        Raises:
            TypeError: if `node` is not a Node instance.
            ValueError: if a node with the same `id` is already present.
        """
        if not isinstance(node, Node):
            raise TypeError("Only Node instances may be added to the Network.")

        if any(
            getattr(n, "id", None) == getattr(node, "id", None) for n in self._nodes
        ):
            raise ValueError(f"Node with id {node.id} already present in network.")

        self._nodes.append(node)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from the network by object or id.

        Args:
            node_id: The `id` of the `Node` instance to remove.

        Raises:
            ValueError: if no node with the given id is found.
        """
        for n in list(self._nodes):
            if getattr(n, "id", None) == node_id:
                self._nodes.remove(n)
                return

        raise ValueError(f"No node with id {node_id} found in network.")

    def get_node(self, id: str) -> Node | None:
        """Return the node with the given id or None if absent."""
        for n in self._nodes:
            if getattr(n, "id", None) == id:
                return n

        return None

    def list_nodes(self) -> list[Node]:
        """Return a shallow copy of the node list."""
        return list(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        for n in self._nodes:
            yield n

    def validate(self) -> bool:
        """
        Validate network structure according to class requirements.

        Requirements validated:
            - Each network must have at least one origin link or onramp
            - Each network must have at least one destination
            - Each offramp needs to be connected to a destination
            - Each node needs to have at least one incoming and one outgoing link
            - All nodes in the network must be connected through links

        Raises:
            ValueError: if any of the requirements are violated.
        """

        # verify that the network is not empty
        if len(self._nodes) < 2:
            raise ValueError("Network contains less than 2 nodes.")

        # call node-level validation (ensures at least one incoming & outgoing per node)
        for node in self._nodes:
            if not isinstance(node, Node):
                raise TypeError("Network contains non-Node object.")

            node.validate()

        # ensure that the network has at least one origin / onramp and one destination
        has_origin_or_onramp = False
        has_destination = False

        # collect all destination instances present in network (to check offramp targets)
        dests: set[Destination] = set()

        for node in self._nodes:
            for link in list(node.incoming) + list(node.outgoing):
                if isinstance(link, (Origin, Onramp)):
                    has_origin_or_onramp = True
                if isinstance(link, Destination):
                    has_destination = True
                    dests.add(link)
                if isinstance(link, Offramp) and link.destination is not None:
                    has_destination = True
                    dests.add(link.destination)

        if not has_origin_or_onramp:
            raise ValueError("Network must contain at least one origin or onramp link.")

        if not has_destination:
            raise ValueError("Network must contain at least one destination.")

        # check that every offramp has a destination
        for node in self._nodes:
            for link in list(node.incoming) + list(node.outgoing):
                if isinstance(link, Offramp):
                    if link.destination is None:
                        raise ValueError("Offramp is not connected to a destination.")

        # connectivity: use two DFS passes (original and reversed edges)
        # build adjacency (directed) between nodes: A -> B if any link in A.outgoing is in B.incoming
        num_nodes = len(self._nodes)

        # validate that each node's incoming links have their destination_node_id set
        # to this node's id, and each outgoing link has its origin_node_id set to
        # this node's id. This ensures the per-link origin/destination metadata is
        # consistent with the node topology.
        for node in self._nodes:
            for link in node.incoming:
                if hasattr(link, "destination_node_id"):
                    if link.destination_node_id is None:
                        raise ValueError(
                            f"Incoming link {getattr(link,'id',repr(link))} has no destination_node_id set for node {node.id}"
                        )
                    if link.destination_node_id != node.id:
                        raise ValueError(
                            f"Incoming link {getattr(link,'id',repr(link))} destination_node_id mismatch: expected {node.id}, got {link.destination_node_id}"
                        )
                else:
                    raise ValueError(
                        f"Incoming link {getattr(link,'id',repr(link))} missing destination_node_id attribute for node {node.id}"
                    )

            for link in node.outgoing:
                if hasattr(link, "origin_node_id"):
                    if link.origin_node_id is None:
                        raise ValueError(
                            f"Outgoing link {getattr(link,'id',repr(link))} has no origin_node_id set for node {node.id}"
                        )
                    if link.origin_node_id != node.id:
                        raise ValueError(
                            f"Outgoing link {getattr(link,'id',repr(link))} origin_node_id mismatch: expected {node.id}, got {link.origin_node_id}"
                        )
                else:
                    raise ValueError(
                        f"Outgoing link {getattr(link,'id',repr(link))} missing origin_node_id attribute for node {node.id}"
                    )

        # build a mapping from node id -> index for quick lookups
        node_id_to_index: dict[str, int] = {n.id: i for i, n in enumerate(self._nodes)}

        # build directed adjacency using each outgoing link's destination id.
        adj_forward: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        for i, node in enumerate(self._nodes):
            for link in node.outgoing:
                dest_id = getattr(link, "destination_node_id", None)
                if dest_id is None:
                    continue

                j = node_id_to_index.get(dest_id)
                if j is not None and i != j:
                    adj_forward[i].add(j)

        # reversed adjacency
        adj_reversed: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        for u, nbrs in adj_forward.items():
            for v in nbrs:
                adj_reversed[v].add(u)

        # choose a start node (index 0)
        start = 0

        vis1 = [False] * num_nodes
        vis2 = [False] * num_nodes

        # iterative DFS on original graph
        stack = [start]
        while stack:
            cur = stack.pop()
            if vis1[cur]:
                continue

            vis1[cur] = True
            for nb in adj_forward.get(cur, ()):
                if not vis1[nb]:
                    stack.append(nb)

        # iterative DFS on reversed graph
        stack = [start]
        while stack:
            cur = stack.pop()
            if vis2[cur]:
                continue
            vis2[cur] = True
            for nb in adj_reversed.get(cur, ()):
                if not vis2[nb]:
                    stack.append(nb)

        # any node that is neither reachable from start nor can reach start is disconnected
        unconnected = [
            self._nodes[i].id for i in range(num_nodes) if (not vis1[i] and not vis2[i])
        ]
        if unconnected:
            raise ValueError(f"Network contains unconnected components: {unconnected}")

        # all checks passed
        return True

    # TODO: during simulation the network requires the passing of demands for origins and onramps, downstream densities for destinations, split ratios at each node, etc.
    # TODO: we should also have a validation function that all these quantities are passed with the correct dimensions
    #   (time and number of exiting indices, matching ids, split ratios add up to one (warning and renormalization?), etc.)
