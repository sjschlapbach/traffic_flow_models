from traffic_flow_models import Origin, Cell, Node
import pytest


class TestOrigin:
    def test_init_assigns_attributes(self):
        o = Origin()
        assert isinstance(o.id, str) and len(o.id) > 0

        # origin should not have destination node id until connected
        assert getattr(o, "destination_node_id", None) is None

        # connect to a node and verify id set
        n = Node(id="n-origin")
        n.add_incoming(o)
        assert o.destination_node_id == n.id
