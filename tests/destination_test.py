from traffic_flow_models import Destination, Node


class TestDestination:
    def test_id_assignment_when_provided(self):
        d = Destination(id="dest-42")
        assert d.id == "dest-42"

    def test_id_generated_when_none_or_empty(self):
        d = Destination(id=None)
        assert isinstance(d.id, str) and len(d.id) > 0
        d2 = Destination()
        assert isinstance(d2.id, str) and len(d2.id) > 0

        # destinations should not have origin node id until connected
        assert getattr(d, "origin_node_id", None) is None

        # connect to node
        n = Node(id="n-dest")
        n.add_outgoing(d)
        assert d.origin_node_id == n.id
