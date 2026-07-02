from traffic_flow_models import Offramp, Destination, Node, MotorwayLink


class TestOfframp:
    def test_init_assigns_attributes(self):
        link = MotorwayLink(length=2.0, lanes=3)
        link.partition_link(max_vf=100.0, preferred_cell_size=2.0, dt=0.001)

        offramp = Offramp(lanes=2)
        node = Node(incoming=[link], outgoing=[offramp])

        # ensure validate() does not raise and check the properties of the offramp
        node.validate()
        assert offramp in node.outgoing
        assert offramp.lanes == 2

    def test_network_cell_assignment_via_constructor(self):
        link = MotorwayLink(length=1.0, lanes=1)

        off = Offramp(lanes=1)

        # connect via a node: mainline -> offramp
        n = Node(incoming=[link], outgoing=[off])

        n.validate()
        assert off in n.outgoing
        assert off.lanes == 1
        assert off.origin_node_id == n.id

    def test_destination_instance_linking(self):
        dest = Destination(id="dest-abc")
        off = Offramp(lanes=1)

        # origin_node_id/destination_node_id should be unset until connected to nodes
        assert getattr(off, "origin_node_id", None) is None
        assert getattr(off, "destination_node_id", None) is None

        n_up = Node(id="n-up")
        n_up.add_outgoing(off)
        assert off.origin_node_id == n_up.id

        n_down = Node(id="n-down", incoming=[off], outgoing=[dest])
        assert off.destination_node_id == n_down.id
        assert dest.origin_node_id == n_down.id

    def test_id_assignment_and_generation(self):
        # provided id is preserved
        off1 = Offramp(id="off-123", lanes=1)
        assert off1.id == "off-123"

        # generated id when not provided
        off2 = Offramp(lanes=1)
        assert isinstance(off2.id, str) and len(off2.id) > 0
