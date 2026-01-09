from traffic_flow_models import Node, MotorwayLink, Onramp, Offramp, Origin, Destination
import pytest


class TestNode:
    def test_validate_raises_when_empty(self):
        n = Node(id=None)
        with pytest.raises(ValueError):
            n.validate()

    def test_add_and_validate_correct_types(self):
        n = Node(id="n1")

        m_in = MotorwayLink()
        on = Onramp(lanes=1, lane_capacity=1000, free_flow_speed=80, jam_density=150)
        org = Origin(
            id=None, lanes=1, lane_capacity=1000, free_flow_speed=80, jam_density=150
        )

        m_out = MotorwayLink()
        off = Offramp(lanes=1, lane_capacity=900, free_flow_speed=70, jam_density=140)
        dest = Destination(id=None)

        # precondition: node id attributes should be unset
        assert getattr(m_in, "destination_node_id", None) is None
        assert getattr(on, "destination_node_id", None) is None
        assert getattr(org, "destination_node_id", None) is None
        assert getattr(m_out, "origin_node_id", None) is None
        assert getattr(off, "origin_node_id", None) is None
        assert getattr(dest, "origin_node_id", None) is None

        # add incoming/outgoing
        n.add_incoming(m_in)
        n.add_incoming(on)
        n.add_incoming(org)

        n.add_outgoing(m_out)
        n.add_outgoing(off)
        n.add_outgoing(dest)

        # ensure stored
        assert m_in in n.incoming
        assert on in n.incoming
        assert org in n.incoming
        assert m_out in n.outgoing
        assert off in n.outgoing
        assert dest in n.outgoing

        # after attaching, node id attributes should be set correctly
        assert m_in.destination_node_id == n.id
        assert on.destination_node_id == n.id
        assert org.destination_node_id == n.id
        assert m_out.origin_node_id == n.id
        assert off.origin_node_id == n.id
        assert dest.origin_node_id == n.id

        # should not raise
        n.validate()

    def test_constructor_initial_links(self):
        m1 = MotorwayLink()
        o1 = Onramp(lanes=2, lane_capacity=1200, free_flow_speed=90, jam_density=160)
        m2 = MotorwayLink()
        d1 = Destination(id=None)

        # precondition: ids unset
        assert getattr(m1, "destination_node_id", None) is None
        assert getattr(o1, "destination_node_id", None) is None
        assert getattr(m2, "origin_node_id", None) is None
        assert getattr(d1, "origin_node_id", None) is None

        n = Node(id="n2", incoming=[m1, o1], outgoing=[m2, d1])

        assert m1 in n.incoming and o1 in n.incoming
        assert m2 in n.outgoing and d1 in n.outgoing

        # constructor should set node ids for attached links
        assert m1.destination_node_id == n.id
        assert o1.destination_node_id == n.id
        assert m2.origin_node_id == n.id
        assert d1.origin_node_id == n.id

        n.validate()

    def test_type_validation_rejects_wrong_types(self):
        n = Node(id="n3")

        # Offramp is not allowed as incoming
        off = Offramp(lanes=1, lane_capacity=900, free_flow_speed=70, jam_density=140)
        with pytest.raises(TypeError):
            n.add_incoming(off)

        # Origin not allowed as outgoing
        org = Origin(
            id=None, lanes=1, lane_capacity=1000, free_flow_speed=80, jam_density=150
        )
        with pytest.raises(TypeError):
            n.add_outgoing(org)

        # completely unrelated type
        with pytest.raises(TypeError):
            n.add_incoming(123)

    def test_remove_and_set_methods(self):
        m1 = MotorwayLink()
        m2 = MotorwayLink()
        o1 = Onramp(lanes=1, lane_capacity=1000, free_flow_speed=80, jam_density=150)
        d1 = Destination(id=None)

        n = Node(id="n4")
        # precondition: ids unset
        assert getattr(m1, "destination_node_id", None) is None
        assert getattr(o1, "destination_node_id", None) is None

        n.add_incoming(m1)
        n.add_incoming(o1)
        assert m1 in n.incoming and o1 in n.incoming

        # after attach
        assert m1.destination_node_id == n.id
        assert o1.destination_node_id == n.id

        # remove by id (remove the motorway link which now has an id)
        n.remove_incoming_by_id(m1.id)
        assert m1 not in n.incoming and o1 in n.incoming

        # removed link should have its node id cleared
        assert m1.destination_node_id is None

        # add and remove outgoing by id
        n.add_outgoing(d1)
        assert d1 in n.outgoing
        assert d1.origin_node_id == n.id
        n.remove_outgoing_by_id(d1.id)
        assert d1 not in n.outgoing

        # removed outgoing should have its origin_node_id cleared
        assert d1.origin_node_id is None

        # set incoming to new list
        n.set_incoming([m2])
        assert n.incoming == [m2]

        # previous incoming links should have their destination_node_id cleared
        assert o1.destination_node_id is None
        # new incoming should have node id set
        assert m2.destination_node_id == n.id

        # set_outgoing with invalid type should raise
        with pytest.raises(TypeError):
            n.set_outgoing([o1])

    def test_id_assignment(self):
        provided = "custom-id"
        n = Node(id=provided)
        assert n.id == provided

        n2 = Node(id=None)
        assert isinstance(n2.id, str) and len(n2.id) > 0
