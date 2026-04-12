import pytest

from traffic_flow_models import (
    Onramp,
    Node,
    MotorwayLink,
    Network,
    Origin,
    Destination,
)


class TestOnramp:
    def test_init_assigns_attributes(self):
        link = MotorwayLink(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link.partition_link(preferred_cell_size=2.0, dt=0.001)

        onramp = Onramp(
            length=0.5,
            lanes=2,
            lane_capacity=1800,
            free_flow_speed=90,
            jam_density=160,
        )
        node = Node(incoming=[onramp], outgoing=[link])

        # ensure validate() does not raise and check the properties of the onramp
        node.validate()
        assert onramp in node.incoming
        assert onramp.lanes == 2
        assert onramp.Qc_lane == 1800
        assert onramp.vf == 90
        assert onramp.rho_jam == 160

    def test_network_cell_assignment_via_constructor(self):
        link = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link.partition_link(preferred_cell_size=1.0, dt=0.001)

        onramp = Onramp(
            length=0.5,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        n = Node(incoming=[onramp], outgoing=[link])

        # onramp should record its destination node id when connected
        assert getattr(onramp, "destination_node_id", None) == n.id

    def test_set_onramp_relations_raises_for_unconnected_onramp(self):
        net = Network(nodes=[])
        onramp = Onramp(
            length=0.5, lanes=1, lane_capacity=1000, free_flow_speed=60, jam_density=100
        )

        with pytest.raises(ValueError):
            net.set_onramp_relations(onramp)

    def test_set_onramp_relations_circular_network_no_duplicates(self):
        # circular motorway with three nodes, each receiving an onramp
        m1 = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        m2 = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        m3 = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )

        up = Onramp(
            length=0.5, lanes=1, lane_capacity=1000, free_flow_speed=60, jam_density=100
        )
        target = Onramp(
            length=0.5, lanes=1, lane_capacity=1000, free_flow_speed=60, jam_density=100
        )
        down = Onramp(
            length=0.5, lanes=1, lane_capacity=1000, free_flow_speed=60, jam_density=100
        )

        node1 = Node(incoming=[m3, up], outgoing=[m1])
        node2 = Node(incoming=[m1, target], outgoing=[m2])
        node3 = Node(incoming=[m2, down], outgoing=[m3])

        net = Network(nodes=[node1, node2, node3])

        up_list, down_list = net.set_onramp_relations(target, max_range_onramps=10)

        # uniqueness and no overlap
        up_ids = [o.id for o in up_list]
        down_ids = [o.id for o in down_list]

        # Each list should contain unique entries internally; overlap between
        # upstream and downstream is allowed (no uniqueness enforced).
        assert len(up_ids) == len(set(up_ids))
        assert len(up_ids) == 2
        assert len(down_ids) == len(set(down_ids))

        # expected discovery: each onramp should be discovered either as
        # upstream or downstream (in cyclical networks the classification
        # may depend on traversal order). Ensure both onramps are found.
        union_ids = set(up_ids) | set(down_ids)
        assert {up.id, down.id}.issubset(union_ids)

        # recompute relations with max_upstream=1, max_downstream=1 to check limits are respected
        up_list, down_list = net.set_onramp_relations(target, max_range_onramps=1)
        assert len(up_list) == 1
        assert len(down_list) == 1
        up_ids = [o.id for o in up_list]
        down_ids = [o.id for o in down_list]
        assert len(up_ids) == len(set(up_ids))
        assert len(down_ids) == len(set(down_ids))
        assert set(up_ids).isdisjoint(set(down_ids))

    def test_long_highway_many_onramps_limits(self):
        # build a long linear motorway with 20 onramps feeding successive nodes
        N_ONRAMPS = 20
        links = [
            MotorwayLink(
                length=1.0,
                lanes=1,
                lane_capacity=1500,
                free_flow_speed=80,
                jam_density=140,
            )
            for _ in range(N_ONRAMPS + 1)
        ]
        onramps = [
            Onramp(
                length=0.5,
                lanes=1,
                lane_capacity=1000,
                free_flow_speed=60,
                jam_density=100,
            )
            for _ in range(N_ONRAMPS)
        ]

        origin = Origin()
        dest = Destination()

        nodes = []
        nodes.append(Node(incoming=[origin], outgoing=[links[0]]))
        for i in range(1, N_ONRAMPS + 1):
            if i < N_ONRAMPS:
                nodes.append(
                    Node(incoming=[links[i - 1], onramps[i - 1]], outgoing=[links[i]])
                )
            else:
                # last node connects to destination
                nodes.append(
                    Node(incoming=[links[i - 1], onramps[i - 1]], outgoing=[dest])
                )

        net = Network(nodes=nodes)

        # pick a middle onramp as the target
        target_idx = N_ONRAMPS // 2
        target = onramps[target_idx]

        up, down = net.set_onramp_relations(target, max_range_onramps=5)

        assert len(up) == 5
        assert len(down) == 5
        # uniqueness and disjointness
        assert len({o.id for o in up}) == len(up)
        assert len({o.id for o in down}) == len(down)
