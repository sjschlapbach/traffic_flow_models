from traffic_flow_models import Onramp, Node, MotorwayLink


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
