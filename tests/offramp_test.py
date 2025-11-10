from traffic_flow_models import Offramp, Link


class TestOfframp:
    def test_init_assigns_attributes(self):
        mainline = Link(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        offramp = Offramp(
            lanes=2,
            lane_capacity=1600,
            free_flow_speed=70,
            jam_density=130,
        )
        mainline.offramp = offramp
        assert offramp.lanes == 2
        assert offramp.lane_capacity == 1600
        assert offramp.free_flow_speed == 70
        assert offramp.jam_density == 130
        assert mainline.offramp is offramp

    def test_network_link_assignment_via_constructor(self):
        mainline = Link(
            length=1.0,
            lanes=1,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            offramp=Offramp(
                lanes=1,
                lane_capacity=1400,
                free_flow_speed=60,
                jam_density=120,
            ),
        )

        assert mainline.offramp is not None
        assert mainline.offramp.lanes == 1
        assert mainline.offramp.lane_capacity == 1400
        assert mainline.offramp.free_flow_speed == 60
        assert mainline.offramp.jam_density == 120
