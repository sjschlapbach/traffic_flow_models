from traffic_flow_models import Onramp, Cell


class TestOnramp:
    def test_init_assigns_attributes(self):
        mainline = Cell(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        onramp = Onramp(
            lanes=2,
            lane_capacity=1800,
            free_flow_speed=90,
            jam_density=160,
        )

        # attach the ramp to the mainline cell (cell holds a single onramp)
        mainline.onramp = onramp
        assert onramp.lanes == 2
        assert onramp.lane_capacity == 1800
        assert onramp.free_flow_speed == 90
        assert onramp.jam_density == 160
        assert mainline.onramp is onramp

    def test_network_cell_assignment_via_constructor(self):
        mainline = Cell(
            length=1.0,
            lanes=1,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            onramp=Onramp(
                lanes=3,
                lane_capacity=2000,
                free_flow_speed=100,
                jam_density=150,
            ),
        )

        assert mainline.onramp is not None
        assert mainline.onramp.lanes == 3
        assert mainline.onramp.lane_capacity == 2000
        assert mainline.onramp.free_flow_speed == 100
        assert mainline.onramp.jam_density == 150
