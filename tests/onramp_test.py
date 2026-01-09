from traffic_flow_models import Onramp, Cell, Node


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
        assert onramp.Qc_lane == 1800
        assert onramp.vf == 90
        assert onramp.rho_jam == 160
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
        assert mainline.onramp.Qc_lane == 2000
        assert mainline.onramp.vf == 100
        assert mainline.onramp.rho_jam == 150

        # onramp should not have destination node id until connected
        assert getattr(mainline.onramp, "destination_node_id", None) is None
        n = Node(id="n-on")
        n.add_incoming(mainline.onramp)
        assert mainline.onramp.destination_node_id == n.id
