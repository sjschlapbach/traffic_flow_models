from traffic_flow_models import Offramp, Cell, Destination, Node


class TestOfframp:
    def test_init_assigns_attributes(self):
        mainline = Cell(
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
        assert offramp.Qc_lane == 1600
        assert offramp.vf == 70
        assert offramp.rho_jam == 130
        assert mainline.offramp is offramp

    def test_network_cell_assignment_via_constructor(self):
        mainline = Cell(
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
        assert mainline.offramp.Qc_lane == 1400
        assert mainline.offramp.vf == 60
        assert mainline.offramp.rho_jam == 120

    def test_destination_instance_linking(self):
        dest = Destination(id="dest-abc")
        off = Offramp(
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
            destination=dest,
        )
        assert off.destination is dest

        # origin_node_id should be unset until connected to a node
        assert getattr(off, "origin_node_id", None) is None
        n = Node(id="n-off")
        n.add_outgoing(off)
        assert off.origin_node_id == n.id

    def test_id_assignment_and_generation(self):
        # provided id is preserved
        off1 = Offramp(
            id="off-123",
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
        )
        assert off1.id == "off-123"

        # generated id when not provided
        off2 = Offramp(
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
        )
        assert isinstance(off2.id, str) and len(off2.id) > 0
