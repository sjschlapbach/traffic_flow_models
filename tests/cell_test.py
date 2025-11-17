from traffic_flow_models import Cell


class TestCell:
    def test_init_assigns_attributes(self):
        cell = Cell(
            length=2.5,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        assert cell.length == 2.5
        assert cell.lanes == 3
        assert cell.Qc_lane == 2000
        assert cell.Qc == 6000
        assert cell.vf == 100
        assert cell.rho_jam == 150
        assert cell.upcoming_lane_drop == 0
