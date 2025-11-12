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
        assert cell.rho_cr == 20.0
        assert cell.downstream_cell is None
        assert cell.upstream_cell is None

    def test_next_cell_assignment_via_constructor(self):
        downstream = Cell(
            length=1.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )
        upstream = Cell(
            length=3.0,
            lanes=1,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            downstream_cell=downstream,
        )
        downstream.upstream_cell = upstream

        assert upstream.downstream_cell is downstream
        assert upstream.downstream_cell is not None
        assert upstream.downstream_cell.length == 1.0
        assert downstream.upstream_cell is upstream

    def test_mutating_downstream_reflects_in_upstream_reference(self):
        a = Cell(
            length=1.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )
        b = Cell(
            length=2.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )

        a.downstream_cell = b
        b.upstream_cell = a
        b.length = 2.5

        # downstream reference should see updated downstream object
        assert a.downstream_cell.length == 2.5

        # cells between network cells are set correctly
        assert b.upstream_cell is a
        assert a.downstream_cell is b
