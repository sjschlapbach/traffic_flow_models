import pytest

from traffic_flow_models import MotorwayLink


class TestCell:
    def test_init_assigns_attributes(self):
        # create a link that holds the physical parameters and a cell
        link = MotorwayLink(length=2.5, lanes=3)
        max_vf = 100.0
        # partition the link into one cell
        link.partition_link(max_vf=max_vf, preferred_cell_size=2.5, dt=0.001)
        cell = link.get_cell(0)

        # verify link-level physical parameters are preserved
        assert cell.length == 2.5
        assert link.lanes == 3
        assert cell.upcoming_lane_drop == 0

        # verify linked list pointers are initialized
        assert cell.upstream is None
        assert cell.downstream is None

    def test_partition_link_creates_cells_of_preferred_size_and_remainder(self):
        link = MotorwayLink(length=3.5, lanes=2)
        preferred = 1.0
        dt = 0.01
        max_vf = 90.0

        # compute expected partitioning using the same rules as the implementation
        min_cell_length = max_vf * dt
        valid_cell_size = max(preferred, min_cell_length + 0.001)
        expected_num = int(link.length // valid_cell_size)
        assert expected_num == 3  # 2 cells of 1.0 m, 1 cell of 1.5 m

        # partition the link into cells
        link.partition_link(max_vf=max_vf, preferred_cell_size=preferred, dt=dt)
        assert len(link) == expected_num

        # collect cell lengths and verify they sum to the link length
        lengths = [c.length for c in link]
        assert sum(lengths) == pytest.approx(link.length)

        # balanced size used for the first (num-1) cells
        balanced = round(link.length / expected_num, 3)
        for l in lengths[:-1]:
            assert l == pytest.approx(balanced)

        # last cell is the remainder
        last_expected = link.length - balanced * (expected_num - 1)
        assert lengths[-1] == pytest.approx(last_expected)
