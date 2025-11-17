import pytest

from traffic_flow_models import Network, Cell, Onramp, Offramp


class TestNetwork:
    def test_add_cell_single(self):
        net = Network()
        l = net.add_cell(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        assert isinstance(l, Cell)
        assert l.length == 2.0
        assert l.lanes == 3
        assert l.Qc_lane == 2000
        assert l.vf == 100
        assert l.rho_jam == 150

        # no chaining for a single cell
        assert net.cells[0] is l

    def test_add_cell_chaining_multiple(self):
        net = Network()
        a = net.add_cell(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        b = net.add_cell(
            length=2.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )
        c = net.add_cell(
            length=3.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=170,
        )

        # order preserved
        assert net.cells == [a, b, c]

    def test_add_cell_with_ramps_instances(self):
        net = Network()
        on = Onramp(lanes=2, lane_capacity=1600, free_flow_speed=70, jam_density=130)
        off = Offramp(
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
            split_ratio=0.2,
        )

        l = net.add_cell(
            length=1.5,
            lanes=2,
            lane_capacity=1800,
            free_flow_speed=90,
            jam_density=160,
            onramp=on,
            offramp=off,
        )
        assert l.onramp is on
        assert l.offramp is off

    def test_add_cell_with_wrong_ramp_type_raises(self):
        net = Network()
        # passing non-Onramp object should raise TypeError
        try:
            net.add_cell(1.0, 1, 1500, 80, 140, onramp={})  # type: ignore
            raised = False
        except TypeError:
            raised = True
        assert raised

        try:
            net.add_cell(1.0, 1, 1500, 80, 140, offramp=123)  # type: ignore
            raised = False
        except TypeError:
            raised = True
        assert raised

    def test_add_onramp_and_offramp_methods_and_duplicates(self):
        net = Network()
        net.add_cell(1.0, 1, 1500, 80, 140)

        # attach via method
        r = net.add_onramp(
            0, lanes=2, lane_capacity=1600, free_flow_speed=70, jam_density=130
        )
        assert isinstance(r, Onramp)
        assert net.get_onramp(0) is r

        # duplicate attach should raise ValueError
        try:
            net.add_onramp(
                0, lanes=1, lane_capacity=1400, free_flow_speed=60, jam_density=120
            )
            raised = False
        except ValueError:
            raised = True
        assert raised

        # attach offramp similarly
        of = net.add_offramp(
            0,
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
            split_ratio=0.3,
        )
        assert isinstance(of, Offramp)
        assert net.get_offramp(0) is of

        try:
            net.add_offramp(
                0,
                lanes=1,
                lane_capacity=1400,
                free_flow_speed=60,
                jam_density=120,
                split_ratio=0.3,
            )
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_get_and_remove_ramps(self):
        net = Network()
        net.add_cell(1.0, 1, 1500, 80, 140)

        # initially none
        assert net.get_onramp(0) is None
        assert net.get_offramp(0) is None

        # add and then remove
        net.add_onramp(
            0, lanes=2, lane_capacity=1600, free_flow_speed=70, jam_density=130
        )
        net.add_offramp(
            0,
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
            split_ratio=0.1,
        )

        assert net.get_onramp(0) is not None
        assert net.get_offramp(0) is not None

        net.remove_onramp(0)
        net.remove_offramp(0)

        assert net.get_onramp(0) is None
        assert net.get_offramp(0) is None

        # removing again should be a no-op (no exception)
        net.remove_onramp(0)
        net.remove_offramp(0)

    def test_index_error_for_invalid_cell_index(self):
        net = Network()
        net.add_cell(1.0, 1, 1500, 80, 140)

        # out of bounds should raise IndexError from list access
        with pytest.raises(IndexError):
            net.add_onramp(
                5, lanes=1, lane_capacity=1400, free_flow_speed=60, jam_density=120
            )

        with pytest.raises(IndexError):
            net.add_offramp(
                5,
                lanes=1,
                lane_capacity=1400,
                free_flow_speed=60,
                jam_density=120,
                split_ratio=0.2,
            )

    def test_network_sizes_and_pointer_integrity(self):
        # build networks of various sizes and check pointers
        for n in (1, 2, 5, 10):
            net = Network()
            for i in range(n):
                net.add_cell(
                    length=float(i + 1),
                    lanes=1 + i,
                    lane_capacity=1500 + i * 100,
                    free_flow_speed=80 + i * 5,
                    jam_density=140 + i * 2,
                )

            assert len(net.cells) == n

    def test_lane_drops(self):
        net = Network()
        c1 = net.add_cell(
            length=2.0,
            lanes=5,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c2 = net.add_cell(
            length=2.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c3 = net.add_cell(
            length=2.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c4 = net.add_cell(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c4 = net.add_cell(
            length=2.0,
            lanes=5,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        assert c1.upcoming_lane_drop == 3  # 5 -> 2 lane drop
        assert c2.upcoming_lane_drop == 0  # no lane drop
        assert c3.upcoming_lane_drop == 0  # -> 1 lane increase
        assert c4.upcoming_lane_drop == 0  # -> default: no lane drop
