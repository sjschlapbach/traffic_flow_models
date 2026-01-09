import pytest

from traffic_flow_models import MotorwayLink, Cell, Onramp, Offramp, Node


class TestMotorwayLink:
    def test_add_cell_single(self):
        link = MotorwayLink()
        l = link.add_cell(
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

        # verify linked list: single cell has no upstream/downstream
        assert l.upstream is None
        assert l.downstream is None
        assert link.first_cell() is l
        assert link.last_cell() is l
        assert len(link) == 1

    def test_add_cell_chaining_multiple(self):
        link = MotorwayLink()
        a = link.add_cell(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        b = link.add_cell(
            length=2.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )
        c = link.add_cell(
            length=3.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=170,
        )

        # verify linked list structure: a -> b -> c
        assert a.upstream is None
        assert a.downstream is b
        assert b.upstream is a
        assert b.downstream is c
        assert c.upstream is b
        assert c.downstream is None

        # verify motorway link helpers
        assert link.first_cell() is a
        assert link.last_cell() is c
        assert len(link) == 3

        # verify iteration order
        cells_list = list(link)
        assert cells_list == [a, b, c]

        # verify get_cell by index
        assert link.get_cell(0) is a
        assert link.get_cell(1) is b
        assert link.get_cell(2) is c

    def test_add_cell_with_ramps_instances(self):
        link = MotorwayLink()
        on = Onramp(lanes=2, lane_capacity=1600, free_flow_speed=70, jam_density=130)
        off = Offramp(
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
        )

        l = link.add_cell(
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
        link = MotorwayLink()

        # passing non-Onramp object should raise TypeError
        try:
            link.add_cell(1.0, 1, 1500, 80, 140, onramp={})  # type: ignore
            raised = False
        except TypeError:
            raised = True
        assert raised

        try:
            link.add_cell(1.0, 1, 1500, 80, 140, offramp=123)  # type: ignore
            raised = False
        except TypeError:
            raised = True
        assert raised

    def test_add_onramp_and_offramp_methods_and_duplicates(self):
        link = MotorwayLink()
        link.add_cell(1.0, 1, 1500, 80, 140)

        # attach via method
        r = link.add_onramp(
            0, lanes=2, lane_capacity=1600, free_flow_speed=70, jam_density=130
        )
        assert isinstance(r, Onramp)
        assert link.get_onramp(0) is r

        # duplicate attach should raise ValueError
        try:
            link.add_onramp(
                0, lanes=1, lane_capacity=1400, free_flow_speed=60, jam_density=120
            )
            raised = False
        except ValueError:
            raised = True
        assert raised

        # attach offramp similarly
        of = link.add_offramp(
            0,
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
        )
        assert isinstance(of, Offramp)
        assert link.get_offramp(0) is of

        try:
            link.add_offramp(
                0,
                lanes=1,
                lane_capacity=1400,
                free_flow_speed=60,
                jam_density=120,
            )
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_get_and_remove_ramps(self):
        link = MotorwayLink()
        link.add_cell(1.0, 1, 1500, 80, 140)

        # initially none
        assert link.get_onramp(0) is None
        assert link.get_offramp(0) is None

        # add and then remove
        link.add_onramp(
            0, lanes=2, lane_capacity=1600, free_flow_speed=70, jam_density=130
        )
        link.add_offramp(
            0,
            lanes=1,
            lane_capacity=1400,
            free_flow_speed=60,
            jam_density=120,
        )

        assert link.get_onramp(0) is not None
        assert link.get_offramp(0) is not None

        link.remove_onramp(0)
        link.remove_offramp(0)

        assert link.get_onramp(0) is None
        assert link.get_offramp(0) is None

        # removing again should be a no-op (no exception)
        link.remove_onramp(0)
        link.remove_offramp(0)

    def test_index_error_for_invalid_cell_index(self):
        link = MotorwayLink()
        link.add_cell(1.0, 1, 1500, 80, 140)

        # out of bounds should raise IndexError from list access
        with pytest.raises(IndexError):
            link.add_onramp(
                5, lanes=1, lane_capacity=1400, free_flow_speed=60, jam_density=120
            )

        with pytest.raises(IndexError):
            link.add_offramp(
                5,
                lanes=1,
                lane_capacity=1400,
                free_flow_speed=60,
                jam_density=120,
            )

    def test_motorway_link_sizes_and_pointer_integrity(self):
        # build motorway link of various sizes and check linked list pointers
        for n in (1, 2, 5, 10):
            link = MotorwayLink()
            cells = []
            for i in range(n):
                cell = link.add_cell(
                    length=float(i + 1),
                    lanes=1 + i,
                    lane_capacity=1500 + i * 100,
                    free_flow_speed=80 + i * 5,
                    jam_density=140 + i * 2,
                )
                cells.append(cell)

            # verify motorway link size
            assert len(link) == n

            # verify first and last cells
            assert link.first_cell() is cells[0]
            assert link.last_cell() is cells[-1]

            # verify linked list integrity
            for i in range(n):
                cell = cells[i]
                # check upstream pointer
                if i == 0:
                    assert cell.upstream is None
                else:
                    assert cell.upstream is cells[i - 1]

                # check downstream pointer
                if i == n - 1:
                    assert cell.downstream is None
                else:
                    assert cell.downstream is cells[i + 1]

            # verify iteration produces correct order
            assert list(link) == cells

            # verify get_cell returns correct cells
            for i in range(n):
                assert link.get_cell(i) is cells[i]

    def test_lane_drops(self):
        link = MotorwayLink()
        c1 = link.add_cell(
            length=2.0,
            lanes=5,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c2 = link.add_cell(
            length=2.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c3 = link.add_cell(
            length=2.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c4 = link.add_cell(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        c4 = link.add_cell(
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

    def test_node_connection_sets_node_ids(self):
        link = MotorwayLink()

        # precondition: node ids unset
        assert getattr(link, "origin_node_id", None) is None
        assert getattr(link, "destination_node_id", None) is None
        n_in = Node(id="n-in")
        n_out = Node(id="n-out")

        # connecting as incoming should set destination_node_id
        n_in.add_incoming(link)
        assert link.destination_node_id == n_in.id

        # connecting as outgoing should set origin_node_id
        n_out.add_outgoing(link)
        assert link.origin_node_id == n_out.id
