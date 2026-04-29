import pytest

from traffic_flow_models import MotorwayLink, Cell, Node


class TestMotorwayLink:
    def test_add_cell_single(self):
        link = MotorwayLink(length=2.0, lanes=3)
        max_vf = 100.0

        # partition into a single cell
        link.partition_link(max_vf=max_vf, preferred_cell_size=2.0, dt=0.001)
        l = link.get_cell(0)

        assert isinstance(l, Cell)
        assert l.length == 2.0
        assert link.lanes == 3

        # verify linked list: single cell has no upstream/downstream
        assert l.upstream is None
        assert l.downstream is None
        assert link.first_cell() is l
        assert link.last_cell() is l
        assert len(link) == 1

    def test_add_cell_chaining_multiple(self):
        link = MotorwayLink(length=6.0, lanes=1)
        max_vf = 80.0

        # partition into three roughly equal cells
        link.partition_link(max_vf=max_vf, preferred_cell_size=2.0, dt=0.001)
        a = link.get_cell(0)
        b = link.get_cell(1)
        c = link.get_cell(2)

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

    def test_index_error_for_invalid_cell_index(self):
        link = MotorwayLink(length=1.0, lanes=1)
        link.partition_link(max_vf=80.0, preferred_cell_size=1.0, dt=0.001)

        # out of bounds should raise IndexError from list access
        with pytest.raises(IndexError):
            link.get_cell(5)

    def test_motorway_link_sizes_and_pointer_integrity(self):
        # build motorway link of various sizes and check linked list pointers
        for n in (1, 2, 5, 10):
            link = MotorwayLink(length=10.0, lanes=1)
            preferred = round(link.length / n, 3)
            link.partition_link(max_vf=80.0, preferred_cell_size=preferred, dt=0.001)
            cells = list(link)

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

    def test_node_connection_sets_node_ids(self):
        link = MotorwayLink(length=1.0, lanes=1)

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
