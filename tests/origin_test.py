from traffic_flow_models import Origin, Cell, Node
import pytest


class TestOrigin:
    def test_init_assigns_attributes(self):
        o = Origin(
            id=None, lanes=2, lane_capacity=1200, free_flow_speed=90, jam_density=150
        )
        assert o.lanes == 2
        assert o.Qc_lane == 1200
        assert o.Qc == 2400
        assert o.vf == 90
        assert o.rho_jam == 150
        assert isinstance(o.id, str) and len(o.id) > 0

        # origin should not have destination node id until connected
        assert getattr(o, "destination_node_id", None) is None

        # connect to a node and verify id set
        n = Node(id="n-origin")
        n.add_incoming(o)
        assert o.destination_node_id == n.id

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            Origin(
                id=None,
                lanes=0,
                lane_capacity=1200,
                free_flow_speed=90,
                jam_density=150,
            )

        with pytest.raises(ValueError):
            Origin(
                id=None, lanes=1, lane_capacity=0, free_flow_speed=90, jam_density=150
            )

        with pytest.raises(ValueError):
            Origin(
                id=None, lanes=1, lane_capacity=1200, free_flow_speed=0, jam_density=150
            )

        with pytest.raises(ValueError):
            Origin(
                id=None, lanes=1, lane_capacity=1200, free_flow_speed=90, jam_density=0
            )
