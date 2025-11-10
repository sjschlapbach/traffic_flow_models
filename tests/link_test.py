from traffic_flow_models import Link


class TestLink:
    def test_init_assigns_attributes(self):
        link = Link(
            length=2.5,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        assert link.length == 2.5
        assert link.lanes == 3
        assert link.lane_capacity == 2000
        assert link.free_flow_speed == 100
        assert link.jam_density == 150
        assert link.downstream_link is None
        assert link.upstream_link is None

    def test_next_link_assignment_via_constructor(self):
        downstream = Link(
            length=1.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )
        upstream = Link(
            length=3.0,
            lanes=1,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            downstream_link=downstream,
        )
        downstream.upstream_link = upstream

        assert upstream.downstream_link is downstream
        assert upstream.downstream_link is not None
        assert upstream.downstream_link.length == 1.0
        assert downstream.upstream_link is upstream

    def test_mutating_downstream_reflects_in_upstream_reference(self):
        a = Link(
            length=1.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )
        b = Link(
            length=2.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=160
        )

        a.downstream_link = b
        b.upstream_link = a
        b.length = 2.5

        # downstream reference should see updated downstream object
        assert a.downstream_link.length == 2.5

        # links between network links are set correctly
        assert b.upstream_link is a
        assert a.downstream_link is b
