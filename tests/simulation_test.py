import numpy as np

from traffic_flow_models import (
    CTM,
    Network,
    Node,
    MotorwayLink,
    Origin,
    Destination,
    Simulation,
)


class TestSimulationComputeMetrics:
    def test_compute_metrics_matches_hand_computed_values(self):
        # single-cell mainline link: Origin -> MotorwayLink -> Destination
        link = MotorwayLink(
            length=1.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link.partition_link(preferred_cell_size=1.0, dt=0.001)
        assert len(link) == 1
        cell = link.get_cell(0)

        origin = Origin()
        destination = Destination()
        node1 = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        sim = Simulation(network, CTM())

        flow_val = 500.0  # veh/h
        density_val = 10.0  # veh/km/lane
        speed_val = 50.0  # km/h
        origin_queue_val = 3.0  # veh

        flow_dict = {
            origin.id: np.array([flow_val]),
            link.id: np.array([flow_val]),
            destination.id: np.array([flow_val]),
        }
        density_dict = {link.id: np.array([density_val])}
        speed_dict = {link.id: np.array([speed_val])}
        origin_queue_dict = {origin.id: origin_queue_val}

        x0, *_ = network.network_dict_to_state_vec(
            flow_dict, density_dict, speed_dict, origin_queue_dict, {}, {}
        )
        # compute_metrics only reads states[:, 0 .. timesteps-2], so a second
        # (unused) column is enough to exercise the full loop range.
        states = np.column_stack([x0, x0])
        dt = 0.01
        timesteps = 2

        VKT, VHT, avg_speed = sim.compute_metrics(
            states=states, dt=dt, timesteps=timesteps
        )

        expected_VKT = cell.length * dt * flow_val
        expected_VHT_queue = dt * origin_queue_val
        expected_VHT_mainline = cell.length * dt * density_val * link.lanes
        expected_VHT = expected_VHT_queue + expected_VHT_mainline
        expected_avg_speed = expected_VKT / expected_VHT

        assert np.isclose(VKT, expected_VKT)
        assert np.isclose(VHT, expected_VHT)
        assert np.isclose(avg_speed, expected_avg_speed)

        # ignore_queues=True should drop the queue contribution to VHT
        # (and therefore change the derived average speed) but leave VKT untouched
        VKT_no_q, VHT_no_q, avg_speed_no_q = sim.compute_metrics(
            states=states, dt=dt, timesteps=timesteps, ignore_queues=True
        )
        assert np.isclose(VKT_no_q, expected_VKT)
        assert np.isclose(VHT_no_q, expected_VHT_mainline)
        assert np.isclose(avg_speed_no_q, expected_VKT / expected_VHT_mainline)

    def test_compute_metrics_returns_zero_avg_speed_when_no_travel(self):
        link = MotorwayLink(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link.partition_link(preferred_cell_size=1.0, dt=0.001)

        origin = Origin()
        destination = Destination()
        node1 = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        sim = Simulation(network, CTM())

        # empty network: zero flow, density, speed, and queues everywhere
        flow_dict = {
            origin.id: np.array([0.0]),
            link.id: np.array([0.0]),
            destination.id: np.array([0.0]),
        }
        density_dict = {link.id: np.array([0.0])}
        speed_dict = {link.id: np.array([0.0])}

        x0, *_ = network.network_dict_to_state_vec(
            flow_dict, density_dict, speed_dict, {origin.id: 0.0}, {}, {}
        )
        states = np.column_stack([x0, x0])

        VKT, VHT, avg_speed = sim.compute_metrics(states=states, dt=0.01, timesteps=2)

        assert VKT == 0.0
        assert VHT == 0.0
        # avg_speed falls back to 0.0 instead of dividing by zero when VHT == 0
        assert avg_speed == 0.0
