import numpy as np

from traffic_flow_models import CTM, Network


class TestCTM:
    def test_cell_update_basic(self):
        model = CTM()
        # simple values chosen to allow manual verification
        next_density, speed = model.cell_update(
            cell_lanes=2,
            cell_length=0.5,
            density=20.0,
            upstream_flow=100.0,
            cell_flow=80.0,
            onramp_flow=10.0,
            offramp_flow=5.0,
            dt=0.25,
        )

        # compute expected next density directly
        expected_next = 20.0 + 0.25 * (100.0 + 10.0 - 5.0 - 80.0) / (0.5 * 2)
        expected_speed = 80.0 / (2 * 20.0)

        assert np.isclose(next_density, expected_next)
        assert np.isclose(speed, expected_speed)

    def test_step_single_cell_no_onramp(self):
        # build a minimal network with one mainline cell and no ramps
        net = Network()
        net.add_cell(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        model = CTM()

        previous_density = np.array([10.0], dtype=np.float64)
        mainline_demand = 500.0
        input_queue = 0
        onramp_demand = np.array([0.0], dtype=np.float64)
        onramp_queue = np.array([0], dtype=np.float64)
        previous_onramp_flow = np.array([0.0], dtype=np.float64)
        dt = 0.25

        flow, density, speed, input_flow, _, onramp_flow, _, next_onramp_queue = (
            model.step(
                network=net,
                density=previous_density,
                speed=np.array([0.0], dtype=np.float64),  # ignored for CTM
                flow=np.array([0.0], dtype=np.float64),  # ignored for CTM
                mainline_demand=mainline_demand,
                input_queue=input_queue,
                onramp_demand=onramp_demand,
                onramp_queue=onramp_queue,
                onramp_flow=previous_onramp_flow,
                dt=dt,
            )
        )

        # shapes and basic invariants
        assert flow.shape == (1,)
        assert density.shape == (1,)
        assert speed.shape == (1,)

        # no onramp attached -> onramp_flow should be zero and queues unchanged
        assert onramp_flow[0] == 0.0
        assert next_onramp_queue[0] == 0

        # the density returned should match a direct call to cell_update for the same arguments
        next_density_direct, speed_direct = model.cell_update(
            cell_lanes=net.cells[0].lanes,
            cell_length=net.cells[0].length,
            density=previous_density[0],
            upstream_flow=input_flow,
            cell_flow=flow[0],
            onramp_flow=0.0,
            offramp_flow=0.0,
            dt=dt,
        )

        assert np.isclose(density[0], next_density_direct)
        assert np.isclose(speed[0], speed_direct)

    def test_critical_density_and_backward_wave(self):
        # create a simple network and CTM instance
        net = Network()
        # choose parameters that allow easy manual verification
        net.add_cell(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        model = CTM()
        cell = net.cells[0]

        # expected critical density: Qc_lane / vf
        expected_rho_cr = cell.Qc_lane / cell.vf
        computed_rho_cr = model.critical_density(cell=cell)
        assert np.isclose(computed_rho_cr, expected_rho_cr)

        # expected backward wave speed: Qc / (rho_jam - rho_cr)
        expected_w = cell.Qc / (cell.rho_jam - expected_rho_cr)
        computed_w = model.backward_wave_speed(cell=cell)
        assert np.isclose(computed_w, expected_w)
