import numpy as np

from traffic_flow_models import CTM, Network


class TestCTM:
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

        prev_flow = np.array([0.0], dtype=np.float64)
        prev_input_flow = mainline_demand

        flow, density, speed, input_flow, _, onramp_flow, _, next_onramp_queue = (
            model.step(
                network=net,
                density=previous_density,
                speed=np.array([0.0], dtype=np.float64),  # ignored for CTM
                flow=prev_flow,  # previous outflow
                mainline_demand=mainline_demand,
                input_queue=input_queue,
                input_flow=prev_input_flow,
                onramp_demand=onramp_demand,
                onramp_queue=onramp_queue,
                onramp_flow=previous_onramp_flow,
                offramp_flow=np.array([0.0], dtype=np.float64),
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

        # compute expected next density directly using conservation
        # using the *previous* timestep flows
        next_density_direct = previous_density[0] + dt * (
            prev_input_flow + previous_onramp_flow[0] - prev_flow[0] - 0.0
        ) / (net.cells[0].length * net.cells[0].lanes)

        speed_direct = (
            flow[0] / (net.cells[0].lanes * next_density_direct)
            if next_density_direct > 0
            else net.cells[0].vf
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
