import numpy as np

from traffic_flow_models import METANET, MotorwayLink


class TestMETANET:
    def test_cell_update_basic(self):
        # choose parameters that eliminate secondary effects so the
        # METANET speed update reduces to a no-op (stationary speed)
        model = METANET(tau=1.0, nu=0.0, kappa=0.0, delta=0.0, phi=0.0, alpha=1.0)

        link = MotorwayLink()
        cell = link.add_cell(
            length=0.5,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        previous_density = 20.0
        # compute stationary speed at the previous density and use it
        # as the previous and upstream speed so convective and relaxation
        # terms vanish (given nu=delta=0 and previous_speed==stationary)
        previous_speed = model.stationary_velocity(cell=cell, density=previous_density)

        upstream_flow = 100.0
        previous_flow = 80.0
        onramp_flow = 10.0
        offramp_flow = 5.0
        dt = 0.25

        next_density, next_speed, next_flow = model.cell_update(
            cell=cell,
            upstream_flow=upstream_flow,
            previous_flow=previous_flow,
            onramp_flow=onramp_flow,
            offramp_flow=offramp_flow,
            previous_density=previous_density,
            downstream_density=previous_density,  # no density gradient
            upstream_speed=previous_speed,
            previous_speed=previous_speed,
            dt=dt,
        )

        expected_next = previous_density + dt * (
            upstream_flow + onramp_flow - offramp_flow - previous_flow
        ) / (cell.length * cell.lanes)

        # since we chose previous_speed == stationary_velocity and nu=delta=phi=0,
        # the speed should remain unchanged (and non-negative)
        expected_speed = previous_speed

        assert np.isclose(next_density, expected_next)
        assert np.isclose(next_speed, expected_speed)

        # flow is density * speed * lanes
        assert np.isclose(next_flow, next_density * next_speed * cell.lanes)

    def test_step_two_cells_no_onramp(self):
        # METANET.step implementation expects at least two cells when
        # evaluating the first cell's downstream density. Test a small
        # two-cell motorway link without ramps to validate consistency between
        # step() and cell_update() for the first cell.
        link = MotorwayLink()
        link.add_cell(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link.add_cell(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        model = METANET(tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.0, alpha=1.0)

        previous_density = np.array([10.0, 12.0], dtype=np.float64)
        previous_speed = np.array([0.0, 0.0], dtype=np.float64)
        previous_flow = np.array([0.0, 0.0], dtype=np.float64)

        mainline_demand = 500.0
        input_queue = 0
        onramp_demand = np.array([0.0, 0.0], dtype=np.float64)
        onramp_queue = np.array([0, 0], dtype=np.float64)
        previous_onramp_flow = np.array([0.0, 0.0], dtype=np.float64)
        dt = 0.25

        (
            flow,
            density,
            speed,
            input_flow,
            _,
            onramp_flow,
            _,
            next_onramp_queue,
        ) = model.step(
            link=link,
            density=previous_density,
            speed=previous_speed,
            flow=previous_flow,
            mainline_demand=mainline_demand,
            input_queue=input_queue,
            input_flow=mainline_demand,
            onramp_demand=onramp_demand,
            onramp_queue=onramp_queue,
            onramp_flow=previous_onramp_flow,
            offramp_flow=np.array([0.0, 0.0], dtype=np.float64),
            dt=dt,
        )

        assert flow.shape == (2,)
        assert density.shape == (2,)
        assert speed.shape == (2,)

        # no onramps -> onramp_flow should be zero and queues unchanged
        assert onramp_flow[0] == 0.0
        assert next_onramp_queue[0] == 0

        # verify that the first cell's density & speed match a direct
        # call to cell_update using the same values
        first_cell = link.get_cell(0)
        next_density_direct, next_speed_direct, _ = model.cell_update(
            cell=first_cell,
            upstream_flow=input_flow,
            previous_flow=previous_flow[0],
            onramp_flow=0.0,
            offramp_flow=0.0,
            previous_density=previous_density[0],
            downstream_density=previous_density[1],
            upstream_speed=previous_speed[0],
            previous_speed=previous_speed[0],
            dt=dt,
        )

        assert np.isclose(density[0], next_density_direct)
        assert np.isclose(speed[0], next_speed_direct)

    def test_critical_density_and_backward_wave(self):
        link = MotorwayLink()
        link.add_cell(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        model = METANET(tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.0, alpha=2.0)
        cell = link.get_cell(0)

        expected_rho_cr = cell.Qc_lane / (cell.vf * np.exp(-1 / model.alpha))
        computed_rho_cr = model.critical_density(cell=cell)
        assert np.isclose(computed_rho_cr, expected_rho_cr)

        expected_w = cell.Qc / (cell.rho_jam - expected_rho_cr)
        computed_w = model.backward_wave_speed(cell=cell)
        assert np.isclose(computed_w, expected_w)

    def test_lane_drop_deceleration(self):
        # verify that an upcoming lane drop reduces the computed speed
        # (phi term should apply additional deceleration)
        # build a two-cell motorway link with an upstream cell having an upcoming drop
        link = MotorwayLink()
        link.add_cell(
            length=1.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        # downstream cell has fewer lanes -> induces upcoming_lane_drop on previous
        link.add_cell(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        # model with no phi
        model_no_phi = METANET(
            tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.0, alpha=1.0
        )
        # model with phi > 0
        model_with_phi = METANET(
            tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.5, alpha=1.0
        )

        cell = link.get_cell(0)
        previous_density = 20.0
        previous_speed = 30.0
        upstream_speed = 30.0
        upstream_flow = 100.0
        previous_flow = 80.0
        onramp_flow = 0.0
        offramp_flow = 0.0
        dt = 0.1

        _, speed_no_phi, _ = model_no_phi.cell_update(
            cell=cell,
            upstream_flow=upstream_flow,
            previous_flow=previous_flow,
            onramp_flow=onramp_flow,
            offramp_flow=offramp_flow,
            previous_density=previous_density,
            downstream_density=previous_density,
            upstream_speed=upstream_speed,
            previous_speed=previous_speed,
            dt=dt,
        )

        _, speed_with_phi, _ = model_with_phi.cell_update(
            cell=cell,
            upstream_flow=upstream_flow,
            previous_flow=previous_flow,
            onramp_flow=onramp_flow,
            offramp_flow=offramp_flow,
            previous_density=previous_density,
            downstream_density=previous_density,
            upstream_speed=upstream_speed,
            previous_speed=previous_speed,
            dt=dt,
        )

        assert speed_with_phi <= speed_no_phi
