import casadi
import numpy as np

from traffic_flow_models import (
    METANET,
    Network,
    Node,
    MotorwayLink,
    Origin,
    Onramp,
    Destination,
)


class TestMETANET:
    def test_cell_update_basic(self):
        # choose parameters that eliminate secondary effects so the
        # METANET speed update reduces to a no-op (stationary speed)
        model = METANET(tau=1.0, nu=0.0, kappa=0.0, delta=0.0, phi=0.0, alpha=1.0)
        link = MotorwayLink(
            length=0.5,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        # partition link into one cell using a small dt for CFL condition
        link.partition_link(preferred_cell_size=0.5, dt=0.001)
        cell = link.get_cell(0)

        previous_density = 20.0
        # compute stationary speed at the previous density and use it
        # as the previous and upstream speed so convective and relaxation
        # terms vanish (given nu=delta=0 and previous_speed==stationary)
        previous_speed = model.stationary_velocity(
            lane_capacity=link.lane_capacity,
            free_flow_speed=link.vf,
            density=previous_density,
        )

        # CasADi type stubs are incorrect - sym() does accept string as first arg
        upstream_flow_sx = casadi.SX.sym("upstream_flow", 1, 1)  # type: ignore
        previous_flow_sx = casadi.SX.sym("previous_flow", 1, 1)  # type: ignore
        previous_density_sx = casadi.SX.sym("previous_density", 1, 1)  # type: ignore
        downstream_density_sx = casadi.SX.sym("downstream_density", 1, 1)  # type: ignore
        upstream_speed_sx = casadi.SX.sym("upstream_speed", 1, 1)  # type: ignore
        previous_speed_sx = casadi.SX.sym("previous_speed", 1, 1)  # type: ignore

        upstream_flow = 100.0
        previous_flow = 80.0
        dt = 0.25

        next_density_sx, next_speed_sx, next_flow_sx = model.cell_update(
            link=link,
            cell=cell,
            upstream_flow=upstream_flow_sx,
            previous_flow=previous_flow_sx,
            previous_density=previous_density_sx,
            downstream_density=downstream_density_sx,  # no density gradient
            upstream_speed=upstream_speed_sx,
            previous_speed=previous_speed_sx,
            dt=dt,
        )

        # create CasADi function for evaluation
        cell_update_fn = casadi.Function(
            "cell_update",
            [
                upstream_flow_sx,
                previous_flow_sx,
                previous_density_sx,
                downstream_density_sx,
                upstream_speed_sx,
                previous_speed_sx,
            ],
            [next_density_sx, next_speed_sx, next_flow_sx],
        )

        res = cell_update_fn(
            upstream_flow,
            previous_flow,
            previous_density,
            previous_density,  # no density gradient
            previous_speed,
            previous_speed,
        )
        next_density = np.array(res).flatten()[0]
        next_speed = np.array(res).flatten()[1]
        next_flow = np.array(res).flatten()[2]

        expected_density = previous_density + dt * (upstream_flow - previous_flow) / (
            cell.length * link.lanes
        )

        # since we chose previous_speed == stationary_velocity and nu=delta=phi=0,
        # the speed should remain unchanged (and non-negative)
        expected_speed = previous_speed
        assert np.isclose(next_density, expected_density)
        assert np.isclose(next_speed, expected_speed)

        # flow is density * speed * lanes
        assert np.isclose(next_flow, next_density * next_speed * link.lanes)

    def test_step_two_cells_no_onramp(self):
        # METANET.step implementation expects at least two cells when
        # evaluating the first cell's downstream density. Test a small
        # two-cell motorway link without ramps to validate consistency between
        # step() and cell_update() for the first cell.
        link = MotorwayLink(
            length=2.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        dt = 0.005
        link.partition_link(preferred_cell_size=1.5, dt=dt)

        origin = Origin()
        onramp = Onramp(
            lanes=1, lane_capacity=1000, free_flow_speed=60, jam_density=100
        )
        destination = Destination()

        node1 = Node(incoming=[origin, onramp], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        model = METANET(tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.0, alpha=1.0)
        previous_density = np.array([10.0, 12.0, 12.0], dtype=np.float64)
        previous_speed = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        previous_flow = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        mainline_demand = lambda t: 500.0
        onramp_demand = lambda t: 0.0
        onramp_queue = np.array([0, 0], dtype=np.float64)

        time_array, states, disturbances = network.simulate(
            duration=dt,
            dt=dt,
            model=model,
            origin_demands={origin.id: mainline_demand},
            onramp_demands={onramp.id: onramp_demand},
            initial_flows={
                link.id: previous_flow,
                onramp.id: 0.0,
                origin.id: mainline_demand(0),
                destination.id: 0.0,
            },
            initial_densities={
                link.id: previous_density,
                onramp.id: np.array([0.0]),
            },
            initial_speeds={link.id: previous_speed, onramp.id: np.array([0.0])},
            turning_rates={},
            destination_boundary_conditions={destination.id: lambda t: 0.0},
            plot_results=False,
        )

        # use the final simulated state (works even if only one timestep)
        (
            flow,
            density,
            speed,
            _,
            onramp_queue,
            _,
        ) = network.state_vec_to_network_dict(states[:, -1])

        # check that flow is not unbound
        assert flow[link.id].shape == (3,)
        assert density[link.id].shape == (3,)
        assert speed[link.id].shape == (3,)

        # no onramps -> onramp_flow should be zero and queues unchanged
        assert flow[onramp.id][0] == 0.0
        assert onramp_queue[onramp.id] == 0

        # verify that the first cell's density & speed match a direct
        # call to cell_update using the same values (wrap in CasADi function)
        first_cell = link.get_cell(0)

        upstream_flow_sx = casadi.SX.sym("upstream_flow", 1, 1)  # type: ignore
        previous_flow_sx = casadi.SX.sym("previous_flow", 1, 1)  # type: ignore
        previous_density_sx = casadi.SX.sym("previous_density", 1, 1)  # type: ignore
        downstream_density_sx = casadi.SX.sym("downstream_density", 1, 1)  # type: ignore
        upstream_speed_sx = casadi.SX.sym("upstream_speed", 1, 1)  # type: ignore
        previous_speed_sx = casadi.SX.sym("previous_speed", 1, 1)  # type: ignore

        next_density_sx, next_speed_sx, next_flow_sx = model.cell_update(
            link=link,
            cell=first_cell,
            upstream_flow=upstream_flow_sx,
            previous_flow=previous_flow_sx,
            previous_density=previous_density_sx,
            downstream_density=downstream_density_sx,
            upstream_speed=upstream_speed_sx,
            previous_speed=previous_speed_sx,
            dt=dt,
        )

        cell_update_fn = casadi.Function(
            "cell_update_first",
            [
                upstream_flow_sx,
                previous_flow_sx,
                previous_density_sx,
                downstream_density_sx,
                upstream_speed_sx,
                previous_speed_sx,
            ],
            [next_density_sx, next_speed_sx, next_flow_sx],
        )

        res = cell_update_fn(
            flow[origin.id][0],
            previous_flow[0],
            previous_density[0],
            previous_density[1],
            previous_speed[0],
            previous_speed[0],
        )
        next_density_direct = np.array(res).flatten()[0]
        next_speed_direct = np.array(res).flatten()[1]

        assert np.isclose(density[link.id][0], next_density_direct)
        assert np.isclose(speed[link.id][0], next_speed_direct)

    def test_critical_density_and_backward_wave(self):
        link = MotorwayLink(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link.partition_link(preferred_cell_size=1.0, dt=0.001)

        model = METANET(tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.0, alpha=2.0)

        expected_rho_cr = link.lane_capacity / (link.vf * np.exp(-1 / model.alpha))
        computed_rho_cr = model.critical_density(
            lane_capacity=link.lane_capacity, free_flow_speed=link.vf
        )
        assert np.isclose(computed_rho_cr, expected_rho_cr)

        expected_w = (link.lane_capacity * link.lanes) / (
            link.rho_jam - expected_rho_cr
        )
        computed_w = model.backward_wave_speed(
            capacity=link.lane_capacity * link.lanes,
            lane_capacity=link.lane_capacity,
            jam_density=link.rho_jam,
            free_flow_speed=link.vf,
        )
        assert np.isclose(computed_w, expected_w)

    def test_lane_drop_deceleration(self):
        # verify that an upcoming lane drop reduces the computed speed
        # (phi term should apply additional deceleration)
        # build a two-link motorway link with an upstream link having an upcoming drop
        dt = 0.001
        link = MotorwayLink(
            length=2.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        link2 = MotorwayLink(
            length=2.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )

        origin = Origin()
        destination = Destination()

        node = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[link2])
        node3 = Node(incoming=[link2], outgoing=[destination])
        network = Network(nodes=[node, node2, node3])

        # run full simulation with both models and compare upstream link speeds
        model_no_phi = METANET(
            tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.0, alpha=1.0
        )
        model_with_phi = METANET(
            tau=1.0, nu=0.0, kappa=0.1, delta=0.0, phi=0.5, alpha=1.0
        )

        # initial states for link (2 cells) and downstream link2 (1 cell)
        init_density_link = np.array([20.0, 20.0], dtype=np.float64)
        init_speed_link = np.array([30.0, 30.0], dtype=np.float64)
        init_flow_link = np.array([80.0, 80.0], dtype=np.float64)

        init_density_link2 = np.array([20.0, 20.0], dtype=np.float64)
        init_speed_link2 = np.array([30.0, 30.0], dtype=np.float64)
        init_flow_link2 = np.array([0.0, 0.0], dtype=np.float64)

        mainline_demand = lambda t: 100.0

        def run_model(model):
            _, states, _ = network.simulate(
                duration=dt,
                dt=dt,
                model=model,
                origin_demands={origin.id: mainline_demand},
                onramp_demands={},
                initial_flows={
                    link.id: init_flow_link,
                    link2.id: init_flow_link2,
                    origin.id: mainline_demand(0),
                    destination.id: 0.0,
                },
                initial_densities={
                    link.id: init_density_link,
                    link2.id: init_density_link2,
                },
                initial_speeds={link.id: init_speed_link, link2.id: init_speed_link2},
                turning_rates={},
                destination_boundary_conditions={destination.id: lambda t: 0.0},
                preferred_cell_size=1.0,
                plot_results=False,
            )
            _, _, speed, _, _, _ = network.state_vec_to_network_dict(states[:, -1])
            return speed[link.id]

        speed_no_phi = run_model(model_no_phi)
        speed_with_phi = run_model(model_with_phi)

        # verify that the last cell in the upstream link has an upcoming lane drop set (the first cell not through)
        # lane drop parameter should be set automatically during simulation of network
        cell1 = link.get_cell(0)
        cell2 = link.get_cell(1)
        assert link._cell_count == 2
        assert cell1.upcoming_lane_drop == 0
        assert cell2.upcoming_lane_drop == 1

        # verify that the speeds for the last cell of the upstream link are lower when phi>0
        assert speed_with_phi.shape == speed_no_phi.shape
        assert speed_with_phi[-1] <= speed_no_phi[-1]
