from typing import cast
import casadi
import numpy as np

from traffic_flow_models import (
    METANET,
    METANETParams,
    METANETSymbolicParams,
    Network,
    Node,
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
    Simulation,
)


class TestMETANET:
    def test_cell_update_basic(self):
        # choose parameters that eliminate secondary effects so the
        # METANET speed update reduces to a no-op (stationary speed)
        model = METANET()
        link = MotorwayLink(length=0.5, lanes=2)
        model_params: METANETParams = {
            "vf": 100.0,
            "qc_lane": 2000.0,
            "rho_jam": 150.0,
            "tau": 1.0,
            "nu": 0.0,
            "kappa": 0.0,
            "delta": 0.0,
            "phi": 0.0,
            "alpha": {link.id: 1.0},
        }

        # partition link into one cell using a small dt for CFL condition
        link.partition_link(
            max_vf=model_params["vf"], preferred_cell_size=0.5, dt=0.001
        )
        cell = link.get_cell(0)

        previous_density = 20.0
        # compute stationary speed at the previous density and use it
        # as the previous and upstream speed so convective and relaxation
        # terms vanish (given nu=delta=0 and previous_speed==stationary)
        previous_speed = model.stationary_velocity(
            params=model_params,
            link_id=link.id,
            density=previous_density,
        )

        # CasADi type stubs are incorrect - sym() does accept string as first arg
        upstream_flow_sx = casadi.SX.sym("upstream_flow", 1, 1)  # type: ignore
        previous_flow_sx = casadi.SX.sym("previous_flow", 1, 1)  # type: ignore
        previous_density_sx = casadi.SX.sym("previous_density", 1, 1)  # type: ignore
        downstream_density_sx = casadi.SX.sym("downstream_density", 1, 1)  # type: ignore
        upstream_speed_sx = casadi.SX.sym("upstream_speed", 1, 1)  # type: ignore
        upstream_onramp_inflows_sx = casadi.SX.sym("upstream_onramp_inflows", 1, 1)  # type: ignore
        previous_speed_sx = casadi.SX.sym("previous_speed", 1, 1)  # type: ignore

        upstream_flow = 100.0
        previous_flow = 80.0
        dt = 0.25

        network = Network(nodes=[Node(incoming=[], outgoing=[link])])
        sx_params = model.set_up_symbolic_model_params(network=network)
        symbolic_params = cast(
            METANETSymbolicParams,
            model.model_params_vec_to_dict(network=network, model_params_vec=sx_params),
        )
        # numeric parameter vector for function evaluation
        numeric_params_vec = model.model_params_to_vec(
            network=network, model_params=model_params
        )
        next_density_sx, next_speed_sx, next_flow_sx = model.cell_update(
            params=symbolic_params,
            link=link,
            cell=cell,
            upstream_flow=upstream_flow_sx,
            previous_flow=previous_flow_sx,
            previous_density=previous_density_sx,
            downstream_density=downstream_density_sx,  # no density gradient
            upstream_speed=upstream_speed_sx,
            upstream_onramp_inflows=upstream_onramp_inflows_sx,
            previous_speed=previous_speed_sx,
            dt=dt,
        )

        # create CasADi function for evaluation
        cell_update_fn = casadi.Function(
            "cell_update",
            [
                sx_params,
                upstream_flow_sx,
                previous_flow_sx,
                previous_density_sx,
                downstream_density_sx,
                upstream_speed_sx,
                upstream_onramp_inflows_sx,
                previous_speed_sx,
            ],
            [next_density_sx, next_speed_sx, next_flow_sx],
        )

        res = cell_update_fn(
            numeric_params_vec,
            upstream_flow,
            previous_flow,
            previous_density,
            previous_density,  # no density gradient
            previous_speed,
            0.0,  # scenario does not contain an onramp upstream of the link
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
        link = MotorwayLink(length=2.0, lanes=1)
        dt = 0.005
        link.partition_link(max_vf=100.0, preferred_cell_size=1.5, dt=dt)

        origin = Origin()
        ramp_origin = Origin()
        onramp = Onramp(length=0.5, lanes=1)
        destination = Destination()

        node_main = Node(incoming=[origin], outgoing=[link])
        node_ramp = Node(incoming=[ramp_origin], outgoing=[onramp])
        node_merge = Node(incoming=[link, onramp], outgoing=[destination])
        network = Network(nodes=[node_main, node_ramp, node_merge])

        model = METANET()
        model_params: METANETParams = {
            "vf": 100.0,
            "qc_lane": 2000.0,
            "rho_jam": 150.0,
            "tau": 1.0,
            "nu": 0.0,
            "kappa": 0.1,
            "delta": 0.0,
            "phi": 0.0,
            "alpha": {onramp.id: 1.0, link.id: 1.0},
        }
        previous_density = np.array([10.0, 12.0, 12.0], dtype=np.float64)
        previous_speed = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        previous_flow = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        mainline_demand = lambda t: 500.0
        onramp_demand = lambda t: 0.0
        onramp_queue = np.array([0, 0], dtype=np.float64)

        sim = Simulation(network, model, model_params)
        _, states, _ = sim.run(
            duration=dt,
            dt=dt,
            origin_demands={
                origin.id: mainline_demand,
                ramp_origin.id: onramp_demand,
            },
            initial_flows={
                link.id: previous_flow,
                onramp.id: 0.0,
                origin.id: mainline_demand(0),
                ramp_origin.id: onramp_demand(0),
                destination.id: 0.0,
            },
            initial_densities={
                link.id: previous_density,
                onramp.id: np.array([0.0]),
            },
            initial_speeds={link.id: previous_speed, onramp.id: np.array([0.0])},
            turning_rates={},
            destination_flow_bc={destination.id: lambda t: 6000.0},
            destination_density_bc={destination.id: lambda t: 0.0},
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
        upstream_onramp_inflows_sx = casadi.SX.sym("upstream_onramp_inflows", 1, 1)  # type: ignore
        previous_speed_sx = casadi.SX.sym("previous_speed", 1, 1)  # type: ignore

        sx_params = model.set_up_symbolic_model_params(network=network)
        symbolic_params = cast(
            METANETSymbolicParams,
            model.model_params_vec_to_dict(network=network, model_params_vec=sx_params),
        )
        numeric_params_vec = model.model_params_to_vec(
            network=network, model_params=model_params
        )
        next_density_sx, next_speed_sx, next_flow_sx = model.cell_update(
            params=symbolic_params,
            link=link,
            cell=first_cell,
            upstream_flow=upstream_flow_sx,
            previous_flow=previous_flow_sx,
            previous_density=previous_density_sx,
            downstream_density=downstream_density_sx,
            upstream_speed=upstream_speed_sx,
            upstream_onramp_inflows=upstream_onramp_inflows_sx,
            previous_speed=previous_speed_sx,
            dt=dt,
        )

        cell_update_fn = casadi.Function(
            "cell_update_first",
            [
                sx_params,
                upstream_flow_sx,
                previous_flow_sx,
                previous_density_sx,
                downstream_density_sx,
                upstream_speed_sx,
                upstream_onramp_inflows_sx,
                previous_speed_sx,
            ],
            [next_density_sx, next_speed_sx, next_flow_sx],
        )

        res = cell_update_fn(
            numeric_params_vec,
            flow[origin.id][0],
            previous_flow[0],
            previous_density[0],
            previous_density[1],
            previous_speed[0],
            flow[onramp.id][0],
            previous_speed[0],
        )
        next_density_direct = np.array(res).flatten()[0]
        next_speed_direct = np.array(res).flatten()[1]

        assert np.isclose(density[link.id][0], next_density_direct)
        assert np.isclose(speed[link.id][0], next_speed_direct)

    def test_critical_density_and_backward_wave(self):
        link = MotorwayLink(length=1.0, lanes=1)
        link.partition_link(max_vf=100.0, preferred_cell_size=1.0, dt=0.001)

        model = METANET()

        const_alpha = 2.0
        model_params: METANETParams = {
            "vf": 100.0,
            "qc_lane": 2000.0,
            "rho_jam": 150.0,
            "tau": 1.0,
            "nu": 0.0,
            "kappa": 0.1,
            "delta": 0.0,
            "phi": 0.0,
            "alpha": {link.id: const_alpha},
        }

        expected_rho_cr = model_params["qc_lane"] / (
            model_params["vf"] * np.exp(-1 / const_alpha)
        )
        computed_rho_cr = model.critical_density(params=model_params, link_id=link.id)
        assert np.isclose(computed_rho_cr, expected_rho_cr)

        expected_w = (
            link.lanes
            * model_params["qc_lane"]
            / (model_params["rho_jam"] - expected_rho_cr)
        )
        computed_w = model.backward_wave_speed(
            params=model_params, link_id=link.id, lanes=link.lanes
        )
        assert np.isclose(computed_w, expected_w)

    def test_lane_drop_deceleration(self):
        # verify that an upcoming lane drop reduces the computed speed
        # (phi term should apply additional deceleration)
        # build a two-link motorway link with an upstream link having an upcoming drop
        dt = 0.001
        link = MotorwayLink(length=2.0, lanes=2)

        link2 = MotorwayLink(length=2.0, lanes=1)

        origin = Origin()
        destination = Destination()

        node = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[link2])
        node3 = Node(incoming=[link2], outgoing=[destination])
        network = Network(nodes=[node, node2, node3])

        # run full simulation with both models and compare upstream link speeds
        model_no_phi = METANET()
        model_with_phi = METANET()
        model_no_phi_params: METANETParams = {
            "vf": 100.0,
            "qc_lane": 2000.0,
            "rho_jam": 150.0,
            "tau": 1.0,
            "nu": 0.0,
            "kappa": 0.1,
            "delta": 0.0,
            "phi": 0.0,
            "alpha": {link.id: 1.0, link2.id: 1.0},
        }
        model_with_phi_params: METANETParams = {**model_no_phi_params, "phi": 0.5}

        # initial states for link (2 cells) and downstream link2 (1 cell)
        init_density_link = np.array([20.0, 20.0], dtype=np.float64)
        init_speed_link = np.array([30.0, 30.0], dtype=np.float64)
        init_flow_link = np.array([80.0, 80.0], dtype=np.float64)

        init_density_link2 = np.array([20.0, 20.0], dtype=np.float64)
        init_speed_link2 = np.array([30.0, 30.0], dtype=np.float64)
        init_flow_link2 = np.array([0.0, 0.0], dtype=np.float64)

        mainline_demand = lambda t: 100.0

        def run_model(model):
            sim = Simulation(
                network,
                model,
                model_no_phi_params if model is model_no_phi else model_with_phi_params,
            )
            _, states, _ = sim.run(
                duration=dt,
                dt=dt,
                origin_demands={origin.id: mainline_demand},
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
                destination_flow_bc={destination.id: lambda t: 6000.0},
                destination_density_bc={destination.id: lambda t: 0.0},
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

    def test_model_params_conversion_and_validation(self):
        # build a minimal network with one motorway link and one onramp
        link = MotorwayLink(length=1.0, lanes=1)
        onramp = Onramp(length=0.5, lanes=1)
        origin = Origin()
        destination = Destination()

        node1 = Node(incoming=[origin, onramp], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        model = METANET()

        # 1) scalar alpha -> vector and back
        scalar_params: METANETParams = {
            "vf": 100.0,
            "qc_lane": 2000.0,
            "rho_jam": 150.0,
            "tau": 1.0,
            "nu": 0.2,
            "kappa": 0.1,
            "delta": 0.0,
            "phi": 0.0,
            "alpha": 1.5,
        }

        vec = model.model_params_to_vec(network=network, model_params=scalar_params)
        # we expect 8 scalars + 2 link-specific alphas (onramp + motorway link)
        assert vec.shape[0] == 10

        unpacked = model.model_params_vec_to_dict(network=network, model_params_vec=vec)
        assert unpacked["tau"] == vec[3]
        # alpha must be a dict mapping link ids -> values
        assert isinstance(unpacked["alpha"], dict)
        assert len(unpacked["alpha"]) == 2
        for v in unpacked["alpha"].values():
            assert float(v) == 1.5

        # 2) dict alpha -> vector and back (link-specific values)
        alpha_map = {onramp.id: 0.9, link.id: 1.1}
        dict_params: METANETParams = {**scalar_params, "alpha": alpha_map}

        vec2 = model.model_params_to_vec(network=network, model_params=dict_params)
        unpacked2 = model.model_params_vec_to_dict(
            network=network, model_params_vec=vec2
        )
        assert isinstance(unpacked2["alpha"], dict)
        assert unpacked2["alpha"][onramp.id] == alpha_map[onramp.id]
        assert unpacked2["alpha"][link.id] == alpha_map[link.id]

        # 3) symbolic parameter vector creation + unpacking should work
        sym = model.set_up_symbolic_model_params(network=network)
        sym_unpacked = model.model_params_vec_to_dict(
            network=network, model_params_vec=sym
        )
        # symbolic unpacking should produce an 'alpha' dict with correct size
        assert isinstance(sym_unpacked["alpha"], dict)
        assert len(sym_unpacked["alpha"]) == 2

        # 4) validate_model_params: accept valid, reject invalid
        # valid should not raise
        model.validate_model_params(scalar_params)

        # missing key triggers ValueError
        bad = scalar_params.copy()
        bad.pop("tau")
        try:
            model.validate_model_params(bad)
            raise AssertionError(
                "validate_model_params should have raised for missing key"
            )
        except ValueError:
            pass

        # invalid alpha type triggers ValueError
        bad2 = scalar_params.copy()
        bad2["alpha"] = [1, 2, 3]  # type: ignore
        try:
            model.validate_model_params(bad2)
            raise AssertionError(
                "validate_model_params should have raised for invalid alpha type"
            )
        except ValueError:
            pass

    def test_circular_network_simulation(self):
        """Test METANET can simulate a circular network topology with feedback loop."""
        # create circular network: Origin -> Link1 -> Link2 -> Link3 -> Link4 -> back to Link2
        # with offramp at node3 for traffic exit
        dt = 0.005

        link1 = MotorwayLink(length=1.0, lanes=2)
        link2 = MotorwayLink(length=1.0, lanes=2)
        link3 = MotorwayLink(length=1.0, lanes=2)
        link4 = MotorwayLink(length=1.0, lanes=2)

        origin = Origin()
        offramp = Offramp(lanes=1)
        dest = Destination()

        # create circular topology
        node1 = Node(id="n1", incoming=[origin], outgoing=[link1])
        node2 = Node(id="n2", incoming=[link1, link4], outgoing=[link2])  # loop closure
        node3 = Node(id="n3", incoming=[link2], outgoing=[link3, offramp])
        node4 = Node(id="n4", incoming=[link3], outgoing=[link4])
        node_off = Node(id="n_off", incoming=[offramp], outgoing=[dest])

        network = Network(nodes=[node1, node2, node3, node4, node_off])

        # partition links
        link1.partition_link(max_vf=80.0, preferred_cell_size=0.5, dt=dt)
        link2.partition_link(max_vf=80.0, preferred_cell_size=0.5, dt=dt)
        link3.partition_link(max_vf=80.0, preferred_cell_size=0.5, dt=dt)
        link4.partition_link(max_vf=80.0, preferred_cell_size=0.5, dt=dt)

        model = METANET()
        model_params: METANETParams = {
            "vf": 80.0,
            "qc_lane": 1500.0,
            "rho_jam": 140.0,
            "tau": 1.0,
            "nu": 0.1,
            "kappa": 0.1,
            "delta": 0.001,
            "phi": 0.0,
            "alpha": 1.0,  # Use scalar alpha for all links
        }

        # initial conditions - small flows/densities throughout
        num_cells_1 = len(link1)
        num_cells_2 = len(link2)
        num_cells_3 = len(link3)
        num_cells_4 = len(link4)

        initial_flow_dict = {
            origin.id: 100.0,
            link1.id: np.ones(num_cells_1) * 100.0,
            link2.id: np.ones(num_cells_2) * 100.0,
            link3.id: np.ones(num_cells_3) * 90.0,
            link4.id: np.ones(num_cells_4) * 90.0,
            offramp.id: 10.0,
            dest.id: 90.0,
        }

        initial_density_dict = {
            link1.id: np.ones(num_cells_1) * 10.0,
            link2.id: np.ones(num_cells_2) * 10.0,
            link3.id: np.ones(num_cells_3) * 10.0,
            link4.id: np.ones(num_cells_4) * 10.0,
            offramp.id: np.array([10.0]),
        }

        initial_speed_dict = {
            link1.id: np.ones(num_cells_1) * 70.0,
            link2.id: np.ones(num_cells_2) * 70.0,
            link3.id: np.ones(num_cells_3) * 70.0,
            link4.id: np.ones(num_cells_4) * 70.0,
            offramp.id: np.array([70.0]),
        }

        # demand and split ratios
        mainline_demand = lambda t: 200.0
        turning_rates = {
            node1.id: lambda t: {link1.id: 1.0},
            node2.id: lambda t: {link2.id: 1.0},
            node3.id: lambda t: {link3.id: 0.9, offramp.id: 0.1},  # 10% exit
            node4.id: lambda t: {link4.id: 1.0},
            node_off.id: lambda t: {dest.id: 1.0},
        }

        # run short simulation
        sim = Simulation(network, model, model_params)
        time_array, states, disturbances = sim.run(
            duration=3 * dt,  # just 3 timesteps
            dt=dt,
            origin_demands={origin.id: mainline_demand},
            initial_flows=initial_flow_dict,
            initial_densities=initial_density_dict,
            initial_speeds=initial_speed_dict,
            initial_origin_queues={origin.id: 0.0},
            initial_onramp_queues={},
            initial_offramp_queues={offramp.id: 0.0},
            turning_rates=turning_rates,
            destination_flow_bc={dest.id: lambda t: 6000.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            preferred_cell_size=0.5,
            plot_results=False,
        )

        # verify simulation completed successfully
        assert time_array is not None
        assert states is not None
        assert disturbances is not None
        assert len(time_array) == 4  # 0, dt, 2*dt, 3*dt

        # unpack final state
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            network.state_vec_to_network_dict(states[:, -1])
        )

        # basic sanity checks - all states should be finite and non-negative
        for link_id in [link1.id, link2.id, link3.id, link4.id]:
            assert link_id in flows
            assert link_id in densities
            assert link_id in speeds
            assert np.all(np.isfinite(flows[link_id]))
            assert np.all(np.isfinite(densities[link_id]))
            assert np.all(np.isfinite(speeds[link_id]))
            assert np.all(flows[link_id] >= 0)
            assert np.all(densities[link_id] >= 0)
            assert np.all(speeds[link_id] >= 0)

        # verify feedback: link4 feeds back into node2, which outputs to link2
        # so link2's initial flow should be influenced by both link1 and link4
        assert flows[link2.id][0] >= 0  # basic sanity check

        # verify offramp receives some flow
        assert offramp.id in flows
        assert flows[offramp.id][-1] >= 0
