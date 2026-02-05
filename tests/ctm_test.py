import numpy as np
import casadi

from traffic_flow_models import (
    CTM,
    Network,
    Node,
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
)


class TestCTM:
    def test_critical_density_and_backward_wave(self):
        """Test the fundamental diagram helper functions for CTM."""
        link = MotorwayLink(
            length=1.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link.partition_link(preferred_cell_size=1.0, dt=0.001)

        model = CTM()

        # CTM critical density: Qc_lane / vf
        expected_rho_cr = link.lane_capacity / link.vf
        computed_rho_cr = model.critical_density(
            lane_capacity=link.lane_capacity, free_flow_speed=link.vf
        )
        assert np.isclose(computed_rho_cr, expected_rho_cr)

        # CTM backward wave speed: Qc / (rho_jam - rho_cr)
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

    def test_basic_network_simulation(self):
        """Test basic CTM simulation with a simple network (no onramps)."""
        # create a simple network: origin -> link -> destination
        link = MotorwayLink(
            length=2.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        dt = 0.005
        link.partition_link(preferred_cell_size=1.0, dt=dt)

        # get actual number of cells after partitioning
        num_cells = len(link)

        origin = Origin()
        destination = Destination()

        node1 = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        model = CTM()

        # initial conditions
        initial_density = np.full(num_cells, 20.0, dtype=np.float64)
        initial_speed = np.full(num_cells, 50.0, dtype=np.float64)
        initial_flow = initial_density * initial_speed * link.lanes

        mainline_demand = lambda t: 500.0

        # run simulation for one timestep
        _, states, _ = network.simulate(
            duration=dt,
            dt=dt,
            model=model,
            model_params=None,  # CTM has no model parameters
            origin_demands={origin.id: mainline_demand},
            onramp_demands={},
            initial_flows={
                link.id: initial_flow,
                origin.id: mainline_demand(0),
                destination.id: 0.0,
            },
            initial_densities={link.id: initial_density},
            initial_speeds={link.id: initial_speed},
            turning_rates={},
            destination_boundary_conditions={destination.id: lambda t: 0.0},
            preferred_cell_size=1.0,  # Match manual partitioning
            plot_results=False,
        )

        # extract final state
        flow, density, speed, _, _, _ = network.state_vec_to_network_dict(states[:, -1])

        # verify shapes
        assert flow[link.id].shape == (num_cells,)
        assert density[link.id].shape == (num_cells,)
        assert speed[link.id].shape == (num_cells,)

        # verify that density is non-negative and within bounds
        assert np.all(density[link.id] >= 0)
        assert np.all(density[link.id] <= link.rho_jam)

        # verify that speed is within bounds
        assert np.all(speed[link.id] >= 0)
        assert np.all(speed[link.id] <= link.vf)

        # verify flow-density-speed relationship: q = rho * v * lanes
        for i in range(len(link)):
            if density[link.id][i] > 1e-6:
                expected_flow = density[link.id][i] * speed[link.id][i] * link.lanes
                assert np.isclose(flow[link.id][i], expected_flow, rtol=1e-5)

    def test_network_with_onramp(self):
        """Test CTM simulation with a network including an onramp."""
        link = MotorwayLink(
            length=2.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        dt = 0.005
        link.partition_link(preferred_cell_size=1.0, dt=dt)

        # get actual number of cells
        num_cells = len(link)
        origin = Origin()
        onramp = Onramp(
            lanes=1, lane_capacity=1000, free_flow_speed=60, jam_density=100
        )
        destination = Destination()

        node1 = Node(incoming=[origin, onramp], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        model = CTM()

        # initial conditions
        initial_density = np.full(num_cells, 10.0, dtype=np.float64)
        initial_speed = np.full(num_cells, 80.0, dtype=np.float64)
        initial_flow = initial_density * initial_speed * link.lanes

        mainline_demand = lambda t: 500.0
        onramp_demand = lambda t: 200.0

        # run simulation
        _, states, _ = network.simulate(
            duration=dt * 2,
            dt=dt,
            model=model,
            model_params=None,
            origin_demands={origin.id: mainline_demand},
            onramp_demands={onramp.id: onramp_demand},
            initial_flows={
                link.id: initial_flow,
                onramp.id: 0.0,
                origin.id: mainline_demand(0),
                destination.id: 0.0,
            },
            initial_densities={
                link.id: initial_density,
                onramp.id: np.array([0.0]),
            },
            initial_speeds={link.id: initial_speed, onramp.id: np.array([0.0])},
            turning_rates={},
            destination_boundary_conditions={destination.id: lambda t: 0.0},
            preferred_cell_size=1.0,  # Match manual partitioning
            plot_results=False,
        )

        # extract final state
        flow, density, speed, _, onramp_queue, _ = network.state_vec_to_network_dict(
            states[:, -1]
        )

        # verify that onramp flow is within capacity
        assert flow[onramp.id][0] <= onramp.Qc + 1e-6  # allow small numerical error

        # verify conservation principles
        assert np.all(density[link.id] >= 0)
        assert np.all(density[link.id] <= link.rho_jam)

    def test_network_with_offramp(self):
        """Test CTM simulation with a network including an offramp."""
        link = MotorwayLink(
            length=2.0,
            lanes=2,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        dt = 0.005
        link.partition_link(preferred_cell_size=1.0, dt=dt)

        # get actual number of cells
        num_cells = len(link)

        origin = Origin()
        offramp = Offramp(
            lanes=1, lane_capacity=500, free_flow_speed=50, jam_density=100
        )
        destination_offramp = Destination()
        destination_mainline = Destination()

        # connect offramp to its destination
        offramp.destination = destination_offramp

        node1 = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[offramp, destination_mainline])
        # Note: Offramps connect directly to their destinations, no intermediate node needed
        network = Network(nodes=[node1, node2])

        model = CTM()

        # initial conditions
        initial_density = np.full(num_cells, 30.0, dtype=np.float64)
        initial_speed = np.full(num_cells, 60.0, dtype=np.float64)
        initial_flow = initial_density * initial_speed * link.lanes

        mainline_demand = lambda t: 1500.0
        offramp_split = 0.3  # 30% to offramp, 70% to mainline
        mainline_split = 0.7  # remaining to mainline destination

        # run simulation
        _, states, _ = network.simulate(
            duration=dt * 2,
            dt=dt,
            model=model,
            model_params=None,
            origin_demands={origin.id: mainline_demand},
            onramp_demands={},
            initial_flows={
                link.id: initial_flow,
                offramp.id: 0.0,
                origin.id: mainline_demand(0),
                destination_offramp.id: 0.0,
                destination_mainline.id: 0.0,
            },
            initial_densities={
                link.id: initial_density,
                offramp.id: np.array([0.0]),
            },
            initial_speeds={link.id: initial_speed, offramp.id: np.array([0.0])},
            turning_rates={
                node2.id: lambda t: {
                    offramp.id: offramp_split,
                    destination_mainline.id: mainline_split,
                }
            },
            destination_boundary_conditions={
                destination_offramp.id: lambda t: 0.0,
                destination_mainline.id: lambda t: 0.0,
            },
            preferred_cell_size=1.0,  # Match manual partitioning
            plot_results=False,
        )

        # extract final state
        flow, density, speed, _, _, offramp_queue = network.state_vec_to_network_dict(
            states[:, -1]
        )

        # verify that offramp flow is within capacity
        assert flow[offramp.id][0] <= offramp.Qc + 1e-6

        # verify that offramp queue is non-negative
        assert offramp_queue[offramp.id] >= 0

        # verify conservation principles
        assert np.all(density[link.id] >= 0)
        assert np.all(density[link.id] <= link.rho_jam)

        # verify that the split ratios are respected
        total_outflow = flow[link.id][-1]
        offramp_flow = flow[offramp.id][0]
        mainline_destination_flow = flow[destination_mainline.id]
        expected_offramp_flow = total_outflow * offramp_split
        expected_mainline_flow = total_outflow * mainline_split
        assert np.isclose(offramp_flow, expected_offramp_flow, rtol=1e-5)
        assert np.isclose(mainline_destination_flow, expected_mainline_flow, rtol=1e-5)

    def test_helper_function_normalized_splits(self):
        """Test the split ratio normalization helper function."""
        # create a node with multiple outgoing links
        link1 = MotorwayLink(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        link2 = MotorwayLink(
            length=1.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        origin = Origin()

        node = Node(incoming=[origin], outgoing=[link1, link2])

        model = CTM()

        # create unnormalized splits (sum to 2.0)
        splits = {
            node.id: {
                link1.id: casadi.SX(0.6),
                link2.id: casadi.SX(1.4),
            }
        }

        normalized = model._compute_normalized_splits(node=node, splits=splits)

        # verify normalization
        total = float(normalized[link1.id] + normalized[link2.id])
        assert np.isclose(total, 1.0)

        # verify proportions are maintained
        expected_link1 = 0.6 / 2.0
        expected_link2 = 1.4 / 2.0
        assert np.isclose(float(normalized[link1.id]), expected_link1)
        assert np.isclose(float(normalized[link2.id]), expected_link2)

    def test_flow_capacity_constraints(self):
        """Test that CTM respects capacity constraints."""
        link = MotorwayLink(
            length=2.0,
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=150,
        )
        dt = 0.005  # Smaller dt to satisfy CFL condition
        link.partition_link(preferred_cell_size=0.5, dt=dt)

        origin = Origin()
        destination = Destination()

        node1 = Node(incoming=[origin], outgoing=[link])
        node2 = Node(incoming=[link], outgoing=[destination])
        network = Network(nodes=[node1, node2])

        model = CTM()

        # get actual number of cells
        num_cells = len(link)

        # initial conditions with low density
        initial_density = np.full(num_cells, 5.0, dtype=np.float64)
        initial_speed = np.full(num_cells, 100.0, dtype=np.float64)
        initial_flow = initial_density * initial_speed * link.lanes

        # very high demand (exceeds capacity)
        high_demand = lambda t: 5000.0

        # run simulation
        _, states, _ = network.simulate(
            duration=dt * 5,
            dt=dt,
            model=model,
            model_params=None,
            origin_demands={origin.id: high_demand},
            onramp_demands={},
            initial_flows={
                link.id: initial_flow,
                origin.id: high_demand(0),
                destination.id: 0.0,
            },
            initial_densities={link.id: initial_density},
            initial_speeds={link.id: initial_speed},
            turning_rates={},
            destination_boundary_conditions={destination.id: lambda t: 0.0},
            preferred_cell_size=0.5,  # match manual partitioning
            plot_results=False,
        )

        # extract all states
        for t in range(states.shape[1]):
            flow, _, _, _, _, _ = network.state_vec_to_network_dict(states[:, t])

            # verify that flow never exceeds capacity
            capacity = link.lane_capacity * link.lanes
            assert np.all(
                flow[link.id] <= capacity + 1e-6
            )  # allow small numerical error
