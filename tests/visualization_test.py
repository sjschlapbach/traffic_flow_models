import pytest
import numpy as np
import tempfile
import os

from traffic_flow_models.network.network import Network
from traffic_flow_models.network.node import Node
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.motorway_link import MotorwayLink
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.simulation import Simulation
from traffic_flow_models import CTM


class TestVisualization:
    def test_density_to_color_low_density(self):
        """Test color mapping for low densities."""
        rho_crit = 20.0
        rho_jam = 180.0

        # At zero density, should be bright green
        r, g, b = Simulation._density_to_color(0.0, rho_crit, rho_jam)
        assert r == Simulation.COLOR_BRIGHT_GREEN[0]
        assert g == Simulation.COLOR_BRIGHT_GREEN[1]
        assert b == Simulation.COLOR_BRIGHT_GREEN[2]

        # At critical density, should be dark green
        r, g, b = Simulation._density_to_color(rho_crit, rho_crit, rho_jam)
        assert r == Simulation.COLOR_DARK_GREEN[0]
        assert g == Simulation.COLOR_DARK_GREEN[1]
        assert b == Simulation.COLOR_DARK_GREEN[2]

    def test_density_to_color_high_density(self):
        """Test color mapping for high densities."""
        rho_crit = 20.0
        rho_jam = 180.0

        # At critical density, should be dark green
        r, g, b = Simulation._density_to_color(rho_crit, rho_crit, rho_jam)
        assert r == Simulation.COLOR_DARK_GREEN[0]
        assert g == Simulation.COLOR_DARK_GREEN[1]
        assert b == Simulation.COLOR_DARK_GREEN[2]

        # At jam density, should be dark red
        r, g, b = Simulation._density_to_color(rho_jam, rho_crit, rho_jam)
        assert r == Simulation.COLOR_DARK_RED[0]
        assert g == Simulation.COLOR_DARK_RED[1]
        assert b == Simulation.COLOR_DARK_RED[2]

        # Above jam density, should still be capped at dark red
        r, g, b = Simulation._density_to_color(rho_jam * 1.5, rho_crit, rho_jam)
        assert r == Simulation.COLOR_DARK_RED[0]
        assert g == Simulation.COLOR_DARK_RED[1]
        assert b == Simulation.COLOR_DARK_RED[2]

    def test_density_to_color_midpoint(self):
        """Test color interpolation at midpoints."""
        rho_crit = 20.0
        rho_jam = 180.0

        # at 50% of critical density
        r, g, b = Simulation._density_to_color(rho_crit * 0.5, rho_crit, rho_jam)
        # should be roughly halfway between bright and dark green
        midpoint_0 = (
            Simulation.COLOR_BRIGHT_GREEN[0] + Simulation.COLOR_DARK_GREEN[0]
        ) / 2.0
        midpoint_1 = (
            Simulation.COLOR_BRIGHT_GREEN[1] + Simulation.COLOR_DARK_GREEN[1]
        ) / 2.0
        midpoint_2 = (
            Simulation.COLOR_BRIGHT_GREEN[2] + Simulation.COLOR_DARK_GREEN[2]
        ) / 2.0
        assert midpoint_0 - 10 < r < midpoint_0 + 10
        assert midpoint_1 - 10 < g < midpoint_1 + 10
        assert midpoint_2 - 10 < b < midpoint_2 + 10

        # at midpoint between critical and jam
        mid_density = (rho_crit + rho_jam) / 2.0
        r, g, b = Simulation._density_to_color(mid_density, rho_crit, rho_jam)
        # should be orange-ish (slightly closer to orange than red)
        orange_0 = (
            1.2 * Simulation.COLOR_ORANGE[0] + Simulation.COLOR_DARK_RED[0]
        ) / 2.2
        orange_1 = (
            1.2 * Simulation.COLOR_ORANGE[1] + Simulation.COLOR_DARK_RED[1]
        ) / 2.2
        orange_2 = (
            1.2 * Simulation.COLOR_ORANGE[2] + Simulation.COLOR_DARK_RED[2]
        ) / 2.2
        assert orange_0 - 10 < r < orange_0 + 10
        assert orange_1 - 10 < g < orange_1 + 10
        assert orange_2 - 10 < b < orange_2 + 10

    def test_interpolate_frames_identity(self):
        """Test that subsampling=1 returns unchanged arrays."""
        state_history = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        time_array = np.array([0.0, 1.0, 2.0])
        interp_time, interp_state = Simulation._interpolate_frames(
            state_history, time_array, subsampling=1
        )

        np.testing.assert_array_equal(interp_time, time_array)
        np.testing.assert_array_equal(interp_state, state_history)

    def test_interpolate_frames_doubles_frames(self):
        """Test that subsampling=2 correctly interpolates."""
        state_history = np.array([[0.0, 2.0, 4.0], [10.0, 20.0, 30.0]])
        time_array = np.array([0.0, 1.0, 2.0])
        interp_time, interp_state = Simulation._interpolate_frames(
            state_history, time_array, subsampling=2
        )

        # should have 5 frames: original 3 + 2 interpolated
        assert len(interp_time) == 5
        assert interp_state.shape[1] == 5

        # check original frames are preserved
        np.testing.assert_array_equal(interp_state[:, 0], state_history[:, 0])
        np.testing.assert_array_equal(interp_state[:, 2], state_history[:, 1])
        np.testing.assert_array_equal(interp_state[:, 4], state_history[:, 2])

        # check interpolated frames
        # between frame 0 and 1: should be average
        expected_mid_1 = (state_history[:, 0] + state_history[:, 1]) / 2.0
        np.testing.assert_allclose(interp_state[:, 1], expected_mid_1)

        # between frame 1 and 2: should be average
        expected_mid_2 = (state_history[:, 1] + state_history[:, 2]) / 2.0
        np.testing.assert_allclose(interp_state[:, 3], expected_mid_2)

    def test_visualize_simulation_creates_video_file(self):
        """Test that visualization creates a valid video file."""
        # create minimal network
        main = MotorwayLink(
            id="m1",
            length=2.0,
            lanes=3,
            lane_capacity=2000.0,
            free_flow_speed=100.0,
            jam_density=180.0,
        )
        origin = Origin(id="o1")
        dest = Destination(id="d1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])

        # set positions
        node1.set_position(0.0, 0.0)
        node2.set_position(2.0, 0.0)

        net = Network(nodes=[node1, node2])
        net.validate()

        # partition link
        main.partition_link(preferred_cell_size=0.5, dt=0.01)

        # create simulation with very short duration
        model = CTM()
        sim = Simulation(net, model)
        sim.run(
            duration=0.03,  # very short
            dt=0.01,
            preferred_cell_size=0.5,
            origin_demands={origin.id: lambda t: 1000.0},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 6000.0},
            destination_density_bc={dest.id: lambda t: 0.0},
        )

        # save results to temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            video_path = os.path.join(tmpdir, "output.avi")

            sim.save_results(filepath=results_path)

            # generate visualization
            sim.visualize(
                results_filepath=results_path,
                output_filepath=video_path,
                fps=1,  # Low fps for fast test
                figsize=(6, 4),
                dpi=50,  # Low dpi for fast test
            )

            # check video file exists and has non-zero size
            assert os.path.exists(video_path)
            assert os.path.getsize(video_path) > 0

    def test_visualize_simulation_with_subsampling(self):
        """Test that subsampling increases frame count."""
        # create minimal network
        main = MotorwayLink(
            id="m1",
            length=2.0,
            lanes=3,
            lane_capacity=2000.0,
            free_flow_speed=100.0,
            jam_density=180.0,
        )
        origin = Origin(id="o1")
        dest = Destination(id="d1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])

        node1.set_position(0.0, 0.0)
        node2.set_position(2.0, 0.0)

        net = Network(nodes=[node1, node2])
        net.validate()

        main.partition_link(preferred_cell_size=0.5, dt=0.01)

        model = CTM()
        sim = Simulation(net, model)
        sim.run(
            duration=0.03,
            dt=0.01,
            preferred_cell_size=0.5,
            origin_demands={origin.id: lambda t: 1000.0},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 6000.0},
            destination_density_bc={dest.id: lambda t: 0.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            video_path = os.path.join(tmpdir, "output_subsampled.avi")

            sim.save_results(filepath=results_path)

            # generate with subsampling
            sim.visualize(
                results_filepath=results_path,
                output_filepath=video_path,
                fps=1,
                subsampling=2,  # double frames
                figsize=(6, 4),
                dpi=50,
            )

            # video should be created
            assert os.path.exists(video_path)
            assert os.path.getsize(video_path) > 0

    def test_visualize_simulation_missing_positions_raises(self):
        """Test that missing node positions raises ValueError."""
        main = MotorwayLink(
            id="m1",
            length=2.0,
            lanes=3,
            lane_capacity=2000.0,
            free_flow_speed=100.0,
            jam_density=180.0,
        )
        origin = Origin(id="o1")
        dest = Destination(id="d1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])

        # DO NOT set positions
        net = Network(nodes=[node1, node2])
        net.validate()

        main.partition_link(preferred_cell_size=0.5, dt=0.01)

        model = CTM()
        sim = Simulation(net, model)
        sim.run(
            duration=0.03,
            dt=0.01,
            preferred_cell_size=0.5,
            origin_demands={origin.id: lambda t: 1000.0},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 6000.0},
            destination_density_bc={dest.id: lambda t: 0.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            video_path = os.path.join(tmpdir, "output.avi")

            sim.save_results(filepath=results_path)

            # should raise ValueError
            with pytest.raises(ValueError, match="lacks position information"):
                sim.visualize(
                    results_filepath=results_path,
                    output_filepath=video_path,
                    fps=1,
                )

    def test_visualize_simulation_with_ramps(self):
        """Test visualization with onramps and offramps."""
        # create network with ramps - separate nodes for onramp and offramp
        main1 = MotorwayLink(
            id="m1",
            length=2.0,  # increased from 1.0
            lanes=3,
            lane_capacity=2000.0,
            free_flow_speed=100.0,
            jam_density=180.0,
        )
        main2 = MotorwayLink(
            id="m2",
            length=2.0,  # increased from 1.0
            lanes=3,
            lane_capacity=2000.0,
            free_flow_speed=100.0,
            jam_density=180.0,
        )
        main3 = MotorwayLink(
            id="m3",
            length=2.0,  # increased from 1.0
            lanes=3,
            lane_capacity=2000.0,
            free_flow_speed=100.0,
            jam_density=180.0,
        )

        origin = Origin(id="o1")
        ramp_origin = Origin(id="o_on")
        dest = Destination(id="d1")
        dest_off = Destination(id="d_off")
        onramp = Onramp(
            id="on1",
            lanes=1,
            lane_capacity=1500.0,
            free_flow_speed=80.0,
            jam_density=160.0,
        )
        offramp = Offramp(
            id="off1",
            lanes=1,
            lane_capacity=1500.0,
            free_flow_speed=80.0,
            jam_density=160.0,
        )

        # separate nodes for onramp and offramp to satisfy validation rules
        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node_ramp = Node(id="n_ramp", incoming=[ramp_origin], outgoing=[onramp])
        node2 = Node(id="n2", incoming=[main1, onramp], outgoing=[main2])
        node3 = Node(id="n3", incoming=[main2], outgoing=[main3, offramp])
        node4 = Node(id="n4", incoming=[main3], outgoing=[dest])
        node_off = Node(id="n_off", incoming=[offramp], outgoing=[dest_off])

        # set positions
        node1.set_position(0.0, 0.0)
        node_ramp.set_position(0.5, -0.5)
        node2.set_position(1.0, 0.0)
        node3.set_position(2.0, 0.0)
        node4.set_position(3.0, 0.0)
        node_off.set_position(2.2, -0.4)

        net = Network(nodes=[node2, node1, node_ramp, node3, node4, node_off])
        net.validate()

        # partition links
        for node in net.list_nodes():
            for link in node.incoming + node.outgoing:
                if isinstance(link, MotorwayLink):
                    link.partition_link(preferred_cell_size=0.5, dt=0.01)

        model = CTM()
        sim = Simulation(net, model)
        sim.run(
            duration=0.03,
            dt=0.01,
            preferred_cell_size=0.5,
            origin_demands={
                origin.id: lambda t: 1000.0,
                ramp_origin.id: lambda t: 500.0,
            },
            turning_rates={
                node3.id: lambda t: {offramp.id: 0.2, main3.id: 0.8},
                node_off.id: lambda t: {dest_off.id: 1.0},
            },
            destination_flow_bc={
                dest.id: lambda t: 6000.0,
                dest_off.id: lambda t: 6000.0,
            },
            destination_density_bc={
                dest.id: lambda t: 0.0,
                dest_off.id: lambda t: 0.0,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            video_path = os.path.join(tmpdir, "output_ramps.avi")

            sim.save_results(filepath=results_path)

            # should complete without error
            sim.visualize(
                results_filepath=results_path,
                output_filepath=video_path,
                fps=1,
                figsize=(6, 4),
                dpi=50,
            )

            assert os.path.exists(video_path)
            assert os.path.getsize(video_path) > 0
