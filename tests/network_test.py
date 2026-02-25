import pytest
import numpy as np
import tempfile
import os
import json
from typing import cast
from numpy.typing import NDArray

from traffic_flow_models.network.network import Network
from traffic_flow_models.network.node import Node
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.motorway_link import MotorwayLink


# helper function to partition motorway links for tests
def partition_motorway_links(
    network: Network, preferred_cell_size: float = 0.5, dt: float = 0.001
) -> None:
    """Partition all MotorwayLink instances in a network."""
    for node in network.list_nodes():
        for link in node.incoming + node.outgoing:
            if isinstance(link, MotorwayLink):
                link.partition_link(preferred_cell_size, dt)


class TestNetwork:
    def test_add_node_duplicate_id_raises(self):
        net = Network()
        n1 = Node(id="n1")
        net.add_node(n1)
        n1_dup = Node(id="n1")

        with pytest.raises(ValueError):
            net.add_node(n1_dup)

    def test_add_remove_get_list_iteration(self):
        n1 = Node(id="a")
        n2 = Node(id="b")
        net = Network(nodes=[n1, n2])

        assert len(net) == 2
        assert net.get_node("a") is n1
        assert net.get_node("missing") is None

        ids = [n.id for n in net.list_nodes()]
        assert "a" in ids and "b" in ids

        # remove by id
        net.remove_node("a")
        assert net.get_node("a") is None

        # remove by id
        net.remove_node(n2.id)
        assert len(net) == 0

    def test_validate_path_connected_nodes(self):
        # create shared mainline link between node1 -> node2
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )

        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # should not raise
        net.validate()

    def test_validate_offramp_without_destination_raises(self):
        # create an offramp without destination
        offr = Offramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            destination=None,
        )
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[offr])
        net = Network(nodes=[node1, node2])

        with pytest.raises(ValueError):
            net.validate()

    def test_validate_unconnected_component_raises(self):
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        # connected pair
        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])

        # isolated node (links not shared)
        origin2 = Origin()
        dest2 = Destination()
        main2 = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        node3 = Node(id="n3", incoming=[origin2], outgoing=[main2])
        node4 = Node(id="n4", incoming=[main2], outgoing=[dest2])

        net = Network(nodes=[node1, node2, node3, node4])
        with pytest.raises(ValueError):
            net.validate()

    def test_validate_origin_node_multiple_motorway_outgoing_raises(self):
        """Test that a node connected to an origin can only have one motorway link as outgoing."""
        # create a node with Origin incoming and two MotorwayLinks outgoing (invalid)
        origin = Origin()
        main1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        dest1 = Destination()
        dest2 = Destination()

        node1 = Node(
            id="n1", incoming=[origin], outgoing=[main1, main2]
        )  # Invalid: multiple motorway links
        node2 = Node(id="n2", incoming=[main1], outgoing=[dest1])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest2])

        net = Network(nodes=[node1, node2, node3])

        with pytest.raises(ValueError):
            net.validate()

    def test_validate_origin_no_motorway_outgoing_raises(self):
        """Test that a node connected to an origin must have at least one motorway link as outgoing."""
        # create a node with Origin incoming and no MotorwayLink outgoing (invalid)
        origin = Origin()
        offramp = Offramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            destination=Destination(),
        )

        node1 = Node(
            id="n1", incoming=[origin], outgoing=[offramp]
        )  # Invalid: no motorway link

        net = Network(nodes=[node1])

        with pytest.raises(ValueError):
            net.validate()

    def test_validate_onramp_node_multiple_motorway_outgoing_raises(self):
        """Test that a node connected to an onramp can only have one motorway link as outgoing."""
        # create a node with Onramp outgoing and two MotorwayLinks outgoing (invalid)
        onramp = Onramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        main1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        dest = Destination()
        dest2 = Destination()

        node1 = Node(
            id="n1", incoming=[onramp], outgoing=[main1, main2]
        )  # Invalid: multiple motorway links
        node2 = Node(id="n2", incoming=[main1], outgoing=[dest])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest2])

        net = Network(nodes=[node1, node2, node3])

        with pytest.raises(ValueError):
            net.validate()

    def test_vlalidate_onramp_no_motorway_outgoing_raises(self):
        """Test that a node connected to an onramp must have at least one motorway link as outgoing."""
        # create a node with Onramp incoming and no MotorwayLink outgoing (invalid)
        onramp = Onramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        offramp = Offramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            destination=Destination(),
        )

        node1 = Node(
            id="n1", incoming=[onramp], outgoing=[offramp]
        )  # Invalid: no motorway link

        net = Network(nodes=[node1])

        with pytest.raises(ValueError):
            net.validate()

    def test_validate_onramp_without_origin_passes(self):
        # create mainline and an onramp feeding into it (no Origin present)
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        onr = Onramp(lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180)
        dest = Destination()

        node_upstream = Node(id="up", incoming=[onr], outgoing=[main])
        node_downstream = Node(id="down", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node_upstream, node_downstream])

        # should not raise: network contains an onramp (counts as origin-type link) and a destination
        net.validate()

    def test_validate_requires_origin_or_onramp_raises(self):
        # network with only motorway links and a destination but no Origin/Onramp
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        dest = Destination()

        node1 = Node(id="n1", incoming=[main], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        with pytest.raises(ValueError):
            net.validate()

    def test_validate_requires_destination_raises(self):
        # network with Origin and motorway links but no Destination anywhere
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[main])
        net = Network(nodes=[node1, node2])

        with pytest.raises(ValueError):
            net.validate()

    def test_node_missing_incoming_raises(self):
        # node with no incoming links
        node = Node(
            id="bad",
            incoming=[],
            outgoing=[
                MotorwayLink(
                    length=1.0,
                    lanes=1,
                    lane_capacity=1500,
                    free_flow_speed=80,
                    jam_density=140,
                )
            ],
        )
        net = Network(nodes=[node])
        with pytest.raises(ValueError):
            net.validate()

    def test_node_missing_outgoing_raises(self):
        # node with no outgoing links
        node = Node(
            id="bad2",
            incoming=[
                MotorwayLink(
                    length=1.0,
                    lanes=1,
                    lane_capacity=1500,
                    free_flow_speed=80,
                    jam_density=140,
                )
            ],
            outgoing=[],
        )
        net = Network(nodes=[node])
        with pytest.raises(ValueError):
            net.validate()

    def test_validate_incoming_destination_id_mismatch_raises(self):

        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # corrupt the destination id stored on the main link
        main.destination_node_id = "wrong"

        with pytest.raises(ValueError):
            net.validate()

    def test_validate_outgoing_origin_id_mismatch_raises(self):
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # corrupt the origin id stored on the main link
        main.origin_node_id = "wrong"
        with pytest.raises(ValueError):
            net.validate()

    def test_validate_missing_destination_or_origin_id_raises(self):
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # remove origin/destination ids
        main.origin_node_id = None
        with pytest.raises(ValueError):
            net.validate()

        # restore origin id and remove destination id instead
        main.origin_node_id = node1.id
        main.destination_node_id = None
        with pytest.raises(ValueError):
            net.validate()

    def test_network_validate_rejects_invalid_link_types_set_directly(self):
        # create a node and bypass the Node helpers by assigning lists directly
        main = MotorwayLink(
            length=1.0, lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        n = Node(id="bad")

        # invalid incoming type (Offramp is not allowed as incoming)
        n.incoming = [
            Offramp(lanes=1, lane_capacity=1000, free_flow_speed=80, jam_density=140)
        ]
        n.outgoing = [main]
        net = Network(nodes=[n])

        # add a second valid node so network-wide checks proceed to node-level validation
        other = Node(
            id="other",
            incoming=[Origin()],
            outgoing=[Destination()],
        )
        net.add_node(other)

        with pytest.raises(TypeError):
            net.validate()

        # invalid outgoing type (Origin is not allowed as outgoing)
        net = Network()
        n2 = Node(id="bad2")
        n2.incoming = [main]
        n2.outgoing = [Origin()]
        net.add_node(n2)

        # add a valid node so network.validate advances to node-level type checks
        other2 = Node(
            id="other2",
            incoming=[Origin()],
            outgoing=[Destination()],
        )
        net.add_node(other2)

        with pytest.raises(TypeError):
            net.validate()

    def test_network_dict_to_state_vec_simple_network(self):
        """Test packing state dictionaries to vector for simple Origin->Motorway->Destination network."""
        # create simple network: Origin -> MotorwayLink -> Destination
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # partition motorway links to create cells
        partition_motorway_links(net)

        # prepare state dictionaries (MotorwayLink has 2 cells with default cell_size=0.5)
        num_cells = len(main)
        assert num_cells == 2
        flow_dict = {
            origin.id: np.array([500.0]),
            main.id: np.array([1000.0, 1100.0]),
            dest.id: np.array([1200.0]),
        }
        density_dict = {main.id: np.array([50.0, 55.0])}
        speed_dict = {main.id: np.array([75.0, 78.0])}
        origin_queue_dict = {origin.id: 10.0}
        onramp_queue_dict = {}
        offramp_queue_dict = {}

        # pack to state vector
        (
            x,
            num_flows,
            num_densities,
            num_speeds,
            num_origin,
            num_onramp,
            num_offramp,
            num_splits,
            num_destinations,
        ) = net.network_dict_to_state_vec(
            flow_dict,
            density_dict,
            speed_dict,
            origin_queue_dict,
            onramp_queue_dict,
            offramp_queue_dict,
        )

        # verify vector structure
        assert isinstance(x, np.ndarray)
        assert (
            len(x) == 9
        )  # expected: origin_flow(1) + origin_queue(1) + main_flows(2) + main_densities(2) + main_speeds(2) + dest_flow(1) = 9
        assert num_flows == 4  # origin(1) + main(2) + dest(1)
        assert num_densities == 2  # main(2)
        assert num_speeds == 2  # main(2)
        assert num_origin == 1
        assert num_onramp == 0
        assert num_offramp == 0
        assert num_destinations == 1
        assert num_splits == 2  # node1(1) + node2(1)

        # verify values in correct order
        np.testing.assert_array_equal(x[0:1], [500.0])  # origin flow
        np.testing.assert_array_equal(x[1:2], [10.0])  # origin queue
        np.testing.assert_array_equal(x[2:4], [1000.0, 1100.0])  # main flows
        np.testing.assert_array_equal(x[4:6], [50.0, 55.0])  # main densities
        np.testing.assert_array_equal(x[6:8], [75.0, 78.0])  # main speeds
        np.testing.assert_array_equal(x[8:9], [1200.0])  # dest flow

    def test_state_vec_to_network_dict_simple_network(self):
        """Test unpacking state vector to dictionaries for simple network."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])
        partition_motorway_links(net)

        # create state vector
        x = np.array([500.0, 10.0, 1000.0, 1100.0, 50.0, 55.0, 75.0, 78.0, 1200.0])

        # unpack
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            net.state_vec_to_network_dict(x)
        )

        # verify dictionaries
        assert origin.id in flows
        assert main.id in flows
        assert dest.id in flows
        assert main.id in densities
        assert main.id in speeds
        assert origin.id in origin_queues
        assert len(onramp_queues) == 0
        assert len(offramp_queues) == 0

        # verify values
        np.testing.assert_array_equal(flows[origin.id], [500.0])
        np.testing.assert_array_equal(flows[main.id], [1000.0, 1100.0])
        np.testing.assert_array_equal(flows[dest.id], [1200.0])
        np.testing.assert_array_equal(densities[main.id], [50.0, 55.0])
        np.testing.assert_array_equal(speeds[main.id], [75.0, 78.0])
        assert origin_queues[origin.id] == 10.0

    def test_round_trip_state_conversion_simple(self):
        """Test dict->vec->dict round-trip conversion preserves all values."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])
        partition_motorway_links(net)

        # original state dictionaries
        flow_dict_orig = {
            origin.id: np.array([500.0]),
            main.id: np.array([1000.0, 1100.0]),
            dest.id: np.array([1200.0]),
        }
        density_dict_orig = {main.id: np.array([50.0, 55.0])}
        speed_dict_orig = {main.id: np.array([75.0, 78.0])}
        origin_queue_dict_orig = {origin.id: 10.0}
        onramp_queue_dict_orig = {}
        offramp_queue_dict_orig = {}

        # round trip: dict -> vec -> dict
        x, *_ = net.network_dict_to_state_vec(
            flow_dict_orig,
            density_dict_orig,
            speed_dict_orig,
            origin_queue_dict_orig,
            onramp_queue_dict_orig,
            offramp_queue_dict_orig,
        )

        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            net.state_vec_to_network_dict(cast(NDArray[np.float64], x))
        )

        # verify all values match
        for link_id in flow_dict_orig:
            np.testing.assert_array_almost_equal(
                flows[link_id], flow_dict_orig[link_id]
            )

        for link_id in density_dict_orig:
            np.testing.assert_array_almost_equal(
                densities[link_id], density_dict_orig[link_id]
            )

        for link_id in speed_dict_orig:
            np.testing.assert_array_almost_equal(
                speeds[link_id], speed_dict_orig[link_id]
            )

        for origin_id in origin_queue_dict_orig:
            assert origin_queues[origin_id] == origin_queue_dict_orig[origin_id]

    def test_network_dict_to_state_vec_complex_network(self):
        """Test state packing for complex network with all link types."""
        # create network with Origin, Onramp, Offramp, MotorwayLink, Destination
        main1 = MotorwayLink(
            length=1.0, lanes=3, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.5, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        onramp = Onramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest1 = Destination()
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest2 = Destination()
        offramp.destination = dest2

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1, onramp], outgoing=[main2, offramp])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest1])

        net = Network(nodes=[node1, node2, node3])
        partition_motorway_links(net)

        # prepare state dictionaries
        num_cells_main1 = len(main1)
        num_cells_main2 = len(main2)

        flow_dict = {
            origin.id: np.array([500.0]),
            onramp.id: np.array([200.0]),
            main1.id: np.ones(num_cells_main1) * 1000.0,
            main2.id: np.ones(num_cells_main2) * 900.0,
            offramp.id: np.array([100.0]),
            dest1.id: np.array([800.0]),
        }
        density_dict = {
            main1.id: np.ones(num_cells_main1) * 50.0,
            main2.id: np.ones(num_cells_main2) * 45.0,
        }
        speed_dict = {
            main1.id: np.ones(num_cells_main1) * 75.0,
            main2.id: np.ones(num_cells_main2) * 70.0,
        }
        origin_queue_dict = {origin.id: 5.0}
        onramp_queue_dict = {onramp.id: 3.0}
        offramp_queue_dict = {offramp.id: 2.0}

        # pack to state vector
        (
            x,
            num_flows,
            num_densities,
            num_speeds,
            num_origin,
            num_onramp,
            num_offramp,
            num_splits,
            num_destinations,
        ) = net.network_dict_to_state_vec(
            flow_dict,
            density_dict,
            speed_dict,
            origin_queue_dict,
            onramp_queue_dict,
            offramp_queue_dict,
        )

        # verify counts
        assert isinstance(x, np.ndarray)
        assert (
            num_flows == 1 + 1 + num_cells_main1 + num_cells_main2 + 1 + 1
        )  # all flows
        assert num_densities == num_cells_main1 + num_cells_main2
        assert num_speeds == num_cells_main1 + num_cells_main2
        assert num_origin == 1
        assert num_onramp == 1
        assert num_offramp == 1
        assert num_destinations == 2  # dest1 + dest2 (connected to offramp)

    def test_round_trip_state_conversion_complex(self):
        """Test dict->vec->dict round-trip for complex network."""
        # create complex network
        main1 = MotorwayLink(
            length=1.0, lanes=3, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.5, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        onramp = Onramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest1 = Destination()
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest2 = Destination()
        offramp.destination = dest2

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1, onramp], outgoing=[main2, offramp])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest1])

        net = Network(nodes=[node1, node2, node3])
        partition_motorway_links(net)

        # original state
        num_cells_main1 = len(main1)
        num_cells_main2 = len(main2)

        flow_dict_orig = {
            origin.id: np.array([500.0]),
            onramp.id: np.array([200.0]),
            main1.id: np.random.rand(num_cells_main1) * 1000.0,
            main2.id: np.random.rand(num_cells_main2) * 900.0,
            offramp.id: np.array([100.0]),
            dest1.id: np.array([800.0]),
        }
        density_dict_orig = {
            main1.id: np.random.rand(num_cells_main1) * 50.0,
            main2.id: np.random.rand(num_cells_main2) * 45.0,
        }
        speed_dict_orig = {
            main1.id: np.random.rand(num_cells_main1) * 75.0,
            main2.id: np.random.rand(num_cells_main2) * 70.0,
        }
        origin_queue_dict_orig = {origin.id: 5.0}
        onramp_queue_dict_orig = {onramp.id: 3.0}
        offramp_queue_dict_orig = {offramp.id: 2.0}

        # round trip
        x, *_ = net.network_dict_to_state_vec(
            flow_dict_orig,
            density_dict_orig,
            speed_dict_orig,
            origin_queue_dict_orig,
            onramp_queue_dict_orig,
            offramp_queue_dict_orig,
        )

        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            net.state_vec_to_network_dict(cast(NDArray[np.float64], x))
        )

        # verify all values match
        for link_id in flow_dict_orig:
            np.testing.assert_array_almost_equal(
                flows[link_id], flow_dict_orig[link_id]
            )

        for link_id in density_dict_orig:
            np.testing.assert_array_almost_equal(
                densities[link_id], density_dict_orig[link_id]
            )

        for link_id in speed_dict_orig:
            np.testing.assert_array_almost_equal(
                speeds[link_id], speed_dict_orig[link_id]
            )

        assert origin_queues[origin.id] == origin_queue_dict_orig[origin.id]
        assert onramp_queues[onramp.id] == onramp_queue_dict_orig[onramp.id]
        assert offramp_queues[offramp.id] == offramp_queue_dict_orig[offramp.id]

    def test_network_dict_to_disturbance_vec_simple(self):
        """Test packing disturbance dictionaries to vector for simple network."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # prepare disturbance dictionaries
        origin_demand_dict = {origin.id: 600.0}
        onramp_demand_dict = {}
        turning_rate_dict = {
            node1.id: {main.id: 1.0},  # node1: all traffic to main
            node2.id: {dest.id: 1.0},  # node2: all traffic to dest
        }
        flow_destination_bc = {dest.id: 1400.0}
        density_destination_bc = {dest.id: 30.0}

        # pack to disturbance vector
        d = net.network_dict_to_disturbance_vec(
            origin_demand_dict,
            onramp_demand_dict,
            turning_rate_dict,
            flow_boundary_condition_dict=flow_destination_bc,
            density_boundary_condition_dict=density_destination_bc,
        )

        # verify vector structure
        assert isinstance(d, np.ndarray)
        assert (
            len(d) == 5
        )  # expected: turning_rate_n1(1) + origin_demand(1) + turning_rate_n2(1) + flow_boundary_cond(1) + density_boundary_cond(1) = 5

        # verify values (order: node1 turning rates, node1 origin demand, node2 turning rates, node2 flow boundary, node2 density boundary)
        assert d[0] == 1.0  # node1 turning rate for main
        assert d[1] == 600.0  # origin demand
        assert d[2] == 1.0  # node2 turning rate for dest
        assert d[3] == 1400.0  # flow boundary condition
        assert d[4] == 30.0  # density boundary condition

    def test_disturbance_vec_to_network_dict_simple(self):
        """Test unpacking disturbance vector to dictionaries."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # create disturbance vector
        d = np.array([1.0, 600.0, 1.0, 1400.0, 30.0])

        # unpack
        (
            origin_demands,
            onramp_demands,
            turning_rates,
            flow_boundary_conditions,
            density_boundary_conditions,
        ) = net.disturbance_vec_to_network_dict(d)

        # verify dictionaries
        assert origin.id in origin_demands
        assert len(onramp_demands) == 0
        assert node1.id in turning_rates
        assert node2.id in turning_rates
        assert dest.id in flow_boundary_conditions
        assert dest.id in density_boundary_conditions

        # verify values
        assert origin_demands[origin.id] == 600.0
        assert turning_rates[node1.id][main.id] == 1.0
        assert turning_rates[node2.id][dest.id] == 1.0
        assert flow_boundary_conditions[dest.id] == 1400.0
        assert density_boundary_conditions[dest.id] == 30.0

    def test_round_trip_disturbance_conversion(self):
        """Test dict->vec->dict round-trip for disturbance conversion."""
        # create network with onramp and offramp
        main1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        onramp = Onramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest1 = Destination()
        dest2 = Destination()
        offramp.destination = dest2

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1, onramp], outgoing=[main2, offramp])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest1])

        net = Network(nodes=[node1, node2, node3])

        # original disturbance dictionaries
        origin_demand_dict_orig = {origin.id: 600.0}
        onramp_demand_dict_orig = {onramp.id: 150.0}
        turning_rate_dict_orig = {
            node1.id: {main1.id: 1.0},
            node2.id: {main2.id: 0.8, offramp.id: 0.2},
            node3.id: {dest1.id: 1.0},
        }
        flow_destination_bc_orig = {dest1.id: 1400.0, dest2.id: 1200.0}
        density_destination_bc_orig = {dest1.id: 30.0, dest2.id: 25.0}

        # round trip: dict -> vec -> dict
        d = net.network_dict_to_disturbance_vec(
            origin_demand_dict_orig,
            onramp_demand_dict_orig,
            turning_rate_dict_orig,
            flow_destination_bc_orig,
            density_destination_bc_orig,
        )

        (
            origin_demands,
            onramp_demands,
            turning_rates,
            flow_boundary_conditions,
            density_boundary_conditions,
        ) = net.disturbance_vec_to_network_dict(d)

        # verify all values match
        assert origin_demands[origin.id] == origin_demand_dict_orig[origin.id]
        assert onramp_demands[onramp.id] == onramp_demand_dict_orig[onramp.id]

        for node_id in turning_rate_dict_orig:
            for link_id in turning_rate_dict_orig[node_id]:
                assert (
                    turning_rates[node_id][link_id]
                    == turning_rate_dict_orig[node_id][link_id]
                )

        for dest_id in flow_destination_bc_orig:
            assert (
                flow_boundary_conditions[dest_id] == flow_destination_bc_orig[dest_id]
            )
        for dest_id in density_destination_bc_orig:
            assert (
                density_boundary_conditions[dest_id]
                == density_destination_bc_orig[dest_id]
            )

    def test_state_vec_too_short_raises(self):
        """Test that unpacking too-short state vector raises ValueError."""
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # too short state vector
        x = np.array([500.0, 10.0])  # only 2 values, needs 11

        with pytest.raises(ValueError, match="State vector too short"):
            net.state_vec_to_network_dict(x)

    def test_disturbance_vec_too_short_raises(self):
        """Test that unpacking too-short disturbance vector raises ValueError."""
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # too short disturbance vector
        d = np.array([1.0])  # only 1 value, needs 5

        with pytest.raises(ValueError, match="Disturbance vector too short"):
            net.disturbance_vec_to_network_dict(d)

    def test_compute_upcoming_lane_drop_with_drop(self):
        """Test lane drop detection when downstream link has fewer lanes."""
        # create network with lane drop: 3 lanes -> 2 lanes
        main1 = MotorwayLink(
            length=1.0, lanes=3, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main3 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1], outgoing=[main2])
        node3 = Node(id="n3", incoming=[main2], outgoing=[main3])
        node4 = Node(id="n4", incoming=[main3], outgoing=[dest])

        net = Network(nodes=[node1, node2, node3, node4])
        partition_motorway_links(net)

        # test lane drop computation
        lane_drop = net._compute_upcoming_lane_drop(main1)
        assert lane_drop == 1  # 3 - 2 = 1 lane dropped
        lane_drop2 = net._compute_upcoming_lane_drop(main2)
        assert lane_drop2 == 0  # no further drop
        lane_drop3 = net._compute_upcoming_lane_drop(main3)
        assert lane_drop3 == 0  # no downstream link

    def test_compute_upcoming_lane_drop_no_drop(self):
        """Test lane drop detection when no lanes are dropped."""
        # create network with same lanes
        main1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1], outgoing=[main2])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest])

        net = Network(nodes=[node1, node2, node3])
        partition_motorway_links(net)

        # test no lane drop
        lane_drop = net._compute_upcoming_lane_drop(main1)
        assert lane_drop == 0

    def test_compute_upcoming_lane_drop_with_offramp(self):
        """Test lane drop detection when downstream is an offramp with fewer lanes."""
        # create network: mainline -> offramp (lane reduction)
        main = MotorwayLink(
            length=1.0, lanes=3, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest = Destination()
        offramp.destination = dest

        origin = Origin()
        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[offramp])

        net = Network(nodes=[node1, node2])
        partition_motorway_links(net)

        # test lane drop to offramp
        lane_drop = net._compute_upcoming_lane_drop(main)
        assert lane_drop == 2  # 3 - 1 = 2 lanes dropped

    def test_compute_upcoming_lane_drop_multiple_outgoing(self):
        """Test lane drop returns 0 when node has multiple outgoing links (not simple continuation)."""
        # create merge node with multiple outgoing links
        main1 = MotorwayLink(
            length=1.0, lanes=3, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest1 = Destination()
        dest2 = Destination()
        offramp.destination = dest2

        origin = Origin()
        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(
            id="n2", incoming=[main1], outgoing=[main2, offramp]
        )  # multiple outgoing
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest1])

        net = Network(nodes=[node1, node2, node3])
        partition_motorway_links(net)

        # should return 0 because node2 has multiple outgoing links (diverge)
        lane_drop = net._compute_upcoming_lane_drop(main1)
        assert lane_drop == 0

    def test_compute_upcoming_lane_drop_missing_destination_raises(self):
        """Test that missing destination_node_id raises ValueError."""
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])

        net = Network(nodes=[node1, node2])
        partition_motorway_links(net)

        # Remove destination_node_id to trigger error
        main.destination_node_id = None

        with pytest.raises(ValueError, match="has no destination_node_id set"):
            net._compute_upcoming_lane_drop(main)

    def test_save_network_structure_txt(self):
        """Test saving network structure to text file."""
        # create simple network
        main = MotorwayLink(
            length=1.5, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network_structure.txt")
            net.save_to_txt(filepath)

            # verify file exists and contains expected content
            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                content = f.read()

            # verify key elements and explicit ids
            assert "NETWORK STRUCTURE" in content
            assert "Total Nodes: 2" in content
            assert "NODE: n1" in content
            assert "NODE: n2" in content
            assert "Origin" in content
            assert "MotorwayLink" in content
            assert "Destination" in content
            assert f"Length: {main.length}" in content
            assert f"Lanes: {main.lanes}" in content

            # verify that the actual link and node ids are stored next to the correct node sections
            # split content into per-node blocks
            lines = content.splitlines()
            node_indices = [i for i, L in enumerate(lines) if L.startswith("NODE:")]
            node_blocks = {}
            for idx, start in enumerate(node_indices):
                end = (
                    node_indices[idx + 1] if idx + 1 < len(node_indices) else len(lines)
                )
                header = lines[start]
                node_id = header.split(":", 1)[1].strip()
                node_blocks[node_id] = "\n".join(lines[start:end])

            # node1 block should contain the origin in incoming and the main link in outgoing
            assert origin.id in node_blocks[node1.id]
            assert main.id in node_blocks[node1.id]
            # verify printed origin/destination node ids in node1 block
            assert (
                f"Origin Node ID: {getattr(origin, 'origin_node_id', 'N/A')}"
                in node_blocks[node1.id]
            )
            assert f"Destination Node ID: {node1.id}" in node_blocks[node1.id]

            # node2 block should contain the main link in incoming and the destination in outgoing
            assert main.id in node_blocks[node2.id]
            assert dest.id in node_blocks[node2.id]
            # verify printed origin/destination node ids in node2 block
            assert f"Origin Node ID: {node2.id}" in node_blocks[node2.id]
            assert (
                f"Destination Node ID: {getattr(dest, 'destination_node_id', 'N/A')}"
                in node_blocks[node2.id]
            )

    def test_circular_network_validates(self):
        """Test that a circular network topology (with feedback loop) passes validation."""
        # create circular network: Origin -> Link1 -> Link2 -> Link3 -> Link4 -> back to Link2
        # add offramp at node 3 for traffic exit
        link1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link3 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link4 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )

        origin = Origin()
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest = Destination()
        offramp.destination = dest

        # Node1: Origin feeds into Link1
        node1 = Node(id="n1", incoming=[origin], outgoing=[link1])
        # Node2: Link1 and Link4 merge into Link2 (this is where the loop closes)
        node2 = Node(id="n2", incoming=[link1, link4], outgoing=[link2])
        # Node3: Link2 splits to Link3 and Offramp
        node3 = Node(id="n3", incoming=[link2], outgoing=[link3, offramp])
        # Node4: Link3 continues to Link4
        node4 = Node(id="n4", incoming=[link3], outgoing=[link4])

        net = Network(nodes=[node1, node2, node3, node4])

        # validation should pass despite circular topology
        assert net.validate() is True

    def test_circular_network_state_conversion(self):
        """Test state vector conversion works correctly for circular network."""
        # create circular network
        link1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link3 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link4 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )

        origin = Origin()
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest = Destination()
        offramp.destination = dest

        node1 = Node(id="n1", incoming=[origin], outgoing=[link1])
        node2 = Node(id="n2", incoming=[link1, link4], outgoing=[link2])
        node3 = Node(id="n3", incoming=[link2], outgoing=[link3, offramp])
        node4 = Node(id="n4", incoming=[link3], outgoing=[link4])

        net = Network(nodes=[node1, node2, node3, node4])
        partition_motorway_links(net)

        # create state dictionaries
        num_cells_1 = len(link1)
        num_cells_2 = len(link2)
        num_cells_3 = len(link3)
        num_cells_4 = len(link4)

        flow_dict = {
            origin.id: np.array([500.0]),
            link1.id: np.ones(num_cells_1) * 500.0,
            link2.id: np.ones(num_cells_2) * 450.0,
            link3.id: np.ones(num_cells_3) * 400.0,
            link4.id: np.ones(num_cells_4) * 400.0,
            offramp.id: np.array([50.0]),
        }
        density_dict = {
            link1.id: np.ones(num_cells_1) * 30.0,
            link2.id: np.ones(num_cells_2) * 32.0,
            link3.id: np.ones(num_cells_3) * 28.0,
            link4.id: np.ones(num_cells_4) * 28.0,
        }
        speed_dict = {
            link1.id: np.ones(num_cells_1) * 70.0,
            link2.id: np.ones(num_cells_2) * 68.0,
            link3.id: np.ones(num_cells_3) * 72.0,
            link4.id: np.ones(num_cells_4) * 72.0,
        }
        origin_queue_dict = {origin.id: 5.0}
        onramp_queue_dict = {}
        offramp_queue_dict = {offramp.id: 2.0}

        # test packing to state vector
        x, *_ = net.network_dict_to_state_vec(
            flow_dict,
            density_dict,
            speed_dict,
            origin_queue_dict,
            onramp_queue_dict,
            offramp_queue_dict,
        )

        # test round-trip conversion
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            net.state_vec_to_network_dict(cast(NDArray[np.float64], x))
        )

        # verify all values preserved
        for link_id in flow_dict:
            np.testing.assert_allclose(flows[link_id], flow_dict[link_id], rtol=1e-10)

        for link_id in density_dict:
            np.testing.assert_allclose(
                densities[link_id], density_dict[link_id], rtol=1e-10
            )

        for link_id in speed_dict:
            np.testing.assert_allclose(speeds[link_id], speed_dict[link_id], rtol=1e-10)

        assert origin_queues[origin.id] == origin_queue_dict[origin.id]
        assert offramp_queues[offramp.id] == offramp_queue_dict[offramp.id]

    def test_circular_network_disturbance_conversion(self):
        """Test disturbance vector conversion for circular network with split ratios."""
        # create circular network
        link1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link2 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link3 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        link4 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )

        origin = Origin()
        offramp = Offramp(
            lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
        )
        dest = Destination()
        offramp.destination = dest

        node1 = Node(id="n1", incoming=[origin], outgoing=[link1])
        node2 = Node(id="n2", incoming=[link1, link4], outgoing=[link2])
        node3 = Node(id="n3", incoming=[link2], outgoing=[link3, offramp])
        node4 = Node(id="n4", incoming=[link3], outgoing=[link4])

        net = Network(nodes=[node1, node2, node3, node4])

        # create disturbance dictionaries with split ratios at diverge node
        origin_demand_dict = {origin.id: 600.0}
        onramp_demand_dict = {}
        turning_rate_dict = {
            node1.id: {link1.id: 1.0},
            node2.id: {link2.id: 1.0},
            node3.id: {link3.id: 0.9, offramp.id: 0.1},  # 90% continue, 10% exit
            node4.id: {link4.id: 1.0},
        }
        flow_destination_bc = {dest.id: 1400.0}
        density_destination_bc = {dest.id: 20.0}

        # test packing
        d = net.network_dict_to_disturbance_vec(
            origin_demand_dict,
            onramp_demand_dict,
            turning_rate_dict,
            flow_boundary_condition_dict=flow_destination_bc,
            density_boundary_condition_dict=density_destination_bc,
        )

        # test round-trip
        (
            origin_demands,
            onramp_demands,
            turning_rates,
            flow_boundary_conditions,
            density_boundary_conditions,
        ) = net.disturbance_vec_to_network_dict(d)

        # verify values preserved
        assert origin_demands[origin.id] == origin_demand_dict[origin.id]
        assert len(onramp_demands) == 0

        for node_id in turning_rate_dict:
            assert node_id in turning_rates
            for link_id in turning_rate_dict[node_id]:
                assert (
                    turning_rates[node_id][link_id]
                    == turning_rate_dict[node_id][link_id]
                )

        assert flow_boundary_conditions[dest.id] == flow_destination_bc[dest.id]
        assert density_boundary_conditions[dest.id] == density_destination_bc[dest.id]

    def test_save_network_to_json_simple(self):
        """Test saving a simple network structure to JSON file."""
        # create simple network
        main = MotorwayLink(
            id="main_link",
            length=1.5,
            lanes=2,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
        )
        origin = Origin(id="origin_1")
        dest = Destination(id="dest_1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network.json")
            net.save_to_json(filepath)

            # verify file exists and can be read
            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)

            # verify structure
            assert "nodes" in data
            assert "links" in data
            assert len(data["nodes"]) == 2
            assert len(data["links"]) == 3  # origin, main, dest

            # verify that the nodes and links have the expected IDs and attributes
            node_ids = {node["id"] for node in data["nodes"]}
            assert "n1" in node_ids
            assert "n2" in node_ids
            link_ids = {link["id"] for link in data["links"]}
            assert "origin_1" in link_ids
            assert "main_link" in link_ids
            assert "dest_1" in link_ids
            main_link_data = next(
                link for link in data["links"] if link["id"] == "main_link"
            )
            assert main_link_data["length"] == 1.5
            assert main_link_data["lanes"] == 2
            assert main_link_data["lane_capacity"] == 1500
            assert main_link_data["free_flow_speed"] == 80
            assert main_link_data["jam_density"] == 140

    def test_load_network_from_json_simple(self):
        """Test loading a simple network structure from JSON file."""
        # create simple network
        main = MotorwayLink(
            id="main_link",
            length=1.5,
            lanes=2,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
        )
        origin = Origin(id="origin_1")
        dest = Destination(id="dest_1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network.json")
            net.save_to_json(filepath)
            loaded_net = Network.load_from_json(filepath)

            # verify loaded network structure
            assert len(loaded_net) == 2
            assert loaded_net.get_node("n1") is not None
            assert loaded_net.get_node("n2") is not None

            # verify links
            loaded_node1 = loaded_net.get_node("n1")
            loaded_node2 = loaded_net.get_node("n2")
            assert loaded_node1 is not None
            assert loaded_node2 is not None
            assert len(loaded_node1.incoming) == 1
            assert len(loaded_node1.outgoing) == 1
            assert len(loaded_node2.incoming) == 1
            assert len(loaded_node2.outgoing) == 1

            # verify link types
            assert isinstance(loaded_node1.incoming[0], Origin)
            assert isinstance(loaded_node1.outgoing[0], MotorwayLink)
            assert isinstance(loaded_node2.outgoing[0], Destination)

            # verify that the properties are loaded correctly
            loaded_main = loaded_node1.outgoing[0]
            assert loaded_node1.outgoing[0].id == main.id
            assert loaded_main.id == main.id
            assert loaded_main.length == main.length
            assert loaded_main.lanes == main.lanes
            assert loaded_main.Qc_lane == main.Qc_lane
            assert loaded_main.vf == main.vf
            assert loaded_main.rho_jam == main.rho_jam

            loaded_origin = loaded_node1.incoming[0]
            assert loaded_node1.incoming[0].id == origin.id
            assert loaded_origin.id == origin.id

            loaded_dest = loaded_node2.outgoing[0]
            assert loaded_node2.outgoing[0].id == dest.id
            assert loaded_dest.id == dest.id

    def test_round_trip_json_simple_network(self):
        """Test that save->load preserves network structure for simple network."""
        # create simple network with explicit IDs
        main = MotorwayLink(
            length=1.5,
            lanes=2,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            id="main_link",
        )
        origin = Origin(id="origin_1")
        dest = Destination(id="dest_1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network.json")
            net.save_to_json(filepath)
            loaded_net = Network.load_from_json(filepath)

            # verify network-level properties
            assert len(loaded_net) == len(net)
            assert len(loaded_net.list_nodes()) == len(net.list_nodes())

            # verify node IDs preserved
            for node in net.list_nodes():
                loaded_node = loaded_net.get_node(node.id)
                assert loaded_node is not None
                assert loaded_node.id == node.id

            # verify link counts
            for orig_node in net.list_nodes():
                loaded_node = loaded_net.get_node(orig_node.id)
                assert loaded_node is not None
                assert len(loaded_node.incoming) == len(orig_node.incoming)
                assert len(loaded_node.outgoing) == len(orig_node.outgoing)
                for link in loaded_node.incoming:
                    assert link.id in [l.id for l in orig_node.incoming]
                for link in loaded_node.outgoing:
                    assert link.id in [l.id for l in orig_node.outgoing]

    def test_round_trip_json_preserves_link_attributes(self):
        """Test that motorway link attributes are preserved through save/load."""
        # create network with specific link attributes
        main = MotorwayLink(
            id="main_link",
            length=2.5,
            lanes=3,
            lane_capacity=1800,
            free_flow_speed=120,
            jam_density=160,
        )
        origin = Origin(id="origin_1")
        dest = Destination(id="dest_1")

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network.json")
            net.save_to_json(filepath)
            loaded_net = Network.load_from_json(filepath)

            # find the motorway link in loaded network
            loaded_node1 = loaded_net.get_node("n1")
            assert loaded_node1 is not None
            loaded_main = loaded_node1.outgoing[0]

            # verify all attributes preserved
            assert isinstance(loaded_main, MotorwayLink)
            assert loaded_main.id == main.id
            assert loaded_main.length == main.length
            assert loaded_main.lanes == main.lanes
            assert loaded_main.Qc_lane == main.Qc_lane
            assert loaded_main.vf == main.vf
            assert loaded_main.rho_jam == main.rho_jam

    def test_round_trip_json_complex_network(self):
        """Test save/load for complex network with all link types."""
        # create complex network
        main1 = MotorwayLink(
            length=1.0,
            lanes=3,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            id="main1",
        )
        main2 = MotorwayLink(
            length=1.5,
            lanes=2,
            lane_capacity=1500,
            free_flow_speed=80,
            jam_density=140,
            id="main2",
        )
        origin = Origin(id="origin_1")
        onramp = Onramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            id="onramp_1",
        )
        dest1 = Destination(id="dest_1")
        offramp = Offramp(
            lanes=1,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
            id="offramp_1",
        )
        dest2 = Destination(id="dest_2")
        offramp.destination = dest2

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1, onramp], outgoing=[main2, offramp])
        node3 = Node(id="n3", incoming=[main2], outgoing=[dest1])

        net = Network(nodes=[node1, node2, node3])

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network.json")
            net.save_to_json(filepath)
            loaded_net = Network.load_from_json(filepath)

            # verify network structure
            assert len(loaded_net) == 3
            loaded_node1 = loaded_net.get_node("n1")
            loaded_node2 = loaded_net.get_node("n2")
            loaded_node3 = loaded_net.get_node("n3")

            assert loaded_node1 is not None
            assert loaded_node2 is not None
            assert loaded_node3 is not None

            # verify node1
            assert len(loaded_node1.incoming) == 1
            assert len(loaded_node1.outgoing) == 1
            assert isinstance(loaded_node1.incoming[0], Origin)
            assert isinstance(loaded_node1.outgoing[0], MotorwayLink)

            # verify node2 (has merge and diverge)
            assert len(loaded_node2.incoming) == 2
            assert len(loaded_node2.outgoing) == 2
            incoming_types = {type(link).__name__ for link in loaded_node2.incoming}
            outgoing_types = {type(link).__name__ for link in loaded_node2.outgoing}
            assert "MotorwayLink" in incoming_types
            assert "Onramp" in incoming_types
            assert "MotorwayLink" in outgoing_types
            assert "Offramp" in outgoing_types

            # verify node3
            assert len(loaded_node3.incoming) == 1
            assert len(loaded_node3.outgoing) == 1
            assert isinstance(loaded_node3.incoming[0], MotorwayLink)
            assert isinstance(loaded_node3.outgoing[0], Destination)

    def test_load_from_json_validates_successfully(self):
        """Test that loaded network passes validation."""
        # create valid network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "network.json")
            net.save_to_json(filepath)
            loaded_net = Network.load_from_json(filepath)

            # should not raise
            assert loaded_net.validate() is True

    def test_load_from_json_missing_file_raises(self):
        """Test that loading from non-existent file raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "nonexistent.json")

            with pytest.raises(FileNotFoundError):
                Network.load_from_json(filepath)

    def test_save_simulation_results_json(self):
        """Test that simulation results can be saved to JSON format."""
        # create simple network and run simulation
        from traffic_flow_models import CTM

        main = MotorwayLink(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # run short simulation
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.1,
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            plot_results=False,
        )

        # save results
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.1,
                preferred_cell_size=0.5,
                model_params=None,
            )

            # verify file exists and is valid JSON
            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)

            # check structure
            assert "time_array" in data
            assert "state_time_series" in data
            assert "disturbance_time_series" in data
            assert len(data["time_array"]) == len(time_array)

    def test_load_simulation_results_json(self):
        """Test that simulation results can be loaded from JSON with validation."""
        from traffic_flow_models import CTM

        # create simple network
        main = MotorwayLink(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # run simulation
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.1,
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            plot_results=False,
        )

        # save and load results
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.1,
                preferred_cell_size=0.5,
                model_params=None,
            )

            loaded_time, loaded_state, loaded_disturbance, metadata = (
                Network.load_simulation_results_json(filepath=filepath, network=net)
            )

            # verify loaded arrays match original
            np.testing.assert_array_almost_equal(loaded_time, time_array)
            np.testing.assert_array_almost_equal(loaded_state, state_history)
            np.testing.assert_array_almost_equal(
                loaded_disturbance, disturbance_history
            )

    def test_save_load_json_includes_metadata(self):
        """Test that JSON save/load includes comprehensive metadata for reproducibility."""
        from traffic_flow_models import CTM

        # create network
        main = MotorwayLink(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # run simulation
        model = CTM()
        dt = 0.01
        duration = 0.1
        preferred_cell_size = 0.5
        time_array, state_history, disturbance_history = net.simulate(
            duration=duration,
            dt=dt,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            preferred_cell_size=preferred_cell_size,
            plot_results=False,
        )

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results_with_metadata.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=dt,
                duration=duration,
                preferred_cell_size=preferred_cell_size,
                model_params=None,
            )

            # verify metadata in file
            with open(filepath, "r") as f:
                data = json.load(f)

            assert "metadata" in data
            metadata_dict = data["metadata"]

            # check model type
            assert "model_type" in metadata_dict
            assert metadata_dict["model_type"] == "CTM"

            # check simulation parameters
            assert "simulation_parameters" in metadata_dict
            sim_params = metadata_dict["simulation_parameters"]
            assert sim_params["dt"] == dt
            assert sim_params["duration"] == duration
            assert sim_params["preferred_cell_size"] == preferred_cell_size

            # check link properties
            assert "link_properties" in metadata_dict
            link_props = metadata_dict["link_properties"]
            assert main.id in link_props
            assert link_props[main.id]["length"] == 2.0
            assert link_props[main.id]["lanes"] == 3
            assert link_props[main.id]["lane_capacity"] == 2000
            assert link_props[main.id]["free_flow_speed"] == 100
            assert link_props[main.id]["jam_density"] == 180

            # check cell discretization info
            assert "num_cells" in link_props[main.id]
            assert link_props[main.id]["num_cells"] == len(main)
            assert "cell_lengths" in link_props[main.id]
            assert isinstance(link_props[main.id]["cell_lengths"], list)
            assert len(link_props[main.id]["cell_lengths"]) == len(main)
            # verify sum of cell lengths equals total link length
            assert abs(sum(link_props[main.id]["cell_lengths"]) - 2.0) < 1e-6

            # check critical densities
            assert "critical_densities" in metadata_dict
            crit_densities = metadata_dict["critical_densities"]
            assert main.id in crit_densities
            expected_rho_crit = model.critical_density(
                lane_capacity=2000, free_flow_speed=100
            )
            assert abs(crit_densities[main.id] - expected_rho_crit) < 1e-6

            # load and verify metadata is returned
            loaded_time, loaded_state, loaded_disturbance, loaded_metadata = (
                Network.load_simulation_results_json(filepath=filepath, network=net)
            )

            assert loaded_metadata is not None
            assert loaded_metadata["model_type"] == "CTM"
            assert loaded_metadata["simulation_parameters"]["dt"] == dt
            assert (
                loaded_metadata["critical_densities"][main.id]
                == crit_densities[main.id]
            )

    def test_load_simulation_results_json_validates_structure(self):
        """Test that loading simulation results validates required fields."""
        # create minimal network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        with tempfile.TemporaryDirectory() as tmpdir:
            # test missing top-level fields
            filepath = os.path.join(tmpdir, "incomplete.json")

            # missing time_array
            with open(filepath, "w") as f:
                json.dump({"state_time_series": {}, "disturbance_time_series": {}}, f)

            with pytest.raises(ValueError, match="Missing required field 'time_array'"):
                Network.load_simulation_results_json(filepath=filepath, network=net)

            # missing state fields
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "time_array": [0, 1, 2],
                        "state_time_series": {"flows": {}},
                        "disturbance_time_series": {},
                    },
                    f,
                )

            with pytest.raises(ValueError, match="Missing required field.*densities"):
                Network.load_simulation_results_json(filepath=filepath, network=net)

    def test_load_simulation_results_json_validates_network_match(self):
        """Test that loader validates saved data matches network structure."""
        from traffic_flow_models import CTM

        # create and simulate first network
        main1 = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin1 = Origin()
        dest1 = Destination()

        node1 = Node(id="n1", incoming=[origin1], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1], outgoing=[dest1])
        net1 = Network(nodes=[node1, node2])

        model = CTM()
        time_array, state_history, disturbance_history = net1.simulate(
            duration=0.05,
            dt=0.01,
            model=model,
            origin_demands={origin1.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest1.id: lambda t: 0.0},
            destination_density_bc={dest1.id: lambda t: 0.0},
            plot_results=False,
        )

        # save results from first network
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net1.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.05,
                preferred_cell_size=0.5,
                model_params=None,
            )

            # create different network with different structure
            main2 = MotorwayLink(
                length=2.0,
                lanes=3,
                lane_capacity=2000,
                free_flow_speed=100,
                jam_density=180,
            )
            onramp = Onramp(
                lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
            )
            dest2 = Destination()

            node3 = Node(id="n3", incoming=[onramp], outgoing=[main2])
            node4 = Node(id="n4", incoming=[main2], outgoing=[dest2])
            net2 = Network(nodes=[node3, node4])

            # trying to load results from net1 into net2 should fail validation
            with pytest.raises(ValueError, match="not found in saved results"):
                Network.load_simulation_results_json(filepath=filepath, network=net2)

    def test_save_load_complex_network_with_onramps_offramps(self):
        """Test save/load with complex network including onramps and offramps."""
        from traffic_flow_models import CTM

        # create complex network
        origin = Origin()
        main1 = MotorwayLink(
            length=3.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        )
        onramp = Onramp(
            lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        main2 = MotorwayLink(
            length=2.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        )
        offramp = Offramp(
            lanes=1, lane_capacity=1500, free_flow_speed=60, jam_density=140
        )
        dest_offramp = Destination()
        dest_main = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main1])
        node2 = Node(id="n2", incoming=[main1, onramp], outgoing=[main2])
        node3 = Node(id="n3", incoming=[main2], outgoing=[offramp, dest_main])

        # Note: Offramps connect to destinations via their destination property
        net = Network(nodes=[node1, node2, node3])
        offramp.destination = dest_offramp

        # simulate
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.05,
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 2000.0},
            onramp_demands={onramp.id: lambda t: 500.0},
            turning_rates={"n3": lambda t: {offramp.id: 0.3, dest_main.id: 0.7}},
            destination_flow_bc={
                dest_offramp.id: lambda t: 0.0,
                dest_main.id: lambda t: 0.0,
            },
            destination_density_bc={
                dest_offramp.id: lambda t: 0.0,
                dest_main.id: lambda t: 0.0,
            },
            plot_results=False,
        )

        # save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "complex_results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.05,
                preferred_cell_size=0.5,
                model_params=None,
            )

            # verify file contents
            with open(filepath, "r") as f:
                data = json.load(f)

            # check all link types are present
            assert origin.id in data["state_time_series"]["origin_queues"]
            assert onramp.id in data["state_time_series"]["onramp_queues"]
            assert offramp.id in data["state_time_series"]["offramp_queues"]
            assert main1.id in data["state_time_series"]["flows"]
            assert main1.id in data["state_time_series"]["densities"]
            assert main1.id in data["state_time_series"]["speeds"]

            # check disturbance data
            assert origin.id in data["disturbance_time_series"]["origin_demands"]
            assert onramp.id in data["disturbance_time_series"]["onramp_demands"]
            assert "n3" in data["disturbance_time_series"]["turning_rates"]

            # load and verify
            loaded_time, loaded_state, loaded_disturbance, metadata = (
                Network.load_simulation_results_json(filepath=filepath, network=net)
            )

            np.testing.assert_array_almost_equal(loaded_time, time_array)
            np.testing.assert_array_almost_equal(loaded_state, state_history)
            np.testing.assert_array_almost_equal(
                loaded_disturbance, disturbance_history
            )

    def test_load_simulation_results_validates_numerical_data(self):
        """Test that loader detects non-numerical data."""
        from traffic_flow_models import CTM

        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # simulate to get valid structure
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.02,
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            plot_results=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.02,
                preferred_cell_size=0.5,
                model_params=None,
            )

            # corrupt the data with non-numeric values
            with open(filepath, "r") as f:
                data = json.load(f)

            # replace numeric data with strings
            origin_queue_key = list(data["state_time_series"]["origin_queues"].keys())[
                0
            ]
            data["state_time_series"]["origin_queues"][origin_queue_key][0] = "invalid"

            # save corrupted data
            corrupted_path = os.path.join(tmpdir, "corrupted.json")
            with open(corrupted_path, "w") as f:
                json.dump(data, f)

            # loading should fail validation (Python's float() will raise ValueError)
            with pytest.raises(ValueError, match="could not convert string to float"):
                Network.load_simulation_results_json(
                    filepath=corrupted_path, network=net
                )

    def test_save_load_preserves_link_ids(self):
        """Test that save/load preserves exact link IDs for all link types."""
        from traffic_flow_models import CTM

        # create network with explicit IDs
        origin = Origin()
        origin_id = origin.id
        main = MotorwayLink(
            length=2.0, lanes=2, lane_capacity=1800, free_flow_speed=90, jam_density=150
        )
        main_id = main.id
        dest = Destination()
        dest_id = dest.id

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.03,
            dt=0.01,
            model=model,
            origin_demands={origin_id: lambda t: 1500.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest_id: lambda t: 0.0},
            destination_density_bc={dest_id: lambda t: 0.0},
            plot_results=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.03,
                preferred_cell_size=0.5,
                model_params=None,
            )

            with open(filepath, "r") as f:
                data = json.load(f)

            # verify exact IDs are preserved
            assert origin_id in data["state_time_series"]["origin_queues"]
            assert main_id in data["state_time_series"]["flows"]
            assert main_id in data["state_time_series"]["densities"]
            assert main_id in data["state_time_series"]["speeds"]
            assert dest_id in data["state_time_series"]["flows"]

    def test_load_simulation_results_with_multiple_timesteps(self):
        """Test that loader correctly handles data with many timesteps."""
        from traffic_flow_models import CTM

        # create simple network
        main = MotorwayLink(
            length=5.0,
            lanes=3,
            lane_capacity=2000,
            free_flow_speed=100,
            jam_density=180,
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # simulate with more timesteps
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.5,  # longer duration
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0 + 500.0 * t},  # time-varying
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            plot_results=False,
        )

        assert len(time_array) > 10  # ensure we have many timesteps

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "multi_timestep.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.5,
                preferred_cell_size=0.5,
                model_params=None,
            )

            with open(filepath, "r") as f:
                data = json.load(f)

            # verify all timesteps are saved
            origin_queue_data = data["state_time_series"]["origin_queues"][origin.id]
            assert len(origin_queue_data) == len(time_array)

            # load and verify
            loaded_time, loaded_state, loaded_disturbance, metadata = (
                Network.load_simulation_results_json(filepath=filepath, network=net)
            )

            # check dimensions
            assert loaded_state.shape[1] == len(time_array)
            assert loaded_disturbance.shape[1] == len(time_array) - 1

            np.testing.assert_array_almost_equal(loaded_time, time_array)
            np.testing.assert_array_almost_equal(loaded_state, state_history)

    def test_load_simulation_results_missing_link_data(self):
        """Test that loader detects when link data is missing from saved file."""
        from traffic_flow_models import CTM

        # create network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # simulate
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.02,
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            plot_results=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.02,
                preferred_cell_size=0.5,
                model_params=None,
            )

            # load and remove a required field
            with open(filepath, "r") as f:
                data = json.load(f)

            # remove density data for main link
            del data["state_time_series"]["densities"][main.id]

            # save modified data
            modified_path = os.path.join(tmpdir, "missing_link.json")
            with open(modified_path, "w") as f:
                json.dump(data, f)

            # loading should fail with clear error message
            with pytest.raises(
                ValueError, match=f"Density data for motorway link '{main.id}'"
            ):
                Network.load_simulation_results_json(
                    filepath=modified_path, network=net
                )

    def test_validate_disturbance_history_numerical_valid_data(self):
        """Test that validate_disturbance_history_numerical accepts valid numerical data."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # valid disturbance data
        origin_demands = {origin.id: 1000.0}
        onramp_demands = {}
        turning_rates = {
            "n1": {main.id: 1.0},
            "n2": {dest.id: 1.0},
        }
        flow_bc = {dest.id: 0.0}
        density_bc = {dest.id: 0.0}

        # should not raise
        net._validate_disturbance_history_numerical(
            origin_demands=origin_demands,
            onramp_demands=onramp_demands,
            turning_rates=turning_rates,
            flow_boundary_conditions=flow_bc,
            density_boundary_conditions=density_bc,
        )

    def test_validate_disturbance_history_numerical_rejects_non_numerical_origin_demands(
        self,
    ):
        """Test that validate_disturbance_history_numerical rejects non-numerical origin demands."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # invalid origin demands (string instead of number)
        origin_demands = {origin.id: "invalid"}
        onramp_demands = {}
        turning_rates = {"n1": {main.id: 1.0}, "n2": {dest.id: 1.0}}
        flow_bc = {dest.id: 0.0}
        density_bc = {dest.id: 0.0}

        with pytest.raises(ValueError, match="Non-numerical values found"):
            net._validate_disturbance_history_numerical(
                origin_demands=origin_demands,  # type: ignore (intentional error for testing)
                onramp_demands=onramp_demands,
                turning_rates=turning_rates,
                flow_boundary_conditions=flow_bc,
                density_boundary_conditions=density_bc,
            )

    def test_validate_disturbance_history_numerical_rejects_non_numerical_onramp_demands(
        self,
    ):
        """Test that validate_disturbance_history_numerical rejects non-numerical onramp demands."""
        # create network with onramp
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        onramp = Onramp(
            lanes=1, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        dest = Destination()

        node1 = Node(id="n1", incoming=[onramp], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # invalid onramp demands
        origin_demands = {}
        onramp_demands = {onramp.id: [1000.0]}  # list instead of scalar
        turning_rates = {"n1": {main.id: 1.0}, "n2": {dest.id: 1.0}}
        flow_bc = {dest.id: 0.0}
        density_bc = {dest.id: 0.0}

        with pytest.raises(ValueError, match="Non-numerical values found"):
            net._validate_disturbance_history_numerical(
                origin_demands=origin_demands,
                onramp_demands=onramp_demands,  # type: ignore (intentional error for testing)
                turning_rates=turning_rates,
                flow_boundary_conditions=flow_bc,
                density_boundary_conditions=density_bc,
            )

    def test_validate_disturbance_history_numerical_rejects_non_dict_turning_rates(
        self,
    ):
        """Test that validate_disturbance_history_numerical rejects non-dict turning rates."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # invalid turning rates (not a dict for node)
        origin_demands = {origin.id: 1000.0}
        onramp_demands = {}
        turning_rates = {"n1": 1.0, "n2": {dest.id: 1.0}}  # scalar instead of dict
        flow_bc = {dest.id: 0.0}
        density_bc = {dest.id: 0.0}

        with pytest.raises(ValueError, match="must be a dictionary"):
            net._validate_disturbance_history_numerical(
                origin_demands=origin_demands,
                onramp_demands=onramp_demands,
                turning_rates=turning_rates,
                flow_boundary_conditions=flow_bc,
                density_boundary_conditions=density_bc,
            )

    def test_validate_disturbance_history_numerical_rejects_non_numerical_turning_rates(
        self,
    ):
        """Test that validate_disturbance_history_numerical rejects non-numerical turning rate values."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # invalid turning rate values (string instead of number)
        origin_demands = {origin.id: 1000.0}
        onramp_demands = {}
        turning_rates = {"n1": {main.id: "invalid"}, "n2": {dest.id: 1.0}}
        flow_bc = {dest.id: 0.0}
        density_bc = {dest.id: 0.0}

        with pytest.raises(
            ValueError, match="Non-numerical turning rate values found for node"
        ):
            net._validate_disturbance_history_numerical(
                origin_demands=origin_demands,
                onramp_demands=onramp_demands,
                turning_rates=turning_rates,
                flow_boundary_conditions=flow_bc,
                density_boundary_conditions=density_bc,
            )

    def test_validate_disturbance_history_numerical_rejects_non_numerical_flow_bc(self):
        """Test that validate_disturbance_history_numerical rejects non-numerical flow boundary conditions."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # invalid flow boundary conditions
        origin_demands = {origin.id: 1000.0}
        onramp_demands = {}
        turning_rates = {"n1": {main.id: 1.0}, "n2": {dest.id: 1.0}}
        flow_bc = {dest.id: None}  # None instead of number
        density_bc = {dest.id: 0.0}

        with pytest.raises(ValueError, match="Non-numerical values found"):
            net._validate_disturbance_history_numerical(
                origin_demands=origin_demands,
                onramp_demands=onramp_demands,
                turning_rates=turning_rates,
                flow_boundary_conditions=flow_bc,  # type: ignore (intentional error for testing)
                density_boundary_conditions=density_bc,
            )

    def test_validate_disturbance_history_numerical_rejects_non_numerical_density_bc(
        self,
    ):
        """Test that validate_disturbance_history_numerical rejects non-numerical density boundary conditions."""
        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # invalid density boundary conditions
        origin_demands = {origin.id: 1000.0}
        onramp_demands = {}
        turning_rates = {"n1": {main.id: 1.0}, "n2": {dest.id: 1.0}}
        flow_bc = {dest.id: 0.0}
        density_bc = {dest.id: {"invalid": "dict"}}  # dict instead of number

        with pytest.raises(ValueError, match="Non-numerical values found"):
            net._validate_disturbance_history_numerical(
                origin_demands=origin_demands,
                onramp_demands=onramp_demands,
                turning_rates=turning_rates,
                flow_boundary_conditions=flow_bc,
                density_boundary_conditions=density_bc,  # type: ignore (intentional error for testing)
            )

    def test_load_simulation_results_validates_disturbance_data(self):
        """Test that loader validates disturbance data using validate_disturbance_history_numerical."""
        from traffic_flow_models import CTM

        # create simple network
        main = MotorwayLink(
            length=1.0, lanes=2, lane_capacity=1500, free_flow_speed=80, jam_density=140
        )
        origin = Origin()
        dest = Destination()

        node1 = Node(id="n1", incoming=[origin], outgoing=[main])
        node2 = Node(id="n2", incoming=[main], outgoing=[dest])
        net = Network(nodes=[node1, node2])

        # simulate
        model = CTM()
        time_array, state_history, disturbance_history = net.simulate(
            duration=0.02,
            dt=0.01,
            model=model,
            origin_demands={origin.id: lambda t: 1000.0},
            onramp_demands={},
            turning_rates={},
            destination_flow_bc={dest.id: lambda t: 0.0},
            destination_density_bc={dest.id: lambda t: 0.0},
            plot_results=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            net.save_simulation_results_json(
                time_array=time_array,
                state_history=state_history,
                disturbance_history=disturbance_history,
                filepath=filepath,
                model=model,
                dt=0.01,
                duration=0.02,
                preferred_cell_size=0.5,
                model_params=None,
            )

            # corrupt disturbance data
            with open(filepath, "r") as f:
                data = json.load(f)

            # replace numeric data with invalid values in origin demands
            origin_demand_key = list(
                data["disturbance_time_series"]["origin_demands"].keys()
            )[0]
            data["disturbance_time_series"]["origin_demands"][origin_demand_key][
                0
            ] = "corrupted"

            # save corrupted data
            corrupted_path = os.path.join(tmpdir, "corrupted_disturbance.json")
            with open(corrupted_path, "w") as f:
                json.dump(data, f)

            # loading should fail validation
            with pytest.raises(ValueError, match="could not convert string to float"):
                Network.load_simulation_results_json(
                    filepath=corrupted_path, network=net
                )
