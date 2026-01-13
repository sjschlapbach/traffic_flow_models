import pytest
from typing import Any, cast

from traffic_flow_models.network.network import Network
from traffic_flow_models.network.node import Node
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.motorway_link import MotorwayLink


def test_add_node_duplicate_id_raises():
    net = Network()
    n1 = Node(id="n1")
    net.add_node(n1)
    n1_dup = Node(id="n1")
    with pytest.raises(ValueError):
        net.add_node(n1_dup)


def test_add_node_requires_node_type():
    net = Network()
    # cast to ``Any`` to avoid static type-checker/linter complaining
    with pytest.raises(TypeError):
        net.add_node(cast(Any, object()))


def test_add_remove_get_list_iteration():
    net = Network()
    n1 = Node(id="a")
    n2 = Node(id="b")
    net.add_node(n1)
    net.add_node(n2)

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


def test_validate_path_connected_nodes():
    net = Network()

    # create shared mainline link between node1 -> node2
    main = MotorwayLink()

    origin = Origin()
    dest = Destination()

    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[dest])

    net.add_node(node1)
    net.add_node(node2)

    # should not raise
    net.validate()


def test_validate_offramp_without_destination_raises():
    net = Network()

    # create an offramp without destination
    offr = Offramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        destination=None,
    )
    main = MotorwayLink()
    origin = Origin()

    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[offr])

    net.add_node(node1)
    net.add_node(node2)

    with pytest.raises(ValueError):
        net.validate()


def test_validate_unconnected_component_raises():
    net = Network()

    main = MotorwayLink()
    origin = Origin()
    dest = Destination()

    # connected pair
    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[dest])

    # isolated node (links not shared)
    origin2 = Origin()
    dest2 = Destination()
    main2 = MotorwayLink()
    node3 = Node(id="n3", incoming=[origin2], outgoing=[main2])
    node4 = Node(id="n4", incoming=[main2], outgoing=[dest2])

    net.add_node(node1)
    net.add_node(node2)
    net.add_node(node3)
    net.add_node(node4)

    with pytest.raises(ValueError):
        net.validate()


def test_validate_onramp_without_origin_passes():
    net = Network()

    # create mainline and an onramp feeding into it (no Origin present)
    main = MotorwayLink()
    onr = Onramp(lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180)
    dest = Destination()

    node_upstream = Node(id="up", incoming=[onr], outgoing=[main])
    node_downstream = Node(id="down", incoming=[main], outgoing=[dest])

    net.add_node(node_upstream)
    net.add_node(node_downstream)

    # should not raise: network contains an onramp (counts as origin-type link) and a destination
    net.validate()


def test_validate_requires_origin_or_onramp_raises():
    net = Network()

    # network with only motorway links and a destination but no Origin/Onramp
    main = MotorwayLink()
    dest = Destination()

    node1 = Node(id="n1", incoming=[main], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[dest])

    net.add_node(node1)
    net.add_node(node2)

    with pytest.raises(ValueError):
        net.validate()


def test_validate_requires_destination_raises():
    net = Network()

    # network with Origin and motorway links but no Destination anywhere
    main = MotorwayLink()
    origin = Origin()

    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[main])

    net.add_node(node1)
    net.add_node(node2)

    with pytest.raises(ValueError):
        net.validate()


def test_node_missing_incoming_raises():
    net = Network()

    # node with no incoming links
    node = Node(id="bad", incoming=[], outgoing=[MotorwayLink()])
    net.add_node(node)
    with pytest.raises(ValueError):
        net.validate()


def test_node_missing_outgoing_raises():
    net = Network()

    # node with no outgoing links
    node = Node(id="bad2", incoming=[MotorwayLink()], outgoing=[])
    net.add_node(node)
    with pytest.raises(ValueError):
        net.validate()


def test_validate_incoming_destination_id_mismatch_raises():
    net = Network()

    main = MotorwayLink()
    origin = Origin()
    dest = Destination()

    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[dest])

    net.add_node(node1)
    net.add_node(node2)

    # corrupt the destination id stored on the main link
    main.destination_node_id = "wrong"

    with pytest.raises(ValueError):
        net.validate()


def test_validate_outgoing_origin_id_mismatch_raises():
    net = Network()

    main = MotorwayLink()
    origin = Origin()
    dest = Destination()

    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[dest])

    net.add_node(node1)
    net.add_node(node2)

    # corrupt the origin id stored on the main link
    main.origin_node_id = "wrong"

    with pytest.raises(ValueError):
        net.validate()


def test_validate_missing_destination_or_origin_id_raises():
    net = Network()

    main = MotorwayLink()
    origin = Origin()
    dest = Destination()

    node1 = Node(id="n1", incoming=[origin], outgoing=[main])
    node2 = Node(id="n2", incoming=[main], outgoing=[dest])

    net.add_node(node1)
    net.add_node(node2)

    # remove origin/destination ids
    main.origin_node_id = None
    with pytest.raises(ValueError):
        net.validate()

    # restore origin id and remove destination id instead
    main.origin_node_id = node1.id
    main.destination_node_id = None
    with pytest.raises(ValueError):
        net.validate()


def test_network_validate_rejects_invalid_link_types_set_directly():
    net = Network()

    # create a node and bypass the Node helpers by assigning lists directly
    main = MotorwayLink()
    n = Node(id="bad")

    # invalid incoming type (Offramp is not allowed as incoming)
    n.incoming = [
        Offramp(lanes=1, lane_capacity=1000, free_flow_speed=80, jam_density=140)
    ]
    n.outgoing = [main]
    net.add_node(n)

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
