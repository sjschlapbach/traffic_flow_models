from traffic_flow_models import (
    MotorwayLink,
    Onramp,
    Offramp,
    Destination,
    Origin,
    Node,
    Network,
)


def demand(time: float, t1: float, t2: float, end: float, max: float) -> float:
    if time < t1:
        return time * max / t1
    elif time > end:
        return 0.0
    elif time > t2:
        return max - max * (time - t2) / (end - t2)
    else:
        return max


def mainline_demand_a(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 4000)


def mainline_demand_b(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 4000)


def mainline_demand_c(time: float) -> float:
    return demand(time, 450 / 3600, 3150 / 3600, 3600 / 3600, 1500)


def onramp_demand_a(time: float) -> float:
    return demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 2000)


def onramp_demand_b(time: float) -> float:
    return demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 2500)


def onramp_demand_c(time: float) -> float:
    return demand(time, 900 / 3600, 2700 / 3600, 3600 / 3600, 1500)


def setup_network_ab() -> tuple[Network, dict]:
    """Create a simple linear `Network` with an onramp attached to the middle link.

    Returns `(network, metadata)` where `metadata` contains the ids of
    origins, onramps and destinations and a `splits` mapping for nodes.
    """

    # three motorway segments approximating the original 6 cells (0.5 km each)
    m1 = MotorwayLink(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m2 = MotorwayLink(
        length=1.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    # add origin, destination and onramp
    origin = Origin(id=None)
    destination = Destination(id=None)
    onr = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )

    # connect the links through nodes and build the network structure
    n0 = Node(incoming=[origin], outgoing=[m1])
    n1 = Node(incoming=[m1, onr], outgoing=[m2])
    n2 = Node(incoming=[m2], outgoing=[destination])
    net = Network(nodes=[n0, n1, n2])

    splits = {
        n0.id: {m1.id: 1.0},
        n1.id: {m2.id: 1.0},
        n2.id: {destination.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id],
        "onramp_ids": [onr.id],
        "motorway_ids": [m1.id, m2.id],
        "offramp_ids": [],
        "destination_ids": [destination.id],
        "splits": splits,
    }

    return net, metadata


def setup_network_c() -> tuple[Network, dict]:
    """
    Create a simple network with a single onramp in the middle and a
    bottleneck with lane drop downstream.

    The bottleneck is created by reducing the number of lanes in the
    downstream cell, which reduces its capacity and creates congestion
    that propagates upstream and interacts with the onramp / virtual input queue.
    """

    # motorway segments with a lane drop downstream
    m1 = MotorwayLink(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m2 = MotorwayLink(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m3 = MotorwayLink(
        length=0.5, lanes=1, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m4 = MotorwayLink(
        length=0.5, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    # add origin, destination and onramp
    origin = Origin()
    destination = Destination(id=None)
    onr = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )

    # connect the links through nodes and build the network structure
    n0 = Node(incoming=[origin], outgoing=[m1])
    n1 = Node(incoming=[m1, onr], outgoing=[m2])
    n2 = Node(incoming=[m2], outgoing=[m3])
    n3 = Node(incoming=[m3], outgoing=[m4])
    n4 = Node(incoming=[m4], outgoing=[destination])
    net = Network(nodes=[n0, n1, n2, n3, n4])

    splits = {
        n0.id: {m1.id: 1.0},
        n1.id: {m2.id: 1.0},
        n2.id: {m3.id: 1.0},
        n3.id: {destination.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id],
        "onramp_ids": [onr.id],
        "motorway_ids": [m1.id, m2.id, m3.id, m4.id],
        "offramp_ids": [],
        "destination_ids": [destination.id],
        "splits": splits,
    }

    return net, metadata


def mainline_demand_d(time: float) -> float:
    # stronger upstream demand that produces a bottleneck downstream
    return demand(time, 300 / 3600, 1800 / 3600, 3600 / 3600, 3500)


def onramp_demand_d(time: float) -> float:
    return demand(time, 300 / 3600, 1500 / 3600, 3600 / 3600, 2000)


def setup_network_d() -> tuple[Network, dict]:
    """Create a network with a mid-network onramp and a downstream offramp.

    The layout is designed so the onramp merges upstream of an offramp
    which takes a non-negligible split of the mainline flow. This makes
    the effects of on-/off-ramps (local queues, flow reductions and
    recovery downstream) clearly visible in the results and plots.
    """

    # create segments: upstream 2 cells, middle (onramp), downstream with offramp
    m1 = MotorwayLink(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m2 = MotorwayLink(
        length=1.0, lanes=3, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )
    m3 = MotorwayLink(
        length=1.5, lanes=2, lane_capacity=2000, free_flow_speed=100, jam_density=180
    )

    # add origin, destination, onramp and offramp
    origin = Origin()
    destination = Destination(id=None)
    onr = Onramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )
    offr = Offramp(
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        destination=destination,
    )

    # connect the links through nodes and build the network structure
    n0 = Node(incoming=[origin], outgoing=[m1])
    n1 = Node(incoming=[m1, onr], outgoing=[m2])
    n2 = Node(incoming=[m2], outgoing=[m3, offr])
    n3 = Node(incoming=[m3], outgoing=[destination])
    net = Network(nodes=[n0, n1, n2, n3])

    # splits at node2: motorway keeps 0.8, offramp 0.2
    splits = {
        n0.id: {m1.id: 1.0},
        n1.id: {m2.id: 1.0},
        n2.id: {m3.id: 0.8, offr.id: 0.2},
        n3.id: {destination.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id],
        "onramp_ids": [onr.id],
        "motorway_ids": [m1.id, m2.id, m3.id],
        "offramp_ids": [offr.id],
        "destination_ids": [destination.id],
        "splits": splits,
    }

    return net, metadata
