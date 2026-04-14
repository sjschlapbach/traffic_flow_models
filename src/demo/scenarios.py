import casadi
import numpy as np
from typing import Any

from traffic_flow_models import (
    Network,
    Node,
    MotorwayLink,
    Onramp,
    Offramp,
    Destination,
    Origin,
    FlowController,
    AlineaController,
    MetalineController,
    CustomController,
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


def _build_ab_base() -> tuple[Network, dict]:
    """Build the base network for scenarios A and B (single onramp).

    Returns `(network, metadata)` where `metadata` contains ids and splits.
    """

    # three motorway segments approximating the original 6 cells (0.5 km each)
    m1 = MotorwayLink(
        id="m1",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m2 = MotorwayLink(
        id="m2",
        length=2.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )

    # add origin, destination and onramp
    origin = Origin(id="origin")
    origin_onr = Origin(id="origin_onr")
    destination = Destination(id="destination")
    onr = Onramp(
        id="onramp",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )

    # connect the links through nodes and build the network structure
    n0 = Node(id="n1", incoming=[origin], outgoing=[m1])
    n0.position = (0.0, 0.0)

    nonr = Node(id="nonr", incoming=[origin_onr], outgoing=[onr])
    nonr.position = (0.8, 0.1)

    n1 = Node(id="n2", incoming=[m1, onr], outgoing=[m2])
    n1.position = (1.0, 0.0)

    n2 = Node(id="n3", incoming=[m2], outgoing=[destination])
    n2.position = (3.0, 0.0)

    net = Network(nodes=[n0, nonr, n1, n2])

    splits = {
        n0.id: {m1.id: 1.0},
        nonr.id: {onr.id: 1.0},
        n1.id: {m2.id: 1.0},
        n2.id: {destination.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id, origin_onr.id],
        "onramp_ids": [onr.id],
        "motorway_ids": [m1.id, m2.id],
        "offramp_ids": [],
        "destination_ids": [destination.id],
        "splits": splits,
    }

    return net, metadata


def setup_network_a() -> tuple[Network, dict, dict]:
    """Scenario A: base network with demand profile A."""

    net, metadata = _build_ab_base()
    origin_demands = {"origin": mainline_demand_a, "origin_onr": onramp_demand_a}
    return net, metadata, origin_demands


def setup_network_b() -> tuple[Network, dict, dict]:
    """Scenario B: base network with demand profile B."""

    net, metadata = _build_ab_base()
    origin_demands = {"origin": mainline_demand_b, "origin_onr": onramp_demand_b}
    return net, metadata, origin_demands


def setup_network_c() -> tuple[Network, dict, dict]:
    """
    Create a simple network with a single onramp in the middle and a
    bottleneck with lane drop downstream.

    The bottleneck is created by reducing the number of lanes in the
    downstream cell, which reduces its capacity and creates congestion
    that propagates upstream and interacts with the onramp / virtual input queue.
    """

    # motorway segments with a lane drop downstream
    m1 = MotorwayLink(
        id="m1",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m2 = MotorwayLink(
        id="m2",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m3 = MotorwayLink(
        id="m3",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m4 = MotorwayLink(
        id="m4",
        length=0.5,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )

    # add origin, destination and onramp
    origin = Origin(id="origin")
    origin_onr = Origin(id="origin_onr")
    destination = Destination(id="destination")
    onr = Onramp(
        id="onramp",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )

    # connect the links through nodes and build the network structure
    n0 = Node(id="n0", incoming=[origin], outgoing=[m1])
    n0.position = (0.0, 0.0)

    nonr = Node(id="nonr", incoming=[origin_onr], outgoing=[onr])
    nonr.position = (0.8, 0.1)

    n1 = Node(id="n1", incoming=[m1, onr], outgoing=[m2])
    n1.position = (1.0, 0.0)

    n2 = Node(id="n2", incoming=[m2], outgoing=[m3])
    n2.position = (2.0, 0.0)

    n3 = Node(id="n3", incoming=[m3], outgoing=[m4])
    n3.position = (2.5, 0.0)

    n4 = Node(id="n4", incoming=[m4], outgoing=[destination])
    n4.position = (3.0, 0.0)

    net = Network(nodes=[n0, nonr, n1, n2, n3, n4])

    splits = {
        n0.id: {m1.id: 1.0},
        nonr.id: {onr.id: 1.0},
        n1.id: {m2.id: 1.0},
        n2.id: {m3.id: 1.0},
        n3.id: {m4.id: 1.0},
        n4.id: {destination.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id, origin_onr.id],
        "onramp_ids": [onr.id],
        "motorway_ids": [m1.id, m2.id, m3.id, m4.id],
        "offramp_ids": [],
        "destination_ids": [destination.id],
        "splits": splits,
    }

    origin_demands = {origin.id: mainline_demand_c, origin_onr.id: onramp_demand_c}
    return net, metadata, origin_demands


def setup_network_c1() -> tuple[Network, dict, dict]:
    """
    Create a variant of scenario C with a fixed-rate ramp metering controller.

    An instance of the FlowController will be attached to the onramp, automatically
    capping its outflow at 900 vehicles per hour regardless of the traffic condition
    on the mainline.
    """

    net, metadata, origin_demands = setup_network_c()

    # find the relevant node in the network
    onramp_node = net.get_node("nonr")
    if onramp_node is None:
        raise ValueError("Onramp node 'nonr' not found in the network.")

    # get the onramp link
    onramp = onramp_node.outgoing[0]
    if not isinstance(onramp, Onramp):
        raise TypeError("Expected 'nonr' node to have an Onramp as outgoing link.")

    # attach a fixed-rate flow controller to the onramp
    onramp.controller = FlowController(onramp, flow=900)

    return net, metadata, origin_demands


def setup_network_c2() -> tuple[Network, dict, dict]:
    """
    Create a variant of scenario C with an ALINEA ramp metering controller.

    An instance of the AlineaController will be attached to the onramp, regulating
    the inflow from the on-ramp into the mainline segment based on the current density
    in the first cell of the downstream motorway link (m2) and a target critical density.

    The target critical density is fixed to an estimated value of 30 veh/km/lane here.
    The feedback gain is set to a value of 5.0 to avoid oscillations.
    """

    net, metadata, origin_demands = setup_network_c()

    # find the relevant node in the network
    onramp_node = net.get_node("nonr")
    if onramp_node is None:
        raise ValueError("Onramp node 'nonr' not found in the network.")

    # get the onramp link
    onramp = onramp_node.outgoing[0]
    if not isinstance(onramp, Onramp):
        raise TypeError("Expected 'nonr' node to have an Onramp as outgoing link.")

    # attach an ALINEA flow controller to the onramp
    onramp.controller = AlineaController(
        onramp=onramp,
        measurement_link_id="m2",
        measurement_cell_idx=0,
        gain=5.0,
        density_setpoint=30.0,
    )

    return net, metadata, origin_demands


def setup_network_c3() -> tuple[Network, dict, dict]:
    """
    Variant of scenario C where a custom controller inspects the current downstream
    flow and decides on a metering rate accordingly according to a switching rule
    between two fixed metering rates.

    The supplied custom function inspects the current onramp (unrestricted)
    flow and returns a CasADi expression with either a low or high fixed
    metering rate (here 600 or 900 vehicles per time unit).
    """

    net, metadata, origin_demands = setup_network_c()

    # find the relevant node in the network
    onramp_node = net.get_node("nonr")
    if onramp_node is None:
        raise ValueError("Onramp node 'nonr' not found in the network.")

    # get the onramp link
    onramp = onramp_node.outgoing[0]
    if not isinstance(onramp, Onramp):
        raise TypeError("Expected 'nonr' node to have an Onramp as outgoing link.")

    # custom metering logic: decide based on downstream motorway flow (m2)
    def metering_fn(
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
    ) -> casadi.SX:
        downstream_flow = flows["m2"][0]

        # if downstream flow < 2100 veh/h -> allow 900, otherwise 600
        return casadi.if_else(
            downstream_flow < casadi.SX(2100.0), casadi.SX(900.0), casadi.SX(600.0)
        )

    onramp.controller = CustomController(onramp, controller_fn=metering_fn)
    return net, metadata, origin_demands


def setup_network_c4() -> tuple[Network, dict, dict]:
    """
    Variant of scenario C where the custom controller accepts a third
    `params` argument. The parameters control the threshold and the
    high/low metering rates returned by the controller.
    """

    net, metadata, origin_demands = setup_network_c()

    # find the relevant node in the network
    onramp_node = net.get_node("nonr")
    if onramp_node is None:
        raise ValueError("Onramp node 'nonr' not found in the network.")

    # get the onramp link
    onramp = onramp_node.outgoing[0]
    if not isinstance(onramp, Onramp):
        raise TypeError("Expected 'nonr' node to have an Onramp as outgoing link.")

    # custom metering logic: decide based on downstream motorway flow (m2)
    def metering_fn(
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        params: dict[str, Any],
    ) -> casadi.SX:
        downstream_flow = flows["m2"][0]
        threshold = casadi.SX(params.get("threshold"))
        high = casadi.SX(params.get("high"))
        low = casadi.SX(params.get("low"))
        return casadi.if_else(downstream_flow < threshold, high, low)

    onramp.controller = CustomController(
        onramp,
        controller_fn=metering_fn,
        params={"threshold": 2050.0, "high": 900.0, "low": 600.0},
    )

    return net, metadata, origin_demands


def mainline_demand_d(time: float) -> float:
    # stronger upstream demand that produces a bottleneck downstream
    return demand(time, 300 / 3600, 1800 / 3600, 3600 / 3600, 3500)


def onramp_demand_d(time: float) -> float:
    return demand(time, 300 / 3600, 1500 / 3600, 3600 / 3600, 2000)


def setup_network_d() -> tuple[Network, dict, dict]:
    """Create a network with a mid-network onramp and a downstream offramp.

    The layout is designed so the onramp merges upstream of an offramp
    which takes a non-negligible split of the mainline flow. This makes
    the effects of on-/off-ramps (local queues, flow reductions and
    recovery downstream) clearly visible in the results and plots.
    """

    # create segments: upstream 2 cells, middle (onramp), downstream with offramp
    m1 = MotorwayLink(
        id="m1",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m2 = MotorwayLink(
        id="m2",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m3 = MotorwayLink(
        id="m3",
        length=1.5,
        lanes=2,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )

    # add origin, destination, onramp and offramp
    origin = Origin(id="origin")
    origin_onr = Origin(id="origin_onr")
    destination = Destination(id="destination")
    destination_offr = Destination(id="destination_offr")
    onr = Onramp(
        id="onramp",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )
    offr = Offramp(
        id="offramp",
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )

    # connect the links through nodes and build the network structure
    n0 = Node(id="n0", incoming=[origin], outgoing=[m1])
    n0.position = (0.0, 0.0)

    nonr = Node(id="nonr", incoming=[origin_onr], outgoing=[onr])
    nonr.position = (0.8, 0.1)

    n1 = Node(id="n1", incoming=[m1, onr], outgoing=[m2])
    n1.position = (1.0, 0.0)

    n2 = Node(id="n2", incoming=[m2], outgoing=[m3, offr])
    n2.position = (2.0, 0.0)

    noffr = Node(id="noffr", incoming=[offr], outgoing=[destination_offr])
    noffr.position = (2.2, -0.1)

    n3 = Node(id="n3", incoming=[m3], outgoing=[destination])
    n3.position = (3.5, 0.0)

    net = Network(nodes=[n0, nonr, n1, n2, n3, noffr])

    # splits at node2: motorway keeps 0.8, offramp 0.2
    splits = {
        n0.id: {m1.id: 1.0},
        nonr.id: {onr.id: 1.0},
        n1.id: {m2.id: 1.0},
        n2.id: {m3.id: 0.8, offr.id: 0.2},
        n3.id: {destination.id: 1.0},
        noffr.id: {destination_offr.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id, origin_onr.id],
        "onramp_ids": [onr.id],
        "motorway_ids": [m1.id, m2.id, m3.id],
        "offramp_ids": [offr.id],
        "destination_ids": [destination.id, destination_offr.id],
        "splits": splits,
    }

    origin_demands = {origin.id: mainline_demand_d, origin_onr.id: onramp_demand_d}
    return net, metadata, origin_demands


def setup_network_e() -> tuple[Network, dict, dict]:
    """
    Scenario E base: mainline with three sequential onramps.

    Returns the network, metadata and origin demand mapping.
    """

    # mainline segments
    m1 = MotorwayLink(
        id="m1",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m2 = MotorwayLink(
        id="m2",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m3 = MotorwayLink(
        id="m3",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    m4 = MotorwayLink(
        id="m4",
        length=1.0,
        lanes=3,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )

    # origins and onramps
    origin = Origin(id="origin")
    origin_onr1 = Origin(id="origin_onr1")
    origin_onr2 = Origin(id="origin_onr2")
    origin_onr3 = Origin(id="origin_onr3")
    destination = Destination(id="destination")

    onr1 = Onramp(
        id="onr1",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )
    onr2 = Onramp(
        id="onr2",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )
    onr3 = Onramp(
        id="onr3",
        length=0.5,
        lanes=1,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
        controller=None,
    )

    # connect nodes
    n0 = Node(id="n0", incoming=[origin], outgoing=[m1])
    n0.position = (0.0, 0.0)

    nonr1 = Node(id="nonr1", incoming=[origin_onr1], outgoing=[onr1])
    nonr1.position = (0.8, 0.1)

    n1 = Node(id="n1", incoming=[m1, onr1], outgoing=[m2])
    n1.position = (1.0, 0.0)

    nonr2 = Node(id="nonr2", incoming=[origin_onr2], outgoing=[onr2])
    nonr2.position = (1.8, 0.1)

    n2 = Node(id="n2", incoming=[m2, onr2], outgoing=[m3])
    n2.position = (2.0, 0.0)

    nonr3 = Node(id="nonr3", incoming=[origin_onr3], outgoing=[onr3])
    nonr3.position = (2.8, 0.1)

    n3 = Node(id="n3", incoming=[m3, onr3], outgoing=[m4])
    n3.position = (3.0, 0.0)

    n4 = Node(id="n4", incoming=[m4], outgoing=[destination])
    n4.position = (4.0, 0.0)

    net = Network(nodes=[n0, nonr1, n1, nonr2, n2, nonr3, n3, n4])

    splits = {
        n0.id: {m1.id: 1.0},
        nonr1.id: {onr1.id: 1.0},
        n1.id: {m2.id: 1.0},
        nonr2.id: {onr2.id: 1.0},
        n2.id: {m3.id: 1.0},
        nonr3.id: {onr3.id: 1.0},
        n3.id: {m4.id: 1.0},
        n4.id: {destination.id: 1.0},
    }

    metadata = {
        "origin_ids": [origin.id, origin_onr1.id, origin_onr2.id, origin_onr3.id],
        "onramp_ids": [onr1.id, onr2.id, onr3.id],
        "motorway_ids": [m1.id, m2.id, m3.id, m4.id],
        "offramp_ids": [],
        "destination_ids": [destination.id],
        "splits": splits,
    }

    # use a mix of demand profiles for the three onramps
    origin_demands = {
        origin.id: mainline_demand_d,
        origin_onr1.id: onramp_demand_a,
        origin_onr2.id: onramp_demand_b,
        origin_onr3.id: onramp_demand_c,
    }

    return net, metadata, origin_demands


def setup_network_e1() -> tuple[Network, dict, dict]:
    """Scenario E1: apply independent ALINEA controllers to each onramp."""
    net, metadata, origin_demands = setup_network_e()

    # map onramps to their downstream measurement links
    mapping = {"onr1": "m2", "onr2": "m3", "onr3": "m4"}

    # attach ALINEA controllers with the same setpoint/gain for simplicity
    for onr_id in metadata["onramp_ids"]:
        onr = net.get_link(onr_id)
        if not isinstance(onr, Onramp):
            raise TypeError("Expected an Onramp link.")

        meas_link = mapping[onr_id]
        onr.controller = AlineaController(
            onramp=onr,
            measurement_link_id=meas_link,
            measurement_cell_idx=0,
            gain=5.0,
            density_setpoint=30.0,
        )

    return net, metadata, origin_demands


def setup_network_e2() -> tuple[Network, dict, dict]:
    """Scenario E2: apply coordinated METALINE across the three onramps."""
    net, metadata, origin_demands = setup_network_e()

    measurement_cells = [("m2", 0), ("m3", 0), ("m4", 0)]
    density_setpoints = [("m2", 0, 30.0), ("m3", 0, 30.0), ("m4", 0, 30.0)]

    # example coordinated gain matrix with coupling
    gain_matrix = {
        "onr1": np.array([[5.0, 10.0, 1.0]], dtype=np.float64),
        "onr2": np.array([[1.0, 5.0, 3.0]], dtype=np.float64),
        "onr3": np.array([[1.0, 1.0, 5.0]], dtype=np.float64),
    }

    # attach Metaline controller instance to each onramp
    for onr_id in metadata["onramp_ids"]:
        onr = net.get_link(onr_id)
        if not isinstance(onr, Onramp):
            raise TypeError("Expected an Onramp link.")

        onr.controller = MetalineController(
            onramp=onr,
            measurement_cells=measurement_cells,
            gain_matrix=gain_matrix,
            density_setpoints=density_setpoints,
        )

    return net, metadata, origin_demands
