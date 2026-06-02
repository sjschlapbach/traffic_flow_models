# Network Structure

A `Network` is a directed graph of **links** and **nodes**. The same topology is shared
by all models (CTM and METANET), ensuring that simulation results are directly comparable
under identical network conditions.

## Link types

| Class          | Role                           | Models state                                         |
| -------------- | ------------------------------ | ---------------------------------------------------- |
| `MotorwayLink` | Homogeneous highway segment    | density \(\rho\), speed \(v\), flow \(q\) per cell   |
| `Onramp`       | Merging link from surface road | flow \(q\), virtual queue \(N\)                      |
| `Offramp`      | Diverging link to surface road | flow \(q\), virtual queue \(N\)                      |
| `Origin`       | Virtual demand source          | demand \(d(k)\), virtual queue \(N\)                 |
| `Destination`  | Virtual flow sink              | boundary condition \(q_{bc}(k)\) or \(\rho_{bc}(k)\) |

`Origin`, `Onramp`, and `Offramp` are all modelled as **store-and-forward** links: they
accept arbitrary inflow and store excess traffic in a virtual queue \(N(k)\), so highway
backpressure is captured without enforcing hard capacity on the ramp itself. This is consistent with the network-level METANET formulation proposed by A. Kotsialos, M. Papageorgiou, et al. (2002).

## Nodes

A `Node` connects a set of incoming links to a set of outgoing links. Nodes are placed
wherever the network is **inhomogeneous**: on-ramp merges, off-ramp diverges, lane drops,
and changes in free-flow speed or jam density all require a node boundary.

```python
from traffic_flow_models import Node, MotorwayLink, Onramp, Origin, Destination

# Single-lane bottleneck preceded by an on-ramp merge
link_a = MotorwayLink(id="m1", length=2.0, lanes=2,
                      lane_capacity=2000, free_flow_speed=100, jam_density=180)
onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180)
link_b = MotorwayLink(id="m2", length=2.0, lanes=1,  # bottleneck
                      lane_capacity=2000, free_flow_speed=100, jam_density=180)

o1 = Origin(id="o1")
o2 = Origin(id="o2")

merge_node = Node(
    id="n_merge",
    incoming=[link_a, onramp],
    outgoing=[link_b],
)
```

## Cell discretisation

`MotorwayLink` instances are automatically split into **cells** (also called _segments_
in the METANET literature). The number of cells is chosen to satisfy the
**Courant–Friedrichs–Lewy (CFL) condition**:

$$L_i^m \;\geq\; \Delta t \cdot v_f^m$$

This prevents vehicles from travelling across more than one cell per time step and
guarantees numerical stability. A desired cell size \(L\_{des}\) can be passed as a
constructor argument; the library adjusts it upward if necessary to satisfy the CFL
condition for the given time step and free-flow speed.

## Regularity assumptions

The following structural rules are enforced during `Network` validation:

- Every `Origin` or `Destination` node has exactly **one** connected link.
- `Origin` nodes directly connected to a highway link or an on-ramp act as
  store-and-forward links; no spillback from the ramp is modelled.
- `Offramp` nodes are connected to exactly one upstream motorway link and one
  downstream `Destination`.

These assumptions simplify the model equations and ensure a well-posed network
topology for both CTM and METANET. They are validated during the construction of the non-linear model equation system and will raise an error if violated.

## Full network example

```python
from traffic_flow_models import (
    Network, Node,
    MotorwayLink, Onramp, Offramp,
    Origin, Destination,
)

# Links — all physical parameters are required
l1  = MotorwayLink(id="m1", length=2.0, lanes=3,
                   lane_capacity=2000, free_flow_speed=100, jam_density=180)
l2  = MotorwayLink(id="m2", length=1.0, lanes=3,
                   lane_capacity=2000, free_flow_speed=100, jam_density=180)
l3  = MotorwayLink(id="m3", length=2.0, lanes=3,
                   lane_capacity=2000, free_flow_speed=100, jam_density=180)
on  = Onramp(id="r1",  length=0.5, lanes=1,
            lane_capacity=1800, free_flow_speed=80, jam_density=180)
off = Offramp(id="f1", lanes=1,        # Offramp has no length parameter
              lane_capacity=1500, free_flow_speed=80, jam_density=180)

# Origins & destinations
o1 = Origin(id="o1")
o2 = Origin(id="o2")  # feeds on-ramp
d1 = Destination(id="d1")
d2 = Destination(id="d2")  # receives off-ramp

# Node topology
n0 = Node(id="n0", incoming=[o1],       outgoing=[l1])
n1 = Node(id="n1", incoming=[o2],       outgoing=[on])
n2 = Node(id="n2", incoming=[l1, on],   outgoing=[l2])       # merge
n3 = Node(id="n3", incoming=[l2],       outgoing=[l3, off])  # diverge
n4 = Node(id="n4", incoming=[l3],       outgoing=[d1])
n5 = Node(id="n5", incoming=[off],      outgoing=[d2])

net = Network(nodes=[n0, n1, n2, n3, n4, n5])
```
