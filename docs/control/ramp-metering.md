# Ramp Metering

Ramp metering regulates inflow through on-ramps to prevent mainline congestion. The
library provides a general control interface and two reference implementations —
**ALINEA** (local) and **METALINE** (coordinated) — that can be used with both CTM and
METANET.

For full algorithmic derivations, see
[Schlapbach et al., STRC 2026](../index.md#citation).

---

## How controllers are attached

A controller is assigned to an `Onramp` via its `controller` attribute. The simulation
reads the controller from the onramp automatically during each time step.

```python
from traffic_flow_models import Onramp

onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180,
                controller=None)  # controller assigned below
```

---

## General metering mechanism

All ramp metering strategies compute a **maximum allowed inflow** \(r_m(k)\) for each
controlled on-ramp \(m\). The actual metered flow is:

$$q_m^r(k+1) = \min\!\bigl\{q_m(k+1),\; \max\{r_m(k+1),\; 0\}\bigr\}$$

where \(q_m(k+1)\) is the unmetered on-ramp outflow. An unmetered on-ramp (no
controller) is equivalent to setting \(r_m = \infty\). Excess vehicles that cannot
enter the mainline are stored in the on-ramp virtual queue \(N_m(k)\).

---

## ALINEA — local ramp metering

ALINEA is a **local I-type feedback controller**. It regulates the density at a
detector location on the mainline to a set point (typically the critical density
\(\rho\_{cr}\) for maximum throughput):

$$r_m(k+1) = q_m^r(k) + K'_I \bigl[\rho_j^\mu(k) - \rho_{set}^\mu\bigr]$$

where \(K'\_I = K_I L_j^\mu / \Delta t\) is the gain (transformed from the original
occupancy-based formulation to density units).

**Choosing the measurement location:** typically a cell **downstream of the on-ramp**
and upstream of a known bottleneck, so the controller can react before congestion
propagates back to the ramp.

```python
from traffic_flow_models import AlineaController, Onramp

# Create the onramp first, then attach the controller
onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180)

onramp.controller = AlineaController(
    onramp=onramp,
    measurement_link_id="m2",  # ID of the mainline link to measure
    measurement_cell_idx=0,    # cell index on that link (0-based)
    gain=5.0,                  # K'_I gain
    density_setpoint=40.0,     # desired density [veh/km/lane]
)
```

---

## METALINE — coordinated ramp metering

METALINE distributes control effort across **multiple on-ramps** simultaneously,
using density measurements from several relevant locations. This reduces the
myopia of local controllers:

$$r_m(k+1) = q_m^r(k) + K_{METALINE}^m \cdot \bigl[\boldsymbol{\rho}(k) - \boldsymbol{\rho}_{set}\bigr]$$

where \(K\_{METALINE}^m\) is the \(m\)-th row of the gain matrix. A purely diagonal
gain matrix reduces to uncoordinated ALINEA-style control.

```python
import numpy as np
from traffic_flow_models import MetalineController, Onramp

onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180)

onramp.controller = MetalineController(
    onramp=onramp,
    # list of (link_id, cell_idx) measurement locations
    measurement_cells=[("m2", 0), ("m3", 1)],
    # gain matrix: dict mapping onramp ID -> numpy array shape (1, n_measurements)
    gain_matrix={"r1": np.array([[10.0, 5.0]])},
    # desired density at each measurement cell: list of (link_id, cell_idx, setpoint)
    density_setpoints=[("m2", 0, 40.0), ("m3", 1, 40.0)],
)
```

---

## Fixed-rate metering

For simple capacity-restriction scenarios use the `FlowController`:

```python
from traffic_flow_models import FlowController

onramp.controller = FlowController(onramp=onramp, flow=600.0)  # 600 veh/h max
```

---

## Custom controllers

Supply any CasADi-compatible callable via `CustomController`. The function receives
`(onramp_queues, flows, densities)` dicts and must return a `casadi.SX` expression:

```python
import casadi
from traffic_flow_models import CustomController

def my_law(onramp_queues, flows, densities):
    """Reduce metering when downstream density exceeds 50 veh/km/lane."""
    rho = densities["m2"][0]
    return casadi.if_else(rho > 50.0, casadi.SX(400.0), casadi.SX(1200.0))

onramp.controller = CustomController(onramp=onramp, controller_fn=my_law)
```
