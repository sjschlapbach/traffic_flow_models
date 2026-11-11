# Ramp Metering

Ramp metering regulates inflow through on-ramps to prevent mainline congestion. The library provides a general control interface and two reference implementations — **ALINEA** (local) and **METALINE** (coordinated) — that can be used with both CTM and METANET.

For full algorithmic derivations, see [Schlapbach et al., STRC 2026](../index.md#citation).

---

## How controllers are attached

A controller is assigned to an `Onramp` via its `controller` attribute. The simulation reads the controller from the onramp automatically during each time step.

```python
from traffic_flow_models import Onramp

onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180,
                controller=None)  # controller assigned below
```

---

## General metering mechanism

All ramp metering strategies compute a **maximum allowed inflow** \(r_m(k)\) for each controlled on-ramp \(m\). The actual metered flow is:

$$q_m^r(k+1) = \min\!\bigl\{q_m(k+1),\; \max\{r_m(k+1),\; 0\}\bigr\}$$

where \(q_m(k+1)\) is the unmetered on-ramp outflow. An unmetered on-ramp (no controller) is equivalent to setting \(r_m = \infty\). Excess vehicles that cannot enter the mainline are stored in the on-ramp virtual queue \(N_m(k)\).

---

## Control interval

Every controller accepts an optional `control_dt` in the same time unit as the simulation `dt`. The controller is then evaluated every `control_dt` and its metering rate is held in between. `control_dt` must be an integer multiple of `dt` (a 10 Hz simulation accepts a 1 Hz controller, a 3 Hz simulation rejects a 2 Hz one with a `ValueError`); the default `None` updates the rate at every simulation step.

```python
onramp.controller = AlineaController(
    onramp=onramp, measurement_link_id="m2", measurement_cell_idx=0,
    gain=5.0, density_setpoint=40.0,
    control_dt=60 / 3600,  # dt = 10 / 3600: new metering rate every 6th step
)
```

The model step has two formulations, selected with `network_update_function(..., dynamics=...)` (type `DynamicsFormulation`):

- `"closed_form"` (default): the controllers are part of the dynamics, `f(params, x, d)`.
- `"exogenous_control"`: the controllers are ignored and the metering rates are an input, `f(params, x, d, u)`, with one entry per on-ramp in the state-vector order of the on-ramp queues (`inf` = unmetered). `Simulation` uses it only when a controller sets `control_dt`.

---

## ALINEA — local ramp metering

ALINEA is a **local I-type feedback controller**. It regulates the density at a detector location on the mainline to a set point (typically the critical density \(\rho\_{cr}\) for maximum throughput):

$$r_m(k+1) = q_m^r(k) + K'_I \bigl[\rho_j^\mu(k) - \rho_{set}^\mu\bigr]$$

where \(K'\_I = K_I L_j^\mu / \Delta t\) is the gain (transformed from the original occupancy-based formulation to density units).

**Choosing the measurement location:** typically a cell **downstream of the on-ramp** and upstream of a known bottleneck, so the controller can react before congestion propagates back to the ramp.

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

METALINE distributes control effort across **multiple on-ramps** simultaneously, using density measurements from several relevant locations. This reduces the myopia of local controllers:

$$r_m(k+1) = q_m^r(k) + K_{METALINE}^m \cdot \bigl[\boldsymbol{\rho}(k) - \boldsymbol{\rho}_{set}\bigr]$$

where \(K\_{METALINE}^m\) is the \(m\)-th row of the gain matrix. A purely diagonal gain matrix reduces to uncoordinated ALINEA-style control.

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

## Maximum queue length (queue control)

Set `max_queue` (veh) on an `Onramp`, e.g. to the vehicles the ramp can store, to keep its queue from spilling back into the adjacent network. ALINEA and METALINE then limit the metering rate with the standard queue-control law:

$$r_w(k) = \frac{w(k) - w_{max}}{T} + d(k-1), \qquad r(k) = \max\bigl\{r_{ctrl}(k),\; r_w(k)\bigr\}$$

where \(w(k)\) is the current queue, \(w_{max}\) the `max_queue`, \(d(k-1)\) the arrival flow from the origin feeding the on-ramp, \(r_{ctrl}\) the ALINEA / METALINE rate and \(T\) the time between two metering-rate updates. Held for \(T\), \(r_w\) brings the queue back to \(w_{max}\) for constant arrivals. The metering rate is raised only as far as the queue requires, and the control law takes over again once the queue is below the limit.

```python
onramp = Onramp(id="r1", length=0.5, lanes=1,
                lane_capacity=1800, free_flow_speed=80, jam_density=180,
                max_queue=60.0)  # at most 60 vehicles waiting on the ramp
onramp.controller = AlineaController(
    onramp=onramp, measurement_link_id="m2", measurement_cell_idx=0,
    gain=5.0, density_setpoint=40.0,
)
```

- \(T\) is the controller's update interval (`control_dt`, or the simulation `dt` if it is None), so \(r_w\) is evaluated together with the control law and held with it. \(d\) is the arrival flow of the feeding origin in the last simulation step, assumed constant over \(T\) (zero-order hold); if arrivals rise within the interval, the queue ends the interval above \(w_{max}\). Both are set automatically when the model step is built.
- Only ALINEA and METALINE support `max_queue`. With a `FlowController` or `CustomController` it is ignored with a warning (a custom function can read `onramp_queues` and `onramp.max_queue` itself); an on-ramp without a controller is unmetered and ignores it. In the `"exogenous_control"` formulation, metering rates passed as `u` are applied unchanged.
- The queue can still exceed `max_queue` when the ramp capacity or the merge (mainline supply) limits the on-ramp outflow; no metering rate can prevent that. Similarly, the zero-hold assumption of the demand can lead to temporary overshoots if the arrival flow rises within the interval.

---

## Fixed-rate metering

For simple capacity-restriction scenarios use the `FlowController`:

```python
from traffic_flow_models import FlowController

onramp.controller = FlowController(onramp=onramp, flow=600.0)  # 600 veh/h max
```

---

## Custom controllers

Supply any CasADi-compatible callable via `CustomController`. The function receives `(onramp_queues, flows, densities)` dicts and must return a `casadi.SX` expression:

```python
import casadi
from traffic_flow_models import CustomController

def my_law(onramp_queues, flows, densities):
    """Reduce metering when downstream density exceeds 50 veh/km/lane."""
    rho = densities["m2"][0]
    return casadi.if_else(rho > 50.0, casadi.SX(400.0), casadi.SX(1200.0))

onramp.controller = CustomController(onramp=onramp, controller_fn=my_law)
```
