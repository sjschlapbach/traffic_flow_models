# Cell Transmission Model (CTM)

CTM is a **first-order** macroscopic traffic flow model. It partitions motorway links
into cells and advances vehicle densities in discrete time steps using a triangular
fundamental diagram. Its simplicity and physical consistency make it the standard
baseline for highway traffic simulation.

For full mathematical derivations, see
[Schlapbach et al., STRC 2026](../index.md#citation).

---

## Fundamental diagram

CTM uses a **triangular fundamental diagram** relating flow, density, and speed. The
three parameters that fully define it are:

| Parameter          | Symbol                 | Unit        | Typical range |
| ------------------ | ---------------------- | ----------- | ------------- |
| Free-flow speed    | \(v_f\)                | km/h        | 80–130        |
| Lane capacity flow | \(q\_{c,\text{lane}}\) | veh/h/lane  | 1500–2200     |
| Jam density        | \(\rho\_{jam}\)        | veh/km/lane | 100–250       |

The critical density (density at capacity) follows directly:

$$\rho_{cr} = \frac{Q_c^m}{\lambda \, v_f}$$

and the backward wave speed (congestion wave propagation):

$$w_m = \frac{Q_c^m}{\rho_{jam}^m - \rho_{cr}^m}$$

---

## Cell update equations

### Density update (first-order conservation law)

$$
\rho_i^m(k+1) = \rho_i^m(k) + \frac{\Delta t}{L_i^m \lambda_m}
  \bigl[q_{i-1}^m(k) - q_i^m(k)\bigr]
$$

### Inter-cell flow (demand–supply min-rule)

For interior cells \(i \in \{1, \ldots, N_m - 1\}\):

$$
q_i^m(k+1) = \min\!\Bigl\{
  Q_c^m,\;
  \underbrace{\lambda_m v_f \rho_i^m(k+1)}_{\text{demand}},\;
  \underbrace{w_m \bigl[\rho_{jam}^m - \rho_{i+1}^m(k+1)\bigr]}_{\text{supply}}
\Bigr\}
$$

The demand term limits flow in free-flow; the supply term limits it under congestion,
reproducing upstream shockwave propagation.

### Speed (derived)

Speed is not an independent state variable in CTM — it is recovered from flow and
density via the fundamental relation:

$$v_i^m(k+1) = \frac{q_i^m(k+1)}{\lambda_m \, \rho_i^m(k+1)}$$

---

## Node flow and proportional reduction

At merge nodes, total incoming flow is accumulated and forwarded according to **turning rates**
\(\beta_n^m(k)\). If the combined demand exceeds the downstream supply of any outgoing
link, all incoming link flows are reduced proportionally to satisfy the constraint — a
standard technique for first-order merge models. This mechanism is handled automatically
by the library.

---

## Parameters

```python
from traffic_flow_models import CTM

# CTM has no constructor arguments.
# All fundamental-diagram parameters (free-flow speed, capacity, jam density)
# are set directly on each MotorwayLink.
ctm = CTM()
```

Fundamental-diagram parameters live on each **link** so heterogeneous networks are
naturally supported:

```python
from traffic_flow_models import MotorwayLink

# Downstream bottleneck has a lower free-flow speed
link = MotorwayLink(id="m_bottleneck", length=1.0, lanes=2,
                    lane_capacity=1800, free_flow_speed=80, jam_density=180)
```

---

## Running a CTM simulation

```python
from traffic_flow_models import CTM, Simulation

sim = Simulation(network=net, model=CTM())
time_arr, states, disturbances = sim.run(
    duration=1.0,
    dt=10 / 3600,
    origin_demands={"o_main": lambda t: 1500.0},
    turning_rates={
        "n_in":  lambda t: {"m1":     1.0},
        "n_out": lambda t: {"d_main": 1.0},
    },
    destination_flow_bc={"d_main": lambda t: 2000.0},
    destination_density_bc={"d_main": lambda t: 0.0},
    plot_results=False,
)
```

---

## Limitations

CTM assumes vehicles **instantaneously** reach the equilibrium speed prescribed by the
fundamental diagram. This means it cannot capture:

- **Capacity drop** after congestion onset
- **Hysteresis** in the flow–density relationship
- **Driver anticipation** of downstream conditions

For these phenomena, use [METANET](metanet.md).
