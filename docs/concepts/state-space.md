# State-Space Formulation

Both CTM and METANET share the same abstract discrete-time state-space representation.
Understanding this formulation helps clarify what each argument to `Simulation.run()`
represents and how models differ internally.

## The unified update equation

Network evolution is governed by a single nonlinear, time-invariant discrete-time
update function:

$$\boxed{x(k+1) = f\!\bigl(x(k),\; u(k),\; d(k),\; \theta\bigr)}$$

The function \(f(\cdot)\) advances the network state by one discrete time step
\(\Delta t\). CTM and METANET implement different \(f(\cdot)\) but operate on the
same state and input structure.

## State vector \(x(k)\)

The state vector aggregates all modelled physical quantities at time step \(k\).
It is packed in network-traversal order: for each node, incoming `Origin` / `Onramp`
entries come first, followed by outgoing link entries.

| Component        | Symbol          | Unit        | Defined on                                                               |
| ---------------- | --------------- | ----------- | ------------------------------------------------------------------------ |
| Per-cell flow    | \(q_i^m(k)\)    | veh/h       | Every cell \(i\) of every `MotorwayLink` \(m\)                           |
| Per-cell density | \(\rho_i^m(k)\) | veh/km/lane | Every cell \(i\) of every `MotorwayLink` \(m\)                           |
| Per-cell speed   | \(v_i^m(k)\)    | km/h        | Every cell \(i\) of every `MotorwayLink` \(m\)                           |
| Outflow          | \(q^m(k)\)      | veh/h       | Every `Origin`, `Onramp`, `Offramp`, and `Destination` (scalar per link) |
| Virtual queue    | \(N^m(k)\)      | veh         | Every `Origin`, `Onramp`, and `Offramp` (scalar per link)                |

!!! note "Speed in CTM"

    In CTM, mean speed is derived algebraically from flow and density via the
    fundamental relation \(v = q / (\lambda \rho)\). It is stored in the state
    vector as a derived quantity at each step, but is not an independent dynamic
    state — unlike METANET where it is integrated via a separate ODE.

!!! note "Store-and-forward links"

    `Origin`, `Onramp`, and `Offramp` are each represented as a single-cell
    store-and-forward element. Their state contributes **two** scalars to \(x\):
    the current outflow \(q^m\) and the virtual queue \(N^m\). `Destination` links
    contribute only the exit flow (no queue).

## Input vector \(u(k)\)

Control inputs applied at each time step:

| Component          | Symbol     | Description                                          |
| ------------------ | ---------- | ---------------------------------------------------- |
| Ramp metering rate | \(r_m(k)\) | Maximum allowed inflow through on-ramp \(m\) (veh/h) |

When no controller is attached, \(r_m(k) = \infty\) (no metering).

## Disturbance vector \(d(k)\)

Exogenous quantities that drive the simulation and are not controlled. The disturbance
vector is packed in the same node-traversal order as the state vector:

| Component                  | Symbol              | Description                                                                                          |
| -------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| Turning rates              | \(\beta_n^m(k)\)    | Fraction of flow at node \(n\) directed towards outgoing link \(m\) (one per outgoing link per node) |
| Origin demand              | \(d_o(k)\)          | Time-varying arrival demand at each `Origin` (veh/h)                                                 |
| Flow boundary condition    | \(q\_{bc}^m(k)\)    | Maximum exit flow at each `Destination` (veh/h)                                                      |
| Density boundary condition | \(\rho\_{bc}^m(k)\) | Downstream density at each `Destination` — used by METANET's speed equation (veh/km/lane)            |

## Parameter vector \(\theta\)

Model parameters that are fixed for a given simulation run. Parameters belonging to the
**fundamental diagram (FD)** are currently fixed and link-specific, thereby fully defining CTM. Additional \*_METANET-specific_ parameters are passed via `METANETParams` and govern the second-order dynamics:

| Parameter                | Symbol     | Unit        | Both / METANET-only |
| ------------------------ | ---------- | ----------- | ------------------- |
| Relaxation time          | \(\tau\)   | h           | METANET             |
| Anticipation coefficient | \(\nu\)    | km²/h       | METANET             |
| Density smoothing        | \(\kappa\) | veh/km/lane | METANET             |
| On-ramp speed-drop       | \(\delta\) | —           | METANET             |
| Lane-drop anticipation   | \(\phi\)   | —           | METANET             |
| FD shape exponent        | \(\alpha\) | —           | METANET             |

<!-- | Free-flow speed          | \(v_f\)                | km/h        | Both                | -->
<!-- | Lane capacity flow       | \(q\_{c,\text{lane}}\) | veh/h/lane  | Both                | -->
<!-- | Jam density              | \(\rho\_{jam}\)        | veh/km/lane | Both                | -->

See [CTM](../models/ctm.md) and [METANET](../models/metanet.md) for detailed parameter
descriptions and typical values.

## Initial conditions

At \(k=0\), the default initial state is **free-flow with no vehicles**:

$$q_i^m(0) = 0, \quad \rho_i^m(0) = 0, \quad v_i^m(0) = v_f^m, \quad N^m(0) = 0$$

A custom initial state \(x_0\) can be passed to `Simulation.run()` for warm-starting.
