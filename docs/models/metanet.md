# METANET

METANET is a **second-order** macroscopic traffic flow model. It supplements the
first-order density conservation equation with an explicit dynamic equation for
**mean speed**, capturing effects that CTM cannot reproduce: driver anticipation of
downstream conditions, speed relaxation towards equilibrium, on-ramp merging speed
drops, and lane-drop anticipation.

For full mathematical derivations, see
[Schlapbach et al., STRC 2026](../index.md#citation).

---

## Fundamental diagram

Unlike the triangular FD in CTM, METANET uses an **exponential equilibrium speed
function**:

$$
V_S^m[\rho_i^m(k)] = v_f^m \exp\!\left[
  -\frac{1}{\alpha_m} \left(\frac{\rho_i^m(k)}{\rho_{cr}^m}\right)^{\!\alpha_m}
\right]
$$

The (optionally link-specific) shape exponent \(\alpha_m\) controls how steeply speed drops with density. This provides more flexibility when fitting real-world flow–density observations compared to the triangular shape.

The critical density is defined implicitly from the capacity condition:

$$\rho_{cr}^m = \frac{Q_c^m}{\lambda_m v_f^m \exp\!\left(-\tfrac{1}{\alpha_m}\right)}$$

---

## Speed update equation

The core of METANET is a second update equation for mean speed. It has four additive
components:

$$
v_i^m(k+1) = v_i^m(k)
  + \underbrace{\frac{\Delta t}{\tau}
    \Bigl[V_S^m[\rho_i^m(k)] - v_i^m(k)\Bigr]}_{\text{relaxation}}
  + \underbrace{\frac{\Delta t}{L_i^m} v_i^m(k)
    \Bigl[v_{i-1}^m(k) - v_i^m(k)\Bigr]}_{\text{convection}}
$$

$$
\qquad\qquad
  - \underbrace{\frac{\nu \Delta t}{\tau L_i^m}
    \frac{\rho_{i+1}^m(k) - \rho_i^m(k)}{\rho_i^m(k) + \kappa}}_{\text{anticipation}}
  - \underbrace{\frac{\delta \Delta t}{\lambda_1 L_1^m}
    \frac{q_{on}(k) \, v_1^m(k)}{\rho_1^m(k) + \kappa}}_{\text{on-ramp speed drop}}
  - \underbrace{\frac{\phi \Delta t}{L_{N_m}^m \lambda_m}
    \frac{\Delta\lambda_m \, \rho_{N_m}^m(k) \, [v_{N_m}^m(k)]^2}{\rho_{cr}^m}}_{\text{lane-drop anticipation}}
$$

| Term         | Parameter                                 | Physical meaning                                               |
| ------------ | ----------------------------------------- | -------------------------------------------------------------- |
| Relaxation   | \(\tau\) (h)                              | Drivers' desire to converge to equilibrium speed               |
| Convection   | —                                         | Speed advection along the link                                 |
| Anticipation | \(\nu\) (km²/h), \(\kappa\) (veh/km/lane) | Reaction to density gradient ahead                             |
| On-ramp drop | \(\delta\)                                | Speed reduction caused by merging flow at the first cell       |
| Lane-drop    | \(\phi\)                                  | Speed reduction in anticipation of a downstream lane reduction |

The on-ramp and lane-drop terms are non-zero only in the relevant cells; all other
cells use \(q\_{on}(k)=0\) and \(\Delta\lambda_m = 0\).

Cell flow is then recovered from the fundamental relation:

$$q_i^m(k) = \lambda_m \cdot \rho_i^m(k) \cdot v_i^m(k)$$

!!! note "Flows may temporarily exceed capacity"

    Unlike CTM, METANET does not enforce a hard upper bound on cell flow. Transient
    flows above \(Q_c^m\) are physically plausible and mimic real-world congestion onset
    dynamics such as capacity drop and hysteresis.

---

## Node mechanics

METANET requires two additional boundary quantities at link boundaries:

**Virtual upstream mean speed** (at the first cell of an outgoing link, merge node):

$$
v_0^m(k) = \frac{\sum_{\mu \in I_n} v_{N_\mu}^\mu(k) \cdot q_{N_\mu}^\mu(k)}
                   {\sum_{\mu \in I_n} q_{N_\mu}^\mu(k)}
$$

A flow-weighted average of the last-cell speeds of all incoming links.

**Virtual downstream density** (at the last cell of an incoming link, diverge node):

$$
\rho_{N_m+1}^m(k) = \frac{\sum_{\mu \in O_n} [\rho_1^\mu(k)]^2}
                            {\sum_{\mu \in O_n} \rho_1^\mu(k)}
$$

A contraharmonic mean of the first-cell densities of all outgoing links. Both
quantities are computed automatically by the library.

---

## Parameters

```python
from traffic_flow_models import METANET, METANETParams

# METANET has no constructor arguments.
# Fundamental-diagram parameters live on each MotorwayLink (lane_capacity,
# free_flow_speed, jam_density). METANETParams holds only the second-order
# coefficients that are passed to Simulation.
model_params: METANETParams = {
    "tau":   22 / 3600,  # relaxation time [h]  (~18–30 s)
    "nu":    15.0,       # anticipation coefficient [km²/h]
    "kappa": 40.0,       # density smoothing [veh/km/lane]
    "delta": 0.012,      # on-ramp speed-drop coefficient [-]
    "phi":   1.0,        # lane-drop anticipation coefficient [-]
    "alpha": 1.8,        # FD shape exponent [-]
}
metanet = METANET()
```

### Parameter reference

| Parameter | Symbol     | Unit        | Typical range  | Effect                                         |
| --------- | ---------- | ----------- | -------------- | ---------------------------------------------- |
| `tau`     | \(\tau\)   | h           | 5–30 s (÷3600) | Speed relaxation rate; larger → slower         |
| `nu`      | \(\nu\)    | km²/h       | 10–60          | Anticipation strength; larger → stronger       |
| `kappa`   | \(\kappa\) | veh/km/lane | 10–60          | Prevents division by zero in anticipation term |
| `delta`   | \(\delta\) | —           | 0.005–0.02     | On-ramp merging speed reduction                |
| `phi`     | \(\phi\)   | —           | 0–2            | Lane-drop anticipation magnitude               |
| `alpha`   | \(\alpha\) | —           | 1.5–2.5        | FD shape; 1.8 is a common default              |

!!! tip "Calibration sensitivity"

    Depending on the chosen calibration approach, different parameters may
    be more or less sensitive and coupled through correlation. See the
    [Calibration guide](../experimental/calibration.md) for details.

---

## Running a METANET simulation

```python
from traffic_flow_models import METANET, METANETParams, Simulation

model_params: METANETParams = {
    "tau": 22/3600, "nu": 15.0, "kappa": 40.0,
    "delta": 0.012, "phi": 1.0, "alpha": 1.8,
}
sim = Simulation(network=net, model=METANET(), model_params=model_params)
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
