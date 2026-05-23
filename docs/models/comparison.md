# CTM vs. METANET

Both models share the same network representation and simulation interface. The choice
between them depends on the phenomena you need to capture and the availability of
calibration data.

## Side-by-side comparison

|                               | CTM                            | METANET                              |
| ----------------------------- | ------------------------------ | ------------------------------------ |
| **Order**                     | First-order                    | Second-order                         |
| **State variables**           | \(\rho, v, q\) (speed derived) | \(\rho, v, q\) (speed dynamic)       |
| **Fundamental diagram**       | Triangular                     | Exponential                          |
| **Parameters**                | 3 (FD only)                    | 9 (3 FD + 6 METANET-specific)        |
| **Calibration data needed**   | Flow + density + speed         | Flow + density + speed               |
| **Backward wave propagation** | Yes                            | Yes                                  |
| **Capacity drop**             | Not captured                   | Partially captured                   |
| **Driver anticipation**       | No                             | Yes (\(\nu\), \(\kappa\) terms)      |
| **On-ramp speed drop**        | No                             | Yes (\(\delta\) term)                |
| **Lane-drop anticipation**    | No                             | Yes (\(\phi\) term)                  |
| **Hard flow upper bound**     | Yes (always \(\leq Q_c\))      | No (transient overshoots allowed)    |
| **Computational cost**        | Lower                          | Slightly higher (extra ODE per cell) |

## When to use CTM

- You need a **fast, robust baseline** with minimal parameters.
- You are studying **large-scale scenarios** where the speed equation overhead matters.
- You need a model that **never violates capacity** by construction.

## When to use METANET

- You want to capture **speed dynamics**: relaxation towards equilibrium, driver
  anticipation, and merging speed drops.
- You are modelling scenarios where **transient flow** behaviour (e.g. congestion onset
  with capacity drop) is important.
- You plan to use the model inside a **model predictive control** or
  **gradient-based optimisation** framework that benefits from smooth dynamics.

## Direct comparison

Because both models share the identical network topology, demands, and turning rates,
running them in parallel under the same conditions requires minimal extra code:

```python
from traffic_flow_models import CTM, METANET, METANETParams, Simulation

ctm_sim  = Simulation(network=net, model=CTM())
meta_sim = Simulation(network=net, model=METANET(), model_params=METANETParams(
    tau=22/3600, nu=15.0, kappa=40.0, delta=0.012, phi=1.0, alpha=1.8,
))

run_kwargs = dict(
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
ctm_time,  ctm_states,  _ = ctm_sim.run(**run_kwargs)
meta_time, meta_states, _ = meta_sim.run(**run_kwargs)
```
