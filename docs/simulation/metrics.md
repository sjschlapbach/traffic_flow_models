# Performance Metrics

After running a simulation, call `compute_metrics()` to obtain standard traffic
performance indicators aggregated over the entire network and simulation period.

`compute_metrics()` takes the raw arrays returned by `run()`:

```python
VKT, VHT, avg_speed = sim.compute_metrics(
    states=states,
    dt=10 / 3600,
    timesteps=len(time_arr),
)

print(f"VKT:       {VKT:.0f} veh·km")
print(f"VHT:       {VHT:.2f} veh·h")
print(f"Avg speed: {avg_speed:.1f} km/h")
```

## Available metrics

| Return value | Unit   | Description                                       |
| ------------ | ------ | ------------------------------------------------- |
| `VKT`        | veh·km | Total vehicle-kilometres travelled                |
| `VHT`        | veh·h  | Total vehicle-hours travelled (mainline + queues) |
| `avg_speed`  | km/h   | Network average speed: VKT / VHT                  |

## Excluding queue time

Pass `ignore_queues=True` to exclude time spent in origin / on-ramp / off-ramp queues
from the VHT calculation (mainline only):

```python
VKT, VHT, avg_speed = sim.compute_metrics(
    states=states,
    dt=10 / 3600,
    timesteps=len(time_arr),
    ignore_queues=True,
)
```
