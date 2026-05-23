# Visualization

## Automatic plots during `run()`

Pass `plot_results=True` to `run()` to automatically generate and save plots (density,
flow, speed per link; queue plots for origins and on-ramps; node summary plots) to
`results_dir`:

```python
time_arr, states, disturbances = sim.run(
    ...,
    plot_results=True,
    show_plots=False,               # set True to display interactively
    results_dir="results/my_run",   # directory for saved figures and JSON
)
```

## Manual plots

Generate the same plots explicitly from the arrays returned by `run()`:

```python
sim.plot_results(
    time_array=time_arr,
    state_history=states,
    micro_state_history=None,       # None unless co-simulating with SUMO
    disturbance_history=disturbances,
    save_dir="results/my_run",
)
```

## Video export

`visualize()` generates an AVI video from a **saved results JSON file** (requires
`opencv-python`). Save results first, then call:

```python
# 1. Save results after run()
sim.save_results("results/my_run/results.json")

# 2. Generate video
sim.visualize(
    results_filepath="results/my_run/results.json",
    output_filepath="results/my_run/simulation.avi",
    fps=10,
    subsampling=2,  # interpolate between time steps for smoother animation
)
```

!!! note "Node coordinates required for video export"

    Node position coordinates must be set on each `Node` for the video to render
    the network topology correctly:

    ```python
    node.position = (x, y)
    ```
