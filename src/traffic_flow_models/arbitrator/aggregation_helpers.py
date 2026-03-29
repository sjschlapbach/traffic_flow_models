"""Helper functions for temporal aggregation of detector data.

This module provides reusable aggregation functions for processing
SUMO detector outputs and creating time-varying demand and turning
rate functions for macroscopic traffic flow models.
"""

from typing import Callable, Mapping, Literal, Sequence

Number = float | int


def make_rolling_window_aggregator(
    intervals: Mapping[str, Sequence[tuple[float, Number]]],
    window_size_sec: float,
    max_time: float,
    aggregation_type: Literal["rate", "demand"] = "rate",
) -> Callable[[float], dict[str, float]]:
    """Create a time-varying function using rolling window aggregation.

    Constructs a callable function that computes values from detector
    observations using a rolling time window. At query time t, the function
    aggregates all vehicle counts within [t - window/2, t + window/2].
    If the window extends beyond the simulation horizon, it is shifted to
    fit within [0, max_time].

    Args:
        intervals: Dictionary mapping keys (e.g., edge IDs) to lists of
            (begin_time, count) tuples representing detector measurements.
        window_size_sec: Size of the rolling window in seconds.
        max_time: Maximum simulation time in seconds.
        aggregation_type: Type of aggregation:
            - "rate": Returns fractions (e.g., for turning rates)
            - "demand": Returns flow rates in veh/h (for demand)

    Returns:
        A callable function that takes time in hours and returns a dictionary
        mapping keys to their aggregated values (fractions or flow rates).

    Raises:
        ValueError: If aggregation_type is not "rate" or "demand".
    """
    if aggregation_type not in ["rate", "demand"]:
        raise ValueError(
            f"aggregation_type must be 'rate' or 'demand', got: {aggregation_type}"
        )

    keys = list(intervals.keys())

    def rolling_window_fn(time_hours: float) -> dict[str, float]:
        query_time_sec = time_hours * 3600

        # compute window bounds
        window_start = query_time_sec - window_size_sec / 2
        window_end = query_time_sec + window_size_sec / 2

        # shift window if it goes beyond simulation horizon
        if window_start < 0:
            window_end = min(window_size_sec, max_time)
            window_start = 0
        elif window_end > max_time:
            window_start = max(0, max_time - window_size_sec)
            window_end = max_time

        # aggregate counts for each key within the window
        key_counts: dict[str, float] = {}
        for key in keys:
            count = 0
            for begin, veh_count in intervals[key]:
                if window_start <= begin < window_end:
                    count += float(veh_count)
            key_counts[key] = count

        total = sum(key_counts.values())

        if aggregation_type == "rate":
            # return fractions (turning rates)
            if total == 0:
                # no vehicles in this window - return equal splits
                return {key: 1.0 / len(keys) for key in keys}

            return {key: key_counts[key] / total for key in keys}

        else:  # aggregation_type == "demand"
            # return flow rate in veh/h
            # total vehicles in window_size_sec → scale to hourly rate
            if window_end - window_start == 0:
                return {key: 0.0 for key in keys}

            window_duration_hours = (window_end - window_start) / 3600.0
            return {key: key_counts[key] / window_duration_hours for key in keys}

    return rolling_window_fn


def make_single_stream_rolling_window_aggregator(
    intervals: Sequence[tuple[float, Number]],
    window_size_sec: float,
    max_time: float,
) -> Callable[[float], float]:
    """Create a time-varying demand function for a single stream using rolling window.

    Similar to make_rolling_window_aggregator but for a single data stream,
    returning a scalar flow rate in veh/h instead of a dictionary.

    Args:
        intervals: List of (begin_time, count) tuples representing detector measurements.
        window_size_sec: Size of the rolling window in seconds.
        max_time: Maximum simulation time in seconds.

    Returns:
        A callable function that takes time in hours and returns flow rate in veh/h.
    """
    # wrap intervals in a dict with a single key
    intervals_dict = {"single": intervals}

    # create the multi-stream aggregator
    multi_stream_fn = make_rolling_window_aggregator(
        intervals=intervals_dict,
        window_size_sec=window_size_sec,
        max_time=max_time,
        aggregation_type="demand",
    )

    # return a function that unwraps the result
    def single_stream_fn(time_hours: float) -> float:
        result = multi_stream_fn(time_hours)
        return result["single"]

    return single_stream_fn
