"""Queue-control law that keeps metered onramp queues at or below their maximum length."""

from typing import TYPE_CHECKING

import casadi

if TYPE_CHECKING:
    from traffic_flow_models.network.onramp import Onramp


def apply_queue_control(
    metering_rate: casadi.SX,
    onramp: "Onramp",
    onramp_queues: dict[str, casadi.SX],
    flows: dict[str, casadi.SX],
    rate_update_dt: float | None,
    origin_id: str | None,
) -> casadi.SX:
    """Raise a metering rate as far as needed to respect the onramp's maximum queue length.

    Standard queue-control law: ``r = max(r_ctrl, r_w)`` with
    ``r_w = (w - w_max) / rate_update_dt + d``, where ``w`` is the current queue, ``w_max`` is
    the onramp's ``max_queue`` and ``d`` the arrival flow from the origin feeding the onramp
    in the last simulation step. ``d`` is assumed constant over ``rate_update_dt``
    (zero-order hold), so ``r_w`` brings the queue back to ``w_max`` for constant arrivals;
    arrivals rising within the interval leave the queue above ``w_max`` at its end.

    Args:
        metering_rate: Metering rate from the control law (vehicles per time unit).
        onramp: Controlled onramp; its ``max_queue`` (None = no limit) sets ``w_max``.
        onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
        flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
        rate_update_dt: Update interval of the controller (`control_dt`, or the simulation
            `dt` if it is None).
        origin_id: ID of the origin feeding the onramp, whose flow is the arrival flow ``d``.

    Returns:
        The metering rate respecting the maximum queue length (vehicles per time unit).

    Raises:
        ValueError: If the onramp has a `max_queue` but `rate_update_dt` or `origin_id` is
            not set (the models set both automatically when they build the step).
    """
    if onramp.max_queue is None:
        return metering_rate

    if rate_update_dt is None or origin_id is None:
        raise ValueError(
            f"Queue control of onramp {onramp.id} is not configured; the models "
            "configure it automatically when they build the step."
        )

    # metering rate that brings the queue back to max_queue within one rate update
    queue_control_rate = (
        onramp_queues[onramp.id] - onramp.max_queue
    ) / rate_update_dt + flows[origin_id][0]
    return casadi.fmax(metering_rate, queue_control_rate)
