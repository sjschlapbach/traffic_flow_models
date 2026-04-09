import casadi
import inspect
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from traffic_flow_models.network.onramp import Onramp


class CustomController:
    """Controller that delegates metering computation to a user-provided callable.

    The provided callable is expected to accept two arguments, ``flows`` and
    ``densities``, in the same form as the other controllers' ``compute_regulated_flow``
    method: dictionaries mapping link IDs to CasADi SX vectors. The callable
    should return a CasADi expression (``casadi.SX`` or ``casadi.DM``) that
    represents the metering rate.
    """

    def __init__(
        self,
        onramp: "Onramp",
        controller_fn: Callable[..., casadi.SX],
        params: dict[str, Any] | None = None,
    ) -> None:
        if not callable(controller_fn):
            raise TypeError("controller_fn must be callable")

        self.onramp = onramp
        self.controller_fn = controller_fn

        # store a mutable params dict for use by the controller function
        self.params: dict = dict(params) if params is not None else {}

    def compute_regulated_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        dt: float,
    ) -> casadi.SX:
        """Call the user-supplied function to compute the metering rate.

        Args:
            onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).
            dt: Simulation time step size (placeholder for other controllers).

        Returns:
            CasADi SX expression representing the metering rate.
        """
        # Inspect the callable signature to decide how to pass params.
        # - If the function accepts a fourth positional argument (or *args),
        #   pass params as the fourth positional argument.
        # - If the function accepts **kwargs or defines a parameter named
        #   'params' (including keyword-only), pass params as a keyword arg.
        try:
            sig = inspect.signature(self.controller_fn)
            params_list = list(sig.parameters.values())
            positional = [
                p
                for p in params_list
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_var_pos = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params_list
            )
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params_list
            )
            has_named_params = any(p.name == "params" for p in params_list)
        except (ValueError, TypeError):
            has_var_pos = has_var_kw = has_named_params = False
            positional = []

        if has_var_pos or len(positional) >= 4:
            # accepts a fourth positional argument
            result = self.controller_fn(onramp_queues, flows, densities, self.params)
        elif has_var_kw or has_named_params:
            # accepts params via keyword
            result = self.controller_fn(
                onramp_queues, flows, densities, params=self.params
            )
        else:
            result = self.controller_fn(onramp_queues, flows, densities)

        # if the result is a CasADi object, return it directly
        if isinstance(result, casadi.SX):
            return result
        elif isinstance(result, casadi.DM):
            return casadi.SX(result)
        try:
            return casadi.SX(result)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                "Custom controller function must return a CasADi SX or a numeric value"
            ) from exc
