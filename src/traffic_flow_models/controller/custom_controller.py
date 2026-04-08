import casadi
from typing import Callable


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
        onramp_id: str,
        controller_fn: Callable[
            [dict[str, casadi.SX], dict[str, casadi.SX]], casadi.SX
        ],
    ) -> None:
        if not callable(controller_fn):
            raise TypeError("controller_fn must be callable")

        self.onramp_id: str = onramp_id
        self.controller_fn = controller_fn

    def compute_regulated_flow(
        self, flows: dict[str, casadi.SX], densities: dict[str, casadi.SX]
    ) -> casadi.SX:
        """Call the user-supplied function to compute the metering rate.

        Args:
            flows: Mapping link id -> CasADi SX vector for flows
            densities: Mapping link id -> CasADi SX vector for densities

        Returns:
            CasADi SX expression representing the metering rate.
        """
        result = self.controller_fn(flows, densities)

        # if the result is not a CasADi object, try to convert it to SX
        if isinstance(result, (casadi.SX, casadi.DM)):
            return result
        try:
            return casadi.SX(result)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                "Custom controller function must return a CasADi SX or a numeric value"
            ) from exc
