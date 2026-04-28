import casadi
import warnings
import numpy as np
from typing import cast, TYPE_CHECKING, Tuple, TypedDict, Union
from numpy.typing import NDArray


from traffic_flow_models.network import (
    MotorwayLink,
    Origin,
    Onramp,
    Offramp,
    Destination,
    Node,
    Cell,
)
from .helpers import store_and_forward_update, compute_node_outflows

if TYPE_CHECKING:
    from traffic_flow_models.network.network import Network


class METANETParams(TypedDict):
    vf: float
    qc_lane: float
    rho_jam: float
    tau: float
    nu: float
    kappa: float
    delta: float
    phi: float
    alpha: float | dict[str, float]


class METANETSymbolicParams(TypedDict):
    vf: float | casadi.SX
    qc_lane: float | casadi.SX
    rho_jam: float | casadi.SX
    tau: float | casadi.SX
    nu: float | casadi.SX
    kappa: float | casadi.SX
    delta: float | casadi.SX
    phi: float | casadi.SX
    alpha: dict[str, float | casadi.SX]


class METANET:
    def __init__(self):
        """Create an empty METANET model instance."""
        return

    # ! Model parameter calibration support
    # region
    def get_default_calibration_params(self) -> METANETParams:
        """Return default initial parameters for calibration.

        These defaults provide reasonable starting values for the optimization
        procedure when no initial guess is provided.

        Returns:
            Dictionary of default METANET parameters.
        """
        return METANETParams(
            vf=100.0,
            qc_lane=2000.0,
            rho_jam=180.0,
            tau=10 / 3600,
            nu=20.0,
            kappa=20.0,
            delta=1.0,
            phi=1.0,
            alpha=1.0,
        )

    def _validate_model_options(self, model_options: dict | None) -> dict:
        """Validate and return model_options for METANET calibration.

        Args:
            model_options: Dictionary of model-specific options.

        Returns:
            Validated model_options dictionary (empty dict if None provided).

        Raises:
            ValueError: If model_options contains invalid keys or values.
        """
        if model_options is None:
            return {}

        # validate that only known options are provided
        valid_options = {"link_specific_alpha"}
        unknown_options = set(model_options.keys()) - valid_options
        if unknown_options:
            raise ValueError(
                f"Unknown model_options for METANET: {unknown_options}. "
                f"Valid options are: {valid_options}"
            )

        # validate link_specific_alpha if present
        if "link_specific_alpha" in model_options:
            link_specific_alpha = model_options["link_specific_alpha"]
            if not isinstance(link_specific_alpha, bool):
                raise ValueError(
                    f"model_options['link_specific_alpha'] must be a bool, "
                    f"got {type(link_specific_alpha).__name__}"
                )

        return model_options

    def get_calibration_param_names(
        self,
        network: "Network",
        model_options: dict | None = None,
    ) -> list[str]:
        """Return ordered calibration parameter names corresponding to the
        calibration vector.

        The returned list matches the packing order used by
        :meth:`model_params_to_vec` and :meth:`model_params_vec_to_dict`.

        Args:
            network: Network instance used to determine the ordering of
                per-link alpha entries when `model_options['link_specific_alpha']` is True.
            model_options: Model options (supports the boolean key
                ``'link_specific_alpha'``). If False, a single global ``alpha``
                parameter name is returned; if True, one ``alpha_<linkid>``
                name is returned per link in the network ordering.

        Returns:
            Ordered list of parameter names. The first five entries are always
            the scalar parameters ``['tau', 'nu', 'kappa', 'delta', 'phi']``
            followed by either a single ``'alpha'`` (global) or per-link
            ``'alpha_<linkid>'`` names (link-specific).
        """
        model_options = self._validate_model_options(model_options)
        link_specific_alpha = model_options.get("link_specific_alpha", False)

        base = ["vf", "qc_lane", "rho_jam", "tau", "nu", "kappa", "delta", "phi"]

        # single global alpha
        if not link_specific_alpha:
            return base + ["alpha"]

        # determine per-link alpha entries from the network topology. Alpha is
        # defined for motorway links, onramps and offramps; the ordering here
        # matches `model_params_to_vec` / `model_params_vec_to_dict`.
        link_ids: list[str] = []
        for node in network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Onramp):
                    link_ids.append(link.id)
            for link in node.outgoing:
                if isinstance(link, MotorwayLink) or isinstance(link, Offramp):
                    link_ids.append(link.id)

        alpha_names = [f"alpha_{lid}" for lid in link_ids]
        return base + alpha_names

    def get_calibration_bounds(
        self,
        network: "Network",
        model_options: dict | None = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return default parameter bounds for calibration.

        Provides conservative bounds that ensure physical validity of parameters
        while allowing sufficient freedom for optimization.

        Args:
            network: Network instance (used to count links for alpha bounds).
            model_options: Dictionary of METANET-specific calibration options:
                - 'link_specific_alpha' (bool): If True, returns bounds for per-link
                  alpha values. If False, returns bounds for a single global alpha
                  parameter. Default: False.

        Returns:
            Tuple of (lower_bounds, upper_bounds) as numpy arrays.
        """
        model_options = self._validate_model_options(model_options)
        link_specific_alpha = model_options.get("link_specific_alpha", False)

        # count number of alpha parameters needed
        if link_specific_alpha:
            # count unique motorway links, onramps, and offramps
            # use a set to avoid double-counting links that appear in both incoming and outgoing
            unique_links = set()
            for node in network.list_nodes():
                for link in node.incoming + node.outgoing:
                    if isinstance(link, (Onramp, MotorwayLink, Offramp)):
                        unique_links.add(link.id)
            num_alpha = len(unique_links)
        else:
            num_alpha = 1

        lower_bounds = np.array(
            [
                30.0,  # vf > 30 m/s
                500.0,  # qc_lane > 500 veh/h/lane
                50.0,  # rho_jam > 50 veh/km/lane
                5 / 3600,  # tau > 5 seconds (typically around 18 s)
                10.0,  # nu >= 10 (typically around 60 km^2/h)
                10,  # kappa > 10 (typically around 40 veh/km/lane)
                1e-3,  # delta >= 0.001 (typically around 0.1-1)
                0.01,  # phi >= 0.01 (typically around 0.1-1)
            ]
            + [1.0] * num_alpha,  # alpha > 0 for each link (or global)
            dtype=np.float64,
        )
        upper_bounds = np.array(
            [
                150.0,  # vf < 150 m/s
                2500.0,  # qc_lane < 2500 veh/h/lane
                250.0,  # rho_jam < 250 veh/km/lane
                50 / 3600,  # tau < 50 seconds
                200.0,  # nu < 200
                150.0,  # kappa < 150
                5.0,  # delta < 5
                15.0,  # phi < 15
            ]
            + [6.0] * num_alpha,  # alpha < 5 for each link (or global)
            dtype=np.float64,
        )

        return lower_bounds, upper_bounds

    def prepare_calibration_params(
        self,
        params: METANETParams,
        network: "Network",
        model_options: dict | None = None,
    ) -> NDArray[np.float64]:
        """Convert METANET parameters to a calibration vector.

        This method extends model_params_to_vec with support for choosing between
        link-specific and global alpha parameters, which is useful for reducing
        degrees of freedom and improving calibration robustness.

        Args:
            params: METANET parameter dictionary.
            network: Network instance.
            model_options: Dictionary of METANET-specific calibration options:
                - 'link_specific_alpha' (bool): If True, treats alpha as link-specific
                  (dict or scalar expanded to all links). If False, uses a single
                  global alpha value. Default: False.

        Returns:
            1-D numpy array of calibration parameters.

        Raises:
            ValueError: If params are invalid or inconsistent.
        """
        model_options = self._validate_model_options(model_options)
        link_specific_alpha = model_options.get("link_specific_alpha", False)

        # validate parameters
        self.validate_model_params(params)

        # extract scalar parameters
        scalar_params = np.array(
            [
                params["vf"],
                params["qc_lane"],
                params["rho_jam"],
                params["tau"],
                params["nu"],
                params["kappa"],
                params["delta"],
                params["phi"],
            ],
            dtype=np.float64,
        )

        # handle alpha parameter
        if link_specific_alpha:
            # use existing model_params_to_vec for link-specific alpha
            return self.model_params_to_vec(network=network, model_params=params)
        else:
            # use single global alpha value
            if isinstance(params["alpha"], dict):
                # if dict provided, take the mean (or first value)
                alpha_val = np.mean(list(params["alpha"].values()))
            else:
                alpha_val = params["alpha"]

            return np.concatenate(
                (scalar_params, np.array([alpha_val], dtype=np.float64))
            )

    def parse_calibration_params(
        self,
        param_vec: NDArray[np.float64],
        network: "Network",
        model_options: dict | None = None,
    ) -> METANETParams:
        """Convert a calibration vector back to METANET parameters.

        Inverse of prepare_calibration_params. Handles both link-specific and
        global alpha configurations.

        Args:
            param_vec: 1-D calibration parameter vector.
            network: Network instance.
            model_options: Dictionary of METANET-specific calibration options:
                - 'link_specific_alpha' (bool): If True, parses link-specific alpha
                  values. If False, parses a single global alpha. Default: False.

        Returns:
            METANET parameter dictionary.
        """
        model_options = self._validate_model_options(model_options)
        link_specific_alpha = model_options.get("link_specific_alpha", False)

        if link_specific_alpha:
            # use existing model_params_vec_to_dict for link-specific alpha
            # return value is known to be numerical based on inputs (non-symbolic)
            return cast(
                METANETParams,
                self.model_params_vec_to_dict(
                    network=network, model_params_vec=param_vec
                ),
            )
        else:
            # parse with global alpha
            return METANETParams(
                vf=param_vec[0],
                qc_lane=param_vec[1],
                rho_jam=param_vec[2],
                tau=param_vec[3],
                nu=param_vec[4],
                kappa=param_vec[5],
                delta=param_vec[6],
                phi=param_vec[7],
                alpha=param_vec[8],  # single global value
            )

    def prepare_system_params(
        self,
        param_vec: NDArray[np.float64],
        network: "Network",
        model_options: dict | None = None,
    ) -> NDArray[np.float64]:
        """Convert calibration parameter vector to system parameter vector.

        The CasADi system function always expects a full parameter vector with
        per-link alpha values. When using global alpha (link_specific_alpha=False),
        this method expands the single alpha value to all links. When using
        link-specific alpha, the parameter vector is returned unchanged.

        This method encapsulates model-specific logic for parameter vector conversion,
        keeping the Network class model-agnostic.

        Args:
            param_vec: 1-D calibration parameter vector. If link_specific_alpha=False,
                      this should have 9 elements (vf, qc_lane, rho_jam, tau, nu, kappa, delta, phi, alpha_global).
                      If True, it should have (8 + num_links) elements with per-link alphas.
            network: Network instance (used to count links for alpha expansion).
            model_options: Dictionary of METANET-specific calibration options:
                - 'link_specific_alpha' (bool): If False, param_vec contains a single
                  global alpha that will be replicated for all links. Default: False.

        Returns:
            System parameter vector ready to be passed to the CasADi system function.
            Always has format: [tau, nu, kappa, delta, phi, alpha_1, alpha_2, ..., alpha_n]
        """
        model_options = self._validate_model_options(model_options)
        link_specific_alpha = model_options.get("link_specific_alpha", False)

        # if using link-specific alpha, parameter vector is already in system format
        if link_specific_alpha:
            return param_vec

        # if using global alpha, expand to per-link format
        # param_vec is [tau, nu, kappa, delta, phi, alpha_global]
        # need to expand to [tau, nu, kappa, delta, phi, alpha_1, alpha_2, ..., alpha_n]
        if len(param_vec) == 9:
            # count unique motorway links, onramps, and offramps
            # use a set to avoid double-counting links that appear in both incoming and outgoing
            unique_links = set()
            for node in network.list_nodes():
                for link in node.incoming + node.outgoing:
                    if isinstance(link, (Onramp, MotorwayLink, Offramp)):
                        unique_links.add(link.id)

            num_links = len(unique_links)

            # expand parameter vector
            system_param_vec = np.zeros(8 + num_links)
            system_param_vec[:8] = param_vec[
                :8
            ]  # vf, qc_lane, rho_jam, tau, nu, kappa, delta, phi
            system_param_vec[8:] = param_vec[8]  # replicate global alpha
            return system_param_vec
        else:
            # parameter vector already in system format
            warnings.warn(
                "Expected 6 parameters for global alpha configuration, but got "
                f"{len(param_vec)}. Assuming parameter vector is already in system format.",
                stacklevel=2,
            )
            return param_vec

    # endregion

    # ! Fundamental diagram helper functions
    # region
    def critical_density(
        self,
        params: METANETParams | METANETSymbolicParams,
        link_id: str,
    ) -> float | casadi.SX:
        """
        Compute the METANET critical density for a link.

        The METANET critical density is defined as::
            rho_crit = lane_capacity / (free_flow_speed * exp(-1 / alpha))

        where ``alpha`` is the fundamental-diagram shape parameter. The
        ``alpha`` value is read from ``params['alpha']`` and may be either a
        scalar or a dictionary mapping ``link_id`` to a per-link value.

        Args:
            params: METANET model parameters (may be numeric or symbolic).
            link_id: Identifier of the link for which to compute ``alpha``.

        Returns:
            Critical density (vehicles per length per lane). When symbolic
            parameters are provided the returned expression may be a CasADi
            symbolic value.

        Raises:
            ValueError: If the resolved ``alpha`` is not positive.
        """
        alpha = (
            params["alpha"][link_id]
            if isinstance(params["alpha"], dict)
            else params["alpha"]
        )

        return params["qc_lane"] / (
            params["vf"]
            * (
                casadi.exp(-1 / alpha)
                if isinstance(alpha, casadi.SX)
                else np.exp(-1 / alpha)
            )
        )

    def backward_wave_speed(
        self,
        params: METANETParams | METANETSymbolicParams,
        link_id: str,
        lanes: float,
    ) -> float | casadi.SX:
        """
        Return the backward (congestion) wave speed for given fundamental parameters.

        The backward wave speed is computed as capacity / (jam_density - rho_crit)
        where rho_crit is the critical density computed from lane_capacity and
        free_flow_speed. This speed describes how congestion propagates upstream
        (length per time).

        Args:
            params: METANET model parameters (may be numeric or symbolic).
            link_id: Identifier of the link for which to compute the backward wave speed.
            lanes: Number of lanes on the link (used to compute total capacity).

        Returns:
            Backward wave speed (length per time).

        Raises:
            ValueError: If jam_density is less than or equal to the critical density.
        """

        rho_crit = self.critical_density(
            params=params,
            link_id=link_id,
        )

        return lanes * params["qc_lane"] / (params["rho_jam"] - rho_crit)

    def stationary_velocity(
        self,
        params: METANETParams | METANETSymbolicParams,
        link_id: str,
        density: float | casadi.SX,
    ) -> float | casadi.SX:
        """Compute the stationary (equilibrium) velocity for a cell.

        The stationary velocity is the speed that the traffic on the cell would
        adopt in the absence of dynamics, given the current density. METANET
        uses an exponential functional form parameterized by ``alpha`` and the
        cell's free-flow speed (fundamental diagram).

        Args:
            params (METANETParams | METANETSymbolicParams): METANET model parameters.
            link_id (str): Identifier of the link for which to compute the stationary velocity.
            density (float | casadi.SX): The density at which to evaluate the
                stationary velocity (vehicles per length per lane).

        Returns:
            The stationary velocity (length per time unit).
        """

        alpha = (
            params["alpha"][link_id]
            if isinstance(params["alpha"], dict)
            else params["alpha"]
        )
        exponent = (
            -1
            / alpha
            * (density / self.critical_density(params=params, link_id=link_id)) ** alpha
        )

        return (
            params["vf"] * casadi.exp(exponent)
            if isinstance(density, casadi.SX)
            else params["vf"] * np.exp(exponent)
        )

    # endregion

    # ! Symbolic model parameter handling (validation, packing, unpacking)
    # region
    def validate_model_params(self, model_params: METANETParams) -> None:
        """Validate a METANET model parameter dictionary.

        Ensures required keys are present and their types are correct. The
        first three parameters (`vf`, `qc_lane`, `rho_jam`) are fundamental
        fundamental diagram parameters and must be numeric (int or float). All
        scalar model parameters (`tau`, `nu`, `kappa`, `delta`, `phi`) must be
        numeric (int or float). The `alpha` parameter may be either a scalar
        (int or float) or a dictionary mapping link ids to numeric values.

        Args:
            model_params: Dictionary containing METANET model parameters.

        Raises:
            ValueError: If a required parameter is missing or has an invalid type.
        """

        required_params = [
            "vf",
            "qc_lane",
            "rho_jam",
            "tau",
            "nu",
            "kappa",
            "delta",
            "phi",
            "alpha",
        ]
        for param in required_params:
            if param not in model_params:
                raise ValueError(f"Missing required METANET model parameter: {param}")

        # make sure that the parameters have the correct type
        # all parameters should be scalars, while the alpha parameter may be a scalar or link-specific
        for param in required_params:
            if param != "alpha":
                if not isinstance(model_params[param], (int, float)):
                    raise ValueError(
                        f"METANET model parameter {param} must be a scalar (int or float)."
                    )
            else:
                if not (
                    isinstance(model_params["alpha"], (int, float))
                    or (
                        isinstance(model_params["alpha"], dict)
                        and all(
                            isinstance(k, str) and isinstance(v, (int, float))
                            for k, v in model_params["alpha"].items()
                        )
                    )
                ):
                    raise ValueError(
                        f"METANET model parameter {param} must be either a scalar (int or float) or a mapping returning a scalar."
                    )

    def model_params_to_vec(
        self,
        network: "Network",
        model_params: METANETParams,
    ) -> NDArray[np.float64]:
        """Convert a METANET parameter dictionary to a numerical vector.

        The returned vector layout is identical to that used by the
        symbolic parameter vector: the five scalar parameters in the order
        `vf, qc_lane, rho_jam, tau, nu, kappa, delta, phi` followed by the
        per-link `alpha` parameters for all motorway links, onramps and
        offramps in the same ordering used throughout the model formulation.

        Args:
            network: Network instance used to determine the ordering of link
                specific `alpha` entries.
            model_params: Parameter dictionary containing scalar parameters
                and either a scalar `alpha` or a dict mapping link ids to
                `alpha` values.

        Returns:
            1-D numpy array of dtype `np.float64` containing the packed
            model parameters suitable for use with the symbolic routines.
        """

        # augment the model parameters dictionary for links where necessary
        alpha_vector = np.array([], dtype=np.float64)

        link_ids: list[str] = []  # links for which alpha values are required
        for node in network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Onramp):
                    link_ids.append(link.id)
            for link in node.outgoing:
                if isinstance(link, MotorwayLink) or isinstance(link, Offramp):
                    link_ids.append(link.id)

        for link_id in link_ids:
            if isinstance(model_params["alpha"], float) or isinstance(
                model_params["alpha"], int
            ):
                alpha_vector = np.append(alpha_vector, model_params["alpha"])
            else:
                casted_params_function = cast(dict[str, float], model_params["alpha"])
                alpha_vector = np.append(alpha_vector, casted_params_function[link_id])

        return np.concatenate(
            (
                np.array(
                    [
                        model_params["vf"],
                        model_params["qc_lane"],
                        model_params["rho_jam"],
                        model_params["tau"],
                        model_params["nu"],
                        model_params["kappa"],
                        model_params["delta"],
                        model_params["phi"],
                    ],
                    dtype=np.float64,
                ),
                alpha_vector,
            )
        )

    def model_params_vec_to_dict(
        self,
        network: "Network",
        model_params_vec: NDArray[np.float64] | casadi.SX,
    ) -> METANETParams | METANETSymbolicParams:
        """Convert a packed parameter vector into a METANET parameter dict.

        This is the inverse of `model_params_to_vec`. The input vector is
        expected to contain the five scalar parameters followed by the per
        link `alpha` values in the ordering implied by `network.list_nodes()`
        and the per-node link ordering used elsewhere in the model.

        Args:
            network: Network instance used to determine link ordering for
                unpacking the `alpha` entries.
            model_params_vec: 1-D numpy array or CasADi SX vector containing
                the packed model parameters.

        Returns:
            A dictionary mapping scalar parameter names to their values and
            `alpha` to a dict of per-link values. When `model_params_vec` is
            symbolic (`casadi.SX`) the returned values will be symbolic as
            well (see `METANETSymbolicParams`).
        """

        # extract the scalar parameters
        vf: float | casadi.SX = model_params_vec[0]
        qc_lane: float | casadi.SX = model_params_vec[1]
        rho_jam: float | casadi.SX = model_params_vec[2]
        tau: float | casadi.SX = model_params_vec[3]
        nu: float | casadi.SX = model_params_vec[4]
        kappa: float | casadi.SX = model_params_vec[5]
        delta: float | casadi.SX = model_params_vec[6]
        phi: float | casadi.SX = model_params_vec[7]

        # extract the alpha parameters for each link
        alpha_vector = model_params_vec[8:]
        alpha_dict: dict[str, float | casadi.SX] = {}
        alpha_index = 0

        link_ids: list[str] = []
        for node in network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Onramp):
                    link_ids.append(link.id)
            for link in node.outgoing:
                if isinstance(link, MotorwayLink) or isinstance(link, Offramp):
                    link_ids.append(link.id)

        for link_id in link_ids:
            alpha_dict[link_id] = alpha_vector[alpha_index]
            alpha_index += 1

        return {
            "vf": vf,
            "qc_lane": qc_lane,
            "rho_jam": rho_jam,
            "tau": tau,
            "nu": nu,
            "kappa": kappa,
            "delta": delta,
            "phi": phi,
            "alpha": alpha_dict,
        }

    def set_up_symbolic_model_params(
        self,
        network: "Network",
    ) -> casadi.SX:
        """Create a CasADi symbolic vector for METANET model parameters.

        The created `casadi.SX` vector contains one entry for each scalar
        parameter (`vf, qc_lane, rho_jam, tau, nu, kappa, delta, phi`) followed by one `alpha`
        entry per motorway link, onramp and offramp in the network. The
        ordering of per-link `alpha` entries matches the unpacking performed
        in `model_params_vec_to_dict`.

        Args:
            network: Network instance used to count links and determine the
                length of the symbolic parameter vector.

        Returns:
            A `casadi.SX` symbolic column vector of shape `(num_links + 8, 1)`.
        """

        # count the number of motorway links, onramps and offramps in the network
        num_links = 0
        for node in network.list_nodes():
            for link in node.incoming:
                if isinstance(link, Onramp):
                    num_links += 1
            for link in node.outgoing:
                if isinstance(link, MotorwayLink) or isinstance(link, Offramp):
                    num_links += 1

        # create symbolic variables for the model parameters (one entry for the alpha of each link)
        # 8 more entries for the parameters vf, qc_lane, rho_jam, tau, nu, kappa, delta, phi
        model_params_sym = casadi.SX.sym("metanet_params", num_links + 8, 1)  # type: ignore
        return model_params_sym

    # endregion

    # ! Network update helper functions
    # region
    def _compute_virtual_downstream_density(
        self,
        node: Node,
        params: METANETSymbolicParams,
        densities: dict[str, casadi.SX],
        density_boundary_conditions: dict[str, casadi.SX],
    ) -> Tuple[casadi.SX, Union[casadi.SX, None], Union[casadi.SX, None]]:
        """Determine a node's virtual downstream density for METANET updates.

        The virtual downstream density is used when computing boundary and
        anticipation terms at nodes with multiple outgoing links. For each
        outgoing link the method selects an appropriate downstream density
        representation:
        - For a `MotorwayLink`: the density of its first cell is used.
        - For an `Onramp`: the density is set to zero (free-flow conditions)
          in order not to influence the inflow from the virtual upstream
          origin onto the store-and-forward onramp link
        - For an `Offramp`: the connected destination's boundary condition
          is used (offramps do not carry internal density in the
          store-and-forward representation).
        - For a `Destination`: the provided boundary condition is used.

        If multiple motorway or destination densities are present they are
        combined using a weighted quadratic mean implemented as
        ``sum(d**2)/sum(d)`` (CasADi symbolic expression).

        Args:
            node (Node): Network node for which to compute the downstream
                density.
            params (METANETSymbolicParams): METANET model parameters (CasADi SX).
            densities (dict[str, casadi.SX]): Mapping link id -> vector of
                cell densities (CasADi SX) for motorway links.
            density_boundary_conditions (dict[str, casadi.SX]): Mapping of link or
                destination id to boundary density (CasADi SX).

        Returns:
            Tuple[casadi.SX, Union[casadi.SX, None], Union[casadi.SX, None]]: A tuple containing:
                - The virtual downstream density of the node (CasADi SX).
                - The virtual downstream jam density of the node (CasADi SX or None).
                - The virtual downstream backward wave speed of the node (CasADi SX or None).

        Raises:
            ValueError: If the node has no outgoing links or an offramp has
                no destination defined.
            TypeError: If an outgoing link has an unexpected type.
        """
        # initialize variables
        node_downstream_density: casadi.SX = casadi.SX(0)
        node_downstream_jam_density: casadi.SX | None = None
        node_downstream_backward_wave_speed: casadi.SX | None = None

        # determine the virtual downstream density of the node based on the outgoing links = q_m,N_m+1(k)
        if len(node.outgoing) > 1:
            out_densities: list[casadi.SX] = []

            for out_link in node.outgoing:
                if isinstance(out_link, MotorwayLink):
                    # motorway link: use the density of the first cell as downstream density
                    out_densities.append(densities[out_link.id][0])

                elif isinstance(out_link, Offramp):
                    # offramp link: the store-and-forward model does not model density / speed on offramps
                    # -> free flow conditions are assumed for traffic entering this link from the mainline
                    # -> if the boundary condition is too restrictive, a virtual queue will form on the offramp
                    # (no density is defined for the offramp in the case of multiple outgoing links)
                    continue

                elif isinstance(out_link, Destination):
                    # destination link: density is provided as boundary condition
                    out_densities.append(density_boundary_conditions[out_link.id])

                elif isinstance(out_link, Onramp):
                    # onramps can only be linked as only outgoing link to a node
                    raise ValueError(
                        f"Onramp {out_link.id} cannot be an outgoing link at a node with multiple outgoing links."
                    )

                else:
                    raise TypeError(f"Unknown outgoing link type {type(out_link)}")

            if len(out_densities) == 0:
                # if only offramps were present as outgoing links at the
                # current node, assume downstream free flow conditions
                node_downstream_density = casadi.SX(0)
            else:
                # combine the different downstream densities (e.g., weighted average)
                numer = casadi.sum(casadi.vertcat(*[d**2 for d in out_densities]))
                denom = casadi.sum(casadi.vertcat(*out_densities))
                node_downstream_density = casadi.if_else(denom == 0, 0, numer / denom)

            # for multiple outgoing links, the virtual downstream jam density
            # and backward wave speed are not well-defined
            node_downstream_jam_density = None
            node_downstream_backward_wave_speed = None

        # single outgoing link: directly use its downstream density
        elif len(node.outgoing) == 1:
            out_link = node.outgoing[0]

            if isinstance(out_link, MotorwayLink):
                # motorway link: use the density of the first cell as downstream density
                node_downstream_density = densities[out_link.id][0]
                node_downstream_jam_density = casadi.SX(params["rho_jam"])
                node_downstream_backward_wave_speed = casadi.SX(
                    self.backward_wave_speed(
                        params=params, link_id=out_link.id, lanes=out_link.lanes
                    )
                )

            elif isinstance(out_link, Offramp):
                # offramp link: the store-and-forward model does not model density / speed on offramps
                # -> free flow conditions are assumed for traffic entering this link from the mainline
                # -> if the boundary condition is too restrictive, a virtual queue will form on the offramp
                node_downstream_density = casadi.SX(0)
                node_downstream_jam_density = None  # no downstream jam density defined -> handling on calling level required
                node_downstream_backward_wave_speed = None  # no downstream backward wave speed defined -> handling on calling level required

            elif isinstance(out_link, Destination):
                # destination link: density is provided as boundary condition
                node_downstream_density = density_boundary_conditions[out_link.id]
                node_downstream_jam_density = None  # no downstream jam density defined -> handling on calling level required
                node_downstream_backward_wave_speed = None  # no downstream backward wave speed defined -> handling on calling level required

            elif isinstance(out_link, Onramp):
                # for onramps no downstream supply of space restrictions should be imposed
                # -> all inflow from the virtual upstream origin should be consumed
                # free-flow conditions downstream of the node
                node_downstream_density = casadi.SX(0)
                node_downstream_jam_density = None  # no downstream jam density defined -> handling on calling level required
                node_downstream_backward_wave_speed = None  # no downstream backward wave speed defined -> handling on calling level required

            else:
                raise TypeError(f"Unknown outgoing link type {type(out_link)}")
        else:
            raise ValueError(f"No outgoing links defined for node {node.id}")

        return (
            node_downstream_density,
            node_downstream_jam_density,
            node_downstream_backward_wave_speed,
        )

    def _compute_node_upstream_speed(
        self,
        params: METANETSymbolicParams,
        node: Node,
        flows: dict[str, casadi.SX],
        speeds: dict[str, casadi.SX],
    ) -> casadi.SX:
        """Compute the virtual upstream speed for METANET updates.

        The method computes the total available flow into the node by summing
        the last cell flows of all incoming motorway links as well as the
        flows from origins and onramps. Based on the total available flow and
        the provided split ratios, the method computes the outflows for each
        outgoing link. Additionally, the method computes the virtual upstream speed
        used in the speed update equations for outgoing motorway links.

        Args:
            node (Node): Network node for which to compute the virtual upstream speed.
            params (METANETSymbolicParams): METANET model parameters (CasADi SX).
            flows (dict[str, casadi.SX]): Mapping link id -> vector of cell
                flows (CasADi SX) for motorway links, origins, onramps and
                offramps.
            speeds (dict[str, casadi.SX]): Mapping link id -> vector of cell
                speeds (CasADi SX) for motorway links.

        Returns:
            casadi.SX: The virtual upstream speed (CasADi SX) used in speed updates.
        """
        # determine the virtual upstream speed of the node (for outgoing motorway links) = v_m,0(k)
        # since a speed parameter is required, only incoming motorway links are considered
        # if all incoming links are origins, assume free flow conditions upstream
        if all(not isinstance(inc, MotorwayLink) for inc in node.incoming):
            # if any onramps are connected to the node, use the minimum free-flow speed of those onramps
            if any(isinstance(inc, Onramp) for inc in node.incoming):
                node_upstream_speed = casadi.SX(
                    min(
                        params["vf"] for inc in node.incoming if isinstance(inc, Onramp)
                    )
                )

            # for incoming offramps, assume free flow speed -> outflow to destination should
            # not be restricted through upstream effects -> accept all discharged flow from the offramp
            elif len(node.incoming) == 1 and isinstance(node.incoming[0], Offramp):
                node_upstream_speed = casadi.SX(params["vf"])

            # if only an origin is connected as an incoming link (and correspondingly only one outgoing
            # motorway link or onramp is allowed), choose the free-flow speed of the outgoing motorway link
            # or onramp for consistency (origin does not have free flow speed defined)
            else:
                if (
                    len(node.incoming) != 1
                    or not isinstance(node.incoming[0], Origin)
                    or len(node.outgoing) != 1
                    or not isinstance(node.outgoing[0], (MotorwayLink, Onramp))
                ):
                    raise ValueError(
                        "Encountered node without expected types of input links (more than one Origin / more than one outgoing link for origin-linked node)."
                    )

                node_upstream_speed = casadi.SX(params["vf"])

        else:
            # keep track of the minimum free-flow speed of incoming motorway links
            # -> in case upstream flow is zero, use this value as upstream speed
            min_vf: casadi.SX = casadi.SX(casadi.inf)

            numer_terms = []
            denom_terms = []
            for inc in node.incoming:
                if isinstance(inc, MotorwayLink):
                    # motorway link: use the last cell speed and flow for upstream speed
                    numer_terms.append(speeds[inc.id][-1] * flows[inc.id][-1])
                    denom_terms.append(flows[inc.id][-1])
                    min_vf = casadi.fmin(min_vf, params["vf"])

            # catch the case where no values were measured -> should not happen
            if len(numer_terms) == 0 or len(denom_terms) == 0:
                raise ValueError(
                    f"No incoming motorway links with defined speeds/flows for node {node.id}."
                )

            numer_sum = casadi.sum(casadi.vertcat(*numer_terms))
            denom_sum = casadi.sum(casadi.vertcat(*denom_terms))
            node_upstream_speed = casadi.if_else(
                denom_sum == 0, min_vf, numer_sum / denom_sum
            )

        return node_upstream_speed

    def _compute_offramp_outflows(
        self,
        params: METANETSymbolicParams,
        offramp: Offramp,
        node_outflows: dict[str, casadi.SX],
        offramp_queues: dict[str, casadi.SX],
        density_boundary_condition: casadi.SX,
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX]:
        """Compute offramp outflow and update the offramp store-and-forward queue.

        Offramps are represented as store-and-forward links with finite
        capacity. This method computes the desired mainline outflow onto the
        offramp (from `node_outflows`) and combines it with the current
        offramp queue to form an offramp demand. The actual offramp outflow
        and the updated queue are computed via `store_and_forward_update`,
        which uses the offramp's capacity, jam density and the downstream
        (destination) boundary density.

        Args:
            offramp (Offramp): The offramp link for which to compute outflow.
            node_outflows (dict[str, casadi.SX]): Mapping of outgoing link id
                to the desired outflow at the node (CasADi SX).
            offramp_queues (dict[str, casadi.SX]): Current queue lengths on
                offramps (CasADi SX).
            density_boundary_condition (casadi.SX): Downstream virtual density
                constraint of the connected destination (vehicles / length / lane)
            dt (float): Simulation timestep.

        Returns:
            Tuple[casadi.SX, casadi.SX]: Tuple `(next_outflow, next_queue)`
            where `next_outflow` is the offramp outflow (vehicles/time) into
            the connected destination, and `next_queue` is the updated queue
            length on the offramp (vehicles).

        Raises:
            ValueError: If the `offramp` does not have a `destination` defined.
        """
        mainline_outflow = node_outflows[
            offramp.id
        ]  # desired offramp flow based on splits = flow onto offramp (queue on offramp itself)

        # update the offramp flow and queue based on the store-and-forward model
        next_outflow, next_queue = store_and_forward_update(
            params=params,
            lanes=offramp.lanes,
            jam_density=params["rho_jam"],
            backward_wave_speed=self.backward_wave_speed(
                params=params, link_id=offramp.id, lanes=offramp.lanes
            ),
            density=density_boundary_condition,
            demand=mainline_outflow,  # (additional demand from queue added automatically)
            queue=offramp_queues[offramp.id],
            dt=dt,
        )

        return next_outflow, next_queue

    def _compute_motorway_link_outflows(
        self,
        params: METANETSymbolicParams,
        link: MotorwayLink,
        node: Node,
        node_outflows: dict[str, casadi.SX],
        node_downstream_density: casadi.SX,
        node_upstream_speed: casadi.SX,
        node_upstream_onramp_inflows: casadi.SX | None,
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        speeds: dict[str, casadi.SX],
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX, casadi.SX]:
        """Compute next-step densities, speeds and flows for a motorway link.

        This helper advances all cells on a `MotorwayLink` by one simulation
        timestep using the METANET `cell_update` routine. The method applies
        the node-level outflow as the upstream boundary condition for the
        first cell and uses `node_downstream_density` for the downstream
        boundary condition of the last cell. Per-cell upstream speeds are
        provided via `node_upstream_speed` for the first cell and the
        `speeds` vector for internal cells.

        Args:
            params (METANETSymbolicParams): Model parameters.
            link (MotorwayLink): Motorway link containing the cells to update.
            node (Node): The upstream node of the link (used for error
                messages and contextual checks).
            node_outflows (dict[str, casadi.SX]): Mapping of outgoing link id
                to the node-level outflow (CasADi SX).
            node_downstream_density (casadi.SX): Virtual downstream density
                at the node used as boundary for the last cell (CasADi SX).
            node_upstream_speed (casadi.SX): Virtual upstream speed at the
                node used as boundary for the first cell (CasADi SX).
            node_upstream_onramp_inflows (casadi.SX | None): Total inflow
                from onramps connected upstream of the node used to account
                for speed reduction terms caused by merging traffic.
            flows (dict[str, casadi.SX]): Current per-link flow vectors
                (CasADi SX).
            densities (dict[str, casadi.SX]): Current per-link density vectors
                (CasADi SX).
            speeds (dict[str, casadi.SX]): Current per-link speed vectors
                (CasADi SX).
            dt (float): Simulation timestep.

        Returns:
            Tuple[casadi.SX, casadi.SX, casadi.SX]: Three CasADi column vectors
            of length equal to the number of cells on `link` containing the
            virtual downstream link densities, speeds and outflows respectively.

        Raises:
            ValueError: If no node outflow has been computed for `link.id`.
        """

        if node_outflows[link.id] is None:
            raise ValueError(
                f"No outflow computed for outgoing motorway link {link.id} at node {node.id}"
            )

        link_flows = flows[link.id]
        link_densities = densities[link.id]
        link_speeds = speeds[link.id]

        next_densities_list = casadi.SX(len(link), 1)
        next_speeds_list = casadi.SX(len(link), 1)
        next_flows_list = casadi.SX(len(link), 1)

        for i, cell in link.enumerate_cells():
            # compute updates for this cell and append to lists
            d_next, s_next, f_next = self.cell_update(
                params=params,
                link=link,
                cell=cell,
                upstream_flow=(
                    link_flows[i - 1]
                    if cell.upstream is not None
                    else node_outflows[link.id]
                ),
                previous_flow=link_flows[i],
                previous_density=link_densities[i],
                downstream_density=(
                    link_densities[i + 1]
                    if cell.downstream is not None
                    else node_downstream_density
                ),
                upstream_speed=(
                    link_speeds[i - 1]
                    if cell.upstream is not None
                    else node_upstream_speed
                ),
                upstream_onramp_inflows=(
                    node_upstream_onramp_inflows if cell.upstream is None else None
                ),
                previous_speed=link_speeds[i],
                dt=dt,
            )

            next_densities_list[i] = d_next
            next_speeds_list[i] = s_next
            next_flows_list[i] = f_next

        return next_densities_list, next_speeds_list, next_flows_list

    def cell_update(
        self,
        params: METANETSymbolicParams,
        link: MotorwayLink,
        cell: Cell,
        upstream_flow: casadi.SX,
        previous_flow: casadi.SX,
        previous_density: casadi.SX,
        downstream_density: casadi.SX,
        upstream_speed: casadi.SX,
        upstream_onramp_inflows: casadi.SX | None,
        previous_speed: casadi.SX,
        dt: float,
    ) -> Tuple[casadi.SX, casadi.SX, casadi.SX]:
        """Compute one-step updates for density, speed and flow of a METANET cell.

        Implements the METANET discrete-time update for a homogeneous motorway
        cell. Density is updated by vehicle conservation using the provided
        upstream and previous outflow. Speed evolves according to METANET's
        second-order dynamics: relaxation toward the stationary velocity,
        convective coupling with upstream speed, anticipation of downstream
        density gradients, and additional deceleration for upcoming lane
        drops. The updated flow is computed from the updated density and
        speed.

        Args:
            params (METANETSymbolicParams): Model parameters.
            link (MotorwayLink): Parent motorway link containing geometric
                and lane information.
            cell (Cell): Cell object with geometry and lane-drop info.
            upstream_flow (casadi.SX): Flow entering the cell from upstream
                (vehicles / time).
            previous_flow (casadi.SX): Flow leaving the cell at the previous
                time step (vehicles / time).
            previous_density (casadi.SX): Density at the previous time step
                (vehicles / length per lane).
            downstream_density (casadi.SX): Density in the downstream cell
                used for anticipation terms (vehicles / length per lane).
            upstream_speed (casadi.SX): Speed in the upstream cell used for
                convective coupling (length / time).
            upstream_onramp_inflows (casadi.SX | None): Total onramp inflow
                entering upstream of the cell used for speed reduction terms.
            previous_speed (casadi.SX): Speed at the previous time step in
                this cell (length / time).
            dt (float): Simulation timestep.

        Returns:
            Tuple[casadi.SX, casadi.SX, casadi.SX]: ``(density, speed, flow)``
            updated for one timestep where:
            - ``density``: Updated density (vehicles / length per lane).
            - ``speed``: Updated speed (length / time).
            - ``flow``: Updated outflow from the cell (vehicles / time).
        """

        # compute the new density based on the flows at the previous timestep
        # Note: off-ramps are modeled as splitting the outflow and do not
        # directly reduce the density update term (matches MATLAB METANET).
        density = previous_density + dt * (upstream_flow - previous_flow) / (
            cell.length * link.lanes
        )

        # compute the new speed based on the previous timestep
        speed = (
            previous_speed
            + dt
            / params["tau"]
            * (
                self.stationary_velocity(
                    params=params,
                    link_id=link.id,
                    density=previous_density,
                )
                - previous_speed
            )
            + dt / cell.length * previous_speed * (upstream_speed - previous_speed)
            - (dt * params["nu"])
            / (params["tau"] * cell.length)
            * (downstream_density - previous_density)
            / (previous_density + params["kappa"])
        )

        # if the considered cell is the last downstream cell of a link into a node with onramp inflows, reduce the speed accordingly
        if upstream_onramp_inflows is not None:
            speed -= (
                (dt * params["delta"])
                / (cell.length * link.lanes)
                * (upstream_onramp_inflows * previous_speed)
                / (previous_density + params["kappa"])
            )

        # if a lane drop is coming up, add an additional term to the speed
        # update equation to account for the additional deceleration
        if cell.upcoming_lane_drop > 0:
            speed -= (
                dt
                * params["phi"]
                * cell.upcoming_lane_drop
                * previous_density
                * previous_speed**2
            ) / (
                cell.length
                * link.lanes
                * self.critical_density(params=params, link_id=link.id)
            )

        # ensure that the speed values remain non-negative
        speed = casadi.fmax(speed, 0)

        # compute the flow update of the cell based on the speed and density
        flow = density * speed * link.lanes

        return density, speed, flow

    # endregion

    def network_update_function(
        self,
        network: "Network",
        num_flows: int,
        num_densities: int,
        num_speeds: int,
        num_origins: int,
        num_onramps: int,
        num_offramps: int,
        num_splits: int,
        num_destinations: int,
        dt: float,
    ) -> casadi.Function:
        """Build a CasADi function implementing one METANET network step.

        The returned CasADi `Function` (named ``metanet_network_step``) maps
        the symbolic model parameter vector, the current state vector ``x``
        and the disturbance vector ``d`` to the next-step state vector
        ``x_next`` according to the METANET dynamics combined with
        store-and-forward updates for origins, onramps and offramps.

        State and disturbance vector layouts follow
        `Network.state_vec_to_network_dict` and
        `Network.disturbance_vec_to_network_dict`. The disturbance vector
        contains origin demands, split ratios and boundary condition
        entries in the ordering expected by the network helpers.

        Args:
            network (Network): Network object containing links, nodes and
                helper methods to convert between vectors and dictionaries.
            num_flows (int): Length of the flow portion of the state vector.
            num_densities (int): Length of the density portion of the state
                vector.
            num_speeds (int): Length of the speed portion of the state vector.
            num_origins (int): Number of origin links (state/disturbance size).
            num_onramps (int): Number of onramp links (state/disturbance size).
            num_offramps (int): Number of offramp links (state/disturbance size).
            num_splits (int): Number of split-ratio disturbance entries.
            num_destinations (int): Number of boundary-condition disturbance entries.
            dt (float): Simulation timestep.

        Returns:
            casadi.Function: A CasADi function `f(params, x, d) -> x_next`
            implementing the network update for one timestep. The first
            argument to the function is the symbolic model parameter vector
            produced by `set_up_symbolic_model_params`.
        """

        # ! Store the model parameters in a dedicated vector to be used for evaluation
        sym_params = self.set_up_symbolic_model_params(network=network)

        # load the model parameters in dictionary form for easy access during function formulation
        params: METANETSymbolicParams = cast(
            METANETSymbolicParams,
            self.model_params_vec_to_dict(
                network=network,
                model_params_vec=sym_params,
            ),
        )

        # ! Set up variables for state update and cast types to be correct
        # set up state and disturbance vectors
        # state: flows, densities, speeds, origin, onramp
        # disturbances: origin_demands, offramp_split_ratios
        # CasADi type stubs are incorrect - sym() does accept string as first arg
        x = casadi.SX.sym(  # type: ignore
            "x",  # type: ignore
            num_flows
            + num_densities
            + num_speeds
            + num_origins
            + num_onramps
            + num_offramps,  # type: ignore
            1,  # type: ignore
        )
        d = casadi.SX.sym("d", num_origins + num_splits + 2 * num_destinations, 1)  # type: ignore

        # split up the state and disturbance vectors to obtain a dictionary for
        # efficient access of the relevant quantities during the state update
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            network.state_vec_to_network_dict(x=x)
        )
        # flow boundary conditions are not extracted for METANET, since they are not needed
        origin_demands, splits, _, density_boundary_conditions = (
            network.disturbance_vec_to_network_dict(d=d)
        )

        # typecast values of the dictionaries to casadi SX for symbolic computation
        flows = {k: casadi.SX(v) for k, v in flows.items()}
        densities = {k: casadi.SX(v) for k, v in densities.items()}
        speeds = {k: casadi.SX(v) for k, v in speeds.items()}
        origin_queues = {k: casadi.SX(v) for k, v in origin_queues.items()}
        onramp_queues = {k: casadi.SX(v) for k, v in onramp_queues.items()}
        offramp_queues = {k: casadi.SX(v) for k, v in offramp_queues.items()}
        origin_demands = {k: casadi.SX(v) for k, v in origin_demands.items()}
        splits = {
            k: {kk: casadi.SX(vv) for kk, vv in v.items()} for k, v in splits.items()
        }
        density_boundary_conditions = {
            k: casadi.SX(v) for k, v in density_boundary_conditions.items()
        }

        # initialize next-step state dictionaries
        next_flows: dict[str, casadi.SX] = {}
        next_densities: dict[str, casadi.SX] = {}
        next_speeds: dict[str, casadi.SX] = {}
        next_origin_queues: dict[str, casadi.SX] = {}
        next_onramp_queues: dict[str, casadi.SX] = {}
        next_offramp_queues: dict[str, casadi.SX] = {}

        # order the nodes such that nodes connected to a destination are processed last
        # -> this is required to ensure that all incoming flows for this node have been computed
        # (otherwise the node inflow computation breaks)
        network_nodes = list(network.list_nodes())
        network_nodes.sort(
            key=lambda n: any(isinstance(out, Destination) for out in n.outgoing)
        )

        # formulate the individual update equations for each node and update the overall system equation and the next step state
        # iterate through all nodes and update the corrresponding quantities of incoming and outgoing links
        for node in network_nodes:
            # ! 1) update the flows and queues for origins and onramps connected to this node
            for inc in node.incoming:
                # compute the virtual downstream density for the current node
                # -> required for onramp and origin store-and-forward state updates
                (
                    node_virtual_downstream_density,
                    node_virtual_downstream_jam_density,
                    node_virtual_downstream_backward_wave_speed,
                ) = self._compute_virtual_downstream_density(
                    node=node,
                    params=params,
                    densities=densities,
                    density_boundary_conditions=density_boundary_conditions,
                )

                if isinstance(inc, Origin):
                    next_inflow, next_queue = store_and_forward_update(
                        params=params,
                        lanes=int(
                            1e8
                        ),  # very large number (virtual infinity) to avoid that this is limiting
                        jam_density=(
                            node_virtual_downstream_jam_density
                            if node_virtual_downstream_jam_density is not None
                            else casadi.inf
                        ),
                        backward_wave_speed=(
                            node_virtual_downstream_backward_wave_speed
                            if node_virtual_downstream_backward_wave_speed is not None
                            else casadi.inf
                        ),
                        density=node_virtual_downstream_density,
                        demand=origin_demands[inc.id],
                        queue=origin_queues[inc.id],
                        dt=dt,
                    )

                    next_flows[inc.id] = next_inflow
                    next_origin_queues[inc.id] = next_queue

                elif isinstance(inc, Onramp):
                    # for onramps, get the outflow of the upstream connected origin node as the inflow
                    # (as validation, ensure that the upstream node is only connected to the origin and onramp)
                    if inc.origin_node_id is None:
                        raise ValueError(
                            f"Onramp {inc.id} does not have an upstream origin node defined."
                        )

                    upstream_node = network.get_node(id=inc.origin_node_id)
                    if (
                        upstream_node is None
                        or len(upstream_node.incoming) > 1
                        or not isinstance(upstream_node.incoming[0], Origin)
                    ):
                        raise ValueError(
                            f"Upstream node {inc.origin_node_id} of onramp {inc.id} is not connected to exactly one origin."
                        )

                    # get the origin outflow
                    origin_outflow = flows[upstream_node.incoming[0].id]

                    # if a controller is defined for the on-ramp, compute the regulated outflow
                    if inc.controller is not None:
                        r_k = inc.controller.compute_regulated_flow(
                            onramp_queues=onramp_queues,
                            flows=flows,
                            densities=densities,
                        )
                    else:
                        r_k = casadi.SX(casadi.inf)

                    # compute the next-step on-ramp flow, respecting queue demand, capacity constraints,
                    # downstream traffic conditions and, if defined, the metering rate from a potential controller
                    next_inflow, next_queue = store_and_forward_update(
                        params=params,
                        lanes=inc.lanes,
                        metering_rate=r_k,
                        jam_density=(
                            node_virtual_downstream_jam_density
                            if node_virtual_downstream_jam_density is not None
                            else casadi.inf
                        ),
                        backward_wave_speed=(
                            node_virtual_downstream_backward_wave_speed
                            if node_virtual_downstream_backward_wave_speed is not None
                            else casadi.inf
                        ),
                        density=node_virtual_downstream_density,
                        demand=origin_outflow,
                        queue=onramp_queues[inc.id],
                        dt=dt,
                    )

                    next_flows[inc.id] = next_inflow
                    next_onramp_queues[inc.id] = next_queue

            # ! 2) compute the required boundary conditions based on the combined incoming / outgoing quantities
            # -> this includes the upstream speed, downstream density, etc. that are required by the model udpate
            # sum up the last cell flows of all incoming links, onramps and origins
            node_outflows = compute_node_outflows(
                node=node,
                flows=flows,
                node_splits=splits[node.id],
            )
            node_upstream_speed = self._compute_node_upstream_speed(
                params=params,
                node=node,
                flows=flows,
                speeds=speeds,
            )

            for out in node.outgoing:
                # ! 3) update the offramp flows (& density/speed) for destinations connected to this node
                if isinstance(out, Onramp):
                    # onramp flows are updated when considered as an incoming link
                    continue

                elif isinstance(out, Destination):
                    # destinations are assumed to consume all incoming flow
                    # (only impact the mainstream through the density boundary condition)
                    # Note: the next-step flows are already used, since the destination is
                    # only a virtual sink that does not have a flow state itself and no length
                    next_node_outflows = compute_node_outflows(
                        node=node,
                        flows=next_flows,  # all incoming flows for this node have already been updated at this point
                        node_splits=splits[node.id],
                    )
                    next_flows[out.id] = next_node_outflows[out.id]

                elif isinstance(out, Offramp):
                    # fetch the node downstream of the offramp and the connected destination
                    # in order to identify the correct downstream density boundary condition
                    if out.destination_node_id is None:
                        raise ValueError(
                            f"Offramp {out.id} does not have a well-defined destination node."
                        )
                    offramp_downstream_node = network.get_node(out.destination_node_id)
                    if offramp_downstream_node is None:
                        raise ValueError(
                            f"Offramp {out.id} has invalid destination node id {out.destination_node_id}."
                        )
                    if len(offramp_downstream_node.outgoing) != 1 or not isinstance(
                        offramp_downstream_node.outgoing[0], Destination
                    ):
                        raise ValueError(
                            f"Offramp {out.id} is not connected to a single destination downstream."
                        )

                    destination = offramp_downstream_node.outgoing[0]
                    destination_density_bc = density_boundary_conditions[destination.id]

                    # previous-step flows are used for the computation of the next-step offramp
                    # flow and queue since the offramp keeps its own flow state and queue with
                    # store-and-forward dynamics
                    next_outflow, next_queue = self._compute_offramp_outflows(
                        params=params,
                        offramp=out,
                        node_outflows=node_outflows,
                        offramp_queues=offramp_queues,
                        density_boundary_condition=destination_density_bc,
                        dt=dt,
                    )

                    # set the offramp flow and the queue on the offramp (part of store-and-forward link)
                    # flow for the connected destination is not set => equal to the offramp flow
                    next_flows[out.id] = next_outflow
                    next_offramp_queues[out.id] = next_queue

                # ! 4) update the outgoing motorway links connected to this node (including all cells)
                elif isinstance(out, MotorwayLink):
                    # check if any onramps are connected to the upstream node of the motorway link
                    # (= currently considered node) and if so, compute the corresponding inflows
                    # to take the merging effect into account in the speed update equations
                    upstream_node = node
                    node_upstream_onramp_inflows: casadi.SX | None = None
                    if any(isinstance(inc, Onramp) for inc in upstream_node.incoming):
                        for inc in upstream_node.incoming:
                            if isinstance(inc, Onramp):
                                if node_upstream_onramp_inflows is None:
                                    node_upstream_onramp_inflows = flows[inc.id]
                                else:
                                    node_upstream_onramp_inflows += flows[inc.id]

                    # get the downstream node of the motorway and compute the virtual
                    # downstream density of the corresponding destination node as a
                    # boundary condition for the last cell of the outgoing link from
                    # the current node
                    destination_node = (
                        network.get_node(id=out.destination_node_id)
                        if out.destination_node_id is not None
                        else None
                    )
                    if destination_node is None:
                        raise ValueError(
                            f"Motorway link {out.id} does not have a valid destination node defined."
                        )

                    node_downstream_virtual_downstream_density, _, _ = (
                        self._compute_virtual_downstream_density(
                            node=destination_node,
                            params=params,
                            densities=densities,
                            density_boundary_conditions=density_boundary_conditions,
                        )
                    )

                    next_densities_list, next_speeds_list, next_flows_list = (
                        self._compute_motorway_link_outflows(
                            params=params,
                            link=out,
                            node=node,
                            node_outflows=node_outflows,
                            node_downstream_density=node_downstream_virtual_downstream_density,
                            node_upstream_speed=node_upstream_speed,
                            node_upstream_onramp_inflows=node_upstream_onramp_inflows,
                            flows=flows,
                            densities=densities,
                            speeds=speeds,
                            dt=dt,
                        )
                    )

                    next_densities[out.id] = casadi.SX(next_densities_list)
                    next_speeds[out.id] = casadi.SX(next_speeds_list)
                    next_flows[out.id] = casadi.SX(next_flows_list)

                else:
                    raise TypeError(f"Unknown outgoing link type {type(out)}")

        # combine the network dictionary values for the next step into a single state vector
        x_next, _, _, _, _, _, _, _, _ = network.network_dict_to_state_vec(
            flow_dict=next_flows,
            density_dict=next_densities,
            speed_dict=next_speeds,
            origin_queue_dict=next_origin_queues,
            onramp_queue_dict=next_onramp_queues,
            offramp_queue_dict=next_offramp_queues,
        )

        # wrap the state update in a nonlinear casadi function
        return casadi.Function("metanet_network_step", [sym_params, x, d], [x_next])
