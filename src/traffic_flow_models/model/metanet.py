import numpy as np
import casadi
import warnings
from typing import Callable, Tuple, TypedDict, cast, Union
from numpy.typing import NDArray


from traffic_flow_models.network.motorway_link import MotorwayLink
from traffic_flow_models.network.onramp import Onramp
from traffic_flow_models.network.origin import Origin
from traffic_flow_models.network.offramp import Offramp
from traffic_flow_models.network.destination import Destination
from traffic_flow_models.network.network import Network
from traffic_flow_models.network.node import Node
from traffic_flow_models.network.cell import Cell
from .helpers import (
    store_and_forward_update,
)


class METANETParams(TypedDict):
    tau: float
    nu: float
    kappa: float
    delta: float
    phi: float
    alpha: float | dict[str, float]


class METANETSymbolicParams(TypedDict):
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

    # ! Fundamental diagram helper functions
    # region
    def critical_density(
        self,
        params: METANETParams | METANETSymbolicParams,
        link_id: str,
        lane_capacity: float,
        free_flow_speed: float,
    ) -> float:
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
            lane_capacity: Lane capacity (vehicles per time).
            free_flow_speed: Free-flow speed (length per time).

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
        return lane_capacity / (free_flow_speed * (casadi.exp(-1 / (alpha)) if isinstance(alpha, casadi.SX) else np.exp(-1 / (alpha))))  # type: ignore

    def backward_wave_speed(
        self,
        params: METANETParams | METANETSymbolicParams,
        link_id: str,
        capacity: float,
        lane_capacity: float,
        jam_density: float,
        free_flow_speed: float,
    ) -> float:
        """
        Return the backward (congestion) wave speed for given fundamental parameters.

        The backward wave speed is computed as capacity / (jam_density - rho_crit)
        where rho_crit is the critical density computed from lane_capacity and
        free_flow_speed. This speed describes how congestion propagates upstream
        (length per time).

        Args:
            capacity: Cell capacity (vehicles per time).
            lane_capacity: Capacity per lane (vehicles per time).
            jam_density: Jam density (vehicles per length per lane).
            free_flow_speed: Free-flow speed (length per time).

        Returns:
            Backward wave speed (length per time).

        Raises:
            ValueError: If jam_density is less than or equal to the critical density.
        """

        rho_crit = self.critical_density(
            params=params,
            link_id=link_id,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
        )

        return capacity / (jam_density - rho_crit)

    def stationary_velocity(
        self,
        params: METANETParams | METANETSymbolicParams,
        link_id: str,
        lane_capacity: float,
        free_flow_speed: float,
        density: float | casadi.SX,
    ) -> float:
        """Compute the stationary (equilibrium) velocity for a cell.

        The stationary velocity is the speed that the traffic on the cell would
        adopt in the absence of dynamics, given the current density. METANET
        uses an exponential functional form parameterized by ``alpha`` and the
        cell's free-flow speed (fundamental diagram).

        Args:
            lane_capacity (float): Capacity per lane used to compute the
                critical density.
            free_flow_speed (float): Free-flow speed for the link.
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
            * (
                density
                / self.critical_density(
                    params=params,
                    link_id=link_id,
                    lane_capacity=lane_capacity,
                    free_flow_speed=free_flow_speed,
                )
            )
            ** alpha
        )

        return (
            free_flow_speed * casadi.exp(exponent)
            if isinstance(density, casadi.SX)
            else free_flow_speed * np.exp(exponent)
        )

    # endregion

    # ! Symbolic model parameter handling (validation, packing, unpacking)
    # region
    def validate_model_params(self, model_params: METANETParams) -> None:
        """Validate a METANET model parameter dictionary.

        Ensures required keys are present and their types are correct. All
        scalar parameters (`tau`, `nu`, `kappa`, `delta`, `phi`) must be
        numeric (int or float). The `alpha` parameter may be either a scalar
        (int or float) or a dictionary mapping link ids to numeric values.

        Args:
            model_params: Dictionary containing METANET model parameters.

        Raises:
            ValueError: If a required parameter is missing or has an invalid type.
        """

        required_params = ["tau", "nu", "kappa", "delta", "phi", "alpha"]
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
        network: Network,
        model_params: METANETParams,
    ) -> NDArray[np.float64]:
        """Convert a METANET parameter dictionary to a numerical vector.

        The returned vector layout is identical to that used by the
        symbolic parameter vector: the five scalar parameters in the order
        `tau, nu, kappa, delta, phi` followed by the per-link `alpha`
        parameters for all motorway links, onramps and offramps in the same
        ordering used throughout the model formulation.

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
        network: Network,
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
        tau: float | casadi.SX = model_params_vec[0]
        nu: float | casadi.SX = model_params_vec[1]
        kappa: float | casadi.SX = model_params_vec[2]
        delta: float | casadi.SX = model_params_vec[3]
        phi: float | casadi.SX = model_params_vec[4]

        # extract the alpha parameters for each link
        alpha_vector = model_params_vec[5:]
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
            "tau": tau,
            "nu": nu,
            "kappa": kappa,
            "delta": delta,
            "phi": phi,
            "alpha": alpha_dict,
        }

    def set_up_symbolic_model_params(
        self,
        network: Network,
    ) -> casadi.SX:
        """Create a CasADi symbolic vector for METANET model parameters.

        The created `casadi.SX` vector contains one entry for each scalar
        parameter (`tau, nu, kappa, delta, phi`) followed by one `alpha`
        entry per motorway link, onramp and offramp in the network. The
        ordering of per-link `alpha` entries matches the unpacking performed
        in `model_params_vec_to_dict`.

        Args:
            network: Network instance used to count links and determine the
                length of the symbolic parameter vector.

        Returns:
            A `casadi.SX` symbolic column vector of shape `(num_links + 5, 1)`.
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
        # 5 more entries for the parameters tau, nu, kappa, delta, phi
        model_params_sym = casadi.SX.sym("metanet_params", num_links + 5, 1)  # type: ignore
        return model_params_sym

    # endregion

    # ! Network update helper functions
    # region
    def _compute_virtual_downstream_density(
        self,
        node: Node,
        params: METANETSymbolicParams,
        densities: dict[str, casadi.SX],
        boundary_conditions: dict[str, casadi.SX],
    ) -> Tuple[casadi.SX, Union[float, None], Union[float, None]]:
        """Determine a node's virtual downstream density for METANET updates.

        The virtual downstream density is used when computing boundary and
        anticipation terms at nodes with multiple outgoing links. For each
        outgoing link the method selects an appropriate downstream density
        representation:
        - For a `MotorwayLink`: the density of its first cell is used.
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
            boundary_conditions (dict[str, casadi.SX]): Mapping of link or
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
        node_downstream_density = None
        node_downstream_jam_density = None
        node_downstream_backward_wave_speed = None

        # determine the virtual downstream density of the node based on the outgoing links = q_m,N_m+1(k)
        if len(node.outgoing) > 1:
            out_densities: list[casadi.SX] = []
            out_jam_densities: list[float] = []
            out_backward_wave_speeds: list[float] = []
            for out_link in node.outgoing:
                if isinstance(out_link, MotorwayLink):
                    # motorway link: use the density of the first cell as downstream density
                    out_densities.append(densities[out_link.id][0])
                    out_jam_densities.append(out_link.rho_jam)
                    out_backward_wave_speeds.append(
                        self.backward_wave_speed(
                            params=params,
                            link_id=out_link.id,
                            capacity=out_link.lane_capacity * out_link.lanes,
                            lane_capacity=out_link.lane_capacity,
                            jam_density=out_link.rho_jam,
                            free_flow_speed=out_link.vf,
                        )
                    )

                elif isinstance(out_link, Offramp):
                    # offramp link: the store-and-forward model does not model density / speed on offramps
                    # -> directly use the boundary condition of the connected destination as downstream density
                    if out_link.destination is None:
                        raise ValueError(
                            f"Offramp {out_link.id} does not have a destination defined."
                        )

                    out_densities.append(boundary_conditions[out_link.destination.id])
                    out_jam_densities.append(np.inf)
                    out_backward_wave_speeds.append(np.inf)

                elif isinstance(out_link, Destination):
                    # destination link: density is provided as boundary condition
                    out_densities.append(boundary_conditions[out_link.id])
                    out_jam_densities.append(np.inf)
                    out_backward_wave_speeds.append(np.inf)

                else:
                    raise TypeError(f"Unknown outgoing link type {type(out_link)}")

            # combine the different downstream densities (e.g., weighted average)
            numer = casadi.vertcat(*[d**2 for d in out_densities])
            denom = casadi.vertcat(*out_densities)
            node_downstream_density = casadi.sum(numer) / casadi.sum(denom)

            # choose the minimum downstream jam density / backward wave speed among the outgoing links
            # set the value to None, if no finite value is available
            finite_jam_densities = [d for d in out_jam_densities if d < np.inf]
            node_downstream_jam_density = (
                min(finite_jam_densities) if len(finite_jam_densities) > 0 else None
            )
            finite_backward_wave_speeds = [
                s for s in out_backward_wave_speeds if s < np.inf
            ]
            node_downstream_backward_wave_speed = (
                min(finite_backward_wave_speeds)
                if len(finite_backward_wave_speeds) > 0
                else None
            )

        # single outgoing link: directly use its downstream density
        elif len(node.outgoing) == 1:
            out_link = node.outgoing[0]

            if isinstance(out_link, MotorwayLink):
                # motorway link: use the density of the first cell as downstream density
                node_downstream_density = densities[out_link.id][0]
                node_downstream_jam_density = out_link.rho_jam
                node_downstream_backward_wave_speed = self.backward_wave_speed(
                    params=params,
                    link_id=out_link.id,
                    capacity=out_link.lane_capacity * out_link.lanes,
                    lane_capacity=out_link.lane_capacity,
                    jam_density=out_link.rho_jam,
                    free_flow_speed=out_link.vf,
                )

            elif isinstance(out_link, Offramp):
                # offramp link: the store-and-forward model does not model density / speed on offramps
                # -> directly use the boundary condition of the connected destination as downstream density
                if out_link.destination is None:
                    raise ValueError(
                        f"Offramp {out_link.id} does not have a destination defined."
                    )

                node_downstream_density = boundary_conditions[out_link.destination.id]
                node_downstream_jam_density = None  # no downstream jam density defined -> handling on calling level required
                node_downstream_backward_wave_speed = None  # no downstream backward wave speed defined -> handling on calling level required
            elif isinstance(out_link, Destination):
                # destination link: density is provided as boundary condition
                node_downstream_density = boundary_conditions[out_link.id]
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

    def _compute_node_outflows_upstream_speed(
        self,
        node: Node,
        flows: dict[str, casadi.SX],
        speeds: dict[str, casadi.SX],
        splits: dict[str, dict[str, casadi.SX]],
    ) -> Tuple[dict[str, casadi.SX], casadi.SX]:
        """Compute the node outflows and virtual upstream speed for METANET updates.

        The method computes the total available flow into the node by summing
        the last cell flows of all incoming motorway links as well as the
        flows from origins and onramps. Based on the total available flow and
        the provided split ratios, the method computes the outflows for each
        outgoing link. Additionally, the method computes the virtual upstream speed
        used in the speed update equations for outgoing motorway links.

        Args:
            node (Node): Network node for which to compute outflows and
                upstream speed.
            flows (dict[str, casadi.SX]): Mapping link id -> vector of cell
                flows (CasADi SX) for motorway links, origins, onramps and
                offramps.
            speeds (dict[str, casadi.SX]): Mapping link id -> vector of cell
                speeds (CasADi SX) for motorway links.
            splits (dict[str, dict[str, casadi.SX]]): Mapping of node id to
                mapping of outgoing link id to split ratio (CasADi SX).

        Returns:
            Tuple[dict[str, casadi.SX], casadi.SX]: A tuple containing:
                - A dictionary mapping outgoing link id to computed outflow
                  (CasADi SX).
                - The virtual upstream speed (CasADi SX) used in speed updates.
        """
        Qn = casadi.SX(0)
        for inc in node.incoming:
            if isinstance(inc, MotorwayLink):
                # motorway link: use the last cell flow as upstream flow
                Qn += flows[inc.id][-1]
            elif isinstance(inc, Origin):
                # origin link: use the flow entering the origin (from state vector) - demand -> flow update separate
                Qn += flows[inc.id][0]
            elif isinstance(inc, Onramp):
                # onramp link: use the flow entering the onramp (from state vector) - demand -> flow update separate
                Qn += flows[inc.id][0]
            else:
                raise TypeError("Unknown incoming link type")

        # compute the node outflows based on the total available flow and the splits
        # node outflows = q_m,0(k) - dictionary with one value per outgoing edge
        node_outflows = {}
        node_splits = splits[node.id]

        if node_splits is None:
            raise ValueError(f"No split ratios defined for node {node.id}")

        for out in node.outgoing:
            out_split = node_splits[out.id]
            if out_split is None:
                raise ValueError(
                    f"No split ratio defined for outgoing link {out.id} (type: {type(out)}) at node {node.id}"
                )

            # re-normalize turning rates to make sure that they properly sum up to 1
            total_splits = casadi.sum(casadi.vertcat(*list(node_splits.values())))
            node_outflows[out.id] = Qn * out_split / casadi.fmax(total_splits, 1.0)

        # determine the virtual upstream speed of the node (for outgoing motorway links) = v_m,0(k)
        # since a speed parameter is required, only incoming motorway links are considered
        # if all incoming links are origins, assume free flow conditions upstream
        if all(not isinstance(inc, MotorwayLink) for inc in node.incoming):
            # if any onramps are connected to the node, use the minimum free-flow speed of those onramps
            if any(isinstance(inc, Onramp) for inc in node.incoming):
                node_upstream_speed = casadi.SX(
                    min(inc.vf for inc in node.incoming if isinstance(inc, Onramp))
                )
            else:
                node_upstream_speed = casadi.SX(
                    min(
                        out.vf for out in node.outgoing if isinstance(out, MotorwayLink)
                    )
                )
        else:
            nom_terms = []
            denom_terms = []
            for inc in node.incoming:
                if isinstance(inc, MotorwayLink):
                    # motorway link: use the last cell speed and flow for upstream speed
                    nom_terms.append(speeds[inc.id][-1] * flows[inc.id][-1])
                    denom_terms.append(flows[inc.id][-1])

            node_upstream_speed = casadi.sum(casadi.vertcat(*nom_terms)) / casadi.sum(
                casadi.vertcat(*denom_terms)
            )

        return node_outflows, node_upstream_speed

    def _compute_offramp_outflows(
        self,
        params: METANETSymbolicParams,
        offramp: Offramp,
        node_outflows: dict[str, casadi.SX],
        offramp_queues: dict[str, casadi.SX],
        boundary_conditions: dict[str, casadi.SX],
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
            boundary_conditions (dict[str, casadi.SX]): Mapping of destination
                id to boundary density (CasADi SX) used as downstream density.
            dt (float): Simulation timestep.

        Returns:
            Tuple[casadi.SX, casadi.SX]: Tuple `(next_outflow, next_queue)`
            where `next_outflow` is the offramp outflow (vehicles/time) into
            the connected destination, and `next_queue` is the updated queue
            length on the offramp (vehicles).

        Raises:
            ValueError: If the `offramp` does not have a `destination` defined.
        """
        if offramp.destination is None:
            raise ValueError(
                f"Offramp {offramp.id} does not have a destination defined."
            )

        mainline_outflow = node_outflows[
            offramp.id
        ]  # desired offramp flow based on splits = flow onto offramp (queue on offramp itself)
        offramp_demand = mainline_outflow + offramp_queues[offramp.id] / dt

        # update the offramp flow and queue based on the store-and-forward model
        next_outflow, next_queue = store_and_forward_update(
            capacity=offramp.Qc,
            jam_density=offramp.rho_jam,
            backward_wave_speed=self.backward_wave_speed(
                params=params,
                link_id=offramp.id,
                capacity=offramp.Qc,
                lane_capacity=offramp.Qc_lane,
                jam_density=offramp.rho_jam,
                free_flow_speed=offramp.vf,
            ),
            density=boundary_conditions[offramp.destination.id],
            demand=offramp_demand,
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
            next-step densities, speeds and outflows respectively.

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
                    node_outflows[link.id]
                    if cell.upstream is None
                    else link_flows[i - 1]
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
                    lane_capacity=link.lane_capacity,
                    free_flow_speed=link.vf,
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
                * self.critical_density(
                    params=params,
                    link_id=link.id,
                    lane_capacity=link.lane_capacity,
                    free_flow_speed=link.vf,
                )
            )

        # ensure that the speed values remain non-negative
        speed = casadi.fmax(speed, 0)

        # compute the flow update of the cell based on the speed and density
        flow = density * speed * link.lanes

        return density, speed, flow

    # endregion

    def network_update_function(
        self,
        network: Network,
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
        contains origin demands, onramp demands, split ratios and boundary
        condition entries in the ordering expected by the network helpers.

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
        # disturbances: origin_demands, onramp_demands, offramp_split_ratios
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
        d = casadi.SX.sym("d", num_origins + num_onramps + num_splits + num_destinations, 1)  # type: ignore

        # split up the state and disturbance vectors to obtain a dictionary for
        # efficient access of the relevant quantities during the state update
        flows, densities, speeds, origin_queues, onramp_queues, offramp_queues = (
            network.state_vec_to_network_dict(x=x)
        )
        origin_demands, onramp_demands, splits, boundary_conditions = (
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
        onramp_demands = {k: casadi.SX(v) for k, v in onramp_demands.items()}
        splits = {
            k: {kk: casadi.SX(vv) for kk, vv in v.items()} for k, v in splits.items()
        }
        boundary_conditions = {k: casadi.SX(v) for k, v in boundary_conditions.items()}

        # initialize next-step state dictionaries
        next_flows: dict[str, casadi.SX] = {}
        next_densities: dict[str, casadi.SX] = {}
        next_speeds: dict[str, casadi.SX] = {}
        next_origin_queues: dict[str, casadi.SX] = {}
        next_onramp_queues: dict[str, casadi.SX] = {}
        next_offramp_queues: dict[str, casadi.SX] = {}

        # formulate the individual update equations for each node and update the overall system equation and the next step state
        # iterate through all nodes and update the corrresponding quantities of incoming and outgoing links
        for node in network.list_nodes():
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
                    boundary_conditions=boundary_conditions,
                )

                if isinstance(inc, Origin):
                    next_inflow, next_queue = store_and_forward_update(
                        capacity=casadi.inf,
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
                    # TODO: include possibility here for ramp metering controller (e.g. through ramp metering rate input)
                    next_inflow, next_queue = store_and_forward_update(
                        capacity=inc.Qc,
                        jam_density=inc.rho_jam,
                        backward_wave_speed=self.backward_wave_speed(
                            params=params,
                            link_id=inc.id,
                            capacity=inc.Qc,
                            lane_capacity=inc.Qc_lane,
                            jam_density=inc.rho_jam,
                            free_flow_speed=inc.vf,
                        ),
                        density=node_virtual_downstream_density,
                        demand=onramp_demands[inc.id],
                        queue=onramp_queues[inc.id],
                        dt=dt,
                    )

                    next_flows[inc.id] = next_inflow
                    next_onramp_queues[inc.id] = next_queue

            # ! 2) compute the required boundary conditions based on the combined incoming / outgoing quantities
            # -> this includes the upstream speed, downstream density, etc. that are required by the model udpate
            # sum up the last cell flows of all incoming links, onramps and origins
            node_outflows, node_upstream_speed = (
                self._compute_node_outflows_upstream_speed(
                    node=node,
                    flows=flows,
                    speeds=speeds,
                    splits=splits,
                )
            )

            for out in node.outgoing:
                # ! 3) update the offramp flows (& density/speed) for destinations connected to this node
                if isinstance(out, Destination):
                    # destinations are assumed to consume all incoming flow
                    # (only impact the mainstream through the density boundary condition)
                    next_flows[out.id] = node_outflows[out.id]
                elif isinstance(out, Offramp):
                    next_outflow, next_queue = self._compute_offramp_outflows(
                        params=params,
                        offramp=out,
                        node_outflows=node_outflows,
                        offramp_queues=offramp_queues,
                        boundary_conditions=boundary_conditions,
                        dt=dt,
                    )

                    # set the offramp flow and the queue on the offramp (part of store-and-forward link)
                    next_flows[out.id] = next_outflow
                    next_offramp_queues[out.id] = next_queue

                    # the flow of the connected destination is equal to the offramp outflow
                    if out.destination is not None:
                        next_flows[out.destination.id] = next_outflow
                    else:
                        raise ValueError(
                            f"Offramp {out.id} does not have a destination defined."
                        )

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
                            boundary_conditions=boundary_conditions,
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
