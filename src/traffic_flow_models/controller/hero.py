import casadi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traffic_flow_models.network.onramp import Onramp


class HeroController:
    # TODO: UDPATE DOCSTRING FOR GIVEN IMPLEMENTATION
    # TODO: ALSO UPDATE TEST SUITE ONCE IMPLEMENTATION ITSELF IS WORKING
    """Simple HERO-style coordinated ramp metering controller.

    This is a pragmatic, lightweight implementation intended for unit tests
    and simple simulations. It monitors the onramp queue relative to the
    onramp's jam-storage capacity (``rho_jam * length * lanes``) and
    activates coordination when the queue exceeds ``activation_threshold``.

    When activated the controller will:
    - check if any downstream onramp is already active; if so, it does nothing
    - otherwise set the local onramp to ``hero_master`` and mark neighbours
      as ``hero_slave`` and assign a ``master_reference`` attribute on slaves
    - master and slaves compute metering rates using a simple capacity-based
      cap (50% of onramp capacity) as a safe default.

    The class is designed to be attached to each involved onramp as a
    per-onramp controller instance; coordination is achieved via the
    mutable attributes placed on the Onramp objects (``control_status`` and
    ``master_reference``) so the same controller object need not be attached
    to every onramp.
    """

    def __init__(
        self,
        onramp: "Onramp",
        measurement_link_id: str,
        measurement_cell_idx: int,
        gain: float,
        density_setpoint: float,
        activation_threshold: float = 0.6,
        deactivation_threshold: float | None = None,
        nonconservative_prediction_param: float = 1.2,
    ) -> None:
        """Create a HERO-style coordinated ramp metering controller instance.

        Args:
            onramp: Onramp object to which the controller is attached.
            measurement_link_id: ID of the link where the density measurement is taken for feedback
            measurement_cell_idx: Index of the cell on the measurement link where the density is measured
            gain: ALINEA controller gain parameter (positive scalar).
            density_setpoint: Desired downstream density setpoint (vehicles per length per lane).
            activation_threshold: Queue length threshold for activating HERO coordination
                (relative to jam storage capacity; in (0,1)).
            deactivation_threshold: Queue length threshold for deactivating HERO coordination
                (relative to jam storage capacity; in (0,1)). If None, defaults to 50% of the activation threshold.
            nonconservative_prediction_param: Parameter in [0,1] to adjust the non-conservative
                prediction of the queue length evolution in the regulation towards N_min in slave mode.
                Value is required to be >= 1 to ensure that the predicted queue length does not underestimate
                the actual queue length.
        """
        if activation_threshold <= 0.0 or activation_threshold >= 1.0:
            raise ValueError("activation_threshold must be in (0,1)")

        if deactivation_threshold is None:
            deactivation_threshold = activation_threshold * 0.5

        if not (0.0 < deactivation_threshold < activation_threshold):
            raise ValueError(
                "deactivation_threshold must be >0 and < activation_threshold"
            )

        self.onramp = onramp

        # HERO thresholds to determine when to activate / deactivate
        # coordination / additional onramps for capacity
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.nonconservative_prediction_param = nonconservative_prediction_param

        # ALINEA reference cell
        self.measurement_link_id: str = measurement_link_id
        self.measurement_cell: int = measurement_cell_idx

        # ALINEA parameters for use in master and base mode regulation
        self.gain: float = gain
        self.density_setpoint: float = density_setpoint

    def alinea_regulate_flow(
        self, measured_density: casadi.SX, previous_flow: casadi.SX
    ) -> casadi.SX:
        """Compute the regulated onramp flow using the ALINEA feedback law.

        Args:
            measured_density: Current measured density for feedback (Casadi SX).
            previous_flow: Previous onramp flow (Casadi SX).
        """
        flow_adjustment = self.gain * (self.density_setpoint - measured_density)
        regulated_flow = previous_flow + flow_adjustment
        return casadi.fmax(regulated_flow, casadi.SX(0.0))  # ensure non-negative flow

    def slave_regulate_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        dt: float,
    ) -> casadi.SX:
        """Compute the regulated onramp flow for a slave onramp according to HERO-like rules.

        Args:
            onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).
            dt: Simulation time step (for queue evolution prediction in slave mode).
        """

        # sum up the current and maximum queue lengths of all involved coordinated onramps
        # iterate downstream until reaching master onramp, upstream until last slave
        sum_N_current = casadi.SX(0.0)
        sum_N_max = casadi.SX(0.0)

        for down in self.onramp.downstream_onramps:
            if down.control_status in ["hero_master", "hero_slave"]:
                sum_N_current += onramp_queues[down.id]
                sum_N_max += down.max_queue_length

                # if a master ramp is reached, break the loop (no further onramps should be considered)
                if down.control_status == "hero_master":
                    break
            else:
                break

        for up in self.onramp.upstream_onramps:
            if up.control_status == "hero_slave":
                sum_N_current += onramp_queues[up.id]
                sum_N_max += up.max_queue_length
            else:
                break

        # compute the minimum queue length towards which the current onramp should be regulated
        N_min = (self.onramp.max_queue_length * sum_N_current) / (sum_N_max)

        # precompute the flow allowed by ALINEA
        alinea_flow = self.alinea_regulate_flow(
            measured_density=densities[self.measurement_link_id][self.measurement_cell],
            previous_flow=flows[self.onramp.id][0],
        )

        # if the current queue length is above the minimum, regulate according to ALINEA
        if onramp_queues[self.onramp.id] > N_min:
            return alinea_flow

        # otherwise, ensure that the queue goes towards the minimum queue length
        # if ALINEA flow would be even smaller, choose this as the setpoint to avoid
        # oscillations around the minimum queue length
        return casadi.fmin(
            alinea_flow,
            self.nonconservative_prediction_param * flows[self.onramp.id][0]
            - (N_min - onramp_queues[self.onramp.id]) / dt,
        )

    def compute_regulated_flow(
        self,
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        dt: float,
    ) -> casadi.SX:
        """Compute regulated flow for this onramp according to HERO-like rules.

        Updates of related onramps into slave mode will follow automatically
        during the next iteration (at the latest) when their regulation function
        is called and the current ramp is recognized to be in master mode

        Args:
            onramp_queues: Dictionary mapping on-ramp IDs to their current queue values (Casadi SX).
            flows: Dictionary mapping link IDs to their current flow values (Casadi SX).
            densities: Dictionary mapping link IDs to their current density values (Casadi SX).
            dt: Simulation time step (for queue evolution prediction in slave mode).
        """
        # obtain previous on-ramp flow (one-element array)
        previous_flow: casadi.SX = flows[self.onramp.id][0]

        # get the current density measurement for feedback
        measured_density = densities[self.measurement_link_id][self.measurement_cell]

        # get the current queue length on the onramp
        queue_length: casadi.SX = onramp_queues[self.onramp.id]

        # check if the current onramp is already in master-mode and if that should be deactivated
        if self.onramp.control_status == "hero_master":
            if (
                queue_length
                < self.deactivation_threshold * self.onramp.max_queue_length
            ):
                # deactivate coordinator and clear slave references
                self.onramp.control_status = "unset"
                for o in self.onramp.upstream_onramps:
                    if o.control_status == "hero_slave":
                        o.control_status = "unset"
            else:
                # remain master -> use regular ALINEA regulation
                pass

            # for both unset and master case, use regular ALINEA
            return self.alinea_regulate_flow(
                measured_density=measured_density, previous_flow=previous_flow
            )

        # check if any downstream onramp is active in master-mode
        if any(
            o.control_status == "hero_master" for o in self.onramp.downstream_onramps
        ):
            # check if the onramp immediately downstream of the current one is
            # active as master or slave with a queue length above its threshold
            following_master = (
                self.onramp.downstream_onramps[0].control_status == "hero_master"
            )
            following_slave = (
                self.onramp.downstream_onramps[0].control_status == "hero_slave"
            )
            following_queue = onramp_queues[self.onramp.downstream_onramps[0].id]
            following_max_queue = (
                self.onramp.downstream_onramps[0].rho_jam
                * self.onramp.downstream_onramps[0].length
                * self.onramp.downstream_onramps[0].lanes
            )
            following_activation_threshold = (
                self.activation_threshold * following_max_queue
            )
            following_deactivation_threshold = (
                self.deactivation_threshold * following_max_queue
            )

            # check if we should activate the current onramp as a slave based
            # on the downstream onramp setting and queue length
            if following_master or (
                following_slave and following_queue > following_activation_threshold
            ):
                self.onramp.control_status = "hero_slave"
                return self.slave_regulate_flow(
                    onramp_queues=onramp_queues, flows=flows, densities=densities, dt=dt
                )

            # if the following ramp does not qualify for propagation currently, we do not
            # update the status of the current ramp except from master activation (may be unset or slave)
            elif following_slave and following_queue > following_deactivation_threshold:
                if (
                    self.onramp.control_status == "unset"
                    and queue_length
                    > self.activation_threshold * self.onramp.max_queue_length
                ):
                    self.onramp.control_status = "hero_master"
                    return self.alinea_regulate_flow(
                        measured_density=measured_density, previous_flow=previous_flow
                    )  # use regular ALINEA regulation
                else:
                    if self.onramp.control_status == "unset":
                        return self.alinea_regulate_flow(
                            measured_density=measured_density,
                            previous_flow=previous_flow,
                        )
                    else:
                        return self.slave_regulate_flow(
                            onramp_queues=onramp_queues,
                            flows=flows,
                            densities=densities,
                            dt=dt,
                        )

            # there is no reason for current ramp to be in slave mode
            # deactivate slave mode if currently active and, if applicable, activate master mode
            else:
                if (
                    queue_length
                    > self.activation_threshold * self.onramp.max_queue_length
                ):
                    self.onramp.control_status = "hero_master"
                else:
                    self.onramp.control_status = "unset"

                # independent of whether in master or unset mode, use regular ALINEA
                return self.alinea_regulate_flow(
                    measured_density=measured_density, previous_flow=previous_flow
                )

        else:
            # no relevant downstream onramp is active
            # -> check if the current one should be activated in master mode
            # (-> maximum considered ramp range limits propagation of HERO)
            if queue_length > self.activation_threshold * self.onramp.max_queue_length:
                self.onramp.control_status = "hero_master"
            else:
                # make sure the current status is unset
                self.onramp.control_status = "unset"

            # for both unset and master case, use regular ALINEA
            return self.alinea_regulate_flow(
                measured_density=measured_density, previous_flow=previous_flow
            )
