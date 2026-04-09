import casadi
import numpy as np

from traffic_flow_models import (
    FlowController,
    AlineaController,
    Onramp,
    CustomController,
    HeroController,
)
from traffic_flow_models.model.helpers import store_and_forward_update


def _eval(exprs):
    """Evaluate CasADi expressions and return list of floats.

    This helper handles different return types from `casadi.Function()` such as
    `DM`, `SX`-backed results, tuples/lists, or dicts and uses NumPy to
    produce stable numeric values (avoiding direct `.full()` calls on plain
    Python/numpy objects).
    """
    out = casadi.Function("f", [], exprs)()
    if isinstance(out, dict):
        vals = list(out.values())
    elif isinstance(out, (list, tuple)):
        vals = list(out)
    else:
        vals = [out]

    res = []
    for v in vals:
        if hasattr(v, "full"):
            arr = np.array(v).flatten()
        else:
            arr = np.asarray(v)
        res.append(float(np.squeeze(arr)))
    return res


def test_flowcontroller_attributes_and_compute():
    onr = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=140,
        id="r1",
    )
    c = FlowController(onramp=onr, flow=900.0)
    assert c.onramp is onr
    assert _eval([c.flow])[0] == 900.0

    c2 = FlowController(onramp=onr, flow=750.0)
    flows = {"r1": casadi.SX([100.0])}
    densities = {"m1": casadi.SX([10.0])}
    onramp_queues = {"r1": casadi.SX([5.0])}
    regulated = c2.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert _eval([regulated])[0] == 750.0


def test_store_and_forward_respects_metering_rate():
    capacity = 100.0
    jam_density = 200.0
    backward_wave_speed = 20.0
    density = casadi.SX(10.0)
    demand = casadi.SX(50.0)
    queue = casadi.SX(10.0)
    dt = 1.0
    metering_rate = casadi.SX(30.0)

    inflow, updated_queue = store_and_forward_update(
        capacity,
        jam_density,
        backward_wave_speed,
        density,
        demand,
        queue,
        dt,
        metering_rate,
    )
    inflow_val, queue_val = _eval([inflow, updated_queue])
    assert inflow_val == 30.0
    assert queue_val == 10.0 + (50.0 - 30.0)


def test_store_and_forward_without_metering_rate():
    capacity = 100.0
    jam_density = 200.0
    backward_wave_speed = 20.0
    density = casadi.SX(10.0)
    demand = casadi.SX(50.0)
    queue = casadi.SX(10.0)
    dt = 1.0

    inflow, updated_queue = store_and_forward_update(
        capacity, jam_density, backward_wave_speed, density, demand, queue, dt
    )
    inflow_val, queue_val = _eval([inflow, updated_queue])
    # without metering, qin_demand = demand + queue/dt = 60 -> inflow = min(capacity,60)=60
    assert inflow_val == 60.0
    assert queue_val == 0.0


def test_onramp_accepts_controllers_and_compute():
    onramp_fc = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=140,
        id="r1",
    )
    fc = FlowController(onramp=onramp_fc, flow=500.0)
    onramp_fc.controller = fc

    onramp_al = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=140,
        id="r2",
    )
    ar = AlineaController(
        onramp=onramp_al,
        measurement_link_id="m1",
        measurement_cell_idx=0,
        gain=2.0,
        density_setpoint=10.0,
    )
    onramp_al.controller = ar

    assert isinstance(onramp_fc.controller, FlowController)
    assert isinstance(onramp_al.controller, AlineaController)

    flows = {"r1": casadi.SX([0.0])}
    densities = {"m1": casadi.SX([0.0])}
    onramp_queues = {"r1": casadi.SX([5.0])}
    regulated = onramp_fc.controller.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert _eval([regulated])[0] == 500.0


def test_alinea_attributes_and_compute():
    onr = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=140,
        id="r1",
    )
    c = AlineaController(
        onramp=onr,
        measurement_link_id="m1",
        measurement_cell_idx=0,
        gain=2.0,
        density_setpoint=10.0,
    )
    assert c.gain == 2.0
    assert c.density_setpoint == 10.0

    flows = {"r1": casadi.SX([100.0])}
    densities = {"m1": casadi.SX([5.0])}
    onramp_queues = {"r1": casadi.SX([5.0])}
    regulated = c.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    val = _eval([regulated])[0]
    prev_val = _eval([flows["r1"]])[0]
    meas_val = _eval([densities["m1"]])[0]
    expected = prev_val + c.gain * (c.density_setpoint - meas_val)
    assert val == expected

    # non-negative behaviour
    c2 = AlineaController(
        onramp=onr,
        measurement_link_id="m1",
        measurement_cell_idx=0,
        gain=1.0,
        density_setpoint=0.0,
    )
    flows = {"r1": casadi.SX([0.0])}
    densities = {"m1": casadi.SX([1000.0])}
    onramp_queues = {"r1": casadi.SX([5.0])}
    regulated2 = c2.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert _eval([regulated2])[0] == 0.0


def test_custom_controller_callable_and_numeric_conversion():
    # controller that uses flows to compute rate (CasADi expression)
    def fn_casadi(
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        _: dict[str, casadi.SX],
    ) -> casadi.SX:
        return flows["r1"][0] * casadi.SX(2.0)

    onr2 = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=140,
        id="r1",
    )
    cc = CustomController(onramp=onr2, controller_fn=fn_casadi)
    flows = {"r1": casadi.SX([10.0])}
    densities = {"m1": casadi.SX([0.0])}
    onramp_queues = {"r1": casadi.SX([5.0])}
    regulated = cc.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert _eval([regulated])[0] == 20.0

    # controller that returns a plain numeric value (should be converted)
    def fn_numeric(
        _: dict[str, casadi.SX], __: dict[str, casadi.SX], ___: dict[str, casadi.SX]
    ) -> float:
        return 333.0

    cc2 = CustomController(onramp=onr2, controller_fn=fn_numeric)  # type: ignore
    regulated2 = cc2.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert _eval([regulated2])[0] == 333.0


def test_custom_controller_with_params():
    # controller that reads a rate from the params dict
    def fn_with_params(
        onramp_queues: dict[str, casadi.SX],
        flows: dict[str, casadi.SX],
        densities: dict[str, casadi.SX],
        params: dict[str, float],
    ) -> casadi.SX:
        return casadi.SX(params.get("rate", 0.0))

    onr3 = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=140,
        id="r1",
    )
    cc = CustomController(
        onramp=onr3, controller_fn=fn_with_params, params={"rate": 777.0}
    )
    flows = {"r1": casadi.SX([10.0])}
    densities = {"m1": casadi.SX([0.0])}
    onramp_queues = {"r1": casadi.SX([5.0])}
    regulated = cc.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert _eval([regulated])[0] == 777.0


def test_hero_activation_and_master_slave_assignment():
    # create three onramps and a simple neighbour relation
    master = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1500,
        free_flow_speed=80,
        jam_density=100,
        id="rm",
    )
    up = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1000,
        free_flow_speed=60,
        jam_density=100,
        id="ru",
    )
    down = Onramp(
        length=0.5,
        lanes=1,
        lane_capacity=1000,
        free_flow_speed=60,
        jam_density=100,
        id="rd",
    )

    # set neighbour lists manually
    master.upstream_onramps = [up]
    master.downstream_onramps = [down]

    # attach HERO controllers to each onramp (distinct instances)
    cm = HeroController(
        onramp=master, activation_threshold=0.5, deactivation_threshold=0.4
    )
    cu = HeroController(onramp=up, activation_threshold=0.5, deactivation_threshold=0.4)
    cd = HeroController(
        onramp=down, activation_threshold=0.5, deactivation_threshold=0.4
    )
    master.controller = cm
    up.controller = cu
    down.controller = cd

    # queues: master exceeds activation threshold (jam_capacity = rho_jam*length*lanes = 50 -> threshold 25)
    flows = {"rm": casadi.SX([100.0]), "ru": casadi.SX([50.0]), "rd": casadi.SX([50.0])}
    densities = {}
    onramp_queues = {
        "rm": casadi.SX([30.0]),
        "ru": casadi.SX([1.0]),
        "rd": casadi.SX([1.0]),
    }

    # perform regulation call for master -> should activate and mark neighbours as slaves
    regulated = master.controller.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert master.control_status == "hero_master"
    assert up.control_status == "hero_slave"
    assert down.control_status == "hero_slave"

    # slaves should report reduced/capped rates when invoked
    reg_u = up.controller.compute_regulated_flow(
        onramp_queues, flows, densities, dt=1.0
    )
    reg_d = down.controller.compute_regulated_flow(
        onramp_queues, flows, densities, dt=1.0
    )
    # ensure values are CasADi expressions and numeric when evaluated
    val_u = _eval([reg_u])[0]
    val_d = _eval([reg_d])[0]
    assert isinstance(val_u, float)
    assert isinstance(val_d, float)

    # if a downstream onramp was already active, activation should do nothing
    # reset statuses
    master.control_status = "unset"
    down.control_status = "hero_master"
    # master queue still high; since a downstream is active, do nothing
    regulated2 = master.controller.compute_regulated_flow(
        onramp_queues=onramp_queues, flows=flows, densities=densities, dt=1.0
    )
    assert master.control_status == "unset"
