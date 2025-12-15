from traffic_flow_models import CTM
from demo.scenarios import (
    mainline_demand_a,
    mainline_demand_b,
    mainline_demand_c,
    onramp_demand_a,
    onramp_demand_b,
    onramp_demand_c,
    setup_network_ab,
    setup_network_c,
    mainline_demand_d,
    onramp_demand_d,
    setup_network_d,
)


if __name__ == "__main__":
    # ! simulation configuration parameters
    scenario = "A"
    alinea_ramp_control = False
    alinea_gain = 5.0
    alinea_setpoint = 2000.0 / 100.0  # = rho_cr = Qc_lane / vf (triangular FD)
    dt = 10.0 / 3600
    duration = 5000.0 / 3600

    # select the appropriate scenario functions
    if scenario == "A":
        mainline_demand = mainline_demand_a
        onramp_demand = onramp_demand_a
        setup_network = setup_network_ab
    elif scenario == "B":
        mainline_demand = mainline_demand_b
        onramp_demand = onramp_demand_b
        setup_network = setup_network_ab
    elif scenario == "C":
        mainline_demand = mainline_demand_c
        onramp_demand = onramp_demand_c
        setup_network = setup_network_c
    elif scenario == "D":
        mainline_demand = mainline_demand_d
        onramp_demand = onramp_demand_d
        setup_network = setup_network_d
    else:
        raise ValueError(f"Scenario {scenario} is not defined.")

    # initialize the CTM model
    ctm = CTM()

    # initialize the network with the correct structure (optionally with ALINEA ramp metering)
    network = setup_network(
        get_critical_density=ctm.critical_density,
        ramp_control=alinea_ramp_control,
        alinea_gain=alinea_gain,
        alinea_setpoint=alinea_setpoint,
    )
    network.plot()

    # run a simulation of the network using the CTM model
    density, flow, speed, input_flow, input_queue, onramp_flow, onramp_queue = (
        network.simulate(
            duration=duration,
            dt=dt,
            model=ctm,
            mainline_demand=mainline_demand,
            onramp_demand=onramp_demand,
            plot_results=True,
        )
    )

    # compute performance metrics and illustrate them
    VKT, VHT, avg_speed = network.compute_performance_metrics(
        density=density,
        flow=flow,
        speed=speed,
        input_queue=input_queue,
        onramp_queues=onramp_queue,
        dt=dt,
        plotting=True,
    )
    print(f"Total VKT: {VKT:.2f} veh-km")
    print(f"Total VHT: {VHT:.2f} veh-h")
    print(f"Overall Average Speed: {avg_speed:.2f} km/h")
