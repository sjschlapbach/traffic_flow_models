from traffic_flow_models import CTM
from demo.scenarios import mainline_demand_a, onramp_demand_a, setup_network_a


if __name__ == "__main__":
    # ! simulation configuration parameters
    scenario = "A"
    alinea_ramp_control = False
    dt = 10.0 / 3600
    duration = 5000.0 / 3600

    # select the appropriate scenario functions
    if scenario == "A":
        mainline_demand = mainline_demand_a
        onramp_demand = onramp_demand_a
        setup_network = setup_network_a
    else:
        raise ValueError(f"Scenario {scenario} is not defined.")

    # initialize the network with the correct structure (optionally with ALINEA ramp metering)
    network = setup_network(ramp_control=alinea_ramp_control)

    # run a simulation of the network using the CTM model
    ctm = CTM()
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
