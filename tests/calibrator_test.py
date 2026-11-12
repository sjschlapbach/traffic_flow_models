import numpy as np
import pytest

from traffic_flow_models import (
    METANET,
    METANETParams,
    Calibrator,
    Destination,
    MotorwayLink,
    Network,
    Node,
    Origin,
    Simulation,
)
from traffic_flow_models.network.calibrator import _centered_moving_average


def _one_link_ground_truth(tmp_path) -> tuple[Network, str]:
    """Simulate a single 2 km link with 10 s steps and return (network, file)."""
    link = MotorwayLink(
        id="m1",
        length=2.0,
        lanes=2,
        lane_capacity=2000,
        free_flow_speed=100,
        jam_density=180,
    )
    origin = Origin(id="o1")
    destination = Destination(id="d1")
    node1 = Node(id="n1", incoming=[origin], outgoing=[link])
    node2 = Node(id="n2", incoming=[link], outgoing=[destination])
    network = Network(nodes=[node1, node2])
    node1.set_position(0.0, 0.0)
    node2.set_position(1.0, 0.0)

    true_params = METANETParams(
        tau=22 / 3600, nu=15.0, kappa=10.0, delta=1.4, phi=10.0, alpha=2.0
    )
    Simulation(network, METANET(), true_params).run(
        duration=0.1,
        dt=10 / 3600,
        preferred_cell_size=0.5,
        origin_demands={"o1": lambda t: 3000.0},
        turning_rates={},
        destination_flow_bc={"d1": lambda t: 6000.0},
        destination_density_bc={"d1": lambda t: 0.0},
        plot_results=True,
        show_plots=False,
        results_dir=str(tmp_path),
    )
    return network, str(tmp_path / "simulation_results.json")


class TestCalibrator:
    def test_log_ratio_penalty_pulls_every_parameter_to_reference(self, tmp_path):
        # the relative penalty acts on tau (hours) as strongly as on nu (km^2/h);
        # a raw-unit penalty (p - p_ref) would leave tau at its data fit
        network, filepath = _one_link_ground_truth(tmp_path)
        reference = METANETParams(
            tau=35 / 3600, nu=40.0, kappa=30.0, delta=2.0, phi=5.0, alpha=1.5
        )

        calibrated, _, _ = Calibrator(network).calibrate_model_params(
            ground_truth_filepath=filepath,
            model=METANET(),
            window_size=10,
            regularization_weight=1e4,
            regularization_reference=reference,
            max_nfev=50,
            verbose=False,
        )

        for name in ("tau", "nu", "kappa", "delta", "phi", "alpha"):
            assert calibrated[name] == pytest.approx(reference[name], rel=0.02)

    def test_tau_lower_bound_is_raised_to_time_step_with_warning(self, tmp_path):
        # default tau bound starts at 5 s, the data use 10 s steps
        network, filepath = _one_link_ground_truth(tmp_path)

        with pytest.warns(UserWarning, match="time step"):
            calibrated, _, history = Calibrator(network).calibrate_model_params(
                ground_truth_filepath=filepath,
                model=METANET(),
                window_size=10,
                max_nfev=5,
                verbose=False,
            )

        assert np.all(history[:, 0] >= 10 / 3600)
        assert calibrated["tau"] >= 10 / 3600

    def test_centered_moving_average_shrinks_at_the_ends(self):
        states = np.array([[0.0, 3.0, 6.0, 9.0]])
        assert np.allclose(_centered_moving_average(states, 3), [[1.5, 3.0, 6.0, 7.5]])
        assert np.allclose(_centered_moving_average(states, 1), states)
        # series shorter than the window keeps its length
        assert np.allclose(_centered_moving_average(states[:, :2], 3), [[1.5, 1.5]])

    def test_even_smoothing_window_is_rejected(self):
        calibrator = Calibrator(Network(nodes=[]))
        with pytest.raises(ValueError, match="odd integer"):
            calibrator.calibrate_model_params(
                ground_truth_filepath="unused.json",
                model=METANET(),
                initial_state_smoothing=4,
            )
