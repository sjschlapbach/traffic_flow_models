import numpy as np

from traffic_flow_models import AlineaController


class TestAlineaController:
    def test_attributes_assigned(self):
        c = AlineaController(gain=100.0, setpoint=30.0, measurement_cell=0)
        assert c.gain == 100.0
        assert c.setpoint == 30.0
        assert c.measurement_cell == 0

    def test_compute_regulated_flow_increases_when_below_setpoint(self):
        c = AlineaController(gain=2.0, setpoint=10.0, measurement_cell=0)
        measured = np.array([5.0], dtype=np.float64)
        prev = 100.0
        regulated = c.compute_regulated_flow(
            measured_densities=measured, previous_flow=prev
        )
        # expected: prev + gain*(setpoint - measured)
        assert regulated == 110.0

    def test_compute_regulated_flow_non_negative(self):
        c = AlineaController(gain=1.0, setpoint=0.0, measurement_cell=0)
        measured = np.array([1000.0], dtype=np.float64)
        regulated = c.compute_regulated_flow(
            measured_densities=measured, previous_flow=0.0
        )
        # ensure non-negative output even when adjustment would go negative
        assert regulated == 0.0
