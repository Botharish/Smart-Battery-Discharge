import unittest

import numpy as np

from models import BatteryConfig, current_at, theoretical_soc
from numerical_methods import euler_step, rk4_step
from simulation import predict_remaining_time_h, simulate
from validation import solver_comparison


class BatteryDischargeTests(unittest.TestCase):
    def make_config(self, **kwargs):
        base = dict(capacity_ah=4.0, initial_soc_percent=80.0, base_current_a=2.0, voltage_v=3.7, duration_h=2.0, step_minutes=5.0)
        base.update(kwargs)
        return BatteryConfig(**base)

    def test_constant_current_matches_theoretical_soc(self):
        config = self.make_config()
        frame = simulate(config, "RK4")
        self.assertTrue(np.allclose(frame.soc, frame.theoretical_soc, atol=1e-12))

    def test_constant_current_runtime_formula(self):
        config = self.make_config(initial_soc_percent=50, base_current_a=1.0)
        self.assertAlmostEqual(predict_remaining_time_h(config), 2.0)

    def test_zero_current_has_no_finite_runtime(self):
        config = self.make_config(base_current_a=0.0)
        self.assertIsNone(predict_remaining_time_h(config))
        self.assertTrue(np.all(simulate(config).battery_percent == 80.0))

    def test_discharge_stops_at_zero_soc(self):
        config = self.make_config(initial_soc_percent=10, base_current_a=10, duration_h=2, step_minutes=20)
        frame = simulate(config)
        self.assertEqual(frame.soc.iloc[-1], 0.0)
        self.assertAlmostEqual(frame.time_h.iloc[-1], 0.04)

    def test_capacity_remaining_consistent_with_soc(self):
        frame = simulate(self.make_config())
        self.assertTrue(np.allclose(frame.remaining_capacity_ah, frame.soc * 4.0))

    def test_power_equals_voltage_times_current(self):
        config = self.make_config()
        frame = simulate(config)
        self.assertTrue(np.allclose(frame.power_w, frame.current_a * config.voltage_v))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_config(capacity_ah=0).validate()
        with self.assertRaises(ValueError):
            self.make_config(initial_soc_percent=110).validate()

    def test_variable_load_is_nonnegative(self):
        config = self.make_config(profile="sinusoidal", variation_percent=100)
        values = current_at(np.linspace(0, 3600, 100), config)
        self.assertGreaterEqual(float(np.min(values)), 0.0)

    def test_rk4_error_smaller_than_euler_for_sinusoidal_current(self):
        config = self.make_config(profile="sinusoidal", variation_percent=80, period_minutes=37, step_minutes=17)
        frame, errors = solver_comparison(config)
        self.assertGreater(len(frame), 2)
        self.assertLess(errors["RK4 maximum absolute SOC error"], errors["Euler maximum absolute SOC error"])


class NumericalMethodTests(unittest.TestCase):
    def test_euler_and_rk4_constant_derivative(self):
        f = lambda _t, _y: -2.0
        self.assertEqual(euler_step(f, 0, 1, 0.5), 0.0)
        self.assertEqual(rk4_step(f, 0, 1, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
