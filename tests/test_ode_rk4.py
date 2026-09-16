import unittest
import math
from c_mathlib.numerical import rk4, Polynomial, ODEResult


class TestRK4(unittest.TestCase):
    """Unit tests for the Runge-Kutta 4th Order (RK4) ODE solver in C."""

    def test_rk4_exponential_decay(self):
        """Test dy/dt = -y, y(0) = 1.0 -> exact analytical y(t) = exp(-t)."""
        res = rk4("exp_decay", y0=1.0, t_span=(0.0, 2.0), steps=100)
        self.assertIsInstance(res, ODEResult)
        self.assertEqual(len(res), 101)
        self.assertAlmostEqual(res.t[0], 0.0)
        self.assertAlmostEqual(res.t[-1], 2.0)

        # Exact solution at t=2.0 is exp(-2.0)
        exact_final = math.exp(-2.0)
        self.assertAlmostEqual(res.y_final, exact_final, places=6)

        # Check trajectory at intermediate points
        for t_val, y_val in zip(res.t, res.y):
            self.assertAlmostEqual(y_val, math.exp(-t_val), places=5)

    def test_rk4_logistic_growth(self):
        """Test dy/dt = y*(1-y), y(0) = 0.5 -> exact analytical y(t) = 1 / (1 + exp(-t))."""
        res = rk4("logistic", y0=0.5, t_span=(0.0, 3.0), steps=150)
        exact_final = 1.0 / (1.0 + math.exp(-3.0))
        self.assertAlmostEqual(res.y_final, exact_final, places=6)

    def test_rk4_sine_ode(self):
        """Test dy/dt = cos(t), y(0) = 0.0 -> exact analytical y(t) = sin(t)."""
        res = rk4("sine", y0=0.0, t_span=(0.0, math.pi), steps=100)
        self.assertAlmostEqual(res.y_final, 0.0, places=6)

        # Midpoint at pi/2 -> sin(pi/2) = 1.0
        mid_idx = 50
        self.assertAlmostEqual(res.t[mid_idx], math.pi / 2.0, places=5)
        self.assertAlmostEqual(res.y[mid_idx], 1.0, places=5)

    def test_rk4_polynomial_derivative(self):
        """Test dy/dt = P(t) = 3t^2 -> exact analytical y(t) = t^3 (with y0=0)."""
        # P(t) = [0, 0, 3] -> 3*t^2
        res = rk4([0, 0, 3], y0=0.0, t_span=(0.0, 2.0), steps=50)
        # y(2) = 2^3 = 8.0
        self.assertAlmostEqual(res.y_final, 8.0, places=6)

        # Polynomial object support
        poly = Polynomial([0, 0, 3])
        res_poly = rk4(poly, y0=0.0, t_span=(0.0, 2.0), steps=50)
        self.assertAlmostEqual(res_poly.y_final, 8.0, places=6)

    def test_rk4_repr_and_properties(self):
        """Test ODEResult representations and property access."""
        res = rk4("exp_decay", y0=1.0, t_span=(0.0, 1.0), steps=10)
        repr_str = repr(res)
        self.assertIn("ODEResult", repr_str)
        self.assertIn("t_span", repr_str)
        self.assertEqual(res.y_final, res.y[-1])

    def test_rk4_validation_errors(self):
        """Test error handling for invalid arguments."""
        with self.assertRaises(ValueError):
            rk4("exp_decay", y0=1.0, steps=0)

        with self.assertRaises(ValueError):
            rk4("non_existent_ode", y0=1.0)

        with self.assertRaises(TypeError):
            rk4("exp_decay", y0="bad_y0")  # type: ignore

        with self.assertRaises(TypeError):
            rk4("exp_decay", y0=1.0, t_span=(0.0,))  # type: ignore

        with self.assertRaises(TypeError):
            rk4("exp_decay", y0=1.0, steps=10.5)  # type: ignore


if __name__ == "__main__":
    unittest.main()
