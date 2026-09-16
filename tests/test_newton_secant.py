import unittest
import math
from c_mathlib.numerical import newton, secant, Polynomial


class TestNewtonSecant(unittest.TestCase):
    """Unit tests for Newton-Raphson and Secant root finding methods in C."""

    # --------------------------------------------------------------------------
    # Newton-Raphson Tests
    # --------------------------------------------------------------------------

    def test_newton_polynomial(self):
        """Test Newton-Raphson on polynomials with exact analytical derivatives."""
        # P(x) = x^2 - 9 = 0 -> root = 3.0
        root1 = newton([-9, 0, 1], x0=5.0, tol=1e-10)
        self.assertAlmostEqual(root1, 3.0, places=9)

        # P(x) = x^3 - x - 2 = 0 -> root ~ 1.5213797068
        p = Polynomial([-2, -1, 0, 1])
        root2 = newton(p, x0=1.5, tol=1e-10)
        self.assertAlmostEqual(root2, 1.5213797068, places=8)

    def test_newton_transcendental(self):
        """Test Newton-Raphson on built-in transcendental functions."""
        # sin(x) = 0 near x0 = 3.0 -> root is pi
        root_sin = newton("sin", x0=3.0, tol=1e-10)
        self.assertAlmostEqual(root_sin, math.pi, places=9)

        # cos(x) = 0 near x0 = 1.0 -> root is pi/2
        root_cos = newton("cos", x0=1.0, tol=1e-10)
        self.assertAlmostEqual(root_cos, math.pi / 2, places=9)

        # x^2 - 4 = 0 near x0 = 3.0 -> root is 2.0
        root_quad = newton("x^2 - 4", x0=3.0, tol=1e-10)
        self.assertAlmostEqual(root_quad, 2.0, places=9)

    def test_newton_zero_derivative_error(self):
        """Zero derivative at starting point must raise ValueError."""
        # x^2 - 4 at x0 = 0.0 has derivative 2(0) = 0
        with self.assertRaises(ValueError):
            newton("x^2 - 4", x0=0.0)

    def test_newton_invalid_parameters(self):
        """Invalid tolerances and types must raise appropriate exceptions."""
        with self.assertRaises(ValueError):
            newton("x^2 - 4", x0=3.0, tol=-1e-5)
        with self.assertRaises(ValueError):
            newton("x^2 - 4", x0=3.0, max_iter=0)
        with self.assertRaises(TypeError):
            newton("x^2 - 4", x0="invalid")  # type: ignore

    # --------------------------------------------------------------------------
    # Secant Method Tests
    # --------------------------------------------------------------------------

    def test_secant_polynomial(self):
        """Test Secant method on polynomials."""
        # P(x) = x^3 - x - 2 = 0
        root = secant([-2, -1, 0, 1], x0=1.0, x1=2.0, tol=1e-10)
        self.assertAlmostEqual(root, 1.5213797068, places=8)

        # P(x) = x^2 - 16 = 0
        root_16 = secant([-16, 0, 1], x0=2.0, x1=6.0, tol=1e-10)
        self.assertAlmostEqual(root_16, 4.0, places=9)

    def test_secant_transcendental(self):
        """Test Secant method on transcendental functions."""
        # sin(x) = 0 on [3, 4] -> pi
        root = secant("sin", x0=3.0, x1=4.0, tol=1e-10)
        self.assertAlmostEqual(root, math.pi, places=9)

    def test_secant_zero_slope_error(self):
        """Symmetric points where f(x0) == f(x1) must raise ValueError."""
        # x^2 - 4 at x0 = -2, x1 = 2 has f(-2) = f(2) = 0 or f(-1) = f(1) = -3
        with self.assertRaises(ValueError):
            secant("x^2 - 4", x0=-1.0, x1=1.0)

    def test_secant_invalid_parameters(self):
        """Invalid bounds and types must raise exceptions."""
        with self.assertRaises(ValueError):
            secant("sin", x0=3.0, x1=4.0, tol=0.0)
        with self.assertRaises(TypeError):
            secant("sin", x0=None, x1=4.0)  # type: ignore


if __name__ == "__main__":
    unittest.main()
