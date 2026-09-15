import unittest
import math
from c_mathlib.numerical import (
    bisection,
    simpson,
    derivative,
    evaluate,
    Polynomial,
)


class TestNumericalMethods(unittest.TestCase):
    """Unit tests for numerical mathematics module in c-mathlib."""

    # --------------------------------------------------------------------------
    # Function Evaluation Tests
    # --------------------------------------------------------------------------

    def test_evaluate_builtin_functions(self):
        """Test basic built-in function evaluation in C."""
        self.assertAlmostEqual(evaluate("x^2", 3.0), 9.0)
        self.assertAlmostEqual(evaluate("x^3", 2.0), 8.0)
        self.assertAlmostEqual(evaluate("sin", math.pi / 2), 1.0)
        self.assertAlmostEqual(evaluate("cos", 0.0), 1.0)
        self.assertAlmostEqual(evaluate("exp", 1.0), math.e)
        self.assertAlmostEqual(evaluate("log", math.e), 1.0)

    def test_evaluate_polynomial(self):
        """Test polynomial evaluation via Horner's method in C."""
        # P(x) = 5 - 2x + 4x^3 at x = 2: 5 - 4 + 32 = 33
        poly = Polynomial([5, -2, 0, 4])
        self.assertAlmostEqual(evaluate(poly, 2.0), 33.0)
        self.assertAlmostEqual(evaluate([5, -2, 0, 4], 2.0), 33.0)

    def test_polynomial_class_features(self):
        """Test Polynomial attributes and edge cases."""
        poly = Polynomial([1, 2, 3])
        self.assertEqual(poly.degree, 2)
        self.assertEqual(poly.coeffs, [1.0, 2.0, 3.0])
        self.assertIn("x^2", repr(poly))

        with self.assertRaises(ValueError):
            Polynomial([])
        with self.assertRaises(TypeError):
            Polynomial("invalid")  # type: ignore

    # --------------------------------------------------------------------------
    # Bisection Root Finding Tests
    # --------------------------------------------------------------------------

    def test_bisection_builtin_quadratic(self):
        """Find root of x^2 - 4 = 0 in [0, 3] (root is 2)."""
        root = bisection("x^2 - 4", 0.0, 3.0, tol=1e-8)
        self.assertAlmostEqual(root, 2.0, places=7)

    def test_bisection_builtin_cubic(self):
        """Find root of x^3 - x - 2 = 0 in [1, 2] (~1.5213797)."""
        root = bisection("x^3 - x - 2", 1.0, 2.0, tol=1e-7)
        self.assertAlmostEqual(root, 1.5213797, places=6)

    def test_bisection_builtin_trig(self):
        """Find root of sin(x) = 0 in [3, 4] (root is pi)."""
        root = bisection("sin", 3.0, 4.0, tol=1e-8)
        self.assertAlmostEqual(root, math.pi, places=7)

        # cos(x) = 0 in [0, 2] (root is pi/2)
        root = bisection("cos", 0.0, 2.0, tol=1e-8)
        self.assertAlmostEqual(root, math.pi / 2, places=7)

    def test_bisection_polynomial(self):
        """Find root of polynomial via coefficients."""
        # P(x) = -9 + x^2 => root = 3 in [0, 5]
        root = bisection([-9, 0, 1], 0.0, 5.0, tol=1e-8)
        self.assertAlmostEqual(root, 3.0, places=7)

    def test_bisection_boundary_is_root(self):
        """Root at exact interval boundary."""
        root = bisection("x^2 - 4", 2.0, 5.0)
        self.assertAlmostEqual(root, 2.0, places=7)

    def test_bisection_unbracketed_error(self):
        """Ensure unbracketed intervals (same sign) raise ValueError."""
        with self.assertRaises(ValueError):
            bisection("x^2", 1.0, 2.0)
        with self.assertRaises(ValueError):
            bisection("x^2 - 4", 3.0, 5.0)

    def test_bisection_invalid_params(self):
        """Ensure invalid tolerances and iteration counts raise ValueError."""
        with self.assertRaises(ValueError):
            bisection("x^2 - 4", 0.0, 3.0, tol=-1e-5)
        with self.assertRaises(ValueError):
            bisection("x^2 - 4", 0.0, 3.0, max_iter=0)
        with self.assertRaises(TypeError):
            bisection("x^2 - 4", 0.0, "3.0")  # type: ignore

    # --------------------------------------------------------------------------
    # Simpson's Rule Integration Tests
    # --------------------------------------------------------------------------

    def test_simpson_polynomial_integrals(self):
        """Test analytical definite integrals of polynomials."""
        # integral_0^1 x^2 dx = 1/3
        val = simpson("x^2", 0.0, 1.0, n=100)
        self.assertAlmostEqual(val, 1.0 / 3.0, places=7)

        # integral_0^1 x^3 dx = 1/4
        val = simpson("x^3", 0.0, 1.0, n=100)
        self.assertAlmostEqual(val, 0.25, places=7)

        # integral_0^2 (1 + 2x + 3x^2) dx = [x + x^2 + x^3]_0^2 = 2 + 4 + 8 = 14
        val = simpson([1, 2, 3], 0.0, 2.0, n=100)
        self.assertAlmostEqual(val, 14.0, places=7)

    def test_simpson_trig_and_exp_integrals(self):
        """Test analytical integrals of transcendental functions."""
        # integral_0^pi sin(x) dx = [-cos(x)]_0^pi = 2.0
        val = simpson("sin", 0.0, math.pi, n=200)
        self.assertAlmostEqual(val, 2.0, places=7)

        # integral_0^(pi/2) cos(x) dx = [sin(x)]_0^(pi/2) = 1.0
        val = simpson("cos", 0.0, math.pi / 2, n=200)
        self.assertAlmostEqual(val, 1.0, places=7)

        # integral_0^1 exp(x) dx = e - 1
        val = simpson("exp", 0.0, 1.0, n=200)
        self.assertAlmostEqual(val, math.e - 1.0, places=7)

    def test_simpson_zero_width_interval(self):
        """Integral over [a, a] must be 0."""
        val = simpson("x^2", 3.0, 3.0, n=10)
        self.assertAlmostEqual(val, 0.0, places=7)

    def test_simpson_invalid_intervals(self):
        """Odd or n < 2 must raise ValueError."""
        with self.assertRaises(ValueError):
            simpson("x^2", 0.0, 1.0, n=3)  # Odd n
        with self.assertRaises(ValueError):
            simpson("x^2", 0.0, 1.0, n=0)  # n < 2
        with self.assertRaises(TypeError):
            simpson("x^2", 0.0, 1.0, n=2.5)  # type: ignore

    # --------------------------------------------------------------------------
    # Numerical Derivative Tests
    # --------------------------------------------------------------------------

    def test_derivative_polynomials(self):
        """Test analytical derivatives of polynomials."""
        # d/dx (x^2) at x = 3 is 6
        self.assertAlmostEqual(derivative("x^2", 3.0), 6.0, places=5)

        # d/dx (x^3) at x = 2 is 12
        self.assertAlmostEqual(derivative("x^3", 2.0), 12.0, places=5)

        # d/dx (5 - 2x + 4x^3) at x = 2 is -2 + 12(4) = 46
        self.assertAlmostEqual(derivative([5, -2, 0, 4], 2.0), 46.0, places=5)

    def test_derivative_transcendental(self):
        """Test analytical derivatives of trig and exp functions."""
        # d/dx (sin x) at x = 0 is cos(0) = 1
        self.assertAlmostEqual(derivative("sin", 0.0), 1.0, places=6)

        # d/dx (cos x) at x = 0 is -sin(0) = 0
        self.assertAlmostEqual(derivative("cos", 0.0), 0.0, places=6)

        # d/dx (exp x) at x = 0 is exp(0) = 1
        self.assertAlmostEqual(derivative("exp", 0.0), 1.0, places=6)

        # d/dx (log x) at x = 1 is 1/1 = 1
        self.assertAlmostEqual(derivative("log", 1.0), 1.0, places=6)

    def test_derivative_invalid_h(self):
        """Ensure non-positive step sizes raise ValueError."""
        with self.assertRaises(ValueError):
            derivative("x^2", 3.0, h=0.0)
        with self.assertRaises(ValueError):
            derivative("x^2", 3.0, h=-1e-5)
        with self.assertRaises(TypeError):
            derivative("x^2", "3.0")  # type: ignore


if __name__ == "__main__":
    unittest.main()
