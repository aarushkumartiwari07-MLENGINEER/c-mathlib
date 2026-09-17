import unittest
import math
from c_mathlib.numerical import minimize_1d, Polynomial, OptimizeResult


class TestOptimization(unittest.TestCase):
    """Unit tests for Golden Section Search 1D optimization in C."""

    def test_minimize_polynomial(self):
        """Test minimizing quadratic P(x) = (x - 3)^2 + 2 = x^2 - 6x + 11 on [0, 6]."""
        # Coefficients: [11, -6, 1]
        res = minimize_1d([11, -6, 1], bracket=(0.0, 6.0), tol=1e-8)
        self.assertIsInstance(res, OptimizeResult)
        self.assertTrue(res.converged)
        self.assertAlmostEqual(res.x, 3.0, places=6)
        self.assertAlmostEqual(res.fun, 2.0, places=6)
        self.assertGreater(res.nit, 0)

        # Using Polynomial object
        poly = Polynomial([11, -6, 1])
        res_poly = minimize_1d(poly, bracket=(0.0, 6.0), tol=1e-8)
        self.assertAlmostEqual(res_poly.x, 3.0, places=6)
        self.assertAlmostEqual(res_poly.fun, 2.0, places=6)

    def test_minimize_builtin_quadratic(self):
        """Test minimizing x^2 - 4 on [-3, 3] -> minimum at x = 0, f(x) = -4."""
        res = minimize_1d("x^2 - 4", bracket=(-3.0, 3.0), tol=1e-7)
        self.assertTrue(res.converged)
        self.assertAlmostEqual(res.x, 0.0, places=6)
        self.assertAlmostEqual(res.fun, -4.0, places=6)

    def test_minimize_trig_functions(self):
        """Test minimizing sin(x) on [3.0, 6.0] -> minimum at 3*pi/2 with fun = -1."""
        res_sin = minimize_1d("sin", bracket=(3.0, 6.0), tol=1e-8)
        self.assertTrue(res_sin.converged)
        self.assertAlmostEqual(res_sin.x, 1.5 * math.pi, places=6)
        self.assertAlmostEqual(res_sin.fun, -1.0, places=6)

        # cos(x) on [2.0, 4.5] -> minimum at pi with fun = -1
        res_cos = minimize_1d("cos", bracket=(2.0, 4.5), tol=1e-8)
        self.assertTrue(res_cos.converged)
        self.assertAlmostEqual(res_cos.x, math.pi, places=6)
        self.assertAlmostEqual(res_cos.fun, -1.0, places=6)

    def test_minimize_reversed_bracket(self):
        """Reversed bracket (b, a) should be automatically ordered."""
        res = minimize_1d("x^2 - 4", bracket=(3.0, -3.0), tol=1e-7)
        self.assertTrue(res.converged)
        self.assertAlmostEqual(res.x, 0.0, places=6)

    def test_minimize_max_iter_not_converged(self):
        """Very small max_iter should return converged=False without crashing."""
        res = minimize_1d("x^2 - 4", bracket=(-3.0, 3.0), tol=1e-12, max_iter=3)
        self.assertFalse(res.converged)
        self.assertEqual(res.nit, 3)

    def test_optimize_result_repr(self):
        """Test OptimizeResult string representation."""
        res = minimize_1d("x^2 - 4", bracket=(-3.0, 3.0), tol=1e-5)
        repr_str = repr(res)
        self.assertIn("OptimizeResult", repr_str)
        self.assertIn("converged=True", repr_str)

    def test_minimize_validation_errors(self):
        """Ensure invalid tolerances, brackets, and types raise appropriate errors."""
        with self.assertRaises(TypeError):
            minimize_1d("sin", bracket="not-a-tuple")  # type: ignore
        with self.assertRaises(TypeError):
            minimize_1d("sin", bracket=(0.0,))  # type: ignore
        with self.assertRaises(TypeError):
            minimize_1d("sin", bracket=("a", "b"))  # type: ignore
        with self.assertRaises(ValueError):
            minimize_1d("sin", bracket=(0.0, 1.0), tol=-1e-5)
        with self.assertRaises(ValueError):
            minimize_1d("sin", bracket=(0.0, 1.0), max_iter=0)
        with self.assertRaises(TypeError):
            minimize_1d("sin", bracket=(0.0, 1.0), max_iter=10.5)  # type: ignore


if __name__ == "__main__":
    unittest.main()
