import unittest
import math
from c_mathlib.special import gamma, lgamma, beta, erf, erfc, sigmoid, softmax


class TestSpecialFunctions(unittest.TestCase):
    """Unit tests for special functions and ML activation primitives in C."""

    # --------------------------------------------------------------------------
    # Gamma & Log-Gamma
    # --------------------------------------------------------------------------

    def test_gamma_integers(self):
        """Gamma(n) = (n - 1)! for positive integers."""
        self.assertAlmostEqual(gamma(1), 1.0, places=9)
        self.assertAlmostEqual(gamma(2), 1.0, places=9)
        self.assertAlmostEqual(gamma(3), 2.0, places=9)
        self.assertAlmostEqual(gamma(4), 6.0, places=9)
        self.assertAlmostEqual(gamma(5), 24.0, places=9)
        self.assertAlmostEqual(gamma(6), 120.0, places=8)

    def test_gamma_half_integers(self):
        """Gamma(1/2) = sqrt(pi) and Gamma(-1/2) = -2*sqrt(pi)."""
        self.assertAlmostEqual(gamma(0.5), math.sqrt(math.pi), places=8)
        self.assertAlmostEqual(gamma(-0.5), -2.0 * math.sqrt(math.pi), places=7)

    def test_lgamma(self):
        """lgamma(x) matches math.lgamma(x)."""
        for x in [0.5, 1.5, 2.5, 5.0, 10.0]:
            self.assertAlmostEqual(lgamma(x), math.lgamma(x), places=7)

    def test_gamma_singularities(self):
        """Gamma is undefined at 0 and negative integers."""
        with self.assertRaises(ValueError):
            gamma(0)
        with self.assertRaises(ValueError):
            gamma(-1)
        with self.assertRaises(ValueError):
            gamma(-5)
        with self.assertRaises(ValueError):
            lgamma(0)
        with self.assertRaises(ValueError):
            lgamma(-3)

    # --------------------------------------------------------------------------
    # Beta Function
    # --------------------------------------------------------------------------

    def test_beta_values(self):
        """Beta(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
        # B(1, 1) = 1.0
        self.assertAlmostEqual(beta(1, 1), 1.0, places=8)
        # B(2, 3) = 1! * 2! / 4! = 2 / 24 = 1/12
        self.assertAlmostEqual(beta(2, 3), 1.0 / 12.0, places=8)
        # Symmetry: B(a, b) == B(b, a)
        self.assertAlmostEqual(beta(2.5, 4.2), beta(4.2, 2.5), places=8)

    # --------------------------------------------------------------------------
    # Error Function (erf / erfc)
    # --------------------------------------------------------------------------

    def test_erf(self):
        """erf(0) = 0, erf(-x) = -erf(x), erf(1) ~ 0.84270079."""
        self.assertAlmostEqual(erf(0.0), 0.0, places=9)
        self.assertAlmostEqual(erf(1.0), 0.84270079, places=6)
        self.assertAlmostEqual(erf(-1.0), -0.84270079, places=6)
        self.assertAlmostEqual(erf(2.0), 0.995322265, places=5)

    def test_erfc(self):
        """erfc(x) = 1 - erf(x)."""
        for x in [-1.5, 0.0, 0.5, 1.0, 2.0]:
            self.assertAlmostEqual(erfc(x), 1.0 - erf(x), places=7)

    # --------------------------------------------------------------------------
    # Machine Learning Primitives: Sigmoid & Softmax
    # --------------------------------------------------------------------------

    def test_sigmoid(self):
        """sigmoid(0) = 0.5, sigmoid(x) + sigmoid(-x) = 1.0."""
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=9)
        self.assertAlmostEqual(sigmoid(2.0), 1.0 / (1.0 + math.exp(-2.0)), places=9)
        # Symmetry
        self.assertAlmostEqual(sigmoid(3.5) + sigmoid(-3.5), 1.0, places=9)
        # Extreme values without numerical overflow
        self.assertAlmostEqual(sigmoid(100.0), 1.0, places=9)
        self.assertAlmostEqual(sigmoid(-100.0), 0.0, places=9)

    def test_softmax(self):
        """softmax output forms a valid probability distribution summing to 1.0."""
        probs = softmax([1.0, 2.0, 3.0])
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertTrue(probs[0] < probs[1] < probs[2])

        # Test extreme logits (overflow resilience via max subtraction)
        large_probs = softmax([1000.0, 1001.0, 1002.0])
        self.assertAlmostEqual(sum(large_probs), 1.0, places=9)
        self.assertAlmostEqual(large_probs[0], probs[0], places=7)
        self.assertAlmostEqual(large_probs[1], probs[1], places=7)
        self.assertAlmostEqual(large_probs[2], probs[2], places=7)

    def test_type_validation_errors(self):
        """Invalid types and empty inputs must raise appropriate errors."""
        with self.assertRaises(TypeError):
            gamma("not_a_number")  # type: ignore
        with self.assertRaises(TypeError):
            beta("a", 1.0)  # type: ignore
        with self.assertRaises(TypeError):
            sigmoid(None)  # type: ignore
        with self.assertRaises(TypeError):
            softmax(123)  # type: ignore
        with self.assertRaises(TypeError):
            softmax(["a", "b"])  # type: ignore
        with self.assertRaises(ValueError):
            softmax([])


if __name__ == "__main__":
    unittest.main()
