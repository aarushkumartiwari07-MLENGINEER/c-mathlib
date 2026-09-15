import unittest
from c_mathlib.combinatorics import (
    combinations,
    permutations,
    euler_totient,
    nCr,
    nPr,
)


class TestCombinatorics(unittest.TestCase):
    """Unit tests for c-mathlib combinatorics module."""

    # --------------------------------------------------------------------------
    # Combinations Tests
    # --------------------------------------------------------------------------

    def test_combinations_standard(self):
        """Test standard binomial coefficients."""
        self.assertEqual(combinations(5, 2), 10)
        self.assertEqual(combinations(10, 3), 120)
        self.assertEqual(combinations(52, 5), 2598960)
        self.assertEqual(nCr(5, 2), 10)

    def test_combinations_boundary(self):
        """Test boundary conditions for combinations."""
        self.assertEqual(combinations(0, 0), 1)
        self.assertEqual(combinations(10, 0), 1)
        self.assertEqual(combinations(10, 10), 1)
        self.assertEqual(combinations(10, 1), 10)
        self.assertEqual(combinations(10, 9), 10)

    def test_combinations_invalid(self):
        """Test domain validation for combinations."""
        with self.assertRaises(ValueError):
            combinations(5, 6)  # k > n
        with self.assertRaises(ValueError):
            combinations(-5, 2)  # n < 0
        with self.assertRaises(ValueError):
            combinations(5, -1)  # k < 0
        with self.assertRaises(TypeError):
            combinations(5.0, 2)  # type: ignore

    # --------------------------------------------------------------------------
    # Permutations Tests
    # --------------------------------------------------------------------------

    def test_permutations_standard(self):
        """Test standard permutation calculations."""
        self.assertEqual(permutations(5, 2), 20)
        self.assertEqual(permutations(10, 3), 720)
        self.assertEqual(permutations(5, 5), 120)
        self.assertEqual(nPr(5, 2), 20)

    def test_permutations_boundary(self):
        """Test boundary conditions for permutations."""
        self.assertEqual(permutations(0, 0), 1)
        self.assertEqual(permutations(10, 0), 1)
        self.assertEqual(permutations(10, 1), 10)

    def test_permutations_invalid(self):
        """Test domain validation for permutations."""
        with self.assertRaises(ValueError):
            permutations(5, 6)  # k > n
        with self.assertRaises(ValueError):
            permutations(-5, 2)  # n < 0
        with self.assertRaises(ValueError):
            permutations(5, -1)  # k < 0
        with self.assertRaises(TypeError):
            permutations("5", 2)  # type: ignore

    # --------------------------------------------------------------------------
    # Euler's Totient Tests
    # --------------------------------------------------------------------------

    def test_euler_totient_standard(self):
        """Test Euler's totient function."""
        self.assertEqual(euler_totient(1), 1)
        self.assertEqual(euler_totient(2), 1)
        self.assertEqual(euler_totient(7), 6)  # Prime: p - 1
        self.assertEqual(euler_totient(13), 12)
        self.assertEqual(euler_totient(9), 6)  # 1, 2, 4, 5, 7, 8
        self.assertEqual(euler_totient(10), 4)  # 1, 3, 7, 9
        self.assertEqual(euler_totient(36), 12)
        self.assertEqual(euler_totient(100), 40)

    def test_euler_totient_invalid(self):
        """Test domain validation for euler_totient."""
        with self.assertRaises(ValueError):
            euler_totient(0)
        with self.assertRaises(ValueError):
            euler_totient(-10)
        with self.assertRaises(TypeError):
            euler_totient(3.14)  # type: ignore


if __name__ == "__main__":
    unittest.main()
