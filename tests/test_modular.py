import unittest
import ctypes
from c_mathlib import mod_pow, extended_gcd, mod_inverse
from c_mathlib.core import _c_mod_pow, _c_extended_gcd, _c_mod_inverse


class TestModularArithmetic(unittest.TestCase):
    """Unit tests for modular arithmetic and extended Euclidean algorithm."""

    def test_c_symbols_are_foreign_functions(self):
        """Verify ctypes bindings for modular functions."""
        self.assertTrue(callable(_c_mod_pow))
        self.assertEqual(
            _c_mod_pow.argtypes,
            [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong],
        )
        self.assertEqual(_c_mod_pow.restype, ctypes.c_longlong)

        self.assertTrue(callable(_c_extended_gcd))
        self.assertTrue(callable(_c_mod_inverse))

    # --------------------------------------------------------------------------
    # mod_pow Tests
    # --------------------------------------------------------------------------

    def test_mod_pow_standard(self):
        """Test typical modular exponentiation cases."""
        self.assertEqual(mod_pow(2, 10, 1000), 24)
        self.assertEqual(mod_pow(3, 4, 5), 1)  # 81 % 5 = 1
        self.assertEqual(mod_pow(7, 2, 100), 49)
        self.assertEqual(mod_pow(7, 3, 100), 43)

    def test_mod_pow_large_exponents(self):
        """Test large exponents that would overflow naive multiplication."""
        # 2^100 % 13: 2^12 = 4096 = 315*13 + 1 = 1 (mod 13) by Fermat's Little Theorem
        # 100 = 12*8 + 4 => 2^4 = 16 = 3 (mod 13)
        self.assertEqual(mod_pow(2, 100, 13), 3)
        # 5^6 = 1 (mod 7). 1000000 % 6 = 4 => 5^4 = 625 = 89*7 + 2 = 2 (mod 7)
        self.assertEqual(mod_pow(5, 1000000, 7), 2)

    def test_mod_pow_edge_cases(self):
        """Test base, exponent, and modulus edge cases."""
        self.assertEqual(mod_pow(5, 0, 13), 1)
        self.assertEqual(mod_pow(0, 5, 13), 0)
        self.assertEqual(mod_pow(0, 0, 13), 1)
        self.assertEqual(mod_pow(10, 5, 1), 0)  # Any value mod 1 is 0
        self.assertEqual(mod_pow(-3, 3, 11), 6)  # (-27) mod 11 = 6


    def test_mod_pow_invalid_inputs(self):
        """Test domain validation for mod_pow."""
        with self.assertRaises(ValueError):
            mod_pow(2, -1, 10)  # Negative exponent
        with self.assertRaises(ValueError):
            mod_pow(2, 5, 0)   # Modulus <= 0
        with self.assertRaises(ValueError):
            mod_pow(2, 5, -5)  # Negative modulus

    def test_mod_pow_type_and_overflow_errors(self):
        """Test type checking and overflow bounds."""
        with self.assertRaises(TypeError):
            mod_pow(2.0, 5, 10)  # type: ignore
        with self.assertRaises(TypeError):
            mod_pow(2, "5", 10)  # type: ignore
        with self.assertRaises(TypeError):
            mod_pow(2, 5, None)  # type: ignore
        with self.assertRaises(TypeError):
            mod_pow(True, 5, 10)  # type: ignore

        with self.assertRaises(OverflowError):
            mod_pow(2**63, 2, 10)

    # --------------------------------------------------------------------------
    # extended_gcd Tests
    # --------------------------------------------------------------------------

    def test_extended_gcd_standard(self):
        """Test Bézout's identity: a*x + b*y == gcd(a, b)."""
        test_pairs = [
            (30, 12),
            (35, 15),
            (17, 19),
            (252, 105),
            (1071, 462),
            (1, 1),
            (0, 5),
            (5, 0),
            (-30, 12),
            (30, -12),
            (-30, -12),
        ]
        for a, b in test_pairs:
            g, x, y = extended_gcd(a, b)
            self.assertEqual(a * x + b * y, g, f"Bézout identity failed for ({a}, {b})")
            self.assertGreaterEqual(g, 0, f"GCD must be non-negative for ({a}, {b})")

    def test_extended_gcd_values(self):
        """Verify explicit return values of extended_gcd."""
        g, x, y = extended_gcd(30, 12)
        self.assertEqual(g, 6)
        self.assertEqual(30 * x + 12 * y, 6)

    def test_extended_gcd_invalid_types(self):
        """Ensure non-integers raise TypeError."""
        with self.assertRaises(TypeError):
            extended_gcd(30.5, 12)  # type: ignore
        with self.assertRaises(TypeError):
            extended_gcd(30, None)  # type: ignore
        with self.assertRaises(TypeError):
            extended_gcd(True, 12)  # type: ignore

    # --------------------------------------------------------------------------
    # mod_inverse Tests
    # --------------------------------------------------------------------------

    def test_mod_inverse_standard(self):
        """Test modular multiplicative inverse calculation."""
        # 3 * 4 = 12 = 1 (mod 11)
        self.assertEqual(mod_inverse(3, 11), 4)
        self.assertEqual((3 * mod_inverse(3, 11)) % 11, 1)

        # 10 * 12 = 120 = 7*17 + 1 = 1 (mod 17)
        self.assertEqual(mod_inverse(10, 17), 12)
        self.assertEqual((10 * mod_inverse(10, 17)) % 17, 1)

        # 7 * x = 1 (mod 26) => 7 * 15 = 105 = 4*26 + 1 => x = 15
        self.assertEqual(mod_inverse(7, 26), 15)

    def test_mod_inverse_negative_a(self):
        """Test modular inverse with negative numbers."""
        inv = mod_inverse(-3, 11)
        self.assertEqual((-3 * inv) % 11, 1)

    def test_mod_inverse_non_existent(self):
        """Test that non-coprime numbers raise ValueError."""
        # gcd(6, 9) = 3 != 1
        with self.assertRaises(ValueError):
            mod_inverse(6, 9)
        # gcd(4, 12) = 4 != 1
        with self.assertRaises(ValueError):
            mod_inverse(4, 12)
        # gcd(0, 5) = 5 != 1
        with self.assertRaises(ValueError):
            mod_inverse(0, 5)

    def test_mod_inverse_invalid_modulus(self):
        """Test invalid modulus values (modulus must be > 1)."""
        with self.assertRaises(ValueError):
            mod_inverse(3, 1)
        with self.assertRaises(ValueError):
            mod_inverse(3, 0)
        with self.assertRaises(ValueError):
            mod_inverse(3, -5)

    def test_mod_inverse_invalid_types(self):
        """Test non-integer types."""
        with self.assertRaises(TypeError):
            mod_inverse(3.0, 11)  # type: ignore
        with self.assertRaises(TypeError):
            mod_inverse(3, "11")  # type: ignore
        with self.assertRaises(TypeError):
            mod_inverse(True, 11)  # type: ignore


if __name__ == "__main__":
    unittest.main()
