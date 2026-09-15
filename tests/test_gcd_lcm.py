import unittest
import ctypes
from c_mathlib import gcd, lcm
from c_mathlib.core import _c_gcd, _c_lcm


class TestGcdLcm(unittest.TestCase):
    """Unit tests for c-mathlib gcd and lcm functionality."""

    def test_c_symbols_are_foreign_functions(self):
        """Verify that the underlying implementations are ctypes foreign C functions."""
        self.assertTrue(callable(_c_gcd))
        self.assertEqual(_c_gcd.argtypes, [ctypes.c_longlong, ctypes.c_longlong])
        self.assertEqual(_c_gcd.restype, ctypes.c_longlong)

        self.assertTrue(callable(_c_lcm))
        self.assertEqual(_c_lcm.argtypes, [ctypes.c_longlong, ctypes.c_longlong])
        self.assertEqual(_c_lcm.restype, ctypes.c_longlong)

    # --------------------------------------------------------------------------
    # GCD Tests
    # --------------------------------------------------------------------------

    def test_gcd_standard_cases(self):
        """Test typical GCD values."""
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(18, 48), 6)
        self.assertEqual(gcd(100, 25), 25)
        self.assertEqual(gcd(54, 24), 6)
        self.assertEqual(gcd(1071, 462), 21)

    def test_gcd_coprime(self):
        """Test coprime numbers whose GCD is 1."""
        self.assertEqual(gcd(13, 7), 1)
        self.assertEqual(gcd(17, 19), 1)
        self.assertEqual(gcd(8, 9), 1)

    def test_gcd_with_zeros(self):
        """Test GCD edge cases involving 0."""
        self.assertEqual(gcd(0, 5), 5)
        self.assertEqual(gcd(5, 0), 5)
        self.assertEqual(gcd(0, 0), 0)
        self.assertEqual(gcd(-7, 0), 7)
        self.assertEqual(gcd(0, -7), 7)

    def test_gcd_negative_numbers(self):
        """Test GCD with negative integers (must return positive result)."""
        self.assertEqual(gcd(-48, 18), 6)
        self.assertEqual(gcd(48, -18), 6)
        self.assertEqual(gcd(-48, -18), 6)
        self.assertEqual(gcd(-10, -5), 5)

    def test_gcd_equal_numbers(self):
        """Test GCD when both arguments are equal."""
        self.assertEqual(gcd(42, 42), 42)
        self.assertEqual(gcd(-42, 42), 42)
        self.assertEqual(gcd(1, 1), 1)

    def test_gcd_large_values(self):
        """Test 64-bit integer values."""
        self.assertEqual(gcd(1000000000000, 250000000000), 250000000000)
        self.assertEqual(gcd(9223372036854775806, 2), 2)

    def test_gcd_invalid_types(self):
        """Ensure non-integers raise TypeError."""
        with self.assertRaises(TypeError):
            gcd(12.5, 4)  # type: ignore
        with self.assertRaises(TypeError):
            gcd(12, "4")  # type: ignore
        with self.assertRaises(TypeError):
            gcd(None, 4)  # type: ignore
        with self.assertRaises(TypeError):
            gcd(True, 4)  # type: ignore
        with self.assertRaises(TypeError):
            gcd(4, False)  # type: ignore

    def test_gcd_overflow(self):
        """Ensure values outside 64-bit signed integer range raise OverflowError."""
        with self.assertRaises(OverflowError):
            gcd(2**63, 10)
        with self.assertRaises(OverflowError):
            gcd(10, -2**63 - 1)

    # --------------------------------------------------------------------------
    # LCM Tests
    # --------------------------------------------------------------------------

    def test_lcm_standard_cases(self):
        """Test typical LCM values."""
        self.assertEqual(lcm(12, 18), 36)
        self.assertEqual(lcm(18, 12), 36)
        self.assertEqual(lcm(4, 6), 12)
        self.assertEqual(lcm(21, 6), 42)

    def test_lcm_coprime(self):
        """Test LCM of coprime numbers (product of the two)."""
        self.assertEqual(lcm(13, 7), 91)
        self.assertEqual(lcm(8, 9), 72)
        self.assertEqual(lcm(1, 5), 5)

    def test_lcm_with_zeros(self):
        """Test LCM edge cases involving 0."""
        self.assertEqual(lcm(0, 5), 0)
        self.assertEqual(lcm(5, 0), 0)
        self.assertEqual(lcm(0, 0), 0)
        self.assertEqual(lcm(-5, 0), 0)

    def test_lcm_negative_numbers(self):
        """Test LCM with negative integers (must return positive result)."""
        self.assertEqual(lcm(-12, 18), 36)
        self.assertEqual(lcm(12, -18), 36)
        self.assertEqual(lcm(-12, -18), 36)

    def test_lcm_equal_numbers(self):
        """Test LCM when both arguments are equal."""
        self.assertEqual(lcm(42, 42), 42)
        self.assertEqual(lcm(-42, 42), 42)

    def test_lcm_large_values(self):
        """Test 64-bit integer values."""
        self.assertEqual(lcm(1000000000, 2000000000), 2000000000)

    def test_lcm_invalid_types(self):
        """Ensure non-integers raise TypeError."""
        with self.assertRaises(TypeError):
            lcm(3.14, 2)  # type: ignore
        with self.assertRaises(TypeError):
            lcm(3, "2")  # type: ignore
        with self.assertRaises(TypeError):
            lcm(None, 2)  # type: ignore
        with self.assertRaises(TypeError):
            lcm(True, 2)  # type: ignore
        with self.assertRaises(TypeError):
            lcm(2, False)  # type: ignore

    def test_lcm_overflow(self):
        """Ensure values outside 64-bit signed integer range raise OverflowError."""
        with self.assertRaises(OverflowError):
            lcm(2**63, 10)
        with self.assertRaises(OverflowError):
            lcm(10, -2**63 - 1)


if __name__ == "__main__":
    unittest.main()
