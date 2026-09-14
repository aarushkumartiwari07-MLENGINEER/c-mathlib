import unittest
import ctypes
import c_mathlib
from c_mathlib import is_prime
from c_mathlib.core import _c_is_prime


class TestIsPrime(unittest.TestCase):
    """Unit tests for c-mathlib is_prime functionality."""

    def test_c_symbol_is_foreign_function(self):
        """Verify that the underlying implementation is a ctypes foreign C function."""
        self.assertTrue(callable(_c_is_prime))
        # Ensure _c_is_prime has ctypes configuration
        self.assertEqual(_c_is_prime.argtypes, [ctypes.c_int])
        self.assertEqual(_c_is_prime.restype, ctypes.c_int)

    def test_negative_numbers(self):
        """Negative numbers must not be prime."""
        self.assertFalse(is_prime(-10))
        self.assertFalse(is_prime(-1))
        self.assertFalse(is_prime(-100))

    def test_zero_and_one(self):
        """0 and 1 are non-prime edge cases."""
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))

    def test_small_primes(self):
        """Test small prime numbers."""
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(7))

    def test_small_composites(self):
        """Test small composite numbers."""
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))
        self.assertFalse(is_prime(8))
        self.assertFalse(is_prime(9))

    def test_larger_primes(self):
        """Test larger prime numbers."""
        self.assertTrue(is_prime(17))
        self.assertTrue(is_prime(31))
        self.assertTrue(is_prime(97))
        self.assertTrue(is_prime(541))
        self.assertTrue(is_prime(104729))

    def test_larger_composites(self):
        """Test larger composite numbers."""
        self.assertFalse(is_prime(25))
        self.assertFalse(is_prime(100))
        self.assertFalse(is_prime(542))
        self.assertFalse(is_prime(100000))

    def test_invalid_types(self):
        """Ensure non-integer types raise TypeError."""
        with self.assertRaises(TypeError):
            is_prime(3.14)  # type: ignore

        with self.assertRaises(TypeError):
            is_prime("17")  # type: ignore

        with self.assertRaises(TypeError):
            is_prime(None)  # type: ignore

        with self.assertRaises(TypeError):
            is_prime(True)  # type: ignore


if __name__ == "__main__":
    unittest.main()
