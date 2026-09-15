import unittest
import ctypes
from c_mathlib import factorial, fibonacci
from c_mathlib.core import _c_factorial, _c_fibonacci


class TestFactorialFibonacci(unittest.TestCase):
    """Unit tests for c-mathlib factorial and fibonacci functionality."""

    def test_c_symbols_are_foreign_functions(self):
        """Verify that the underlying implementations are ctypes foreign C functions."""
        self.assertTrue(callable(_c_factorial))
        self.assertEqual(_c_factorial.argtypes, [ctypes.c_int])
        self.assertEqual(_c_factorial.restype, ctypes.c_longlong)

        self.assertTrue(callable(_c_fibonacci))
        self.assertEqual(_c_fibonacci.argtypes, [ctypes.c_int])
        self.assertEqual(_c_fibonacci.restype, ctypes.c_longlong)

    # --------------------------------------------------------------------------
    # Factorial Tests
    # --------------------------------------------------------------------------

    def test_factorial_base_cases(self):
        """0! and 1! must equal 1."""
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)

    def test_factorial_standard_cases(self):
        """Test standard small-to-medium factorial values."""
        self.assertEqual(factorial(2), 2)
        self.assertEqual(factorial(3), 6)
        self.assertEqual(factorial(4), 24)
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(6), 720)
        self.assertEqual(factorial(7), 5040)
        self.assertEqual(factorial(10), 3628800)
        self.assertEqual(factorial(12), 479001600)

    def test_factorial_boundary(self):
        """Test maximum 64-bit factorial (n = 20)."""
        self.assertEqual(factorial(20), 2432902008176640000)

    def test_factorial_negative_values(self):
        """Negative inputs must raise ValueError."""
        with self.assertRaises(ValueError):
            factorial(-1)
        with self.assertRaises(ValueError):
            factorial(-10)

    def test_factorial_overflow(self):
        """Inputs above 20 exceed 64-bit precision and must raise OverflowError."""
        with self.assertRaises(OverflowError):
            factorial(21)
        with self.assertRaises(OverflowError):
            factorial(100)

    def test_factorial_invalid_types(self):
        """Non-integer types must raise TypeError."""
        with self.assertRaises(TypeError):
            factorial(5.0)  # type: ignore
        with self.assertRaises(TypeError):
            factorial("5")  # type: ignore
        with self.assertRaises(TypeError):
            factorial(None)  # type: ignore
        with self.assertRaises(TypeError):
            factorial(True)  # type: ignore
        with self.assertRaises(TypeError):
            factorial(False)  # type: ignore

    # --------------------------------------------------------------------------
    # Fibonacci Tests
    # --------------------------------------------------------------------------

    def test_fibonacci_base_cases(self):
        """F(0) = 0, F(1) = 1, F(2) = 1."""
        self.assertEqual(fibonacci(0), 0)
        self.assertEqual(fibonacci(1), 1)
        self.assertEqual(fibonacci(2), 1)

    def test_fibonacci_standard_cases(self):
        """Test standard Fibonacci values."""
        self.assertEqual(fibonacci(3), 2)
        self.assertEqual(fibonacci(4), 3)
        self.assertEqual(fibonacci(5), 5)
        self.assertEqual(fibonacci(6), 8)
        self.assertEqual(fibonacci(7), 13)
        self.assertEqual(fibonacci(8), 21)
        self.assertEqual(fibonacci(9), 34)
        self.assertEqual(fibonacci(10), 55)
        self.assertEqual(fibonacci(20), 6765)
        self.assertEqual(fibonacci(30), 832040)

    def test_fibonacci_boundary(self):
        """Test maximum 64-bit Fibonacci value (n = 92)."""
        self.assertEqual(fibonacci(92), 7540113804746346429)

    def test_fibonacci_negative_values(self):
        """Negative indices must raise ValueError."""
        with self.assertRaises(ValueError):
            fibonacci(-1)
        with self.assertRaises(ValueError):
            fibonacci(-10)

    def test_fibonacci_overflow(self):
        """Indices above 92 exceed 64-bit precision and must raise OverflowError."""
        with self.assertRaises(OverflowError):
            fibonacci(93)
        with self.assertRaises(OverflowError):
            fibonacci(1000)

    def test_fibonacci_invalid_types(self):
        """Non-integer types must raise TypeError."""
        with self.assertRaises(TypeError):
            fibonacci(10.5)  # type: ignore
        with self.assertRaises(TypeError):
            fibonacci("10")  # type: ignore
        with self.assertRaises(TypeError):
            fibonacci(None)  # type: ignore
        with self.assertRaises(TypeError):
            fibonacci(True)  # type: ignore
        with self.assertRaises(TypeError):
            fibonacci(False)  # type: ignore


if __name__ == "__main__":
    unittest.main()
