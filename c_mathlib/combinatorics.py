"""
Combinatorics and discrete mathematics module for c-mathlib.

Provides binomial coefficients (combinations), permutations, and Euler's totient function
executed in native C.
"""

import ctypes
from typing import Union
from .core import _lib, _INT64_MIN, _INT64_MAX


# ==============================================================================
# Ctypes Signatures Configuration
# ==============================================================================

_c_math_combinations = _lib.math_combinations
_c_math_combinations.argtypes = [ctypes.c_int, ctypes.c_int]
_c_math_combinations.restype = ctypes.c_longlong

_c_math_permutations = _lib.math_permutations
_c_math_permutations.argtypes = [ctypes.c_int, ctypes.c_int]
_c_math_permutations.restype = ctypes.c_longlong

_c_euler_phi = _lib.euler_phi
_c_euler_phi.argtypes = [ctypes.c_longlong]
_c_euler_phi.restype = ctypes.c_longlong


# ==============================================================================
# Public Combinatorics Functions
# ==============================================================================

def combinations(n: int, k: int) -> int:
    """
    Compute the number of ways to choose k items from n items without repetition (n choose k):
        C(n, k) = n! / (k! * (n - k)!)

    Computed in native C using an incremental multiplicative loop to avoid premature integer overflow.

    Parameters
    ----------
    n : int
        Total number of items (n >= 0).
    k : int
        Number of items to choose (0 <= k <= n).

    Returns
    -------
    int
        Binomial coefficient C(n, k).

    Raises
    ------
    TypeError
        If n or k is not an integer.
    ValueError
        If n < 0, k < 0, or k > n.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"combinations() n must be an integer, got {type(n).__name__}")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"combinations() k must be an integer, got {type(k).__name__}")

    if n < 0:
        raise ValueError(f"combinations() n must be non-negative (>= 0), got {n}")
    if k < 0 or k > n:
        raise ValueError(f"combinations() k must satisfy 0 <= k <= n (got n={n}, k={k})")

    result = _c_math_combinations(n, k)
    if result < 0:
        raise OverflowError("combinations() calculation exceeded 64-bit integer range")
    return int(result)


def permutations(n: int, k: int) -> int:
    """
    Compute the number of k-permutations of n items (ordered arrangements):
        P(n, k) = n! / (n - k)! = n * (n - 1) * ... * (n - k + 1)

    Computed in native C.

    Parameters
    ----------
    n : int
        Total number of items (n >= 0).
    k : int
        Number of items to arrange (0 <= k <= n).

    Returns
    -------
    int
        Permutation count P(n, k).

    Raises
    ------
    TypeError
        If n or k is not an integer.
    ValueError
        If n < 0, k < 0, or k > n.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"permutations() n must be an integer, got {type(n).__name__}")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"permutations() k must be an integer, got {type(k).__name__}")

    if n < 0:
        raise ValueError(f"permutations() n must be non-negative (>= 0), got {n}")
    if k < 0 or k > n:
        raise ValueError(f"permutations() k must satisfy 0 <= k <= n (got n={n}, k={k})")

    result = _c_math_permutations(n, k)
    if result < 0:
        raise OverflowError("permutations() calculation exceeded 64-bit integer range")
    return int(result)


def euler_totient(n: int) -> int:
    """
    Compute Euler's totient function phi(n), counting the number of positive integers
    up to n that are relatively prime (coprime) to n:
        phi(n) = n * Product_{p | n} (1 - 1/p)

    Computed in native C in O(sqrt(n)) time via prime factor decomposition.

    Parameters
    ----------
    n : int
        A positive integer (n >= 1).

    Returns
    -------
    int
        The count of integers in [1, n] coprime to n.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n <= 0.
    OverflowError
        If n exceeds 64-bit signed integer range.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"euler_totient() argument must be an integer, got {type(n).__name__}")
    if not (_INT64_MIN <= n <= _INT64_MAX):
        raise OverflowError("euler_totient() argument exceeds 64-bit signed integer range")
    if n <= 0:
        raise ValueError(f"euler_totient() argument must be a positive integer (>= 1), got {n}")

    result = _c_euler_phi(n)
    return int(result)


# Aliases
nCr = combinations
nPr = permutations

__all__ = [
    "combinations",
    "permutations",
    "euler_totient",
    "nCr",
    "nPr",
]
