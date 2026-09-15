import ctypes
import os
import sys
from pathlib import Path


def _find_and_load_library() -> ctypes.CDLL:
    """
    Locates and loads the compiled C shared library/DLL.
    
    Search strategy:
    1. Checks the current package directory (`c_mathlib/`).
    2. Checks the project root directory.
    3. Checks platform-specific file extensions:
       - Windows: `libmymath.dll`, `mymath.dll`
       - Linux: `libmymath.so`
       - macOS: `libmymath.dylib`
    """
    pkg_dir = Path(__file__).resolve().parent
    project_root = pkg_dir.parent

    # Determine potential library filenames based on OS platform
    if sys.platform.startswith("win"):
        lib_names = ["libmymath.dll", "mymath.dll"]
    elif sys.platform.startswith("darwin"):
        lib_names = ["libmymath.dylib", "mymath.dylib"]
    else:
        lib_names = ["libmymath.so", "mymath.so"]

    # Search candidates: package directory first, then root/build directories
    search_dirs = [
        pkg_dir,
        project_root,
        project_root / "build",
        project_root / "dist",
    ]

    for directory in search_dirs:
        for name in lib_names:
            candidate = directory / name
            if candidate.is_file():
                try:
                    return ctypes.CDLL(str(candidate))
                except OSError as err:
                    raise OSError(
                        f"Failed to load native library at '{candidate}': {err}"
                    ) from err

    tried_locations = [str(d / name) for d in search_dirs for name in lib_names]
    raise FileNotFoundError(
        "Could not find compiled native library 'libmymath'.\n"
        f"Searched in:\n  - " + "\n  - ".join(tried_locations) + "\n\n"
        "To compile on Windows using GCC:\n"
        "  gcc -O2 -shared -o c_mathlib/libmymath.dll src/mymath.c\n"
    )


# Load the compiled native C library
_lib = _find_and_load_library()

# ==============================================================================
# Ctypes Signature Configuration
# ==============================================================================
# In Python's ctypes, setting `argtypes` and `restype` is critical:
#
# 1. `argtypes`: Defines the sequence of expected argument types for the C function.
#    Without `argtypes`, ctypes assumes standard integer/pointer types but won't
#    typecheck arguments in Python before calling the foreign function, which can
#    lead to memory corruption or crashes if mismatched arguments are passed.
#
# 2. `restype`: Defines the return type of the C function.
#    By default, ctypes assumes all C functions return a 32-bit signed C `int`.
#    Explicitly declaring `restype` guarantees proper type conversion from C
#    machine registers into Python objects across different CPU architectures.
# ==============================================================================

_c_is_prime = _lib.is_prime
_c_is_prime.argtypes = [ctypes.c_int]
_c_is_prime.restype = ctypes.c_int

_c_gcd = _lib.gcd
_c_gcd.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
_c_gcd.restype = ctypes.c_longlong

_c_lcm = _lib.lcm
_c_lcm.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
_c_lcm.restype = ctypes.c_longlong

_c_factorial = _lib.factorial
_c_factorial.argtypes = [ctypes.c_int]
_c_factorial.restype = ctypes.c_longlong

_c_fibonacci = _lib.fibonacci
_c_fibonacci.argtypes = [ctypes.c_int]
_c_fibonacci.restype = ctypes.c_longlong

_c_mod_pow = _lib.mod_pow
_c_mod_pow.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
_c_mod_pow.restype = ctypes.c_longlong

_c_extended_gcd = _lib.extended_gcd
_c_extended_gcd.argtypes = [
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_longlong),
    ctypes.POINTER(ctypes.c_longlong),
]
_c_extended_gcd.restype = ctypes.c_longlong

_c_mod_inverse = _lib.mod_inverse
_c_mod_inverse.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
_c_mod_inverse.restype = ctypes.c_longlong

_INT64_MIN = -9223372036854775808
_INT64_MAX = 9223372036854775807




def is_prime(n: int) -> bool:
    """
    Check whether an integer n is a prime number.

    This function delegates prime testing to the compiled native C library
    via Python's `ctypes` foreign function interface.

    Parameters
    ----------
    n : int
        The integer to test for primality.

    Returns
    -------
    bool
        True if n is prime, False otherwise.

    Raises
    ------
    TypeError
        If n is not an integer.
    OverflowError
        If n is outside the 32-bit signed integer range.
    """
    # Reject non-integers (note: in Python bool is a subclass of int, so we reject bool explicitly)
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"is_prime() argument must be an integer, got {type(n).__name__}")

    # Ensure n fits within standard 32-bit signed integer range (C int)
    if not (-2147483648 <= n <= 2147483647):
        raise OverflowError(
            f"is_prime() argument {n} exceeds 32-bit signed integer range (-2147483648 to 2147483647)"
        )

    # Call the native C function and convert the integer return code (0 or 1) to a Python boolean
    result = _c_is_prime(n)
    return bool(result)


def gcd(a: int, b: int) -> int:
    """
    Compute the Greatest Common Divisor (GCD) of two integers.

    Uses the Euclidean algorithm executed in native C.
    The result is always non-negative.

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    int
        The non-negative greatest common divisor of a and b.

    Raises
    ------
    TypeError
        If either a or b is not an integer.
    OverflowError
        If either a or b exceeds 64-bit signed integer range.
    """
    if not isinstance(a, int) or isinstance(a, bool):
        raise TypeError(f"gcd() arguments must be integers, got {type(a).__name__}")
    if not isinstance(b, int) or isinstance(b, bool):
        raise TypeError(f"gcd() arguments must be integers, got {type(b).__name__}")

    if not (_INT64_MIN <= a <= _INT64_MAX) or not (_INT64_MIN <= b <= _INT64_MAX):
        raise OverflowError("gcd() arguments must fit within 64-bit signed integer range")

    return int(_c_gcd(a, b))


def lcm(a: int, b: int) -> int:
    """
    Compute the Least Common Multiple (LCM) of two integers.

    Computed in native C using the relationship lcm(a, b) = (|a| / gcd(a, b)) * |b|.
    The result is always non-negative. If either a or b is 0, the result is 0.

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    int
        The non-negative least common multiple of a and b.

    Raises
    ------
    TypeError
        If either a or b is not an integer.
    OverflowError
        If either a or b exceeds 64-bit signed integer range.
    """
    if not isinstance(a, int) or isinstance(a, bool):
        raise TypeError(f"lcm() arguments must be integers, got {type(a).__name__}")
    if not isinstance(b, int) or isinstance(b, bool):
        raise TypeError(f"lcm() arguments must be integers, got {type(b).__name__}")

    if not (_INT64_MIN <= a <= _INT64_MAX) or not (_INT64_MIN <= b <= _INT64_MAX):
        raise OverflowError("lcm() arguments must fit within 64-bit signed integer range")

    return int(_c_lcm(a, b))


def factorial(n: int) -> int:
    """
    Compute the factorial of a non-negative integer n (n!).

    Computed in native C. Supports values up to n = 20 (the maximum value
    representable in a 64-bit signed integer: 20! = 2,432,902,008,176,640,000).

    Parameters
    ----------
    n : int
        A non-negative integer (0 <= n <= 20).

    Returns
    -------
    int
        The factorial of n (0! = 1).

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n is negative.
    OverflowError
        If n > 20 (exceeds 64-bit signed integer precision).
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"factorial() argument must be an integer, got {type(n).__name__}")

    if n < 0:
        raise ValueError("factorial() not defined for negative integers")

    if n > 20:
        raise OverflowError("factorial() input exceeds 64-bit integer range (n <= 20)")

    return int(_c_factorial(n))


def fibonacci(n: int) -> int:
    """
    Compute the n-th Fibonacci number in the standard 0-indexed Fibonacci sequence.

    Computed iteratively in native C with O(n) time and O(1) space.
    Sequence convention: F(0) = 0, F(1) = 1, F(2) = 1, F(3) = 2, ..., F(10) = 55.
    Supports indices up to n = 92 (F(92) = 7,540,113,804,746,346,429).

    Parameters
    ----------
    n : int
        A non-negative sequence index (0 <= n <= 92).

    Returns
    -------
    int
        The n-th Fibonacci number.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n is negative.
    OverflowError
        If n > 92 (exceeds 64-bit signed integer precision).
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"fibonacci() argument must be an integer, got {type(n).__name__}")

    if n < 0:
        raise ValueError("fibonacci() not defined for negative index")

    if n > 92:
        raise OverflowError("fibonacci() index exceeds 64-bit integer range (n <= 92)")

    return int(_c_fibonacci(n))


def mod_pow(base: int, exponent: int, modulus: int) -> int:
    """
    Compute (base ^ exponent) % modulus using binary exponentiation (exponentiation by squaring).

    Executed in native C in O(log exponent) time complexity.
    Exponentiation by squaring decomposes the exponent into powers of two,
    avoiding both astronomical intermediate values and slow O(exponent) naive multiplication loops.

    Parameters
    ----------
    base : int
        The base integer.
    exponent : int
        The non-negative exponent (power).
    modulus : int
        The positive integer modulus (> 0).

    Returns
    -------
    int
        (base ^ exponent) % modulus in the range [0, modulus - 1].

    Raises
    ------
    TypeError
        If any argument is not an integer.
    ValueError
        If modulus <= 0 or exponent < 0.
    OverflowError
        If any argument exceeds the 64-bit signed integer range.
    """
    for name, val in [("base", base), ("exponent", exponent), ("modulus", modulus)]:
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"mod_pow() {name} must be an integer, got {type(val).__name__}")
        if not (_INT64_MIN <= val <= _INT64_MAX):
            raise OverflowError(f"mod_pow() {name} must fit within 64-bit signed integer range")

    if modulus <= 0:
        raise ValueError(f"mod_pow() modulus must be a positive integer (> 0), got {modulus}")
    if exponent < 0:
        raise ValueError(f"mod_pow() exponent must be non-negative (>= 0), got {exponent}")

    return int(_c_mod_pow(base, exponent, modulus))


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Compute the Extended Euclidean Algorithm for two integers a and b.

    Finds the greatest common divisor g = gcd(|a|, |b|) along with Bézout coefficients
    x and y satisfying the identity:
        a * x + b * y = g

    Parameters
    ----------
    a : int
        First integer.
    b : int
        Second integer.

    Returns
    -------
    tuple[int, int, int]
        A 3-tuple (g, x, y) where g is the non-negative GCD, and x, y are Bézout coefficients.

    Raises
    ------
    TypeError
        If either a or b is not an integer.
    OverflowError
        If either a or b exceeds the 64-bit signed integer range.
    """
    for name, val in [("a", a), ("b", b)]:
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"extended_gcd() argument {name} must be an integer, got {type(val).__name__}")
        if not (_INT64_MIN <= val <= _INT64_MAX):
            raise OverflowError(f"extended_gcd() argument {name} must fit within 64-bit signed integer range")

    c_x = ctypes.c_longlong()
    c_y = ctypes.c_longlong()
    g = _c_extended_gcd(a, b, ctypes.byref(c_x), ctypes.byref(c_y))
    return (int(g), int(c_x.value), int(c_y.value))


def mod_inverse(a: int, modulus: int) -> int:
    """
    Compute the modular multiplicative inverse of a modulo m, such that:
        (a * x) = 1 (mod m)

    Computed in native C using the Extended Euclidean Algorithm rather than brute force.
    The inverse exists if and only if a and modulus are coprime (i.e., gcd(a, modulus) == 1).

    Parameters
    ----------
    a : int
        The integer whose modular inverse is sought.
    modulus : int
        The integer modulus (must be > 1).

    Returns
    -------
    int
        The unique modular multiplicative inverse x in the range [0, modulus - 1].

    Raises
    ------
    TypeError
        If either a or modulus is not an integer.
    ValueError
        If modulus <= 1 or if no modular inverse exists (gcd(a, modulus) != 1).
    OverflowError
        If either a or modulus exceeds the 64-bit signed integer range.
    """
    for name, val in [("a", a), ("modulus", modulus)]:
        if not isinstance(val, int) or isinstance(val, bool):
            raise TypeError(f"mod_inverse() argument {name} must be an integer, got {type(val).__name__}")
        if not (_INT64_MIN <= val <= _INT64_MAX):
            raise OverflowError(f"mod_inverse() argument {name} must fit within 64-bit signed integer range")

    if modulus <= 1:
        raise ValueError(f"mod_inverse() modulus must be greater than 1, got {modulus}")

    result = _c_mod_inverse(a, modulus)
    if result == -1:
        raise ValueError(
            f"mod_inverse() does not exist: {a} and {modulus} are not coprime (gcd != 1)"
        )

    return int(result)



