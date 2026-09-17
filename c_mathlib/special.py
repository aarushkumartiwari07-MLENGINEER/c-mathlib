"""
Special Mathematical Functions and ML Activation Primitives.

Provides native C implementations of:
- Gamma function Gamma(x) and log-Gamma ln|Gamma(x)|
- Beta function B(a, b)
- Gauss Error Function erf(x) and complementary erfc(x)
- Logistic sigmoid activation function
- Numerically stable softmax probability distribution
"""

import ctypes
from typing import List, Sequence, Union
from c_mathlib.core import _lib

_c_double_p = ctypes.POINTER(ctypes.c_double)

# ==============================================================================
# ctypes Function Signatures
# ==============================================================================

_c_math_lgamma = _lib.math_lgamma
_c_math_lgamma.argtypes = [ctypes.c_double, _c_double_p]
_c_math_lgamma.restype = ctypes.c_int

_c_math_gamma = _lib.math_gamma
_c_math_gamma.argtypes = [ctypes.c_double, _c_double_p]
_c_math_gamma.restype = ctypes.c_int

_c_math_beta = _lib.math_beta
_c_math_beta.argtypes = [ctypes.c_double, ctypes.c_double, _c_double_p]
_c_math_beta.restype = ctypes.c_int

_c_math_erf = _lib.math_erf
_c_math_erf.argtypes = [ctypes.c_double, _c_double_p]
_c_math_erf.restype = ctypes.c_int

_c_math_erfc = _lib.math_erfc
_c_math_erfc.argtypes = [ctypes.c_double, _c_double_p]
_c_math_erfc.restype = ctypes.c_int

_c_math_sigmoid = _lib.math_sigmoid
_c_math_sigmoid.argtypes = [ctypes.c_double]
_c_math_sigmoid.restype = ctypes.c_double

_c_math_softmax = _lib.math_softmax
_c_math_softmax.argtypes = [_c_double_p, ctypes.c_int, _c_double_p]
_c_math_softmax.restype = ctypes.c_int


# ==============================================================================
# Public Python API
# ==============================================================================

def gamma(x: Union[int, float]) -> float:
    """
    Compute the Gamma function Gamma(x) via Lanczos approximation in native C.

    Parameters
    ----------
    x : float or int
        Input real number. Must not be a non-positive integer (0, -1, -2, ...).

    Returns
    -------
    float
        The Gamma value Gamma(x).

    Raises
    ------
    TypeError
        If x is not a numeric value.
    ValueError
        If x is a pole/singularity (0 or negative integer).
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"gamma() argument must be numeric, got {type(x).__name__}")

    out = ctypes.c_double()
    status = _c_math_gamma(float(x), ctypes.byref(out))
    if status != 0:
        raise ValueError(f"gamma() is undefined at singularity x = {x}")
    return out.value


def lgamma(x: Union[int, float]) -> float:
    """
    Compute the natural logarithm of the absolute value of Gamma(x): ln|Gamma(x)|.

    Parameters
    ----------
    x : float or int
        Input real number. Must not be a non-positive integer.

    Returns
    -------
    float
        The value ln|Gamma(x)|.

    Raises
    ------
    TypeError
        If x is not a numeric value.
    ValueError
        If x is 0 or a negative integer.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"lgamma() argument must be numeric, got {type(x).__name__}")

    out = ctypes.c_double()
    status = _c_math_lgamma(float(x), ctypes.byref(out))
    if status != 0:
        raise ValueError(f"lgamma() is undefined at singularity x = {x}")
    return out.value


def beta(a: Union[int, float], b: Union[int, float]) -> float:
    """
    Compute the Euler Beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).

    Parameters
    ----------
    a : float or int
        First parameter (must not be a non-positive integer).
    b : float or int
        Second parameter (must not be a non-positive integer).

    Returns
    -------
    float
        Computed Beta function value.

    Raises
    ------
    TypeError
        If a or b are non-numeric.
    ValueError
        If parameters hit singularities.
    """
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        raise TypeError(f"beta() argument 'a' must be numeric, got {type(a).__name__}")
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise TypeError(f"beta() argument 'b' must be numeric, got {type(b).__name__}")

    out = ctypes.c_double()
    status = _c_math_beta(float(a), float(b), ctypes.byref(out))
    if status != 0:
        raise ValueError(f"beta() undefined for parameters a={a}, b={b}")
    return out.value


def erf(x: Union[int, float]) -> float:
    """
    Compute the Gauss Error Function erf(x) = (2 / sqrt(pi)) * int_0^x exp(-t^2) dt.

    Parameters
    ----------
    x : float or int
        Input real number.

    Returns
    -------
    float
        Computed erf(x) value in (-1, 1).
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"erf() argument must be numeric, got {type(x).__name__}")

    out = ctypes.c_double()
    status = _c_math_erf(float(x), ctypes.byref(out))
    if status != 0:
        raise ValueError(f"erf() calculation error for x = {x}")
    return out.value


def erfc(x: Union[int, float]) -> float:
    """
    Compute the Complementary Error Function erfc(x) = 1 - erf(x).

    Parameters
    ----------
    x : float or int
        Input real number.

    Returns
    -------
    float
        Computed erfc(x) value in (0, 2).
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"erfc() argument must be numeric, got {type(x).__name__}")

    out = ctypes.c_double()
    status = _c_math_erfc(float(x), ctypes.byref(out))
    if status != 0:
        raise ValueError(f"erfc() calculation error for x = {x}")
    return out.value


def sigmoid(x: Union[int, float]) -> float:
    """
    Compute the logistic sigmoid activation function sigma(x) = 1 / (1 + exp(-x)).

    Numerically stable for extreme positive and negative values.

    Parameters
    ----------
    x : float or int
        Input scalar.

    Returns
    -------
    float
        Computed sigmoid value in (0, 1).
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"sigmoid() argument must be numeric, got {type(x).__name__}")

    return _c_math_sigmoid(float(x))


def softmax(x: Sequence[Union[int, float]]) -> List[float]:
    """
    Compute the numerically stable softmax probability distribution over an array.

    Subtracts the maximum logit before exponentiation to prevent floating-point overflow.

    Parameters
    ----------
    x : Sequence[float]
        1D sequence of logits / scores.

    Returns
    -------
    List[float]
        Probability distribution vector that sums to 1.0.

    Raises
    ------
    TypeError
        If x is not a sequence or contains non-numeric elements.
    ValueError
        If sequence x is empty.
    """
    if not hasattr(x, "__len__") or isinstance(x, (str, bytes)):
        raise TypeError(f"softmax() requires a sequence of numbers, got {type(x).__name__}")
    if len(x) == 0:
        raise ValueError("softmax() requires a non-empty sequence")

    float_vals = []
    for item in x:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"softmax() sequence elements must be numeric, got {type(item).__name__}")
        float_vals.append(float(item))

    n = len(float_vals)
    c_in = (ctypes.c_double * n)(*float_vals)
    c_out = (ctypes.c_double * n)()

    status = _c_math_softmax(c_in, n, c_out)
    if status != 0:
        raise ValueError("softmax() computation failed")

    return list(c_out)


__all__ = [
    "gamma",
    "lgamma",
    "beta",
    "erf",
    "erfc",
    "sigmoid",
    "softmax",
]
