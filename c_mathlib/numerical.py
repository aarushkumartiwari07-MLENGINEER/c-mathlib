"""
Numerical mathematics module for c-mathlib.

Provides root finding (bisection), numerical integration (Simpson's 1/3 rule),
and numerical differentiation (central difference) executed in native C.
"""

import ctypes
from typing import List, Sequence, Tuple, Union
from .core import _lib


# ==============================================================================
# Ctypes Signature Configuration for Numerical Routines
# ==============================================================================

_c_double_p = ctypes.POINTER(ctypes.c_double)

_c_eval_math_func = _lib.eval_math_func
_c_eval_math_func.argtypes = [ctypes.c_int, ctypes.c_double]
_c_eval_math_func.restype = ctypes.c_double

_c_eval_polynomial = _lib.eval_polynomial
_c_eval_polynomial.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_double]
_c_eval_polynomial.restype = ctypes.c_double

_c_bisection_method = _lib.bisection_method
_c_bisection_method.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    _c_double_p,
]
_c_bisection_method.restype = ctypes.c_int

_c_bisection_poly = _lib.bisection_poly
_c_bisection_poly.argtypes = [
    _c_double_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    _c_double_p,
]
_c_bisection_poly.restype = ctypes.c_int

_c_simpson_rule = _lib.simpson_rule
_c_simpson_rule.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    _c_double_p,
]
_c_simpson_rule.restype = ctypes.c_int

_c_simpson_poly = _lib.simpson_poly
_c_simpson_poly.argtypes = [
    _c_double_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    _c_double_p,
]
_c_simpson_poly.restype = ctypes.c_int

_c_numerical_derivative = _lib.numerical_derivative
_c_numerical_derivative.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    _c_double_p,
]
_c_numerical_derivative.restype = ctypes.c_int

_c_numerical_derivative_poly = _lib.numerical_derivative_poly
_c_numerical_derivative_poly.argtypes = [
    _c_double_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    _c_double_p,
]
_c_numerical_derivative_poly.restype = ctypes.c_int


# ==============================================================================
# Built-in Function ID Mappings
# ==============================================================================

_FUNC_NAME_TO_ID = {
    "x": 0,
    "identity": 0,
    "x^2": 1,
    "square": 1,
    "x^3": 2,
    "cube": 2,
    "x^2 - 4": 3,
    "x^2-4": 3,
    "quadratic": 3,
    "x^3 - x - 2": 4,
    "x^3-x-2": 4,
    "cubic_test": 4,
    "sin": 5,
    "sin(x)": 5,
    "cos": 6,
    "cos(x)": 6,
    "exp": 7,
    "exp(x)": 7,
    "log": 8,
    "ln": 8,
    "log(x)": 8,
    "1/x": 9,
    "reciprocal": 9,
}


class Polynomial:
    """
    Representation of a single-variable polynomial P(x) = c_0 + c_1*x + ... + c_n*x^n.

    Parameters
    ----------
    coeffs : Sequence[float]
        Coefficients in ascending degree order: [c_0, c_1, ..., c_n].
    """

    def __init__(self, coeffs: Sequence[Union[int, float]]):
        if not hasattr(coeffs, "__iter__") or isinstance(coeffs, (str, bytes)):
            raise TypeError("Polynomial coefficients must be an iterable sequence of numbers")
        
        parsed = []
        for c in coeffs:
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise TypeError(f"Polynomial coefficient must be numeric, got {type(c).__name__}")
            parsed.append(float(c))
        
        if not parsed:
            raise ValueError("Polynomial must contain at least one coefficient")
        
        self._coeffs = parsed

    @property
    def coeffs(self) -> List[float]:
        return list(self._coeffs)

    @property
    def degree(self) -> int:
        return len(self._coeffs) - 1

    def __repr__(self) -> str:
        terms = []
        for deg, c in enumerate(self._coeffs):
            if c == 0 and len(self._coeffs) > 1:
                continue
            if deg == 0:
                terms.append(f"{c:g}")
            elif deg == 1:
                terms.append(f"{c:g}*x")
            else:
                terms.append(f"{c:g}*x^{deg}")
        return " + ".join(terms) if terms else "0"


def _resolve_target(
    func: Union[str, Sequence[Union[int, float]], Polynomial]
) -> Tuple[bool, Union[int, List[float]]]:
    """
    Resolves the input function into either a built-in function ID or a polynomial coefficient array.
    
    Returns
    -------
    Tuple[bool, Union[int, List[float]]]
        (is_poly, payload) where payload is func_id if is_poly is False, else list of floats.
    """
    if isinstance(func, Polynomial):
        return True, func.coeffs

    if isinstance(func, str):
        normalized = func.strip().lower()
        if normalized in _FUNC_NAME_TO_ID:
            return False, _FUNC_NAME_TO_ID[normalized]
        valid_names = ", ".join(sorted(set(_FUNC_NAME_TO_ID.keys())))
        raise ValueError(
            f"Unknown built-in function name '{func}'. Supported names: {valid_names}\n"
            "Or provide a list/tuple of polynomial coefficients, e.g. [-2, -1, 0, 1] for x^3 - x - 2."
        )

    if hasattr(func, "__iter__") and not isinstance(func, (bytes, bytearray)):
        poly = Polynomial(func)
        return True, poly.coeffs

    raise TypeError(
        f"Unsupported function representation of type {type(func).__name__}. "
        "Expected function name string (e.g. 'x^2', 'sin') or polynomial coefficient sequence."
    )


def evaluate(
    func: Union[str, Sequence[Union[int, float]], Polynomial],
    x: Union[int, float],
) -> float:
    """
    Evaluate a mathematical function or polynomial at point x in native C.

    Parameters
    ----------
    func : str or Sequence[float] or Polynomial
        The function identifier or polynomial coefficients.
    x : float or int
        The evaluation point.

    Returns
    -------
    float
        The evaluated value f(x).
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"evaluate() point x must be a number, got {type(x).__name__}")

    is_poly, payload = _resolve_target(func)
    x_val = float(x)

    if is_poly:
        coeffs_array = (ctypes.c_double * len(payload))(*payload)
        return float(_c_eval_polynomial(coeffs_array, len(payload), x_val))
    else:
        return float(_c_eval_math_func(int(payload), x_val))


def bisection(
    func: Union[str, Sequence[Union[int, float]], Polynomial],
    a: Union[int, float],
    b: Union[int, float],
    tol: float = 1e-7,
    max_iter: int = 100,
) -> float:
    """
    Find a root of f(x) = 0 on interval [a, b] using the bisection method.

    Executed in native C. The bisection method iteratively halves the bracket [a, b],
    guaranteeing linear convergence when f(a) and f(b) have opposite signs (Intermediate Value Theorem).

    Parameters
    ----------
    func : str or Sequence[float] or Polynomial
        The mathematical function or polynomial to solve.
    a : float or int
        Left interval boundary.
    b : float or int
        Right interval boundary.
    tol : float, default=1e-7
        Convergence tolerance on interval width and residual.
    max_iter : int, default=100
        Maximum number of bisection iterations.

    Returns
    -------
    float
        Approximate root c in [a, b] such that |f(c)| < tol or |b - a|/2 < tol.

    Raises
    ------
    TypeError
        If bounds, tolerance, or iteration count have invalid types.
    ValueError
        If tol <= 0, max_iter <= 0, or if root is not bracketed (f(a) and f(b) have the same sign).
    RuntimeError
        If maximum iterations are reached without converging to tolerance.
    """
    for name, val in [("a", a), ("b", b), ("tol", tol)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"bisection() parameter '{name}' must be numeric, got {type(val).__name__}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError(f"bisection() max_iter must be an integer, got {type(max_iter).__name__}")

    if tol <= 0:
        raise ValueError(f"bisection() tolerance tol must be positive (> 0), got {tol}")
    if max_iter <= 0:
        raise ValueError(f"bisection() max_iter must be positive (> 0), got {max_iter}")

    is_poly, payload = _resolve_target(func)
    c_a = float(a)
    c_b = float(b)
    c_tol = float(tol)
    c_max_iter = int(max_iter)
    c_root = ctypes.c_double()

    if is_poly:
        coeffs_array = (ctypes.c_double * len(payload))(*payload)
        status = _c_bisection_poly(coeffs_array, len(payload), c_a, c_b, c_tol, c_max_iter, ctypes.byref(c_root))
    else:
        status = _c_bisection_method(int(payload), c_a, c_b, c_tol, c_max_iter, ctypes.byref(c_root))

    if status == -1:
        raise ValueError(
            f"Root is not bracketed on interval [{a}, {b}]: f(a) and f(b) have the same sign"
        )
    if status == -2:
        raise RuntimeError(
            f"Bisection failed to converge within {max_iter} iterations (tol={tol})"
        )
    if status < 0:
        raise ValueError(f"Bisection failed with status code {status}")

    return float(c_root.value)


def simpson(
    func: Union[str, Sequence[Union[int, float]], Polynomial],
    a: Union[int, float],
    b: Union[int, float],
    n: int = 100,
) -> float:
    """
    Approximate the definite integral of f(x) from a to b using composite Simpson's 1/3 rule.

    Executed in native C. Simpson's rule approximates the integrand using quadratic polynomials
    across consecutive subintervals, achieving an O(h^4) truncation error for smooth functions.

    Parameters
    ----------
    func : str or Sequence[float] or Polynomial
        The integrand function or polynomial.
    a : float or int
        Lower limit of integration.
    b : float or int
        Upper limit of integration.
    n : int, default=100
        Number of subdivisions. Must be a positive even integer (n >= 2).

    Returns
    -------
    float
        The approximate definite integral value.

    Raises
    ------
    TypeError
        If bounds or subdivision count have invalid types.
    ValueError
        If n is not an even positive integer (n >= 2).
    """
    for name, val in [("a", a), ("b", b)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"simpson() integration limit '{name}' must be numeric, got {type(val).__name__}")
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"simpson() subdivision count n must be an integer, got {type(n).__name__}")

    if n < 2 or (n % 2 != 0):
        raise ValueError(f"simpson() subdivision count n must be a positive even integer (>= 2), got {n}")

    is_poly, payload = _resolve_target(func)
    c_a = float(a)
    c_b = float(b)
    c_n = int(n)
    c_result = ctypes.c_double()

    if is_poly:
        coeffs_array = (ctypes.c_double * len(payload))(*payload)
        status = _c_simpson_poly(coeffs_array, len(payload), c_a, c_b, c_n, ctypes.byref(c_result))
    else:
        status = _c_simpson_rule(int(payload), c_a, c_b, c_n, ctypes.byref(c_result))

    if status < 0:
        raise ValueError(f"simpson() integration failed with status code {status}")

    return float(c_result.value)


def derivative(
    func: Union[str, Sequence[Union[int, float]], Polynomial],
    x: Union[int, float],
    h: float = 1e-5,
) -> float:
    """
    Approximate the first derivative f'(x) using the central difference formula:
        f'(x) ~ [f(x + h) - f(x - h)] / (2*h)

    Executed in native C. The central difference formula achieves an O(h^2) truncation error,
    providing higher accuracy than forward or backward differences.

    Note on step size h:
    -------------------
    Choosing h involves balancing truncation error (which grows with larger h) against
    floating-point cancellation/roundoff error (which grows with smaller h).
    A default of h = 1e-5 provides optimal precision for double-precision floats.

    Parameters
    ----------
    func : str or Sequence[float] or Polynomial
        The function or polynomial to differentiate.
    x : float or int
        The point at which to evaluate the derivative.
    h : float, default=1e-5
        The finite difference step size (must be > 0).

    Returns
    -------
    float
        The approximate first derivative f'(x).

    Raises
    ------
    TypeError
        If x or h have invalid types.
    ValueError
        If h <= 0.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"derivative() point x must be numeric, got {type(x).__name__}")
    if isinstance(h, bool) or not isinstance(h, (int, float)):
        raise TypeError(f"derivative() step size h must be numeric, got {type(h).__name__}")

    if h <= 0:
        raise ValueError(f"derivative() step size h must be positive (> 0), got {h}")

    is_poly, payload = _resolve_target(func)
    c_x = float(x)
    c_h = float(h)
    c_result = ctypes.c_double()

    if is_poly:
        coeffs_array = (ctypes.c_double * len(payload))(*payload)
        status = _c_numerical_derivative_poly(coeffs_array, len(payload), c_x, c_h, ctypes.byref(c_result))
    else:
        status = _c_numerical_derivative(int(payload), c_x, c_h, ctypes.byref(c_result))

    if status < 0:
        raise ValueError(f"derivative() differentiation failed with status code {status}")

    return float(c_result.value)


__all__ = [
    "Polynomial",
    "evaluate",
    "bisection",
    "simpson",
    "derivative",
]
