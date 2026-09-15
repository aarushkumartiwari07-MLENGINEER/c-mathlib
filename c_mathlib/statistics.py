"""
Statistics and data analysis module for c-mathlib.

Provides descriptive statistics (mean, variance, std dev, median, quantiles),
covariance, Pearson correlation, and Ordinary Least Squares (OLS) linear regression
executed in native C.
"""

import ctypes
from dataclasses import dataclass
from typing import List, Sequence, Union
from .core import _lib


# ==============================================================================
# Ctypes Signatures Configuration
# ==============================================================================

_c_double_p = ctypes.POINTER(ctypes.c_double)

_c_stat_mean = _lib.stat_mean
_c_stat_mean.argtypes = [_c_double_p, ctypes.c_int]
_c_stat_mean.restype = ctypes.c_double

_c_stat_variance = _lib.stat_variance
_c_stat_variance.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_int]
_c_stat_variance.restype = ctypes.c_double

_c_stat_std_dev = _lib.stat_std_dev
_c_stat_std_dev.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_int]
_c_stat_std_dev.restype = ctypes.c_double

_c_stat_median = _lib.stat_median
_c_stat_median.argtypes = [_c_double_p, ctypes.c_int]
_c_stat_median.restype = ctypes.c_double

_c_stat_quantile = _lib.stat_quantile
_c_stat_quantile.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_double]
_c_stat_quantile.restype = ctypes.c_double

_c_stat_covariance = _lib.stat_covariance
_c_stat_covariance.argtypes = [_c_double_p, _c_double_p, ctypes.c_int, ctypes.c_int]
_c_stat_covariance.restype = ctypes.c_double

_c_stat_correlation = _lib.stat_correlation
_c_stat_correlation.argtypes = [_c_double_p, _c_double_p, ctypes.c_int]
_c_stat_correlation.restype = ctypes.c_double

_c_stat_linear_regression = _lib.stat_linear_regression
_c_stat_linear_regression.argtypes = [
    _c_double_p,
    _c_double_p,
    ctypes.c_int,
    _c_double_p,
    _c_double_p,
    _c_double_p,
]
_c_stat_linear_regression.restype = ctypes.c_int


# ==============================================================================
# Helpers & Data Classes
# ==============================================================================

@dataclass(frozen=True)
class LinearRegressionResult:
    """Results from an Ordinary Least Squares (OLS) linear regression model."""
    slope: float
    intercept: float
    r_squared: float

    def predict(self, x: Union[int, float]) -> float:
        """Evaluate fitted model y = slope * x + intercept."""
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"predict() x must be a number, got {type(x).__name__}")
        return self.slope * float(x) + self.intercept

    def __repr__(self) -> str:
        sign = "+" if self.intercept >= 0 else "-"
        return f"LinearRegression(y = {self.slope:g}*x {sign} {abs(self.intercept):g}, R^2 = {self.r_squared:.4f})"


def _parse_numeric_sequence(data: Sequence[Union[int, float]], param_name: str = "data") -> List[float]:
    if not hasattr(data, "__iter__") or isinstance(data, (str, bytes)):
        raise TypeError(f"{param_name} must be an iterable sequence of numbers")
    
    parsed = []
    for val in data:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{param_name} elements must be numeric, got {type(val).__name__}")
        parsed.append(float(val))

    if not parsed:
        raise ValueError(f"{param_name} cannot be empty")
    return parsed


# ==============================================================================
# Public Statistical Functions
# ==============================================================================

def mean(data: Sequence[Union[int, float]]) -> float:
    """
    Compute the arithmetic mean of a dataset in C.

    Parameters
    ----------
    data : Sequence[float]
        Numeric dataset.

    Returns
    -------
    float
        Sample mean sum(x_i) / n.
    """
    parsed = _parse_numeric_sequence(data, "mean() data")
    arr = (ctypes.c_double * len(parsed))(*parsed)
    return float(_c_stat_mean(arr, len(parsed)))


def variance(data: Sequence[Union[int, float]], ddof: int = 1) -> float:
    """
    Compute sample or population variance in C using a two-pass numerically stable algorithm.

    Parameters
    ----------
    data : Sequence[float]
        Numeric dataset.
    ddof : int, default=1
        Delta Degrees of Freedom (1 for sample variance, 0 for population variance).

    Returns
    -------
    float
        Variance sum((x_i - mean)^2) / (n - ddof).
    """
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError(f"ddof must be an integer, got {type(ddof).__name__}")
    parsed = _parse_numeric_sequence(data, "variance() data")
    if len(parsed) <= ddof:
        raise ValueError(f"Sample size ({len(parsed)}) must be strictly greater than ddof ({ddof})")

    arr = (ctypes.c_double * len(parsed))(*parsed)
    return float(_c_stat_variance(arr, len(parsed), ddof))


def std_dev(data: Sequence[Union[int, float]], ddof: int = 1) -> float:
    """
    Compute sample or population standard deviation in C.

    Parameters
    ----------
    data : Sequence[float]
        Numeric dataset.
    ddof : int, default=1
        Delta Degrees of Freedom (1 for sample std dev, 0 for population std dev).

    Returns
    -------
    float
        Standard deviation sqrt(variance).
    """
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError(f"ddof must be an integer, got {type(ddof).__name__}")
    parsed = _parse_numeric_sequence(data, "std_dev() data")
    if len(parsed) <= ddof:
        raise ValueError(f"Sample size ({len(parsed)}) must be strictly greater than ddof ({ddof})")

    arr = (ctypes.c_double * len(parsed))(*parsed)
    return float(_c_stat_std_dev(arr, len(parsed), ddof))


def median(data: Sequence[Union[int, float]]) -> float:
    """
    Compute the median of a dataset in C.

    Parameters
    ----------
    data : Sequence[float]
        Numeric dataset.

    Returns
    -------
    float
        The 50th percentile (median) value.
    """
    parsed = _parse_numeric_sequence(data, "median() data")
    arr = (ctypes.c_double * len(parsed))(*parsed)
    return float(_c_stat_median(arr, len(parsed)))


def quantile(data: Sequence[Union[int, float]], q: float) -> float:
    """
    Compute the q-th quantile of a dataset in C using linear interpolation.

    Parameters
    ----------
    data : Sequence[float]
        Numeric dataset.
    q : float
        Quantile level in the range [0.0, 1.0].

    Returns
    -------
    float
        The interpolated q-th quantile.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise TypeError(f"quantile q must be numeric, got {type(q).__name__}")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"quantile q must be between 0.0 and 1.0, got {q}")

    parsed = _parse_numeric_sequence(data, "quantile() data")
    arr = (ctypes.c_double * len(parsed))(*parsed)
    return float(_c_stat_quantile(arr, len(parsed), float(q)))


def covariance(
    x: Sequence[Union[int, float]],
    y: Sequence[Union[int, float]],
    ddof: int = 1,
) -> float:
    """
    Compute sample or population covariance between two datasets x and y in C.

    Parameters
    ----------
    x : Sequence[float]
        First dataset.
    y : Sequence[float]
        Second dataset.
    ddof : int, default=1
        Delta Degrees of Freedom (1 for sample covariance, 0 for population covariance).

    Returns
    -------
    float
        Covariance sum((x_i - mean_x) * (y_i - mean_y)) / (n - ddof).
    """
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError(f"ddof must be an integer, got {type(ddof).__name__}")
    parsed_x = _parse_numeric_sequence(x, "covariance() x")
    parsed_y = _parse_numeric_sequence(y, "covariance() y")

    if len(parsed_x) != len(parsed_y):
        raise ValueError(f"Datasets must have matching lengths: len(x)={len(parsed_x)} != len(y)={len(parsed_y)}")
    if len(parsed_x) <= ddof:
        raise ValueError(f"Sample size ({len(parsed_x)}) must be strictly greater than ddof ({ddof})")

    arr_x = (ctypes.c_double * len(parsed_x))(*parsed_x)
    arr_y = (ctypes.c_double * len(parsed_y))(*parsed_y)
    return float(_c_stat_covariance(arr_x, arr_y, len(parsed_x), ddof))


def correlation(
    x: Sequence[Union[int, float]],
    y: Sequence[Union[int, float]],
) -> float:
    """
    Compute the Pearson product-moment correlation coefficient r in [-1, 1] in C.

    Parameters
    ----------
    x : Sequence[float]
        First dataset.
    y : Sequence[float]
        Second dataset.

    Returns
    -------
    float
        Pearson r coefficient.
    """
    parsed_x = _parse_numeric_sequence(x, "correlation() x")
    parsed_y = _parse_numeric_sequence(y, "correlation() y")

    if len(parsed_x) != len(parsed_y):
        raise ValueError(f"Datasets must have matching lengths: len(x)={len(parsed_x)} != len(y)={len(parsed_y)}")
    if len(parsed_x) < 2:
        raise ValueError("Correlation requires at least 2 data points")

    arr_x = (ctypes.c_double * len(parsed_x))(*parsed_x)
    arr_y = (ctypes.c_double * len(parsed_y))(*parsed_y)
    return float(_c_stat_correlation(arr_x, arr_y, len(parsed_x)))


def linear_regression(
    x: Sequence[Union[int, float]],
    y: Sequence[Union[int, float]],
) -> LinearRegressionResult:
    """
    Fit an Ordinary Least Squares (OLS) simple linear regression line y = slope * x + intercept in C.

    Parameters
    ----------
    x : Sequence[float]
        Independent variable data.
    y : Sequence[float]
        Dependent variable data.

    Returns
    -------
    LinearRegressionResult
        Dataclass containing slope, intercept, and R^2 coefficient of determination.
    """
    parsed_x = _parse_numeric_sequence(x, "linear_regression() x")
    parsed_y = _parse_numeric_sequence(y, "linear_regression() y")

    if len(parsed_x) != len(parsed_y):
        raise ValueError(f"Datasets must have matching lengths: len(x)={len(parsed_x)} != len(y)={len(parsed_y)}")
    if len(parsed_x) < 2:
        raise ValueError("Linear regression requires at least 2 data points")

    arr_x = (ctypes.c_double * len(parsed_x))(*parsed_x)
    arr_y = (ctypes.c_double * len(parsed_y))(*parsed_y)

    c_slope = ctypes.c_double()
    c_intercept = ctypes.c_double()
    c_r_squared = ctypes.c_double()

    status = _c_stat_linear_regression(
        arr_x,
        arr_y,
        len(parsed_x),
        ctypes.byref(c_slope),
        ctypes.byref(c_intercept),
        ctypes.byref(c_r_squared),
    )

    if status < 0:
        raise ValueError("Linear regression failed: independent variable x has zero variance")

    return LinearRegressionResult(
        slope=float(c_slope.value),
        intercept=float(c_intercept.value),
        r_squared=float(c_r_squared.value),
    )


__all__ = [
    "mean",
    "variance",
    "std_dev",
    "median",
    "quantile",
    "covariance",
    "correlation",
    "linear_regression",
    "LinearRegressionResult",
]
