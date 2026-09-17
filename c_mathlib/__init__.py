"""
c-mathlib: A lightweight mathematical library with a C core and Python ctypes interface.
"""

from .core import (
    is_prime,
    gcd,
    lcm,
    factorial,
    fibonacci,
    mod_pow,
    extended_gcd,
    mod_inverse,
)
from .combinatorics import (
    combinations,
    permutations,
    euler_totient,
    nCr,
    nPr,
)
from .special import (
    gamma,
    lgamma,
    beta,
    erf,
    erfc,
    sigmoid,
    softmax,
)
from . import numerical
from . import linear_algebra
from . import statistics
from . import combinatorics
from . import special

__version__ = "0.1.0"
__all__ = [
    "is_prime",
    "gcd",
    "lcm",
    "factorial",
    "fibonacci",
    "mod_pow",
    "extended_gcd",
    "mod_inverse",
    "combinations",
    "permutations",
    "euler_totient",
    "nCr",
    "nPr",
    "gamma",
    "lgamma",
    "beta",
    "erf",
    "erfc",
    "sigmoid",
    "softmax",
    "numerical",
    "linear_algebra",
    "statistics",
    "combinatorics",
    "special",
]







