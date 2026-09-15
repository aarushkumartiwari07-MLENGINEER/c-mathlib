"""
c-mathlib: A lightweight mathematical library with a C core and Python ctypes interface.
"""

from .core import is_prime, gcd, lcm

__version__ = "0.1.0"
__all__ = ["is_prime", "gcd", "lcm"]

