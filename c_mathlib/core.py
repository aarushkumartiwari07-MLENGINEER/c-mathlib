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
    """
    # Reject non-integers (note: in Python bool is a subclass of int, so we reject bool explicitly)
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"is_prime() argument must be an integer, got {type(n).__name__}")

    # Call the native C function and convert the integer return code (0 or 1) to a Python boolean
    result = _c_is_prime(n)
    return bool(result)
