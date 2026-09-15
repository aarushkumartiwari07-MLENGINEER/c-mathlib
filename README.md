# c-mathlib

A lightweight mathematical library whose core implementations are written in native **C** and exposed to **Python** via `ctypes`.

---

## Why This Project Exists

`c-mathlib` is built to explore and demonstrate how Python interfaces with native C code. Rather than relying on heavyweight wrapper frameworks or high-level abstractions, this project provides a clean, transparent foundation for understanding:

1. How native C code is written, compiled, and exported as a dynamic shared library (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS).
2. How Python's `ctypes` foreign function interface (FFI) dynamically loads the library and marshals data across the language boundary.
3. How a native C extension project can be structured as an installable, reusable Python package.

---

## Supported Mathematical Functions

| Function | Description | Supported Input Range | Edge Case / Invalid Handling |
| :--- | :--- | :--- | :--- |
| `is_prime(n)` | Check whether an integer $n$ is prime | 32-bit signed integers | Negative, $0, 1 \rightarrow \text{False}$ |
| `gcd(a, b)` | Greatest Common Divisor (Euclidean algorithm) | 64-bit signed integers | Returns non-negative; $\gcd(0,0)=0$ |
| `lcm(a, b)` | Least Common Multiple (via GCD) | 64-bit signed integers | Returns non-negative; $\text{lcm}(0,x)=0$ |
| `factorial(n)` | Factorial of non-negative integer $n!$ | $0 \le n \le 20$ (64-bit limit) | $n < 0 \rightarrow \text{ValueError}$, $n > 20 \rightarrow \text{OverflowError}$ |
| `fibonacci(n)` | $n$-th Fibonacci sequence number ($F_0=0, F_1=1$) | $0 \le n \le 92$ (64-bit limit) | $n < 0 \rightarrow \text{ValueError}$, $n > 92 \rightarrow \text{OverflowError}$ |

---

## Architecture

```text
Python Application (e.g. from c_mathlib import is_prime, gcd, lcm, factorial, fibonacci)
       │
       ▼
Python Wrapper (`c_mathlib/core.py`)
  • Validates input types & ranges in Python
  • Calls C foreign function symbols via ctypes
  • Translates return codes and data types
       │
       ▼
Python `ctypes` FFI Layer
  • Maps Python arguments to C ABI (`argtypes`)
  • Defines return register conventions (`restype`)
       │
       ▼
Compiled Dynamic Library (`libmymath.dll` / `.so` / `.dylib`)
       │
       ▼
C Implementation (`src/mymath.c`, `src/mymath.h`)
  • Computes mathematical primitives directly on CPU in native C
```

---

## Repository Structure

```text
c-mathlib/
├── src/
│   ├── mymath.h                      # C header declaring functions and export macros
│   └── mymath.c                      # C implementation of mathematical algorithms
├── c_mathlib/
│   ├── __init__.py                   # Public package API
│   ├── core.py                       # ctypes loader and Python wrappers
│   └── libmymath.dll                 # Compiled native dynamic library
├── tests/
│   ├── test_is_prime.py              # Tests for primality testing
│   ├── test_gcd_lcm.py               # Tests for GCD and LCM
│   └── test_factorial_fibonacci.py   # Tests for Factorial and Fibonacci
├── examples/
│   └── basic.py                      # Runnable demonstration script
├── pyproject.toml                    # PEP 517/621 package metadata & configuration
├── .gitignore                        # Git ignore rules for build and cache artifacts
├── LICENSE                           # MIT License
└── README.md                         # Project documentation
```

---

## Prerequisites

To build the native C library, you need a C compiler installed on your system:

- **Windows**: MinGW-w64 (`gcc`), Clang (`clang`), or MSVC (`cl.exe`).
- **Linux**: `gcc` or `clang` (`build-essential` on Ubuntu/Debian).
- **macOS**: `clang` (Xcode Command Line Tools).

---

## Building the Native Library

Compile the C source file into a shared library within the `c_mathlib` directory.

### On Windows (MinGW / GCC)
```powershell
gcc -O2 -shared -o c_mathlib/libmymath.dll src/mymath.c
```

### On Windows (MSVC)
```powershell
cl /O2 /LD src/mymath.c /Fe:c_mathlib/libmymath.dll
```

### On Linux
```bash
gcc -O2 -fPIC -shared -o c_mathlib/libmymath.so src/mymath.c
```

### On macOS
```bash
clang -O2 -dynamiclib -o c_mathlib/libmymath.dylib src/mymath.c
```

---

## Local Installation

Once the native library is compiled, install the package in editable (development) mode using `pip`:

```powershell
python -m pip install -e .
```

This registers `c_mathlib` into your current Python environment, allowing any script or project on your machine to import it directly.

---

## Usage

```python
from c_mathlib import (
    is_prime,
    gcd,
    lcm,
    factorial,
    fibonacci,
)

# Primality
print(is_prime(97))      # True
print(is_prime(100))     # False

# Greatest Common Divisor & Least Common Multiple
print(gcd(48, 18))       # 6
print(lcm(12, 18))       # 36

# Factorial
print(factorial(5))      # 120
print(factorial(20))     # 2432902008176640000

# Fibonacci sequence (0-indexed: F_0 = 0, F_1 = 1)
print(fibonacci(10))     # 55
print(fibonacci(50))     # 12586269025
```

Run the included example script:
```powershell
python examples/basic.py
```

---

## Running Tests

Run the full test suite using Python's built-in `unittest` module:

```powershell
python -m unittest discover -s tests -v
```

Or using `pytest` if installed:
```powershell
pytest -v
```

---

## What I Am Learning

- **Symbol Visibility & Exporting**: Understanding `__declspec(dllexport)` on Windows vs. default symbol visibility on ELF/Mach-O systems.
- **Dynamic Linking**: How dynamic linkers find and map `.dll` / `.so` files into the process's virtual address space at runtime.
- **Foreign Function Interfaces**: Why `argtypes` and `restype` are necessary in `ctypes` to correctly align stack frames and register conventions according to the C Application Binary Interface (ABI).
- **Domain & Precision Boundaries**: Handling signed 64-bit integer limits across C and Python with graceful overflow and type errors.
- **Python Packaging with Native Extensions**: How to bundle compiled binary artifacts (`package_data`) alongside pure Python code inside `pyproject.toml`.

---

## Planned Future Functions

The core architecture will continue to expand with additional mathematical functions:

- `pow_mod(base, exp, mod)` (Modular exponentiation)
- `nCr(n, r)` / `nPr(n, r)` (Combinations and permutations)
- `is_perfect_square(n)`
- Prime factorization helpers
