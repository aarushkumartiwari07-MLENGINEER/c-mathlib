# c-mathlib

A lightweight mathematical library whose core implementations are written in native **C** and exposed to **Python** via `ctypes`.

---

## Why This Project Exists

`c-mathlib` is built to explore and demonstrate how Python interfaces with native C code. Rather than relying on heavyweight wrapper frameworks or high-level abstractions, this project provides a clean, transparent foundation for understanding:

1. How native C code is written, compiled, and exported as a dynamic shared library (`.dll` on Windows, `.so` on Linux, `.dylib` on macOS).
2. How Python's `ctypes` foreign function interface (FFI) dynamically loads the library and marshals data across the language boundary.
3. How a native C extension project can be structured as an installable, reusable Python package.

---

### Mathematical Functions Reference

### 1. Arithmetic & Sequence Primitives

| Function | Mathematical Principle | Input Range | Edge Cases / Error Handling |
| :--- | :--- | :--- | :--- |
| `is_prime(n)` | 6k ± 1 primality test in C | 32-bit signed ints | Negative, $0, 1 \rightarrow \text{False}$ |
| `gcd(a, b)` | Euclidean algorithm ($O(\log(\min(a,b)))$) | 64-bit signed ints | Returns non-negative; $\gcd(0,0)=0$ |
| `lcm(a, b)` | $(|a| / \gcd(a,b)) \times |b|$ | 64-bit signed ints | Returns non-negative; $\text{lcm}(0,x)=0$ |
| `factorial(n)` | Iterative product $n! = \prod_{i=1}^n i$ | $0 \le n \le 20$ (64-bit limit) | $n < 0 \rightarrow \text{ValueError}$, $n > 20 \rightarrow \text{OverflowError}$ |
| `fibonacci(n)` | Iterative $O(n)$ time / $O(1)$ space | $0 \le n \le 92$ (64-bit limit) | $n < 0 \rightarrow \text{ValueError}$, $n > 92 \rightarrow \text{OverflowError}$ |

### 2. Number Theory & Modular Arithmetic

| Function | Mathematical Principle | Input Range | Edge Cases / Error Handling |
| :--- | :--- | :--- | :--- |
| `mod_pow(base, exp, mod)` | Exponentiation by squaring in $O(\log(\text{exp}))$ | `exp >= 0`, `mod > 0` | `mod <= 0` or `exp < 0` $\rightarrow \text{ValueError}$ |
| `extended_gcd(a, b)` | Extended Euclidean algorithm: $a \cdot x + b \cdot y = \gcd(a, b)$ | 64-bit signed ints | Returns `(g, x, y)` tuple |
| `mod_inverse(a, mod)` | Multiplicative inverse $(a \cdot x \equiv 1 \pmod m)$ via Extended GCD | `mod > 1` | Non-coprime $(\gcd \ne 1) \rightarrow \text{ValueError}$ |

### 3. Numerical Mathematics (`c_mathlib.numerical`)

| Function | Method / Mathematical Principle | Convergence & Error | Key Notes |
| :--- | :--- | :--- | :--- |
| `bisection(func, a, b, tol, max_iter)` | Interval halving on $[a, b]$ | Linear convergence | $f(a) \cdot f(b) > 0 \rightarrow \text{ValueError}$ (not bracketed) |
| `simpson(func, a, b, n)` | Composite Simpson's 1/3 rule with quadratic interpolation | $O(h^4)$ truncation error | Requires positive even $n \ge 2$; continuous integrand |
| `derivative(func, x, h)` | Central difference: $\frac{f(x+h) - f(x-h)}{2h}$ | $O(h^2)$ truncation error | Balances truncation error vs roundoff cancellation |
| `Polynomial(coeffs)` | Polynomial representation with Horner's evaluation | $O(d)$ Horner evaluation | Direct evaluation in C without FFI callback overhead |

---

## Architecture

```text
Python Application Layer
  ├── Root Package: `from c_mathlib import is_prime, gcd, lcm, mod_pow, extended_gcd, mod_inverse`
  └── Numerical Submodule: `from c_mathlib.numerical import bisection, simpson, derivative, Polynomial`
        │
        ▼
Python Wrapper & Validation Layer (`core.py`, `numerical.py`)
  • Validates parameter types, domains, and bounds in Python
  • Marshals inputs to C ABI formats
  • Converts C status codes to descriptive Python exceptions
        │
        ▼
Python `ctypes` FFI Layer
  • Maps 64-bit integers (`c_longlong`), doubles (`c_double`), and pointers (`POINTER`)
  • Binds directly to the compiled native dynamic library symbols
        │
        ▼
Compiled Dynamic Library (`libmymath.dll` / `.so` / `.dylib`)
        │
        ▼
Native C Engine (`src/mymath.c`, `src/mymath.h`)
  • High-performance CPU-bound number theory & numerical algorithms
  • Zero Python callback overhead: Horner's method & internal function dispatch in pure C
```

---

## Repository Structure

```text
c-mathlib/
├── src/
│   ├── mymath.h                      # C header declaring symbols, export macros, and function IDs
│   └── mymath.c                      # C implementation of algorithms and numerical solvers
├── c_mathlib/
│   ├── __init__.py                   # Public package API and submodule exports
│   ├── core.py                       # ctypes loader and fundamental math wrappers
│   ├── numerical.py                  # High-level numerical routines (bisection, Simpson, derivative)
│   └── libmymath.dll                 # Compiled native dynamic library
├── tests/
│   ├── test_is_prime.py              # Tests for primality testing
│   ├── test_gcd_lcm.py               # Tests for GCD and LCM
│   ├── test_factorial_fibonacci.py   # Tests for Factorial and Fibonacci
│   ├── test_modular.py               # Tests for Modular arithmetic and Extended GCD
│   └── test_numerical.py             # Tests for Root finding, Integration, and Differentiation
├── examples/
│   ├── basic.py                      # Fundamental arithmetic & number theory demo
│   └── numerical_demo.py             # Numerical calculus and root finding demo
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
gcc -O2 -shared -static-libgcc -o c_mathlib/libmymath.dll src/mymath.c -lm
```

### On Windows (MSVC)
```powershell
cl /O2 /LD src/mymath.c /Fe:c_mathlib/libmymath.dll
```

### On Linux
```bash
gcc -O2 -fPIC -shared -o c_mathlib/libmymath.so src/mymath.c -lm
```

### On macOS
```bash
clang -O2 -dynamiclib -o c_mathlib/libmymath.dylib src/mymath.c
```

---

## Local Installation

Install the package in editable (development) mode using `pip`:

```powershell
python -m pip install -e .
```

---

## Usage Examples

### 1. Arithmetic & Number Theory

```python
from c_mathlib import (
    is_prime,
    gcd,
    lcm,
    factorial,
    fibonacci,
    mod_pow,
    extended_gcd,
    mod_inverse,
)

# Primality & Sequences
print(is_prime(97))         # True
print(factorial(5))         # 120
print(fibonacci(10))        # 55

# GCD & LCM
print(gcd(48, 18))          # 6
print(lcm(12, 18))          # 36

# Modular Exponentiation (Exponentiation by squaring in C)
print(mod_pow(2, 10, 1000)) # 24  (2^10 % 1000)

# Extended Euclidean Algorithm (Bézout's identity: a*x + b*y == gcd(a, b))
g, x, y = extended_gcd(30, 12)
print(f"gcd={g}, x={x}, y={y} -> 30*({x}) + 12*({y}) = {30*x + 12*y}")

# Modular Multiplicative Inverse
print(mod_inverse(3, 11))   # 4  (because (3 * 4) % 11 == 1)
```

### 2. Numerical Mathematics

```python
import math
from c_mathlib.numerical import bisection, simpson, derivative, Polynomial

# Root Finding (Bisection Method)
# Solve x^3 - x - 2 = 0 on [1, 2]
root = bisection("x^3 - x - 2", 1.0, 2.0, tol=1e-7)
print(f"Root: {root:.7f}")   # ~1.5213797

# Numerical Integration (Composite Simpson's 1/3 Rule)
# Approximate integral of sin(x) from 0 to pi (= 2.0)
integral = simpson("sin", 0.0, math.pi, n=100)
print(f"Integral: {integral:.7f}")  # 2.0000000

# Numerical Differentiation (Central Difference Approximation)
# Approximate derivative of x^2 at x = 3.0 (= 6.0)
slope = derivative("x^2", 3.0, h=1e-5)
print(f"Derivative: {slope:.6f}")   # 6.000000

# Custom Polynomials via Coefficients (e.g. P(x) = 5 - 2x + 4x^3)
p = Polynomial([5, -2, 0, 4])
print("P'(2) =", derivative(p, 2.0))      # 46.0
print("Integral_0^1 P(x) dx =", simpson(p, 0.0, 1.0)) # 5.0
```

Run the demonstration scripts:
```powershell
python examples/basic.py
python examples/numerical_demo.py
```

---

## Running Tests

Run the complete test suite using Python's built-in `unittest` module:

```powershell
python -m unittest discover -s tests -v
```

---

## What I Am Learning

- **Modular Arithmetic & Cryptographic Primitives**: Implementing $O(\log e)$ binary exponentiation by squaring and Bézout coefficient extraction via the Extended Euclidean algorithm.
- **Numerical Calculus in Native C**: Implementing composite Simpson's 1/3 rule with $O(h^4)$ accuracy, central difference numerical differentiation, and interval-halving root finding.
- **Horner's Method for Polynomials**: Efficient $O(d)$ polynomial evaluation directly in compiled C, avoiding floating-point instability and Python callback overhead.
- **Step-Size Tradeoffs**: Understanding the balance between truncation error ($O(h^2)$) and machine roundoff error when selecting finite difference steps $h$.
- **Modular Library Architecture**: Structuring a Python package with dedicated domain submodules (`c_mathlib.numerical`).

---

## Planned Future Functions

- Linear algebra: vector dot products, matrix multiplication, Gaussian elimination
- Statistics: mean, variance, covariance, correlation
- Advanced numerical methods: Newton-Raphson method, Runge-Kutta ODE solvers (RK4)
- Combinatorics: `nCr(n, r)` / `nPr(n, r)` (Combinations and permutations)
- Number theory: `is_perfect_square(n)`, prime factorization helpers

