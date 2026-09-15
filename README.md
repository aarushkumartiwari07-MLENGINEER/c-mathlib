# c-mathlib

A high-performance, multi-module mathematical and numerical computing library whose core algorithms are implemented in **pure C** and exposed to **Python** through `ctypes`.

---

## Overview

`c-mathlib` is built to provide an educational, transparent, and robust native numerical computing pipeline. Rather than relying on heavyweight automated wrapper generators (SWIG, Cython, PyBind11), this library uses a clean C application binary interface (ABI) paired with Python's standard `ctypes` foreign function interface (FFI).

### Key Architectural Pillars
- **Zero Python-Callback Overhead**: Numerical algorithms (root finding, Simpson quadrature, differentiation) evaluate polynomial and transcendental expressions inside compiled C via Horner's method and internal function dispatch tables.
- **Cache-Friendly Row-Major Memory**: Linear algebra routines use contiguous 1D row-major buffers (`double*`) for GEMM multiplication, transpositions, and partial-pivoting Gaussian elimination.
- **Numerically Stable Algorithms**: Two-pass variance computation (canceling catastrophic cancellation in single-pass formulas), $O(\log n)$ modular exponentiation with 128-bit intermediate products, and Quickselect-based quantile calculations.
- **Pure Native Dependency**: Zero third-party dependencies. Requires only a C99/C11 compiler and Python 3.8+.

---

## Mathematical Capabilities Reference

### 1. Arithmetic & Sequence Primitives

| Function | Mathematical Principle | Input Range | Edge Cases & Error Handling |
| :--- | :--- | :--- | :--- |
| `is_prime(n)` | $6k \pm 1$ primality test in C | 32-bit signed ints | Negative, $0, 1 \rightarrow \text{False}$ |
| `gcd(a, b)` | Euclidean algorithm ($O(\log(\min(a,b)))$) | 64-bit signed ints | Returns non-negative; $\gcd(0,0)=0$ |
| `lcm(a, b)` | $(|a| / \gcd(a,b)) \times |b|$ | 64-bit signed ints | Returns non-negative; $\text{lcm}(0,x)=0$ |
| `factorial(n)` | Iterative product $n! = \prod_{i=1}^n i$ | $0 \le n \le 20$ (64-bit limit) | $n < 0 \rightarrow \text{ValueError}$, $n > 20 \rightarrow \text{OverflowError}$ |
| `fibonacci(n)` | Iterative $O(n)$ time / $O(1)$ space | $0 \le n \le 92$ (64-bit limit) | $n < 0 \rightarrow \text{ValueError}$, $n > 92 \rightarrow \text{OverflowError}$ |

### 2. Number Theory & Modular Arithmetic

| Function | Mathematical Principle | Input Range | Edge Cases & Error Handling |
| :--- | :--- | :--- | :--- |
| `mod_pow(base, exp, mod)` | Exponentiation by squaring in $O(\log(\text{exp}))$ | `exp >= 0`, `mod > 0` | `mod <= 0` or `exp < 0` $\rightarrow \text{ValueError}$ |
| `extended_gcd(a, b)` | Extended Euclidean: $a \cdot x + b \cdot y = \gcd(a, b)$ | 64-bit signed ints | Returns `(g, x, y)` tuple |
| `mod_inverse(a, mod)` | Multiplicative inverse $(a \cdot x \equiv 1 \pmod m)$ | `mod > 1` | Non-coprime $(\gcd \ne 1) \rightarrow \text{ValueError}$ |
| `euler_totient(n)` | Euler's phi function $\phi(n) = n \prod (1 - 1/p)$ | $n \ge 1$ (64-bit int) | $n \le 0 \rightarrow \text{ValueError}$; computed in $O(\sqrt{n})$ in C |

### 3. Combinatorics & Discrete Mathematics

| Function | Mathematical Principle | Input Range | Key Notes |
| :--- | :--- | :--- | :--- |
| `combinations(n, k)` / `nCr` | Binomial coefficient $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ | $0 \le k \le n$ (64-bit limit) | Multiplicative loop in C; returns 0 for $k > n$ |
| `permutations(n, k)` / `nPr` | Ordered arrangements $P(n, k) = \frac{n!}{(n-k)!}$ | $0 \le k \le n$ (64-bit limit) | Multiplicative loop in C; returns 0 for $k > n$ |

### 4. Numerical Mathematics (`c_mathlib.numerical`)

| Function | Method / Principle | Convergence & Error | Key Notes |
| :--- | :--- | :--- | :--- |
| `bisection(func, a, b, tol, max_iter)` | Interval halving on $[a, b]$ | Linear convergence | $f(a) \cdot f(b) > 0 \rightarrow \text{ValueError}$ (not bracketed) |
| `simpson(func, a, b, n)` | Composite Simpson's 1/3 rule | $O(h^4)$ truncation error | Requires positive even $n \ge 2$; continuous integrand |
| `derivative(func, x, h)` | Central difference: $\frac{f(x+h) - f(x-h)}{2h}$ | $O(h^2)$ truncation error | Balances truncation error vs roundoff cancellation |
| `Polynomial(coeffs)` | Polynomial representation with Horner's evaluation | $O(d)$ Horner evaluation | Direct evaluation in C without FFI callback overhead |

### 5. Linear Algebra (`c_mathlib.linear_algebra`)

| Class / Function | Mathematical Operation | Complexity | Description |
| :--- | :--- | :--- | :--- |
| `Vector` | $n$-dimensional vector algebra | $O(n)$ | Supports `+`, `-`, `*` (scalar), `@` (dot), `.norm(p)`, `.cross()`, `.cosine_similarity()` |
| `Matrix` | 2D matrix stored in row-major order | $O(mn)$ | Supports `+`, `-`, `*` (scalar), `@` (GEMM & matrix-vector), `.T`, `.trace()`, `.det()`, `.solve()` |
| `dot(u, v)` | Inner product $\mathbf{u} \cdot \mathbf{v}$ | $O(n)$ | Native C vector accumulation |
| `norm(v, p)` | Vector $L_1$, $L_2$, or $L_\infty$ norm | $O(n)$ | $p=1, 2, \infty$ (Manhattan, Euclidean, Chebyshev) |
| `cross(u, v)` | 3D vector cross product $\mathbf{u} \times \mathbf{v}$ | $O(1)$ | Orthogonal vector in $\mathbb{R}^3$ |
| `cosine_similarity(u, v)` | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | $O(n)$ | Normalized angular similarity $\in [-1, 1]$ |
| `matrix_mult(A, B)` | Matrix product $C_{m \times p} = A_{m \times k} B_{k \times p}$ | $O(mkp)$ | General matrix multiply (GEMM) in C |
| `transpose(A)` | Matrix transpose $A^T$ | $O(mn)$ | Transposes rows and columns in C |
| `determinant(A)` / `A.det()` | Determinant $\det(A)$ | $O(n^3)$ | Gaussian elimination with partial pivoting in C |
| `solve_linear(A, b)` / `A.solve(b)`| Solve linear system $A \mathbf{x} = \mathbf{b}$ | $O(n^3)$ | LU-style forward elimination & back-substitution in C |

### 6. Statistics & Regression (`c_mathlib.statistics`)

| Function | Mathematical Principle | Complexity | Description |
| :--- | :--- | :--- | :--- |
| `mean(data)` | Arithmetic average $\bar{x} = \frac{1}{n} \sum x_i$ | $O(n)$ | Computed in native C |
| `variance(data, ddof)` | Two-pass sample/population variance | $O(n)$ | Numerically stable; default sample variance (`ddof=1`) |
| `std_dev(data, ddof)` | Standard deviation $\sqrt{\text{Var}(x)}$ | $O(n)$ | Default sample standard deviation (`ddof=1`) |
| `median(data)` | 50th percentile | $O(n)$ avg | Computed via C Quickselect partitioning |
| `quantile(data, q)` | Linear interpolation quantile ($q \in [0, 1]$) | $O(n \log n)$ | Interpolates between ranks $(n-1)q$ in C |
| `covariance(x, y, ddof)` | Sample/population covariance $\text{Cov}(X, Y)$ | $O(n)$ | Two-pass joint variance in C |
| `correlation(x, y)` | Pearson correlation coefficient $r \in [-1, 1]$ | $O(n)$ | $\frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$ |
| `linear_regression(x, y)` | Ordinary Least Squares (OLS) $y = \beta_0 + \beta_1 x$ | $O(n)$ | Computes $\beta_1, \beta_0, R^2$, and provides `.predict(x)` |

---

## Architecture

```text
Python Application Layer
  ├── Root Package: `from c_mathlib import is_prime, gcd, lcm, mod_pow, combinations, ...`
  ├── Submodules:
  │     ├── `c_mathlib.numerical`      (bisection, simpson, derivative, Polynomial)
  │     ├── `c_mathlib.linear_algebra` (Vector, Matrix, dot, norm, determinant, solve_linear)
  │     ├── `c_mathlib.statistics`    (mean, variance, correlation, linear_regression)
  │     └── `c_mathlib.combinatorics` (combinations, permutations, euler_totient)
        │
        ▼
Python Wrapper & Validation Layer
  • Type & domain boundary checks (e.g. non-empty vectors, square matrices, even Simpson intervals)
  • Marshaling sequence inputs to contiguous C arrays (`ctypes.POINTER(ctypes.c_double)`)
  • Translates C error return codes into Python ValueError/ZeroDivisionError/OverflowError
        │
        ▼
Python `ctypes` FFI Layer
  • Directly links to compiled dynamic library (`libmymath.dll` / `.so` / `.dylib`)
        │
        ▼
Native C Engine (`src/*.c`, `src/*.h`)
  • `src/mymath.c`: Primes, GCD, LCM, modular arithmetic, Horner evaluation, Simpson, bisection
  • `src/linear_algebra.c`: Vector ops, norms, cosine similarity, GEMM, partial pivoting Gaussian elimination
  • `src/statistics.c`: Two-pass variance, quickselect median/quantiles, covariance, OLS regression
  • `src/combinatorics.c`: Overflow-checked combinations (nCr), permutations (nPr), Euler's totient
```

---

## Repository Structure

```text
c-mathlib/
├── src/
│   ├── mymath.h                      # Declarations for core math, modular arithmetic, numerical routines
│   ├── mymath.c                      # Implementations of core arithmetic and numerical calculus
│   ├── linear_algebra.h              # Declarations for vector/matrix structures and routines
│   ├── linear_algebra.c              # Implementations of vector ops, GEMM, determinant, linear solver
│   ├── statistics.h                  # Declarations for descriptive statistics and OLS regression
│   ├── statistics.c                  # Implementations of two-pass stats, quantiles, and linear regression
│   ├── combinatorics.h               # Declarations for combinations, permutations, and Euler totient
│   └── combinatorics.c               # Implementations of nCr, nPr, and phi(n)
├── c_mathlib/
│   ├── __init__.py                   # Public package API and root symbol exports
│   ├── core.py                       # ctypes dynamic library loader and core arithmetic bindings
│   ├── numerical.py                  # High-level numerical routines (bisection, Simpson, derivative)
│   ├── linear_algebra.py             # Vector and Matrix classes, norms, GEMM, and linear solver
│   ├── statistics.py                 # Descriptive statistics, correlation, and OLS regression model
│   ├── combinatorics.py              # Combinations, permutations, and totient wrappers
│   └── libmymath.dll                 # Compiled native dynamic library
├── tests/
│   ├── test_is_prime.py              # Tests for primality testing (10 tests)
│   ├── test_gcd_lcm.py               # Tests for GCD and LCM (17 tests)
│   ├── test_factorial_fibonacci.py   # Tests for Factorial and Fibonacci (13 tests)
│   ├── test_modular.py               # Tests for Modular arithmetic and Extended GCD (14 tests)
│   ├── test_numerical.py             # Tests for Root finding, Simpson, derivative (17 tests)
│   ├── test_linear_algebra.py        # Tests for Vector, Matrix, GEMM, solver (12 tests)
│   ├── test_statistics.py            # Tests for mean, variance, OLS regression (7 tests)
│   └── test_combinatorics.py         # Tests for nCr, nPr, Euler's totient (8 tests)
├── examples/
│   ├── basic.py                      # Fundamental arithmetic & number theory demo
│   ├── numerical_demo.py             # Numerical calculus and root finding demo
│   ├── linear_algebra_demo.py        # Vector/matrix operations, determinants, and linear solver demo
│   ├── statistics_demo.py            # Descriptive statistics, correlation, and OLS regression demo
│   └── combinatorics_demo.py         # Combinations, permutations, and totient demo
├── pyproject.toml                    # PEP 517/621 package metadata & configuration
├── .gitignore                        # Git ignore rules for build and cache artifacts
├── LICENSE                           # MIT License
└── README.md                         # Complete project documentation
```

---

## Building the Native Library

To compile the multi-file C engine into the shared library within `c_mathlib`:

### On Windows (MinGW / GCC)
```powershell
gcc -Wall -Wextra -O2 -shared -static-libgcc -o c_mathlib/libmymath.dll src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c -lm
```

### On Windows (MSVC)
```powershell
cl /O2 /LD src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c /Fe:c_mathlib/libmymath.dll
```

### On Linux
```bash
gcc -Wall -Wextra -O2 -fPIC -shared -o c_mathlib/libmymath.so src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c -lm
```

### On macOS
```bash
clang -Wall -Wextra -O2 -dynamiclib -o c_mathlib/libmymath.dylib src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c -lm
```

---

## Local Installation

Install the package in editable (development) mode using `pip`:

```powershell
python -m pip install -e .
```

---

## Usage Examples

### 1. Linear Algebra (Vectors, Matrices, Linear Systems)

```python
from c_mathlib.linear_algebra import Vector, Matrix, solve_linear, cosine_similarity

# Vector geometry
u = Vector([1.0, 2.0, 3.0])
v = Vector([4.0, 5.0, 6.0])

print(u + v)                         # Vector([5, 7, 9])
print(u @ v)                         # 32.0 (dot product)
print(u.norm(p=2))                   # 3.741657 (L2 norm)
print(cosine_similarity(u, v))       # 0.974632

# Matrix GEMM Multiplication & Determinants
A = Matrix([[1.0, 2.0], [3.0, 4.0]])
B = Matrix([[2.0, 0.0], [1.0, 2.0]])
print(A @ B)                         # Matrix([[4, 4], [10, 8]])
print(A.det())                       # -2.0

# Solve System of Linear Equations: A * x = b
# 2x + y = 5
# x + 3y = 5
sys_A = Matrix([[2.0, 1.0], [1.0, 3.0]])
sys_b = Vector([5.0, 5.0])
x_sol = solve_linear(sys_A, sys_b)
print("Solution x =", x_sol)         # Vector([2, 1])
```

### 2. Statistics & Ordinary Least Squares (OLS) Regression

```python
from c_mathlib.statistics import mean, variance, std_dev, correlation, linear_regression

hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
scores = [50.0, 55.0, 65.0, 70.0, 75.0, 85.0]

print("Mean score:", mean(scores))                 # 66.67
print("Sample Std Dev:", std_dev(scores, ddof=1))  # 12.91
print("Correlation (r):", correlation(hours, scores))  # 0.9959

# Fit OLS Line: y = beta_0 + beta_1 * x
model = linear_regression(hours, scores)
print(model)                         # LinearRegression(y = 6.857*x + 42.667, R^2 = 0.9918)
print("Predicted score for 7 hrs:", model.predict(7.0))  # 90.67
```

### 3. Combinatorics & Discrete Mathematics

```python
from c_mathlib.combinatorics import combinations, permutations, euler_totient, nCr, nPr

# Binomial Coefficients (nCr) & Permutations (nPr)
print("5-card poker hands:", combinations(52, 5))   # 2,598,960
print("Permutations P(10, 3):", permutations(10, 3)) # 720

# Euler's Totient Function (Count of coprimes up to n)
print("phi(36) =", euler_totient(36))               # 12
print("phi(100) =", euler_totient(100))             # 40
```

### 4. Numerical Calculus & Root Finding

```python
import math
from c_mathlib.numerical import bisection, simpson, derivative, Polynomial

# Bisection Root Finding: Solve x^3 - x - 2 = 0 on [1, 2]
root = bisection("x^3 - x - 2", 1.0, 2.0, tol=1e-7)
print("Root:", root)                 # ~1.5213797

# Composite Simpson's 1/3 Rule: Integrate sin(x) on [0, pi]
integral = simpson("sin", 0.0, math.pi, n=100)
print("Integral:", integral)         # 2.0000000

# Central Difference Derivative: d/dx (x^2) at x=3.0
print("Derivative:", derivative("x^2", 3.0))  # 6.000000

# Polynomials in C (Horner's Method): P(x) = 5 - 2x + 4x^3
p = Polynomial([5, -2, 0, 4])
print("P'(2) =", derivative(p, 2.0)) # 46.0
```

### 5. Arithmetic & Modular Cryptographic Primitives

```python
from c_mathlib import is_prime, gcd, lcm, mod_pow, extended_gcd, mod_inverse

# Primality & Division
print(is_prime(97))                  # True
print(gcd(48, 18), lcm(12, 18))      # 6, 36

# Modular Exponentiation (O(log exp) in C)
print(mod_pow(2, 10, 1000))          # 24

# Bézout Identity & Modular Inversion
g, x, y = extended_gcd(30, 12)       # gcd=6, x=1, y=-2 -> 30*(1) + 12*(-2) = 6
print(mod_inverse(3, 11))            # 4 (since 3 * 4 == 12 == 1 mod 11)
```

---

## Running Example Demos

Run the interactive demonstration scripts located in `examples/`:

```powershell
python examples/basic.py
python examples/numerical_demo.py
python examples/linear_algebra_demo.py
python examples/statistics_demo.py
python examples/combinatorics_demo.py
```

---

## Running Tests

Run the complete test suite (98 unit tests covering all C routines and edge cases) using `unittest`:

```powershell
python -m unittest discover -s tests -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
