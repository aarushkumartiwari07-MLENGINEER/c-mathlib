# c-mathlib

A high-performance, multi-module mathematical and numerical computing library whose core algorithms are implemented in **pure C** and exposed to **Python** through `ctypes`.

---

## Overview

`c-mathlib` is built to provide an educational, transparent, and robust native numerical computing pipeline. Rather than relying on heavyweight automated wrapper generators (SWIG, Cython, PyBind11), this library uses a clean C application binary interface (ABI) paired with Python's standard `ctypes` foreign function interface (FFI).

### Key Architectural Pillars
- **Zero Python-Callback Overhead**: Numerical algorithms (root finding, Simpson quadrature, differentiation, RK4 ODE integration, Golden Section optimization) evaluate polynomial and transcendental expressions inside compiled C via Horner's method and internal function dispatch tables.
- **Cache-Friendly Row-Major Memory**: Linear algebra routines use contiguous 1D row-major buffers (`double*`) for GEMM multiplication, transpositions, LU / QR / Cholesky factorizations, and least squares solvers.
- **Numerically Stable Algorithms**: Two-pass variance computation (canceling catastrophic cancellation in single-pass formulas), $O(\log n)$ modular exponentiation with 128-bit intermediate products, Lanczos Gamma approximation, and max-subtracted softmax.
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

### 4. Numerical Calculus, Solvers & Optimization (`c_mathlib.numerical`)

| Function | Method / Principle | Convergence & Error | Key Notes |
| :--- | :--- | :--- | :--- |
| `bisection(func, a, b, tol, max_iter)` | Interval halving on $[a, b]$ | Linear convergence | $f(a) \cdot f(b) > 0 \rightarrow \text{ValueError}$ (not bracketed) |
| `newton(func, x0, tol, max_iter)` | Newton-Raphson root finding: $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$ | Quadratic convergence | Exact analytical Horner derivative for polynomials in C |
| `secant(func, x0, x1, tol, max_iter)`| Secant root finding (quasi-Newton) | Superlinear ($r \approx 1.618$) | Derivative-free root solving with finite difference slope in C |
| `simpson(func, a, b, n)` | Composite Simpson's 1/3 rule | $O(h^4)$ truncation error | Requires positive even $n \ge 2$; continuous integrand |
| `derivative(func, x, h)` | Central difference: $\frac{f(x+h) - f(x-h)}{2h}$ | $O(h^2)$ truncation error | Balances truncation error vs roundoff cancellation |
| `rk4(func, y0, t_span, steps)` | Runge-Kutta 4th Order ODE initial-value solver | $O(h^4)$ global truncation error | Returns `ODEResult` with time grid and state trajectory |
| `minimize_1d(func, bracket, tol)` | Golden Section Search 1D minimizer | Linear ($r \approx 0.618$) | Derivative-free unimodal optimization returning `OptimizeResult` |
| `Polynomial(coeffs)` | Polynomial representation with Horner's evaluation | $O(d)$ Horner evaluation | Direct evaluation in C without FFI callback overhead |

### 5. Special Functions & ML Primitives (`c_mathlib.special`)

| Function | Mathematical Principle | Implementation | Key Notes |
| :--- | :--- | :--- | :--- |
| `gamma(x)` | Euler Gamma function $\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt$ | Lanczos ($g=7, N=9$) & reflection | Generalizes factorial: $\Gamma(n) = (n-1)!$; $\Gamma(0.5) = \sqrt{\pi}$ |
| `lgamma(x)` | Natural logarithm $\ln\|\Gamma(x)\|$ | Lanczos approximation | Prevents overflow for large arguments |
| `beta(a, b)` | Euler Beta function $B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ | Log-gamma exponentiation | Symmetric: $B(a, b) = B(b, a)$ |
| `erf(x)` | Gauss Error Function $\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2} dt$ | Abramowitz & Stegun 7.1.26 | Accurate to $< 1.5 \times 10^{-7}$ across $\mathbb{R}$ |
| `erfc(x)` | Complementary Error Function $1 - \text{erf}(x)$ | High-precision complementary | Returns values in $[0, 2]$ |
| `sigmoid(x)` | Logistic activation $\sigma(x) = \frac{1}{1 + e^{-x}}$ | Branching for negative $x$ | Numerically stable against large positive/negative inputs |
| `softmax(x)` | Softmax probabilities $\frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$ | Max-subtracted exponential | Overflow-resilient probability distribution summing to 1.0 |

### 6. Linear Algebra (`c_mathlib.linear_algebra`)

| Class / Function | Mathematical Operation | Complexity | Description |
| :--- | :--- | :--- | :--- |
| `Vector` | $n$-dimensional vector algebra | $O(n)$ | Supports `+`, `-`, `*` (scalar), `@` (dot), `.norm(p)`, `.cross()`, `.cosine_similarity()` |
| `Matrix` | 2D matrix stored in row-major order | $O(mn)$ | Supports `+`, `-`, `*` (scalar), `@` (GEMM & matrix-vector), `.T`, `.trace()`, `.det()`, `.solve()`, `.lu()`, `.qr()`, `.cholesky()`, `.inv()` |
| `dot(u, v)` | Inner product $\mathbf{u} \cdot \mathbf{v}$ | $O(n)$ | Native C vector accumulation |
| `norm(v, p)` | Vector $L_1$, $L_2$, or $L_\infty$ norm | $O(n)$ | $p=1, 2, \infty$ (Manhattan, Euclidean, Chebyshev) |
| `cross(u, v)` | 3D vector cross product $\mathbf{u} \times \mathbf{v}$ | $O(1)$ | Orthogonal vector in $\mathbb{R}^3$ |
| `cosine_similarity(u, v)` | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | $O(n)$ | Normalized angular similarity $\in [-1, 1]$ |
| `matrix_mult(A, B)` | Matrix product $C_{m \times p} = A_{m \times k} B_{k \times p}$ | $O(mkp)$ | General matrix multiply (GEMM) in C |
| `transpose(A)` | Matrix transpose $A^T$ | $O(mn)$ | Transposes rows and columns in C |
| `determinant(A)` / `A.det()` | Determinant $\det(A)$ | $O(n^3)$ | Gaussian elimination with partial pivoting in C |
| `solve_linear(A, b)` / `A.solve(b)`| Solve linear system $A \mathbf{x} = \mathbf{b}$ | $O(n^3)$ | LU-style forward elimination & back-substitution in C |
| `lu(A)` / `A.lu()` | LU Decomposition $P A = L U$ | $O(n^3)$ | Partial-pivoting factorization into permutation $P$, unit lower $L$, upper $U$ |
| `qr(A)` / `A.qr()` | QR Decomposition $A = Q R$ | $O(mn^2)$ | Modified Gram-Schmidt orthogonalization ($Q^T Q = I$) |
| `cholesky(A)` / `A.cholesky()` | Cholesky Decomposition $A = L L^T$ | $O(n^3/3)$ | Factorization for symmetric positive-definite (SPD) matrices |
| `inv(A)` / `A.inv()` | Matrix Multiplicative Inverse $A^{-1}$ | $O(n^3)$ | Computes inverse via LU column solves against $I_n$ |
| `solve_least_squares(A, b)` | Linear Least Squares $\min \|A\mathbf{x} - \mathbf{b}\|_2$ | $O(mn^2)$ | Overdetermined solver ($m \ge n$) using native QR decomposition |

### 7. Statistics & Regression (`c_mathlib.statistics`)

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
  ├── Root Package: `from c_mathlib import is_prime, gcd, lcm, mod_pow, combinations, gamma, erf, ...`
  ├── Submodules:
  │     ├── `c_mathlib.numerical`      (bisection, newton, secant, simpson, derivative, rk4, minimize_1d)
  │     ├── `c_mathlib.special`        (gamma, lgamma, beta, erf, erfc, sigmoid, softmax)
  │     ├── `c_mathlib.linear_algebra` (Vector, Matrix, dot, norm, lu, qr, cholesky, inv, least_squares)
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
  • `src/mymath.c`: Primes, GCD, modular arithmetic, Simpson, bisection, Newton, Secant, RK4, Golden Section
  • `src/special.c`: Lanczos Gamma, Log-Gamma, Beta, Abramowitz-Stegun erf/erfc, sigmoid, softmax
  • `src/linear_algebra.c`: Vector ops, norms, GEMM, LU, QR, Cholesky, inversion, least squares
  • `src/statistics.c`: Two-pass variance, quickselect median/quantiles, covariance, OLS regression
  • `src/combinatorics.c`: Overflow-checked combinations (nCr), permutations (nPr), Euler's totient
```

---

## Repository Structure

```text
c-mathlib/
├── src/
│   ├── mymath.h                      # Declarations for core math, root finding, ODEs, optimization
│   ├── mymath.c                      # Implementations of core arithmetic, Newton, RK4, Golden Section
│   ├── special.h                     # Declarations for special functions (Gamma, Beta, erf, softmax)
│   ├── special.c                     # Implementations of Lanczos Gamma, Beta, erf/erfc, softmax
│   ├── linear_algebra.h              # Declarations for vector/matrix structures, decompositions
│   ├── linear_algebra.c              # Implementations of GEMM, LU, QR, Cholesky, inv, least squares
│   ├── statistics.h                  # Declarations for descriptive statistics and OLS regression
│   ├── statistics.c                  # Implementations of two-pass stats, quantiles, and linear regression
│   ├── combinatorics.h               # Declarations for combinations, permutations, and Euler totient
│   └── combinatorics.c               # Implementations of nCr, nPr, and phi(n)
├── c_mathlib/
│   ├── __init__.py                   # Public package API and root symbol exports
│   ├── core.py                       # ctypes dynamic library loader and core arithmetic bindings
│   ├── numerical.py                  # High-level numerical routines (Newton, RK4, Golden Section, etc.)
│   ├── special.py                    # Special functions and ML activations (Gamma, Beta, erf, softmax)
│   ├── linear_algebra.py             # Vector and Matrix classes, decompositions, and solvers
│   ├── statistics.py                 # Descriptive statistics, correlation, and OLS regression model
│   ├── combinatorics.py              # Combinations, permutations, and totient wrappers
│   └── libmymath.dll                 # Compiled native dynamic library
├── tests/
│   ├── test_is_prime.py              # Tests for primality testing (10 tests)
│   ├── test_gcd_lcm.py               # Tests for GCD and LCM (17 tests)
│   ├── test_factorial_fibonacci.py   # Tests for Factorial and Fibonacci (13 tests)
│   ├── test_modular.py               # Tests for Modular arithmetic and Extended GCD (14 tests)
│   ├── test_numerical.py             # Tests for Bisection, Simpson, derivative (17 tests)
│   ├── test_newton_secant.py         # Tests for Newton-Raphson and Secant methods (8 tests)
│   ├── test_ode_rk4.py               # Tests for Runge-Kutta 4th Order ODE solver (6 tests)
│   ├── test_optimization.py          # Tests for Golden Section 1D Minimization (7 tests)
│   ├── test_special.py               # Tests for Gamma, Beta, erf, sigmoid, softmax (10 tests)
│   ├── test_linear_algebra.py        # Tests for Vector, Matrix, GEMM, solver (12 tests)
│   ├── test_decompositions.py         # Tests for LU, QR, Cholesky, Inversion, Least Squares (13 tests)
│   ├── test_statistics.py            # Tests for mean, variance, OLS regression (7 tests)
│   └── test_combinatorics.py         # Tests for nCr, nPr, Euler's totient (8 tests)
├── examples/
│   ├── basic.py                      # Fundamental arithmetic & number theory demo
│   ├── numerical_demo.py             # Numerical calculus and root finding demo
│   ├── ode_optimization_demo.py      # RK4 ODE solver and Golden Section optimization demo
│   ├── special_functions_demo.py     # Gamma, Beta, erf, sigmoid, and softmax demo
│   ├── linear_algebra_demo.py        # Vector/matrix operations, determinants, and linear solver demo
│   ├── decompositions_demo.py        # LU, QR least squares, Cholesky, and matrix inversion demo
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
gcc -Wall -Wextra -O2 -shared -static-libgcc -o c_mathlib/libmymath.dll src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c src/special.c -lm
```

### On Linux
```bash
gcc -Wall -Wextra -O2 -fPIC -shared -o c_mathlib/libmymath.so src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c src/special.c -lm
```

### On macOS
```bash
clang -Wall -Wextra -O2 -dynamiclib -o c_mathlib/libmymath.dylib src/mymath.c src/linear_algebra.c src/statistics.c src/combinatorics.c src/special.c -lm
```

---

## Local Installation

Install the package in editable (development) mode using `pip`:

```powershell
python -m pip install -e .
```

---

## Usage Examples

### 1. Special Functions & Machine Learning Primitives

```python
from c_mathlib.special import gamma, beta, erf, erfc, sigmoid, softmax

# Gamma function (generalizes factorial: Gamma(n) = (n-1)!)
print("Gamma(5) =", gamma(5))              # 24.0
print("Gamma(0.5) =", gamma(0.5))          # 1.77245385 (sqrt(pi))

# Beta function: B(2, 3) = 1! * 2! / 4! = 1/12
print("Beta(2, 3) =", beta(2, 3))          # 0.08333333

# Error function (erf / erfc)
print("erf(1.0) =", erf(1.0))              # 0.842701
print("erfc(1.0) =", erfc(1.0))            # 0.157299

# Logistic Sigmoid activation
print("sigmoid(2.0) =", sigmoid(2.0))      # 0.880797

# Numerically Stable Softmax (overflow safe)
probs = softmax([1000.0, 1002.0, 999.0])
print("Probabilities:", probs)             # [0.1142, 0.8438, 0.0420]
```

### 2. Numerical Root Finding, ODEs & Optimization

```python
import math
from c_mathlib.numerical import newton, secant, rk4, minimize_1d, Polynomial

# Newton-Raphson: Solve x^3 - x - 2 = 0 using exact Horner analytical derivative
root = newton([-2, -1, 0, 1], x0=1.5, tol=1e-10)
print("Newton root:", root)                # 1.5213797068

# Secant method: Find root of sin(x) = 0 on [3, 4]
root_sec = secant("sin", x0=3.0, x1=4.0, tol=1e-10)
print("Secant root:", root_sec)            # 3.1415926535 (pi)

# Runge-Kutta 4th Order (RK4): Solve dy/dt = -y, y(0) = 1 on [0, 2]
res_ode = rk4("exp_decay", y0=1.0, t_span=(0.0, 2.0), steps=50)
print("RK4 y(2.0) =", res_ode.y_final)     # 0.13533528 (exp(-2))

# Golden Section 1D Optimization: Minimize (x - 3)^2 + 2 on [0, 6]
res_opt = minimize_1d([11, -6, 1], bracket=(0.0, 6.0), tol=1e-8)
print(f"Minimum at x={res_opt.x:.4f} with f(x)={res_opt.fun:.4f}")  # x=3.0000, f(x)=2.0000
```

### 3. Linear Algebra (Vectors, Matrices, Decompositions, Solvers)

```python
from c_mathlib.linear_algebra import (
    Vector,
    Matrix,
    lu,
    qr,
    cholesky,
    solve_linear,
    solve_least_squares,
    cosine_similarity,
)

# Vector geometry & Norms
u = Vector([1.0, 2.0, 3.0])
v = Vector([4.0, 5.0, 6.0])

print(u + v)                         # Vector([5, 7, 9])
print(u @ v)                         # 32.0 (dot product)
print(u.norm(p=2))                   # 3.741657 (L2 norm)
print(cosine_similarity(u, v))       # 0.974632

# Matrix Multiplications, Inverses, & Determinants
A = Matrix([[1.0, 2.0], [3.0, 4.0]])
B = Matrix([[2.0, 0.0], [1.0, 2.0]])
print(A @ B)                         # Matrix([[4, 4], [10, 8]])
print(A.det())                       # -2.0
print(A.inv())                       # Matrix([[-2, 1], [1.5, -0.5]])

# LU Factorization with Partial Pivoting: P @ A == L @ U
P, L, U = A.lu()

# QR Decomposition & Linear Least Squares Fitting (min ||A*x - b||_2)
design_A = Matrix([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
y_data = Vector([2.1, 3.9, 6.2, 8.0])
Q, R = design_A.qr()
params = solve_least_squares(design_A, y_data)
print(f"Fitted: y = {params[0]:.2f} + {params[1]:.2f}*x")  # y = 0.05 + 2.00*x

# Cholesky Factorization for Symmetric Positive-Definite (SPD) Matrices: A == L @ L.T
A_spd = Matrix([[4.0, 2.0], [2.0, 5.0]])
L_spd = A_spd.cholesky()
print("Cholesky L:", L_spd)          # Matrix([[2, 0], [1, 2]])

# Solve System of Linear Equations: A * x = b
sys_A = Matrix([[2.0, 1.0], [1.0, 3.0]])
sys_b = Vector([5.0, 5.0])
x_sol = solve_linear(sys_A, sys_b)
print("Solution x =", x_sol)         # Vector([2, 1])
```

### 4. Statistics & Ordinary Least Squares (OLS) Regression

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

### 5. Combinatorics & Discrete Mathematics

```python
from c_mathlib.combinatorics import combinations, permutations, euler_totient, nCr, nPr

# Binomial Coefficients (nCr) & Permutations (nPr)
print("5-card poker hands:", combinations(52, 5))   # 2,598,960
print("Permutations P(10, 3):", permutations(10, 3)) # 720

# Euler's Totient Function (Count of coprimes up to n)
print("phi(36) =", euler_totient(36))               # 12
print("phi(100) =", euler_totient(100))             # 40
```

---

## Running Example Demos

Run the interactive demonstration scripts located in `examples/`:

```powershell
python examples/basic.py
python examples/numerical_demo.py
python examples/ode_optimization_demo.py
python examples/special_functions_demo.py
python examples/linear_algebra_demo.py
python examples/decompositions_demo.py
python examples/statistics_demo.py
python examples/combinatorics_demo.py
```

---

## Running Tests

Run the complete test suite (**142 unit tests** covering all C routines and edge cases) using `unittest`:

```powershell
python -m unittest discover -s tests -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
