"""
Demonstration script for numerical mathematics in c-mathlib (c_mathlib.numerical).
"""

import math
from c_mathlib.numerical import (
    bisection,
    simpson,
    derivative,
    Polynomial,
)


def main():
    print("=== c-mathlib Numerical Mathematics Demonstration ===")
    print("All numerical routines (Bisection, Simpson, Central Difference) run in native C.\n")

    # 1. Root Finding via Bisection Method
    print("--- 1. Root Finding (bisection) ---")
    
    # Solve x^3 - x - 2 = 0 on [1, 2]
    root_cubic = bisection("x^3 - x - 2", 1.0, 2.0, tol=1e-7)
    print(f"Root of x^3 - x - 2 = 0 on [1, 2]:   x = {root_cubic:.7f}")

    # Solve x^2 - 4 = 0 on [0, 3]
    root_quad = bisection("x^2 - 4", 0.0, 3.0, tol=1e-8)
    print(f"Root of x^2 - 4 = 0 on [0, 3]:       x = {root_quad:.7f}")

    # Solve sin(x) = 0 on [3, 4] -> pi
    root_sin = bisection("sin", 3.0, 4.0, tol=1e-8)
    print(f"Root of sin(x) = 0 on [3, 4]:        x = {root_sin:.7f} (pi ~ {math.pi:.7f})")

    # 2. Numerical Integration via Composite Simpson's 1/3 Rule
    print("\n--- 2. Numerical Integration (simpson) ---")
    
    # Integral of x^2 from 0 to 1 = 1/3
    int_x2 = simpson("x^2", 0.0, 1.0, n=100)
    print(f"integral of x^2 from 0 to 1:         {int_x2:.8f} (exact: 1/3 = {1/3:.8f})")

    # Integral of sin(x) from 0 to pi = 2.0
    int_sin = simpson("sin", 0.0, math.pi, n=100)
    print(f"integral of sin(x) from 0 to pi:     {int_sin:.8f} (exact: 2.00000000)")

    # Integral of exp(x) from 0 to 1 = e - 1
    int_exp = simpson("exp", 0.0, 1.0, n=100)
    print(f"integral of exp(x) from 0 to 1:     {int_exp:.8f} (exact: {math.e - 1:.8f})")

    # 3. Numerical Differentiation via Central Difference
    print("\n--- 3. Numerical Differentiation (derivative) ---")
    
    # d/dx (x^2) at x = 3 is 6.0
    d_x2 = derivative("x^2", 3.0, h=1e-5)
    print(f"d/dx (x^2) at x = 3.0:               {d_x2:.6f} (exact: 6.0)")

    # d/dx (sin x) at x = 0 is 1.0
    d_sin = derivative("sin", 0.0, h=1e-5)
    print(f"d/dx (sin x) at x = 0.0:             {d_sin:.6f} (exact: 1.0)")

    # d/dx (exp x) at x = 1 is e
    d_exp = derivative("exp", 1.0, h=1e-5)
    print(f"d/dx (exp x) at x = 1.0:             {d_exp:.6f} (exact: {math.e:.6f})")

    # 4. Polynomial Evaluation and Calculus via Horner's Method
    print("\n--- 4. Arbitrary Polynomials in Native C ---")
    # P(x) = 5 - 2x + 4x^3
    poly = Polynomial([5, -2, 0, 4])
    print(f"Polynomial P(x) = {poly}")
    
    # Derivative P'(2) = -2 + 12(4) = 46
    p_prime_2 = derivative(poly, 2.0)
    print(f"P'(2.0) =                           {p_prime_2:.6f} (exact: 46.0)")

    # Integral_0^1 (5 - 2x + 4x^3) dx = [5x - x^2 + x^4]_0^1 = 5 - 1 + 1 = 5.0
    p_int = simpson(poly, 0.0, 1.0, n=100)
    print(f"integral of P(x) from 0 to 1:        {p_int:.8f} (exact: 5.00000000)")


if __name__ == "__main__":
    main()
