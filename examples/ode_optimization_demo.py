"""
Demo script for ODE Initial-Value Solvers (RK4) and 1D Optimization.
"""

import math
from c_mathlib.numerical import rk4, minimize_1d, Polynomial


def demo_ode_solvers():
    print("=" * 60)
    print("1. RUNGE-KUTTA 4TH ORDER (RK4) ODE SOLVER")
    print("=" * 60)

    # 1. Exponential decay: dy/dt = -y, y(0) = 1.0 over [0, 2]
    res_decay = rk4("exp_decay", y0=1.0, t_span=(0.0, 2.0), steps=20)
    exact_decay = math.exp(-2.0)
    print(f"Model: dy/dt = -y, y(0) = 1.0 on [0, 2]")
    print(f"  Result: {res_decay}")
    print(f"  Computed y(2.0) = {res_decay.y_final:.8f}")
    print(f"  Exact    y(2.0) = {exact_decay:.8f}")
    print(f"  Absolute Error  = {abs(res_decay.y_final - exact_decay):.2e}")

    # 2. Logistic Population Growth: dy/dt = y * (1 - y), y(0) = 0.1 over [0, 6]
    res_log = rk4("logistic", y0=0.1, t_span=(0.0, 6.0), steps=30)
    exact_log = 1.0 / (1.0 + 9.0 * math.exp(-6.0))
    print(f"\nModel: dy/dt = y*(1 - y), y(0) = 0.1 on [0, 6]")
    print(f"  Result: {res_log}")
    print(f"  Computed y(6.0) = {res_log.y_final:.8f}")
    print(f"  Exact    y(6.0) = {exact_log:.8f}")

    # 3. Polynomial derivative: dy/dt = 3t^2, y(0) = 0.0 -> exact y(t) = t^3
    poly_deriv = Polynomial([0, 0, 3])  # 3*t^2
    res_poly = rk4(poly_deriv, y0=0.0, t_span=(0.0, 3.0), steps=30)
    print(f"\nModel: dy/dt = 3*t^2, y(0) = 0.0 on [0, 3]")
    print(f"  Computed y(3.0) = {res_poly.y_final:.8f} (Exact: 27.0)")


def demo_optimization():
    print("\n" + "=" * 60)
    print("2. GOLDEN SECTION 1D FUNCTION OPTIMIZATION")
    print("=" * 60)

    # 1. Quadratic Polynomial: P(x) = (x - 4)^2 + 3 = x^2 - 8x + 19 on [0, 10]
    p = Polynomial([19, -8, 1])
    opt_p = minimize_1d(p, bracket=(0.0, 10.0), tol=1e-8)
    print(f"Objective: P(x) = (x - 4)^2 + 3 on [0, 10]")
    print(f"  Optimal x*   = {opt_p.x:.8f} (Expected: 4.0)")
    print(f"  Minimum f(x) = {opt_p.fun:.8f} (Expected: 3.0)")
    print(f"  Iterations   = {opt_p.nit}, Converged: {opt_p.converged}")

    # 2. Trigonometric minimum: sin(x) on [3.0, 6.0] -> min at 3*pi/2
    opt_sin = minimize_1d("sin", bracket=(3.0, 6.0), tol=1e-8)
    print(f"\nObjective: f(x) = sin(x) on [3.0, 6.0]")
    print(f"  Optimal x*   = {opt_sin.x:.8f} (Expected 3*pi/2: {1.5 * math.pi:.8f})")
    print(f"  Minimum f(x) = {opt_sin.fun:.8f} (Expected: -1.0)")
    print(f"  Iterations   = {opt_sin.nit}, Converged: {opt_sin.converged}")


if __name__ == "__main__":
    demo_ode_solvers()
    demo_optimization()
