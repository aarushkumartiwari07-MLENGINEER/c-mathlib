"""
Demonstration script showcasing c-mathlib's native C linear algebra engine.
"""

from c_mathlib.linear_algebra import (
    Vector,
    Matrix,
    dot,
    norm,
    cross,
    cosine_similarity,
    matrix_mult,
    transpose,
    determinant,
    solve_linear,
)


def main():
    print("=== c-mathlib Linear Algebra Demonstration ===")
    print("All matrix and vector computations are executed in native C.\n")

    # 1. Vector Operations
    print("--- 1. Vector Operations ---")
    u = Vector([1.0, 2.0, 3.0])
    v = Vector([4.0, 5.0, 6.0])
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"u + v = {u + v}")
    print(f"u - v = {u - v}")
    print(f"3.0 * u = {3.0 * u}")
    print(f"u @ v (dot product) = {u @ v}")
    print(f"norm(u, p=1) = {u.norm(p=1):.4f}")
    print(f"norm(u, p=2) = {u.norm(p=2):.4f}")
    print(f"norm(u, p=inf) = {u.norm(p=float('inf')):.4f}")
    print(f"u normalized = {(1.0 / u.norm(2)) * u}")
    print(f"cosine_similarity(u, v) = {cosine_similarity(u, v):.6f}")

    # 3D Cross Product
    i_hat = Vector([1.0, 0.0, 0.0])
    j_hat = Vector([0.0, 1.0, 0.0])
    k_hat = i_hat.cross(j_hat)
    print(f"i_hat x j_hat = {k_hat} (should be [0, 0, 1])")

    # 2. Matrix Arithmetic & Transformations
    print("\n--- 2. Matrix Multiplication & Transformations ---")
    A = Matrix([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    B = Matrix([
        [7.0, 8.0],
        [9.0, 1.0],
        [2.0, 3.0],
    ])
    print("Matrix A (2x3):")
    print(A)
    print("\nMatrix B (3x2):")
    print(B)

    C = A @ B
    print("\nMatrix C = A @ B (2x2) [C GEMM engine]:")
    print(C)

    print("\nA.T (Transpose of A):")
    print(A.T)

    # 3. Determinant & Invertibility
    print("\n--- 3. Determinant via Gaussian Elimination (Partial Pivoting) ---")
    M = Matrix([
        [6.0, 1.0, 1.0],
        [4.0, -2.0, 5.0],
        [2.0, 8.0, 7.0],
    ])
    print("Matrix M (3x3):")
    print(M)
    det_M = M.determinant()
    print(f"det(M) = {det_M:.4f}")

    # 4. Solving Linear Systems A x = b
    print("\n--- 4. Solving Systems of Linear Equations (A x = b) ---")
    # System:
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    A_sys = Matrix([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ])
    b_sys = Vector([8.0, -11.0, -3.0])

    print("System Coefficient Matrix A:")
    print(A_sys)
    print(f"Right-hand side b: {b_sys}")

    x_sol = solve_linear(A_sys, b_sys)
    print(f"Solution vector x = {x_sol}")
    print(f"Verification A @ x = {A_sys @ x_sol} (Expected: {b_sys})")


if __name__ == "__main__":
    main()
