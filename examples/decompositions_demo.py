"""
Demonstration script showcasing c-mathlib's matrix decompositions (LU, QR, Cholesky, Inversion, Least Squares) in native C.
"""

from c_mathlib.linear_algebra import (
    Vector,
    Matrix,
    lu,
    qr,
    cholesky,
    inverse,
    solve_least_squares,
)


def main():
    print("=== c-mathlib Matrix Decompositions & Least Squares Demonstration ===")
    print("All factorizations and solvers are executed in native C.\n")

    # 1. LU Decomposition with Partial Pivoting (P @ A = L @ U)
    print("--- 1. LU Decomposition with Partial Pivoting (P @ A = L @ U) ---")
    A = Matrix([
        [2.0, 1.0, 1.0],
        [4.0, 3.0, 3.0],
        [8.0, 7.0, 9.0],
    ])
    print("Matrix A (3x3):")
    print(A)

    P, L, U = A.lu()
    print("\nPermutation Matrix P:")
    print(P)
    print("\nUnit Lower Triangular Matrix L:")
    print(L)
    print("\nUpper Triangular Matrix U:")
    print(U)

    # Verify P @ A == L @ U
    print(f"\nVerification: P @ A == L @ U -> {P @ A == L @ U}")

    # 2. QR Decomposition & Overdetermined Least Squares
    print("\n--- 2. QR Decomposition & Overdetermined Least Squares ---")
    # Data points: (1, 2.1), (2, 3.9), (3, 6.2), (4, 8.0) -> fit y = c + m*x
    A_ls = Matrix([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ])
    b_ls = Vector([2.1, 3.9, 6.2, 8.0])

    Q, R = A_ls.qr()
    print("Design Matrix A (4x2):")
    print(A_ls)
    print("\nOrthogonal Matrix Q (4x2):")
    print(Q)
    print("\nUpper Triangular Matrix R (2x2):")
    print(R)
    print(f"Orthonormality Q.T @ Q == I_2 -> {Q.T @ Q == Matrix.identity(2)}")

    x_ls = solve_least_squares(A_ls, b_ls)
    print(f"\nLeast Squares Solution [intercept, slope] = {x_ls}")
    print(f"Fitted Line: y = {x_ls[0]:.3f} + {x_ls[1]:.3f} * x")

    # 3. Cholesky Decomposition for Symmetric Positive-Definite (SPD) Matrices
    print("\n--- 3. Cholesky Factorization (A = L @ L.T) for SPD Matrices ---")
    A_spd = Matrix([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ])
    print("Symmetric Positive-Definite Matrix A (3x3):")
    print(A_spd)

    L_cholesky = A_spd.cholesky()
    print("\nCholesky Factor L (Lower Triangular):")
    print(L_cholesky)
    print(f"Reconstruction L @ L.T == A -> {L_cholesky @ L_cholesky.T == A_spd}")

    # 4. Matrix Inversion via LU Factorization
    print("\n--- 4. Matrix Inversion (A^(-1)) via C LU Engine ---")
    M = Matrix([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [5.0, 6.0, 0.0],
    ])
    print("Matrix M (3x3):")
    print(M)

    M_inv = M.inv()
    print("\nInverse Matrix M^(-1):")
    print(M_inv)
    print(f"Verification M @ M^(-1) == I_3 -> {M @ M_inv == Matrix.identity(3)}")


if __name__ == "__main__":
    main()
