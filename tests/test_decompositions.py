import unittest
import math
from c_mathlib.linear_algebra import (
    Vector,
    Matrix,
    lu,
    qr,
    cholesky,
    inv,
    inverse,
    solve_least_squares,
)


class TestMatrixDecompositions(unittest.TestCase):
    """Unit tests for matrix factorizations (LU, QR, Cholesky, Inversion, Least Squares)."""

    def assertMatrixAlmostEqual(self, A: Matrix, B: Matrix, places: int = 7):
        """Helper to assert that two matrices are element-wise approximately equal."""
        self.assertEqual(A.shape, B.shape, f"Shape mismatch: {A.shape} != {B.shape}")
        for r in range(A.rows):
            for c in range(A.cols):
                self.assertAlmostEqual(
                    A[r, c],
                    B[r, c],
                    places=places,
                    msg=f"Matrix mismatch at ({r}, {c}): {A[r, c]} != {B[r, c]}",
                )

    # --------------------------------------------------------------------------
    # LU Decomposition Tests
    # --------------------------------------------------------------------------

    def test_lu_standard_3x3(self):
        """Test LU factorization: P @ A == L @ U on 3x3 matrix."""
        A = Matrix([
            [2.0, 1.0, 1.0],
            [4.0, 3.0, 3.0],
            [8.0, 7.0, 9.0],
        ])
        P, L, U = A.lu()

        # Check P is orthogonal permutation matrix: P @ P^T == I
        I3 = Matrix.identity(3)
        self.assertMatrixAlmostEqual(P @ P.T, I3)

        # Check L is unit lower triangular
        for i in range(3):
            self.assertAlmostEqual(L[i, i], 1.0)
            for j in range(i + 1, 3):
                self.assertAlmostEqual(L[i, j], 0.0)

        # Check U is upper triangular
        for i in range(3):
            for j in range(0, i):
                self.assertAlmostEqual(U[i, j], 0.0)

        # Verify identity: P @ A == L @ U
        self.assertMatrixAlmostEqual(P @ A, L @ U)

    def test_lu_functional_wrapper(self):
        """Test functional lu() wrapper."""
        raw_A = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        P, L, U = lu(raw_A)
        A = Matrix(raw_A)
        self.assertMatrixAlmostEqual(P @ A, L @ U)

    def test_lu_singular_and_non_square(self):
        """Ensure singular and rectangular matrices raise ValueError for LU."""
        singular = Matrix([
            [1.0, 2.0],
            [2.0, 4.0],
        ])
        with self.assertRaises(ValueError):
            singular.lu()

        rect = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaises(ValueError):
            rect.lu()

    # --------------------------------------------------------------------------
    # QR Decomposition Tests
    # --------------------------------------------------------------------------

    def test_qr_square_3x3(self):
        """Test QR decomposition on square 3x3 matrix."""
        A = Matrix([
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ])
        Q, R = A.qr()

        # Q must have orthonormal columns: Q^T @ Q == I
        I3 = Matrix.identity(3)
        self.assertMatrixAlmostEqual(Q.T @ Q, I3, places=6)

        # R must be upper triangular
        for i in range(3):
            for j in range(0, i):
                self.assertAlmostEqual(R[i, j], 0.0, places=6)

        # Q @ R must reconstruct A
        self.assertMatrixAlmostEqual(Q @ R, A, places=6)

    def test_qr_overdetermined_rectangular(self):
        """Test QR decomposition on 4x3 overdetermined matrix."""
        A = Matrix([
            [1.0, -1.0, 4.0],
            [1.0, 4.0, -2.0],
            [1.0, 4.0, 2.0],
            [1.0, -1.0, 0.0],
        ])
        Q, R = A.qr()

        self.assertEqual(Q.shape, (4, 3))
        self.assertEqual(R.shape, (3, 3))

        # Q^T @ Q == I_3
        I3 = Matrix.identity(3)
        self.assertMatrixAlmostEqual(Q.T @ Q, I3, places=6)

        # Q @ R == A
        self.assertMatrixAlmostEqual(Q @ R, A, places=6)

    def test_qr_invalid_dimensions(self):
        """QR requires m >= n."""
        underdetermined = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with self.assertRaises(ValueError):
            underdetermined.qr()

    # --------------------------------------------------------------------------
    # Least Squares Solver Tests
    # --------------------------------------------------------------------------

    def test_least_squares_exact_and_overdetermined(self):
        """Test linear least squares fitting using QR."""
        # Overdetermined system fitting line y = 2x + 1 to points (1, 3), (2, 5), (3, 7), (4, 9)
        # [1  1] [c]   [3]
        # [1  2] [m] = [5]
        # [1  3]       [7]
        # [1  4]       [9]
        A = [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
        b = [3.0, 5.0, 7.0, 9.0]
        x = solve_least_squares(A, b)

        self.assertEqual(len(x), 2)
        self.assertAlmostEqual(x[0], 1.0, places=6)  # intercept c
        self.assertAlmostEqual(x[1], 2.0, places=6)  # slope m

    def test_least_squares_invalid(self):
        """Ensure invalid shapes raise ValueError."""
        with self.assertRaises(ValueError):
            solve_least_squares([[1.0, 2.0, 3.0]], [1.0])  # rows < cols
        with self.assertRaises(ValueError):
            solve_least_squares([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0])  # RHS mismatch

    # --------------------------------------------------------------------------
    # Cholesky Factorization Tests
    # --------------------------------------------------------------------------

    def test_cholesky_standard_spd(self):
        """Test Cholesky decomposition A = L @ L^T on SPD matrix."""
        # Symmetric positive-definite matrix
        A = Matrix([
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ])
        L = A.cholesky()

        # Check L is lower triangular
        for i in range(3):
            self.assertGreater(L[i, i], 0.0)  # Positive diagonal
            for j in range(i + 1, 3):
                self.assertAlmostEqual(L[i, j], 0.0)

        # L @ L^T == A
        self.assertMatrixAlmostEqual(L @ L.T, A, places=6)

    def test_cholesky_non_spd_error(self):
        """Ensure non-positive-definite matrices raise ValueError."""
        # Matrix with negative eigenvalue
        non_spd = Matrix([
            [1.0, 2.0],
            [2.0, 1.0],
        ])
        with self.assertRaises(ValueError):
            non_spd.cholesky()

        # Matrix with non-square dimensions
        with self.assertRaises(ValueError):
            Matrix([[1.0, 2.0, 3.0]]).cholesky()

    # --------------------------------------------------------------------------
    # Matrix Inversion Tests
    # --------------------------------------------------------------------------

    def test_matrix_inverse_1x1_and_2x2(self):
        """Test inversion of 1x1 and 2x2 matrices."""
        M1 = Matrix([[4.0]])
        self.assertAlmostEqual(M1.inv()[0, 0], 0.25)

        M2 = Matrix([
            [4.0, 7.0],
            [2.0, 6.0],
        ])
        # det = 24 - 14 = 10 -> inv = [[0.6, -0.7], [-0.2, 0.4]]
        inv2 = M2.inv()
        I2 = Matrix.identity(2)
        self.assertMatrixAlmostEqual(M2 @ inv2, I2, places=6)
        self.assertMatrixAlmostEqual(inv2 @ M2, I2, places=6)
        self.assertMatrixAlmostEqual(inverse(M2), inv2, places=6)

    def test_matrix_inverse_3x3_and_4x4(self):
        """Test inversion on 3x3 and 4x4 invertible matrices."""
        A = Matrix([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ])
        invA = A.inv()
        I3 = Matrix.identity(3)
        self.assertMatrixAlmostEqual(A @ invA, I3, places=6)
        self.assertMatrixAlmostEqual(invA @ A, I3, places=6)

        B = Matrix([
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ])
        invB = B.inv()
        I4 = Matrix.identity(4)
        self.assertMatrixAlmostEqual(B @ invB, I4, places=6)
        self.assertMatrixAlmostEqual(invB @ B, I4, places=6)

    def test_matrix_inverse_singular(self):
        """Singular matrices must raise ValueError."""
        singular = Matrix([
            [1.0, 2.0],
            [2.0, 4.0],
        ])
        with self.assertRaises(ValueError):
            singular.inv()


if __name__ == "__main__":
    unittest.main()
