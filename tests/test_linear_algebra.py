import unittest
import math
from c_mathlib.linear_algebra import (
    Vector,
    Matrix,
    dot,
    norm,
    cosine_similarity,
    cross,
    matrix_mult,
    transpose,
    determinant,
    solve_linear,
)


class TestLinearAlgebra(unittest.TestCase):
    """Unit tests for c-mathlib linear algebra module."""

    # --------------------------------------------------------------------------
    # Vector Tests
    # --------------------------------------------------------------------------

    def test_vector_creation_and_properties(self):
        """Test vector initialization and basic properties."""
        v = Vector([1, 2, 3])
        self.assertEqual(v.dim, 3)
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[2], 3.0)
        self.assertEqual(v.to_list(), [1.0, 2.0, 3.0])
        self.assertIn("Vector", repr(v))

        zeros = Vector.zeros(4)
        self.assertEqual(zeros.to_list(), [0.0, 0.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            Vector([])
        with self.assertRaises(TypeError):
            Vector("123")  # type: ignore

    def test_vector_dot_product(self):
        """Test vector dot product via .dot() and @ operator."""
        u = Vector([1, 2, 3])
        v = Vector([4, 5, 6])
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertAlmostEqual(u.dot(v), 32.0)
        self.assertAlmostEqual(u @ v, 32.0)
        self.assertAlmostEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)

        with self.assertRaises(ValueError):
            u.dot(Vector([1, 2]))

    def test_vector_norms(self):
        """Test L1, L2, and Linf norms."""
        v = Vector([3, -4])
        # L1: |3| + |-4| = 7
        self.assertAlmostEqual(v.norm(p=1), 7.0)
        # L2: sqrt(9 + 16) = 5
        self.assertAlmostEqual(v.norm(p=2), 5.0)
        self.assertAlmostEqual(v.norm(), 5.0)
        # Linf: max(|3|, |-4|) = 4
        self.assertAlmostEqual(v.norm(p=0), 4.0)
        self.assertAlmostEqual(v.norm(p="inf"), 4.0)
        self.assertAlmostEqual(norm([3, -4], p=2), 5.0)

    def test_vector_arithmetic(self):
        """Test vector addition, subtraction, and scaling."""
        u = Vector([1, 2, 3])
        v = Vector([4, 5, 6])

        self.assertEqual((u + v).to_list(), [5.0, 7.0, 9.0])
        self.assertEqual((v - u).to_list(), [3.0, 3.0, 3.0])
        self.assertEqual((u * 2.5).to_list(), [2.5, 5.0, 7.5])
        self.assertEqual((2.5 * u).to_list(), [2.5, 5.0, 7.5])

    def test_vector_cross_product(self):
        """Test 3D cross product."""
        i = Vector([1, 0, 0])
        j = Vector([0, 1, 0])
        k = Vector([0, 0, 1])

        # i x j = k
        self.assertEqual(i.cross(j), k)
        # j x i = -k
        self.assertEqual(j.cross(i), Vector([0, 0, -1]))
        # cross functional wrapper
        self.assertEqual(cross([1, 0, 0], [0, 1, 0]), k)

        with self.assertRaises(ValueError):
            Vector([1, 2]).cross(Vector([3, 4]))

    def test_cosine_similarity(self):
        """Test cosine similarity between vectors."""
        u = Vector([1, 2, 3])
        v = Vector([2, 4, 6])  # parallel
        self.assertAlmostEqual(u.cosine_similarity(v), 1.0)

        w = Vector([-1, -2, -3])  # opposite
        self.assertAlmostEqual(u.cosine_similarity(w), -1.0)

        x = Vector([1, 0])
        y = Vector([0, 1])  # orthogonal
        self.assertAlmostEqual(x.cosine_similarity(y), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    # --------------------------------------------------------------------------
    # Matrix Tests
    # --------------------------------------------------------------------------

    def test_matrix_creation_and_properties(self):
        """Test matrix initialization, shapes, indexing."""
        A = Matrix([[1, 2], [3, 4]])
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(A.rows, 2)
        self.assertEqual(A.cols, 2)
        self.assertTrue(A.is_square)
        self.assertEqual(A[0, 1], 2.0)
        self.assertEqual(A[1, 0], 3.0)
        self.assertEqual(A[0], [1.0, 2.0])

        I = Matrix.identity(3)
        self.assertEqual(I.to_list(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        Z = Matrix.zeros(2, 3)
        self.assertEqual(Z.shape, (2, 3))
        self.assertFalse(Z.is_square)

        with self.assertRaises(ValueError):
            Matrix([])
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])  # Ragged rows

    def test_matrix_transpose(self):
        """Test matrix transpose."""
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        AT = A.T
        self.assertEqual(AT.shape, (3, 2))
        self.assertEqual(AT.to_list(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(AT.T, A)
        self.assertEqual(transpose([[1, 2], [3, 4]]), Matrix([[1, 3], [2, 4]]))

    def test_matrix_trace(self):
        """Test matrix trace."""
        A = Matrix([[5, 1, 2], [3, 7, 4], [1, 2, 9]])
        self.assertAlmostEqual(A.trace(), 21.0)  # 5 + 7 + 9 = 21

        rect = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            rect.trace()

    def test_matrix_multiplication(self):
        """Test matrix multiplication A @ B and A @ v."""
        # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        # A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = A @ B
        self.assertEqual(C.to_list(), [[19.0, 22.0], [43.0, 50.0]])
        self.assertEqual(matrix_mult(A, B).to_list(), [[19.0, 22.0], [43.0, 50.0]])

        # Matrix @ Vector
        v = Vector([1, 2])
        Av = A @ v
        # [[1, 2], [3, 4]] @ [1, 2] = [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
        self.assertEqual(Av.to_list(), [5.0, 11.0])

        with self.assertRaises(ValueError):
            A @ Matrix([[1, 2, 3]])  # Incompatible dimensions

    def test_matrix_determinant(self):
        """Test matrix determinant via Gaussian elimination."""
        # 2x2: det([[1, 2], [3, 4]]) = 4 - 6 = -2
        A = Matrix([[1, 2], [3, 4]])
        self.assertAlmostEqual(A.det(), -2.0)
        self.assertAlmostEqual(determinant([[1, 2], [3, 4]]), -2.0)

        # 3x3: det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) = -306
        B = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        self.assertAlmostEqual(B.det(), -306.0)

        # Singular matrix (linearly dependent rows)
        C = Matrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
        self.assertAlmostEqual(C.det(), 0.0)

    def test_solve_linear_system(self):
        """Test solving A * x = b."""
        # 2x + y = 5
        # x + 3y = 5
        # => x = 2, y = 1
        A = Matrix([[2, 1], [1, 3]])
        b = Vector([5, 5])
        x = A.solve(b)
        self.assertAlmostEqual(x[0], 2.0)
        self.assertAlmostEqual(x[1], 1.0)
        self.assertEqual(solve_linear(A, b), Vector([2.0, 1.0]))

        # Verify A * x == b
        residual = (A @ x) - b
        self.assertAlmostEqual(residual.norm(), 0.0, places=10)

        # 3x3 System:
        # 2x + y - z = 8
        # -3x - y + 2z = -11
        # -2x + y + 2z = -3
        # Solution: x = 2, y = 3, z = -1
        A3 = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b3 = Vector([8, -11, -3])
        x3 = A3.solve(b3)
        self.assertAlmostEqual(x3[0], 2.0)
        self.assertAlmostEqual(x3[1], 3.0)
        self.assertAlmostEqual(x3[2], -1.0)

        # Singular matrix raises ValueError
        singular_A = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            singular_A.solve(Vector([1, 2]))


if __name__ == "__main__":
    unittest.main()
