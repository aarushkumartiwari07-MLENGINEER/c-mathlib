"""
Linear Algebra module for c-mathlib.

Provides high-performance vector and matrix algebra, norms, dot products,
matrix multiplication, determinants, and linear system solvers executed in native C.
"""

import ctypes
import math
from typing import Iterator, List, Sequence, Tuple, Union
from .core import _lib


# ==============================================================================
# Ctypes Signatures Configuration
# ==============================================================================

_c_double_p = ctypes.POINTER(ctypes.c_double)

# Vector Foreign Functions
_c_vector_dot = _lib.vector_dot
_c_vector_dot.argtypes = [_c_double_p, _c_double_p, ctypes.c_int]
_c_vector_dot.restype = ctypes.c_double

_c_vector_norm = _lib.vector_norm
_c_vector_norm.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_int]
_c_vector_norm.restype = ctypes.c_double

_c_vector_add = _lib.vector_add
_c_vector_add.argtypes = [_c_double_p, _c_double_p, ctypes.c_int, _c_double_p]
_c_vector_add.restype = ctypes.c_int

_c_vector_sub = _lib.vector_sub
_c_vector_sub.argtypes = [_c_double_p, _c_double_p, ctypes.c_int, _c_double_p]
_c_vector_sub.restype = ctypes.c_int

_c_vector_scale = _lib.vector_scale
_c_vector_scale.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_double, _c_double_p]
_c_vector_scale.restype = ctypes.c_int

_c_vector_cross_3d = _lib.vector_cross_3d
_c_vector_cross_3d.argtypes = [_c_double_p, _c_double_p, _c_double_p]
_c_vector_cross_3d.restype = ctypes.c_int

_c_vector_cosine_similarity = _lib.vector_cosine_similarity
_c_vector_cosine_similarity.argtypes = [_c_double_p, _c_double_p, ctypes.c_int]
_c_vector_cosine_similarity.restype = ctypes.c_double

# Matrix Foreign Functions
_c_matrix_multiply = _lib.matrix_multiply
_c_matrix_multiply.argtypes = [
    _c_double_p,
    _c_double_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    _c_double_p,
]
_c_matrix_multiply.restype = ctypes.c_int

_c_matrix_transpose = _lib.matrix_transpose
_c_matrix_transpose.argtypes = [
    _c_double_p,
    ctypes.c_int,
    ctypes.c_int,
    _c_double_p,
]
_c_matrix_transpose.restype = ctypes.c_int

_c_matrix_trace = _lib.matrix_trace
_c_matrix_trace.argtypes = [_c_double_p, ctypes.c_int]
_c_matrix_trace.restype = ctypes.c_double

_c_matrix_determinant = _lib.matrix_determinant
_c_matrix_determinant.argtypes = [_c_double_p, ctypes.c_int, _c_double_p]
_c_matrix_determinant.restype = ctypes.c_int

_c_solve_linear_system = _lib.solve_linear_system
_c_solve_linear_system.argtypes = [
    _c_double_p,
    _c_double_p,
    ctypes.c_int,
    _c_double_p,
]
_c_solve_linear_system.restype = ctypes.c_int

_c_int_p = ctypes.POINTER(ctypes.c_int)

_c_matrix_lu = _lib.matrix_lu
_c_matrix_lu.argtypes = [_c_double_p, ctypes.c_int, _c_double_p, _c_double_p, _c_int_p]
_c_matrix_lu.restype = ctypes.c_int

_c_lu_solve = _lib.lu_solve
_c_lu_solve.argtypes = [_c_double_p, _c_double_p, _c_int_p, _c_double_p, ctypes.c_int, _c_double_p]
_c_lu_solve.restype = ctypes.c_int

_c_matrix_qr = _lib.matrix_qr
_c_matrix_qr.argtypes = [_c_double_p, ctypes.c_int, ctypes.c_int, _c_double_p, _c_double_p]
_c_matrix_qr.restype = ctypes.c_int

_c_qr_solve = _lib.qr_solve
_c_qr_solve.argtypes = [_c_double_p, _c_double_p, _c_double_p, ctypes.c_int, ctypes.c_int, _c_double_p]
_c_qr_solve.restype = ctypes.c_int

_c_matrix_cholesky = _lib.matrix_cholesky
_c_matrix_cholesky.argtypes = [_c_double_p, ctypes.c_int, _c_double_p]
_c_matrix_cholesky.restype = ctypes.c_int

_c_cholesky_solve = _lib.cholesky_solve
_c_cholesky_solve.argtypes = [_c_double_p, _c_double_p, ctypes.c_int, _c_double_p]
_c_cholesky_solve.restype = ctypes.c_int

_c_matrix_inverse = _lib.matrix_inverse
_c_matrix_inverse.argtypes = [_c_double_p, ctypes.c_int, _c_double_p]
_c_matrix_inverse.restype = ctypes.c_int


# ==============================================================================
# Vector Class
# ==============================================================================

class Vector:
    """
    Mathematical vector supporting n-dimensional arithmetic operations executed in C.
    """

    def __init__(self, elements: Union["Vector", Sequence[Union[int, float]]]):
        if isinstance(elements, Vector):
            self._data = list(elements._data)
            return

        if not hasattr(elements, "__iter__") or isinstance(elements, (str, bytes)):
            raise TypeError("Vector elements must be an iterable sequence of numbers")
        
        parsed = []
        for val in elements:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Vector elements must be numeric, got {type(val).__name__}")
            parsed.append(float(val))

        if not parsed:
            raise ValueError("Vector cannot be empty")

        self._data = parsed


    @classmethod
    def zeros(cls, n: int) -> "Vector":
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {n}")
        return cls([0.0] * n)

    @property
    def dim(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def to_list(self) -> List[float]:
        return list(self._data)

    def _to_c_array(self):
        return (ctypes.c_double * len(self._data))(*self._data)

    def dot(self, other: Union["Vector", Sequence[Union[int, float]]]) -> float:
        """Compute the dot product (inner product) with another vector."""
        if not isinstance(other, Vector):
            other = Vector(other)
        if self.dim != other.dim:
            raise ValueError(f"Vector dimensions must match for dot product: {self.dim} != {other.dim}")

        u_arr = self._to_c_array()
        v_arr = other._to_c_array()
        return float(_c_vector_dot(u_arr, v_arr, self.dim))

    def norm(self, p: Union[int, float, str] = 2) -> float:
        """
        Compute the p-norm of the vector.
        - p=1: L1 norm (Manhattan distance)
        - p=2: L2 norm (Euclidean distance)
        - p=inf or p=0: Linf norm (Chebyshev norm)
        """
        if p == float("inf") or p == "inf" or p == 0:
            c_p = 0
        elif p == 1:
            c_p = 1
        elif p == 2:
            c_p = 2
        else:
            raise ValueError(f"Supported norms are p=1, p=2, or p=inf (0), got {p}")

        v_arr = self._to_c_array()
        return float(_c_vector_norm(v_arr, self.dim, c_p))

    def cosine_similarity(self, other: Union["Vector", Sequence[Union[int, float]]]) -> float:
        """Compute cosine similarity between two vectors."""
        if not isinstance(other, Vector):
            other = Vector(other)
        if self.dim != other.dim:
            raise ValueError(f"Vector dimensions must match for cosine similarity: {self.dim} != {other.dim}")

        u_arr = self._to_c_array()
        v_arr = other._to_c_array()
        return float(_c_vector_cosine_similarity(u_arr, v_arr, self.dim))

    def cross(self, other: Union["Vector", Sequence[Union[int, float]]]) -> "Vector":
        """Compute the 3D cross product with another vector."""
        if not isinstance(other, Vector):
            other = Vector(other)
        if self.dim != 3 or other.dim != 3:
            raise ValueError(f"Cross product is defined for 3D vectors only, got dim={self.dim} and dim={other.dim}")

        u_arr = self._to_c_array()
        v_arr = other._to_c_array()
        out_arr = (ctypes.c_double * 3)()
        _c_vector_cross_3d(u_arr, v_arr, out_arr)
        return Vector(list(out_arr))

    def __add__(self, other: Union["Vector", Sequence[Union[int, float]]]) -> "Vector":
        if not isinstance(other, Vector):
            other = Vector(other)
        if self.dim != other.dim:
            raise ValueError(f"Vector dimensions must match for addition: {self.dim} != {other.dim}")

        u_arr = self._to_c_array()
        v_arr = other._to_c_array()
        out_arr = (ctypes.c_double * self.dim)()
        _c_vector_add(u_arr, v_arr, self.dim, out_arr)
        return Vector(list(out_arr))

    def __sub__(self, other: Union["Vector", Sequence[Union[int, float]]]) -> "Vector":
        if not isinstance(other, Vector):
            other = Vector(other)
        if self.dim != other.dim:
            raise ValueError(f"Vector dimensions must match for subtraction: {self.dim} != {other.dim}")

        u_arr = self._to_c_array()
        v_arr = other._to_c_array()
        out_arr = (ctypes.c_double * self.dim)()
        _c_vector_sub(u_arr, v_arr, self.dim, out_arr)
        return Vector(list(out_arr))

    def __mul__(self, scalar: Union[int, float]) -> "Vector":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise TypeError(f"Scalar multiplication requires a numeric scalar, got {type(scalar).__name__}")
        v_arr = self._to_c_array()
        out_arr = (ctypes.c_double * self.dim)()
        _c_vector_scale(v_arr, self.dim, float(scalar), out_arr)
        return Vector(list(out_arr))

    def __rmul__(self, scalar: Union[int, float]) -> "Vector":
        return self.__mul__(scalar)

    def __matmul__(self, other: Union["Vector", Sequence[Union[int, float]]]) -> float:
        return self.dot(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            if hasattr(other, "__iter__"):
                try:
                    other = Vector(other)  # type: ignore
                except Exception:
                    return False
            else:
                return False
        if self.dim != other.dim:
            return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self._data, other._data))

    def __repr__(self) -> str:
        formatted = ", ".join(f"{x:g}" for x in self._data)
        return f"Vector([{formatted}])"


# ==============================================================================
# Matrix Class
# ==============================================================================

class Matrix:
    """
    2D Matrix supporting high-performance linear algebra operations in native C.
    Stored in row-major layout.
    """

    def __init__(self, data: Union["Matrix", Sequence[Sequence[Union[int, float]]]]):
        if isinstance(data, Matrix):
            self._rows = data._rows
            self._cols = data._cols
            self._flat = list(data._flat)
            return

        if not hasattr(data, "__iter__") or isinstance(data, (str, bytes)):
            raise TypeError("Matrix data must be a 2D sequence of numbers")

        rows = []

        num_cols = None
        for r_idx, row in enumerate(data):
            if not hasattr(row, "__iter__") or isinstance(row, (str, bytes)):
                raise TypeError(f"Matrix row {r_idx} must be an iterable sequence of numbers")
            parsed_row = []
            for val in row:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"Matrix elements must be numeric, got {type(val).__name__}")
                parsed_row.append(float(val))
            
            if num_cols is None:
                num_cols = len(parsed_row)
                if num_cols == 0:
                    raise ValueError("Matrix rows cannot be empty")
            elif len(parsed_row) != num_cols:
                raise ValueError("All rows in a Matrix must have the same number of columns")
            
            rows.append(parsed_row)

        if not rows:
            raise ValueError("Matrix cannot be empty")

        self._rows = len(rows)
        self._cols = num_cols
        # Flattened row-major array
        self._flat = [elem for row in rows for elem in row]

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Matrix dimensions must be positive integers, got ({rows}, {cols})")
        return cls([[0.0] * cols for _ in range(rows)])

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        if n <= 0:
            raise ValueError(f"Identity matrix dimension must be a positive integer, got {n}")
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return cls(data)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def is_square(self) -> bool:
        return self._rows == self._cols

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[List[float], float]:
        if isinstance(key, tuple):
            r, c = key
            if not (0 <= r < self._rows and 0 <= c < self._cols):
                raise IndexError(f"Matrix index out of range: ({r}, {c}) for shape {self.shape}")
            return self._flat[r * self._cols + c]
        elif isinstance(key, int):
            if not (0 <= key < self._rows):
                raise IndexError(f"Matrix row index out of range: {key} for {self._rows} rows")
            start = key * self._cols
            return self._flat[start : start + self._cols]
        raise TypeError(f"Invalid matrix index type: {type(key).__name__}")

    def to_list(self) -> List[List[float]]:
        return [self._flat[i * self._cols : (i + 1) * self._cols] for i in range(self._rows)]

    def _to_c_array(self):
        return (ctypes.c_double * len(self._flat))(*self._flat)

    @property
    def T(self) -> "Matrix":
        """Compute matrix transpose A^T in C."""
        in_arr = self._to_c_array()
        out_arr = (ctypes.c_double * (self._rows * self._cols))()
        _c_matrix_transpose(in_arr, self._rows, self._cols, out_arr)
        transposed_data = [
            list(out_arr[i * self._rows : (i + 1) * self._rows]) for i in range(self._cols)
        ]
        return Matrix(transposed_data)

    def trace(self) -> float:
        """Compute the trace (sum of main diagonal elements) in C."""
        if not self.is_square:
            raise ValueError(f"Trace is defined for square matrices only, got shape {self.shape}")
        arr = self._to_c_array()
        return float(_c_matrix_trace(arr, self._rows))

    def det(self) -> float:
        """Compute the determinant of a square matrix via Gaussian elimination in C."""
        if not self.is_square:
            raise ValueError(f"Determinant is defined for square matrices only, got shape {self.shape}")
        arr = self._to_c_array()
        c_det = ctypes.c_double()
        status = _c_matrix_determinant(arr, self._rows, ctypes.byref(c_det))
        if status < 0:
            raise ValueError(f"Failed to compute determinant, error code {status}")
        return float(c_det.value)

    def determinant(self) -> float:
        """Alias for det()."""
        return self.det()

    def solve(self, b: Union[Vector, Sequence[Union[int, float]]]) -> Vector:
        """
        Solve linear system A * x = b for x using Gaussian elimination with partial pivoting in C.
        """
        if not self.is_square:
            raise ValueError(f"Coefficient matrix must be square to solve A*x = b, got shape {self.shape}")
        if not isinstance(b, Vector):
            b = Vector(b)
        if b.dim != self._rows:
            raise ValueError(
                f"Dimension mismatch between matrix ({self._rows}x{self._cols}) and RHS vector ({b.dim})"
            )

        A_arr = self._to_c_array()
        b_arr = b._to_c_array()
        x_arr = (ctypes.c_double * self._rows)()

        status = _c_solve_linear_system(A_arr, b_arr, self._rows, x_arr)
        if status == -1:
            raise ValueError("Matrix is singular or nearly singular; linear system has no unique solution")
        elif status < 0:
            raise ValueError(f"Linear solver failed with error code {status}")

        return Vector(list(x_arr))

    def lu(self) -> Tuple["Matrix", "Matrix", "Matrix"]:
        """
        Compute LU decomposition with partial pivoting: P @ A = L @ U.
        
        Returns
        -------
        Tuple[Matrix, Matrix, Matrix]
            (P, L, U) where P is the permutation matrix, L is unit lower triangular,
            and U is upper triangular.
        """
        if not self.is_square:
            raise ValueError(f"LU decomposition requires a square matrix, got shape {self.shape}")
        
        n = self._rows
        A_arr = self._to_c_array()
        L_arr = (ctypes.c_double * (n * n))()
        U_arr = (ctypes.c_double * (n * n))()
        P_arr = (ctypes.c_int * n)()

        status = _c_matrix_lu(A_arr, n, L_arr, U_arr, P_arr)
        if status == -1:
            raise ValueError("Matrix is singular; LU decomposition cannot be completed")
        elif status < 0:
            raise ValueError(f"LU decomposition failed with error code {status}")

        L_data = [list(L_arr[i * n : (i + 1) * n]) for i in range(n)]
        U_data = [list(U_arr[i * n : (i + 1) * n]) for i in range(n)]
        P_data = [[1.0 if j == P_arr[i] else 0.0 for j in range(n)] for i in range(n)]

        return (Matrix(P_data), Matrix(L_data), Matrix(U_data))

    def qr(self) -> Tuple["Matrix", "Matrix"]:
        """
        Compute QR decomposition of an m x n matrix (m >= n): A = Q @ R.
        
        Returns
        -------
        Tuple[Matrix, Matrix]
            (Q, R) where Q is an m x n orthogonal matrix (Q^T @ Q = I)
            and R is an n x n upper triangular matrix.
        """
        m, n = self._rows, self._cols
        if m < n:
            raise ValueError(f"QR decomposition requires rows >= cols (m >= n), got shape {self.shape}")

        A_arr = self._to_c_array()
        Q_arr = (ctypes.c_double * (m * n))()
        R_arr = (ctypes.c_double * (n * n))()

        status = _c_matrix_qr(A_arr, m, n, Q_arr, R_arr)
        if status == -1:
            raise ValueError("Matrix has linearly dependent columns; QR decomposition failed")
        elif status < 0:
            raise ValueError(f"QR decomposition failed with error code {status}")

        Q_data = [list(Q_arr[i * n : (i + 1) * n]) for i in range(m)]
        R_data = [list(R_arr[i * n : (i + 1) * n]) for i in range(n)]

        return (Matrix(Q_data), Matrix(R_data))

    def cholesky(self) -> "Matrix":
        """
        Compute Cholesky decomposition of a symmetric positive-definite matrix: A = L @ L^T.
        
        Returns
        -------
        Matrix
            Lower triangular matrix L with positive diagonal elements.
        """
        if not self.is_square:
            raise ValueError(f"Cholesky decomposition requires a square symmetric matrix, got shape {self.shape}")

        n = self._rows
        A_arr = self._to_c_array()
        L_arr = (ctypes.c_double * (n * n))()

        status = _c_matrix_cholesky(A_arr, n, L_arr)
        if status == -1:
            raise ValueError("Matrix is not symmetric positive-definite; Cholesky decomposition failed")
        elif status < 0:
            raise ValueError(f"Cholesky decomposition failed with error code {status}")

        L_data = [list(L_arr[i * n : (i + 1) * n]) for i in range(n)]
        return Matrix(L_data)

    def inv(self) -> "Matrix":
        """
        Compute the multiplicative inverse of a square matrix A^(-1) in native C via LU decomposition.
        """
        if not self.is_square:
            raise ValueError(f"Matrix inverse requires a square matrix, got shape {self.shape}")

        n = self._rows
        A_arr = self._to_c_array()
        inv_arr = (ctypes.c_double * (n * n))()

        status = _c_matrix_inverse(A_arr, n, inv_arr)
        if status == -1:
            raise ValueError("Matrix is singular and cannot be inverted")
        elif status < 0:
            raise ValueError(f"Matrix inversion failed with error code {status}")

        inv_data = [list(inv_arr[i * n : (i + 1) * n]) for i in range(n)]
        return Matrix(inv_data)

    def inverse(self) -> "Matrix":
        """Alias for inv()."""
        return self.inv()

    def __matmul__(self, other: Union["Matrix", Vector, Sequence[Sequence[Union[int, float]]]]) -> Union["Matrix", Vector]:
        if isinstance(other, Vector):
            # Matrix-vector multiplication: A (m x n) * v (n) -> Vector (m)
            if self._cols != other.dim:
                raise ValueError(
                    f"Shape mismatch in Matrix @ Vector: matrix has {self._cols} columns, vector has dimension {other.dim}"
                )
            v_mat = Matrix([[x] for x in other])
            res_mat = self.__matmul__(v_mat)
            assert isinstance(res_mat, Matrix)
            return Vector([row[0] for row in res_mat.to_list()])

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self._cols != other.rows:
            raise ValueError(
                f"Incompatible shapes for matrix multiplication: {self.shape} @ {other.shape}"
            )

        A_arr = self._to_c_array()
        B_arr = other._to_c_array()
        C_arr = (ctypes.c_double * (self._rows * other.cols))()

        status = _c_matrix_multiply(A_arr, B_arr, self._rows, self._cols, other.cols, C_arr)
        if status < 0:
            raise ValueError(f"Matrix multiplication failed with status {status}")

        result_rows = [
            list(C_arr[i * other.cols : (i + 1) * other.cols]) for i in range(self._rows)
        ]
        return Matrix(result_rows)

    def __mul__(self, scalar: Union[int, float]) -> "Matrix":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise TypeError(f"Matrix scalar multiplication requires a number, got {type(scalar).__name__}")
        s = float(scalar)
        scaled = [[val * s for val in row] for row in self.to_list()]
        return Matrix(scaled)

    def __rmul__(self, scalar: Union[int, float]) -> "Matrix":
        return self.__mul__(scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            if hasattr(other, "__iter__"):
                try:
                    other = Matrix(other)  # type: ignore
                except Exception:
                    return False
            else:
                return False
        if self.shape != other.shape:
            return False
        return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(self._flat, other._flat))

    def __repr__(self) -> str:
        rows_str = ", ".join(f"[{', '.join(f'{x:g}' for x in row)}]" for row in self.to_list())
        return f"Matrix([{rows_str}])"


# ==============================================================================
# Functional API Convenience Wrappers
# ==============================================================================

def dot(u: Sequence[Union[int, float]], v: Sequence[Union[int, float]]) -> float:
    """Compute dot product of two vectors."""
    return Vector(u).dot(Vector(v))


def norm(v: Sequence[Union[int, float]], p: Union[int, float, str] = 2) -> float:
    """Compute vector p-norm."""
    return Vector(v).norm(p=p)


def cosine_similarity(u: Sequence[Union[int, float]], v: Sequence[Union[int, float]]) -> float:
    """Compute cosine similarity between two vectors."""
    return Vector(u).cosine_similarity(Vector(v))


def cross(u: Sequence[Union[int, float]], v: Sequence[Union[int, float]]) -> Vector:
    """Compute 3D cross product."""
    return Vector(u).cross(Vector(v))


def matrix_mult(
    A: Sequence[Sequence[Union[int, float]]],
    B: Sequence[Sequence[Union[int, float]]],
) -> Matrix:
    """Compute matrix multiplication A @ B."""
    return Matrix(A) @ Matrix(B)


def transpose(A: Sequence[Sequence[Union[int, float]]]) -> Matrix:
    """Compute matrix transpose A^T."""
    return Matrix(A).T


def determinant(A: Sequence[Sequence[Union[int, float]]]) -> float:
    """Compute matrix determinant."""
    return Matrix(A).det()


def solve_linear(
    A: Sequence[Sequence[Union[int, float]]],
    b: Sequence[Union[int, float]],
) -> Vector:
    """Solve linear system A * x = b."""
    return Matrix(A).solve(Vector(b))


def lu(A: Sequence[Sequence[Union[int, float]]]) -> Tuple[Matrix, Matrix, Matrix]:
    """Compute LU decomposition with partial pivoting: P @ A = L @ U."""
    return Matrix(A).lu()


def qr(A: Sequence[Sequence[Union[int, float]]]) -> Tuple[Matrix, Matrix]:
    """Compute QR decomposition: A = Q @ R."""
    return Matrix(A).qr()


def cholesky(A: Sequence[Sequence[Union[int, float]]]) -> Matrix:
    """Compute Cholesky decomposition of symmetric positive-definite matrix: A = L @ L^T."""
    return Matrix(A).cholesky()


def inv(A: Sequence[Sequence[Union[int, float]]]) -> Matrix:
    """Compute matrix inverse A^(-1)."""
    return Matrix(A).inv()


def inverse(A: Sequence[Sequence[Union[int, float]]]) -> Matrix:
    """Alias for inv(A)."""
    return Matrix(A).inv()


def solve_least_squares(
    A: Sequence[Sequence[Union[int, float]]],
    b: Sequence[Union[int, float]],
) -> Vector:
    """
    Solve linear least squares problem min ||A * x - b||_2 for overdetermined system (m >= n)
    using QR decomposition in native C.
    """
    A_mat = Matrix(A)
    b_vec = Vector(b)
    m, n = A_mat.rows, A_mat.cols

    if m < n:
        raise ValueError(f"Least squares solver requires rows >= cols (m >= n), got shape {A_mat.shape}")
    if b_vec.dim != m:
        raise ValueError(f"Dimension mismatch: matrix has {m} rows, but RHS vector has dimension {b_vec.dim}")

    A_arr = A_mat._to_c_array()
    Q_arr = (ctypes.c_double * (m * n))()
    R_arr = (ctypes.c_double * (n * n))()

    status = _c_matrix_qr(A_arr, m, n, Q_arr, R_arr)
    if status == -1:
        raise ValueError("Matrix has linearly dependent columns; least squares solution is not unique")
    elif status < 0:
        raise ValueError(f"QR decomposition failed with error code {status}")

    x_arr = (ctypes.c_double * n)()
    b_arr = b_vec._to_c_array()
    status = _c_qr_solve(Q_arr, R_arr, b_arr, m, n, x_arr)
    if status < 0:
        raise ValueError(f"Least squares solve failed with error code {status}")

    return Vector(list(x_arr))


__all__ = [
    "Vector",
    "Matrix",
    "dot",
    "norm",
    "cosine_similarity",
    "cross",
    "matrix_mult",
    "transpose",
    "determinant",
    "solve_linear",
    "lu",
    "qr",
    "cholesky",
    "inv",
    "inverse",
    "solve_least_squares",
]
