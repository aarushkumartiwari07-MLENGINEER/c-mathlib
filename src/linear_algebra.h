#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#ifndef MYMATH_API
  #ifdef _WIN32
    #ifdef BUILDING_MYMATH_DLL
      #define MYMATH_API __declspec(dllexport)
    #else
      #define MYMATH_API __declspec(dllimport)
    #endif
  #else
    #define MYMATH_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Vector Operations                                                          */
/* ========================================================================== */

/**
 * Computes the dot product (scalar product) of two vectors u and v.
 *
 * @param u Pointer to first vector elements.
 * @param v Pointer to second vector elements.
 * @param n Dimension of the vectors.
 * @return Scalar dot product sum(u_i * v_i).
 */
MYMATH_API double vector_dot(const double *u, const double *v, int n);

/**
 * Computes the p-norm of a vector v.
 * Supported p values:
 * - p = 1: L1 norm (Manhattan norm, sum |v_i|)
 * - p = 2: L2 norm (Euclidean norm, sqrt(sum v_i^2))
 * - p = 0: Linf norm (Chebyshev norm, max |v_i|)
 *
 * @param v Pointer to vector elements.
 * @param n Dimension of the vector.
 * @param p Norm type (1, 2, or 0 for Linf).
 * @return Computed vector norm.
 */
MYMATH_API double vector_norm(const double *v, int n, int p);

/**
 * Adds two vectors: out = u + v.
 */
MYMATH_API int vector_add(const double *u, const double *v, int n, double *out);

/**
 * Subtracts two vectors: out = u - v.
 */
MYMATH_API int vector_sub(const double *u, const double *v, int n, double *out);

/**
 * Multiplies a vector by a scalar: out = scalar * v.
 */
MYMATH_API int vector_scale(const double *v, int n, double scalar, double *out);

/**
 * Computes the 3D cross product: out = u x v.
 */
MYMATH_API int vector_cross_3d(const double *u, const double *v, double *out);

/**
 * Computes the cosine similarity between two vectors: (u . v) / (||u||_2 * ||v||_2).
 * Returns 0.0 if either vector has norm 0.
 */
MYMATH_API double vector_cosine_similarity(const double *u, const double *v, int n);

/* ========================================================================== */
/* Matrix Operations (Row-Major Storage)                                      */
/* ========================================================================== */

/**
 * Multiplies two matrices: C = A (rows_A x cols_A) * B (cols_A x cols_B).
 * C has dimensions (rows_A x cols_B).
 *
 * @return 0 on success, -1 on invalid dimensions/pointers.
 */
MYMATH_API int matrix_multiply(
    const double *A,
    const double *B,
    int rows_A,
    int cols_A,
    int cols_B,
    double *C
);

/**
 * Computes the transpose of a matrix: out = A^T.
 * A is (rows x cols), out is (cols x rows).
 */
MYMATH_API int matrix_transpose(const double *A, int rows, int cols, double *out);

/**
 * Computes the trace of a square matrix (sum of diagonal elements).
 */
MYMATH_API double matrix_trace(const double *A, int n);

/**
 * Computes the determinant of an n x n square matrix using Gaussian elimination
 * with partial pivoting.
 *
 * @param A Pointer to square matrix (n x n, row-major).
 * @param n Dimension of the matrix.
 * @param det Output pointer for determinant.
 * @return 0 on success, -1 on invalid parameter.
 */
MYMATH_API int matrix_determinant(const double *A, int n, double *det);

/**
 * Solves a system of linear equations A * x = b for an n x n matrix A using
 * Gaussian elimination with partial pivoting and back-substitution.
 *
 * @param A Pointer to coefficient matrix (n x n, row-major).
 * @param b Pointer to right-hand side vector (dimension n).
 * @param n Dimension of the linear system.
 * @param x Output pointer for solution vector (dimension n).
 * @return 0 on success, -1 if matrix is singular / non-invertible, -2 on invalid input.
 */
MYMATH_API int solve_linear_system(const double *A, const double *b, int n, double *x);

/* ========================================================================== */
/* Matrix Decompositions & Advanced Solvers                                   */
/* ========================================================================== */

/**
 * Computes the LU decomposition with partial pivoting: P * A = L * U.
 *
 * @param A Pointer to square matrix (n x n, row-major).
 * @param n Dimension of matrix.
 * @param L Output pointer for unit lower triangular matrix (n x n).
 * @param U Output pointer for upper triangular matrix (n x n).
 * @param P Output pointer for pivot permutation vector (dimension n).
 * @return 0 on success, -1 if matrix is singular, -2 on invalid input.
 */
MYMATH_API int matrix_lu(
    const double *A,
    int n,
    double *L,
    double *U,
    int *P
);

/**
 * Solves A * x = b given precomputed LU factorization (P * A = L * U).
 * Uses forward substitution on L * y = P * b, then back-substitution on U * x = y.
 *
 * @param L Pointer to unit lower triangular matrix (n x n).
 * @param U Pointer to upper triangular matrix (n x n).
 * @param P Pointer to pivot permutation vector (dimension n).
 * @param b Pointer to right-hand side vector (dimension n).
 * @param n Dimension of linear system.
 * @param x Output pointer for solution vector (dimension n).
 * @return 0 on success, -1 on singular U (zero diagonal), -2 on invalid input.
 */
MYMATH_API int lu_solve(
    const double *L,
    const double *U,
    const int *P,
    const double *b,
    int n,
    double *x
);

/**
 * Computes QR decomposition of an m x n matrix (m >= n) using Modified Gram-Schmidt:
 * A = Q * R, where Q is m x n (orthogonal columns) and R is n x n (upper triangular).
 *
 * @param A Pointer to m x n matrix.
 * @param m Number of rows (m >= n).
 * @param n Number of columns.
 * @param Q Output pointer for m x n matrix with orthogonal columns.
 * @param R Output pointer for n x n upper triangular matrix.
 * @return 0 on success, -1 if columns are linearly dependent, -2 on invalid input.
 */
MYMATH_API int matrix_qr(
    const double *A,
    int m,
    int n,
    double *Q,
    double *R
);

/**
 * Solves the linear least squares problem min ||A * x - b||_2 given QR decomposition A = Q * R.
 * Computes x = R^(-1) * (Q^T * b) via back-substitution.
 *
 * @param Q Pointer to m x n orthogonal matrix.
 * @param R Pointer to n x n upper triangular matrix.
 * @param b Pointer to right-hand side vector (dimension m).
 * @param m Number of rows.
 * @param n Number of columns (m >= n).
 * @param x Output pointer for solution vector (dimension n).
 * @return 0 on success, -1 if R is singular, -2 on invalid input.
 */
MYMATH_API int qr_solve(
    const double *Q,
    const double *R,
    const double *b,
    int m,
    int n,
    double *x
);

/**
 * Computes Cholesky decomposition of a symmetric positive-definite (SPD) matrix:
 * A = L * L^T, where L is lower triangular with positive diagonal elements.
 *
 * @param A Pointer to n x n symmetric matrix.
 * @param n Dimension of matrix.
 * @param L Output pointer for n x n lower triangular matrix.
 * @return 0 on success, -1 if matrix is not positive-definite, -2 on invalid input.
 */
MYMATH_API int matrix_cholesky(
    const double *A,
    int n,
    double *L
);

/**
 * Solves A * x = b for a symmetric positive-definite matrix given its Cholesky factor L (A = L * L^T).
 *
 * @param L Pointer to lower triangular matrix (n x n).
 * @param b Pointer to right-hand side vector (dimension n).
 * @param n Dimension of linear system.
 * @param x Output pointer for solution vector (dimension n).
 * @return 0 on success, -1 if L is singular, -2 on invalid input.
 */
MYMATH_API int cholesky_solve(
    const double *L,
    const double *b,
    int n,
    double *x
);

/**
 * Computes the inverse of an n x n square matrix A using LU decomposition.
 *
 * @param A Pointer to n x n square matrix.
 * @param n Dimension of matrix.
 * @param inv Output pointer for n x n inverse matrix.
 * @return 0 on success, -1 if matrix is singular, -2 on invalid input.
 */
MYMATH_API int matrix_inverse(
    const double *A,
    int n,
    double *inv
);

#ifdef __cplusplus
}
#endif

#endif /* LINEAR_ALGEBRA_H */
