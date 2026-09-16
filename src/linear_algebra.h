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

#ifdef __cplusplus
}
#endif

#endif /* LINEAR_ALGEBRA_H */
