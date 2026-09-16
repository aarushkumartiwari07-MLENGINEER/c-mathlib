#define BUILDING_MYMATH_DLL
#include "linear_algebra.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================== */
/* Vector Operations Implementation                                           */
/* ========================================================================== */

double vector_dot(const double *u, const double *v, int n) {
    if (u == NULL || v == NULL || n <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += u[i] * v[i];
    }
    return sum;
}

double vector_norm(const double *v, int n, int p) {
    if (v == NULL || n <= 0) {
        return 0.0;
    }

    if (p == 1) {
        /* L1 Norm: Manhattan distance */
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += fabs(v[i]);
        }
        return sum;
    } else if (p == 0) {
        /* Linf Norm: Maximum absolute element */
        double max_val = 0.0;
        for (int i = 0; i < n; i++) {
            double abs_val = fabs(v[i]);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
        return max_val;
    } else {
        /* Default L2 Norm: Euclidean distance */
        double sum_sq = 0.0;
        for (int i = 0; i < n; i++) {
            sum_sq += v[i] * v[i];
        }
        return sqrt(sum_sq);
    }
}

int vector_add(const double *u, const double *v, int n, double *out) {
    if (u == NULL || v == NULL || out == NULL || n <= 0) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        out[i] = u[i] + v[i];
    }
    return 0;
}

int vector_sub(const double *u, const double *v, int n, double *out) {
    if (u == NULL || v == NULL || out == NULL || n <= 0) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        out[i] = u[i] - v[i];
    }
    return 0;
}

int vector_scale(const double *v, int n, double scalar, double *out) {
    if (v == NULL || out == NULL || n <= 0) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        out[i] = v[i] * scalar;
    }
    return 0;
}

int vector_cross_3d(const double *u, const double *v, double *out) {
    if (u == NULL || v == NULL || out == NULL) {
        return -1;
    }
    out[0] = u[1] * v[2] - u[2] * v[1];
    out[1] = u[2] * v[0] - u[0] * v[2];
    out[2] = u[0] * v[1] - u[1] * v[0];
    return 0;
}

double vector_cosine_similarity(const double *u, const double *v, int n) {
    if (u == NULL || v == NULL || n <= 0) {
        return 0.0;
    }
    double dot = vector_dot(u, v, n);
    double norm_u = vector_norm(u, n, 2);
    double norm_v = vector_norm(v, n, 2);

    if (norm_u == 0.0 || norm_v == 0.0) {
        return 0.0;
    }
    return dot / (norm_u * norm_v);
}

/* ========================================================================== */
/* Matrix Operations Implementation                                           */
/* ========================================================================== */

int matrix_multiply(
    const double *A,
    const double *B,
    int rows_A,
    int cols_A,
    int cols_B,
    double *C
) {
    if (A == NULL || B == NULL || C == NULL) {
        return -1;
    }
    if (rows_A <= 0 || cols_A <= 0 || cols_B <= 0) {
        return -1;
    }

    /* Initialize output to 0 */
    memset(C, 0, sizeof(double) * rows_A * cols_B);

    /* Cache-friendly loop order (i, k, j) */
    for (int i = 0; i < rows_A; i++) {
        for (int k = 0; k < cols_A; k++) {
            double a_ik = A[i * cols_A + k];
            for (int j = 0; j < cols_B; j++) {
                C[i * cols_B + j] += a_ik * B[k * cols_B + j];
            }
        }
    }
    return 0;
}

int matrix_transpose(const double *A, int rows, int cols, double *out) {
    if (A == NULL || out == NULL || rows <= 0 || cols <= 0) {
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = A[i * cols + j];
        }
    }
    return 0;
}

double matrix_trace(const double *A, int n) {
    if (A == NULL || n <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += A[i * n + i];
    }
    return sum;
}

int matrix_determinant(const double *A, int n, double *det) {
    if (A == NULL || det == NULL || n <= 0) {
        return -1;
    }

    if (n == 1) {
        *det = A[0];
        return 0;
    }

    if (n == 2) {
        *det = A[0] * A[3] - A[1] * A[2];
        return 0;
    }

    /* Allocate working copy of matrix */
    double *M = (double *)malloc(sizeof(double) * n * n);
    if (M == NULL) {
        return -1;
    }
    memcpy(M, A, sizeof(double) * n * n);

    double d = 1.0;
    int sign = 1;

    for (int i = 0; i < n; i++) {
        /* Partial pivoting: find maximum element in column i from row i down */
        int pivot_row = i;
        double max_val = fabs(M[i * n + i]);
        for (int k = i + 1; k < n; k++) {
            double val = fabs(M[k * n + i]);
            if (val > max_val) {
                max_val = val;
                pivot_row = k;
            }
        }

        if (max_val < 1e-15) {
            /* Singular matrix: determinant is 0 */
            free(M);
            *det = 0.0;
            return 0;
        }

        if (pivot_row != i) {
            /* Swap rows i and pivot_row */
            for (int j = 0; j < n; j++) {
                double temp = M[i * n + j];
                M[i * n + j] = M[pivot_row * n + j];
                M[pivot_row * n + j] = temp;
            }
            sign = -sign;
        }

        double pivot = M[i * n + i];
        d *= pivot;

        /* Eliminate rows below pivot */
        for (int k = i + 1; k < n; k++) {
            double factor = M[k * n + i] / pivot;
            for (int j = i; j < n; j++) {
                M[k * n + j] -= factor * M[i * n + j];
            }
        }
    }

    free(M);
    *det = d * sign;
    return 0;
}

int solve_linear_system(const double *A, const double *b, int n, double *x) {
    if (A == NULL || b == NULL || x == NULL || n <= 0) {
        return -2;
    }

    /* Build augmented matrix [A | b] with dimension n x (n + 1) */
    int cols = n + 1;
    double *M = (double *)malloc(sizeof(double) * n * cols);
    if (M == NULL) {
        return -2;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i * cols + j] = A[i * n + j];
        }
        M[i * cols + n] = b[i];
    }

    /* Forward elimination with partial pivoting */
    for (int i = 0; i < n; i++) {
        int pivot_row = i;
        double max_val = fabs(M[i * cols + i]);
        for (int k = i + 1; k < n; k++) {
            double val = fabs(M[k * cols + i]);
            if (val > max_val) {
                max_val = val;
                pivot_row = k;
            }
        }

        if (max_val < 1e-14) {
            free(M);
            return -1; /* Singular or nearly singular system */
        }

        if (pivot_row != i) {
            for (int j = 0; j < cols; j++) {
                double temp = M[i * cols + j];
                M[i * cols + j] = M[pivot_row * cols + j];
                M[pivot_row * cols + j] = temp;
            }
        }

        double pivot = M[i * cols + i];
        for (int k = i + 1; k < n; k++) {
            double factor = M[k * cols + i] / pivot;
            for (int j = i; j < cols; j++) {
                M[k * cols + j] -= factor * M[i * cols + j];
            }
        }
    }

    /* Back-substitution */
    for (int i = n - 1; i >= 0; i--) {
        double sum = M[i * cols + n];
        for (int j = i + 1; j < n; j++) {
            sum -= M[i * cols + j] * x[j];
        }
        x[i] = sum / M[i * cols + i];
    }

    free(M);
    return 0;
}

/* ========================================================================== */
/* Matrix Decompositions Implementation                                       */
/* ========================================================================== */

int matrix_lu(
    const double *A,
    int n,
    double *L,
    double *U,
    int *P
) {
    if (A == NULL || L == NULL || U == NULL || P == NULL || n <= 0) {
        return -2;
    }

    double *M = (double *)malloc(sizeof(double) * n * n);
    if (M == NULL) {
        return -2;
    }
    memcpy(M, A, sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    for (int k = 0; k < n; k++) {
        /* Partial pivoting: find maximum element in column k from row k down */
        int pivot_row = k;
        double max_val = fabs(M[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double val = fabs(M[i * n + k]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }

        if (max_val < 1e-15) {
            free(M);
            return -1; /* Singular matrix */
        }

        if (pivot_row != k) {
            /* Swap rows in M */
            for (int j = 0; j < n; j++) {
                double temp = M[k * n + j];
                M[k * n + j] = M[pivot_row * n + j];
                M[pivot_row * n + j] = temp;
            }
            /* Swap permutation entries */
            int p_temp = P[k];
            P[k] = P[pivot_row];
            P[pivot_row] = p_temp;
        }

        double pivot = M[k * n + k];
        for (int i = k + 1; i < n; i++) {
            M[i * n + k] /= pivot;
            double factor = M[i * n + k];
            for (int j = k + 1; j < n; j++) {
                M[i * n + j] -= factor * M[k * n + j];
            }
        }
    }

    /* Extract L and U matrices */
    memset(L, 0, sizeof(double) * n * n);
    memset(U, 0, sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        L[i * n + i] = 1.0;
        for (int j = 0; j < n; j++) {
            if (i > j) {
                L[i * n + j] = M[i * n + j];
            } else {
                U[i * n + j] = M[i * n + j];
            }
        }
    }

    free(M);
    return 0;
}

int lu_solve(
    const double *L,
    const double *U,
    const int *P,
    const double *b,
    int n,
    double *x
) {
    if (L == NULL || U == NULL || P == NULL || b == NULL || x == NULL || n <= 0) {
        return -2;
    }

    double *y = (double *)malloc(sizeof(double) * n);
    if (y == NULL) {
        return -2;
    }

    /* Apply permutation P to RHS: y_i = b_{P_i} */
    for (int i = 0; i < n; i++) {
        y[i] = b[P[i]];
    }

    /* Forward substitution: L * y = P * b */
    for (int i = 0; i < n; i++) {
        double sum = y[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum; /* L_ii is 1.0 */
    }

    /* Back-substitution: U * x = y */
    for (int i = n - 1; i >= 0; i--) {
        double diag = U[i * n + i];
        if (fabs(diag) < 1e-15) {
            free(y);
            return -1; /* Singular U matrix */
        }
        double sum = y[i];
        for (int j = i + 1; j < n; j++) {
            sum -= U[i * n + j] * x[j];
        }
        x[i] = sum / diag;
    }

    free(y);
    return 0;
}

int matrix_qr(
    const double *A,
    int m,
    int n,
    double *Q,
    double *R
) {
    if (A == NULL || Q == NULL || R == NULL || m <= 0 || n <= 0 || m < n) {
        return -2;
    }

    /* Modified Gram-Schmidt Orthogonalization */
    double *V = (double *)malloc(sizeof(double) * m * n);
    if (V == NULL) {
        return -2;
    }
    memcpy(V, A, sizeof(double) * m * n);

    memset(Q, 0, sizeof(double) * m * n);
    memset(R, 0, sizeof(double) * n * n);

    for (int k = 0; k < n; k++) {
        /* Compute norm of column k */
        double norm_sq = 0.0;
        for (int i = 0; i < m; i++) {
            double val = V[i * n + k];
            norm_sq += val * val;
        }
        double norm_k = sqrt(norm_sq);

        if (norm_k < 1e-15) {
            free(V);
            return -1; /* Linearly dependent columns */
        }

        R[k * n + k] = norm_k;

        /* Normalize column k to get Q_k */
        for (int i = 0; i < m; i++) {
            Q[i * n + k] = V[i * n + k] / norm_k;
        }

        /* Orthogonalize subsequent columns against Q_k */
        for (int j = k + 1; j < n; j++) {
            double dot = 0.0;
            for (int i = 0; i < m; i++) {
                dot += Q[i * n + k] * V[i * n + j];
            }
            R[k * n + j] = dot;
            for (int i = 0; i < m; i++) {
                V[i * n + j] -= dot * Q[i * n + k];
            }
        }
    }

    free(V);
    return 0;
}

int qr_solve(
    const double *Q,
    const double *R,
    const double *b,
    int m,
    int n,
    double *x
) {
    if (Q == NULL || R == NULL || b == NULL || x == NULL || m <= 0 || n <= 0 || m < n) {
        return -2;
    }

    /* Compute d = Q^T * b (dimension n) */
    double *d = (double *)malloc(sizeof(double) * n);
    if (d == NULL) {
        return -2;
    }

    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += Q[i * n + j] * b[i];
        }
        d[j] = sum;
    }

    /* Back-substitution on R * x = d */
    for (int i = n - 1; i >= 0; i--) {
        double diag = R[i * n + i];
        if (fabs(diag) < 1e-15) {
            free(d);
            return -1;
        }
        double sum = d[i];
        for (int j = i + 1; j < n; j++) {
            sum -= R[i * n + j] * x[j];
        }
        x[i] = sum / diag;
    }

    free(d);
    return 0;
}

int matrix_cholesky(
    const double *A,
    int n,
    double *L
) {
    if (A == NULL || L == NULL || n <= 0) {
        return -2;
    }

    memset(L, 0, sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }

            if (i == j) {
                double val = A[i * n + i] - sum;
                if (val <= 1e-15) {
                    return -1; /* Matrix is not positive-definite */
                }
                L[i * n + i] = sqrt(val);
            } else {
                double diag = L[j * n + j];
                if (diag < 1e-15) {
                    return -1;
                }
                L[i * n + j] = (A[i * n + j] - sum) / diag;
            }
        }
    }

    return 0;
}

int cholesky_solve(
    const double *L,
    const double *b,
    int n,
    double *x
) {
    if (L == NULL || b == NULL || x == NULL || n <= 0) {
        return -2;
    }

    double *y = (double *)malloc(sizeof(double) * n);
    if (y == NULL) {
        return -2;
    }

    /* Forward substitution: L * y = b */
    for (int i = 0; i < n; i++) {
        double diag = L[i * n + i];
        if (diag < 1e-15) {
            free(y);
            return -1;
        }
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / diag;
    }

    /* Back-substitution: L^T * x = y */
    for (int i = n - 1; i >= 0; i--) {
        double diag = L[i * n + i];
        double sum = y[i];
        for (int j = i + 1; j < n; j++) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / diag;
    }

    free(y);
    return 0;
}

int matrix_inverse(
    const double *A,
    int n,
    double *inv
) {
    if (A == NULL || inv == NULL || n <= 0) {
        return -2;
    }

    if (n == 1) {
        if (fabs(A[0]) < 1e-15) {
            return -1;
        }
        inv[0] = 1.0 / A[0];
        return 0;
    }

    if (n == 2) {
        double det = A[0] * A[3] - A[1] * A[2];
        if (fabs(det) < 1e-15) {
            return -1;
        }
        double inv_det = 1.0 / det;
        inv[0] = A[3] * inv_det;
        inv[1] = -A[1] * inv_det;
        inv[2] = -A[2] * inv_det;
        inv[3] = A[0] * inv_det;
        return 0;
    }

    double *L = (double *)malloc(sizeof(double) * n * n);
    double *U = (double *)malloc(sizeof(double) * n * n);
    int *P = (int *)malloc(sizeof(int) * n);
    double *e = (double *)malloc(sizeof(double) * n);
    double *x_col = (double *)malloc(sizeof(double) * n);

    if (L == NULL || U == NULL || P == NULL || e == NULL || x_col == NULL) {
        free(L); free(U); free(P); free(e); free(x_col);
        return -2;
    }

    int status = matrix_lu(A, n, L, U, P);
    if (status != 0) {
        free(L); free(U); free(P); free(e); free(x_col);
        return -1;
    }

    /* Solve A * x_j = e_j for each column of identity matrix */
    for (int j = 0; j < n; j++) {
        memset(e, 0, sizeof(double) * n);
        e[j] = 1.0;

        status = lu_solve(L, U, P, e, n, x_col);
        if (status != 0) {
            free(L); free(U); free(P); free(e); free(x_col);
            return -1;
        }

        for (int i = 0; i < n; i++) {
            inv[i * n + j] = x_col[i];
        }
    }

    free(L); free(U); free(P); free(e); free(x_col);
    return 0;
}
