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
