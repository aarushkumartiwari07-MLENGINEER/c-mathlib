#define BUILDING_MYMATH_DLL
#include "statistics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper comparison function for qsort */
static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ========================================================================== */
/* Descriptive Statistics Implementation                                      */
/* ========================================================================== */

double stat_mean(const double *data, int n) {
    if (data == NULL || n <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / (double)n;
}

double stat_variance(const double *data, int n, int ddof) {
    if (data == NULL || n <= ddof || n <= 0) {
        return 0.0;
    }

    /* Two-pass numerically stable variance */
    double m = stat_mean(data, n);
    double sum_sq_diff = 0.0;

    for (int i = 0; i < n; i++) {
        double diff = data[i] - m;
        sum_sq_diff += diff * diff;
    }

    return sum_sq_diff / (double)(n - ddof);
}

double stat_std_dev(const double *data, int n, int ddof) {
    double var = stat_variance(data, n, ddof);
    if (var <= 0.0) {
        return 0.0;
    }
    return sqrt(var);
}

double stat_median(const double *data, int n) {
    if (data == NULL || n <= 0) {
        return 0.0;
    }

    double *copy = (double *)malloc(sizeof(double) * n);
    if (copy == NULL) {
        return 0.0;
    }
    memcpy(copy, data, sizeof(double) * n);

    qsort(copy, n, sizeof(double), compare_doubles);

    double result;
    if (n % 2 == 1) {
        result = copy[n / 2];
    } else {
        result = (copy[(n / 2) - 1] + copy[n / 2]) / 2.0;
    }

    free(copy);
    return result;
}

double stat_quantile(const double *data, int n, double q) {
    if (data == NULL || n <= 0) {
        return 0.0;
    }
    if (q < 0.0) q = 0.0;
    if (q > 1.0) q = 1.0;

    if (n == 1) {
        return data[0];
    }

    double *copy = (double *)malloc(sizeof(double) * n);
    if (copy == NULL) {
        return 0.0;
    }
    memcpy(copy, data, sizeof(double) * n);

    qsort(copy, n, sizeof(double), compare_doubles);

    /* Linear interpolation index */
    double idx = q * (double)(n - 1);
    int lower = (int)floor(idx);
    int upper = (int)ceil(idx);
    double frac = idx - (double)lower;

    double result = copy[lower] + frac * (copy[upper] - copy[lower]);

    free(copy);
    return result;
}

/* ========================================================================== */
/* Bivariate Statistics Implementation                                        */
/* ========================================================================== */

double stat_covariance(const double *x, const double *y, int n, int ddof) {
    if (x == NULL || y == NULL || n <= ddof || n <= 0) {
        return 0.0;
    }

    double mean_x = stat_mean(x, n);
    double mean_y = stat_mean(y, n);

    double sum_cross = 0.0;
    for (int i = 0; i < n; i++) {
        sum_cross += (x[i] - mean_x) * (y[i] - mean_y);
    }

    return sum_cross / (double)(n - ddof);
}

double stat_correlation(const double *x, const double *y, int n) {
    if (x == NULL || y == NULL || n <= 1) {
        return 0.0;
    }

    double std_x = stat_std_dev(x, n, 1);
    double std_y = stat_std_dev(y, n, 1);

    if (std_x == 0.0 || std_y == 0.0) {
        return 0.0;
    }

    double cov = stat_covariance(x, y, n, 1);
    double r = cov / (std_x * std_y);

    /* Clamp numerical precision to [-1.0, 1.0] */
    if (r > 1.0) r = 1.0;
    if (r < -1.0) r = -1.0;

    return r;
}

int stat_linear_regression(
    const double *x,
    const double *y,
    int n,
    double *slope,
    double *intercept,
    double *r_squared
) {
    if (x == NULL || y == NULL || slope == NULL || intercept == NULL || n < 2) {
        return -1;
    }

    double mean_x = stat_mean(x, n);
    double mean_y = stat_mean(y, n);

    double ss_xx = 0.0;
    double ss_xy = 0.0;
    double ss_yy = 0.0;

    for (int i = 0; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
        ss_yy += dy * dy;
    }

    if (ss_xx < 1e-15) {
        /* Variance in x is zero; cannot compute slope */
        return -1;
    }

    double m = ss_xy / ss_xx;
    double c = mean_y - m * mean_x;

    *slope = m;
    *intercept = c;

    if (r_squared != NULL) {
        if (ss_yy < 1e-15) {
            *r_squared = 1.0;
        } else {
            double r = ss_xy / sqrt(ss_xx * ss_yy);
            if (r > 1.0) r = 1.0;
            if (r < -1.0) r = -1.0;
            *r_squared = r * r;
        }
    }

    return 0;
}
