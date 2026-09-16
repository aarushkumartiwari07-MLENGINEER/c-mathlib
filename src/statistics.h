#ifndef STATISTICS_H
#define STATISTICS_H

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
/* Descriptive Statistics                                                     */
/* ========================================================================== */

/**
 * Computes the arithmetic mean of a dataset: sum(x_i) / n.
 */
MYMATH_API double stat_mean(const double *data, int n);

/**
 * Computes sample or population variance using a numerically stable two-pass method.
 *
 * @param data Array of numeric values.
 * @param n Sample size.
 * @param ddof Delta degrees of freedom (0 for population variance, 1 for sample variance).
 * @return Variance, or 0.0 if n <= ddof.
 */
MYMATH_API double stat_variance(const double *data, int n, int ddof);

/**
 * Computes standard deviation: sqrt(variance(data, n, ddof)).
 */
MYMATH_API double stat_std_dev(const double *data, int n, int ddof);

/**
 * Computes the median using the Quickselect algorithm in O(n) average time.
 * Note: A temporary copy of data is sorted/partitioned to preserve original data.
 */
MYMATH_API double stat_median(const double *data, int n);

/**
 * Computes the q-th quantile (0 <= q <= 1) using linear interpolation between order statistics.
 */
MYMATH_API double stat_quantile(const double *data, int n, double q);

/* ========================================================================== */
/* Bivariate Statistics & Linear Regression                                   */
/* ========================================================================== */

/**
 * Computes the covariance between two datasets x and y:
 * cov(x, y) = sum((x_i - mean_x) * (y_i - mean_y)) / (n - ddof).
 */
MYMATH_API double stat_covariance(const double *x, const double *y, int n, int ddof);

/**
 * Computes the Pearson product-moment correlation coefficient r in [-1, 1].
 */
MYMATH_API double stat_correlation(const double *x, const double *y, int n);

/**
 * Fits a simple Ordinary Least Squares (OLS) linear regression model y = slope * x + intercept.
 *
 * @param x Independent variable array.
 * @param y Dependent variable array.
 * @param n Number of data points.
 * @param slope Output pointer for regression slope (m).
 * @param intercept Output pointer for regression y-intercept (c).
 * @param r_squared Output pointer for coefficient of determination (R^2).
 * @return 0 on success, -1 on zero variance in x or invalid parameters.
 */
MYMATH_API int stat_linear_regression(
    const double *x,
    const double *y,
    int n,
    double *slope,
    double *intercept,
    double *r_squared
);

#ifdef __cplusplus
}
#endif

#endif /* STATISTICS_H */
