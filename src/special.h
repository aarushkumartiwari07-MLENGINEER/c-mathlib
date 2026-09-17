#ifndef SPECIAL_H
#define SPECIAL_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifndef MYMATH_API
    #if defined(BUILDING_MYMATH_DLL) || defined(BUILDING_MYMATH)
      #define MYMATH_API __declspec(dllexport)
    #else
      #define MYMATH_API __declspec(dllimport)
    #endif
  #endif
#else
  #ifndef MYMATH_API
    #define MYMATH_API __attribute__((visibility("default")))
  #endif
#endif

/**
 * Computes the natural logarithm of the absolute value of the Gamma function, ln|Gamma(x)|.
 * Uses the Lanczos approximation (g=7).
 *
 * @param x Input real number. Must not be a non-positive integer.
 * @param result Output pointer for ln|Gamma(x)|.
 * @return 0 on success, -1 on invalid pointer/domain error (x <= 0 and x is an integer).
 */
MYMATH_API int math_lgamma(double x, double *result);

/**
 * Computes the Gamma function Gamma(x) via Lanczos approximation and reflection formula.
 *
 * @param x Input real number.
 * @param result Output pointer for Gamma(x).
 * @return 0 on success, -1 on invalid pointer or non-positive integer singularity.
 */
MYMATH_API int math_gamma(double x, double *result);

/**
 * Computes the Beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).
 *
 * @param a First parameter (must not be <= 0 integer).
 * @param b Second parameter (must not be <= 0 integer).
 * @param result Output pointer for B(a, b).
 * @return 0 on success, -1 on domain error or invalid pointer.
 */
MYMATH_API int math_beta(double a, double b, double *result);

/**
 * Computes the Gauss Error Function erf(x) = (2 / sqrt(pi)) * int_0^x exp(-t^2) dt.
 *
 * @param x Input value.
 * @param result Output pointer for erf(x).
 * @return 0 on success, -1 on invalid pointer.
 */
MYMATH_API int math_erf(double x, double *result);

/**
 * Computes the Complementary Error Function erfc(x) = 1 - erf(x).
 *
 * @param x Input value.
 * @param result Output pointer for erfc(x).
 * @return 0 on success, -1 on invalid pointer.
 */
MYMATH_API int math_erfc(double x, double *result);

/**
 * Computes the logistic sigmoid activation function: sigma(x) = 1 / (1 + exp(-x)).
 * Numerically stable for both large positive and negative inputs.
 *
 * @param x Input scalar.
 * @return Computed sigmoid value in (0, 1).
 */
MYMATH_API double math_sigmoid(double x);

/**
 * Computes the numerically stable softmax probability distribution over an array x:
 * out[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x))).
 *
 * @param x Input array of logits.
 * @param n Length of array (must be >= 1).
 * @param out Output array of probabilities (size n, sums to 1.0).
 * @return 0 on success, -1 on invalid inputs or n < 1.
 */
MYMATH_API int math_softmax(const double *x, int n, double *out);

#ifdef __cplusplus
}
#endif

#endif /* SPECIAL_H */
