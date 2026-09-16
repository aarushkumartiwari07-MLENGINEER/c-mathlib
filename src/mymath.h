#ifndef MYMATH_H
#define MYMATH_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cross-platform symbol visibility macro:
 * On Windows, symbols inside a DLL are private by default unless explicitly
 * exported with __declspec(dllexport).
 * On Linux/macOS, functions are visible by default.
 */
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

/**
 * Checks whether an integer n is a prime number.
 *
 * Edge cases handled:
 * - Negative numbers, 0, 1 -> return 0 (not prime)
 * - 2, 3 -> return 1 (prime)
 * - Multiples of 2 or 3 -> return 0
 * - General integers tested up to sqrt(n)
 *
 * @param n Integer to check for primality.
 * @return 1 if n is prime, 0 otherwise.
 */
MYMATH_API int is_prime(int n);

/**
 * Computes the Greatest Common Divisor (GCD) of two integers using the Euclidean algorithm.
 *
 * Properties & Edge cases:
 * - Returns a non-negative value: gcd(a, b) >= 0
 * - gcd(a, 0) = |a|, gcd(0, b) = |b|
 * - gcd(0, 0) = 0
 * - Handles negative integers by taking absolute values.
 *
 * @param a First integer (64-bit signed).
 * @param b Second integer (64-bit signed).
 * @return Non-negative greatest common divisor.
 */
MYMATH_API long long gcd(long long a, long long b);

/**
 * Computes the Least Common Multiple (LCM) of two integers.
 * Reuses gcd() to compute (|a| / gcd(a, b)) * |b| to minimize overflow risk.
 *
 * Properties & Edge cases:
 * - Returns a non-negative value: lcm(a, b) >= 0
 * - lcm(a, 0) = 0, lcm(0, b) = 0
 * - lcm(0, 0) = 0
 * - Handles negative integers by taking absolute values.
 *
 * @param a First integer (64-bit signed).
 * @param b Second integer (64-bit signed).
 * @return Non-negative least common multiple.
 */
MYMATH_API long long lcm(long long a, long long b);

/**
 * Computes the factorial of a non-negative integer n (n!).
 *
 * Domain & Range:
 * - Valid for 0 <= n <= 20 (fits within standard 64-bit signed integer).
 * - 0! = 1, 1! = 1.
 * - Returns -1 if n < 0 (negative input) or n > 20 (overflow).
 *
 * @param n Non-negative integer.
 * @return Factorial of n, or -1 on domain/range error.
 */
MYMATH_API long long factorial(int n);

/**
 * Computes the n-th Fibonacci number (0-indexed: F_0 = 0, F_1 = 1, F_2 = 1, ...).
 *
 * Domain & Range:
 * - Valid for 0 <= n <= 92 (fits within standard 64-bit signed integer).
 * - F_0 = 0, F_1 = 1, F_2 = 1, F_10 = 55.
 * - Returns -1 if n < 0 (negative input) or n > 92 (overflow).
 *
 * @param n Non-negative sequence index.
 * @return n-th Fibonacci number, or -1 on domain/range error.
 */
MYMATH_API long long fibonacci(int n);

/**
 * Computes (base ^ exponent) % modulus using exponentiation by squaring (O(log exponent)).
 *
 * Properties & Constraints:
 * - Requires modulus > 0 and exponent >= 0.
 * - Negative base is normalized to [0, modulus - 1].
 * - Returns -1 if modulus <= 0 or exponent < 0.
 *
 * @param base The base integer.
 * @param exponent Non-negative power to raise base to.
 * @param modulus Positive modulus.
 * @return (base ^ exponent) % modulus, or -1 on invalid domain.
 */
MYMATH_API long long mod_pow(long long base, long long exponent, long long modulus);

/**
 * Extended Euclidean Algorithm: computes gcd(a, b) and Bezout coefficients x and y
 * such that: a*x + b*y = gcd(a, b).
 *
 * @param a First integer.
 * @param b Second integer.
 * @param x Output pointer for Bezout coefficient of a.
 * @param y Output pointer for Bezout coefficient of b.
 * @return The greatest common divisor gcd(|a|, |b|).
 */
MYMATH_API long long extended_gcd(long long a, long long b, long long *x, long long *y);

/**
 * Computes the modular multiplicative inverse of a modulo m, such that:
 * (a * x) = 1 (mod m).
 *
 * Uses the Extended Euclidean Algorithm.
 *
 * Properties & Constraints:
 * - Requires modulus > 1 and gcd(a, modulus) == 1.
 * - Returns the unique inverse in range [0, modulus - 1].
 * - Returns -1 if the inverse does not exist (i.e. gcd(a, modulus) != 1 or modulus <= 1).
 *
 * @param a Integer whose inverse is sought.
 * @param modulus Positive integer modulus (> 1).
 * @return Modular inverse in [0, modulus - 1], or -1 if no inverse exists.
 */
MYMATH_API long long mod_inverse(long long a, long long modulus);

/*
 * Built-in mathematical function IDs for numerical routines.
 */
enum MathFuncId {
    FUNC_IDENTITY = 0,    /* f(x) = x */
    FUNC_SQUARE = 1,      /* f(x) = x^2 */
    FUNC_CUBE = 2,        /* f(x) = x^3 */
    FUNC_QUADRATIC = 3,   /* f(x) = x^2 - 4 */
    FUNC_CUBIC_TEST = 4,  /* f(x) = x^3 - x - 2 */
    FUNC_SIN = 5,         /* f(x) = sin(x) */
    FUNC_COS = 6,         /* f(x) = cos(x) */
    FUNC_EXP = 7,         /* f(x) = exp(x) */
    FUNC_LOG = 8,         /* f(x) = ln(x) */
    FUNC_RECIPROCAL = 9   /* f(x) = 1/x */
};

/**
 * Evaluates a built-in mathematical function at point x.
 */
MYMATH_API double eval_math_func(int func_id, double x);

/**
 * Evaluates a polynomial P(x) = c_0 + c_1*x + ... + c_n*x^n at point x using Horner's method.
 *
 * @param coeffs Array of coefficients in ascending degree order: [c_0, c_1, ..., c_n].
 * @param num_coeffs Number of elements in coeffs (degree + 1).
 * @param x Evaluation point.
 * @return Evaluated value P(x).
 */
MYMATH_API double eval_polynomial(const double *coeffs, int num_coeffs, double x);

/**
 * Finds a root of f(x) = 0 on [a, b] using the bisection method.
 *
 * Status return codes:
 *  0 = success (root written to *root)
 * -1 = invalid bracket (f(a) and f(b) have same sign)
 * -2 = failed to converge within max_iter
 * -3 = invalid parameters (tol <= 0, max_iter <= 0, or unknown func_id)
 *
 * @param func_id Identifier of the built-in function.
 * @param a Left interval bracket.
 * @param b Right interval bracket.
 * @param tol Convergence tolerance.
 * @param max_iter Maximum allowed iterations.
 * @param root Output pointer to store the computed root.
 * @return Status code (0 on success, negative on error).
 */
MYMATH_API int bisection_method(int func_id, double a, double b, double tol, int max_iter, double *root);

/**
 * Finds a root of a polynomial P(x) = 0 on [a, b] using the bisection method.
 */
MYMATH_API int bisection_poly(const double *coeffs, int num_coeffs, double a, double b, double tol, int max_iter, double *root);

/**
 * Approximates definite integral of f(x) from a to b using composite Simpson's 1/3 rule.
 *
 * Status return codes:
 *  0 = success (result written to *result)
 * -1 = invalid subdivision count (n must be even and >= 2)
 * -2 = invalid function ID
 *
 * @param func_id Identifier of the built-in function.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param n Number of intervals (must be positive and even).
 * @param result Output pointer for computed integral approximation.
 * @return Status code (0 on success, negative on error).
 */
MYMATH_API int simpson_rule(int func_id, double a, double b, int n, double *result);

/**
 * Approximates definite integral of a polynomial from a to b using composite Simpson's rule.
 */
MYMATH_API int simpson_poly(const double *coeffs, int num_coeffs, double a, double b, int n, double *result);

/**
 * Computes numerical derivative f'(x) using central difference approximation:
 * f'(x) ~ [f(x + h) - f(x - h)] / (2h).
 *
 * Status return codes:
 *  0 = success (result written to *result)
 * -1 = invalid step size h (must be > 0)
 * -2 = invalid function ID
 *
 * @param func_id Identifier of the built-in function.
 * @param x Point at which to evaluate derivative.
 * @param h Finite difference step size.
 * @param result Output pointer for computed derivative.
 * @return Status code (0 on success, negative on error).
 */
MYMATH_API int numerical_derivative(int func_id, double x, double h, double *result);

/**
 * Computes numerical derivative of a polynomial using central difference.
 */
MYMATH_API int numerical_derivative_poly(const double *coeffs, int num_coeffs, double x, double h, double *result);

#ifdef __cplusplus
}
#endif

#endif /* MYMATH_H */




