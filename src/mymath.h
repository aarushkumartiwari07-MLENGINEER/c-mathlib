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
#ifdef _WIN32
  #ifdef BUILDING_MYMATH_DLL
    #define MYMATH_API __declspec(dllexport)
  #else
    #define MYMATH_API __declspec(dllimport)
  #endif
#else
  #define MYMATH_API
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
 * Extended Euclidean Algorithm: computes gcd(a, b) and Bézout coefficients x and y
 * such that: a*x + b*y = gcd(a, b).
 *
 * @param a First integer.
 * @param b Second integer.
 * @param x Output pointer for Bézout coefficient of a.
 * @param y Output pointer for Bézout coefficient of b.
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

#ifdef __cplusplus
}
#endif

#endif /* MYMATH_H */



