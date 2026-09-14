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

#ifdef __cplusplus
}
#endif

#endif /* MYMATH_H */
