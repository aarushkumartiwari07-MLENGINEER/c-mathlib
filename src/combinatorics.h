#ifndef COMBINATORICS_H
#define COMBINATORICS_H

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

/**
 * Computes combinations n choose k (binomial coefficient nCk):
 * C(n, k) = n! / (k! * (n - k)!).
 *
 * Uses an incremental multiplicative loop to avoid premature integer overflow.
 *
 * @param n Total number of items (n >= 0).
 * @param k Number of items to choose (0 <= k <= n).
 * @return Binomial coefficient C(n, k), or -1 on invalid input.
 */
MYMATH_API long long math_combinations(int n, int k);

/**
 * Computes permutations P(n, k) (k-permutations of n):
 * P(n, k) = n! / (n - k)! = n * (n - 1) * ... * (n - k + 1).
 *
 * @param n Total number of items (n >= 0).
 * @param k Number of items to arrange (0 <= k <= n).
 * @return Permutation count P(n, k), or -1 on invalid input.
 */
MYMATH_API long long math_permutations(int n, int k);

/**
 * Computes Euler's totient function phi(n), which counts the positive integers
 * up to n that are relatively prime (coprime) to n:
 * phi(n) = n * Product_{p | n} (1 - 1/p).
 *
 * @param n Positive integer.
 * @return phi(n), or -1 if n <= 0.
 */
MYMATH_API long long euler_phi(long long n);

#ifdef __cplusplus
}
#endif

#endif /* COMBINATORICS_H */
