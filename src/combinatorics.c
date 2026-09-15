#define BUILDING_MYMATH_DLL
#include "combinatorics.h"

long long math_combinations(int n, int k) {
    if (n < 0 || k < 0 || k > n) {
        return -1;
    }
    if (k == 0 || k == n) {
        return 1;
    }

    /* Symmetry property: C(n, k) = C(n, n - k) */
    if (k > n - k) {
        k = n - k;
    }

    long long result = 1;
    for (int i = 1; i <= k; i++) {
        result = result * (n - k + i) / i;
    }
    return result;
}

long long math_permutations(int n, int k) {
    if (n < 0 || k < 0 || k > n) {
        return -1;
    }
    if (k == 0) {
        return 1;
    }

    long long result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
    }
    return result;
}

long long euler_phi(long long n) {
    if (n <= 0) {
        return -1;
    }
    if (n == 1) {
        return 1;
    }

    long long result = n;

    /* Check divisibility by 2 */
    if (n % 2 == 0) {
        while (n % 2 == 0) {
            n /= 2;
        }
        result -= result / 2;
    }

    /* Check odd factors up to sqrt(n) */
    for (long long p = 3; p * p <= n; p += 2) {
        if (n % p == 0) {
            while (n % p == 0) {
                n /= p;
            }
            result -= result / p;
        }
    }

    /* If remaining n is a prime greater than 2 */
    if (n > 1) {
        result -= result / n;
    }

    return result;
}
