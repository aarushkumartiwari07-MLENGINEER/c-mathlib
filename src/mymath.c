#define BUILDING_MYMATH_DLL
#include "mymath.h"

int is_prime(int n) {
    /* Numbers <= 1 (including 0 and all negative integers) are not prime */
    if (n <= 1) {
        return 0;
    }

    /* 2 and 3 are prime */
    if (n <= 3) {
        return 1;
    }

    /* Multiples of 2 and 3 (except 2 and 3 themselves) are composite */
    if (n % 2 == 0 || n % 3 == 0) {
        return 0;
    }

    /*
     * Check divisors of the form (6k ± 1) up to sqrt(n).
     * Any integer can be expressed as (6k + i) for i in {0, 1, 2, 3, 4, 5}.
     * Since 6k, 6k+2, 6k+3, 6k+4 are divisible by 2 or 3, we only need to test
     * 6k-1 and 6k+1 (starting at i = 5).
     * Using i * i <= n avoids floating-point operations.
     */
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return 0;
        }
    }

    return 1;
}
