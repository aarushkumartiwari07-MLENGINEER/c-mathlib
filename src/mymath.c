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
     * Using (long long)i * i <= n avoids 32-bit signed overflow when n is close to INT_MAX.
     */
    for (int i = 5; (long long)i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return 0;
        }
    }

    return 1;
}

long long gcd(long long a, long long b) {
    /* Normalize negative numbers to positive */
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }

    /* Standard Euclidean algorithm: gcd(a, b) = gcd(b, a % b) */
    while (b != 0) {
        long long temp = b;
        b = a % b;
        a = temp;
    }

    return a;
}

long long lcm(long long a, long long b) {
    if (a == 0 || b == 0) {
        return 0;
    }

    long long g = gcd(a, b);
    if (g == 0) {
        return 0;
    }

    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }

    /* Divide first to avoid premature integer overflow */
    return (a / g) * b;
}

long long factorial(int n) {
    /* Factorial is undefined for negative integers; 20! is max signed 64-bit value */
    if (n < 0 || n > 20) {
        return -1;
    }

    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }

    return result;
}

long long fibonacci(int n) {
    /* 0-indexed Fibonacci sequence: F_0 = 0, F_1 = 1; F_92 is max signed 64-bit value */
    if (n < 0 || n > 92) {
        return -1;
    }

    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return 1;
    }

    long long prev = 0;
    long long curr = 1;

    for (int i = 2; i <= n; i++) {
        long long next = prev + curr;
        prev = curr;
        curr = next;
    }

    return curr;
}


