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

long long mod_pow(long long base, long long exponent, long long modulus) {
    if (modulus <= 0 || exponent < 0) {
        return -1;
    }
    if (modulus == 1) {
        return 0;
    }

    long long result = 1;
    base = base % modulus;
    if (base < 0) {
        base += modulus;
    }

    /* Exponentiation by squaring: O(log exponent) complexity */
    while (exponent > 0) {
        if (exponent & 1) {
            result = (long long)(((unsigned __int128)result * (unsigned long long)base) % (unsigned long long)modulus);
        }
        base = (long long)(((unsigned __int128)base * (unsigned long long)base) % (unsigned long long)modulus);
        exponent >>= 1;
    }

    return result;
}

long long extended_gcd(long long a, long long b, long long *x, long long *y) {
    long long old_r = a, r = b;
    long long old_s = 1, s = 0;
    long long old_t = 0, t = 1;

    /* Iterative Extended Euclidean algorithm */
    while (r != 0) {
        long long quotient = old_r / r;
        long long temp_r = old_r - quotient * r;
        old_r = r;
        r = temp_r;

        long long temp_s = old_s - quotient * s;
        old_s = s;
        s = temp_s;

        long long temp_t = old_t - quotient * t;
        old_t = t;
        t = temp_t;
    }

    /* Ensure returned GCD is non-negative and Bézout identity a*x + b*y = g holds */
    if (old_r < 0) {
        old_r = -old_r;
        old_s = -old_s;
        old_t = -old_t;
    }

    if (x != (void*)0) {
        *x = old_s;
    }
    if (y != (void*)0) {
        *y = old_t;
    }

    return old_r;
}

long long mod_inverse(long long a, long long modulus) {
    if (modulus <= 1) {
        return -1;
    }

    long long x, y;
    long long g = extended_gcd(a, modulus, &x, &y);

    if (g != 1) {
        /* Modular inverse exists if and only if gcd(a, modulus) == 1 */
        return -1;
    }

    /* Map Bézout coefficient to standard range [0, modulus - 1] */
    long long inv = (x % modulus + modulus) % modulus;
    return inv;
}



