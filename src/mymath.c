#define BUILDING_MYMATH_DLL
#include "mymath.h"
#include <math.h>


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

double eval_math_func(int func_id, double x) {
    switch (func_id) {
        case FUNC_IDENTITY:   return x;
        case FUNC_SQUARE:     return x * x;
        case FUNC_CUBE:       return x * x * x;
        case FUNC_QUADRATIC:  return x * x - 4.0;
        case FUNC_CUBIC_TEST: return x * x * x - x - 2.0;
        case FUNC_SIN:        return sin(x);
        case FUNC_COS:        return cos(x);
        case FUNC_EXP:        return exp(x);
        case FUNC_LOG:        return log(x);
        case FUNC_RECIPROCAL: return 1.0 / x;
        default:              return 0.0;
    }
}

double eval_polynomial(const double *coeffs, int num_coeffs, double x) {
    if (num_coeffs <= 0 || coeffs == (void*)0) {
        return 0.0;
    }
    double result = coeffs[num_coeffs - 1];
    for (int i = num_coeffs - 2; i >= 0; i--) {
        result = result * x + coeffs[i];
    }
    return result;
}

int bisection_method(int func_id, double a, double b, double tol, int max_iter, double *root) {
    if (tol <= 0.0 || max_iter <= 0 || root == (void*)0) {
        return -3;
    }
    if (func_id < 0 || func_id > 9) {
        return -3;
    }

    double fa = eval_math_func(func_id, a);
    double fb = eval_math_func(func_id, b);

    if (fa == 0.0) {
        *root = a;
        return 0;
    }
    if (fb == 0.0) {
        *root = b;
        return 0;
    }

    if (fa * fb > 0.0) {
        return -1; /* Root is not bracketed: f(a) and f(b) have the same sign */
    }

    double c = a;
    for (int iter = 0; iter < max_iter; iter++) {
        c = a + (b - a) / 2.0;
        double fc = eval_math_func(func_id, c);

        if (fabs(fc) < tol || (b - a) / 2.0 < tol) {
            *root = c;
            return 0;
        }

        if (fa * fc < 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    *root = c;
    return -2; /* Max iterations reached */
}

int bisection_poly(const double *coeffs, int num_coeffs, double a, double b, double tol, int max_iter, double *root) {
    if (tol <= 0.0 || max_iter <= 0 || root == (void*)0 || coeffs == (void*)0 || num_coeffs <= 0) {
        return -3;
    }

    double fa = eval_polynomial(coeffs, num_coeffs, a);
    double fb = eval_polynomial(coeffs, num_coeffs, b);

    if (fa == 0.0) {
        *root = a;
        return 0;
    }
    if (fb == 0.0) {
        *root = b;
        return 0;
    }

    if (fa * fb > 0.0) {
        return -1;
    }

    double c = a;
    for (int iter = 0; iter < max_iter; iter++) {
        c = a + (b - a) / 2.0;
        double fc = eval_polynomial(coeffs, num_coeffs, c);

        if (fabs(fc) < tol || (b - a) / 2.0 < tol) {
            *root = c;
            return 0;
        }

        if (fa * fc < 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    *root = c;
    return -2;
}

int simpson_rule(int func_id, double a, double b, int n, double *result) {
    if (n < 2 || (n % 2 != 0) || result == (void*)0) {
        return -1;
    }
    if (func_id < 0 || func_id > 9) {
        return -2;
    }

    double h = (b - a) / (double)n;
    double sum = eval_math_func(func_id, a) + eval_math_func(func_id, b);

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        double weight = (i % 2 == 0) ? 2.0 : 4.0;
        sum += weight * eval_math_func(func_id, x);
    }

    *result = sum * (h / 3.0);
    return 0;
}

int simpson_poly(const double *coeffs, int num_coeffs, double a, double b, int n, double *result) {
    if (n < 2 || (n % 2 != 0) || result == (void*)0 || coeffs == (void*)0 || num_coeffs <= 0) {
        return -1;
    }

    double h = (b - a) / (double)n;
    double sum = eval_polynomial(coeffs, num_coeffs, a) + eval_polynomial(coeffs, num_coeffs, b);

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        double weight = (i % 2 == 0) ? 2.0 : 4.0;
        sum += weight * eval_polynomial(coeffs, num_coeffs, x);
    }

    *result = sum * (h / 3.0);
    return 0;
}

int numerical_derivative(int func_id, double x, double h, double *result) {
    if (h <= 0.0 || result == (void*)0) {
        return -1;
    }
    if (func_id < 0 || func_id > 9) {
        return -2;
    }

    double f_plus = eval_math_func(func_id, x + h);
    double f_minus = eval_math_func(func_id, x - h);

    *result = (f_plus - f_minus) / (2.0 * h);
    return 0;
}

int numerical_derivative_poly(const double *coeffs, int num_coeffs, double x, double h, double *result) {
    if (h <= 0.0 || result == (void*)0 || coeffs == (void*)0 || num_coeffs <= 0) {
        return -1;
    }

    double f_plus = eval_polynomial(coeffs, num_coeffs, x + h);
    double f_minus = eval_polynomial(coeffs, num_coeffs, x - h);

    *result = (f_plus - f_minus) / (2.0 * h);
    return 0;
}




