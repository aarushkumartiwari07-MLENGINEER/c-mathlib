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
     * Check divisors of the form (6k +/- 1) up to sqrt(n).
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

static inline unsigned long long mul_mod_u64(unsigned long long a, unsigned long long b, unsigned long long m) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__SIZEOF_INT128__)
    __extension__ typedef unsigned __int128 uint128_t;
    return (unsigned long long)(((uint128_t)a * (uint128_t)b) % (uint128_t)m);
#else
    unsigned long long res = 0;
    a %= m;
    while (b > 0) {
        if (b & 1) {
            res = (res + a) % m;
        }
        a = (a * 2) % m;
        b >>= 1;
    }
    return res;
#endif
}

long long mod_pow(long long base, long long exponent, long long modulus) {
    if (modulus <= 0 || exponent < 0) {
        return -1;
    }
    if (modulus == 1) {
        return 0;
    }

    unsigned long long result = 1;
    long long b_rem = base % modulus;
    if (b_rem < 0) {
        b_rem += modulus;
    }
    unsigned long long b_u = (unsigned long long)b_rem;
    unsigned long long m_u = (unsigned long long)modulus;

    /* Exponentiation by squaring: O(log exponent) complexity */
    while (exponent > 0) {
        if (exponent & 1) {
            result = mul_mod_u64(result, b_u, m_u);
        }
        b_u = mul_mod_u64(b_u, b_u, m_u);
        exponent >>= 1;
    }

    return (long long)result;
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

    /* Ensure returned GCD is non-negative and Bezout identity a*x + b*y = g holds */
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

    /* Map Bezout coefficient to standard range [0, modulus - 1] */
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

static double eval_poly_derivative(const double *coeffs, int num_coeffs, double x) {
    if (num_coeffs <= 1 || coeffs == (void*)0) {
        return 0.0;
    }
    double result = (num_coeffs - 1) * coeffs[num_coeffs - 1];
    for (int i = num_coeffs - 2; i >= 1; i--) {
        result = result * x + i * coeffs[i];
    }
    return result;
}

int newton_raphson_method(int func_id, double x0, double tol, int max_iter, double *root) {
    if (tol <= 0.0 || max_iter <= 0 || root == (void*)0) {
        return -3;
    }
    if (func_id < 0 || func_id > 9) {
        return -3;
    }

    double x = x0;
    double h = 1e-6;

    for (int iter = 0; iter < max_iter; iter++) {
        double fx = eval_math_func(func_id, x);
        if (fabs(fx) < tol) {
            *root = x;
            return 0;
        }

        double dfx;
        numerical_derivative(func_id, x, h, &dfx);
        if (fabs(dfx) < 1e-14) {
            return -4;
        }

        double delta = fx / dfx;
        x -= delta;

        if (fabs(delta) < tol) {
            *root = x;
            return 0;
        }
    }

    *root = x;
    return -2;
}

int newton_poly(const double *coeffs, int num_coeffs, double x0, double tol, int max_iter, double *root) {
    if (tol <= 0.0 || max_iter <= 0 || root == (void*)0 || coeffs == (void*)0 || num_coeffs <= 0) {
        return -3;
    }

    double x = x0;
    for (int iter = 0; iter < max_iter; iter++) {
        double fx = eval_polynomial(coeffs, num_coeffs, x);
        if (fabs(fx) < tol) {
            *root = x;
            return 0;
        }

        double dfx = eval_poly_derivative(coeffs, num_coeffs, x);
        if (fabs(dfx) < 1e-14) {
            return -4;
        }

        double delta = fx / dfx;
        x -= delta;

        if (fabs(delta) < tol) {
            *root = x;
            return 0;
        }
    }

    *root = x;
    return -2;
}

int secant_method(int func_id, double x0, double x1, double tol, int max_iter, double *root) {
    if (tol <= 0.0 || max_iter <= 0 || root == (void*)0) {
        return -3;
    }
    if (func_id < 0 || func_id > 9) {
        return -3;
    }

    double f0 = eval_math_func(func_id, x0);
    double f1 = eval_math_func(func_id, x1);

    for (int iter = 0; iter < max_iter; iter++) {
        if (fabs(f1) < tol) {
            *root = x1;
            return 0;
        }

        double denom = f1 - f0;
        if (fabs(denom) < 1e-14) {
            return -4;
        }

        double x_next = x1 - f1 * (x1 - x0) / denom;
        x0 = x1;
        f0 = f1;
        x1 = x_next;
        f1 = eval_math_func(func_id, x1);

        if (fabs(x1 - x0) < tol) {
            *root = x1;
            return 0;
        }
    }

    *root = x1;
    return -2;
}

int secant_poly(const double *coeffs, int num_coeffs, double x0, double x1, double tol, int max_iter, double *root) {
    if (tol <= 0.0 || max_iter <= 0 || root == (void*)0 || coeffs == (void*)0 || num_coeffs <= 0) {
        return -3;
    }

    double f0 = eval_polynomial(coeffs, num_coeffs, x0);
    double f1 = eval_polynomial(coeffs, num_coeffs, x1);

    for (int iter = 0; iter < max_iter; iter++) {
        if (fabs(f1) < tol) {
            *root = x1;
            return 0;
        }

        double denom = f1 - f0;
        if (fabs(denom) < 1e-14) {
            return -4;
        }

        double x_next = x1 - f1 * (x1 - x0) / denom;
        x0 = x1;
        f0 = f1;
        x1 = x_next;
        f1 = eval_polynomial(coeffs, num_coeffs, x1);

        if (fabs(x1 - x0) < tol) {
            *root = x1;
            return 0;
        }
    }

    *root = x1;
    return -2;
}

static double eval_ode_func(int ode_id, double t, double y) {
    switch (ode_id) {
        case ODE_EXP_DECAY:       return -y;
        case ODE_LOGISTIC:        return y * (1.0 - y);
        case ODE_LINEAR:          return t + y;
        case ODE_SINE:            return cos(t);
        case ODE_HARMONIC_ACCEL:  return -t * y;
        default:                  return 0.0;
    }
}

int rk4_solve(
    int ode_id,
    double y0,
    double t0,
    double t1,
    int steps,
    double *t_out,
    double *y_out
) {
    if (steps < 1 || t_out == (void*)0 || y_out == (void*)0) {
        return -1;
    }
    if (ode_id < 0 || ode_id > 4) {
        return -2;
    }

    double h = (t1 - t0) / (double)steps;
    double t = t0;
    double y = y0;

    t_out[0] = t;
    y_out[0] = y;

    for (int i = 0; i < steps; i++) {
        double k1 = eval_ode_func(ode_id, t, y);
        double k2 = eval_ode_func(ode_id, t + 0.5 * h, y + 0.5 * h * k1);
        double k3 = eval_ode_func(ode_id, t + 0.5 * h, y + 0.5 * h * k2);
        double k4 = eval_ode_func(ode_id, t + h, y + h * k3);

        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t = t0 + (i + 1) * h;

        t_out[i + 1] = t;
        y_out[i + 1] = y;
    }

    return 0;
}

int rk4_poly(
    const double *coeffs,
    int num_coeffs,
    double y0,
    double t0,
    double t1,
    int steps,
    double *t_out,
    double *y_out
) {
    if (steps < 1 || t_out == (void*)0 || y_out == (void*)0 || coeffs == (void*)0 || num_coeffs <= 0) {
        return -1;
    }

    double h = (t1 - t0) / (double)steps;
    double t = t0;
    double y = y0;

    t_out[0] = t;
    y_out[0] = y;

    for (int i = 0; i < steps; i++) {
        double k1 = eval_polynomial(coeffs, num_coeffs, t);
        double k2 = eval_polynomial(coeffs, num_coeffs, t + 0.5 * h);
        double k3 = k2;
        double k4 = eval_polynomial(coeffs, num_coeffs, t + h);

        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t = t0 + (i + 1) * h;

        t_out[i + 1] = t;
        y_out[i + 1] = y;
    }

    return 0;
}

#define GOLDEN_RATIO_INV 0.61803398874989484820

int golden_section_minimize(
    int func_id,
    double a,
    double b,
    double tol,
    int max_iter,
    double *min_x,
    double *min_val
) {
    if (tol <= 0.0 || max_iter <= 0 || min_x == (void*)0 || min_val == (void*)0) {
        return -1;
    }
    if (func_id < 0 || func_id > 9) {
        return -2;
    }

    if (a > b) {
        double tmp = a;
        a = b;
        b = tmp;
    }

    double c = b - GOLDEN_RATIO_INV * (b - a);
    double d = a + GOLDEN_RATIO_INV * (b - a);
    double fc = eval_math_func(func_id, c);
    double fd = eval_math_func(func_id, d);

    int iter = 0;
    while ((b - a) > tol && iter < max_iter) {
        iter++;
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - GOLDEN_RATIO_INV * (b - a);
            fc = eval_math_func(func_id, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + GOLDEN_RATIO_INV * (b - a);
            fd = eval_math_func(func_id, d);
        }
    }

    *min_x = 0.5 * (a + b);
    *min_val = eval_math_func(func_id, *min_x);

    if ((b - a) > tol) {
        return -2;
    }

    return iter;
}

int golden_section_poly(
    const double *coeffs,
    int num_coeffs,
    double a,
    double b,
    double tol,
    int max_iter,
    double *min_x,
    double *min_val
) {
    if (coeffs == (void*)0 || num_coeffs <= 0 || tol <= 0.0 || max_iter <= 0 || min_x == (void*)0 || min_val == (void*)0) {
        return -1;
    }

    if (a > b) {
        double tmp = a;
        a = b;
        b = tmp;
    }

    double c = b - GOLDEN_RATIO_INV * (b - a);
    double d = a + GOLDEN_RATIO_INV * (b - a);
    double fc = eval_polynomial(coeffs, num_coeffs, c);
    double fd = eval_polynomial(coeffs, num_coeffs, d);

    int iter = 0;
    while ((b - a) > tol && iter < max_iter) {
        iter++;
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - GOLDEN_RATIO_INV * (b - a);
            fc = eval_polynomial(coeffs, num_coeffs, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + GOLDEN_RATIO_INV * (b - a);
            fd = eval_polynomial(coeffs, num_coeffs, d);
        }
    }

    *min_x = 0.5 * (a + b);
    *min_val = eval_polynomial(coeffs, num_coeffs, *min_x);

    if ((b - a) > tol) {
        return -2;
    }

    return iter;
}




