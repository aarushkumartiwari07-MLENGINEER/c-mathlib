#define BUILDING_MYMATH_DLL
#include "special.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_LN2PI
#define M_LN2PI 1.83787706640934548356  /* ln(2 * pi) */
#endif

#ifndef M_LNPI
#define M_LNPI 1.14472988584940017414   /* ln(pi) */
#endif

/* Lanczos approximation coefficients for g = 7, n = 9 */
static const double LANCZOS_P[] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109583115912,
    9.9843695780195716e-6,
    1.5056327351493116e-7
};

static int is_non_positive_integer(double x) {
    if (x > 0.0) return 0;
    double rounded = floor(x + 0.5);
    return fabs(x - rounded) < 1e-15;
}

int math_lgamma(double x, double *result) {
    if (result == (void*)0) {
        return -1;
    }
    if (is_non_positive_integer(x)) {
        return -1;
    }

    if (x < 0.5) {
        /* Reflection formula: ln|Gamma(x)| = ln(pi) - ln|sin(pi * x)| - ln|Gamma(1 - x)| */
        double lg_1_minus_x;
        int status = math_lgamma(1.0 - x, &lg_1_minus_x);
        if (status != 0) {
            return status;
        }
        double sin_val = fabs(sin(M_PI * x));
        if (sin_val == 0.0) {
            return -1;
        }
        *result = M_LNPI - log(sin_val) - lg_1_minus_x;
        return 0;
    }

    double z = x - 1.0;
    double base = LANCZOS_P[0];
    for (int i = 1; i < 9; i++) {
        base += LANCZOS_P[i] / (z + (double)i);
    }

    double t = z + 7.5; /* z + g + 0.5 where g = 7 */
    *result = 0.5 * M_LN2PI + (z + 0.5) * log(t) - t + log(base);
    return 0;
}

int math_gamma(double x, double *result) {
    if (result == (void*)0) {
        return -1;
    }
    if (is_non_positive_integer(x)) {
        return -1;
    }

    if (x < 0.5) {
        /* Reflection formula: Gamma(x) = pi / (sin(pi * x) * Gamma(1 - x)) */
        double g_1_minus_x;
        int status = math_gamma(1.0 - x, &g_1_minus_x);
        if (status != 0) {
            return status;
        }
        double sin_val = sin(M_PI * x);
        if (fabs(sin_val) < 1e-15 || g_1_minus_x == 0.0) {
            return -1;
        }
        *result = M_PI / (sin_val * g_1_minus_x);
        return 0;
    }

    double z = x - 1.0;
    double base = LANCZOS_P[0];
    for (int i = 1; i < 9; i++) {
        base += LANCZOS_P[i] / (z + (double)i);
    }

    double t = z + 7.5;
    *result = sqrt(2.0 * M_PI) * pow(t, z + 0.5) * exp(-t) * base;
    return 0;
}

int math_beta(double a, double b, double *result) {
    if (result == (void*)0 || is_non_positive_integer(a) || is_non_positive_integer(b) || is_non_positive_integer(a + b)) {
        return -1;
    }

    double lga, lgb, lgab;
    if (math_lgamma(a, &lga) != 0 || math_lgamma(b, &lgb) != 0 || math_lgamma(a + b, &lgab) != 0) {
        return -1;
    }

    *result = exp(lga + lgb - lgab);
    return 0;
}

int math_erf(double x, double *result) {
    if (result == (void*)0) {
        return -1;
    }
    if (x == 0.0) {
        *result = 0.0;
        return 0;
    }

    /* Abramowitz and Stegun formula 7.1.26 (max error: 1.5e-7) */
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x < 0.0) ? -1 : 1;
    double abs_x = fabs(x);

    double t = 1.0 / (1.0 + p * abs_x);
    double poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    double y = 1.0 - poly * exp(-abs_x * abs_x);

    *result = sign * y;
    return 0;
}

int math_erfc(double x, double *result) {
    if (result == (void*)0) {
        return -1;
    }

    double erf_val;
    int status = math_erf(x, &erf_val);
    if (status != 0) {
        return status;
    }

    *result = 1.0 - erf_val;
    return 0;
}

double math_sigmoid(double x) {
    if (x >= 0.0) {
        double exp_neg = exp(-x);
        return 1.0 / (1.0 + exp_neg);
    } else {
        double exp_pos = exp(x);
        return exp_pos / (1.0 + exp_pos);
    }
}

int math_softmax(const double *x, int n, double *out) {
    if (x == (void*)0 || out == (void*)0 || n < 1) {
        return -1;
    }

    /* Find maximum element for numerical stability */
    double max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    /* Exponentiate shifted values and accumulate denominator sum */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(x[i] - max_val);
        sum += out[i];
    }

    if (sum <= 0.0) {
        return -1;
    }

    /* Normalize to probability distribution */
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }

    return 0;
}
