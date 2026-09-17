"""
Demo script for Special Functions and Machine Learning Activation Primitives.
"""

import math
from c_mathlib.special import gamma, lgamma, beta, erf, erfc, sigmoid, softmax


def demo_special_functions():
    print("=" * 60)
    print("1. SPECIAL FUNCTIONS (GAMMA, BETA, ERROR FUNCTION)")
    print("=" * 60)

    # Gamma and Factorials
    print("Gamma Function:")
    for n in range(1, 7):
        print(f"  Gamma({n}) = {gamma(n):8.1f} (matches ({n-1})! = {math.factorial(n-1)})")
    print(f"  Gamma(0.5) = {gamma(0.5):.8f} (matches sqrt(pi) = {math.sqrt(math.pi):.8f})")
    print(f"  ln|Gamma(10)| = {lgamma(10):.8f}")

    # Beta Function
    print("\nBeta Function:")
    print(f"  Beta(2, 3) = {beta(2, 3):.8f} (1/12 = {1/12:.8f})")
    print(f"  Beta(0.5, 0.5) = {beta(0.5, 0.5):.8f} (pi = {math.pi:.8f})")

    # Error Function
    print("\nError Functions (erf & erfc):")
    for x in [0.0, 0.5, 1.0, 1.5, 2.0]:
        print(f"  x = {x:3.1f} -> erf(x) = {erf(x):.6f}, erfc(x) = {erfc(x):.6f} (sum = {erf(x) + erfc(x):.6f})")


def demo_ml_primitives():
    print("\n" + "=" * 60)
    print("2. MACHINE LEARNING ACTIVATION PRIMITIVES")
    print("=" * 60)

    # Sigmoid
    print("Logistic Sigmoid sigma(x):")
    for x in [-5.0, -2.0, 0.0, 2.0, 5.0]:
        print(f"  sigmoid({x:+4.1f}) = {sigmoid(x):.6f}")

    # Softmax
    logits = [2.0, 1.0, 0.1]
    probs = softmax(logits)
    print(f"\nSoftmax Probabilities for logits {logits}:")
    for logit, prob in zip(logits, probs):
        print(f"  logit = {logit:4.1f} -> P(class) = {prob:6.4f} ({prob*100:5.1f}%)")
    print(f"  Total sum = {sum(probs):.6f}")

    # Numerical stability test with extreme logits
    extreme_logits = [1000.0, 1002.0, 999.0]
    extreme_probs = softmax(extreme_logits)
    print(f"\nSoftmax on extreme logits (overflow-safe) {extreme_logits}:")
    for logit, prob in zip(extreme_logits, extreme_probs):
        print(f"  logit = {logit:6.1f} -> P(class) = {prob:6.4f} ({prob*100:5.1f}%)")


if __name__ == "__main__":
    demo_special_functions()
    demo_ml_primitives()
