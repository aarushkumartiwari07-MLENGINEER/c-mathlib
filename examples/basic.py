"""
Demonstration script showcasing c-mathlib functions executed in native C.
"""

from c_mathlib import (
    is_prime,
    gcd,
    lcm,
    factorial,
    fibonacci,
    __version__,
)


def main():
    print(f"=== c-mathlib v{__version__} Demonstration ===")
    print("All mathematical computations are executed in native C via ctypes.\n")

    # 1. Primality Testing
    print("--- 1. Primality Testing (is_prime) ---")
    primes_to_test = [-5, 0, 1, 2, 3, 4, 17, 18, 25, 97, 100, 104729]
    for n in primes_to_test:
        result = is_prime(n)
        status = "PRIME" if result else "COMPOSITE / NON-PRIME"
        print(f"is_prime({n:7d})  ->  {str(result):5s} ({status})")

    # 2. GCD & LCM
    print("\n--- 2. GCD & LCM ---")
    pairs = [(48, 18), (12, 18), (1071, 462), (13, 7), (100, 25), (-48, 18), (0, 5)]
    for a, b in pairs:
        g = gcd(a, b)
        l = lcm(a, b)
        print(f"gcd({a:5d}, {b:5d}) = {g:5d}    |    lcm({a:5d}, {b:5d}) = {l:5d}")

    # 3. Factorial
    print("\n--- 3. Factorial ---")
    for n in [0, 1, 3, 5, 7, 10, 12, 20]:
        print(f"{n:2d}! = {factorial(n)}")

    # 4. Fibonacci Sequence
    print("\n--- 4. Fibonacci Sequence ---")
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 92]:
        print(f"F({n:2d}) = {fibonacci(n)}")


if __name__ == "__main__":
    main()

