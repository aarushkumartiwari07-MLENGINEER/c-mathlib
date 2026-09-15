"""
Demonstration script showcasing c-mathlib functions executed in native C.
"""

from c_mathlib import (
    is_prime,
    gcd,
    lcm,
    factorial,
    fibonacci,
    mod_pow,
    extended_gcd,
    mod_inverse,
    __version__,
)


def main():
    print(f"=== c-mathlib v{__version__} Arithmetic & Number Theory Demonstration ===")
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

    # 3. Factorial & Fibonacci
    print("\n--- 3. Factorial & Fibonacci ---")
    for n in [0, 1, 5, 10, 15, 20]:
        print(f"{n:2d}! = {factorial(n):20d}    |    F({n:2d}) = {fibonacci(n):10d}")

    # 4. Modular Exponentiation
    print("\n--- 4. Modular Exponentiation (mod_pow) ---")
    pow_cases = [(2, 10, 1000), (3, 4, 5), (7, 100, 13), (5, 1000000, 7)]
    for base, exp, mod in pow_cases:
        res = mod_pow(base, exp, mod)
        print(f"({base}^{exp}) mod {mod} = {res}")

    # 5. Extended Euclidean Algorithm (Bézout's identity)
    print("\n--- 5. Extended Euclidean Algorithm (extended_gcd) ---")
    for a, b in [(30, 12), (35, 15), (1071, 462), (17, 19)]:
        g, x, y = extended_gcd(a, b)
        print(f"extended_gcd({a:4d}, {b:4d}) -> gcd={g:2d}, x={x:3d}, y={y:3d}  |  {a}*({x}) + {b}*({y}) = {a*x + b*y}")

    # 6. Modular Multiplicative Inverse
    print("\n--- 6. Modular Inverse (mod_inverse) ---")
    inv_cases = [(3, 11), (10, 17), (7, 26), (15, 26)]
    for a, m in inv_cases:
        inv = mod_inverse(a, m)
        print(f"mod_inverse({a:2d}, mod {m:2d}) = {inv:2d}  (check: ({a} * {inv}) % {m} = {(a * inv) % m})")


if __name__ == "__main__":
    main()


