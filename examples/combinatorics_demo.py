"""
Demonstration script showcasing c-mathlib's combinatorics and Euler totient functions in native C.
"""

from c_mathlib.combinatorics import (
    combinations,
    permutations,
    euler_totient,
    nCr,
    nPr,
)


def main():
    print("=== c-mathlib Combinatorics & Discrete Mathematics Demonstration ===")
    print("All combinatorial routines are executed in native C.\n")

    # 1. Combinations nCr
    print("--- 1. Combinations nCr ---")
    comb_cases = [(5, 2), (10, 3), (20, 5), (52, 5), (60, 4)]
    for n, k in comb_cases:
        res = combinations(n, k)
        print(f"C({n:2d}, {k:2d}) = {nCr(n, k):10d}")
    print("5-card poker hands from 52-card deck: C(52, 5) =", combinations(52, 5))

    # 2. Permutations nPr
    print("\n--- 2. Permutations nPr ---")
    perm_cases = [(5, 2), (10, 3), (12, 4), (20, 2)]
    for n, k in perm_cases:
        res = permutations(n, k)
        print(f"P({n:2d}, {k:2d}) = {nPr(n, k):10d}")

    # 3. Euler's Totient Function phi(n)
    print("\n--- 3. Euler's Totient Function phi(n) ---")
    totient_cases = [1, 7, 9, 10, 12, 36, 100, 1000]
    for n in totient_cases:
        phi = euler_totient(n)
        print(f"phi({n:4d}) = {phi:4d}  (number of integers 1 <= k <= {n} coprime to {n})")


if __name__ == "__main__":
    main()
