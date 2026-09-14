"""
Basic usage example for c-mathlib.
"""

from c_mathlib import is_prime, __version__


def main():
    print(f"=== c-mathlib v{__version__} Demonstration ===")
    print("Underlying primality algorithm is executed in native C.\n")

    test_numbers = [-5, 0, 1, 2, 3, 4, 17, 18, 25, 97, 100, 104729]

    for n in test_numbers:
        result = is_prime(n)
        status = "PRIME" if result else "COMPOSITE / NON-PRIME"
        print(f"is_prime({n:6d}) -> {str(result):5s} ({status})")


if __name__ == "__main__":
    main()
