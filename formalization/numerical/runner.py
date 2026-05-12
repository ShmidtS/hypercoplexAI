"""Runner for HDIM numerical verification categories."""
from __future__ import annotations

import argparse

from . import algebra_checks, hallucination_checks, memory_checks, moe_checks

CATEGORIES = {
    "algebra": algebra_checks.run_checks,
    "memory": memory_checks.run_checks,
    "moe": moe_checks.run_checks,
    "hallucination": hallucination_checks.run_checks,
}


def run_category(category: str) -> list[tuple[str, str]]:
    if category == "all":
        results: list[tuple[str, str]] = []
        for run_checks in CATEGORIES.values():
            results.extend(run_checks())
        return results
    return CATEGORIES[category]()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Numerical verification of Lean4 formalization theorems for HDIM.")
    parser.add_argument("--category", choices=["algebra", "memory", "moe", "hallucination", "all"], default="all")
    args = parser.parse_args(argv)

    print('='*60)
    print('LEAN4 NUMERICAL VERIFICATION - HDIM Core Theorems')
    print('='*60)

    results = run_category(args.category)

    print('\n' + '='*60)
    passed = sum(1 for _, s in results if s == 'PASS')
    total = len(results)
    print(f'RESULTS: {passed}/{total} PASS')
    for name, status in results:
        mark = 'OK' if status == 'PASS' else 'FAIL'
        print(f' [{mark}] {name}')
    print('='*60)
    return 0 if passed == total else 1
