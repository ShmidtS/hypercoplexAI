"""Numerical verification of all Lean4 formalization theorems for HDIM."""
import argparse

try:
    from .numerical.runner import main
except ImportError:
    from numerical.runner import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Numerical verification of Lean4 formalization theorems for HDIM.")
    parser.add_argument("--category", choices=["algebra", "memory", "moe", "hallucination", "all"], default="all")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(["--category", args.category]))
