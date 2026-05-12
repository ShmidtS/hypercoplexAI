"""Unified HDIM command line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


def _forward_to(module_main: Callable[[], object], argv: Sequence[str]) -> object:
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0], *argv]
    try:
        return module_main()
    finally:
        sys.argv = original_argv


def _run_train(argv: Sequence[str]) -> object:
    from .gpu_train import main as train_main

    return _forward_to(train_main, argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hdim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run training", add_help=False)

    return parser


def main(argv: Sequence[str] | None = None) -> object:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, remaining = parser.parse_known_args(raw_argv)

    if args.command == "train":
        return _run_train(remaining)

    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main()
