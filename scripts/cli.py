"""Unified HDIM command line interface."""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


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


def _run_tune(argv: Sequence[str]) -> object:
    from .auto_tune import main as tune_main

    return _forward_to(tune_main, argv)


def _run_benchmark(argv: Sequence[str]) -> object:
    if any(arg in {"-h", "--help"} for arg in argv):
        argparse.ArgumentParser(prog="hdim benchmark", description="Run SOTA benchmarks").parse_args(argv)

    from .benchmark_comparison import main as benchmark_main

    return benchmark_main()


def _run_chat(argv: Sequence[str]) -> object:
    from .interactive_kernel_chat import main as chat_main

    return _forward_to(chat_main, argv)


def _run_test(argv: Sequence[str]) -> object:
    if any(arg in {"-h", "--help"} for arg in argv):
        argparse.ArgumentParser(prog="hdim test", description="Run module tests").parse_args(argv)

    return runpy.run_path(str(Path(__file__).with_name("test_all_modules.py")), run_name="__main__")


def _run_profile(argv: Sequence[str]) -> None:
    profile_parser = argparse.ArgumentParser(prog="hdim profile")
    profile_parser.add_argument(
        "--mode",
        choices=["all", "perf", "memory"],
        default="all",
        help="Profile mode: performance, GPU memory, or both",
    )
    args = profile_parser.parse_args(argv)

    if args.mode in {"all", "perf"}:
        from .perf_profile import main as perf_main

        perf_main()

    if args.mode in {"all", "memory"}:
        memory_module = importlib.import_module("scripts.gpu_memory_profile")
        memory_module.profile_gpu_memory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hdim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Run training", add_help=False)
    subparsers.add_parser("tune", help="Hyperparameter tuning", add_help=False)
    subparsers.add_parser("benchmark", help="Run SOTA benchmarks")
    subparsers.add_parser("chat", help="Run interactive chat", add_help=False)
    subparsers.add_parser("test", help="Run module tests")
    subparsers.add_parser("profile", help="Run performance and GPU memory profiling", add_help=False)

    return parser


def main(argv: Sequence[str] | None = None) -> object:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, remaining = parser.parse_known_args(raw_argv)

    if args.command == "train":
        return _run_train(remaining)
    if args.command == "tune":
        return _run_tune(remaining)
    if args.command == "benchmark":
        return _run_benchmark(remaining)
    if args.command == "chat":
        return _run_chat(remaining)
    if args.command == "test":
        return _run_test(remaining)
    if args.command == "profile":
        return _run_profile(remaining)

    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main()
