#!/usr/bin/env python3
"""Run nnops benchmarks with criterion.

Examples:
  # Run all CPU benchmarks
  python scripts/bench.py

  # Run specific CPU benchmark
  python scripts/bench.py --filter matmul

  # Run CPU + MPS benchmarks
  python scripts/bench.py --mps

  # Quick validation with minimal sampling
  python scripts/bench.py --quick --filter binary_add/1024

  # List all available benchmarks
  python scripts/bench.py --list

  # Save baseline for later comparison
  python scripts/bench.py --save-baseline v1
  # ... after code changes, compare:
  python scripts/bench.py --baseline v1

  # Open HTML report after running
  python scripts/bench.py --open
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nnops benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mps", "-m", action="store_true",
        help="also run MPS benchmarks (requires --features mps)",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="run with minimal sampling for quick validation",
    )
    parser.add_argument(
        "--filter", "-f",
        help="only run benchmarks matching this pattern (e.g. 'matmul', 'binary_add/1024')",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="list all benchmarks without running them",
    )
    parser.add_argument(
        "--save-baseline",
        help="save results as a baseline for future comparison (e.g. --save-baseline v1)",
    )
    parser.add_argument(
        "--baseline",
        help="compare against a previously saved baseline (e.g. --baseline v1)",
    )
    parser.add_argument(
        "--open", "-o", action="store_true",
        help="open HTML report in browser after completion",
    )

    args = parser.parse_args()

    # Arguments passed to the benchmark binary (everything after --)
    bench_args: list[str] = []
    if args.filter:
        bench_args.append(args.filter)
    if args.quick:
        bench_args.append("--quick")
    if args.save_baseline:
        bench_args.extend(["--save-baseline", args.save_baseline])
    if args.baseline:
        bench_args.extend(["--baseline", args.baseline])

    # Determine which bench targets to run
    benches: list[tuple[str, str | None]] = [("cpu_ops", None)]
    if args.mps:
        benches.append(("mps_ops", "mps"))

    for bench_name, features in benches:
        cmd = ["cargo", "bench", "--bench", bench_name]
        if features:
            cmd.extend(["--features", features])
        if args.list:
            # --list is a criterion binary flag, must go after --
            cmd.append("--")
            cmd.append("--list")
        elif bench_args:
            cmd.append("--")
            cmd.extend(bench_args)
        run(cmd)

    # Open HTML report
    if args.open:
        report_dir = ROOT / "target" / "criterion"
        if report_dir.is_dir():
            index = report_dir / "report" / "index.html"
            if index.is_file():
                import webbrowser
                webbrowser.open(index.as_uri())
                print(f"\nOpened {index}")
            else:
                print(f"\nReport not found at {index}")
        else:
            print(f"\nNo criterion report found at {report_dir}")
            print("Run a benchmark first, e.g.  python scripts/bench.py")


if __name__ == "__main__":
    main()