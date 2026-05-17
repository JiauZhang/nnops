#!/usr/bin/env python3
import subprocess, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run(cmd):
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)

parser = argparse.ArgumentParser(description="Build nnops locally")
parser.add_argument("--feature", "-f", choices=["mps", "cuda"],
                    help="additional backend feature to build with")
parser.add_argument("--release", "-r", action="store_true", help="build in release mode")
parser.add_argument("--wheel", "-w", action="store_true", help="build wheel without installing")
parser.add_argument("--clean", "-c", action="store_true", help="clean before building")
args = parser.parse_args()

feature_args = []
if args.feature:
    feature_args = ["--features", args.feature]

if args.clean:
    run(["cargo", "clean", "-p", "nnops-python"])

if args.release:
    feature_args.append("--release")

if args.wheel:
    run(["maturin", "build", *feature_args])
    print("\nDone! Install with: pip install target/wheels/nnops-*.whl")
else:
    run(["maturin", "develop", *feature_args])