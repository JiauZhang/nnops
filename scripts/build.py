#!/usr/bin/env python3
import subprocess
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
DIST = ROOT / "target" / "wheels"

def run(cmd):
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)

# auto-install maturin if missing
try:
    subprocess.check_call(
        [PYTHON, "-m", "maturin", "--version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    print("maturin not found, installing...")
    run([PYTHON, "-m", "pip", "install", "maturin", "numpy"])

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

# build wheel
run([PYTHON, "-m", "maturin", "build", "--out", str(DIST), *feature_args])

if args.wheel:
    print(f"\nDone! Install with: pip install {DIST}/nnops-*.whl")
else:
    # find the built wheel and install to current Python
    wheels = sorted(DIST.glob("nnops-*.whl"))
    if not wheels:
        print("No wheel found in", DIST)
        exit(1)
    wheel = wheels[-1]
    run([PYTHON, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)])
    print("\nDone! Run 'pytest' to verify.")
    if args.feature == "mps":
        print("MPS tests will be included because MPS is available.")