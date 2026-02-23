#!/usr/bin/env python3
"""
calibrate_frozen.py — CLI entry point for MABE scorer calibration.

Usage:
    python calibrate_frozen.py              # Run calibration, show summary
    python calibrate_frozen.py --verbose    # Show per-complex breakdown
    python calibrate_frozen.py --baseline   # Evaluate current params only (no optimization)
"""

import sys
from cal_optimizer import run_calibration, evaluate
from cal_params import get_x0


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    baseline_only = "--baseline" in sys.argv

    if baseline_only:
        print("── Baseline evaluation (no optimization) ──")
        x0 = get_x0()
        evaluate(x0, verbose=verbose)
    else:
        x_opt, r2, mae = run_calibration(verbose=verbose, export=True)


if __name__ == "__main__":
    main()