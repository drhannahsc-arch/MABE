#!/usr/bin/env python3
"""
design_binder.py — CLI interface for the MABE Generative Design Engine.

Usage:
    python design_binder.py --target "Pb(II)" --matrix "pH 5, Ca, Mg, Fe" --top 10
    python design_binder.py --target Hg2+ --interferents "Pb2+ Cd2+ Zn2+" --pH 7
    python design_binder.py --target Au --matrix "industrial pH 2" --top 5
    python design_binder.py --target Cu --matrix mine_amd --verbose
    python design_binder.py --list-metals
    python design_binder.py --list-matrices

Matrix presets: mine_water, mine_amd, drinking_water, seawater, blood, industrial
"""

import argparse
import sys

from design_engine import (
    design_binder, print_design_result,
    MATRIX_PRESETS, METAL_ALIASES, resolve_metal
)
from scorer_frozen import METAL_DB


def main():
    parser = argparse.ArgumentParser(
        description="MABE Generative Design Engine — Design selective metal binders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target "Pb(II)" --matrix "pH 5, Ca, Mg, Fe" --top 10
  %(prog)s --target Hg2+ --interferents "Pb2+ Cd2+ Zn2+" --pH 7
  %(prog)s --target Au --matrix "industrial pH 2" --top 5 --verbose
  %(prog)s --list-metals
  %(prog)s --list-matrices
        """
    )

    parser.add_argument("--target", "-t",
        help="Target metal (name or formula, e.g. 'Pb(II)', 'Pb2+', 'lead')")
    parser.add_argument("--matrix", "-m",
        help="Matrix specification: 'pH X, Metal1, Metal2, ...' or preset name")
    parser.add_argument("--interferents", "-i",
        help="Space-separated interferent metals (alternative to --matrix)")
    parser.add_argument("--pH", "-p", type=float, default=7.0,
        help="Operating pH (default: 7.0, overridden by matrix)")
    parser.add_argument("--top", "-n", type=int, default=10,
        help="Number of top candidates to show (default: 10)")
    parser.add_argument("--max-enumerate", type=int, default=500,
        help="Maximum candidates to enumerate (default: 500)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Show detailed scoring for top candidate")
    parser.add_argument("--list-metals", action="store_true",
        help="List all available metals and exit")
    parser.add_argument("--list-matrices", action="store_true",
        help="List matrix presets and exit")

    args = parser.parse_args()

    # ── List modes ───────────────────────────────────────────────────
    if args.list_metals:
        print(f"\n  Available metals ({len(METAL_DB)}):\n")
        by_charge = {}
        for formula, m in sorted(METAL_DB.items()):
            by_charge.setdefault(m.charge, []).append(m)
        for charge in sorted(by_charge.keys()):
            metals = by_charge[charge]
            names = [f"{m.formula} ({m.name})" for m in metals]
            print(f"  +{charge}: {', '.join(names)}")
        print()
        return

    if args.list_matrices:
        print("\n  Matrix presets:\n")
        for name, metals in sorted(MATRIX_PRESETS.items()):
            print(f"  {name:20s}  {', '.join(metals)}")
        print(f"\n  Usage: --matrix '{list(MATRIX_PRESETS.keys())[0]} pH 5'")
        print()
        return

    # ── Validate target ──────────────────────────────────────────────
    if not args.target:
        parser.error("--target is required (or use --list-metals)")

    try:
        resolve_metal(args.target)
    except ValueError as e:
        parser.error(str(e))

    # ── Build interferent list ───────────────────────────────────────
    interferents = None
    if args.interferents:
        interferents = args.interferents.split()

    # ── Run design ───────────────────────────────────────────────────
    result = design_binder(
        target=args.target,
        interferents=interferents,
        matrix=args.matrix,
        pH=args.pH,
        top_n=args.top,
        max_enumerate=args.max_enumerate,
    )

    print_design_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()