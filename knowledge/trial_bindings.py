#!/usr/bin/env python
"""
trial_bindings.py -- Run trial glycan binding predictions from your local MABE repo.

Usage (from MABE root):
    python trial_bindings.py
    python trial_bindings.py --target GalNAc
    python trial_bindings.py --target Glc --candidates 500
    python trial_bindings.py --score-only  (skip de novo, just run lectin scorer)

Requires bootstraps applied:
    bootstrap_glycan_expansion.py
    bootstrap_glycan_denovo.py

Requires: RDKit (pip install rdkit)
"""

import argparse
import sys
import os
import time

# ── Ensure repo root is on path ──
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(_here, "core", "universal_schema.py")):
    ROOT = _here
elif os.path.isfile(os.path.join(_here, "..", "core", "universal_schema.py")):
    ROOT = os.path.join(_here, "..")
else:
    print("ERROR: Run from MABE repo root.")
    sys.exit(1)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_scorer():
    """Run the glycan scorer on all known lectin-sugar pairs."""
    print("=" * 70)
    print("PART 1: GLYCAN SCORER -- Lectin binding predictions")
    print("=" * 70)
    print()

    from glycan.scorer import GlycanScorer

    s = GlycanScorer()
    preds = s.score_all()

    # Header
    print(f"{'Scaffold':<10} {'Ligand':<16} {'Pred':>7} {'Obs':>7} "
          f"{'Resid':>7}  {'HB':>6} {'CH-pi':>6} {'Desolv':>6} {'Conf':>6}  Conf.")
    print("-" * 100)

    for p in preds:
        obs = f"{p.dG_obs:7.2f}" if p.dG_obs is not None else "   NB  "
        res = f"{p.residual:+6.2f}" if p.residual is not None else "   -- "
        print(f"{p.scaffold:<10} {p.ligand:<16} {p.dG_pred:7.2f} {obs} "
              f"{res}  {p.dG_HB:6.2f} {p.dG_CHP:6.2f} {p.dG_desolv:6.2f} "
              f"{p.dG_conf:6.2f}  [{p.confidence}]")

    # Stats
    valid = [p for p in preds if p.dG_obs is not None and p.residual is not None]
    stats = s.compute_r2(preds)
    stats_high = s.compute_r2(preds, confidence_filter=["HIGH"])

    print()
    print(f"  ALL:  R2 = {stats['r2']:.4f}  MAE = {stats['mae']:.2f} kJ/mol  "
          f"RMSE = {stats['rmse']:.2f}  n = {stats['n']}")
    print(f"  HIGH: R2 = {stats_high['r2']:.4f}  MAE = {stats_high['mae']:.2f} kJ/mol  "
          f"RMSE = {stats_high['rmse']:.2f}  n = {stats_high['n']}")
    print()

    # Selectivity checks
    print("  Selectivity checks:")
    for sc, a, b in [("ConA", "Man", "Glc"), ("Davis", "Glc", "Gal"),
                      ("Davis", "Glc", "Man"), ("DGL", "Man", "Glc")]:
        pa = s.score(sc, a)
        pb = s.score(sc, b)
        ok = "PASS" if pa.dG_pred < pb.dG_pred else "FAIL"
        print(f"    {sc} {a} > {b}: ddG = {pa.dG_pred - pb.dG_pred:+.2f} kJ/mol  [{ok}]")
    print()


def run_denovo(target="Glc", max_candidates=300):
    """Run de novo binder generation for a target sugar."""
    print("=" * 70)
    print(f"PART 2: DE NOVO BINDER DESIGN -- Target: {target}")
    print("=" * 70)
    print()

    from glycan.de_novo_binder import (
        generate_glycan_binders, GlycanBinderSpec,
        score_glycan_binder, selectivity_score,
        compute_receptor_descriptor, get_sugar_target,
    )

    # Show target info
    sugar = get_sugar_target(target)
    print(f"  Target: {sugar.name} ({sugar.ring_type})")
    print(f"  Eq OH: {sugar.n_eq_OH}  Ax OH: {sugar.n_ax_OH}  "
          f"C6 OH: {sugar.n_primary_OH}  NAc: {sugar.n_NAc}")
    print(f"  Axial CH (CH-pi donors): {sugar.n_axial_CH}")
    print(f"  cis-1,2-diol: {sugar.has_cis_12_diol}  Vol: {sugar.molecular_volume_A3} A3")
    print()

    # Generate
    t0 = time.time()
    spec = GlycanBinderSpec(target=target, max_candidates=max_candidates)
    result = generate_glycan_binders(spec=spec)
    elapsed = time.time() - t0

    print(f"  Generated: {result.n_scored} candidates in {elapsed:.1f}s")
    print(f"  Pareto front: {result.n_pareto_front}")
    print()

    # Top 15
    print(f"  {'#':<3} {'Front':<6} {'dG':>6} {'Sel':>6} {'SA':>5}  "
          f"{'CH-pi':>6} {'HB':>6} {'Desolv':>6} {'Boron':>6}  Backbone / Arms")
    print("  " + "-" * 95)

    for i, c in enumerate(result.candidates[:15]):
        front_tag = "P" if c.pareto_rank == 0 else str(c.pareto_rank)
        arms_str = ", ".join(c.arm_names)
        print(f"  {i+1:<3} {front_tag:<6} {c.score.dG_total:>6.1f} "
              f"{c.selectivity_ddG:>6.1f} {c.sa_score:>5.1f}  "
              f"{c.score.dG_chpi:>6.1f} {c.score.dG_hb:>6.1f} "
              f"{c.score.dG_desolv:>6.1f} {c.score.dG_boronic:>6.1f}  "
              f"{c.backbone_name} / {arms_str}")
    print()

    # Score known binders against this target for comparison
    KNOWN = {
        "Davis anth.+diurea": "NC(=O)Nc1ccc2cc3ccc(NC(N)=O)cc3cc2c1",
        "Ke anthracene": "c1ccc2cc3ccccc3cc2c1",
        "Phenylboronic acid": "OB(O)c1ccccc1",
        "Diboronic biphenyl": "OB(O)c1ccc(-c2ccc(B(O)O)cc2)cc1",
        "Anth.+boronic": "OB(O)c1ccc2cc3ccccc3cc2c1",
        "Pyridine+diurea": "NC(=O)Nc1cccc(NC(N)=O)n1",
        "Indole+amide": "NC(=O)c1c[nH]c2ccccc12",
        "Catechol+amine": "NCc1ccc(O)c(O)c1",
    }

    print(f"  Known binder comparison (scoring against {target}):")
    print(f"  {'Receptor':<24} {'dG':>6} {'CH-pi':>6} {'HB':>6} "
          f"{'Desolv':>6} {'Boron':>6} {'Shape':>6}")
    print("  " + "-" * 80)

    for name, smi in KNOWN.items():
        desc = compute_receptor_descriptor(smi)
        bs = score_glycan_binder(smi, target, desc)
        print(f"  {name:<24} {bs.dG_total:>6.1f} {bs.dG_chpi:>6.1f} "
              f"{bs.dG_hb:>6.1f} {bs.dG_desolv:>6.1f} "
              f"{bs.dG_boronic:>6.1f} {bs.dG_shape:>6.1f}")

    # Selectivity of best candidate
    if result.best:
        print()
        print(f"  Best candidate selectivity (dG_target - dG_competitor):")
        best = result.best
        for comp, ddg in sorted(best.selectivity_map.items(), key=lambda x: x[1]):
            marker = " <-- selective" if ddg < -2.0 else ""
            print(f"    vs {comp:<8} ddG = {ddg:+.2f} kJ/mol{marker}")
    print()


def run_custom_smiles(smiles, target="Glc"):
    """Score a custom SMILES against a target sugar."""
    print("=" * 70)
    print(f"PART 3: CUSTOM SMILES SCORING")
    print("=" * 70)
    print()

    from glycan.de_novo_binder import (
        score_glycan_binder, selectivity_score,
        compute_receptor_descriptor, list_sugar_targets,
    )

    desc = compute_receptor_descriptor(smiles)
    if not desc.valid:
        print(f"  ERROR: Invalid SMILES: {smiles}")
        return

    print(f"  SMILES: {smiles}")
    print(f"  MW: {desc.molecular_weight:.1f}  Aromatic rings: {desc.n_aromatic_rings}  "
          f"Large aromatic: {desc.n_large_aromatic}")
    print(f"  HB donors: {desc.n_hb_donors}  Urea groups: {desc.n_urea_groups}  "
          f"Boronic acid: {desc.n_boronic_acid}")
    print(f"  Rotatable bonds: {desc.n_rotatable_bonds}  SA score: {desc.sa_score:.2f}")
    print(f"  Est. cavity: {desc.estimated_cavity_A3:.0f} A3")
    print()

    # Score against all sugars
    print(f"  {'Sugar':<10} {'dG':>7} {'CH-pi':>7} {'HB':>7} "
          f"{'Desolv':>7} {'Boron':>7} {'Shape':>7}")
    print("  " + "-" * 60)

    all_scores = {}
    for t in list_sugar_targets():
        bs = score_glycan_binder(smiles, t, desc)
        all_scores[t] = bs.dG_total
        print(f"  {t:<10} {bs.dG_total:>7.2f} {bs.dG_chpi:>7.2f} "
              f"{bs.dG_hb:>7.2f} {bs.dG_desolv:>7.2f} "
              f"{bs.dG_boronic:>7.2f} {bs.dG_shape:>7.2f}")

    best = min(all_scores, key=all_scores.get)
    print(f"\n  Best target: {best} (dG = {all_scores[best]:.2f} kJ/mol)")

    # Selectivity for best target
    sel, sel_map = selectivity_score(smiles, best, descriptor=desc)
    print(f"  Selectivity for {best}:")
    for comp, ddg in sorted(sel_map.items(), key=lambda x: x[1]):
        marker = " <-- selective" if ddg < -2.0 else ""
        print(f"    vs {comp:<8} ddG = {ddg:+.2f}{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MABE Glycan Trial Bindings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python trial_bindings.py                          # Full run: scorer + Glc de novo
  python trial_bindings.py --target GalNAc          # Design binders for Tn antigen
  python trial_bindings.py --target Fru             # Design binders for fructose
  python trial_bindings.py --score-only             # Just run lectin scorer
  python trial_bindings.py --candidates 500         # More candidates (slower)
  python trial_bindings.py --smiles "OB(O)c1ccccc1" # Score a custom molecule
  python trial_bindings.py --smiles "c1ccc2cc3ccccc3cc2c1" --target Man
""")
    parser.add_argument("--target", default="Glc",
                        choices=["Glc", "Gal", "Man", "GalNAc", "GlcNAc",
                                 "Fuc", "Fru", "Neu5Ac"],
                        help="Sugar target for de novo design (default: Glc)")
    parser.add_argument("--candidates", type=int, default=300,
                        help="Max de novo candidates (default: 300)")
    parser.add_argument("--score-only", action="store_true",
                        help="Only run lectin scorer, skip de novo")
    parser.add_argument("--smiles", type=str, default=None,
                        help="Score a custom SMILES against all sugars")
    parser.add_argument("--skip-scorer", action="store_true",
                        help="Skip lectin scorer, run de novo only")

    args = parser.parse_args()

    print()
    print("  MABE Glycan Trial Bindings")
    print("  MAAD Scientist Technologies Inc.")
    print()

    if not args.skip_scorer:
        run_scorer()

    if args.smiles:
        run_custom_smiles(args.smiles, args.target)

    if not args.score_only:
        try:
            from rdkit import Chem
        except ImportError:
            print("ERROR: RDKit required for de novo generation.")
            print("  Install: pip install rdkit")
            print("  (scorer results above are still valid)")
            sys.exit(1)
        run_denovo(args.target, args.candidates)


if __name__ == "__main__":
    main()