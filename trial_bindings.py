#!/usr/bin/env python
"""
trial_bindings.py -- MABE glycan binder design: full modular stack.

Usage (from MABE root):
    python trial_bindings.py                                  # scorer + Glc de novo
    python trial_bindings.py --target GalNAc                  # different sugar
    python trial_bindings.py --score-only                     # lectin scorer only
    python trial_bindings.py --smiles "OB(O)c1ccccc1"         # score custom molecule
    python trial_bindings.py --construct                      # full construct assembly
    python trial_bindings.py --demand                         # show demand vectors
    python trial_bindings.py --compare                        # fixed vs grammar comparison
    python trial_bindings.py --metal Cu2+                     # metal chelator design
    python trial_bindings.py --host beta-CD                   # host-guest design

Requires bootstraps (in order):
    bootstrap_glycan_expansion.py
    bootstrap_glycan_denovo.py
    bootstrap_ring_grammar.py
    bootstrap_core_demand.py
    bootstrap_modular_stack.py

Requires: RDKit (pip install rdkit)
"""

import argparse
import sys
import os
import time

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


# ======================================================================
# PART 1: Glycan scorer
# ======================================================================

def run_scorer():
    print("=" * 70)
    print("PART 1: GLYCAN SCORER -- Lectin binding predictions")
    print("=" * 70)
    print()

    from glycan.scorer import GlycanScorer

    s = GlycanScorer()
    preds = s.score_all()

    print(f"{'Scaffold':<10} {'Ligand':<16} {'Pred':>7} {'Obs':>7} "
          f"{'Resid':>7}  {'HB':>6} {'CH-pi':>6} {'Desolv':>6} {'Conf':>6}  Conf.")
    print("-" * 100)

    for p in preds:
        obs = f"{p.dG_obs:7.2f}" if p.dG_obs is not None else "   NB  "
        res = f"{p.residual:+6.2f}" if p.residual is not None else "   -- "
        print(f"{p.scaffold:<10} {p.ligand:<16} {p.dG_pred:7.2f} {obs} "
              f"{res}  {p.dG_HB:6.2f} {p.dG_CHP:6.2f} {p.dG_desolv:6.2f} "
              f"{p.dG_conf:6.2f}  [{p.confidence}]")

    valid = [p for p in preds if p.dG_obs is not None and p.residual is not None]
    stats = s.compute_r2(preds)
    stats_high = s.compute_r2(preds, confidence_filter=["HIGH"])

    print()
    print(f"  ALL:  R2 = {stats['r2']:.4f}  MAE = {stats['mae']:.2f} kJ/mol  "
          f"RMSE = {stats['rmse']:.2f}  n = {stats['n']}")
    print(f"  HIGH: R2 = {stats_high['r2']:.4f}  MAE = {stats_high['mae']:.2f} kJ/mol  "
          f"RMSE = {stats_high['rmse']:.2f}  n = {stats_high['n']}")
    print()

    print("  Selectivity checks:")
    for sc, a, b in [("ConA", "Man", "Glc"), ("Davis", "Glc", "Gal"),
                      ("Davis", "Glc", "Man"), ("DGL", "Man", "Glc")]:
        pa = s.score(sc, a)
        pb = s.score(sc, b)
        ok = "PASS" if pa.dG_pred < pb.dG_pred else "FAIL"
        print(f"    {sc} {a} > {b}: ddG = {pa.dG_pred - pb.dG_pred:+.2f} kJ/mol  [{ok}]")
    print()


# ======================================================================
# PART 2: De novo binder design (grammar-driven)
# ======================================================================

def run_denovo(target="Glc", max_candidates=300):
    print("=" * 70)
    print(f"PART 2: DE NOVO BINDER DESIGN -- Target: {target}")
    print("=" * 70)
    print()

    from glycan.de_novo_binder import get_sugar_target
    from glycan.demand_grammar import compute_demand, generate_from_demand

    sugar = get_sugar_target(target)
    demand = compute_demand(target)

    print(f"  Target: {sugar.name} ({sugar.ring_type})")
    print(f"  Eq OH: {sugar.n_eq_OH}  Ax OH: {sugar.n_ax_OH}  "
          f"C6 OH: {sugar.n_primary_OH}  NAc: {sugar.n_NAc}")
    print(f"  Axial CH: {sugar.n_axial_CH}  cis-1,2-diol: {sugar.has_cis_12_diol}  "
          f"Vol: {sugar.molecular_volume_A3} A3")
    print()
    print(f"  Demand vector:")
    print(f"    Aromatic: >={demand.min_aromatic_atoms} atoms, "
          f"large={demand.prefer_large_aromatic}")
    print(f"    HBD strategy: {demand.hbd_strategy}")
    print(f"    Boronic: useful={demand.boronic_useful}, "
          f"preferred={demand.boronic_preferred}")
    print(f"    Selectivity axis: {demand.selectivity_axis}")
    print()

    t0 = time.time()
    result = generate_from_demand(target=target, max_candidates=max_candidates)
    elapsed = time.time() - t0

    print(f"  Generated: {result.n_scored} candidates in {elapsed:.1f}s")
    print(f"  Pareto front: {result.n_pareto_front}")
    print()

    print(f"  {'#':<3} {'Frt':<4} {'dG':>6} {'Sel':>6} {'SA':>5}  "
          f"{'CH-pi':>6} {'HB':>6} {'Desol':>6} {'Boron':>6}  Scaffold / Arms")
    print("  " + "-" * 90)

    for i, c in enumerate(result.candidates[:15]):
        ft = str(c.pareto_rank) if c.pareto_rank >= 0 else "-"
        print(f"  {i+1:<3} {ft:<4} {c.score.dG_total:>6.1f} "
              f"{c.selectivity_ddG:>6.1f} {c.sa_score:>5.1f}  "
              f"{c.score.dG_chpi:>6.1f} {c.score.dG_hb:>6.1f} "
              f"{c.score.dG_desolv:>6.1f} {c.score.dG_boronic:>6.1f}  "
              f"{c.backbone_name} / {', '.join(c.arm_names)}")
    print()

    # Known binder comparison
    KNOWN = {
        "Davis anth.+diurea": "NC(=O)Nc1ccc2cc3ccc(NC(N)=O)cc3cc2c1",
        "Ke anthracene": "c1ccc2cc3ccccc3cc2c1",
        "Phenylboronic acid": "OB(O)c1ccccc1",
        "Diboronic biphenyl": "OB(O)c1ccc(-c2ccc(B(O)O)cc2)cc1",
        "Anth.+boronic": "OB(O)c1ccc2cc3ccccc3cc2c1",
    }

    from glycan.de_novo_binder import score_glycan_binder, compute_receptor_descriptor

    print(f"  Known binder comparison (vs {target}):")
    print(f"  {'Receptor':<24} {'dG':>6} {'CH-pi':>6} {'HB':>6} "
          f"{'Desol':>6} {'Boron':>6} {'Shape':>6}")
    print("  " + "-" * 72)

    for name, smi in KNOWN.items():
        desc = compute_receptor_descriptor(smi)
        bs = score_glycan_binder(smi, target, desc)
        print(f"  {name:<24} {bs.dG_total:>6.1f} {bs.dG_chpi:>6.1f} "
              f"{bs.dG_hb:>6.1f} {bs.dG_desolv:>6.1f} "
              f"{bs.dG_boronic:>6.1f} {bs.dG_shape:>6.1f}")
    print()


# ======================================================================
# PART 3: Custom SMILES scoring
# ======================================================================

def run_custom_smiles(smiles, target="Glc"):
    print("=" * 70)
    print(f"PART 3: CUSTOM SMILES SCORING")
    print("=" * 70)
    print()

    from glycan.de_novo_binder import (
        score_glycan_binder, selectivity_score,
        compute_receptor_descriptor, list_sugar_targets,
    )
    from core.property_predictor import predict_properties

    desc = compute_receptor_descriptor(smiles)
    if not desc.valid:
        print(f"  ERROR: Invalid SMILES: {smiles}")
        return

    props = predict_properties(smiles)

    print(f"  SMILES: {smiles}")
    print(f"  MW: {props.molecular_weight:.1f}  logP: {props.logP:.2f}  "
          f"TPSA: {props.tpsa:.1f}  logS: {props.logS_esol:.2f} ({props.solubility_class})")
    print(f"  Aromatic rings: {desc.n_aromatic_rings}  Large: {desc.n_large_aromatic}  "
          f"HBD: {desc.n_hb_donors}  Urea: {desc.n_urea_groups}  "
          f"Boronic: {desc.n_boronic_acid}")
    print(f"  SA: {props.sa_score:.2f}  Lipinski: {'PASS' if props.lipinski_pass else 'FAIL'}  "
          f"Stable: {props.aqueous_stable}")

    if props.acidic_groups:
        print(f"  Acidic groups: {', '.join(props.acidic_groups)} (pKa~{props.strongest_acidic_pka})")
    if props.basic_groups:
        print(f"  Basic groups: {', '.join(props.basic_groups)} (pKa~{props.strongest_basic_pka})")
    print()

    # Synthesis route
    try:
        from core.reaction_assembler import validate_synthesis
        synth = validate_synthesis(smiles)
        if synth.reactions_used:
            print(f"  Synthesis: {synth.route_summary}")
            print(f"  Coverage: {synth.annotation_coverage:.0%}  "
                  f"Reliability: {synth.max_reliability}")
        print()
    except ImportError:
        pass

    # Score against all sugars
    print(f"  {'Sugar':<10} {'dG':>7} {'CH-pi':>7} {'HB':>7} "
          f"{'Desol':>7} {'Boron':>7} {'Shape':>7}")
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

    sel, sel_map = selectivity_score(smiles, best, descriptor=desc)
    print(f"  Selectivity for {best}:")
    for comp, ddg in sorted(sel_map.items(), key=lambda x: x[1]):
        marker = " <-- selective" if ddg < -2.0 else ""
        print(f"    vs {comp:<8} ddG = {ddg:+.2f}{marker}")
    print()


# ======================================================================
# PART 4: Demand vectors
# ======================================================================

def run_demand():
    print("=" * 70)
    print("PART 4: PHYSICS DEMAND VECTORS")
    print("=" * 70)
    print()

    from glycan.demand_grammar import compute_demand
    from glycan.de_novo_binder import list_sugar_targets

    print(f"  {'Sugar':<10} {'Arom>=':<7} {'Large':<6} {'HBD':>10} "
          f"{'Boronic':<10} {'Selectivity axis'}")
    print("  " + "-" * 75)

    for t in list_sugar_targets():
        d = compute_demand(t)
        print(f"  {t:<10} {d.min_aromatic_atoms:>5}  {str(d.prefer_large_aromatic):<6} "
              f"{d.hbd_strategy:>10} {str(d.boronic_useful):<10} {d.selectivity_axis}")
    print()


# ======================================================================
# PART 5: Fixed vs grammar comparison
# ======================================================================

def run_compare(target="Glc", max_candidates=200):
    print("=" * 70)
    print(f"PART 5: FIXED LIBRARY vs GRAMMAR -- Target: {target}")
    print("=" * 70)
    print()

    from glycan.demand_grammar import compare_approaches

    comp = compare_approaches(target, max_candidates=max_candidates)
    f = comp["fixed"]
    g = comp["grammar"]

    print(f"  Fixed library:")
    print(f"    Scored: {f['n_scored']}  Pareto: {f['n_pareto']}  "
          f"Best dG: {f['best_dG']:+.1f}  Time: {f['elapsed']:.1f}s")
    print(f"  Grammar-driven:")
    print(f"    Scored: {g['n_scored']}  Pareto: {g['n_pareto']}  "
          f"Best dG: {g['best_dG']:+.1f}  Time: {g['elapsed']:.1f}s")
    print(f"  Improvement: {f['best_dG'] - g['best_dG']:+.1f} kJ/mol")
    print()

    gr = comp["grammar_result"]
    print(f"  Top 5 grammar candidates:")
    for i, c in enumerate(gr.candidates[:5]):
        print(f"    {i+1}. dG={c.score.dG_total:+6.1f} sel={c.selectivity_ddG:+5.1f} "
              f"SA={c.sa_score:.1f}  {c.backbone_name} / {', '.join(c.arm_names)}")
    print()


# ======================================================================
# PART 6: Construct assembly
# ======================================================================

def run_construct(target="Glc", max_candidates=100):
    print("=" * 70)
    print(f"PART 6: CONSTRUCT ASSEMBLY -- Target: {target}")
    print("=" * 70)
    print()

    from glycan.demand_grammar import generate_from_demand
    from core.construct_assembler import (
        assemble_construct, ConstructSpec, suggest_construct, compatible_pairs,
    )
    from core.property_predictor import predict_properties
    from core.reaction_assembler import validate_synthesis

    # Generate best binder
    result = generate_from_demand(target=target, max_candidates=max_candidates)
    if not result.best:
        print("  No candidates generated.")
        return

    best = result.best
    print(f"  Best binder: dG={best.score.dG_total:+.1f} kJ/mol")
    print(f"    Scaffold: {best.backbone_name}")
    print(f"    Arms: {', '.join(best.arm_names)}")
    print(f"    SMILES: {best.smiles[:60]}")
    print()

    # Need attachment point for construct -- add [*] to binder
    # Use the binder SMILES with a dummy appended
    binder_smi = f"[*]{best.smiles}" if "[*]" not in best.smiles else best.smiles

    # Show compatible click+support pairs
    pairs = compatible_pairs()
    print(f"  Compatible click + support pairs:")
    for click, support, rxn in pairs:
        print(f"    {click:20s} + {support:20s} via {rxn}")
    print()

    # Assemble with each linker option
    linkers = ["none", "PEG2", "PEG4", "PEG8"]
    print(f"  Construct options (azide + DBCO-Fe3O4):")
    print(f"  {'Linker':<12} {'MW':>6} {'logP':>6} {'Sol':>13} "
          f"{'Synth cov':>10} {'Rxns':>5}  Compatible")
    print("  " + "-" * 70)

    for linker_name in linkers:
        spec = ConstructSpec(
            recognition_smiles=binder_smi,
            recognition_name=f"{target}-binder",
            linker=linker_name,
            click_handle="azide",
            support="DBCO-Fe3O4",
            target=target,
        )
        c = assemble_construct(spec)
        if c.soluble_valid:
            print(f"  {linker_name:<12} {c.molecular_weight:>6.0f} {c.logP:>6.1f} "
                  f"{c.solubility_class:>13} {c.synthesis_coverage:>9.0%} "
                  f"{c.n_reactions:>5}  {c.click_compatible}")
        else:
            print(f"  {linker_name:<12}  -- assembly failed: {c.errors}")

    # Detailed best construct
    print()
    spec = suggest_construct(binder_smi, target=target)
    c = assemble_construct(spec)
    if c.soluble_valid:
        print(f"  Recommended construct:")
        print(f"    Linker: {spec.linker}")
        print(f"    Click: {spec.click_handle} -> {spec.support}")
        print(f"    Reaction: {c.click_reaction}")
        print(f"    Magnetic: {c.support_magnetic}")
        print(f"    SMILES: {c.soluble_smiles[:80]}")
        print(f"    MW: {c.molecular_weight:.0f}  logP: {c.logP:.1f}  "
              f"Sol: {c.solubility_class}")
        print(f"    Synth route: {', '.join(c.reactions)}")
        print(f"    Readout: {', '.join(c.readout_options)}")
    print()


# ======================================================================
# PART 7: Metal chelator design (core demand generator)
# ======================================================================

def run_metal(metal="Cu2+", max_candidates=200, max_scored=30):
    print("=" * 70)
    print(f"PART 7: METAL CHELATOR DESIGN -- Target: {metal}")
    print("=" * 70)
    print()

    from core.demand_generator import generate_from_demand, metal_demand

    demand = metal_demand(metal)
    print(f"  Demand: HSAB={demand.required_hardness or 'borderline'} "
          f"donor={demand.required_donor_element or 'any'} "
          f"scaffolds={demand.scaffold_categories}")
    print(f"  Notes: {demand.notes}")
    print()

    result = generate_from_demand("metal", metal,
                                   max_candidates=max_candidates,
                                   max_scored=max_scored)
    print(f"  Enumerated: {result.n_enumerated}  Scored: {result.n_scored}  "
          f"Errors: {result.n_failed}")
    print()

    if result.candidates:
        print(f"  {'#':<3} {'logKa':>6} {'dG':>8} {'SA':>5}  Scaffold / Arms")
        print("  " + "-" * 65)
        for i, c in enumerate(result.candidates[:10]):
            print(f"  {i+1:<3} {c.log_Ka_pred:>6.1f} {c.dg_total_kj:>8.1f} "
                  f"{c.sa_score_val:>5.1f}  {c.backbone_name} / {', '.join(c.arm_names)}")
    else:
        print(f"  No scored candidates (errors: {result.errors[:2]})")
    print()


# ======================================================================
# PART 8: Host-guest design
# ======================================================================

def run_host(host="beta-CD", max_candidates=200, max_scored=30):
    print("=" * 70)
    print(f"PART 8: HOST-GUEST DESIGN -- Target host: {host}")
    print("=" * 70)
    print()

    from core.demand_generator import generate_from_demand, host_guest_demand

    demand = host_guest_demand(host)
    print(f"  Cavity: {demand.target_volume_A3:.0f} A3 (optimal guest)")
    print(f"  Max MW: {demand.max_mw:.0f}  Notes: {demand.notes}")
    print()

    result = generate_from_demand("host_guest", host,
                                   max_candidates=max_candidates,
                                   max_scored=max_scored)
    print(f"  Enumerated: {result.n_enumerated}  Scored: {result.n_scored}  "
          f"Errors: {result.n_failed}")
    print()

    if result.candidates:
        print(f"  {'#':<3} {'logKa':>6} {'dG':>8} {'SA':>5}  Scaffold / Arms")
        print("  " + "-" * 65)
        for i, c in enumerate(result.candidates[:10]):
            print(f"  {i+1:<3} {c.log_Ka_pred:>6.1f} {c.dg_total_kj:>8.1f} "
                  f"{c.sa_score_val:>5.1f}  {c.backbone_name} / {', '.join(c.arm_names)}")
    else:
        print(f"  No scored candidates (errors: {result.errors[:2]})")
    print()


# ======================================================================
# PART 9: Property report for custom SMILES
# ======================================================================

def run_properties(smiles):
    print("=" * 70)
    print(f"PART 9: PROPERTY REPORT")
    print("=" * 70)
    print()

    from core.property_predictor import predict_properties

    props = predict_properties(smiles)
    if not props.valid:
        print(f"  Invalid SMILES: {smiles}")
        return

    print(f"  SMILES: {smiles}")
    print()
    print(f"  Molecular weight:    {props.molecular_weight:.1f} Da")
    print(f"  Heavy atoms:         {props.heavy_atom_count}")
    print(f"  logP (Crippen):      {props.logP:.2f}")
    print(f"  TPSA:                {props.tpsa:.1f} A2")
    print(f"  H-bond donors:       {props.n_hbd}")
    print(f"  H-bond acceptors:    {props.n_hba}")
    print(f"  Rotatable bonds:     {props.n_rotatable}")
    print(f"  Rings:               {props.n_rings} ({props.n_aromatic_rings} aromatic)")
    print(f"  Fraction sp3:        {props.fraction_sp3:.2f}")
    print()
    print(f"  logS (ESOL):         {props.logS_esol:.2f}")
    print(f"  Solubility class:    {props.solubility_class}")
    print(f"  Est. solubility:     {props.solubility_mg_ml:.3f} mg/mL")
    print()
    print(f"  pKa (acidic):        {props.strongest_acidic_pka}  {props.acidic_groups}")
    print(f"  pKa (basic):         {props.strongest_basic_pka}  {props.basic_groups}")
    print()
    print(f"  Lipinski:            {'PASS' if props.lipinski_pass else 'FAIL'} "
          f"({props.lipinski_violations} violations)")
    print(f"  Veber:               {'PASS' if props.veber_pass else 'FAIL'}")
    print(f"  SA score:            {props.sa_score:.2f}")
    print(f"  Aqueous stable:      {props.aqueous_stable}")
    if props.hydrolyzable_groups:
        print(f"  Hydrolyzable:        {', '.join(props.hydrolyzable_groups)}")

    # Synthesis route
    try:
        from core.reaction_assembler import validate_synthesis
        synth = validate_synthesis(smiles)
        print()
        print(f"  Synthesis route:")
        print(f"    Reactions: {synth.route_summary}")
        print(f"    Coverage:  {synth.annotation_coverage:.0%}")
        print(f"    Reliability: {synth.max_reliability}")
        if synth.requires_catalyst:
            print(f"    Catalysts: {', '.join(synth.requires_catalyst)}")
    except ImportError:
        pass
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MABE Glycan Binder Design -- Full Modular Stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python trial_bindings.py                              # Scorer + Glc de novo
  python trial_bindings.py --target GalNAc              # Tn antigen binders
  python trial_bindings.py --target Fru --construct     # Fructose + full construct
  python trial_bindings.py --smiles "OB(O)c1ccccc1"    # Score phenylboronic acid
  python trial_bindings.py --props "OB(O)c1ccccc1"     # Full property report
  python trial_bindings.py --demand                     # Show all demand vectors
  python trial_bindings.py --compare                    # Fixed vs grammar
  python trial_bindings.py --metal Cu2+                 # Metal chelator design
  python trial_bindings.py --metal Fe3+                 # Iron chelator
  python trial_bindings.py --host beta-CD               # beta-CD guest design
  python trial_bindings.py --score-only                 # Lectin scorer only
  python trial_bindings.py --skip-scorer --target Man   # Skip scorer
""")
    parser.add_argument("--target", default="Glc",
                        choices=["Glc", "Gal", "Man", "GalNAc", "GlcNAc",
                                 "Fuc", "Fru", "Neu5Ac"])
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--skip-scorer", action="store_true")
    parser.add_argument("--smiles", type=str, default=None,
                        help="Score custom SMILES against all sugars")
    parser.add_argument("--props", type=str, default=None,
                        help="Full property + synthesis report for SMILES")
    parser.add_argument("--construct", action="store_true",
                        help="Full construct assembly (recognition+linker+click+bead)")
    parser.add_argument("--demand", action="store_true",
                        help="Show physics demand vectors for all sugars")
    parser.add_argument("--compare", action="store_true",
                        help="Fixed library vs grammar comparison")
    parser.add_argument("--metal", type=str, default=None,
                        help="Design metal chelator (e.g., Cu2+, Fe3+, Pb2+)")
    parser.add_argument("--host", type=str, default=None,
                        help="Design guest for host (e.g., beta-CD, alpha-CD)")

    args = parser.parse_args()

    print()
    print("  MABE Trial Bindings -- Full Modular Stack")
    print("  MAAD Scientist Technologies Inc.")
    print()

    # Scorer
    if not args.skip_scorer and not args.metal and not args.host:
        if not args.demand and not args.props:
            run_scorer()

    # Custom SMILES
    if args.smiles:
        run_custom_smiles(args.smiles, args.target)

    # Property report
    if args.props:
        run_properties(args.props)

    # Demand vectors
    if args.demand:
        run_demand()

    # Comparison
    if args.compare:
        run_compare(args.target, args.candidates)

    # Metal
    if args.metal:
        run_metal(args.metal)

    # Host-guest
    if args.host:
        run_host(args.host)

    # De novo (glycan)
    if not args.score_only and not args.metal and not args.host and not args.demand and not args.props:
        try:
            from rdkit import Chem
        except ImportError:
            print("ERROR: RDKit required for de novo generation.")
            print("  Install: pip install rdkit")
            sys.exit(1)
        run_denovo(args.target, args.candidates)

    # Construct assembly
    if args.construct:
        run_construct(args.target, min(args.candidates, 100))


if __name__ == "__main__":
    main()
