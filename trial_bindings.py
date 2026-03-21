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
# PART 10: PFAS-free coating design
# ======================================================================

def run_coating(substrate="polyester", color="blue", oil=True, breathable=True):
    print("=" * 70)
    print(f"PART 10: PFAS-FREE COATING DESIGN")
    print("=" * 70)
    print()

    from core.pfas_free_coating import design_coating, CoatingTargets, SUBSTRATES

    sub = SUBSTRATES.get(substrate)
    if sub is None:
        print(f"  Unknown substrate '{substrate}'. Available: {', '.join(SUBSTRATES.keys())}")
        return

    tgt = CoatingTargets(
        water_contact_angle=150,
        oil_contact_angle=130 if oil else 0,
        color=color if color else "",
        wvtr=10000 if breathable else 0,
        durable=True,
    )

    d = design_coating(sub, tgt)
    print(d.summary)
    print()
    for k, (t, a, ok) in d.target_checks.items():
        tag = "PASS" if ok else "FAIL"
        if isinstance(a, float):
            print(f"  {k}: target={t} actual={a:.1f} [{tag}]")
        else:
            print(f"  {k}: [{tag}]")
    print()


# ======================================================================
# PART 11: Smart window design
# ======================================================================

def run_window(n_panes=2, iridescent=True):
    print("=" * 70)
    print(f"PART 11: SMART WINDOW DESIGN")
    print("=" * 70)
    print()

    from core.smart_window import design_window, WindowTargets

    tgt = WindowTargets(
        iridescent=iridescent,
        n_panes=n_panes,
        u_value_target=2.0 if n_panes == 2 else 1.0,
    )
    d = design_window(tgt)
    print(d.summary)
    print()
    for k, (t, a, ok) in d.target_checks.items():
        tag = "PASS" if ok else "FAIL"
        if isinstance(a, float):
            print(f"  {k}: target={t} actual={a:.3f} [{tag}]")
        else:
            print(f"  {k}: [{tag}]")
    print()


# ======================================================================
# PART 12: Switchable window
# ======================================================================

def run_switchable(mechanism="electrochromic"):
    print("=" * 70)
    print(f"PART 12: SWITCHABLE WINDOW -- {mechanism}")
    print("=" * 70)
    print()

    from core.switchable_window import design_switchable_window, SwitchableTargets

    tgt = SwitchableTargets(switching_mechanism=mechanism)
    d = design_switchable_window(tgt)
    print(d.summary)
    print()
    for k, (t, a, ok) in d.target_checks.items():
        tag = "PASS" if ok else "FAIL"
        if isinstance(a, float):
            print(f"  {k}: target={t} actual={a:.3f} [{tag}]")
        else:
            print(f"  {k}: [{tag}]")
    print()


# ======================================================================
# PART 13: Oriented window
# ======================================================================

def run_oriented(orientation="interior", switchable=False):
    print("=" * 70)
    print(f"PART 13: ORIENTED WINDOW -- {orientation.upper()}")
    print("=" * 70)
    print()

    from core.window_orientation import (
        design_oriented_window, OrientedWindowTargets, Orientation,
    )

    orient_map = {
        "exterior": Orientation.EXTERIOR,
        "interior": Orientation.INTERIOR,
        "igu": Orientation.IGU_SURFACE_2,
        "dual": Orientation.DUAL,
    }
    orient = orient_map.get(orientation.lower(), Orientation.INTERIOR)

    tgt = OrientedWindowTargets(orientation=orient, switchable=switchable)
    d = design_oriented_window(tgt)
    print(d.summary)
    print()
    for k, (t, a, ok) in d.target_checks.items():
        tag = "PASS" if ok else "FAIL"
        if isinstance(a, float):
            print(f"  {k}: target={t} actual={a:.3f} [{tag}]")
        else:
            print(f"  {k}: [{tag}]")
    print()


# ======================================================================
# PART 14: Self-assembly material design
# ======================================================================

def run_material(monomer_name=None):
    print("=" * 70)
    print(f"PART 14: SELF-ASSEMBLY MATERIAL DESIGN")
    print("=" * 70)
    print()

    from core.assembly_engine import (
        design_material, urea_tape_monomer, tripodal_linker_monomer,
        mof_paddle_wheel, pi_stacking_monomer,
    )

    library = {
        "urea-tape": urea_tape_monomer,
        "tripodal": tripodal_linker_monomer,
        "paddlewheel": mof_paddle_wheel,
        "pyrene-stack": pi_stacking_monomer,
    }

    if monomer_name and monomer_name in library:
        monomers = [(monomer_name, library[monomer_name]())]
    else:
        monomers = [(n, f()) for n, f in library.items()]

    for name, mono in monomers:
        d = design_material(mono)
        print(d.summary)
        if d.stacking_scores:
            print(f"  Stacking interactions:")
            for s in d.stacking_scores[:4]:
                print(f"    {s.face_a} <-> {s.face_b}: dG={s.dG_net:+.1f} kJ/mol "
                      f"({s.interaction.value})")
        print()


# ======================================================================
# PART 15: Scaffold-to-monomer catalog
# ======================================================================

def run_catalog():
    print("=" * 70)
    print(f"PART 15: SCAFFOLD -> MATERIAL CATALOG")
    print("=" * 70)
    print()

    from core.scaffold_to_monomer import print_material_catalog
    print_material_catalog()
    print()


# ======================================================================
# PART 16: Photonic nanoparticle design
# ======================================================================

def run_photonic(color="blue", material="SiO2", ordered=True):
    print("=" * 70)
    print(f"PART 16: PHOTONIC NANOPARTICLE DESIGN -- {color} {'opal' if ordered else 'glass'}")
    print("=" * 70)
    print()

    from core.photonic_assembly import design_photonic_particles

    d = design_photonic_particles(color, material, ordered)
    print(d.summary)
    print()

    if not ordered:
        print("  Comparison: ordered vs disordered for same color:")
        d_ord = design_photonic_particles(color, material, ordered=True)
        print(f"    Ordered:    d={d_ord.diameter_nm:.0f}nm lambda={d_ord.peak_wavelength_nm:.0f}nm "
              f"angle-dependent")
        print(f"    Disordered: d={d.diameter_nm:.0f}nm lambda={d.peak_wavelength_nm:.0f}nm "
              f"angle-independent")
        print()

    # All colors for this material
    print(f"  Color palette ({material}, {'ordered' if ordered else 'glass'}):")
    from core.photonic_assembly import _COLOR_WAVELENGTHS
    for c in ["violet", "blue", "green", "yellow", "orange", "red"]:
        dc = design_photonic_particles(c, material, ordered)
        print(f"    {c:8s}: d={dc.diameter_nm:5.0f}nm -> {dc.peak_wavelength_nm:.0f}nm")
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MABE -- Modality-Agnostic Binding Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  === Glycan / Molecular Binding ===
  python trial_bindings.py                              # Scorer + Glc de novo
  python trial_bindings.py --target GalNAc              # Tn antigen binders
  python trial_bindings.py --target Fru --construct     # Fructose + construct
  python trial_bindings.py --smiles "OB(O)c1ccccc1"    # Score custom SMILES
  python trial_bindings.py --props "OB(O)c1ccccc1"     # Full property report
  python trial_bindings.py --demand                     # Demand vectors
  python trial_bindings.py --compare                    # Fixed vs grammar
  python trial_bindings.py --metal Cu2+                 # Metal chelator
  python trial_bindings.py --host beta-CD               # Host-guest

  === Coatings ===
  python trial_bindings.py --coating                    # PFAS-free (polyester, blue)
  python trial_bindings.py --coating --substrate glass --color green
  python trial_bindings.py --coating --substrate steel --no-color --no-breathe

  === Windows ===
  python trial_bindings.py --window                     # Iridescent double-pane
  python trial_bindings.py --window --panes 3           # Triple-pane
  python trial_bindings.py --switchable                 # Electrochromic
  python trial_bindings.py --switchable --mechanism photochromic
  python trial_bindings.py --oriented interior          # Interior retrofit
  python trial_bindings.py --oriented dual --switchable # Both sides + switching

  === Materials ===
  python trial_bindings.py --material                   # All library monomers
  python trial_bindings.py --material paddlewheel       # MOF node
  python trial_bindings.py --catalog                    # Full scaffold catalog

  === Photonics ===
  python trial_bindings.py --photonic blue              # Blue FCC opal
  python trial_bindings.py --photonic red --glass       # Red photonic glass
  python trial_bindings.py --photonic green --particle TiO2
""")

    # Glycan args
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
                        help="Full construct assembly")
    parser.add_argument("--demand", action="store_true",
                        help="Show physics demand vectors")
    parser.add_argument("--compare", action="store_true",
                        help="Fixed library vs grammar comparison")
    parser.add_argument("--metal", type=str, default=None,
                        help="Design metal chelator (e.g., Cu2+, Fe3+)")
    parser.add_argument("--host", type=str, default=None,
                        help="Design guest for host (e.g., beta-CD)")

    # Coating args
    parser.add_argument("--coating", action="store_true",
                        help="PFAS-free coating design")
    parser.add_argument("--substrate", type=str, default="polyester",
                        help="Substrate material (cotton, polyester, glass, steel, etc.)")
    parser.add_argument("--color", type=str, default="blue",
                        help="Structural color for coating")
    parser.add_argument("--no-color", action="store_true",
                        help="No structural color")
    parser.add_argument("--no-oil", action="store_true",
                        help="Skip oil repulsion")
    parser.add_argument("--no-breathe", action="store_true",
                        help="Non-breathable coating")

    # Window args
    parser.add_argument("--window", action="store_true",
                        help="Smart window design")
    parser.add_argument("--panes", type=int, default=2,
                        help="Number of panes (2 or 3)")
    parser.add_argument("--no-iridescent", action="store_true",
                        help="Disable iridescent layer")
    parser.add_argument("--switchable", action="store_true",
                        help="Switchable opacity window")
    parser.add_argument("--mechanism", type=str, default="electrochromic",
                        choices=["electrochromic", "photochromic", "magnetochromic"],
                        help="Switching mechanism")
    parser.add_argument("--oriented", type=str, default=None,
                        choices=["exterior", "interior", "igu", "dual"],
                        help="Window orientation")

    # Material args
    parser.add_argument("--material", nargs="?", const="all", default=None,
                        help="Self-assembly material design (name or 'all')")
    parser.add_argument("--catalog", action="store_true",
                        help="Print full scaffold -> material catalog")

    # Photonic args
    parser.add_argument("--photonic", type=str, default=None,
                        help="Photonic nanoparticle design (color name)")
    parser.add_argument("--particle", type=str, default="SiO2",
                        help="Particle material (SiO2, polystyrene, TiO2, etc.)")
    parser.add_argument("--glass", action="store_true",
                        help="Photonic glass (disordered) instead of opal")

    args = parser.parse_args()

    # Detect which mode
    any_new_mode = (args.coating or args.window or args.switchable
                    or args.oriented or args.material or args.catalog
                    or args.photonic)

    print()
    print("  MABE -- Modality-Agnostic Binding Engine")
    print("  MAAD Scientist Technologies Inc.")
    print()

    # ── Glycan / Molecular modes ──
    if not any_new_mode:
        if not args.skip_scorer and not args.metal and not args.host:
            if not args.demand and not args.props:
                run_scorer()

        if args.smiles:
            run_custom_smiles(args.smiles, args.target)

        if args.props:
            run_properties(args.props)

        if args.demand:
            run_demand()

        if args.compare:
            run_compare(args.target, args.candidates)

        if args.metal:
            run_metal(args.metal)

        if args.host:
            run_host(args.host)

        if not args.score_only and not args.metal and not args.host and not args.demand and not args.props:
            try:
                from rdkit import Chem
            except ImportError:
                print("ERROR: RDKit required for de novo generation.")
                sys.exit(1)
            run_denovo(args.target, args.candidates)

        if args.construct:
            run_construct(args.target, min(args.candidates, 100))

    # ── Coating mode ──
    if args.coating:
        c = "" if args.no_color else args.color
        run_coating(args.substrate, c, not args.no_oil, not args.no_breathe)

    # ── Window modes ──
    if args.window:
        run_window(args.panes, not args.no_iridescent)

    if args.switchable and not args.oriented:
        run_switchable(args.mechanism)

    if args.oriented:
        run_oriented(args.oriented, args.switchable)

    # ── Material modes ──
    if args.material:
        name = None if args.material == "all" else args.material
        run_material(name)

    if args.catalog:
        run_catalog()

    # ── Photonic mode ──
    if args.photonic:
        run_photonic(args.photonic, args.particle, not args.glass)


if __name__ == "__main__":
    main()
