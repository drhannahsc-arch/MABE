"""
core/generative_integration.py - Top-level generative_design() pipeline.
"""
from core.coordination_generator import generate_coordination_environments
from core.donor_enumerator import enumerate_donor_arrangements
from core.scaffold_matcher import match_scaffolds, assemble_generative_binders

def generative_design(target_identity, target_formula, charge=2,
    d_electrons=None, hsab_softness=None, ionic_radius_pm=None,
    hydrated_radius_nm=0.2, working_ph=7.0, working_temp_c=25.0,
    ionic_strength_mm=10.0, special_effects=None,
    max_coord_envs=4, max_donor_arrangements=4, max_scaffold_matches=3):
    all_assemblies = []
    coord_envs = generate_coordination_environments(
        target_identity, target_formula, charge, d_electrons,
        hsab_softness, ionic_radius_pm, special_effects, max_coord_envs)
    if not coord_envs: return []
    for env in coord_envs:
        arrangements = enumerate_donor_arrangements(env, working_ph, max_donor_arrangements)
        for arr in arrangements:
            matches = match_scaffolds(arr, env, working_ph, working_temp_c,
                                       ionic_strength_mm, hydrated_radius_nm, max_scaffold_matches)
            assemblies = assemble_generative_binders(env, arr, matches, target_identity)
            all_assemblies.extend(assemblies)
    for a in all_assemblies:
        a._rs = (a.hsab_match_score * 50 + a.lfse_stabilization_kj * 0.15
                 + a.geometry_preference_kj * 0.05 + a.scaffold_match_score * 15
                 + a.synthetic_feasibility * 10 + a.chelate_rings * 3 - a.cost_relative * 0.5)
    all_assemblies.sort(key=lambda a: a._rs, reverse=True)
    for a in all_assemblies:
        if hasattr(a, "_rs"): delattr(a, "_rs")
    return all_assemblies

def print_generative_results(assemblies, top_n=5):
    print(f"\n{'='*70}")
    print(f"  GENERATIVE COORDINATION ENGINE - {len(assemblies)} designs")
    print(f"{'='*70}")
    for i, a in enumerate(assemblies[:top_n]):
        print(f"\n  #{i+1}: {a.name}")
        print(f"  {'-'*60}")
        print(f"  Target:     {a.target_formula} ({a.geometry}, CN={a.coordination_number})")
        print(f"  Donors:     {', '.join(a.donor_groups)} ({a.donor_type})")
        print(f"  Denticity:  {a.effective_denticity}, Chelate rings: {a.chelate_rings}")
        print(f"  Scaffold:   {a.scaffold_type} ({a.scaffold_backbone})")
        print(f"  HSAB match: {a.hsab_match_score:.2f}")
        print(f"  LFSE:       {a.lfse_stabilization_kj:.0f} kJ/mol")
        print(f"  pH range:   {a.ph_working_range[0]:.1f}-{a.ph_working_range[1]:.1f}")
        print(f"  Feasibility:{a.synthetic_feasibility:.2f}")
        print(f"  Cost:       {a.cost_relative:.1f}x baseline")
        if a.confidence_reasoning:
            print(f"  Confidence:")
            for c in a.confidence_reasoning: print(f"    + {c}")
        if a.failure_modes:
            print(f"  Failure modes:")
            for f in a.failure_modes: print(f"    ! {f}")
    print(f"\n{'='*70}")
    if len(assemblies) > top_n: print(f"  ({len(assemblies) - top_n} more not shown)")
    print()


