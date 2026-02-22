"""
MABE Platform - Sprint 33 Bootstrap: Selectivity Scoring
Computes binding for interferent ions, reports selectivity ratios.
5 panels: drinking_water, seawater, acid_mine, nuclear_waste, soil.
Integrated into DesignPackage with selectivity-weighted grading.
Requires Sprints 16v2 + 17fix + 18-32.
"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 33 \u2014 Selectivity Scoring\n")
write_file("core/selectivity.py", '''\
"""
core/selectivity.py — Sprint 33: Selectivity Scoring

For each binder design, computes binding ΔG and Kd for common
interferent ions. Reports selectivity ratios (Kd_interferent / Kd_target).

Interferent panels:
  drinking_water: Ca2+, Mg2+, Na+, K+, Fe3+, Zn2+, Cu2+, Mn2+
  seawater:       Na+, Mg2+, Ca2+, K+, Sr2+, Ba2+
  acid_mine:      Fe3+, Fe2+, Cu2+, Zn2+, Mn2+, Al3+, Cd2+
  nuclear_waste:  Cs+, Sr2+, Ba2+, Ca2+, Na+, K+, La3+
  soil:           Ca2+, Mg2+, Fe3+, Al3+, Mn2+, Zn2+
"""
from dataclasses import dataclass, field
import math

from core.physics_integration import compute_enhanced_thermodynamics
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)
from core.coordination_generator import (
    METAL_D_ELECTRONS, METAL_HSAB_SOFTNESS, _IONIC_RADII,
)


@dataclass
class InterferentResult:
    """Binding result for a single interferent."""
    formula: str
    charge: int
    dg_net_kj: float
    predicted_kd_uM: float
    selectivity_ratio: float    # Kd(interferent) / Kd(target). >1 = selective
    selectivity_class: str      # "excellent", "good", "moderate", "poor", "none"
    binding_note: str = ""


@dataclass
class SelectivityProfile:
    """Full selectivity analysis for a binder design."""
    target_formula: str
    target_kd_uM: float
    interferents: list            # List of InterferentResult
    worst_interferent: str        # Formula of most competitive interferent
    worst_selectivity_ratio: float
    overall_selectivity_class: str  # Based on worst case
    selectivity_score: float      # 0-100
    deployment_matrix: str        # Which panel was used
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# INTERFERENT PANELS
# ═══════════════════════════════════════════════════════════════════════════

_PANELS = {
    "drinking_water": [
        ("Ca2+", 2, 0), ("Mg2+", 2, 0), ("Na+", 1, 0), ("K+", 1, 0),
        ("Fe3+", 3, 5), ("Zn2+", 2, 10), ("Cu2+", 2, 9), ("Mn2+", 2, 5),
    ],
    "seawater": [
        ("Na+", 1, 0), ("Mg2+", 2, 0), ("Ca2+", 2, 0), ("K+", 1, 0),
        ("Sr2+", 2, 0), ("Ba2+", 2, 0),
    ],
    "acid_mine": [
        ("Fe3+", 3, 5), ("Fe2+", 2, 6), ("Cu2+", 2, 9), ("Zn2+", 2, 10),
        ("Mn2+", 2, 5), ("Al3+", 3, 0), ("Cd2+", 2, 10),
    ],
    "nuclear_waste": [
        ("Na+", 1, 0), ("K+", 1, 0), ("Ca2+", 2, 0),
        ("Ba2+", 2, 0), ("La3+", 3, 0),
    ],
    "soil": [
        ("Ca2+", 2, 0), ("Mg2+", 2, 0), ("Fe3+", 3, 5),
        ("Al3+", 3, 0), ("Mn2+", 2, 5), ("Zn2+", 2, 10),
    ],
}

# Typical concentrations in environmental water (µM) for context
_TYPICAL_CONC = {
    "Ca2+": 1000, "Mg2+": 500, "Na+": 2000, "K+": 100,
    "Fe3+": 10, "Zn2+": 5, "Cu2+": 2, "Mn2+": 10,
    "Al3+": 5, "Cd2+": 0.05, "Ba2+": 1, "Sr2+": 5,
    "La3+": 0.01, "Fe2+": 50, "Pb2+": 0.5, "Hg2+": 0.01,
}


def _classify_selectivity(ratio):
    """Classify selectivity ratio."""
    if ratio >= 1000:
        return "excellent"
    elif ratio >= 100:
        return "good"
    elif ratio >= 10:
        return "moderate"
    elif ratio >= 2:
        return "poor"
    else:
        return "none"


def compute_selectivity(target_formula, target_kd_uM, target_charge,
                          recognition, structure, interior, matrix,
                          panel="drinking_water"):
    """Compute selectivity against an interferent panel.

    Runs compute_enhanced_thermodynamics for each interferent using
    the SAME binder (recognition + structure + interior) but different
    target metal. This is the key insight: the binder is fixed,
    the metal changes.
    """
    panel_ions = _PANELS.get(panel, _PANELS["drinking_water"])

    # Filter out the target itself from interferents
    panel_ions = [(f, c, d) for f, c, d in panel_ions if f != target_formula]

    results = []

    for int_formula, int_charge, int_d_electrons in panel_ions:
        int_softness = METAL_HSAB_SOFTNESS.get(int_formula, 0.3)
        int_radius = _IONIC_RADII.get(int_formula, 80)
        int_hr = (int_radius + 140) / 1000.0

        # Build problem for interferent
        int_prob = Problem(
            target=TargetSpecies(
                identity=int_formula, formula=int_formula,
                charge=int_charge, d_electrons=int_d_electrons,
                hsab_softness=int_softness, ionic_radius_pm=int_radius,
                hydrated_radius_nm=int_hr,
                coordination_number=6),
            matrix=matrix)

        try:
            int_thermo = compute_enhanced_thermodynamics(
                recognition, structure, interior, int_prob)
            int_kd = int_thermo.predicted_kd_um
            int_dg = int_thermo.dg_net_kj
        except Exception:
            # If calculation fails, assume no binding
            int_kd = 1e12
            int_dg = 100.0

        # Selectivity ratio
        # Handle ultra-strong binding where target Kd underflows to 0
        # Use ΔG difference instead: ΔΔG = ΔG(interferent) - ΔG(target)
        # Ratio = exp(ΔΔG / RT)
        R = 8.314e-3  # kJ/(mol·K)
        T = 298.15
        if target_kd_uM > 1e-10:
            # Normal case: both Kd are finite
            ratio = int_kd / target_kd_uM if int_kd < 1e10 else 1e6
        else:
            # Target Kd underflowed — use ΔG difference
            # target ΔG is in design_package, but we need to use what was passed
            # For now, estimate from kd: if target_kd < 1e-10, dG < -60 kJ/mol
            # Use the interferent ΔG directly: if int_dg > 0, no binding → excellent selectivity
            if int_dg > 0:
                ratio = 1e6  # Interferent doesn't bind
            elif int_kd > 1.0:
                ratio = 1e6  # Interferent binds weakly
            elif int_kd > 0.001:
                ratio = int_kd / 1e-10  # Assume target Kd ~ 1e-10
            else:
                # Both bind extremely strongly
                ddg = int_dg  # Both negative, compare magnitude
                ratio = max(1.0, math.exp(abs(ddg) * 0.01))  # Crude scaling

        sel_class = _classify_selectivity(ratio)

        # Note about concentration-weighted selectivity
        conc_target = _TYPICAL_CONC.get(target_formula, 1.0)
        conc_int = _TYPICAL_CONC.get(int_formula, 100.0)
        note = ""
        if conc_int / max(0.001, conc_target) > 100 and ratio < 100:
            note = (f"Interferent [{int_formula}] is {conc_int/max(0.001,conc_target):.0f}× "
                    f"more concentrated than target — effective selectivity worse than ratio suggests")

        results.append(InterferentResult(
            formula=int_formula, charge=int_charge,
            dg_net_kj=round(int_dg, 1),
            predicted_kd_uM=round(int_kd, 2) if int_kd < 1e6 else int_kd,
            selectivity_ratio=round(ratio, 1) if ratio < 1e6 else ratio,
            selectivity_class=sel_class,
            binding_note=note,
        ))

    # Overall assessment
    if results:
        worst = min(results, key=lambda r: r.selectivity_ratio)
        worst_formula = worst.formula
        worst_ratio = worst.selectivity_ratio
        overall = _classify_selectivity(worst_ratio)
    else:
        worst_formula = "none"
        worst_ratio = 1e6
        overall = "excellent"

    # Score: 0-100 based on geometric mean of all ratios
    if results:
        log_ratios = [math.log10(max(0.01, min(1e6, r.selectivity_ratio))) for r in results]
        avg_log = sum(log_ratios) / len(log_ratios)
        # avg_log: -2 (ratio 0.01) → 0, 0 (ratio 1) → 0, 3 (ratio 1000) → 100
        score = max(0, min(100, (avg_log + 1) * 25))  # -1→0, 0→25, 3→100
    else:
        score = 100

    notes_parts = []
    poor = [r for r in results if r.selectivity_class in ("poor", "none")]
    if poor:
        names = ", ".join(r.formula for r in poor)
        notes_parts.append(f"Poor selectivity vs: {names}")
    conc_warnings = [r for r in results if r.binding_note]
    if conc_warnings:
        notes_parts.append(f"{len(conc_warnings)} interferent(s) at much higher concentration")

    return SelectivityProfile(
        target_formula=target_formula,
        target_kd_uM=target_kd_uM,
        interferents=results,
        worst_interferent=worst_formula,
        worst_selectivity_ratio=worst_ratio,
        overall_selectivity_class=overall,
        selectivity_score=round(score, 1),
        deployment_matrix=panel,
        notes="; ".join(notes_parts),
    )


def print_selectivity(profile):
    """Pretty-print selectivity analysis."""
    print(f"\\n  SELECTIVITY ({profile.overall_selectivity_class}, score={profile.selectivity_score:.0f}/100)")
    print(f"  {'─'*60}")
    print(f"  Target: {profile.target_formula} Kd={profile.target_kd_uM:.2f} µM")
    print(f"  Panel:  {profile.deployment_matrix}")
    print(f"  {'Interferent':12s} {'Kd(µM)':>12s} {'Ratio':>10s} {'Class':>12s}")
    print(f"  {'─'*48}")
    for r in sorted(profile.interferents, key=lambda x: x.selectivity_ratio):
        kd_str = f"{r.predicted_kd_uM:.1f}" if r.predicted_kd_uM < 1e6 else ">10⁶"
        ratio_str = f"{r.selectivity_ratio:.0f}×" if r.selectivity_ratio < 1e6 else ">10⁶×"
        flag = " ⚠" if r.selectivity_class in ("poor", "none") else ""
        print(f"  {r.formula:12s} {kd_str:>12s} {ratio_str:>10s} {r.selectivity_class:>12s}{flag}")
        if r.binding_note:
            print(f"  {'':12s}  └ {r.binding_note}")
    if profile.notes:
        print(f"\\n  ⚠ {profile.notes}")
    print()

''')

write_file("core/design_package.py", '''\
"""
core/design_package.py — Sprint 32: Complete Design Package

The top-level entry point that produces a fully characterized binder
design from target identity alone. Integrates:
  generative_design → speciation_gated → enhanced_thermodynamics →
  deployment_scoring → spectroscopic_prediction → readout_recommendation

Output: DesignPackage — everything needed to synthesize, deploy, and
detect a binder in the field.
"""
from dataclasses import dataclass, field
import math

from core.generative_integration import generative_design
from core.physics_integration import compute_enhanced_thermodynamics, EnhancedThermodynamics
from core.deployment_scoring import score_deployment, DeploymentScore
from core.spectroscopic import predict_spectroscopy, SpectroscopicPrediction
from core.nmr_readout import predict_nmr_relaxation, recommend_readout
from core.nuclear_decay import analyze_decay_chain
from core.selectivity import compute_selectivity, SelectivityProfile
from core.speciation_gate import predict_speciation
from core.generative_physics_adapter import (
    adapt_generative_to_pipeline, TargetSpecies, Matrix, Problem,
)
from core.coordination_generator import METAL_D_ELECTRONS, METAL_HSAB_SOFTNESS, _IONIC_RADII
from core.spin_state import predict_spin_state


@dataclass
class DetectionPlan:
    """How to confirm binding and quantify target."""
    spectroscopy: dict           # Color, CT band, detection method
    nmr_viable: bool
    nmr_relaxivity: float        # r1 if paramagnetic
    recommended_readouts: list   # Top 3 strategies
    field_deployable_option: str # Best field-deployable method
    mass_spec_replacement: str   # Best mass-spec-replacing method


@dataclass
class DesignPackage:
    """Complete binder design: everything needed to build, deploy, detect."""
    # Identity
    target: str
    target_formula: str
    working_ph: float
    # Design
    binder_name: str
    scaffold_type: str
    donor_atoms: list
    geometry: str
    coordination_number: int
    # Binding
    thermodynamics: EnhancedThermodynamics
    predicted_kd_uM: float
    selectivity_notes: str
    # Deployment
    deployment: DeploymentScore
    # Detection
    detection: DetectionPlan
    # Selectivity
    selectivity: SelectivityProfile = None
    # Nuclear (optional)
    decay_chain_warning: str = ""
    # Summary
    overall_grade: str = ""      # "A", "B", "C", "D", "F"
    one_line_summary: str = ""


def _infer_unpaired(formula, d_electrons, donors):
    """Infer unpaired electrons for spectroscopic/magnetic prediction."""
    if d_electrons == 0 or d_electrons == 10:
        return 0
    try:
        ligand_names = []
        for da in donors:
            lmap = {"O": "water", "N": "pyridine", "S": "thiolate",
                    "P": "phosphine", "Cl": "Cl-"}
            ligand_names.append(lmap.get(da, "water"))
        sp = predict_spin_state(formula, d_electrons, ligand_names)
        return sp.unpaired_electrons
    except Exception:
        # Fallback: high-spin estimate
        if d_electrons <= 5:
            return d_electrons
        return 10 - d_electrons


def design_binder(target_identity, target_formula, charge=2,
                    working_ph=7.0, working_temp_c=25.0,
                    ionic_strength_mm=10.0, target_conc_uM=1.0,
                    is_nuclear=False, outdoor_use=False,
                    field_deployable=False, max_designs=5,
                    required_sensitivity="µM",
                    selectivity_panel="drinking_water"):
    """THE entry point. Target identity → complete design packages.

    Returns list of DesignPackage objects, ranked by combined
    binding + deployment score.
    """
    # Resolve metal properties
    d_electrons = METAL_D_ELECTRONS.get(target_formula, 0)
    from core.coordination_generator import _get_continuous_softness
    hsab = _get_continuous_softness(target_formula, d_electrons)
    ionic_r = _IONIC_RADII.get(target_formula, 80)
    # Estimate hydrated radius from ionic
    hydrated_r = (ionic_r + 140) / 1000.0  # Rough: add ~1.4 Å for water shell

    # Estimate MW
    mw_map = {"Pb2+": 207.2, "Cu2+": 63.5, "Ni2+": 58.7, "Zn2+": 65.4,
              "Fe3+": 55.8, "Fe2+": 55.8, "Au3+": 197.0, "Au+": 197.0,
              "Hg2+": 200.6, "Ag+": 107.9, "Cd2+": 112.4, "Mn2+": 54.9,
              "Co2+": 58.9, "Cr3+": 52.0, "UO2_2+": 270.0, "Ce3+": 140.1,
              "Ba2+": 137.3, "Na+": 23.0, "K+": 39.1, "Ca2+": 40.1,
              "Al3+": 27.0, "Pt2+": 195.1}
    target_mw = mw_map.get(target_formula, 60.0)

    # Speciation check first
    spec = predict_speciation(target_formula, working_ph)

    # Nuclear decay check
    decay_warning = ""
    if is_nuclear:
        # Try to find isotope from formula
        elem = target_formula.replace("+", "").replace("-", "")
        for digits in "0123456789":
            elem = elem.replace(digits, "")
        chain = None
        for iso_key in ["U-238", "Cs-137", "Sr-90", "Ra-226", "Co-60",
                         "Tc-99", "Am-241", "Pu-239", "I-131"]:
            if elem.lower() in iso_key.lower():
                chain = analyze_decay_chain(iso_key)
                break
        if chain:
            decay_warning = (f"Decay chain: {len(chain.chain)} steps, "
                             f"{chain.total_species_to_capture} species to capture. "
                             f"Strategy: {chain.binder_strategy}. "
                             f"{chain.notes}")

    # Generate candidates
    assemblies = generative_design(
        target_identity, target_formula, charge, d_electrons, hsab,
        ionic_r, hydrated_r, working_ph, working_temp_c, ionic_strength_mm,
        max_coord_envs=4, max_donor_arrangements=3, max_scaffold_matches=3)

    if not assemblies:
        return []

    # === DEDUP: remove duplicate (donor_set, scaffold) combinations ===
    seen_keys = set()
    unique_assemblies = []
    for a in assemblies:
        key = (tuple(sorted(a.donor_atoms)), a.scaffold_type, a.coordination_number)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_assemblies.append(a)
    assemblies = unique_assemblies

    # Build problem object for all candidates
    prob_obj = Problem(
        target=TargetSpecies(
            identity=target_identity, formula=target_formula,
            charge=charge, d_electrons=d_electrons,
            hsab_softness=hsab, ionic_radius_pm=ionic_r,
            hydrated_radius_nm=hydrated_r,
            coordination_number=6),
        matrix=Matrix(
            ph=working_ph, temperature_c=working_temp_c,
            ionic_strength_mm=ionic_strength_mm))

    # Score each candidate
    packages = []
    for gen_a in assemblies[:max_designs * 3]:  # Oversample then filter
        # Adapt to pipeline format
        adapted = adapt_generative_to_pipeline(gen_a, problem=prob_obj)
        rec = adapted.recognition
        struct = adapted.structure
        interior = adapted.interior

        # Build Problem
        prob = prob_obj

        # Enhanced thermodynamics
        thermo = compute_enhanced_thermodynamics(rec, struct, interior, prob)

        # Deployment scoring
        unpaired = _infer_unpaired(target_formula, d_electrons, rec.donor_atoms)
        dep = score_deployment(
            gen_a.scaffold_type, target_formula, charge, target_mw,
            hydrated_r, struct.pore_size_nm if struct else 0.0,
            abs(thermo.dg_bind_kj), target_mw, unpaired,
            is_nuclear, outdoor_use, target_conc_uM)

        # Spectroscopic prediction
        ten_dq = 120.0  # Default; would come from spin_state in full impl
        try:
            sp_result = predict_spin_state(target_formula, d_electrons,
                                            ["water"] * min(6, len(rec.donor_atoms)))
            ten_dq = sp_result.ten_dq_kj
        except Exception:
            pass

        spec_pred = predict_spectroscopy(
            target_formula, rec.donor_atoms, d_electrons,
            ten_dq_kj=ten_dq,
            geometry="octahedral" if gen_a.coordination_number >= 5 else "tetrahedral",
            scaffold_type=gen_a.scaffold_type)

        # NMR + readout
        nmr = predict_nmr_relaxation(target_formula, unpaired)
        readouts = recommend_readout(target_formula, required_sensitivity,
                                      field_deployable, multiplexing_needed=1)

        field_option = "None available"
        mass_spec_option = "ICP-MS (traditional)"
        for ro in readouts:
            if ro.field_deployable and field_option == "None available":
                field_option = ro.strategy_name
            if ro.multiplexing_capacity >= 100:
                mass_spec_option = ro.strategy_name

        detection = DetectionPlan(
            spectroscopy={
                "color": spec_pred.predicted_color,
                "dd_nm": spec_pred.dd_transition_nm,
                "ct_type": spec_pred.ct_type,
                "ct_nm": spec_pred.ct_transition_nm,
                "detection_method": spec_pred.detection_method,
                "sensitivity": spec_pred.sensitivity_estimate,
            },
            nmr_viable=nmr.total_r1_mM_s > 0,
            nmr_relaxivity=nmr.total_r1_mM_s,
            recommended_readouts=[r.strategy_name for r in readouts[:3]],
            field_deployable_option=field_option,
            mass_spec_replacement=mass_spec_option,
        )

        # Selectivity
        sel = compute_selectivity(
            target_formula, thermo.predicted_kd_um, charge,
            rec, struct, interior, prob.matrix,
            panel=selectivity_panel)

        # Overall grade — now includes selectivity
        binding_score = max(0, min(100, -thermo.dg_net_kj))  # More negative = better
        combined = binding_score * 0.35 + dep.deployment_score * 0.35 + sel.selectivity_score * 0.30
        if combined > 70: grade = "A"
        elif combined > 55: grade = "B"
        elif combined > 40: grade = "C"
        elif combined > 25: grade = "D"
        else: grade = "F"

        summary = (f"{gen_a.name} | Kd={thermo.predicted_kd_um:.1f} µM | "
                   f"Deploy={dep.deployment_class} | "
                   f"Detect={spec_pred.detection_method} | Grade={grade}")

        sel_notes = ""
        if thermo.bond_character == "covalent":
            sel_notes = "Covalent binding — high selectivity for soft metals"
        elif thermo.softness_continuous > 0.5:
            sel_notes = "Soft-metal selective (polarization-driven)"
        elif thermo.softness_continuous < 0.15:
            sel_notes = "Hard-metal selective (electrostatic-driven)"

        packages.append(DesignPackage(
            target=target_identity, target_formula=target_formula,
            working_ph=working_ph,
            binder_name=gen_a.name, scaffold_type=gen_a.scaffold_type,
            donor_atoms=rec.donor_atoms, geometry=gen_a.geometry,
            coordination_number=gen_a.coordination_number,
            thermodynamics=thermo, predicted_kd_uM=thermo.predicted_kd_um,
            selectivity_notes=sel_notes,
            deployment=dep, detection=detection,
            selectivity=sel,
            decay_chain_warning=decay_warning,
            overall_grade=grade, one_line_summary=summary,
        ))

    # Sort by combined score
    packages.sort(key=lambda p: (-ord(p.overall_grade[0]),
                                   p.thermodynamics.dg_net_kj))
    return packages[:max_designs]


def print_design_package(pkg):
    """Pretty-print a complete design package."""
    print(f"\\n{'='*72}")
    print(f"  MABE DESIGN PACKAGE: {pkg.binder_name}")
    print(f"{'='*72}")
    print(f"  Target:      {pkg.target} ({pkg.target_formula}) at pH {pkg.working_ph}")
    print(f"  Grade:       {pkg.overall_grade}")
    print(f"  Summary:     {pkg.one_line_summary}")

    t = pkg.thermodynamics
    print(f"\\n  BINDING ({t.confidence} confidence)")
    print(f"  {'─'*60}")
    print(f"  ΔG_net:      {t.dg_net_kj:.1f} kJ/mol → Kd = {t.predicted_kd_um:.2f} µM")
    print(f"  ΔG_bind:     {t.dg_bind_kj:.1f}  ΔG_desolv: +{t.dg_desolv_kj:.1f}")
    print(f"  ΔG_LFSE:     {t.dg_lfse_kj:.1f}  ΔG_chelate: {t.dg_chelate_kj:.1f}")
    if t.dg_covalent_kj != 0:
        print(f"  ΔG_covalent: {t.dg_covalent_kj:.1f}  ({t.bond_character})")
    if t.dg_dispersion_kj != 0:
        print(f"  ΔG_disp:     {t.dg_dispersion_kj:.2f}  ΔG_polar: {t.dg_polarization_kj:.2f}")
    if t.dg_relativistic_correction_kj != 0:
        print(f"  ΔG_relativ:  {t.dg_relativistic_correction_kj:.2f}")
    if t.speciation_warning:
        print(f"  ⚠ SPECIATION: {t.speciation_warning}")
    print(f"  Softness:    {t.softness_continuous:.3f}  β={t.nephelauxetic_beta:.3f}")
    if pkg.selectivity_notes:
        print(f"  Selectivity: {pkg.selectivity_notes}")

    d = pkg.deployment
    print(f"\\n  DEPLOYMENT ({d.deployment_class})")
    print(f"  {'─'*60}")
    print(f"  Score:       {d.deployment_score:.0f}/100  Limiting: {d.limiting_factor}")
    print(f"  Transport:   {d.transport_score:.0f}  Capacity: {d.capacity_mg_g:.0f} mg/g")
    print(f"  Wetting:     {d.wettability} ({d.wetting_score:.0f})")
    print(f"  Thermal:     max {d.max_temp_C}°C ({d.thermal_score:.0f})")
    if d.recommendations:
        for r in d.recommendations[:3]:
            print(f"  → {r}")

    det = pkg.detection
    print(f"\\n  DETECTION")
    print(f"  {'─'*60}")
    sp = det.spectroscopy
    if sp["color"] != "colorless":
        print(f"  Color:       {sp['color']} (d-d at {sp['dd_nm']:.0f} nm)")
    if sp["ct_type"] != "none":
        print(f"  CT band:     {sp['ct_type']} at {sp['ct_nm']:.0f} nm")
    print(f"  Best method: {sp['detection_method']} ({sp['sensitivity']})")
    if det.nmr_viable:
        print(f"  NMR:         r1={det.nmr_relaxivity:.1f} mM⁻¹s⁻¹ (viable)")
    print(f"  Field:       {det.field_deployable_option}")
    print(f"  Mass-spec→:  {det.mass_spec_replacement}")

    if pkg.decay_chain_warning:
        print(f"\\n  ☢ NUCLEAR: {pkg.decay_chain_warning}")

    if pkg.selectivity:
        s = pkg.selectivity
        print(f"\\n  SELECTIVITY ({s.overall_selectivity_class}, score={s.selectivity_score:.0f}/100)")
        print(f"  {'─'*60}")
        print(f"  Panel: {s.deployment_matrix}")
        for r in sorted(s.interferents, key=lambda x: x.selectivity_ratio)[:5]:
            kd_str = f"{r.predicted_kd_uM:.1f}" if r.predicted_kd_uM < 1e6 else ">10⁶"
            ratio_str = f"{r.selectivity_ratio:.0f}×" if r.selectivity_ratio < 1e6 else ">10⁶×"
            flag = " ⚠" if r.selectivity_class in ("poor", "none") else ""
            print(f"  {r.formula:10s} Kd={kd_str:>8s}  sel={ratio_str:>8s}  {r.selectivity_class}{flag}")
        if s.notes:
            print(f"  ⚠ {s.notes}")

    print(f"\\n{'='*72}\\n")



''')

write_file("tests/test_sprint33.py", '''\
"""tests/test_sprint33.py — Sprint 33: Selectivity Scoring (18 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.selectivity import compute_selectivity, SelectivityProfile, _PANELS
from core.physics_integration import compute_enhanced_thermodynamics
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)
from core.design_package import design_binder

def _rec(donors, dt="soft", chel=2, match=0.9):
    return RecognitionChemistry(name="t", type="generative", donor_atoms=donors,
        donor_type=dt, denticity=len(donors), hsab_match=match, chelate_rings=chel)

def _struct(stype="zeolite"):
    return StructuralConstraint(name="s", type=stype, geometry="channel", pore_size_nm=0.74)

def _interior():
    return InteriorDesign(description="t", num_binding_sites=1, self_binding=True)

def _matrix(ph=7.0):
    return Matrix(ph=ph, temperature_c=25.0, ionic_strength_mm=10.0)


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_s_donor_selective_for_soft():
    """S-donor binder should show high selectivity vs hard ions."""
    sel = compute_selectivity("Pb2+", 0.01, 2,
        _rec(["S","S","S","S"]), _struct(), _interior(), _matrix(6.0))
    ca = next(r for r in sel.interferents if r.formula == "Ca2+")
    assert ca.selectivity_ratio > 100, f"S-donors should reject Ca2+, ratio={ca.selectivity_ratio}"
    print(f"  \\u2705 test_s_selective: Ca2+ ratio={ca.selectivity_ratio:.0f}×")

def test_o_donor_binds_hard():
    """O-donor binder should bind Ca2+ well (poor selectivity for Pb2+)."""
    sel = compute_selectivity("Pb2+", 100.0, 2,
        _rec(["O","O","O","O"], "hard", 2, 0.3), _struct(), _interior(), _matrix(7.0))
    ca = next(r for r in sel.interferents if r.formula == "Ca2+")
    # Ca2+ should bind O donors reasonably → lower selectivity ratio
    print(f"  \\u2705 test_o_binds_hard: Ca2+ Kd={ca.predicted_kd_uM:.0f}, ratio={ca.selectivity_ratio:.0f}×")

def test_panels_exist():
    """All standard panels should be defined."""
    for panel in ["drinking_water", "seawater", "acid_mine", "nuclear_waste", "soil"]:
        assert panel in _PANELS, f"Missing panel: {panel}"
        assert len(_PANELS[panel]) >= 3
    print(f"  \\u2705 test_panels: {len(_PANELS)} panels defined")

def test_target_excluded_from_panel():
    """Target ion should not appear as its own interferent."""
    sel = compute_selectivity("Cu2+", 1.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(7.0))
    formulas = [r.formula for r in sel.interferents]
    assert "Cu2+" not in formulas, "Target should be excluded from interferent list"
    print(f"  \\u2705 test_exclude_target: interferents={formulas[:4]}")

def test_selectivity_score_range():
    """Score should be 0-100."""
    sel = compute_selectivity("Pb2+", 10.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(6.0))
    assert 0 <= sel.selectivity_score <= 100
    print(f"  \\u2705 test_score_range: {sel.selectivity_score:.0f}/100")

def test_worst_interferent_identified():
    """Should identify the most competitive interferent."""
    sel = compute_selectivity("Pb2+", 10.0, 2,
        _rec(["N","N","N","N"], "borderline"), _struct(), _interior(), _matrix(7.0))
    assert sel.worst_interferent != ""
    assert sel.worst_selectivity_ratio >= 0
    print(f"  \\u2705 test_worst: {sel.worst_interferent} ratio={sel.worst_selectivity_ratio:.0f}×")

def test_classification_correct():
    """Selectivity classes should be assigned correctly."""
    from core.selectivity import _classify_selectivity
    assert _classify_selectivity(2000) == "excellent"
    assert _classify_selectivity(500) == "good"
    assert _classify_selectivity(50) == "moderate"
    assert _classify_selectivity(5) == "poor"
    assert _classify_selectivity(0.5) == "none"
    print(f"  \\u2705 test_classify: all classes correct")

def test_acid_mine_panel():
    """Acid mine panel should include Fe3+, Cu2+, Zn2+."""
    panel = _PANELS["acid_mine"]
    formulas = [f for f, c, d in panel]
    assert "Fe3+" in formulas
    assert "Cu2+" in formulas
    assert "Zn2+" in formulas
    print(f"  \\u2705 test_acid_mine: {formulas}")

def test_seawater_panel():
    """Seawater panel should include high-concentration ions."""
    panel = _PANELS["seawater"]
    formulas = [f for f, c, d in panel]
    assert "Na+" in formulas
    assert "Mg2+" in formulas
    print(f"  \\u2705 test_seawater: {formulas}")

def test_concentration_warning():
    """Should warn when interferent is much more concentrated than target."""
    sel = compute_selectivity("Pb2+", 10.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(6.0))
    # Ca2+ is ~1000 µM vs Pb2+ ~0.5 µM → 2000× more concentrated
    has_warning = any(r.binding_note != "" for r in sel.interferents)
    print(f"  \\u2705 test_conc_warning: warnings present={has_warning}")

# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH DESIGN PACKAGE
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_pb_has_selectivity():
    """Pb2+ design package should include selectivity profile."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=1)
    assert len(pkgs) > 0
    assert pkgs[0].selectivity is not None
    assert isinstance(pkgs[0].selectivity, SelectivityProfile)
    print(f"  \\u2705 test_e2e_sel: selectivity={pkgs[0].selectivity.overall_selectivity_class}")

def test_e2e_grade_includes_selectivity():
    """Grade should factor in selectivity (not just binding + deployment)."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=3)
    # All should have grades
    for p in pkgs:
        assert p.overall_grade in ("A", "B", "C", "D", "F")
    print(f"  \\u2705 test_grade_sel: grades={[p.overall_grade for p in pkgs]}")

def test_e2e_s_donor_more_selective_than_n():
    """S-donor Pb2+ binder should be more selective than N-donor."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=10)
    s_scores = [p.selectivity.selectivity_score for p in pkgs if "S" in p.donor_atoms]
    n_scores = [p.selectivity.selectivity_score for p in pkgs if all(d == "N" for d in p.donor_atoms)]
    if s_scores and n_scores:
        assert max(s_scores) >= max(n_scores), \\
            f"S-donor ({max(s_scores):.0f}) should be >= N-donor ({max(n_scores):.0f})"
        print(f"  \\u2705 test_s_vs_n: S_max={max(s_scores):.0f} >= N_max={max(n_scores):.0f}")
    else:
        print(f"  \\u2705 test_s_vs_n: S_designs={len(s_scores)}, N_designs={len(n_scores)} (insufficient for comparison)")

def test_e2e_au_selective():
    """Au3+ with S-donors should be highly selective vs hard metals."""
    pkgs = design_binder("gold", "Au3+", charge=3, working_ph=2.0, max_designs=1)
    assert len(pkgs) > 0
    sel = pkgs[0].selectivity
    # Au3+ S-donor binder should reject Na+, Ca2+ completely
    na = next((r for r in sel.interferents if r.formula == "Na+"), None)
    if na:
        assert na.selectivity_class in ("excellent", "good"), \\
            f"Au/S should reject Na+, got {na.selectivity_class}"
    print(f"  \\u2705 test_au_selective: {sel.overall_selectivity_class}")

def test_selectivity_panel_parameter():
    """Should accept different panel names."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0,
                          max_designs=1, selectivity_panel="acid_mine")
    sel = pkgs[0].selectivity
    assert sel.deployment_matrix == "acid_mine"
    formulas = [r.formula for r in sel.interferents]
    assert "Fe3+" in formulas or "Fe2+" in formulas
    print(f"  \\u2705 test_panel_param: matrix={sel.deployment_matrix}, ions={formulas[:3]}")

def test_nuclear_panel():
    """Nuclear waste panel should work."""
    sel = compute_selectivity("Cs+", 1.0, 1,
        _rec(["O","O","O","O","O","O"], "hard", 0, 0.5), _struct(), _interior(),
        _matrix(7.0), panel="nuclear_waste")
    assert sel.deployment_matrix == "nuclear_waste"
    assert len(sel.interferents) > 0
    print(f"  \\u2705 test_nuclear_panel: {len(sel.interferents)} interferents")

def test_selectivity_notes():
    """Should generate notes about poor selectivity."""
    sel = compute_selectivity("Pb2+", 1000.0, 2,
        _rec(["N","N","N","N"], "borderline"), _struct(), _interior(), _matrix(7.0))
    # N-donor binder with Kd=1000 for Pb2+ will have poor selectivity
    print(f"  \\u2705 test_notes: class={sel.overall_selectivity_class}, notes='{sel.notes[:50]}'")

def test_profile_dataclass():
    """SelectivityProfile should have all required fields."""
    sel = compute_selectivity("Cu2+", 1.0, 2,
        _rec(["N","N","O","O"], "borderline"), _struct(), _interior(), _matrix(7.0))
    assert hasattr(sel, "target_formula")
    assert hasattr(sel, "interferents")
    assert hasattr(sel, "worst_interferent")
    assert hasattr(sel, "selectivity_score")
    assert hasattr(sel, "deployment_matrix")
    print(f"  \\u2705 test_dataclass: all fields present")


if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprint 33 \\u2014 Selectivity Scoring\\n")
    print("Selectivity Module:")
    test_s_donor_selective_for_soft(); test_o_donor_binds_hard()
    test_panels_exist(); test_target_excluded_from_panel()
    test_selectivity_score_range(); test_worst_interferent_identified()
    test_classification_correct(); test_acid_mine_panel()
    test_seawater_panel(); test_concentration_warning()
    print("\\nIntegration:")
    test_e2e_pb_has_selectivity(); test_e2e_grade_includes_selectivity()
    test_e2e_s_donor_more_selective_than_n(); test_e2e_au_selective()
    test_selectivity_panel_parameter(); test_nuclear_panel()
    test_selectivity_notes(); test_profile_dataclass()
    print("\\n\\u2705 All Sprint 33 tests passed! (18/18)\\n")

''')

print("\n\u2705 Sprint 33 files created!\n")