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
from core.synthesis import generate_synthesis_protocol, SynthesisProtocol
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
    # Synthesis
    synthesis: SynthesisProtocol = None
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
            panel=selectivity_panel, target_dg_kj=thermo.dg_net_kj)

        # Synthesis protocol
        synth = generate_synthesis_protocol(
            gen_a.name, target_formula, gen_a.scaffold_type,
            rec.donor_atoms, rec.donor_type,
            target_softness=hsab)

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
            synthesis=synth,
            decay_chain_warning=decay_warning,
            overall_grade=grade, one_line_summary=summary,
        ))

    # Sort by combined score
    packages.sort(key=lambda p: (-ord(p.overall_grade[0]),
                                   p.thermodynamics.dg_net_kj))
    return packages[:max_designs]


def print_design_package(pkg):
    """Pretty-print a complete design package."""
    print(f"\n{'='*72}")
    print(f"  MABE DESIGN PACKAGE: {pkg.binder_name}")
    print(f"{'='*72}")
    print(f"  Target:      {pkg.target} ({pkg.target_formula}) at pH {pkg.working_ph}")
    print(f"  Grade:       {pkg.overall_grade}")
    print(f"  Summary:     {pkg.one_line_summary}")

    t = pkg.thermodynamics
    print(f"\n  BINDING ({t.confidence} confidence)")
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
    print(f"\n  DEPLOYMENT ({d.deployment_class})")
    print(f"  {'─'*60}")
    print(f"  Score:       {d.deployment_score:.0f}/100  Limiting: {d.limiting_factor}")
    print(f"  Transport:   {d.transport_score:.0f}  Capacity: {d.capacity_mg_g:.0f} mg/g")
    print(f"  Wetting:     {d.wettability} ({d.wetting_score:.0f})")
    print(f"  Thermal:     max {d.max_temp_C}°C ({d.thermal_score:.0f})")
    if d.recommendations:
        for r in d.recommendations[:3]:
            print(f"  → {r}")

    det = pkg.detection
    print(f"\n  DETECTION")
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
        print(f"\n  ☢ NUCLEAR: {pkg.decay_chain_warning}")

    if pkg.selectivity:
        s = pkg.selectivity
        print(f"\n  SELECTIVITY ({s.overall_selectivity_class}, score={s.selectivity_score:.0f}/100)")
        print(f"  {'─'*60}")
        print(f"  Panel: {s.deployment_matrix}")
        for r in sorted(s.interferents, key=lambda x: x.selectivity_ratio)[:5]:
            kd_str = f"{r.predicted_kd_uM:.1f}" if r.predicted_kd_uM < 1e6 else ">10⁶"
            ratio_str = f"{r.selectivity_ratio:.0f}×" if r.selectivity_ratio < 1e6 else ">10⁶×"
            flag = " ⚠" if r.selectivity_class in ("poor", "none") else ""
            print(f"  {r.formula:10s} Kd={kd_str:>8s}  sel={ratio_str:>8s}  {r.selectivity_class}{flag}")
        if s.notes:
            print(f"  ⚠ {s.notes}")

    if pkg.synthesis:
        sy = pkg.synthesis
        print(f"\n  SYNTHESIS")
        print(f"  {'─'*60}")
        print(f"  {sy.difficulty} | {len(sy.steps)} steps | {sy.total_time_hours:.0f}h | "
              f"${sy.total_cost_usd_per_gram:.2f}/g | {sy.scalability}")
        for step in sy.steps:
            print(f"  {step.step_number}. {step.name} ({step.time_hours:.0f}h)")
        if sy.alternative_routes:
            print(f"  Alternatives: {sy.alternative_routes[0]}")

    print(f"\n{'='*72}\n")



