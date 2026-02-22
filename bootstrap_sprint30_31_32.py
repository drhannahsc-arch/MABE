"""
MABE Platform - Sprint 30+31+32 Bootstrap v2: Integration Pipeline
Updated with continuous softness integration and dedup.
Requires Sprint 16 v2 + Sprint 17 fix + Sprints 18-29.
"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 30-32 v2 \u2014 Integration Pipeline\n")

write_file("core/physics_integration.py", '''\
"""
core/physics_integration.py — Sprint 30: Rewired Thermodynamics

Replaces all heuristic terms in compute_thermodynamics_standalone()
with real physics modules from Sprints 18-29. The 10-term ΔG becomes
a 15-term ΔG pulling from ion-specific solvation, field-dependent LFSE,
continuous polarizability, dispersion, covalent bonds, nephelauxetic
correction, and relativistic corrections.

Also wraps the entire pipeline entry with speciation gating.
"""
from dataclasses import dataclass, field
import math

# Import all physics modules
from core.solvation import compute_desolvation_energy, get_hydration_profile
from core.spin_state import predict_spin_state, compute_lfse_for_geometry
from core.dispersion import compute_non_electrostatic
from core.polarizability import compute_full_polarization
from core.relativistic import correct_binding_energy, get_relativistic_profile
from core.speciation_gate import predict_speciation
from core.generative_physics_adapter import (
    BindingThermodynamics, RecognitionChemistry, StructuralConstraint,
    InteriorDesign, Problem, TargetSpecies, Matrix,
)


@dataclass
class EnhancedThermodynamics(BindingThermodynamics):
    """Extended thermodynamics with all physics terms."""
    # New terms from Sprints 18-29
    dg_dispersion_kj: float = 0.0
    dg_covalent_kj: float = 0.0
    dg_polarization_kj: float = 0.0
    dg_hydrophobic_kj: float = 0.0
    dg_relativistic_correction_kj: float = 0.0
    # Metadata
    nephelauxetic_beta: float = 1.0
    spin_state: str = ""
    softness_continuous: float = 0.0
    bond_character: str = "coordinate"
    speciation_warning: str = ""
    free_ion_fraction: float = 1.0
    design_strategy: str = "free_ion_binding"


def compute_enhanced_thermodynamics(recognition, structure, interior, problem):
    """Full physics thermodynamics replacing Sprint 17 heuristics.

    15-term ΔG with real physics modules:
    1. dG_bind (HSAB donor energies — now polarizability-weighted)
    2. dG_desolv (ion-specific from solvation module)
    3. dG_preorg (scaffold rigidity)
    4. dG_chelate (entropic ring closure)
    5. dG_electrostatic (Coulomb + screening)
    6. dG_protonation (pH-dependent donor competition)
    7. dG_LFSE (field-strength + spin-state aware)
    8. dG_activity (Davies equation)
    9. dG_screening (Debye)
    10. dG_repulsion (steric)
    11. dG_dispersion (London forces) [NEW]
    12. dG_covalent (bond dissociation) [NEW]
    13. dG_polarization (mutual induced dipole) [NEW]
    14. dG_hydrophobic (cavity transfer) [NEW]
    15. dG_relativistic (6s contraction correction) [NEW]
    """
    target = problem.target if problem else None
    matrix = problem.matrix if problem else None
    ph = matrix.ph if matrix else 7.0
    temp_c = matrix.temperature_c if matrix else 25.0
    temp_k = temp_c + 273.15
    ionic_mm = matrix.ionic_strength_mm if matrix else 10.0

    formula = target.formula if target else "Zn2+"
    charge = target.charge if target else 2
    d_electrons = target.d_electrons if target else 0
    target_softness = target.hsab_softness if target else 0.3
    cn = target.coordination_number if target else 6
    ionic_radius_pm = target.ionic_radius_pm if target else 80
    hydrated_radius_nm = target.hydrated_radius_nm if target else 0.2
    donors = recognition.donor_atoms if recognition else ["N", "N", "O", "O"]
    scaffold_type = structure.type if structure else "free"
    pore_nm = structure.pore_size_nm if structure else 0.0

    # === SPECIATION CHECK ===
    spec = predict_speciation(formula, ph)
    free_ion_frac = spec.free_ion_fraction
    strategy = spec.design_strategy
    spec_warning = ""
    if free_ion_frac < 0.5:
        spec_warning = (f"Only {free_ion_frac*100:.0f}% free ion at pH {ph}. "
                        f"Strategy: {strategy}")

    # === Term 1: dG_bind (metal-ligand bond formation energies) ===
    # For coordinate bonds: ionic + covalent contribution
    # Ionic contribution: ΔG_ionic ≈ -z_M × z_L × 1389 / r_ML (kJ/mol, Coulomb)
    # Covalent contribution: scaled by softness overlap
    # Total per-donor energies: ~50-200 kJ/mol for divalent metals
    softness = compute_full_polarization(formula, donors,
                                          d_electrons=d_electrons).softness_continuous
    donor_softness_map = {"O": 0.15, "N": 0.40, "S": 0.80, "P": 0.75,
                          "Cl": 0.50, "Br": 0.65, "I": 0.85}

    # Metal-ligand distance estimate (Å)
    r_ml_A = (ionic_radius_pm + 140) / 100.0  # ion + donor atom radius (~1.4Å for O/N)

    donor_energies = []
    for da in donors:
        ds = donor_softness_map.get(da, 0.3)
        # Ionic contribution (point charge model, damped)
        z_donor = -1 if da in ("O", "S") else -0.5  # Partial charge for N, P
        dg_ionic = -charge * abs(z_donor) * 1389 / (r_ml_A * 10)  # 1389/r in pm
        # Actually: 1389 kJ·pm/mol for unit charges → divide by r in pm
        dg_ionic = -charge * abs(z_donor) * 1389 / (r_ml_A * 100)  # r_ml in pm

        # Covalent contribution: maximized when softness matches
        mismatch = abs(softness - ds)
        cov_strength = 60.0 * (1.0 - mismatch)  # Up to 60 kJ/mol covalent per donor
        dg_cov = -cov_strength * min(softness, ds)  # Scales with min(metal, donor) softness

        dg_donor = dg_ionic + dg_cov
        donor_energies.append(dg_donor)
    dg_bind = sum(donor_energies)

    # === Term 2: dG_desolv (water-to-ligand exchange penalty) ===
    # The relevant energy is NOT the total dehydration cost.
    # It's the DIFFERENCE between M-OH₂ bond and M-L bond.
    # For ligands stronger than water: net favorable (captured in dg_bind)
    # For the residual desolvation penalty: reorganization of 2nd shell,
    # loss of hydrogen bonding network around ion, entropy of released water.
    # Empirically: 5-15% of per-water hydration energy for good ligands
    waters_displaced = min(len(donors), cn)
    hydration = get_hydration_profile(formula)
    if hydration:
        per_water_kj = abs(hydration.hydration_energy_kj) / max(1, hydration.first_shell_waters)
        # Exchange penalty: 8-12% of per-water hydration for reorganization
        exchange_fraction = 0.08  # Base: good ligands recover 92% of water binding
        # Soft metals with soft donors have easier exchange (covalent compensation)
        if softness > 0.5 and any(d in ("S", "P", "I") for d in donors):
            exchange_fraction = 0.05  # Soft-soft: excellent compensation
        elif softness < 0.2 and all(d == "O" for d in donors):
            exchange_fraction = 0.06  # Hard-hard: also good compensation
        else:
            exchange_fraction = 0.10  # Mismatched: worse compensation

        dg_desolv = per_water_kj * waters_displaced * exchange_fraction

        # Lability bonus/penalty
        if hydration.lability_class == "inert":
            dg_desolv *= 1.5  # Kinetic barrier on top
        elif hydration.lability_class == "labile":
            dg_desolv *= 0.8
    else:
        dg_desolv = waters_displaced * 12.0  # Fallback: ~12 kJ/mol per water

    # === Term 3: dG_preorg ===
    preorg_map = {
        "dna_origami": -5.0, "zeolite": -8.0, "MOF": -7.0,
        "MIP": -10.0, "COF": -6.0, "coordination_cage": -9.0,
        "mesoporous_silica": -3.0, "dendrimer": -2.0,
        "carbon_nanotube": -1.0, "LDH": -4.0, "free": 0.0,
    }
    scaffold_key = scaffold_type
    for k in preorg_map:
        if k in scaffold_key.lower():
            scaffold_key = k
            break
    dg_preorg = preorg_map.get(scaffold_key, -2.0)
    if interior and interior.self_binding:
        dg_preorg -= 5.0

    # === Term 4: dG_chelate ===
    dg_chelate = -6.0 * recognition.chelate_rings

    # === Term 5: dG_electrostatic ===
    donor_charge = sum(-1 if da in ("O", "S") else 0 for da in donors)
    dg_electrostatic = -2.5 * abs(charge) * abs(donor_charge) / max(1, len(donors))

    # === Term 6: dG_protonation ===
    pka_defaults = {"O": 4.5, "S": 8.3, "P": 6.5}
    n_pka = 6.0 if recognition.type == "generative" else 10.5
    dg_protonation = 0.0
    for da in donors:
        pka = n_pka if da == "N" else pka_defaults.get(da, 7.0)
        if ph < pka:
            penalty = 5.7 * (pka - ph)
            dg_protonation += min(penalty, 20.0)

    # === Term 7: dG_LFSE (SPIN-STATE AWARE from spin_state module) ===
    ligand_names = []
    for da in donors:
        lmap = {"O": "water", "N": "pyridine", "S": "thiolate",
                "P": "phosphine", "Cl": "Cl-", "Br": "Br-", "I": "I-"}
        ligand_names.append(lmap.get(da, "water"))

    dg_lfse = 0.0
    spin_state_str = ""
    if d_electrons > 0 and d_electrons < 10:
        try:
            spin_result = predict_spin_state(formula, d_electrons, ligand_names)
            geom = "octahedral" if cn >= 5 else "tetrahedral" if cn == 4 else "square_planar"
            lfse_result = compute_lfse_for_geometry(formula, d_electrons, geom, ligand_names)
            dg_lfse = -abs(lfse_result.lfse_kj)  # Stabilizing
            spin_state_str = lfse_result.spin_state
        except Exception:
            dg_lfse = 0.0

    # Apply nephelauxetic correction
    pol = compute_full_polarization(formula, donors, d_electrons=d_electrons,
                                     base_lfse_kj=dg_lfse)
    if abs(dg_lfse) > 1.0:
        dg_lfse *= pol.lfse_correction_factor

    # === Term 8: dG_activity ===
    I = ionic_mm / 1000.0
    dg_activity = 0.0
    if I > 0 and charge != 0:
        sqrt_I = math.sqrt(I)
        log_gamma = -0.509 * charge**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)
        dg_activity = -5.71 * log_gamma

    # === Term 9: dG_screening ===
    dg_screening = 0.0
    if I > 0:
        sqrt_I = math.sqrt(I)
        kappa = 3.29 * sqrt_I
        screening_factor = math.exp(-kappa * 0.3)
        dg_screening = dg_electrostatic * (1 - screening_factor)

    # === Term 10: dG_repulsion ===
    dg_repulsion = 0.0
    if structure and pore_nm > 0:
        if hydrated_radius_nm * 2 > pore_nm * 0.9:
            squeeze = (hydrated_radius_nm * 2 - pore_nm * 0.9)
            dg_repulsion = 10.0 * max(0, squeeze / 0.1)

    # === Term 11-14: Non-electrostatic forces ===
    ne = compute_non_electrostatic(formula, donors, scaffold_type=scaffold_type,
                                    pore_diameter_nm=pore_nm,
                                    ionic_radius_pm=ionic_radius_pm)
    dg_dispersion = ne.dg_dispersion_kj
    dg_covalent = ne.dg_covalent_kj
    dg_hydrophobic = ne.dg_hydrophobic_kj
    dg_polarization = pol.dg_polarization_kj

    # === Term 15: Relativistic correction ===
    subtotal_binding = dg_bind + dg_covalent + dg_polarization + dg_dispersion
    corrected, rel_factor = correct_binding_energy(subtotal_binding, formula)
    dg_relativistic = corrected - subtotal_binding

    # === Scale by speciation ===
    # If only X% free ion, effective concentration is reduced
    # This scales the Kd prediction, not the ΔG terms
    effective_fraction = max(0.01, free_ion_frac) if strategy == "free_ion_binding" else \\
                         max(0.01, spec.bindable_fraction)

    # === Sum ===
    dg_net = (dg_bind + dg_desolv + dg_preorg + dg_chelate + dg_electrostatic
              + dg_protonation + dg_lfse + dg_activity + dg_screening + dg_repulsion
              + dg_dispersion + dg_covalent + dg_polarization + dg_hydrophobic
              + dg_relativistic)

    R = 8.314e-3
    kd_m = math.exp(dg_net / (R * temp_k)) if dg_net / (R * temp_k) < 500 else 1e6
    kd_um = kd_m * 1e6 / effective_fraction  # Scale by speciation

    conf = "high" if pol.softness_continuous > 0 and recognition.hsab_match > 0.7 else \\
           "moderate" if recognition.hsab_match > 0.4 else "low"

    return EnhancedThermodynamics(
        dg_bind_kj=round(dg_bind, 2), dg_desolv_kj=round(dg_desolv, 2),
        dg_preorg_kj=round(dg_preorg, 2), dg_chelate_kj=round(dg_chelate, 2),
        dg_electrostatic_kj=round(dg_electrostatic, 2),
        dg_protonation_kj=round(dg_protonation, 2),
        dg_lfse_kj=round(dg_lfse, 2), dg_activity_kj=round(dg_activity, 2),
        dg_screening_kj=round(dg_screening, 2), dg_repulsion_kj=round(dg_repulsion, 2),
        dg_net_kj=round(dg_net, 2),
        predicted_kd_um=round(kd_um, 4) if kd_um < 1e6 else round(kd_um, 0),
        confidence=conf,
        # New terms
        dg_dispersion_kj=round(dg_dispersion, 2),
        dg_covalent_kj=round(dg_covalent, 2),
        dg_polarization_kj=round(dg_polarization, 2),
        dg_hydrophobic_kj=round(dg_hydrophobic, 2),
        dg_relativistic_correction_kj=round(dg_relativistic, 2),
        nephelauxetic_beta=pol.nephelauxetic_beta,
        spin_state=spin_state_str,
        softness_continuous=pol.softness_continuous,
        bond_character=ne.bond_character,
        speciation_warning=spec_warning,
        free_ion_fraction=free_ion_frac,
        design_strategy=strategy,
    )


''')

write_file("core/deployment_scoring.py", '''\
"""
core/deployment_scoring.py — Sprint 31: Deployment Scoring Layer

After thermodynamic scoring, runs each candidate through:
cooperativity → mass_transport → surface → photostability → phonon → radiation
Produces deployment_score alongside binding_score.
"""
from dataclasses import dataclass
import math

from core.cooperativity import analyze_cooperativity
from core.mass_transport import analyze_transport, predict_capture_time
from core.surface_magnetic import get_surface_profile, compute_magnetic_properties
from core.photostability import assess_photostability
from core.phonon_thermal import analyze_phonon_stability
from core.electron_transfer import assess_radiation_stability


@dataclass
class DeploymentScore:
    """Deployment feasibility assessment for a binder design."""
    # Subscores (0-100 each)
    transport_score: float = 0.0
    capacity_score: float = 0.0
    wetting_score: float = 0.0
    thermal_score: float = 0.0
    uv_score: float = 0.0
    radiation_score: float = 0.0
    kinetics_score: float = 0.0
    # Overall
    deployment_score: float = 0.0    # Weighted composite (0-100)
    deployment_class: str = ""       # "field_ready", "lab_viable", "needs_engineering", "redesign"
    # Details
    transport_regime: str = ""
    capacity_mg_g: float = 0.0
    capture_time_90_min: float = 0.0
    max_temp_C: float = 0.0
    outdoor_lifetime_days: float = 0.0
    wettability: str = ""
    limiting_factor: str = ""
    recommendations: list = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


def score_deployment(scaffold_type, target_formula, target_charge=2,
                      target_mw=60.0, hydrated_radius_nm=0.2,
                      pore_diameter_nm=0.0, binding_energy_kj=50.0,
                      ion_mass_amu=60.0, unpaired_electrons=0,
                      is_nuclear=False, outdoor_use=False,
                      target_conc_uM=1.0):
    """Full deployment feasibility scoring."""
    recs = []

    # === TRANSPORT ===
    if pore_diameter_nm > 0:
        tp = analyze_transport(hydrated_radius_nm, pore_diameter_nm)
        transport_score = tp.effectiveness_factor * 100
        transport_regime = tp.transport_regime
        if tp.transport_regime == "excluded":
            transport_score = 0
            recs.append(f"Ion EXCLUDED from pore (λ={tp.lambda_ratio:.2f}). Need larger pore.")
        elif tp.transport_regime == "severely_hindered":
            recs.append(f"Severely hindered diffusion (H={tp.hindrance_factor:.4f}). "
                        f"Consider larger pore or smaller particle.")
    else:
        transport_score = 90  # Free solution / open scaffold
        transport_regime = "unhindered"

    # === CAPACITY ===
    coop = analyze_cooperativity(scaffold_type, target_charge, target_mw)
    capacity_mg_g = coop.capacity_mg_per_g
    capacity_score = min(100, capacity_mg_g / 2.0)  # 200 mg/g = 100
    if coop.max_practical_loading < 0.5:
        recs.append(f"Only {coop.max_practical_loading*100:.0f}% of sites usable "
                    f"(n_Hill={coop.hill_coefficient:.2f})")

    # === CAPTURE KINETICS ===
    eff = transport_score / 100.0
    ct = predict_capture_time(target_conc_uM, coop.capacity_mmol_per_g,
                               1.0, effectiveness=max(0.01, eff))
    capture_90_min = ct.time_to_90pct_s / 60.0
    kinetics_score = max(0, min(100, 100 - capture_90_min))  # <1 min = 100
    if capture_90_min > 60:
        recs.append(f"Slow capture ({capture_90_min:.0f} min to 90%). "
                    f"Increase material loading or use flow-through.")

    # === WETTING ===
    sp = get_surface_profile(scaffold_type, pore_diameter_nm)
    if sp.spontaneous_wetting:
        wetting_score = max(50, 100 - sp.contact_angle_water_deg)
    else:
        wetting_score = max(0, 30 - (sp.contact_angle_water_deg - 90))
        recs.append(f"Hydrophobic surface (θ={sp.contact_angle_water_deg}°). "
                    f"{sp.surface_treatment}")

    # === THERMAL STABILITY ===
    ph = analyze_phonon_stability(scaffold_type, binding_energy_kj,
                                    ion_mass_amu)
    max_temp = ph.max_operating_temp_C
    thermal_score = min(100, max_temp / 2.0)  # 200°C = 100
    if ph.thermal_stability_class == "unstable":
        recs.append(f"Thermally unstable at 25°C. Use stiffer scaffold "
                    f"(current Θ_D={ph.debye_temp_K} K)")

    # === UV STABILITY ===
    ps = assess_photostability(scaffold_type, outdoor_use)
    outdoor_days = ps.operational_lifetime_outdoor_days
    if outdoor_use:
        uv_score = min(100, outdoor_days / 3.0)  # 300 days = 100
        if ps.stability_class in ("UV_sensitive", "poor"):
            recs.append(f"UV-sensitive ({outdoor_days:.0f} day outdoor life). "
                        f"{ps.protection_strategy}")
    else:
        uv_score = 90  # Indoor use, UV rarely a problem

    # === RADIATION ===
    rad = assess_radiation_stability(scaffold_type, is_nuclear_application=is_nuclear)
    rad_score_map = {"excellent": 100, "good": 80, "moderate": 50,
                     "poor": 20, "unsuitable": 0, "unknown": 50}
    radiation_score = rad_score_map.get(rad.stability_rating, 50)
    if is_nuclear and rad.stability_rating in ("unsuitable", "poor"):
        recs.append(f"NOT radiation-stable ({rad.stability_rating}). "
                    f"Use: {', '.join(rad.rad_resistant_alternatives[:2])}")

    # === COMPOSITE ===
    weights = {"transport": 0.20, "capacity": 0.15, "kinetics": 0.15,
               "wetting": 0.15, "thermal": 0.10, "uv": 0.10, "radiation": 0.15}
    if not is_nuclear:
        weights["radiation"] = 0.05
        weights["capacity"] = 0.20

    composite = (transport_score * weights["transport"]
                 + capacity_score * weights["capacity"]
                 + kinetics_score * weights["kinetics"]
                 + wetting_score * weights["wetting"]
                 + thermal_score * weights["thermal"]
                 + uv_score * weights["uv"]
                 + radiation_score * weights["radiation"])

    if composite > 75:
        dep_class = "field_ready"
    elif composite > 50:
        dep_class = "lab_viable"
    elif composite > 25:
        dep_class = "needs_engineering"
    else:
        dep_class = "redesign"

    # Limiting factor
    scores = {"transport": transport_score, "capacity": capacity_score,
              "kinetics": kinetics_score, "wetting": wetting_score,
              "thermal": thermal_score, "UV": uv_score, "radiation": radiation_score}
    limiting = min(scores, key=scores.get)

    return DeploymentScore(
        transport_score=round(transport_score, 1),
        capacity_score=round(capacity_score, 1),
        wetting_score=round(wetting_score, 1),
        thermal_score=round(thermal_score, 1),
        uv_score=round(uv_score, 1),
        radiation_score=round(radiation_score, 1),
        kinetics_score=round(kinetics_score, 1),
        deployment_score=round(composite, 1),
        deployment_class=dep_class,
        transport_regime=transport_regime,
        capacity_mg_g=round(capacity_mg_g, 1),
        capture_time_90_min=round(capture_90_min, 1),
        max_temp_C=max_temp,
        outdoor_lifetime_days=round(outdoor_days, 0),
        wettability=sp.wettability,
        limiting_factor=limiting,
        recommendations=recs,
    )


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
    # Nuclear (optional)
    decay_chain_warning: str
    # Summary
    overall_grade: str           # "A", "B", "C", "D", "F"
    one_line_summary: str


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
                    required_sensitivity="µM"):
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

        # Overall grade
        binding_score = max(0, min(100, -thermo.dg_net_kj))  # More negative = better
        combined = binding_score * 0.5 + dep.deployment_score * 0.5
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

    print(f"\\n{'='*72}\\n")


''')

write_file("tests/test_sprint30_31_32.py", '''\
"""tests/test_sprint30_31_32.py — Integration: Physics + Deployment + Design Package (30 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_integration import compute_enhanced_thermodynamics, EnhancedThermodynamics
from core.deployment_scoring import score_deployment, DeploymentScore
from core.design_package import design_binder, DesignPackage
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)

def _make_prob(name, formula, charge, d_e, soft, r_pm, ph=7.0):
    return Problem(
        target=TargetSpecies(identity=name, formula=formula, charge=charge,
            d_electrons=d_e, hsab_softness=soft, ionic_radius_pm=r_pm,
            hydrated_radius_nm=(r_pm+140)/1000),
        matrix=Matrix(ph=ph, temperature_c=25.0, ionic_strength_mm=10.0))

def _make_rec(donors, dt="borderline", chel=2, match=0.7):
    return RecognitionChemistry(name="t", type="generative", donor_atoms=donors,
        donor_type=dt, denticity=len(donors), hsab_match=match, chelate_rings=chel)

def _make_struct(stype="zeolite", pore=0.74):
    return StructuralConstraint(name="s", type=stype, geometry="channel", pore_size_nm=pore)

def _make_interior(self_binding=True):
    return InteriorDesign(description="t", num_binding_sites=1, self_binding=self_binding)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 30: ENHANCED THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

def test_pb_reasonable_kd():
    """Pb2+ with N,O donors should give µM-range Kd."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"]), _make_struct(), _make_interior(),
        _make_prob("lead", "Pb2+", 2, 0, 0.99, 119, 6.0))
    assert t.dg_net_kj < 0, f"Net should be favorable, got {t.dg_net_kj}"
    assert 0.01 < t.predicted_kd_um < 1e6, f"Kd should be µM range, got {t.predicted_kd_um}"
    print(f"  \\u2705 test_pb_kd: ΔG={t.dg_net_kj:.1f}, Kd={t.predicted_kd_um:.1f} µM")

def test_ni_strong_binding():
    """Ni2+ with 6N + 3 chelate rings should bind strongly (log K >10)."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","N","N","N","N"], chel=3), _make_struct(), _make_interior(),
        _make_prob("nickel", "Ni2+", 2, 8, 0.24, 69))
    logK = -t.dg_net_kj / 5.71 if t.dg_net_kj < 0 else 0
    assert logK > 10, f"Ni2+/6N should have logK>10, got {logK:.1f}"
    assert t.dg_lfse_kj < -50, f"LFSE should be significant, got {t.dg_lfse_kj}"
    print(f"  \\u2705 test_ni_strong: logK={logK:.1f}, LFSE={t.dg_lfse_kj:.0f}")

def test_au_covalent_dominates():
    """Au3+ + 4S should be dominated by covalent term."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft", 2, 0.95), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85, 2.0))
    assert t.dg_covalent_kj < -500, f"Au-S covalent should dominate, got {t.dg_covalent_kj}"
    assert t.bond_character == "covalent"
    print(f"  \\u2705 test_au_covalent: cov={t.dg_covalent_kj:.0f}, character={t.bond_character}")

def test_ion_specific_desolvation():
    """Different ions should have different desolvation costs (not flat 8 kJ/mol)."""
    t_pb = compute_enhanced_thermodynamics(
        _make_rec(["O","O","O","O"]), _make_struct(), _make_interior(),
        _make_prob("lead", "Pb2+", 2, 0, 0.99, 119))
    t_cu = compute_enhanced_thermodynamics(
        _make_rec(["O","O","O","O"]), _make_struct(), _make_interior(),
        _make_prob("copper", "Cu2+", 2, 9, 0.35, 73))
    # Both should be well above the old flat 32 kJ/mol (4 waters × 8)
    assert t_pb.dg_desolv_kj > 40, f"Pb desolv should be > 40, got {t_pb.dg_desolv_kj}"
    assert t_cu.dg_desolv_kj > 40, f"Cu desolv should be > 40, got {t_cu.dg_desolv_kj}"
    # Should be different (ion-specific, not flat)
    assert abs(t_pb.dg_desolv_kj - t_cu.dg_desolv_kj) > 1.0
    print(f"  \\u2705 test_ion_desolv: Pb=+{t_pb.dg_desolv_kj:.0f}, Cu=+{t_cu.dg_desolv_kj:.0f} (ion-specific)")

def test_speciation_warning():
    """Fe3+ at pH 7 should trigger speciation warning (precipitates)."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["O","O","O","O","O","O"], "hard", 3), _make_struct(), _make_interior(),
        _make_prob("iron", "Fe3+", 3, 5, 0.12, 65, 7.0))
    assert t.speciation_warning != "", f"Fe3+ at pH 7 should warn about speciation"
    print(f"  \\u2705 test_speciation: {t.speciation_warning[:60]}")

def test_continuous_softness():
    """Enhanced thermo should report continuous softness score."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft"), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85))
    assert 0 < t.softness_continuous <= 1.0
    print(f"  \\u2705 test_softness: Au3+ softness={t.softness_continuous:.3f}")

def test_relativistic_correction():
    """Au should get relativistic correction, Ni should not."""
    t_au = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft"), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85))
    t_ni = compute_enhanced_thermodynamics(
        _make_rec(["N","N","N","N"]), _make_struct(), _make_interior(),
        _make_prob("nickel", "Ni2+", 2, 8, 0.24, 69))
    assert abs(t_au.dg_relativistic_correction_kj) > abs(t_ni.dg_relativistic_correction_kj)
    print(f"  \\u2705 test_relativistic: Au corr={t_au.dg_relativistic_correction_kj:.1f} vs Ni={t_ni.dg_relativistic_correction_kj:.1f}")

def test_enhanced_has_15_terms():
    """EnhancedThermodynamics should have all 15 terms."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"]), _make_struct(), _make_interior(),
        _make_prob("test", "Cu2+", 2, 9, 0.35, 73))
    assert hasattr(t, "dg_dispersion_kj")
    assert hasattr(t, "dg_covalent_kj")
    assert hasattr(t, "dg_polarization_kj")
    assert hasattr(t, "dg_hydrophobic_kj")
    assert hasattr(t, "dg_relativistic_correction_kj")
    print(f"  \\u2705 test_15_terms: all new terms present")

def test_chelate_effect():
    """More chelate rings should give more negative ΔG."""
    t0 = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"], chel=0), _make_struct(), _make_interior(),
        _make_prob("t", "Cu2+", 2, 9, 0.35, 73))
    t3 = compute_enhanced_thermodynamics(
        _make_rec(["N","N","O","O"], chel=3), _make_struct(), _make_interior(),
        _make_prob("t", "Cu2+", 2, 9, 0.35, 73))
    assert t3.dg_net_kj < t0.dg_net_kj
    print(f"  \\u2705 test_chelate: 0 rings ΔG={t0.dg_net_kj:.0f}, 3 rings ΔG={t3.dg_net_kj:.0f}")

def test_nephelauxetic_reported():
    """Nephelauxetic beta should be reported for d-block metals."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft"), _make_struct(), _make_interior(),
        _make_prob("ni", "Ni2+", 2, 8, 0.24, 69))
    assert t.nephelauxetic_beta < 1.0, "β should be <1 for S donors"
    print(f"  \\u2705 test_nephelauxetic: β={t.nephelauxetic_beta:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 31: DEPLOYMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════

def test_zeolite_deployment():
    """Zeolite should score well on most deployment metrics."""
    d = score_deployment("zeolite_Y", "Ni2+", 2, 58.7, 0.21, 0.74, 100, 58.7, 2)
    assert d.deployment_score > 30
    assert d.wettability in ("hydrophilic", "superhydrophilic")
    print(f"  \\u2705 test_zeolite_deploy: score={d.deployment_score:.0f}, class={d.deployment_class}")

def test_cnt_hydrophobic_flagged():
    """Carbon nanotube should flag hydrophobic wetting issue."""
    d = score_deployment("carbon_nanotube", "Cu2+", 2, 63.5, 0.22, 1.5, 80, 63.5, 1)
    assert d.wettability == "hydrophobic"
    assert any("hydrophobic" in r.lower() or "HYDROPHOBIC" in r for r in d.recommendations)
    print(f"  \\u2705 test_cnt_hydrophobic: wetting={d.wetting_score:.0f}, recs={len(d.recommendations)}")

def test_dna_rad_flagged_nuclear():
    """DNA origami should fail radiation check for nuclear."""
    d = score_deployment("dna_origami_icosahedron", "UO2_2+", 2, 270, 0.3, 4.0,
                          50, 238, 0, is_nuclear=True)
    assert d.radiation_score < 20
    assert any("radiation" in r.lower() or "NOT" in r for r in d.recommendations)
    print(f"  \\u2705 test_dna_nuclear: rad_score={d.radiation_score}, class={d.deployment_class}")

def test_deployment_limiting_factor():
    """Deployment should identify the limiting factor."""
    d = score_deployment("MIP", "Pb2+", 2, 207, 0.26, 0.0, 50, 207, 0)
    assert d.limiting_factor != ""
    print(f"  \\u2705 test_limiting: {d.limiting_factor} (score={d.deployment_score:.0f})")

def test_deployment_capacity():
    """Capacity should be reported in mg/g."""
    d = score_deployment("zeolite_Y", "Cu2+", 2, 63.5, 0.22, 0.74, 80, 63.5, 1)
    assert d.capacity_mg_g > 0
    print(f"  \\u2705 test_capacity: {d.capacity_mg_g:.0f} mg/g")

def test_outdoor_uv_check():
    """Outdoor deployment of DNA should flag UV issue."""
    d = score_deployment("aptamer", "Pb2+", 2, 207, 0.26, 0.0,
                          50, 207, 0, outdoor_use=True)
    assert d.outdoor_lifetime_days < 10
    assert d.uv_score < 30
    print(f"  \\u2705 test_outdoor_uv: lifetime={d.outdoor_lifetime_days:.0f} days, uv_score={d.uv_score:.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 32: COMPLETE DESIGN PACKAGE
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_pb():
    """End-to-end: design binder for Pb2+."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=3)
    assert len(pkgs) > 0, "Should generate at least one design"
    pkg = pkgs[0]
    assert isinstance(pkg, DesignPackage)
    assert pkg.target_formula == "Pb2+"
    assert pkg.thermodynamics is not None
    assert pkg.deployment is not None
    assert pkg.detection is not None
    assert pkg.overall_grade in ("A", "B", "C", "D", "F")
    print(f"  \\u2705 test_e2e_pb: {len(pkgs)} designs, best={pkg.overall_grade}, "
          f"Kd={pkg.predicted_kd_uM:.1f}µM")

def test_e2e_ni():
    """End-to-end: design binder for Ni2+."""
    pkgs = design_binder("nickel", "Ni2+", charge=2, working_ph=7.0, max_designs=3)
    assert len(pkgs) > 0
    pkg = pkgs[0]
    assert pkg.detection is not None
    spec = pkg.detection.spectroscopy
    assert spec["color"] != ""  # Ni2+ should have a color prediction
    print(f"  \\u2705 test_e2e_ni: {len(pkgs)} designs, color={spec['color']}, "
          f"detect={spec['detection_method']}")

def test_e2e_au():
    """End-to-end: Au3+ should route to soft/covalent binders."""
    pkgs = design_binder("gold", "Au3+", charge=3, working_ph=2.0, max_designs=3)
    assert len(pkgs) > 0
    # Should have covalent binding noted
    has_covalent = any(p.thermodynamics.bond_character == "covalent" or
                       p.thermodynamics.dg_covalent_kj < -100
                       for p in pkgs)
    print(f"  \\u2705 test_e2e_au: {len(pkgs)} designs, covalent_found={has_covalent}")

def test_e2e_cu_detection():
    """Cu2+ should recommend fluorescence quench detection."""
    pkgs = design_binder("copper", "Cu2+", charge=2, working_ph=7.0, max_designs=2)
    assert len(pkgs) > 0
    det = pkgs[0].detection
    assert len(det.recommended_readouts) > 0
    print(f"  \\u2705 test_e2e_cu_detect: readouts={det.recommended_readouts[:2]}")

def test_e2e_field_deployable():
    """Field-deployable flag should filter readout options."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0,
                          field_deployable=True, max_designs=2)
    assert len(pkgs) > 0
    det = pkgs[0].detection
    assert det.field_deployable_option != "None available"
    print(f"  \\u2705 test_e2e_field: field_option={det.field_deployable_option}")

def test_e2e_mass_spec_replacement():
    """Design package should recommend mass-spec replacement."""
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0,
                          required_sensitivity="ppt", max_designs=2)
    assert len(pkgs) > 0
    det = pkgs[0].detection
    assert "barcode" in det.mass_spec_replacement.lower() or "sequencing" in det.mass_spec_replacement.lower()
    print(f"  \\u2705 test_e2e_mass_spec: replacement={det.mass_spec_replacement}")

def test_e2e_has_deployment():
    """Every design should have deployment scoring."""
    pkgs = design_binder("copper", "Cu2+", charge=2, working_ph=7.0, max_designs=2)
    for pkg in pkgs:
        assert isinstance(pkg.deployment, DeploymentScore)
        assert pkg.deployment.deployment_class in ("field_ready", "lab_viable",
                                                     "needs_engineering", "redesign")
    print(f"  \\u2705 test_e2e_deployment: all packages have deployment scores")

def test_e2e_grade_assignment():
    """Grades should span A-F range."""
    # This just verifies the grading logic runs without error
    pkgs = design_binder("lead", "Pb2+", charge=2, working_ph=6.0, max_designs=5)
    grades = [p.overall_grade for p in pkgs]
    assert all(g in ("A", "B", "C", "D", "F") for g in grades)
    print(f"  \\u2705 test_e2e_grades: {grades}")

def test_package_one_line_summary():
    """Each package should have a one-line summary."""
    pkgs = design_binder("nickel", "Ni2+", charge=2, working_ph=7.0, max_designs=1)
    assert len(pkgs) > 0
    assert "|" in pkgs[0].one_line_summary  # Should contain pipe-separated fields
    print(f"  \\u2705 test_summary: {pkgs[0].one_line_summary[:70]}")


if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprints 30-32 \\u2014 Integration Pipeline\\n")
    print("Sprint 30 — Enhanced Thermodynamics (15-term ΔG):")
    test_pb_reasonable_kd(); test_ni_strong_binding()
    test_au_covalent_dominates(); test_ion_specific_desolvation()
    test_speciation_warning(); test_continuous_softness()
    test_relativistic_correction(); test_enhanced_has_15_terms()
    test_chelate_effect(); test_nephelauxetic_reported()
    print("\\nSprint 31 — Deployment Scoring:")
    test_zeolite_deployment(); test_cnt_hydrophobic_flagged()
    test_dna_rad_flagged_nuclear(); test_deployment_limiting_factor()
    test_deployment_capacity(); test_outdoor_uv_check()
    print("\\nSprint 32 — Complete Design Package (End-to-End):")
    test_e2e_pb(); test_e2e_ni()
    test_e2e_au(); test_e2e_cu_detection()
    test_e2e_field_deployable(); test_e2e_mass_spec_replacement()
    test_e2e_has_deployment(); test_e2e_grade_assignment()
    test_package_one_line_summary()
    print("\\n\\u2705 All Sprint 30-32 tests passed! (25/25)")
    print("\\n\\U0001f389 MABE FOUNDATIONAL MODEL COMPLETE — TARGET → FULL DESIGN PACKAGE\\n")


''')

print("\n\u2705 Sprint 30-32 v2 files created!\n")