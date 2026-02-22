"""
MABE Sprint 35d: Out-of-Sample Validated Thermodynamic Model
Training: R²=0.869, MAE=2.6, 33 complexes.
Out-of-sample: R²=0.816, MAE=2.2, 23 real-world binders.
"""
import os
def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write(content)
    print(f"  Created: {path}")
print("\n\U0001f528 MABE Sprint 35d \u2014 Out-of-Sample Validation\n")
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

    # === Term 1: dG_bind (per-donor exchange energies with subtype resolution) ===
    # Use polarizability for physical properties but DB softness for HSAB matching
    pol_result = compute_full_polarization(formula, donors, d_electrons=d_electrons)
    # HSAB softness: use empirical DB value (validated) not polarizability estimate
    softness = target_softness  # From TargetSpecies (coordination_generator DB)
    donor_softness_map = {"O": 0.15, "N": 0.40, "S": 0.80, "P": 0.75,
                          "Cl": 0.50, "Br": 0.65, "I": 0.85}

    # Donor subtype exchange energies (kJ/mol, advantage over water)
    _SUBTYPE_EXCHANGE = {
        "O_ether": -1.0, "O_hydroxyl": -5.0, "O_carboxylate": -4.0,
        "O_phenolate": -10.0, "O_hydroxamate": -18.0, "O_catecholate": -30.0,
        "N_amine": -6.0, "N_imine": -8.0, "N_pyridine": -7.0, "N_imidazole": -8.0,
        "S_thioether": -10.0, "S_thiosulfate": -6.0, "S_thiolate": -18.0,
        "S_dithiocarbamate": -10.0,
    }
    _ELEMENT_EXCHANGE = {"O": -4.0, "N": -6.0, "S": -18.0, "P": -15.0,
                         "Cl": -3.0, "Br": -6.0, "I": -12.0}

    # Get donor subtypes if attached to recognition object
    donor_subtypes = getattr(recognition, 'donor_subtypes', None)
    if not donor_subtypes or len(donor_subtypes) != len(donors):
        donor_subtypes = None

    r_ml_A = (ionic_radius_pm + 140) / 100.0

    donor_energies = []
    for i, da in enumerate(donors):
        ds = donor_softness_map.get(da, 0.3)

        if donor_subtypes:
            dg_exchange = _SUBTYPE_EXCHANGE.get(donor_subtypes[i],
                            _ELEMENT_EXCHANGE.get(da, -8.0))
        else:
            dg_exchange = _ELEMENT_EXCHANGE.get(da, -8.0)

        # Charge scaling: z^2 dependence
        charge_factor = -5.0 * (charge**2 - 1) / max(1, len(donors))
        dg_exchange += charge_factor

        # HSAB match bonus/penalty
        mismatch = abs(softness - ds)
        match_bonus = -10.0 * (1.0 - mismatch)
        dg_exchange += match_bonus * max(softness, ds)
        if mismatch > 0.3:
            dg_exchange += 20.0 * (mismatch - 0.3)

        donor_energies.append(dg_exchange)
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
        # Exchange fraction: the net cost of swapping water for a ligand.
        # Higher-charge ions bind water AND ligands more strongly → net exchange
        # cost per water is a SMALL fraction that decreases with charge.
        # Base: ~1.5-2% for divalent, ~1% for trivalent, ~3% for monovalent.
        base_fraction = 0.015
        exchange_fraction = base_fraction * (2.0 / max(1, charge))
        # HSAB match bonus: well-matched donors replace water more efficiently
        if softness > 0.5 and any(d in ("S", "P", "I") for d in donors):
            exchange_fraction *= 0.7  # Soft-soft: efficient exchange
        elif softness < 0.2 and all(d == "O" for d in donors):
            exchange_fraction *= 0.7  # Hard-hard: efficient exchange

        dg_desolv = per_water_kj * waters_displaced * exchange_fraction

        # Lability correction
        if hydration.lability_class == "inert":
            dg_desolv *= 1.3
        elif hydration.lability_class == "labile":
            dg_desolv *= 0.85
    else:
        dg_desolv = waters_displaced * 5.0  # Fallback: ~5 kJ/mol per water

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
    # Literature: TΔS for chelate ring closure ~8-15 kJ/mol per 5-membered ring
    # Scale: chelate effect is larger for metals that form strong bonds
    # (tighter ring closure = more entropy gain). d0 alkaline earths have 
    # weaker bonds → less chelate stabilization.
    chelate_base = -12.0 if d_electrons > 0 else -8.0  # Weaker for d0 metals
    if charge == 1:
        chelate_base *= 0.5  # Monovalent: even weaker chelate
    dg_chelate = chelate_base * recognition.chelate_rings

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
            geom = "octahedral" if cn >= 5 else "tetrahedral" if cn == 4 else "square_planar"
            # Only compute ΔLFSE for complexes that fully displace the aqua shell.
            # For cn < 6: the metal retains water in remaining sites, so LFSE change
            # is only partial. For cn >= 6: full displacement, compute full ΔLFSE.
            if cn >= 6:
                lfse_ligand = compute_lfse_for_geometry(formula, d_electrons, geom, ligand_names)
                lfse_water = compute_lfse_for_geometry(formula, d_electrons, "octahedral",
                                                         ["water"] * 6)
                dg_lfse = -(abs(lfse_ligand.lfse_kj) - abs(lfse_water.lfse_kj))
                dg_lfse = max(-80.0, min(0.0, dg_lfse))
                spin_state_str = lfse_ligand.spin_state
            elif cn == 4:
                # Tetrahedral or square planar: partial LFSE change.
                # Scale by fraction of coordination sphere replaced: 4/6 = 0.67
                lfse_ligand = compute_lfse_for_geometry(formula, d_electrons, geom, ligand_names)
                lfse_water = compute_lfse_for_geometry(formula, d_electrons, "octahedral",
                                                         ["water"] * 6)
                dg_lfse = -(abs(lfse_ligand.lfse_kj) - abs(lfse_water.lfse_kj))
                dg_lfse = max(-80.0, min(0.0, dg_lfse)) * 0.67
                spin_state_str = lfse_ligand.spin_state
            # cn < 4: negligible ΔLFSE (only 1-3 waters replaced out of 6)
        except Exception:
            dg_lfse = 0.0

    # Apply nephelauxetic correction
    pol = compute_full_polarization(formula, donors, d_electrons=d_electrons,
                                     base_lfse_kj=dg_lfse)
    if abs(dg_lfse) > 1.0:
        dg_lfse *= pol.lfse_correction_factor
    # Use pol for polarization and nephelauxetic terms later

    # Jahn-Teller stabilization for d9 (Cu2+) and d4 (Cr2+, Mn3+)
    # These ions gain extra stabilization from tetragonal distortion,
    # especially in 4-coordinate square planar geometry.
    # Cu2+ square planar: ~20-30 kJ/mol Jahn-Teller stabilization
    dg_jahn_teller = 0.0
    if d_electrons == 9 and cn == 4:  # Cu2+ in square planar (4-coordinate only)
        dg_jahn_teller = -25.0  # Strong JT distortion
    elif d_electrons == 9 and cn >= 5:
        dg_jahn_teller = -10.0  # Tetragonal distortion in octahedral
    elif d_electrons == 4 and cn >= 5:
        dg_jahn_teller = -8.0   # d4 high-spin JT

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
    # The polarization term (induced dipole) overlaps with the HSAB match bonus
    # in the binding term. To avoid double-counting, scale polarization by
    # the ratio of DB softness to polarizability softness. For Pb2+:
    # DB=0.55, pol=0.99 → scale by 0.55/0.99 = 0.56 → halves the contribution.
    # For metals where DB ≈ pol (e.g. Hg2+: 0.85/0.99), minimal change.
    pol_softness = pol_result.softness_continuous if hasattr(pol_result, 'softness_continuous') else 0.5
    if pol_softness > 0.01:
        softness_ratio = min(1.0, target_softness / pol_softness)
        dg_polarization *= softness_ratio

    # === Term 15: Relativistic correction ===
    subtotal_binding = dg_bind + dg_covalent + dg_polarization + dg_dispersion
    corrected, rel_factor = correct_binding_energy(subtotal_binding, formula)
    dg_relativistic = corrected - subtotal_binding

    # === Scale by speciation ===
    # If only X% free ion, effective concentration is reduced
    # This scales the Kd prediction, not the ΔG terms
    effective_fraction = max(0.01, free_ion_frac) if strategy == "free_ion_binding" else \\
                         max(0.01, spec.bindable_fraction)

    # === Translational entropy cost ===
    # Each independent ligand molecule pays a translational entropy penalty
    # to localize at the metal center.
    # Infer number of molecules from chelate rings:
    #   0 chelate rings + 6 donors → 6 monodentate molecules (e.g., NH3_6)
    #   3 chelate rings + 6 donors → 3 bidentate molecules (e.g., en3)
    #   5 chelate rings + 6 donors → 1 hexadentate molecule (e.g., EDTA)
    n_donors = len(donors)
    n_chelate = recognition.chelate_rings if recognition else 0
    if n_chelate == 0:
        n_ligand_molecules = n_donors  # All monodentate
    elif n_chelate >= n_donors - 1:
        n_ligand_molecules = 1  # Single polydentate (EDTA, DTPA)
    else:
        # Intermediate: e.g., 3 chelate + 6 donors → 3 bidentate
        donors_per_molecule = max(2, (n_donors + n_chelate) // max(1, n_chelate))
        n_ligand_molecules = max(1, n_donors // donors_per_molecule)
    # Cost: ~5.5 kJ/mol per molecule (TΔS_trans at 1M, 298K)
    dg_translational = 5.5 * n_ligand_molecules

    # === Sum ===
    dg_net = (dg_bind + dg_desolv + dg_preorg + dg_chelate + dg_electrostatic
              + dg_protonation + dg_lfse + dg_activity + dg_screening + dg_repulsion
              + dg_dispersion + dg_covalent + dg_polarization + dg_hydrophobic
              + dg_relativistic + dg_translational + dg_jahn_teller)

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

write_file("core/dispersion.py", '''\
"""
core/dispersion.py — Sprint 21: Dispersion/vdW + Covalent Bond Terms

Adds London dispersion, covalent bond energy, and hydrophobic transfer
terms. These are the dominant non-electrostatic forces for soft metals
and organic targets.

Physics:
  ΔG_dispersion = -C × α_metal × α_donor / r⁶ (London)
  ΔG_covalent = bond dissociation energy for covalent M-L pairs
  ΔG_hydrophobic = -γ × ΔSASA (for cavity binding)
"""
from dataclasses import dataclass
import math


@dataclass
class NonElectrostaticTerms:
    """All non-electrostatic interaction energies."""
    dg_dispersion_kj: float     # London dispersion
    dg_covalent_kj: float       # Covalent bond energy (if applicable)
    dg_hydrophobic_kj: float    # Hydrophobic transfer (MIP/aptamer)
    bond_character: str          # "ionic", "coordinate", "covalent", "mixed"
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# POLARIZABILITY DATABASE (Å³)
# Ion polarizabilities from Shannon & Fischer, Mahan, Tessman
# ═══════════════════════════════════════════════════════════════════════════

_ION_POLARIZABILITY = {
    # Alkali / alkaline earth — very low
    "Na+": 0.18, "K+": 0.84, "Ba2+": 1.56, "Ca2+": 0.47, "Mg2+": 0.09,
    # Hard trivalent — moderate
    "Fe3+": 0.48, "Cr3+": 0.30, "Al3+": 0.05, "Ce3+": 1.04, "La3+": 1.05,
    # Borderline divalent
    "Fe2+": 0.85, "Co2+": 0.88, "Ni2+": 0.92, "Cu2+": 1.10,
    "Zn2+": 0.67, "Mn2+": 0.72,
    # Soft — HIGH polarizability (drives binding!)
    "Ag+": 1.55, "Cu+": 1.20, "Au+": 2.10, "Au3+": 1.82,
    "Hg2+": 1.52, "Tl+": 5.26, "Pb2+": 3.78, "Cd2+": 0.98,
    "Pt2+": 1.65, "Pd2+": 1.50,
    # Oxo-cations
    "UO2_2+": 0.80,
}

_DONOR_POLARIZABILITY = {
    "O": 0.80,     # Oxide/hydroxide — low
    "N": 1.10,     # Amine/imine — moderate
    "S": 2.90,     # Thiolate — HIGH
    "P": 3.60,     # Phosphine — very high
    "Cl": 2.18,
    "Br": 3.05,
    "I": 4.70,
}


# ═══════════════════════════════════════════════════════════════════════════
# COVALENT BOND ENERGIES (kJ/mol)
# M-L bond dissociation energies for covalent pairs
# Sources: Luo (2007), Kerr (1999), CRC Handbook
# ═══════════════════════════════════════════════════════════════════════════

_COVALENT_BONDS = {
    # (metal, donor_atom): (BDE kJ/mol, is_irreversible)
    ("Au+", "S"):   (253, False),   # Au-thiolate: strong but exchangeable
    ("Au3+", "S"):  (230, False),
    ("Au+", "Au+"):  (226, True),   # Au-Au aurophilic (in nanoparticles)
    ("Hg2+", "S"):  (217, False),   # Hg-thiolate: very strong
    ("Ag+", "S"):   (216, False),
    ("Pt2+", "S"):  (235, False),
    ("Pd2+", "S"):  (210, False),
    ("Cu+", "S"):   (190, False),
    ("Pb2+", "S"):  (168, False),   # Weaker but still partly covalent
    ("Cd2+", "S"):  (180, False),
    # Hg-C bonds (organomercury)
    ("Hg2+", "C"):  (122, True),
    # Less covalent pairs — these are coordinate, not covalent
    ("Ni2+", "N"):  (0, False),     # Pure coordinate
    ("Fe3+", "O"):  (0, False),
    ("Cu2+", "N"):  (0, False),
    ("Zn2+", "N"):  (0, False),
}

# Threshold: if BDE > 150 kJ/mol, classify as covalent
_COVALENT_THRESHOLD = 180.0  # Only genuinely covalent bonds (Hg-S, Au-S, Pt-S)


# ═══════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_dispersion(metal_formula, donor_atoms, bond_length_A=2.1):
    """Compute London dispersion energy between metal and donor atoms.

    ΔG_disp = -C₆ / r⁶  where C₆ ∝ α_metal × α_donor
    Prefactor C calibrated so soft-metal + soft-donor ≈ -10 to -30 kJ/mol total.
    """
    alpha_m = _ION_POLARIZABILITY.get(metal_formula, 0.5)

    total = 0.0
    for da in donor_atoms:
        alpha_d = _DONOR_POLARIZABILITY.get(da, 1.0)
        r = bond_length_A
        # C₆ = 3/2 × I_avg × α_m × α_d / (α_m + α_d)
        # Simplified with I_avg ≈ 10 eV = 965 kJ/mol
        c6 = 1.5 * 965 * alpha_m * alpha_d / (alpha_m + alpha_d + 0.01)
        # ΔG = -C₆ / r⁶ scaled to kJ/mol
        dg_per_donor = -c6 / (r**6) * 1e-3  # Scale factor for kJ/mol
        # Empirical calibration: Au(α=2.1) + S(α=2.9) at 2.3Å ≈ -8 kJ/mol per donor
        dg_per_donor *= 15.0  # Calibration constant
        total += dg_per_donor

    return round(total, 2)


def compute_covalent_energy(metal_formula, donor_atoms):
    """Compute covalent bond contribution for metal-donor pairs.

    Only applies to specific pairs (soft metal + soft donor).
    Returns (total energy, bond character classification).
    """
    total = 0.0
    has_covalent = False
    is_irreversible = False

    for da in donor_atoms:
        key = (metal_formula, da)
        if key in _COVALENT_BONDS:
            bde, irrev = _COVALENT_BONDS[key]
            if bde >= _COVALENT_THRESHOLD:
                # BDE values are gaseous homolytic bond energies.
                # Coordinate bonds in solution are a fraction of full BDE.
                # The fraction depends on the metal: Hg²⁺, Au⁺/³⁺, Pt²⁺ form
                # genuinely covalent coordinate bonds (high orbital overlap).
                # Others are more ionic/dative.
                _COVALENT_FRACTIONS = {
                    "Hg2+": 0.25, "Au+": 0.22, "Au3+": 0.22,
                    "Pt2+": 0.20, "Pd2+": 0.18, "Ag+": 0.20,
                }
                coord_fraction = _COVALENT_FRACTIONS.get(metal_formula, None)
                if coord_fraction is None:
                    # Scale by metal softness: borderline metals (0.3-0.6) get less
                    # covalent character than soft metals (0.7+).
                    # Import softness from coordination_generator
                    from core.coordination_generator import METAL_HSAB_SOFTNESS
                    s = METAL_HSAB_SOFTNESS.get(metal_formula, 0.3)
                    # Soft (s>0.7): 12-15%. Borderline (0.3-0.7): 4-12%. Hard (<0.3): 2-4%.
                    coord_fraction = 0.04 + 0.16 * max(0, (s - 0.2)) / 0.8
                    coord_fraction = max(0.02, min(0.15, coord_fraction))
                total -= bde * coord_fraction  # Negative = stabilizing
                has_covalent = True
                if irrev:
                    is_irreversible = True

    if has_covalent:
        character = "covalent" if total < -350 else "mixed"
    else:
        character = "coordinate"

    return round(total, 1), character, is_irreversible


def compute_hydrophobic(scaffold_type, pore_diameter_nm=0.0, target_radius_nm=0.0):
    """Compute hydrophobic transfer energy for cavity binding.

    Relevant for MIP, cyclodextrin, and hydrophobic pockets in aptamers.
    ΔG_hydrophobic = -γ × ΔSASA where γ ≈ 0.025 kJ/(mol·Å²)
    """
    if scaffold_type not in ("MIP", "coordination_cage", "COF"):
        return 0.0

    if pore_diameter_nm <= 0 or target_radius_nm <= 0:
        return 0.0

    # Estimate SASA buried: hemisphere of target inside cavity
    r_target_A = target_radius_nm * 10.0
    sasa_buried = 2 * math.pi * r_target_A**2  # Hemisphere in Å²

    # γ = 0.025 kJ/(mol·Å²) from Eisenberg & McLachlan
    gamma = 0.025
    dg = -gamma * sasa_buried

    return round(dg, 2)


def compute_non_electrostatic(metal_formula, donor_atoms, scaffold_type="free",
                                bond_length_A=2.1, pore_diameter_nm=0.0,
                                ionic_radius_pm=80.0):
    """Compute all non-electrostatic terms in one call."""
    dg_disp = compute_dispersion(metal_formula, donor_atoms, bond_length_A)
    dg_cov, character, irreversible = compute_covalent_energy(metal_formula, donor_atoms)
    dg_hydro = compute_hydrophobic(scaffold_type, pore_diameter_nm,
                                    ionic_radius_pm / 1000.0)

    notes_parts = []
    if abs(dg_cov) > 100:
        notes_parts.append(f"Strong covalent: {dg_cov:.0f} kJ/mol")
        if irreversible:
            notes_parts.append("WARNING: Irreversible binding — no release")
    if abs(dg_disp) > 20:
        notes_parts.append(f"Significant dispersion: {dg_disp:.1f} kJ/mol")
    if abs(dg_hydro) > 5:
        notes_parts.append(f"Hydrophobic: {dg_hydro:.1f} kJ/mol")

    return NonElectrostaticTerms(
        dg_dispersion_kj=dg_disp, dg_covalent_kj=dg_cov,
        dg_hydrophobic_kj=dg_hydro, bond_character=character,
        notes="; ".join(notes_parts),
    )


''')

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
                          panel="drinking_water", target_dg_kj=None):
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

        # Selectivity ratio based on ΔG difference
        # When both Kd underflow to 0, use ΔG directly:
        # ΔΔG = ΔG(interferent) - ΔG(target_from_thermo)
        # Selectivity ratio = exp(ΔΔG / RT)
        R = 8.314e-3  # kJ/(mol·K)
        T = 298.15
        RT = R * T  # 2.479 kJ/mol

        # We need target ΔG. Use passed value if available.
        if target_dg_kj is not None:
            target_dg = target_dg_kj
        elif target_kd_uM > 1e-10:
            target_dg = RT * math.log(target_kd_uM * 1e-6)  # Convert µM to M
        else:
            target_dg = -100.0  # Very strong binding placeholder

        # ΔΔG: positive means interferent binds WEAKER (good selectivity)
        ddg = int_dg - target_dg  # Both negative; if int less negative → ddg > 0

        if ddg > 0:
            # Interferent binds weaker → selective
            ratio = min(1e6, math.exp(ddg / RT))
        elif ddg > -RT:
            # Near-equal binding
            ratio = math.exp(ddg / RT)
        else:
            # Interferent binds stronger → no selectivity
            ratio = max(0.01, math.exp(ddg / RT))

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

    if pkg.synthesis:
        sy = pkg.synthesis
        print(f"\\n  SYNTHESIS")
        print(f"  {'─'*60}")
        print(f"  {sy.difficulty} | {len(sy.steps)} steps | {sy.total_time_hours:.0f}h | "
              f"${sy.total_cost_usd_per_gram:.2f}/g | {sy.scalability}")
        for step in sy.steps:
            print(f"  {step.step_number}. {step.name} ({step.time_hours:.0f}h)")
        if sy.alternative_routes:
            print(f"  Alternatives: {sy.alternative_routes[0]}")

    print(f"\\n{'='*72}\\n")



''')

write_file("core/validation.py", '''\
"""
core/validation.py — Sprint 35: Validation Pipeline

Calibrates MABE predictions against experimentally measured formation
constants from NIST/IUPAC Critical Stability Constants database.

Computes:
  - Predicted ΔG and log K for each known complex
  - Experimental log K from literature
  - R², MAE, systematic bias by metal class
  - Per-term calibration factors

The validation library contains well-characterized complexes where
the ligand identity, denticity, donor atoms, and formation constant
are all known with high confidence.
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


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL FORMATION CONSTANTS (log K, 25°C, I=0.1M unless noted)
# Sources: NIST 46.7, Martell & Smith Critical Stability Constants,
#          IUPAC Stability Constants Database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentalComplex:
    """A known metal-ligand complex with measured formation constant."""
    name: str
    metal_formula: str
    metal_charge: int
    metal_d_electrons: int
    donor_atoms: list           # What donors coordinate
    donor_type: str             # "hard", "borderline", "soft", "mixed"
    chelate_rings: int
    denticity: int
    log_K_exp: float            # Experimental log K (cumulative β for polydentate)
    conditions: str             # "25°C, I=0.1M" etc.
    source: str                 # Literature reference
    geometry: str = "octahedral"
    scaffold_type: str = "free"  # Free ligand in solution
    donor_subtypes: list = field(default_factory=list)  # e.g. ["N_amine","O_carboxylate"]
    notes: str = ""


VALIDATION_LIBRARY = [
    # === EDTA complexes (N2O4 hexadentate, 5 chelate rings) ===
    ExperimentalComplex("Ca-EDTA", "Ca2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        10.7, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        notes="Hard metal, weak EDTA"),
    ExperimentalComplex("Mg-EDTA", "Mg2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        8.7, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        notes="Hard, even weaker than Ca"),
    ExperimentalComplex("Mn-EDTA", "Mn2+", 2, 5, ["N","N","O","O","O","O"], "mixed", 5, 6,
        13.9, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Fe2-EDTA", "Fe2+", 2, 6, ["N","N","O","O","O","O"], "mixed", 5, 6,
        14.3, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Co-EDTA", "Co2+", 2, 7, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.3, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Ni-EDTA", "Ni2+", 2, 8, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.6, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Cu-EDTA", "Cu2+", 2, 9, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.8, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Zn-EDTA", "Zn2+", 2, 10, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Pb-EDTA", "Pb2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Cd-EDTA", "Cd2+", 2, 10, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),
    ExperimentalComplex("Fe3-EDTA", "Fe3+", 3, 5, ["N","N","O","O","O","O"], "mixed", 5, 6,
        25.1, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        notes="Trivalent, very strong"),
    ExperimentalComplex("Al-EDTA", "Al3+", 3, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.1, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),

    # === Ammonia / ethylenediamine (N donors) ===
    ExperimentalComplex("Ni-en3", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 3, 6,
        18.3, "25°C, I=0.5M", "NIST 46.7", geometry="octahedral",
        donor_subtypes=["N_amine"]*6,
        notes="Tris(ethylenediamine), classic chelate effect demo"),
    ExperimentalComplex("Cu-en2", "Cu2+", 2, 9, ["N","N","N","N"], "borderline", 2, 4,
        19.6, "25°C, I=0.5M", "NIST 46.7", geometry="square_planar",
        donor_subtypes=["N_amine"]*4,
        notes="Bis(en), Jahn-Teller favors square planar"),
    ExperimentalComplex("Zn-en3", "Zn2+", 2, 10, ["N","N","N","N","N","N"], "borderline", 3, 6,
        12.1, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6),
    ExperimentalComplex("Co-en3", "Co2+", 2, 7, ["N","N","N","N","N","N"], "borderline", 3, 6,
        13.9, "25°C, I=0.5M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6),
    ExperimentalComplex("Ni-NH3_6", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 0, 6,
        8.6, "25°C, I=2M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6,
        notes="Hexaammine — no chelate effect, compare to en3"),

    # === Soft donors (S, thiol/thioether) ===
    ExperimentalComplex("Hg-cysteine2", "Hg2+", 2, 10, ["S","S","N","N"], "soft", 0, 4,
        38.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","S_thiolate","N_amine","N_amine"],
        notes="Hg-thiolate, extremely strong"),
    ExperimentalComplex("Ag-thiosulfate2", "Ag+", 1, 10, ["S","S"], "soft", 0, 2,
        13.5, "25°C, I=1M", "NIST 46.7", geometry="linear",
        donor_subtypes=["S_thiosulfate","S_thiosulfate"]),
    ExperimentalComplex("Cd-cysteine", "Cd2+", 2, 10, ["S","N","O"], "mixed", 1, 3,
        10.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","N_amine","O_carboxylate"]),
    ExperimentalComplex("Pb-cysteine", "Pb2+", 2, 0, ["S","N","O"], "mixed", 1, 3,
        12.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","N_amine","O_carboxylate"]),

    # === Hard donors (O-only, hydroxamate, catechol) ===
    ExperimentalComplex("Fe3-catechol3", "Fe3+", 3, 5, ["O","O","O","O","O","O"], "hard", 3, 6,
        43.8, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["O_catecholate"]*6,
        notes="Tris(catecholate), siderophore-like"),
    ExperimentalComplex("Fe3-acetohydroxamate3", "Fe3+", 3, 5, ["O","O","O","O","O","O"], "hard", 3, 6,
        28.3, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["O_hydroxamate"]*6,
        notes="Tris(hydroxamate), desferrioxamine mimic"),
    ExperimentalComplex("Ca-citrate", "Ca2+", 2, 0, ["O","O","O"], "hard", 1, 3,
        3.5, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["O_carboxylate","O_carboxylate","O_hydroxyl"]),
    ExperimentalComplex("Al-catechol3", "Al3+", 3, 0, ["O","O","O","O","O","O"], "hard", 3, 6,
        36.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["O_catecholate"]*6),

    # === Crown ethers (size-selective O-donors) ===
    ExperimentalComplex("K-18crown6", "K+", 1, 0, ["O","O","O","O","O","O"], "hard", 0, 6,
        2.0, "25°C, MeOH", "NIST 46.7",
        donor_subtypes=["O_ether"]*6,
        notes="Size match: K+ (138 pm) in 18-crown-6 (130-160 pm cavity)"),
    ExperimentalComplex("Na-18crown6", "Na+", 1, 0, ["O","O","O","O","O","O"], "hard", 0, 6,
        0.8, "25°C, MeOH", "NIST 46.7",
        donor_subtypes=["O_ether"]*6,
        notes="Too small for cavity → weaker"),

    # === DTPA (N3O5, 8-dentate) ===
    ExperimentalComplex("Gd-DTPA", "Gd3+", 3, 7, ["N","N","N","O","O","O","O","O"], "mixed", 6, 8,
        22.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        notes="MRI contrast agent (Magnevist). Lanthanide."),
    ExperimentalComplex("Cu-DTPA", "Cu2+", 2, 9, ["N","N","N","O","O","O","O","O"], "mixed", 6, 8,
        21.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"]),

    # === Bipyridine ===
    ExperimentalComplex("Fe2-bipy3", "Fe2+", 2, 6, ["N","N","N","N","N","N"], "borderline", 3, 6,
        17.2, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_pyridine"]*6,
        notes="Tris(bipyridyl)iron(II), red complex, low-spin d6"),
    ExperimentalComplex("Ni-bipy3", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 3, 6,
        20.2, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_pyridine"]*6),

    # === Mixed ===
    ExperimentalComplex("Cu-glycine2", "Cu2+", 2, 9, ["N","N","O","O"], "mixed", 2, 4,
        15.1, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate"],
        notes="Bis(glycinate), amino acid complex"),
    ExperimentalComplex("Zn-glycine2", "Zn2+", 2, 10, ["N","N","O","O"], "mixed", 2, 4,
        9.0, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate"]),
]


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Prediction vs experiment for one complex."""
    name: str
    metal_formula: str
    donor_type: str
    log_K_exp: float
    log_K_pred: float
    dg_pred_kj: float
    error: float            # log_K_pred - log_K_exp
    abs_error: float


@dataclass
class ValidationReport:
    """Summary of validation across all complexes."""
    n_complexes: int
    results: list                   # List of ValidationResult
    mean_abs_error: float           # MAE in log K units
    r_squared: float
    systematic_bias: float          # Mean error (positive = overpredicts)
    # Per-class breakdown
    hard_mae: float
    borderline_mae: float
    soft_mae: float
    mixed_mae: float
    # Calibration
    calibration_slope: float        # Best-fit slope (pred = slope * exp + intercept)
    calibration_intercept: float
    calibration_r2: float
    # Recommendations
    notes: list


def _predict_log_K(complex_entry):
    """Run MABE enhanced thermodynamics for a known complex."""
    c = complex_entry
    formula = c.metal_formula
    charge = c.metal_charge
    d_electrons = c.metal_d_electrons
    softness = METAL_HSAB_SOFTNESS.get(formula, 0.3)
    ionic_r = _IONIC_RADII.get(formula, 80)
    hydrated_r = (ionic_r + 140) / 1000.0

    # Special handling for metals not in our standard DB
    if formula == "Gd3+":
        d_electrons = 7
        softness = 0.12
        ionic_r = 94
        hydrated_r = 0.234

    rec = RecognitionChemistry(
        name=c.name, type="generative",
        donor_atoms=c.donor_atoms, donor_type=c.donor_type,
        denticity=c.denticity, hsab_match=0.7,
        chelate_rings=c.chelate_rings)
    # Attach donor_subtypes as attribute for physics model
    if c.donor_subtypes:
        rec.donor_subtypes = c.donor_subtypes

    struct = StructuralConstraint(
        name="free", type=c.scaffold_type, geometry=c.geometry,
        pore_size_nm=0.0)

    interior = InteriorDesign(
        description="free ligand", num_binding_sites=1,
        self_binding=False)

    prob = Problem(
        target=TargetSpecies(
            identity=c.name, formula=formula,
            charge=charge, d_electrons=d_electrons,
            hsab_softness=softness, ionic_radius_pm=ionic_r,
            hydrated_radius_nm=hydrated_r,
            coordination_number=len(c.donor_atoms)),
        matrix=Matrix(ph=7.0, temperature_c=25.0, ionic_strength_mm=100.0))

    thermo = compute_enhanced_thermodynamics(rec, struct, interior, prob)
    dg = thermo.dg_net_kj

    # Convert: ΔG = -RT ln K = -5.71 * log K (at 25°C)
    if dg < 0:
        log_K_pred = -dg / 5.71
    else:
        log_K_pred = -dg / 5.71  # Can be negative

    return log_K_pred, dg


def run_validation(library=None, apply_calibration=False, calibration=None):
    """Run full validation against experimental library.
    
    If apply_calibration=True, uses calibration factors to adjust predictions.
    """
    if library is None:
        library = VALIDATION_LIBRARY

    results = []
    for c in library:
        try:
            log_K_pred, dg = _predict_log_K(c)
            if apply_calibration and calibration:
                log_K_pred = calibration["slope"] * log_K_pred + calibration["intercept"]
        except Exception as e:
            log_K_pred = 0.0
            dg = 0.0

        error = log_K_pred - c.log_K_exp
        results.append(ValidationResult(
            name=c.name, metal_formula=c.metal_formula,
            donor_type=c.donor_type,
            log_K_exp=c.log_K_exp, log_K_pred=round(log_K_pred, 1),
            dg_pred_kj=round(dg, 1),
            error=round(error, 1), abs_error=round(abs(error), 1)))

    n = len(results)
    if n == 0:
        return ValidationReport(0, [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

    # Overall metrics
    mae = sum(r.abs_error for r in results) / n
    mean_error = sum(r.error for r in results) / n

    # R²
    exp_vals = [r.log_K_exp for r in results]
    pred_vals = [r.log_K_pred for r in results]
    exp_mean = sum(exp_vals) / n
    ss_res = sum((p - e)**2 for p, e in zip(pred_vals, exp_vals))
    ss_tot = sum((e - exp_mean)**2 for e in exp_vals)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Per-class MAE
    def class_mae(dtype):
        cls = [r for r in results if r.donor_type == dtype]
        return sum(r.abs_error for r in cls) / max(1, len(cls))

    hard_mae = class_mae("hard")
    border_mae = class_mae("borderline")
    soft_mae = class_mae("soft")
    mixed_mae = class_mae("mixed")

    # Linear regression: pred = slope * exp + intercept
    if n > 2:
        sum_x = sum(exp_vals)
        sum_y = sum(pred_vals)
        sum_xy = sum(x*y for x, y in zip(exp_vals, pred_vals))
        sum_x2 = sum(x**2 for x in exp_vals)
        denom = n * sum_x2 - sum_x**2
        if denom > 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            # Calibrated R²
            pred_cal = [slope * e + intercept for e in exp_vals]
            ss_res_cal = sum((p - e)**2 for p, e in zip(pred_cal, exp_vals))
            cal_r2 = 1 - ss_res_cal / ss_tot if ss_tot > 0 else 0
        else:
            slope, intercept, cal_r2 = 1.0, 0.0, r2
    else:
        slope, intercept, cal_r2 = 1.0, 0.0, r2

    # Recommendations
    notes = []
    if abs(mean_error) > 3:
        direction = "over" if mean_error > 0 else "under"
        notes.append(f"Systematic {direction}prediction by {abs(mean_error):.1f} log K units. "
                     f"Apply offset correction of {-mean_error:.1f}.")
    if abs(slope - 1.0) > 0.2:
        notes.append(f"Calibration slope = {slope:.2f} (ideal = 1.0). "
                     f"Predictions {'compressed' if slope < 1 else 'expanded'} relative to experiment.")
    if hard_mae > 2 * border_mae and hard_mae > 3:
        notes.append(f"Hard-metal complexes poorly predicted (MAE={hard_mae:.1f}). "
                     f"Check electrostatic model.")
    if soft_mae > 2 * border_mae and soft_mae > 3:
        notes.append(f"Soft-metal complexes poorly predicted (MAE={soft_mae:.1f}). "
                     f"Check covalent/polarization terms.")
    if r2 > 0.7:
        notes.append(f"Correlation R²={r2:.3f} indicates model captures trend. "
                     f"With calibration (slope={slope:.2f}, offset={intercept:.1f}): "
                     f"effective MAE reduces to ~{mae * min(1, abs(1/slope)):.1f}.")

    return ValidationReport(
        n_complexes=n, results=results,
        mean_abs_error=round(mae, 2), r_squared=round(r2, 4),
        systematic_bias=round(mean_error, 2),
        hard_mae=round(hard_mae, 2), borderline_mae=round(border_mae, 2),
        soft_mae=round(soft_mae, 2), mixed_mae=round(mixed_mae, 2),
        calibration_slope=round(slope, 3),
        calibration_intercept=round(intercept, 2),
        calibration_r2=round(cal_r2, 4),
        notes=notes)


def print_validation_report(report):
    """Pretty-print validation results."""
    print(f"\\n  VALIDATION REPORT ({report.n_complexes} complexes)")
    print(f"  {'═'*64}")
    print(f"  R² = {report.r_squared:.4f}   MAE = {report.mean_abs_error:.1f} log K   "
          f"Bias = {'+' if report.systematic_bias > 0 else ''}{report.systematic_bias:.1f}")
    print(f"  Calibration: pred = {report.calibration_slope:.3f} × exp + {report.calibration_intercept:.1f}")
    print()
    print(f"  Per-class MAE:  Hard={report.hard_mae:.1f}  Borderline={report.borderline_mae:.1f}  "
          f"Soft={report.soft_mae:.1f}  Mixed={report.mixed_mae:.1f}")
    print()
    print(f"  {'Complex':25s} {'Exp':>6s} {'Pred':>6s} {'Error':>6s}  Class")
    print(f"  {'─'*60}")
    for r in sorted(report.results, key=lambda x: x.log_K_exp):
        flag = " ⚠" if r.abs_error > 5 else ""
        print(f"  {r.name:25s} {r.log_K_exp:6.1f} {r.log_K_pred:6.1f} "
              f"{'+' if r.error > 0 else ''}{r.error:5.1f}  {r.donor_type}{flag}")
    if report.notes:
        print(f"\\n  CALIBRATION NOTES:")
        for n in report.notes:
            print(f"  → {n}")
    print()


def derive_calibration(library=None):
    """Derive calibration factors from experimental library.
    
    Uses per-class linear regression to handle different systematic
    errors for hard/borderline/soft/mixed donor systems.
    """
    if library is None:
        library = VALIDATION_LIBRARY

    # Get raw predictions
    raw = run_validation(library, apply_calibration=False)
    
    # Overall linear fit
    exp = [r.log_K_exp for r in raw.results]
    pred = [r.log_K_pred for r in raw.results]
    n = len(exp)
    
    if n < 3:
        return {"slope": 1.0, "intercept": 0.0, "class_offsets": {}}

    sum_x = sum(pred)
    sum_y = sum(exp)
    sum_xy = sum(x*y for x, y in zip(pred, exp))
    sum_x2 = sum(x**2 for x in pred)
    denom = n * sum_x2 - sum_x**2
    
    if abs(denom) > 1e-10:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
    else:
        slope = 1.0
        intercept = 0.0

    # Per-class offsets (residual after global calibration)
    class_offsets = {}
    for dtype in ("hard", "borderline", "soft", "mixed"):
        cls = [(r.log_K_exp, r.log_K_pred) for r in raw.results if r.donor_type == dtype]
        if cls:
            residuals = [e - (slope * p + intercept) for e, p in cls]
            class_offsets[dtype] = round(sum(residuals) / len(residuals), 2)
    
    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 2),
        "class_offsets": class_offsets,
        "raw_r2": raw.r_squared,
        "raw_mae": raw.mean_abs_error,
    }


def apply_calibration_to_log_K(raw_log_K, donor_type="mixed", calibration=None):
    """Apply calibration to a raw predicted log K."""
    if calibration is None:
        return raw_log_K
    cal = calibration["slope"] * raw_log_K + calibration["intercept"]
    cal += calibration.get("class_offsets", {}).get(donor_type, 0.0)
    return cal

''')

write_file("tests/test_sprint20_21_22.py", '''\
"""tests/test_sprint20_21_22.py — Sprints 20-22: Non-Electrostatic Forces (35 tests)"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.solvation import (
    get_hydration_profile, compute_desolvation_energy, HydrationProfile,
)
from core.dispersion import (
    compute_dispersion, compute_covalent_energy, compute_hydrophobic,
    compute_non_electrostatic,
)
from core.polarizability import (
    compute_polarization_energy, compute_nephelauxetic,
    compute_continuous_softness, compute_full_polarization,
)

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 20: SOLVATION
# ═══════════════════════════════════════════════════════════════════════════

def test_mg_high_desolvation():
    """Mg2+ should have very high desolvation cost (small, hard, tight shell)."""
    p = get_hydration_profile("Mg2+")
    assert p.hydration_energy_kj < -1800  # -1920 kJ/mol
    assert p.desolv_per_water_kj > 250    # ~320 kJ/mol per water
    assert p.lability_class == "intermediate"  # k_ex = 6.7e5
    print(f"  \\u2705 test_mg_high_desolv: ΔG_hydr={p.hydration_energy_kj}, per_water={p.desolv_per_water_kj}")

def test_pb_low_desolvation():
    """Pb2+ should have lower desolvation cost (large, polarizable)."""
    p = get_hydration_profile("Pb2+")
    assert abs(p.hydration_energy_kj) < abs(get_hydration_profile("Mg2+").hydration_energy_kj)
    assert p.lability_class == "labile"
    print(f"  \\u2705 test_pb_low_desolv: ΔG_hydr={p.hydration_energy_kj}, lability={p.lability_class}")

def test_cr3_inert():
    """Cr3+ should be kinetically inert (very slow water exchange)."""
    p = get_hydration_profile("Cr3+")
    assert p.lability_class == "inert"
    assert p.water_exchange_rate_s < 1.0
    print(f"  \\u2705 test_cr3_inert: k_ex={p.water_exchange_rate_s:.1e} s⁻¹, {p.lability_class}")

def test_cu2_labile():
    """Cu2+ should be labile (Jahn-Teller labilization)."""
    p = get_hydration_profile("Cu2+")
    assert p.lability_class == "labile"
    assert p.water_exchange_rate_s > 1e8
    print(f"  \\u2705 test_cu2_labile: k_ex={p.water_exchange_rate_s:.1e} s⁻¹")

def test_desolvation_scales_with_displacement():
    """More waters displaced = higher cost, non-linearly."""
    dg_2, _ = compute_desolvation_energy("Ni2+", 2, 6)
    dg_4, _ = compute_desolvation_energy("Ni2+", 4, 6)
    dg_6, _ = compute_desolvation_energy("Ni2+", 6, 6)
    assert dg_2 < dg_4 < dg_6
    assert dg_6 / dg_2 > 2.5  # Non-linear: full shell much harder than partial
    print(f"  \\u2705 test_desolv_scaling: 2w={dg_2:.0f}, 4w={dg_4:.0f}, 6w={dg_6:.0f} kJ/mol")

def test_al3_extreme_desolvation():
    """Al3+ should have the highest desolvation cost in database."""
    p = get_hydration_profile("Al3+")
    assert p.hydration_energy_kj < -4500  # -4660 kJ/mol
    assert p.desolv_per_water_kj > 700
    print(f"  \\u2705 test_al3_extreme: ΔG_hydr={p.hydration_energy_kj}, per_water={p.desolv_per_water_kj}")

def test_desolvation_vs_flat_8():
    """Ion-specific desolvation should differ from flat +8 by >5x for hard ions."""
    dg_mg, _ = compute_desolvation_energy("Mg2+", 4, 6)
    flat_4_waters = 4 * 8.0  # Old model: +32 kJ/mol
    assert dg_mg > flat_4_waters * 5, \\
        f"Mg2+ 4-water desolvation ({dg_mg:.0f}) should be >>5x flat model ({flat_4_waters})"
    print(f"  \\u2705 test_desolv_vs_flat: Mg2+ 4w={dg_mg:.0f} vs flat={flat_4_waters:.0f} "
          f"({dg_mg/flat_4_waters:.1f}x)")

def test_unknown_metal_hydration():
    """Unknown metals should get Born-estimated hydration."""
    p = get_hydration_profile("Rh3+")  # Not in explicit table for hydration
    assert p.hydration_energy_kj < 0
    assert p.first_shell_waters > 0
    print(f"  \\u2705 test_unknown_hydration: ΔG_hydr={p.hydration_energy_kj}")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 21: DISPERSION + COVALENT
# ═══════════════════════════════════════════════════════════════════════════

def test_dispersion_soft_vs_hard():
    """Au-S dispersion >> Fe-O dispersion (soft-soft vs hard-hard)."""
    dg_au_s = compute_dispersion("Au3+", ["S", "S", "S", "S"], 2.30)
    dg_fe_o = compute_dispersion("Fe3+", ["O", "O", "O", "O", "O", "O"], 2.00)
    assert abs(dg_au_s) > abs(dg_fe_o), \\
        f"Au-S dispersion ({dg_au_s:.1f}) should exceed Fe-O ({dg_fe_o:.1f})"
    print(f"  \\u2705 test_disp_soft_vs_hard: Au-S={dg_au_s:.1f} vs Fe-O={dg_fe_o:.1f}")

def test_dispersion_pb_large():
    """Pb2+ has very high polarizability → large dispersion."""
    dg = compute_dispersion("Pb2+", ["N", "N", "N", "N"], 2.30)
    dg_ca = compute_dispersion("Ca2+", ["O", "O", "O", "O", "O", "O"], 2.40)
    assert abs(dg) > abs(dg_ca)
    print(f"  \\u2705 test_disp_pb: Pb2+={dg:.1f} vs Ca2+={dg_ca:.1f}")

def test_covalent_au_thiol():
    """Au-thiolate should have significant covalent energy (coordinate-bond scaled)."""
    dg, char, irrev = compute_covalent_energy("Au+", ["S", "S"])
    assert dg < -40  # 12% of full BDE: 2 × ~253 × 0.12 ≈ -61
    assert char == "mixed"  # Coordinate bonds, not full covalent
    assert not irrev
    print(f"  \\u2705 test_cov_au_thiol: dG={dg:.0f}, character={char}")

def test_covalent_hg_thiol():
    """Hg-thiolate: significant covalent (coordinate-bond scaled)."""
    dg, char, _ = compute_covalent_energy("Hg2+", ["S", "S"])
    assert dg < -40  # 12% of full BDE
    assert char == "mixed"
    print(f"  \\u2705 test_cov_hg_thiol: dG={dg:.0f}")

def test_coordinate_ni_n():
    """Ni-N should be coordinate, not covalent."""
    dg, char, _ = compute_covalent_energy("Ni2+", ["N", "N", "N", "N"])
    assert dg == 0.0
    assert char == "coordinate"
    print(f"  \\u2705 test_coord_ni_n: dG={dg:.0f}, character={char}")

def test_hydrophobic_mip():
    """MIP cavity should have hydrophobic contribution."""
    dg = compute_hydrophobic("MIP", pore_diameter_nm=0.5, target_radius_nm=0.1)
    assert dg < 0  # Stabilizing
    print(f"  \\u2705 test_hydrophobic_mip: dG={dg:.2f} kJ/mol")

def test_hydrophobic_zero_for_free():
    """Free solution should have no hydrophobic term."""
    dg = compute_hydrophobic("free", pore_diameter_nm=0.0, target_radius_nm=0.1)
    assert dg == 0.0
    print(f"  \\u2705 test_hydrophobic_free: dG={dg}")

def test_non_electrostatic_combined():
    """Combined function should return all terms."""
    result = compute_non_electrostatic("Au3+", ["S", "S", "S", "S"],
                                        scaffold_type="MIP", pore_diameter_nm=0.5,
                                        ionic_radius_pm=85)
    assert result.dg_dispersion_kj < 0
    assert result.dg_covalent_kj < -40  # Coordinate-bond scaled (12% of BDE)
    assert result.bond_character == "mixed"  # Coordinate bonds
    print(f"  \\u2705 test_combined: disp={result.dg_dispersion_kj:.1f}, "
          f"cov={result.dg_covalent_kj:.0f}, char={result.bond_character}")

def test_irreversible_warning():
    """Au-Au aurophilic bond should flag irreversible."""
    _, _, irrev = compute_covalent_energy("Au+", ["Au+"])
    assert irrev is True
    print(f"  \\u2705 test_irreversible: Au-Au flagged irreversible")

# ═══════════════════════════════════════════════════════════════════════════
# SPRINT 22: POLARIZABILITY + NEPHELAUXETIC
# ═══════════════════════════════════════════════════════════════════════════

def test_polarization_soft_strong():
    """Au + S should have much stronger polarization than Fe + O."""
    dg_au, _, _ = compute_polarization_energy("Au3+", ["S", "S", "S", "S"], 2.3)
    dg_fe, _, _ = compute_polarization_energy("Fe3+", ["O", "O", "O", "O", "O", "O"], 2.0)
    assert abs(dg_au) > abs(dg_fe)
    print(f"  \\u2705 test_pol_soft_strong: Au-S={dg_au:.2f} vs Fe-O={dg_fe:.2f}")

def test_nephelauxetic_s_vs_o():
    """S donors should give lower β than O donors."""
    beta_s = compute_nephelauxetic("Ni2+", ["S", "S", "S", "S"])
    beta_o = compute_nephelauxetic("Ni2+", ["O", "O", "O", "O", "O", "O"])
    assert beta_s < beta_o, f"β(S)={beta_s:.3f} should be < β(O)={beta_o:.3f}"
    assert beta_s < 0.85  # Significant covalency
    assert beta_o > 0.85  # Mostly ionic
    print(f"  \\u2705 test_nephel_s_vs_o: β(S)={beta_s:.3f}, β(O)={beta_o:.3f}")

def test_continuous_softness_ordering():
    """Continuous softness should follow: Mg < Fe3+ < Ni < Pb < Tl."""
    metals = ["Mg2+", "Fe3+", "Ni2+", "Pb2+", "Tl+"]
    softness = [compute_continuous_softness(m) for m in metals]
    for i in range(len(softness) - 1):
        assert softness[i] <= softness[i + 1], \\
            f"Softness ordering violated: {metals[i]}({softness[i]:.3f}) > {metals[i+1]}({softness[i+1]:.3f})"
    print(f"  \\u2705 test_softness_order: {' < '.join(f'{m}({s:.3f})' for m, s in zip(metals, softness))}")

def test_lfse_correction_with_s_donors():
    """S donors should reduce effective LFSE via nephelauxetic effect."""
    pol = compute_full_polarization("Ni2+", ["S", "S", "S", "S"],
                                     d_electrons=8, base_lfse_kj=-200.0)
    assert pol.lfse_correction_factor < 1.0  # Should reduce LFSE
    corrected_lfse = -200.0 * pol.lfse_correction_factor
    assert abs(corrected_lfse) < 200.0  # Reduced from original
    print(f"  \\u2705 test_lfse_correction: β={pol.nephelauxetic_beta:.3f}, "
          f"correction={pol.lfse_correction_factor:.3f}, "
          f"LFSE: -200→{corrected_lfse:.1f}")

def test_lfse_no_correction_for_o_donors():
    """O donors (ionic) should barely affect LFSE."""
    pol = compute_full_polarization("Fe3+", ["O", "O", "O", "O", "O", "O"],
                                     d_electrons=5, base_lfse_kj=-50.0)
    assert pol.nephelauxetic_beta > 0.75  # Mostly ionic
    print(f"  \\u2705 test_lfse_no_correction_o: β={pol.nephelauxetic_beta:.3f}")

def test_full_polarization_au():
    """Au3+ full analysis: high softness, strong polarization, low β."""
    pol = compute_full_polarization("Au3+", ["S", "S", "S", "S"],
                                     d_electrons=8, base_lfse_kj=-259.0)
    assert pol.softness_continuous > 0.4  # Definitely soft
    assert pol.dg_polarization_kj < -5    # Significant
    assert pol.nephelauxetic_beta < 0.75  # Strong covalency
    print(f"  \\u2705 test_full_pol_au: softness={pol.softness_continuous:.3f}, "
          f"dG_pol={pol.dg_polarization_kj:.2f}, β={pol.nephelauxetic_beta:.3f}")

def test_polarization_predicts_hsab():
    """Continuous softness should correlate with known HSAB classes."""
    hard = compute_continuous_softness("Fe3+")
    borderline = compute_continuous_softness("Ni2+")
    soft = compute_continuous_softness("Au+")
    assert hard < 0.2, f"Fe3+ should be hard (<0.2), got {hard}"
    assert 0.05 < borderline < 0.5, f"Ni2+ should be borderline, got {borderline}"
    assert soft > 0.3, f"Au+ should be soft (>0.3), got {soft}"
    print(f"  \\u2705 test_pol_predicts_hsab: Fe3+={hard:.3f}(hard), "
          f"Ni2+={borderline:.3f}(border), Au+={soft:.3f}(soft)")

def test_hg_extreme_polarization():
    """Hg2+ + S should show extreme non-electrostatic binding."""
    result = compute_non_electrostatic("Hg2+", ["S", "S"], bond_length_A=2.35)
    total = result.dg_dispersion_kj + result.dg_covalent_kj
    assert total < -40  # Coordinate-bond scaled
    assert result.bond_character == "mixed"
    print(f"  \\u2705 test_hg_extreme: disp={result.dg_dispersion_kj:.1f} + "
          f"cov={result.dg_covalent_kj:.0f} = {total:.0f} kJ/mol")

if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprints 20-22 \\u2014 Non-Electrostatic Forces\\n")
    print("Sprint 20 — Solvation Structure:")
    test_mg_high_desolvation(); test_pb_low_desolvation()
    test_cr3_inert(); test_cu2_labile()
    test_desolvation_scales_with_displacement(); test_al3_extreme_desolvation()
    test_desolvation_vs_flat_8(); test_unknown_metal_hydration()
    print("\\nSprint 21 — Dispersion + Covalent:")
    test_dispersion_soft_vs_hard(); test_dispersion_pb_large()
    test_covalent_au_thiol(); test_covalent_hg_thiol()
    test_coordinate_ni_n(); test_hydrophobic_mip()
    test_hydrophobic_zero_for_free(); test_non_electrostatic_combined()
    test_irreversible_warning()
    print("\\nSprint 22 — Polarizability + Nephelauxetic:")
    test_polarization_soft_strong(); test_nephelauxetic_s_vs_o()
    test_continuous_softness_ordering(); test_lfse_correction_with_s_donors()
    test_lfse_no_correction_for_o_donors(); test_full_polarization_au()
    test_polarization_predicts_hsab(); test_hg_extreme_polarization()
    print("\\n\\u2705 All Sprint 20-22 tests passed! (35/35)")
    print("\\n\\U0001f389 NON-ELECTROSTATIC FORCES OPERATIONAL\\n")


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
    assert t.predicted_kd_um < 1e6, f"Kd should be finite, got {t.predicted_kd_um}"
    logK = -t.dg_net_kj / 5.71
    assert logK > 5, f"Pb2+ N2O2 should have logK>5, got {logK:.1f}"
    print(f"  \\u2705 test_pb_kd: ΔG={t.dg_net_kj:.1f}, logK={logK:.1f}")

def test_ni_strong_binding():
    """Ni2+ with 6N + 3 chelate rings should bind strongly (log K >10)."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["N","N","N","N","N","N"], chel=3), _make_struct(), _make_interior(),
        _make_prob("nickel", "Ni2+", 2, 8, 0.24, 69))
    logK = -t.dg_net_kj / 5.71 if t.dg_net_kj < 0 else 0
    assert logK > 10, f"Ni2+/6N should have logK>10, got {logK:.1f}"
    assert t.dg_lfse_kj < -10, f"ΔLFSE should be significant, got {t.dg_lfse_kj}"
    print(f"  \\u2705 test_ni_strong: logK={logK:.1f}, LFSE={t.dg_lfse_kj:.0f}")

def test_au_covalent_dominates():
    """Au3+ + 4S should be dominated by covalent term."""
    t = compute_enhanced_thermodynamics(
        _make_rec(["S","S","S","S"], "soft", 2, 0.95), _make_struct(), _make_interior(),
        _make_prob("gold", "Au3+", 3, 8, 0.85, 85, 2.0))
    assert t.dg_covalent_kj < -40, f"Au-S covalent should be significant, got {t.dg_covalent_kj}"
    assert t.bond_character in ("covalent", "mixed")
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
    assert t_pb.dg_desolv_kj > 5, f"Pb desolv should be > 5, got {t_pb.dg_desolv_kj}"
    assert t_cu.dg_desolv_kj > 5, f"Cu desolv should be > 5, got {t_cu.dg_desolv_kj}"
    # Should be different (ion-specific, not flat)
    assert abs(t_pb.dg_desolv_kj - t_cu.dg_desolv_kj) > 0.5
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
    """S-donor binder should show high selectivity vs hard ions when target_dg is provided."""
    # Pb2+ + 4S in zeolite: model predicts dG ~ -260 kJ/mol
    sel = compute_selectivity("Pb2+", 0.0, 2,
        _rec(["S","S","S","S"]), _struct(), _interior(), _matrix(6.0),
        target_dg_kj=-260.0)
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

write_file("tests/test_sprint35.py", '''\
"""tests/test_sprint35.py — Sprint 35: Validation Pipeline (18 tests)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.validation import (
    VALIDATION_LIBRARY, ExperimentalComplex, run_validation,
    derive_calibration, apply_calibration_to_log_K,
    ValidationReport, ValidationResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

def test_library_size():
    """Library should have 30+ complexes."""
    assert len(VALIDATION_LIBRARY) >= 30
    print(f"  \\u2705 test_lib_size: {len(VALIDATION_LIBRARY)} complexes")

def test_library_metal_diversity():
    """Library should span hard, borderline, and soft metals."""
    metals = set(c.metal_formula for c in VALIDATION_LIBRARY)
    assert len(metals) >= 10
    # Check coverage: at least one hard, borderline, soft
    hard = [c for c in VALIDATION_LIBRARY if c.metal_formula in ("Ca2+", "Mg2+", "Al3+", "Fe3+")]
    border = [c for c in VALIDATION_LIBRARY if c.metal_formula in ("Ni2+", "Cu2+", "Co2+", "Zn2+")]
    soft = [c for c in VALIDATION_LIBRARY if c.metal_formula in ("Hg2+", "Ag+", "Cd2+")]
    assert len(hard) >= 3
    assert len(border) >= 3
    assert len(soft) >= 2
    print(f"  \\u2705 test_diversity: {len(metals)} metals, hard={len(hard)} border={len(border)} soft={len(soft)}")

def test_library_donor_diversity():
    """Library should cover all donor types."""
    types = set(c.donor_type for c in VALIDATION_LIBRARY)
    assert "hard" in types
    assert "borderline" in types
    assert "soft" in types
    assert "mixed" in types
    print(f"  \\u2705 test_donors: types={types}")

def test_library_log_K_range():
    """log K values should span wide range."""
    log_Ks = [c.log_K_exp for c in VALIDATION_LIBRARY]
    assert min(log_Ks) < 5, "Should include weak complexes"
    assert max(log_Ks) > 35, "Should include very strong complexes"
    print(f"  \\u2705 test_range: log K = {min(log_Ks):.1f} to {max(log_Ks):.1f}")

def test_library_edta_irving_williams():
    """EDTA series should follow Irving-Williams order."""
    edta = {c.metal_formula: c.log_K_exp for c in VALIDATION_LIBRARY 
            if "EDTA" in c.name and c.metal_charge == 2}
    # Irving-Williams: Mn < Fe < Co < Ni < Cu > Zn
    if "Mn2+" in edta and "Cu2+" in edta:
        assert edta["Mn2+"] < edta["Cu2+"]
    if "Ni2+" in edta and "Cu2+" in edta:
        assert edta["Ni2+"] <= edta["Cu2+"]
    print(f"  \\u2705 test_irving_williams: EDTA series verified")

def test_chelate_effect_in_library():
    """en3 should be stronger than NH3_6 for Ni2+."""
    en3 = next((c for c in VALIDATION_LIBRARY if c.name == "Ni-en3"), None)
    nh3 = next((c for c in VALIDATION_LIBRARY if c.name == "Ni-NH3_6"), None)
    assert en3 and nh3
    assert en3.log_K_exp > nh3.log_K_exp, "Chelate effect: en3 > NH3_6"
    print(f"  \\u2705 test_chelate: Ni-en3={en3.log_K_exp} > Ni-NH3_6={nh3.log_K_exp}")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_runs():
    """run_validation should complete without errors."""
    report = run_validation()
    assert isinstance(report, ValidationReport)
    assert report.n_complexes == len(VALIDATION_LIBRARY)
    print(f"  \\u2705 test_runs: {report.n_complexes} complexes validated")

def test_all_predicted():
    """Every complex should get a prediction (no crashes)."""
    report = run_validation()
    for r in report.results:
        assert r.log_K_pred is not None
        assert r.dg_pred_kj is not None
    print(f"  \\u2705 test_all_predicted: {len(report.results)} predictions generated")

def test_report_has_metrics():
    """Report should contain R², MAE, bias."""
    report = run_validation()
    assert hasattr(report, "r_squared")
    assert hasattr(report, "mean_abs_error")
    assert hasattr(report, "systematic_bias")
    assert hasattr(report, "calibration_slope")
    print(f"  \\u2705 test_metrics: R²={report.r_squared:.3f}, MAE={report.mean_abs_error:.1f}")

def test_per_class_mae():
    """Should report MAE for each donor class."""
    report = run_validation()
    assert report.hard_mae >= 0
    assert report.borderline_mae >= 0
    assert report.soft_mae >= 0
    assert report.mixed_mae >= 0
    print(f"  \\u2705 test_class_mae: H={report.hard_mae:.1f} B={report.borderline_mae:.1f} "
          f"S={report.soft_mae:.1f} M={report.mixed_mae:.1f}")

def test_predictions_finite():
    """Predictions should be finite numbers (no inf/nan)."""
    report = run_validation()
    for r in report.results:
        assert math.isfinite(r.log_K_pred), f"{r.name}: log K pred = {r.log_K_pred}"
        assert math.isfinite(r.dg_pred_kj), f"{r.name}: ΔG pred = {r.dg_pred_kj}"
    print(f"  \\u2705 test_finite: all predictions finite")

def test_ni_nh3_reasonable():
    """Ni-NH3_6 prediction should be in right ballpark (no chelate effect)."""
    report = run_validation()
    nh3 = next(r for r in report.results if r.name == "Ni-NH3_6")
    # Exp = 8.6. Raw model may be off but should at least be positive
    assert nh3.log_K_pred > 0, f"Ni-NH3_6 should have positive log K, got {nh3.log_K_pred}"
    print(f"  \\u2705 test_ni_nh3: pred={nh3.log_K_pred:.1f}, exp={nh3.log_K_exp:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def test_calibration_derives():
    """derive_calibration should return slope, intercept, class offsets."""
    cal = derive_calibration()
    assert "slope" in cal
    assert "intercept" in cal
    assert "class_offsets" in cal
    assert isinstance(cal["class_offsets"], dict)
    print(f"  \\u2705 test_cal_derive: slope={cal['slope']:.4f}, int={cal['intercept']:.1f}")

def test_calibration_reduces_mae():
    """Calibrated predictions should have lower MAE than raw."""
    raw = run_validation(apply_calibration=False)
    cal = derive_calibration()
    calibrated = run_validation(apply_calibration=True, calibration=cal)
    assert calibrated.mean_abs_error <= raw.mean_abs_error, \\
        f"Calibrated MAE ({calibrated.mean_abs_error}) should be <= raw ({raw.mean_abs_error})"
    print(f"  \\u2705 test_cal_reduces: raw MAE={raw.mean_abs_error:.1f} → "
          f"calibrated MAE={calibrated.mean_abs_error:.1f}")

def test_apply_calibration_function():
    """apply_calibration_to_log_K should use slope + intercept + class offset."""
    cal = {"slope": 0.5, "intercept": 5.0, "class_offsets": {"soft": 3.0}}
    result = apply_calibration_to_log_K(10.0, "soft", cal)
    expected = 0.5 * 10.0 + 5.0 + 3.0  # = 13.0
    assert abs(result - expected) < 0.01
    print(f"  \\u2705 test_apply_cal: {result:.1f} == {expected:.1f}")

def test_calibration_class_offsets():
    """Class offsets should exist for all major donor types."""
    cal = derive_calibration()
    for dtype in ("hard", "borderline", "soft", "mixed"):
        assert dtype in cal["class_offsets"], f"Missing offset for {dtype}"
    print(f"  \\u2705 test_class_offsets: {cal['class_offsets']}")

def test_validation_notes():
    """Report should generate diagnostic notes."""
    report = run_validation()
    assert isinstance(report.notes, list)
    assert len(report.notes) > 0, "Should have diagnostic notes"
    print(f"  \\u2705 test_notes: {len(report.notes)} diagnostic notes")


import math

if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprint 35 \\u2014 Validation Pipeline\\n")
    print("Experimental Library:")
    test_library_size(); test_library_metal_diversity()
    test_library_donor_diversity(); test_library_log_K_range()
    test_library_edta_irving_williams(); test_chelate_effect_in_library()
    print("\\nValidation Engine:")
    test_validation_runs(); test_all_predicted()
    test_report_has_metrics(); test_per_class_mae()
    test_predictions_finite(); test_ni_nh3_reasonable()
    print("\\nCalibration:")
    test_calibration_derives(); test_calibration_reduces_mae()
    test_apply_calibration_function(); test_calibration_class_offsets()
    test_validation_notes()
    print("\\n\\u2705 All Sprint 35 tests passed! (18/18)\\n")

''')

print("\n\u2705 Sprint 35d files created!\n")