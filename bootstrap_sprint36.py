"""
MABE Sprint 36: Deep Physics Calibration
5 physics improvements from coordination chemistry literature:
  1. Metal-specific ΔG_hyd desolvation (Marcus 1991 / Kepp 2019)
  2. Macrocyclic + cavity size-match effect
  3. Chelate ring size correction (Hancock rule)
  4. Electrostatic z·z for anionic donors (Coulombic)
  5. HSAB multiplicative scaling (replaces additive penalty)

Requires Sprint 35d.
"""
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n\U0001f528 MABE Sprint 36 — Deep Physics Calibration\n")

# ═══════════════════════════════════════════════════════════════════════════
# FILE 1: core/physics_integration.py  (REWRITE — 5 term upgrades)
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/physics_integration.py", '''\
"""
core/physics_integration.py — Sprint 36: Deep Physics Calibration

Builds on Sprint 35d's 15-term ΔG model with 5 targeted improvements
derived from coordination chemistry literature:

  1. Metal-specific ΔG_hyd desolvation (Marcus 1991 / Kepp 2019)
     — Replaces charge-dependent scaling with ion-specific hydration
       free energies. Automatically encodes Irving-Williams, size effects.
  2. Macrocyclic + cavity size-match effect
     — New preorganization term for crown ethers, cryptands, cages.
  3. Chelate ring size correction (Hancock rule)
     — 5-membered rings strain-free; 6-membered penalize large metals.
  4. Electrostatic z·z for anionic donors (Coulombic)
     — Proper charge-charge stabilization for trivalent + carboxylate.
  5. HSAB multiplicative scaling
     — Replaces additive match/mismatch with multiplicative factor on
       per-donor exchange energy.

All other terms (3, 6-15, translational, Jahn-Teller) unchanged from 35d.
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


# ═══════════════════════════════════════════════════════════════════════════
# HYDRATION FREE ENERGY TABLE (kJ/mol, Marcus 1991 + Kepp 2019)
# Negative values = energy released upon hydration.
# Used for desolvation: cost to remove each water from inner sphere.
# ═══════════════════════════════════════════════════════════════════════════

_HYDRATION_FREE_ENERGY = {
    # Monovalent (CN_aqua = 4-6)
    "Li+":   -481,   "Na+":  -365,   "K+":    -295,   "Rb+":  -275,
    "Cs+":   -250,   "Ag+":  -430,   "Cu+":   -520,   "Tl+":  -310,
    # Divalent (CN_aqua = 6 typical)
    "Mg2+": -1830,   "Ca2+": -1505,  "Sr2+": -1380,   "Ba2+": -1250,
    "Mn2+": -1760,   "Fe2+": -1840,  "Co2+": -1915,   "Ni2+": -1980,
    "Cu2+": -2010,   "Zn2+": -1955,  "Cd2+": -1755,   "Pb2+": -1425,
    "Hg2+": -1760,   "Pd2+": -1910,  "Pt2+": -1960,   "UO2_2+": -1350,
    # Trivalent (CN_aqua = 6 typical)
    "Fe3+": -4265,   "Al3+": -4525,  "Cr3+": -4010,   "Co3+": -4495,
    "Ga3+": -4515,   "In3+": -3980,  "Tl3+": -4015,   "Au3+": -4420,
    "La3+": -3145,   "Ce3+": -3200,  "Gd3+": -3425,   "Lu3+": -3545,
    "Bi3+": -3480,
}

# Aqua coordination numbers (first shell)
_AQUA_CN = {
    "Li+": 4, "Na+": 6, "K+": 6, "Rb+": 8, "Cs+": 8,
    "Ag+": 4, "Cu+": 4, "Tl+": 6,
    "Mg2+": 6, "Ca2+": 6, "Sr2+": 8, "Ba2+": 8,
    "Mn2+": 6, "Fe2+": 6, "Co2+": 6, "Ni2+": 6,
    "Cu2+": 6, "Zn2+": 6, "Cd2+": 6, "Pb2+": 6,
    "Hg2+": 6, "Pd2+": 4, "Pt2+": 4, "UO2_2+": 5,
    "Fe3+": 6, "Al3+": 6, "Cr3+": 6, "Co3+": 6,
    "Ga3+": 6, "In3+": 6, "Tl3+": 6, "Au3+": 4,
    "La3+": 9, "Ce3+": 9, "Gd3+": 8, "Lu3+": 8,
    "Bi3+": 8,
}


@dataclass
class EnhancedThermodynamics(BindingThermodynamics):
    """Extended thermodynamics with all physics terms."""
    # New terms from Sprints 18-29
    dg_dispersion_kj: float = 0.0
    dg_covalent_kj: float = 0.0
    dg_polarization_kj: float = 0.0
    dg_hydrophobic_kj: float = 0.0
    dg_relativistic_correction_kj: float = 0.0
    # Sprint 36 additions
    dg_macrocyclic_kj: float = 0.0
    dg_ring_strain_kj: float = 0.0
    dg_zz_electrostatic_kj: float = 0.0
    # Metadata
    nephelauxetic_beta: float = 1.0
    spin_state: str = ""
    softness_continuous: float = 0.0
    bond_character: str = "coordinate"
    speciation_warning: str = ""
    free_ion_fraction: float = 1.0
    design_strategy: str = "free_ion_binding"


def compute_enhanced_thermodynamics(recognition, structure, interior, problem):
    """Full physics thermodynamics — Sprint 36 calibrated.

    18-term ΔG:
    1.  dG_bind (HSAB multiplicative donor exchange)       [S36: HSAB mult]
    2.  dG_desolv (metal-specific ΔG_hyd)                  [S36: Marcus table]
    3.  dG_preorg (scaffold rigidity)
    3b. dG_macrocyclic (crown/cryptand/cage preorg)        [S36: NEW]
    4.  dG_chelate (ring-size-aware)                        [S36: Hancock rule]
    5.  dG_electrostatic (z·z Coulombic for anionic donors) [S36: Coulombic]
    6.  dG_protonation (pH-dependent)
    7.  dG_LFSE (spin-state aware)
    8.  dG_activity (Davies)
    9.  dG_screening (Debye)
    10. dG_repulsion (steric)
    11. dG_dispersion (London)
    12. dG_covalent (BDE)
    13. dG_polarization (induced dipole)
    14. dG_hydrophobic (cavity)
    15. dG_relativistic (6s contraction)
    16. dG_translational (entropy)
    17. dG_jahn_teller (d9/d4)
    18. dG_ring_strain (6-membered ring penalty)            [S36: NEW]
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

    # ═══════════════════════════════════════════════════════════════════════
    # Term 1: dG_bind — HSAB MULTIPLICATIVE donor exchange  [S36 UPGRADE]
    # ═══════════════════════════════════════════════════════════════════════
    # Each donor contributes a base exchange energy (subtype-resolved).
    # The HSAB match/mismatch MULTIPLIES this energy rather than adding
    # a separate penalty. This means strong donors amplify selectivity.
    #
    # f_HSAB: 0.3 (hard-soft mismatch) → 1.0 (neutral) → 1.5 (perfect match)
    # Physical basis: orbital overlap integral scales with hardness match.

    pol_result = compute_full_polarization(formula, donors, d_electrons=d_electrons)
    softness = target_softness

    donor_softness_map = {"O": 0.15, "N": 0.40, "S": 0.80, "P": 0.75,
                          "Cl": 0.50, "Br": 0.65, "I": 0.85}

    _SUBTYPE_EXCHANGE = {
        "O_ether": -1.0, "O_hydroxyl": -5.0, "O_carboxylate": -4.0,
        "O_phenolate": -10.0, "O_hydroxamate": -18.0, "O_catecholate": -30.0,
        "N_amine": -6.0, "N_imine": -8.0, "N_pyridine": -7.0, "N_imidazole": -8.0,
        "S_thioether": -10.0, "S_thiosulfate": -6.0, "S_thiolate": -18.0,
        "S_dithiocarbamate": -10.0,
    }
    _ELEMENT_EXCHANGE = {"O": -4.0, "N": -6.0, "S": -18.0, "P": -15.0,
                         "Cl": -3.0, "Br": -6.0, "I": -12.0}

    donor_subtypes = getattr(recognition, 'donor_subtypes', None)
    if not donor_subtypes or len(donor_subtypes) != len(donors):
        donor_subtypes = None

    donor_energies = []
    for i, da in enumerate(donors):
        ds = donor_softness_map.get(da, 0.3)

        # Base exchange energy (subtype-resolved)
        if donor_subtypes:
            dg_exchange = _SUBTYPE_EXCHANGE.get(donor_subtypes[i],
                            _ELEMENT_EXCHANGE.get(da, -8.0))
        else:
            dg_exchange = _ELEMENT_EXCHANGE.get(da, -8.0)

        # Charge scaling: z² dependence (electrostatic field strength)
        charge_factor = -5.0 * (charge**2 - 1) / max(1, len(donors))
        dg_exchange += charge_factor

        # HSAB MULTIPLICATIVE FACTOR [S36]
        # mismatch ∈ [0, ~0.7]: 0 = perfect match, 0.7 = extreme mismatch
        # f_HSAB: perfect match → 1.5, neutral (0.3) → 1.0, extreme → 0.3
        mismatch = abs(softness - ds)
        if mismatch < 0.15:
            # Good match: amplify binding (up to 1.5×)
            f_hsab = 1.0 + 0.5 * max(softness, ds) * (1.0 - mismatch / 0.15)
        elif mismatch < 0.35:
            # Neutral zone: ~1.0×
            f_hsab = 1.0
        else:
            # Mismatch: attenuate binding (down to 0.3× for extreme)
            # Steeper penalty for larger mismatch
            f_hsab = max(0.3, 1.0 - 1.5 * (mismatch - 0.35))

        dg_exchange *= f_hsab

        donor_energies.append(dg_exchange)
    dg_bind = sum(donor_energies)

    # ═══════════════════════════════════════════════════════════════════════
    # Term 2: dG_desolv — METAL-SPECIFIC ΔG_hyd  [S36 UPGRADE]
    # ═══════════════════════════════════════════════════════════════════════
    # The desolvation penalty is: cost to remove n waters from the inner
    # coordination sphere. Using Marcus/Kepp hydration free energies:
    #   ΔG_desolv = f_exchange × |ΔG_hyd(M)| / CN_aqua × n_waters_displaced
    #
    # f_exchange is the FRACTION of per-water hydration energy that must be
    # paid when a ligand replaces water. For strong ligands (better than
    # water), f_exchange is small (2-5%). For weak ligands (worse than
    # water, e.g. ether oxygens), f_exchange is larger (5-10%).
    #
    # This automatically encodes:
    #   - Irving-Williams order (Cu²⁺ > Ni²⁺ > Co²⁺ etc.)
    #   - Trivalent penalty (Fe³⁺ hydration 2× divalent)
    #   - Size effects (small ions have tighter water shells)

    waters_displaced = min(len(donors), cn)

    dg_hyd = _HYDRATION_FREE_ENERGY.get(formula, None)
    cn_aqua = _AQUA_CN.get(formula, 6)

    if dg_hyd is not None:
        per_water_kj = abs(dg_hyd) / max(1, cn_aqua)

        # Base exchange fraction: how much of per-water hydration energy
        # must be paid as a net penalty when swapping water → ligand.
        # Lower charge → weaker M-L bonds → higher net penalty.
        # Monovalent ~4%, divalent ~2.5%, trivalent ~1.5%.
        if charge >= 3:
            base_f = 0.015
        elif charge == 2:
            base_f = 0.025
        else:
            base_f = 0.04

        # HSAB match: well-matched donors replace water more efficiently
        if softness > 0.5 and any(d in ("S", "P", "I") for d in donors):
            base_f *= 0.7    # Soft-soft: efficient exchange
        elif softness < 0.2 and all(d == "O" for d in donors):
            base_f *= 0.7    # Hard-hard: efficient exchange

        dg_desolv = per_water_kj * waters_displaced * base_f

        # Lability correction (from solvation module)
        hydration = get_hydration_profile(formula)
        if hydration:
            if hydration.lability_class == "inert":
                dg_desolv *= 1.3   # Kinetic barrier adds effective cost
            elif hydration.lability_class == "labile":
                dg_desolv *= 0.85  # Easy exchange reduces cost
    else:
        # Fallback: use solvation module's hydration profile
        hydration = get_hydration_profile(formula)
        if hydration:
            per_water_kj = abs(hydration.hydration_energy_kj) / max(1, hydration.first_shell_waters)
            base_f = 0.015 * (2.0 / max(1, charge))
            if softness > 0.5 and any(d in ("S", "P", "I") for d in donors):
                base_f *= 0.7
            elif softness < 0.2 and all(d == "O" for d in donors):
                base_f *= 0.7
            dg_desolv = per_water_kj * waters_displaced * base_f
            if hydration.lability_class == "inert":
                dg_desolv *= 1.3
            elif hydration.lability_class == "labile":
                dg_desolv *= 0.85
        else:
            dg_desolv = waters_displaced * 5.0

    # === Term 3: dG_preorg (unchanged from 35d) ===
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

    # ═══════════════════════════════════════════════════════════════════════
    # Term 3b: dG_macrocyclic — CROWN/CRYPTAND/CAGE  [S36 NEW]
    # ═══════════════════════════════════════════════════════════════════════
    # Macrocyclic effect: preorganized cyclic ligand gains entropy (no
    # conformational penalty) and enthalpy (pre-positioned donors).
    # Crown ethers: +5 to +15 kJ/mol total (mostly entropic).
    # Tetra-aza macrocycles: +10 to +30 kJ/mol.
    # Cryptands (3D cage): additional +5 to +20 kJ/mol.
    #
    # Cavity size-match: Gaussian selectivity function.
    #   K-18crown6 (r_ion=138pm, r_cavity=134pm) → near-perfect match
    #   Na-18crown6 (r_ion=102pm, r_cavity=134pm) → mismatch → weaker
    #
    # Fields read from recognition object (attached as attributes):
    #   is_macrocyclic: bool
    #   cavity_radius_nm: float (0 if unknown)
    #   is_cage: bool (cryptand/cage → extra stabilization)

    is_macrocyclic = getattr(recognition, 'is_macrocyclic', False)
    cavity_radius_nm = getattr(recognition, 'cavity_radius_nm', 0.0)
    is_cage = getattr(recognition, 'is_cage', False)

    dg_macrocyclic = 0.0
    if is_macrocyclic:
        n_donor_atoms = len(donors)
        # Base macrocyclic stabilization: -1.5 kJ/mol per donor atom
        # O-donors (crown): ~-1.0/donor. N-donors (aza-macro): ~-2.5/donor.
        n_O = sum(1 for d in donors if d == "O")
        n_N = sum(1 for d in donors if d == "N")
        n_other = n_donor_atoms - n_O - n_N
        dg_macro_base = -(n_O * 1.2 + n_N * 2.5 + n_other * 1.5)

        # Cavity size-match: Gaussian selectivity
        if cavity_radius_nm > 0 and ionic_radius_pm > 0:
            r_ion_nm = ionic_radius_pm / 1000.0
            # sigma: width of size-match window (~0.03 nm for crowns)
            sigma = 0.03
            size_match = math.exp(-((r_ion_nm - cavity_radius_nm)**2) / (2 * sigma**2))
            # Size match scales the macrocyclic bonus: perfect match → full bonus
            # Large mismatch → bonus reduced to ~20% (still some preorg benefit)
            dg_macrocyclic = dg_macro_base * (0.2 + 0.8 * size_match)
        else:
            # No cavity info: apply 60% of base (average match)
            dg_macrocyclic = dg_macro_base * 0.6

        # Cryptate/cage effect: additional stabilization from 3D encapsulation
        if is_cage:
            dg_macrocyclic *= 1.6  # ~60% additional for full encapsulation

    # ═══════════════════════════════════════════════════════════════════════
    # Term 4: dG_chelate — RING-SIZE CORRECTED  [S36 UPGRADE]
    # ═══════════════════════════════════════════════════════════════════════
    # Base chelate effect: TΔS for ring closure ~8-15 kJ/mol per ring.
    # Hancock rule: 5-membered rings are strain-free for all metals.
    # 6-membered rings favor SMALL metals (r < 0.065 nm).
    # Large metals in 6-membered rings pay an enthalpic strain penalty.
    #
    # ΔG_ring_strain(6-mem) = k_strain × (r_ion - 0.065)  for r > 0.065 nm
    #   k_strain ≈ 80 kJ/mol/nm (from Hancock 1989 data)
    #   Cd²⁺ (0.095 nm): ~2.4 kJ/mol penalty per 6-mem ring
    #   Pb²⁺ (0.119 nm): ~4.3 kJ/mol penalty per 6-mem ring
    #   Cu²⁺ (0.073 nm): ~0.6 kJ/mol penalty (small)

    chelate_base = -12.0 if d_electrons > 0 else -8.0
    if charge == 1:
        chelate_base *= 0.5

    n_chelate = recognition.chelate_rings if recognition else 0
    ring_sizes = getattr(recognition, 'ring_sizes', None)

    dg_chelate = 0.0
    dg_ring_strain = 0.0

    if ring_sizes and len(ring_sizes) == n_chelate:
        # Per-ring calculation with size correction
        r_ion_nm = ionic_radius_pm / 1000.0
        k_strain = 80.0  # kJ/mol/nm — from Hancock empirical fit
        for rs in ring_sizes:
            dg_chelate += chelate_base  # Base chelate effect per ring
            if rs >= 6 and r_ion_nm > 0.065:
                # 6-membered ring penalty for large metals
                penalty = k_strain * (r_ion_nm - 0.065)
                dg_ring_strain += penalty
    else:
        # No ring size info: assume all 5-membered (standard)
        dg_chelate = chelate_base * n_chelate

    # ═══════════════════════════════════════════════════════════════════════
    # Term 5: dG_electrostatic — z·z COULOMBIC  [S36 UPGRADE]
    # ═══════════════════════════════════════════════════════════════════════
    # For anionic donors (carboxylate COO⁻, catecholate O⁻, hydroxamate,
    # thiolate S⁻), there is a direct Coulombic attraction to the metal.
    # This is MUCH larger for trivalent metals than divalent:
    #   Fe³⁺-COO⁻ electrostatic >> Cu²⁺-COO⁻
    #
    # ΔG_zz = -k_elec × z_metal × |z_donor| / (r_ion + r_donor)
    #   k_elec = 138.9 kJ·pm/mol (Coulomb constant in solution, ε_eff ≈ 10)
    #   r_donor: O⁻ ≈ 140 pm, S⁻ ≈ 184 pm
    #
    # The old term (-2.5 × |z| × |z_L| / n_donors) severely undercounted
    # this for trivalent metals.

    _DONOR_CHARGE = {
        # Anionic donors carry formal -1 charge
        "O_carboxylate": -1, "O_phenolate": -1, "O_catecholate": -1,
        "O_hydroxamate": -1, "O_hydroxyl": 0,  "O_ether": 0,
        "S_thiolate": -1, "S_thiosulfate": -1, "S_thioether": 0,
        "S_dithiocarbamate": -1,
        "N_amine": 0, "N_imine": 0, "N_pyridine": 0, "N_imidazole": 0,
    }
    _DONOR_CHARGE_ELEMENT = {"O": -1, "S": -1, "N": 0, "P": 0,
                              "Cl": -1, "Br": -1, "I": -1}
    _DONOR_RADIUS_PM = {"O": 140, "S": 184, "N": 155, "P": 195,
                         "Cl": 181, "Br": 196, "I": 220}

    # Effective dielectric for inner-sphere: ~10 (between vacuum=1 and water=78)
    # k_elec = 1389.4 / ε_eff  (Coulomb's law in kJ·pm/mol)
    k_elec = 1389.4 / 10.0   # = 138.94

    dg_zz = 0.0
    for i, da in enumerate(donors):
        if donor_subtypes:
            z_donor = _DONOR_CHARGE.get(donor_subtypes[i],
                        _DONOR_CHARGE_ELEMENT.get(da, 0))
        else:
            z_donor = _DONOR_CHARGE_ELEMENT.get(da, 0)

        if z_donor < 0:
            r_d = _DONOR_RADIUS_PM.get(da, 140)
            # Coulombic: ΔG = -k × z_M × |z_D| / (r_M + r_D)
            dg_zz -= k_elec * charge * abs(z_donor) / (ionic_radius_pm + r_d)

    # The old flat electrostatic term is replaced
    dg_electrostatic = dg_zz

    # === Term 6: dG_protonation (unchanged from 35d) ===
    pka_defaults = {"O": 4.5, "S": 8.3, "P": 6.5}
    n_pka = 6.0 if recognition.type == "generative" else 10.5
    dg_protonation = 0.0
    for da in donors:
        pka = n_pka if da == "N" else pka_defaults.get(da, 7.0)
        if ph < pka:
            penalty = 5.7 * (pka - ph)
            dg_protonation += min(penalty, 20.0)

    # === Term 7: dG_LFSE (unchanged from 35d) ===
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
            if cn >= 6:
                lfse_ligand = compute_lfse_for_geometry(formula, d_electrons, geom, ligand_names)
                lfse_water = compute_lfse_for_geometry(formula, d_electrons, "octahedral",
                                                         ["water"] * 6)
                dg_lfse = -(abs(lfse_ligand.lfse_kj) - abs(lfse_water.lfse_kj))
                dg_lfse = max(-80.0, min(0.0, dg_lfse))
                spin_state_str = lfse_ligand.spin_state
            elif cn == 4:
                lfse_ligand = compute_lfse_for_geometry(formula, d_electrons, geom, ligand_names)
                lfse_water = compute_lfse_for_geometry(formula, d_electrons, "octahedral",
                                                         ["water"] * 6)
                dg_lfse = -(abs(lfse_ligand.lfse_kj) - abs(lfse_water.lfse_kj))
                dg_lfse = max(-80.0, min(0.0, dg_lfse)) * 0.67
                spin_state_str = lfse_ligand.spin_state
        except Exception:
            dg_lfse = 0.0

    # Apply nephelauxetic correction
    pol = compute_full_polarization(formula, donors, d_electrons=d_electrons,
                                     base_lfse_kj=dg_lfse)
    if abs(dg_lfse) > 1.0:
        dg_lfse *= pol.lfse_correction_factor

    # Jahn-Teller (unchanged from 35d)
    dg_jahn_teller = 0.0
    if d_electrons == 9 and cn == 4:
        dg_jahn_teller = -25.0
    elif d_electrons == 9 and cn >= 5:
        dg_jahn_teller = -10.0
    elif d_electrons == 4 and cn >= 5:
        dg_jahn_teller = -8.0

    # === Terms 8-15 (unchanged from 35d) ===
    I = ionic_mm / 1000.0
    dg_activity = 0.0
    if I > 0 and charge != 0:
        sqrt_I = math.sqrt(I)
        log_gamma = -0.509 * charge**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)
        dg_activity = -5.71 * log_gamma

    dg_screening = 0.0
    if I > 0:
        sqrt_I = math.sqrt(I)
        kappa = 3.29 * sqrt_I
        screening_factor = math.exp(-kappa * 0.3)
        dg_screening = dg_electrostatic * (1 - screening_factor)

    dg_repulsion = 0.0
    if structure and pore_nm > 0:
        if hydrated_radius_nm * 2 > pore_nm * 0.9:
            squeeze = (hydrated_radius_nm * 2 - pore_nm * 0.9)
            dg_repulsion = 10.0 * max(0, squeeze / 0.1)

    ne = compute_non_electrostatic(formula, donors, scaffold_type=scaffold_type,
                                    pore_diameter_nm=pore_nm,
                                    ionic_radius_pm=ionic_radius_pm)
    dg_dispersion = ne.dg_dispersion_kj
    dg_covalent = ne.dg_covalent_kj
    dg_hydrophobic = ne.dg_hydrophobic_kj
    dg_polarization = pol.dg_polarization_kj
    pol_softness = pol_result.softness_continuous if hasattr(pol_result, 'softness_continuous') else 0.5
    if pol_softness > 0.01:
        softness_ratio = min(1.0, target_softness / pol_softness)
        dg_polarization *= softness_ratio

    subtotal_binding = dg_bind + dg_covalent + dg_polarization + dg_dispersion
    corrected, rel_factor = correct_binding_energy(subtotal_binding, formula)
    dg_relativistic = corrected - subtotal_binding

    effective_fraction = max(0.01, free_ion_frac) if strategy == "free_ion_binding" else \\
                         max(0.01, spec.bindable_fraction)

    # Translational entropy (unchanged)
    n_donors = len(donors)
    if n_chelate == 0:
        n_ligand_molecules = n_donors
    elif n_chelate >= n_donors - 1:
        n_ligand_molecules = 1
    else:
        donors_per_molecule = max(2, (n_donors + n_chelate) // max(1, n_chelate))
        n_ligand_molecules = max(1, n_donors // donors_per_molecule)
    dg_translational = 5.5 * n_ligand_molecules

    # === Sum (now includes macrocyclic + ring_strain) ===
    dg_net = (dg_bind + dg_desolv + dg_preorg + dg_macrocyclic
              + dg_chelate + dg_ring_strain
              + dg_electrostatic + dg_protonation + dg_lfse
              + dg_activity + dg_screening + dg_repulsion
              + dg_dispersion + dg_covalent + dg_polarization + dg_hydrophobic
              + dg_relativistic + dg_translational + dg_jahn_teller)

    R = 8.314e-3
    kd_m = math.exp(dg_net / (R * temp_k)) if dg_net / (R * temp_k) < 500 else 1e6
    kd_um = kd_m * 1e6 / effective_fraction

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
        dg_dispersion_kj=round(dg_dispersion, 2),
        dg_covalent_kj=round(dg_covalent, 2),
        dg_polarization_kj=round(dg_polarization, 2),
        dg_hydrophobic_kj=round(dg_hydrophobic, 2),
        dg_relativistic_correction_kj=round(dg_relativistic, 2),
        dg_macrocyclic_kj=round(dg_macrocyclic, 2),
        dg_ring_strain_kj=round(dg_ring_strain, 2),
        dg_zz_electrostatic_kj=round(dg_zz, 2),
        nephelauxetic_beta=pol.nephelauxetic_beta,
        spin_state=spin_state_str,
        softness_continuous=pol.softness_continuous,
        bond_character=ne.bond_character,
        speciation_warning=spec_warning,
        free_ion_fraction=free_ion_frac,
        design_strategy=strategy,
    )



''')


# ═══════════════════════════════════════════════════════════════════════════
# FILE 2: core/validation.py  (schema additions + annotated crown entries)
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/validation.py", '''\
"""
core/validation.py — Sprint 36: Validation with Deep Physics

Extended validation library with Sprint 36 schema additions:
  - ring_sizes: List[int] per chelate ring (5 or 6)
  - is_macrocyclic: bool
  - cavity_radius_nm: float (crown/cryptand cavity)
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
class ExperimentalComplex:
    """A known metal-ligand complex with measured formation constant."""
    name: str
    metal_formula: str
    metal_charge: int
    metal_d_electrons: int
    donor_atoms: list
    donor_type: str
    chelate_rings: int
    denticity: int
    log_K_exp: float
    conditions: str
    source: str
    geometry: str = "octahedral"
    scaffold_type: str = "free"
    donor_subtypes: list = field(default_factory=list)
    # Sprint 36 schema additions
    ring_sizes: list = field(default_factory=list)    # e.g. [5,5,5] for EDTA
    is_macrocyclic: bool = False
    cavity_radius_nm: float = 0.0                     # Crown/cryptand cavity radius
    is_cage: bool = False                              # 3D encapsulation (cryptand)
    notes: str = ""


VALIDATION_LIBRARY = [
    # === EDTA complexes (N2O4 hexadentate, 5 chelate rings, all 5-membered) ===
    ExperimentalComplex("Ca-EDTA", "Ca2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        10.7, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5], notes="Hard metal, weak EDTA"),
    ExperimentalComplex("Mg-EDTA", "Mg2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        8.7, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5], notes="Hard, even weaker than Ca"),
    ExperimentalComplex("Mn-EDTA", "Mn2+", 2, 5, ["N","N","O","O","O","O"], "mixed", 5, 6,
        13.9, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Fe2-EDTA", "Fe2+", 2, 6, ["N","N","O","O","O","O"], "mixed", 5, 6,
        14.3, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Co-EDTA", "Co2+", 2, 7, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.3, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Ni-EDTA", "Ni2+", 2, 8, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.6, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Cu-EDTA", "Cu2+", 2, 9, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.8, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Zn-EDTA", "Zn2+", 2, 10, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Pb-EDTA", "Pb2+", 2, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        18.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Cd-EDTA", "Cd2+", 2, 10, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),
    ExperimentalComplex("Fe3-EDTA", "Fe3+", 3, 5, ["N","N","O","O","O","O"], "mixed", 5, 6,
        25.1, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5], notes="Trivalent, very strong"),
    ExperimentalComplex("Al-EDTA", "Al3+", 3, 0, ["N","N","O","O","O","O"], "mixed", 5, 6,
        16.1, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5]),

    # === Ammonia / ethylenediamine (N donors, 5-membered rings) ===
    ExperimentalComplex("Ni-en3", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 3, 6,
        18.3, "25°C, I=0.5M", "NIST 46.7", geometry="octahedral",
        donor_subtypes=["N_amine"]*6, ring_sizes=[5,5,5],
        notes="Tris(ethylenediamine)"),
    ExperimentalComplex("Cu-en2", "Cu2+", 2, 9, ["N","N","N","N"], "borderline", 2, 4,
        19.6, "25°C, I=0.5M", "NIST 46.7", geometry="square_planar",
        donor_subtypes=["N_amine"]*4, ring_sizes=[5,5],
        notes="Bis(en), Jahn-Teller"),
    ExperimentalComplex("Zn-en3", "Zn2+", 2, 10, ["N","N","N","N","N","N"], "borderline", 3, 6,
        12.1, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6, ring_sizes=[5,5,5]),
    ExperimentalComplex("Co-en3", "Co2+", 2, 7, ["N","N","N","N","N","N"], "borderline", 3, 6,
        13.9, "25°C, I=0.5M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6, ring_sizes=[5,5,5]),
    ExperimentalComplex("Ni-NH3_6", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 0, 6,
        8.6, "25°C, I=2M", "NIST 46.7",
        donor_subtypes=["N_amine"]*6,
        notes="Hexaammine — no chelate rings"),

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
        donor_subtypes=["S_thiolate","N_amine","O_carboxylate"],
        ring_sizes=[5]),
    ExperimentalComplex("Pb-cysteine", "Pb2+", 2, 0, ["S","N","O"], "mixed", 1, 3,
        12.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["S_thiolate","N_amine","O_carboxylate"],
        ring_sizes=[5]),

    # === Hard donors (O-only, hydroxamate, catechol) ===
    ExperimentalComplex("Fe3-catechol3", "Fe3+", 3, 5, ["O","O","O","O","O","O"], "hard", 3, 6,
        43.8, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["O_catecholate"]*6, ring_sizes=[5,5,5],
        notes="Tris(catecholate), siderophore-like"),
    ExperimentalComplex("Fe3-acetohydroxamate3", "Fe3+", 3, 5, ["O","O","O","O","O","O"], "hard", 3, 6,
        28.3, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["O_hydroxamate"]*6, ring_sizes=[5,5,5],
        notes="Tris(hydroxamate)"),
    ExperimentalComplex("Ca-citrate", "Ca2+", 2, 0, ["O","O","O"], "hard", 1, 3,
        3.5, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["O_carboxylate","O_carboxylate","O_hydroxyl"],
        ring_sizes=[5]),
    ExperimentalComplex("Al-catechol3", "Al3+", 3, 0, ["O","O","O","O","O","O"], "hard", 3, 6,
        36.0, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["O_catecholate"]*6, ring_sizes=[5,5,5]),

    # === Crown ethers (MACROCYCLIC, O-donors, cavity size-match) ===
    ExperimentalComplex("K-18crown6", "K+", 1, 0, ["O","O","O","O","O","O"], "hard", 0, 6,
        2.0, "25°C, MeOH", "NIST 46.7",
        donor_subtypes=["O_ether"]*6,
        is_macrocyclic=True, cavity_radius_nm=0.134,
        notes="Size match: K+ (138 pm) ≈ 18C6 cavity (134 pm)"),
    ExperimentalComplex("Na-18crown6", "Na+", 1, 0, ["O","O","O","O","O","O"], "hard", 0, 6,
        0.8, "25°C, MeOH", "NIST 46.7",
        donor_subtypes=["O_ether"]*6,
        is_macrocyclic=True, cavity_radius_nm=0.134,
        notes="Na+ (102 pm) too small for 18C6 cavity"),

    # === DTPA (N3O5, 8-dentate, all 5-membered rings) ===
    ExperimentalComplex("Gd-DTPA", "Gd3+", 3, 7, ["N","N","N","O","O","O","O","O"], "mixed", 6, 8,
        22.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5,5], notes="MRI contrast agent"),
    ExperimentalComplex("Cu-DTPA", "Cu2+", 2, 9, ["N","N","N","O","O","O","O","O"], "mixed", 6, 8,
        21.5, "25°C, I=0.1M", "Martell & Smith",
        donor_subtypes=["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5,5,5,5,5]),

    # === Bipyridine (5-membered chelate rings) ===
    ExperimentalComplex("Fe2-bipy3", "Fe2+", 2, 6, ["N","N","N","N","N","N"], "borderline", 3, 6,
        17.2, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_pyridine"]*6, ring_sizes=[5,5,5],
        notes="Tris(bipyridyl)iron(II), low-spin d6"),
    ExperimentalComplex("Ni-bipy3", "Ni2+", 2, 8, ["N","N","N","N","N","N"], "borderline", 3, 6,
        20.2, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_pyridine"]*6, ring_sizes=[5,5,5]),

    # === Mixed (glycinate, 5-membered rings) ===
    ExperimentalComplex("Cu-glycine2", "Cu2+", 2, 9, ["N","N","O","O"], "mixed", 2, 4,
        15.1, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5], notes="Bis(glycinate)"),
    ExperimentalComplex("Zn-glycine2", "Zn2+", 2, 10, ["N","N","O","O"], "mixed", 2, 4,
        9.0, "25°C, I=0.1M", "NIST 46.7",
        donor_subtypes=["N_amine","N_amine","O_carboxylate","O_carboxylate"],
        ring_sizes=[5,5]),
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
    error: float
    abs_error: float


@dataclass
class ValidationReport:
    """Summary of validation across all complexes."""
    n_complexes: int
    results: list
    mean_abs_error: float
    r_squared: float
    systematic_bias: float
    hard_mae: float
    borderline_mae: float
    soft_mae: float
    mixed_mae: float
    calibration_slope: float
    calibration_intercept: float
    calibration_r2: float
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

    # Attach Sprint 35d donor_subtypes
    if c.donor_subtypes:
        rec.donor_subtypes = c.donor_subtypes

    # Attach Sprint 36 schema fields
    if c.ring_sizes:
        rec.ring_sizes = c.ring_sizes
    if c.is_macrocyclic:
        rec.is_macrocyclic = True
        rec.cavity_radius_nm = c.cavity_radius_nm
    if c.is_cage:
        rec.is_cage = True

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
    log_K_pred = -dg / 5.71

    return log_K_pred, dg


def run_validation(library=None, apply_calibration=False, calibration=None):
    """Run full validation against experimental library."""
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

    mae = sum(r.abs_error for r in results) / n
    mean_error = sum(r.error for r in results) / n

    exp_vals = [r.log_K_exp for r in results]
    pred_vals = [r.log_K_pred for r in results]
    exp_mean = sum(exp_vals) / n
    ss_res = sum((p - e)**2 for p, e in zip(pred_vals, exp_vals))
    ss_tot = sum((e - exp_mean)**2 for e in exp_vals)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    def class_mae(dtype):
        cls = [r for r in results if r.donor_type == dtype]
        return sum(r.abs_error for r in cls) / max(1, len(cls))

    hard_mae = class_mae("hard")
    border_mae = class_mae("borderline")
    soft_mae = class_mae("soft")
    mixed_mae = class_mae("mixed")

    if n > 2:
        sum_x = sum(exp_vals)
        sum_y = sum(pred_vals)
        sum_xy = sum(x*y for x, y in zip(exp_vals, pred_vals))
        sum_x2 = sum(x**2 for x in exp_vals)
        denom = n * sum_x2 - sum_x**2
        if denom > 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            pred_cal = [slope * e + intercept for e in exp_vals]
            ss_res_cal = sum((p - e)**2 for p, e in zip(pred_cal, exp_vals))
            cal_r2 = 1 - ss_res_cal / ss_tot if ss_tot > 0 else 0
        else:
            slope, intercept, cal_r2 = 1.0, 0.0, r2
    else:
        slope, intercept, cal_r2 = 1.0, 0.0, r2

    notes = []
    if abs(mean_error) > 3:
        direction = "over" if mean_error > 0 else "under"
        notes.append(f"Systematic {direction}prediction by {abs(mean_error):.1f} log K units.")
    if abs(slope - 1.0) > 0.2:
        notes.append(f"Calibration slope = {slope:.2f} (ideal = 1.0).")
    if hard_mae > 2 * border_mae and hard_mae > 3:
        notes.append(f"Hard-metal MAE={hard_mae:.1f}. Check electrostatic model.")
    if soft_mae > 2 * border_mae and soft_mae > 3:
        notes.append(f"Soft-metal MAE={soft_mae:.1f}. Check covalent/polarization.")
    if r2 > 0.7:
        notes.append(f"R²={r2:.3f} — model captures trend.")

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
    """Derive calibration factors from experimental library."""
    if library is None:
        library = VALIDATION_LIBRARY

    raw = run_validation(library, apply_calibration=False)
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


# ═══════════════════════════════════════════════════════════════════════════
# FILE 3: tests/test_sprint36.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint36.py", '''\
"""tests/test_sprint36.py — Sprint 36: Deep Physics Calibration (22 tests)

Tests the 5 physics improvements:
  1. Metal-specific ΔG_hyd desolvation
  2. Macrocyclic + cavity size-match
  3. Ring size correction (Hancock)
  4. Electrostatic z·z for anionic donors
  5. HSAB multiplicative scaling
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_integration import (
    compute_enhanced_thermodynamics, EnhancedThermodynamics,
    _HYDRATION_FREE_ENERGY, _AQUA_CN,
)
from core.validation import (
    VALIDATION_LIBRARY, ExperimentalComplex, run_validation,
    derive_calibration, apply_calibration_to_log_K,
    ValidationReport, ValidationResult, _predict_log_K,
)
from core.generative_physics_adapter import (
    RecognitionChemistry, StructuralConstraint, InteriorDesign,
    TargetSpecies, Matrix, Problem,
)

def _prob(name, formula, charge, d_e, soft, r_pm, ph=7.0):
    return Problem(
        target=TargetSpecies(identity=name, formula=formula, charge=charge,
            d_electrons=d_e, hsab_softness=soft, ionic_radius_pm=r_pm,
            hydrated_radius_nm=(r_pm+140)/1000),
        matrix=Matrix(ph=ph, temperature_c=25.0, ionic_strength_mm=100.0))

def _rec(donors, dt="borderline", chel=2, match=0.7):
    return RecognitionChemistry(name="t", type="generative", donor_atoms=donors,
        donor_type=dt, denticity=len(donors), hsab_match=match, chelate_rings=chel)

def _struct(stype="free", pore=0.0):
    return StructuralConstraint(name="s", type=stype, geometry="octahedral", pore_size_nm=pore)

def _interior():
    return InteriorDesign(description="t", num_binding_sites=1, self_binding=False)


# ═══════════════════════════════════════════════════════════════════════════
# 1. METAL-SPECIFIC ΔG_hyd DESOLVATION
# ═══════════════════════════════════════════════════════════════════════════

def test_hydration_table_coverage():
    """ΔG_hyd table should cover all metals in validation library."""
    metals = set(c.metal_formula for c in VALIDATION_LIBRARY)
    covered = sum(1 for m in metals if m in _HYDRATION_FREE_ENERGY)
    assert covered >= len(metals) - 1, f"Only {covered}/{len(metals)} metals in ΔG_hyd table"
    print(f"  \\u2705 test_hyd_coverage: {covered}/{len(metals)} metals covered")

def test_hydration_ordering():
    """ΔG_hyd should follow: |Al3+| > |Fe3+| > |Ni2+| > |Ca2+| > |K+|."""
    al = abs(_HYDRATION_FREE_ENERGY["Al3+"])
    fe = abs(_HYDRATION_FREE_ENERGY["Fe3+"])
    ni = abs(_HYDRATION_FREE_ENERGY["Ni2+"])
    ca = abs(_HYDRATION_FREE_ENERGY["Ca2+"])
    k = abs(_HYDRATION_FREE_ENERGY["K+"])
    assert al > fe > ni > ca > k
    print(f"  \\u2705 test_hyd_order: Al={al} > Fe={fe} > Ni={ni} > Ca={ca} > K={k}")

def test_desolv_fe3_vs_fe2():
    """Fe3+ desolvation should cost more than Fe2+ (higher charge, tighter shell)."""
    t_fe3 = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O","O","O"], "hard", 3), _struct(), _interior(),
        _prob("fe3", "Fe3+", 3, 5, 0.12, 65))
    t_fe2 = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O","O","O"], "hard", 3), _struct(), _interior(),
        _prob("fe2", "Fe2+", 2, 6, 0.25, 78))
    assert t_fe3.dg_desolv_kj > t_fe2.dg_desolv_kj, \\
        f"Fe3+ desolv ({t_fe3.dg_desolv_kj}) should > Fe2+ ({t_fe2.dg_desolv_kj})"
    print(f"  \\u2705 test_desolv_fe3_vs_fe2: Fe3+=+{t_fe3.dg_desolv_kj:.1f} > Fe2+=+{t_fe2.dg_desolv_kj:.1f}")

def test_desolv_irving_williams():
    """Desolvation cost across EDTA divalents should track Irving-Williams."""
    # Mn < Fe < Co < Ni < Cu (desolvation should increase along series)
    metals = [("Mn2+", 5, 0.25, 83), ("Fe2+", 6, 0.25, 78),
              ("Co2+", 7, 0.24, 75), ("Ni2+", 8, 0.24, 69), ("Cu2+", 9, 0.35, 73)]
    desolvs = []
    for f, de, s, r in metals:
        t = compute_enhanced_thermodynamics(
            _rec(["N","N","O","O","O","O"], "mixed", 5), _struct(), _interior(),
            _prob("t", f, 2, de, s, r))
        desolvs.append(t.dg_desolv_kj)
    # General trend should be increasing (Mn < Cu)
    assert desolvs[-1] > desolvs[0], f"Cu desolv ({desolvs[-1]}) should > Mn ({desolvs[0]})"
    print(f"  \\u2705 test_iw_desolv: Mn={desolvs[0]:.1f} → Cu={desolvs[-1]:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. MACROCYCLIC + CAVITY SIZE-MATCH
# ═══════════════════════════════════════════════════════════════════════════

def test_macrocyclic_term_present():
    """Macrocyclic complexes should have non-zero dg_macrocyclic_kj."""
    rec = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec.is_macrocyclic = True
    rec.cavity_radius_nm = 0.134  # 18-crown-6
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("k_crown", "K+", 1, 0, 0.01, 138))
    assert t.dg_macrocyclic_kj < 0, f"Macrocyclic should be stabilizing, got {t.dg_macrocyclic_kj}"
    print(f"  \\u2705 test_macro_present: dG_macro = {t.dg_macrocyclic_kj:.1f} kJ/mol")

def test_macrocyclic_absent_for_free():
    """Non-macrocyclic ligands should have dg_macrocyclic = 0."""
    t = compute_enhanced_thermodynamics(
        _rec(["N","N","N","N","N","N"], "borderline", 3), _struct(), _interior(),
        _prob("ni_en3", "Ni2+", 2, 8, 0.24, 69))
    assert t.dg_macrocyclic_kj == 0.0
    print(f"  \\u2705 test_macro_absent: dG_macro = {t.dg_macrocyclic_kj}")

def test_crown_k_better_than_na():
    """K+ in 18-crown-6 should bind more strongly than Na+ (size match)."""
    # K+: r=138pm, cavity=134pm → near-perfect match
    # Na+: r=102pm, cavity=134pm → mismatch
    rec_k = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_k.is_macrocyclic = True
    rec_k.cavity_radius_nm = 0.134
    rec_na = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_na.is_macrocyclic = True
    rec_na.cavity_radius_nm = 0.134

    t_k = compute_enhanced_thermodynamics(
        rec_k, _struct(), _interior(),
        _prob("k18c6", "K+", 1, 0, 0.01, 138))
    t_na = compute_enhanced_thermodynamics(
        rec_na, _struct(), _interior(),
        _prob("na18c6", "Na+", 1, 0, 0.01, 102))
    # K should be more negative (stronger binding) due to better size match
    assert t_k.dg_macrocyclic_kj < t_na.dg_macrocyclic_kj, \\
        f"K macro ({t_k.dg_macrocyclic_kj}) should be more negative than Na ({t_na.dg_macrocyclic_kj})"
    print(f"  \\u2705 test_crown_k_vs_na: K={t_k.dg_macrocyclic_kj:.1f} < Na={t_na.dg_macrocyclic_kj:.1f}")

def test_cage_stronger_than_macrocycle():
    """Cage (cryptand) should give stronger macrocyclic effect."""
    rec_mac = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_mac.is_macrocyclic = True
    rec_mac.cavity_radius_nm = 0.134
    rec_cage = _rec(["O","O","O","O","O","O"], "hard", 0)
    rec_cage.is_macrocyclic = True
    rec_cage.cavity_radius_nm = 0.134
    rec_cage.is_cage = True

    t_mac = compute_enhanced_thermodynamics(
        rec_mac, _struct(), _interior(),
        _prob("t", "K+", 1, 0, 0.01, 138))
    t_cage = compute_enhanced_thermodynamics(
        rec_cage, _struct(), _interior(),
        _prob("t", "K+", 1, 0, 0.01, 138))
    assert t_cage.dg_macrocyclic_kj < t_mac.dg_macrocyclic_kj
    print(f"  \\u2705 test_cage: cage={t_cage.dg_macrocyclic_kj:.1f} < mac={t_mac.dg_macrocyclic_kj:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. RING SIZE CORRECTION (HANCOCK RULE)
# ═══════════════════════════════════════════════════════════════════════════

def test_ring_strain_absent_5mem():
    """5-membered rings should have zero strain for any metal."""
    rec = _rec(["N","N","O","O","O","O"], "mixed", 5)
    rec.ring_sizes = [5,5,5,5,5]
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("pb", "Pb2+", 2, 0, 0.55, 119))
    assert t.dg_ring_strain_kj == 0.0, f"5-mem rings should have no strain, got {t.dg_ring_strain_kj}"
    print(f"  \\u2705 test_5mem_no_strain: Pb2+ strain = {t.dg_ring_strain_kj}")

def test_ring_strain_6mem_large_metal():
    """6-membered rings should penalize large metals like Pb2+."""
    rec = _rec(["N","N","O","O","O","O"], "mixed", 5)
    rec.ring_sizes = [6,6,6,6,6]
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("pb", "Pb2+", 2, 0, 0.55, 119))
    assert t.dg_ring_strain_kj > 0, f"6-mem rings + Pb2+ should have strain, got {t.dg_ring_strain_kj}"
    print(f"  \\u2705 test_6mem_large: Pb2+ strain = +{t.dg_ring_strain_kj:.1f} kJ/mol")

def test_ring_strain_small_metal_ok():
    """6-membered rings should have minimal strain for small metals like Cu2+."""
    rec = _rec(["N","N","O","O"], "mixed", 2)
    rec.ring_sizes = [6,6]
    t_cu = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("cu", "Cu2+", 2, 9, 0.35, 73))
    rec2 = _rec(["N","N","O","O"], "mixed", 2)
    rec2.ring_sizes = [6,6]
    t_pb = compute_enhanced_thermodynamics(
        rec2, _struct(), _interior(),
        _prob("pb", "Pb2+", 2, 0, 0.55, 119))
    assert t_cu.dg_ring_strain_kj < t_pb.dg_ring_strain_kj, \\
        f"Cu2+ strain ({t_cu.dg_ring_strain_kj}) should < Pb2+ ({t_pb.dg_ring_strain_kj})"
    print(f"  \\u2705 test_6mem_small: Cu={t_cu.dg_ring_strain_kj:.1f} < Pb={t_pb.dg_ring_strain_kj:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ELECTROSTATIC z·z FOR ANIONIC DONORS
# ═══════════════════════════════════════════════════════════════════════════

def test_zz_trivalent_stronger():
    """Fe3+-carboxylate z·z should be much larger than Cu2+-carboxylate."""
    rec = _rec(["O","O","O","O","O","O"], "hard", 3)
    rec.donor_subtypes = ["O_carboxylate"]*6
    t_fe3 = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("fe3", "Fe3+", 3, 5, 0.12, 65))
    t_cu = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("cu", "Cu2+", 2, 9, 0.35, 73))
    assert abs(t_fe3.dg_zz_electrostatic_kj) > abs(t_cu.dg_zz_electrostatic_kj), \\
        f"Fe3+ zz ({t_fe3.dg_zz_electrostatic_kj}) should > Cu2+ ({t_cu.dg_zz_electrostatic_kj})"
    print(f"  \\u2705 test_zz_trivalent: Fe3+={t_fe3.dg_zz_electrostatic_kj:.1f} vs Cu2+={t_cu.dg_zz_electrostatic_kj:.1f}")

def test_zz_zero_for_neutral_donors():
    """N-amine donors (neutral) should have zero z·z contribution."""
    rec = _rec(["N","N","N","N","N","N"], "borderline", 3)
    rec.donor_subtypes = ["N_amine"]*6
    t = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("ni", "Ni2+", 2, 8, 0.24, 69))
    assert abs(t.dg_zz_electrostatic_kj) < 0.1, \\
        f"N_amine z·z should be ~0, got {t.dg_zz_electrostatic_kj}"
    print(f"  \\u2705 test_zz_neutral: N_amine z·z = {t.dg_zz_electrostatic_kj:.2f}")

def test_zz_scales_with_charge():
    """z·z should scale roughly linearly with metal charge."""
    rec = _rec(["O","O","O","O"], "hard", 2)
    rec.donor_subtypes = ["O_carboxylate"]*4
    t2 = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("ca", "Ca2+", 2, 0, 0.01, 100))
    t3 = compute_enhanced_thermodynamics(
        rec, _struct(), _interior(),
        _prob("al", "Al3+", 3, 0, 0.01, 54))
    ratio = abs(t3.dg_zz_electrostatic_kj) / max(0.1, abs(t2.dg_zz_electrostatic_kj))
    assert ratio > 1.3, f"z·z ratio Al3+/Ca2+ should > 1.3, got {ratio:.2f}"
    print(f"  \\u2705 test_zz_charge: Ca2+={t2.dg_zz_electrostatic_kj:.1f}, Al3+={t3.dg_zz_electrostatic_kj:.1f}, ratio={ratio:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. HSAB MULTIPLICATIVE SCALING
# ═══════════════════════════════════════════════════════════════════════════

def test_hsab_soft_soft_amplified():
    """Hg2+(soft) + S donors should amplify binding vs neutral."""
    # Compare Hg2+ with S (matched) vs O (mismatched)
    t_s = compute_enhanced_thermodynamics(
        _rec(["S","S","S","S"], "soft", 2, 0.95), _struct(), _interior(),
        _prob("hg", "Hg2+", 2, 10, 0.85, 102))
    t_o = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O"], "hard", 2, 0.3), _struct(), _interior(),
        _prob("hg", "Hg2+", 2, 10, 0.85, 102))
    # S should be much more negative (stronger binding)
    assert t_s.dg_bind_kj < t_o.dg_bind_kj, \\
        f"Hg+S ({t_s.dg_bind_kj}) should be more negative than Hg+O ({t_o.dg_bind_kj})"
    print(f"  \\u2705 test_hsab_ss: Hg+S={t_s.dg_bind_kj:.0f} << Hg+O={t_o.dg_bind_kj:.0f}")

def test_hsab_hard_hard_amplified():
    """Ca2+(hard) + O donors should be amplified vs S donors."""
    t_o = compute_enhanced_thermodynamics(
        _rec(["O","O","O","O"], "hard", 2, 0.9), _struct(), _interior(),
        _prob("ca", "Ca2+", 2, 0, 0.01, 100))
    t_s = compute_enhanced_thermodynamics(
        _rec(["S","S","S","S"], "soft", 2, 0.3), _struct(), _interior(),
        _prob("ca", "Ca2+", 2, 0, 0.01, 100))
    # O should be more negative (harder = better match for Ca)
    assert t_o.dg_bind_kj < t_s.dg_bind_kj, \\
        f"Ca+O ({t_o.dg_bind_kj}) should be more negative than Ca+S ({t_s.dg_bind_kj})"
    print(f"  \\u2705 test_hsab_hh: Ca+O={t_o.dg_bind_kj:.0f} < Ca+S={t_s.dg_bind_kj:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: FULL VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_runs():
    """Validation should complete without errors."""
    report = run_validation()
    assert isinstance(report, ValidationReport)
    assert report.n_complexes == len(VALIDATION_LIBRARY)
    print(f"  \\u2705 test_runs: {report.n_complexes} complexes validated")

def test_all_predicted():
    """Every complex should get a prediction."""
    report = run_validation()
    for r in report.results:
        assert r.log_K_pred is not None
        assert math.isfinite(r.log_K_pred), f"{r.name}: {r.log_K_pred}"
    print(f"  \\u2705 test_all_predicted: {len(report.results)} predictions")

def test_metrics_reported():
    """Report should have R², MAE, bias."""
    report = run_validation()
    assert hasattr(report, "r_squared")
    assert hasattr(report, "mean_abs_error")
    print(f"  \\u2705 test_metrics: R²={report.r_squared:.3f}, MAE={report.mean_abs_error:.1f}")

def test_new_terms_in_output():
    """EnhancedThermodynamics should have Sprint 36 fields."""
    t = compute_enhanced_thermodynamics(
        _rec(["N","N","O","O"]), _struct(), _interior(),
        _prob("t", "Cu2+", 2, 9, 0.35, 73))
    assert hasattr(t, "dg_macrocyclic_kj")
    assert hasattr(t, "dg_ring_strain_kj")
    assert hasattr(t, "dg_zz_electrostatic_kj")
    print(f"  \\u2705 test_new_fields: macro={t.dg_macrocyclic_kj}, strain={t.dg_ring_strain_kj}, zz={t.dg_zz_electrostatic_kj}")

def test_calibration_works():
    """Calibration should reduce MAE."""
    raw = run_validation(apply_calibration=False)
    cal = derive_calibration()
    calibrated = run_validation(apply_calibration=True, calibration=cal)
    assert calibrated.mean_abs_error <= raw.mean_abs_error
    print(f"  \\u2705 test_calibration: raw MAE={raw.mean_abs_error:.1f} → cal MAE={calibrated.mean_abs_error:.1f}")

def test_full_report():
    """Print full validation report for manual inspection."""
    from core.validation import print_validation_report
    report = run_validation()
    print_validation_report(report)
    print(f"  \\u2705 test_full_report: printed above")


if __name__ == "__main__":
    print("\\n\\U0001f9ea Sprint 36 — Deep Physics Calibration\\n")
    print("1. Metal-specific ΔG_hyd:")
    test_hydration_table_coverage(); test_hydration_ordering()
    test_desolv_fe3_vs_fe2(); test_desolv_irving_williams()
    print("\\n2. Macrocyclic + Cavity Size-Match:")
    test_macrocyclic_term_present(); test_macrocyclic_absent_for_free()
    test_crown_k_better_than_na(); test_cage_stronger_than_macrocycle()
    print("\\n3. Ring Size Correction (Hancock):")
    test_ring_strain_absent_5mem(); test_ring_strain_6mem_large_metal()
    test_ring_strain_small_metal_ok()
    print("\\n4. Electrostatic z·z:")
    test_zz_trivalent_stronger(); test_zz_zero_for_neutral_donors()
    test_zz_scales_with_charge()
    print("\\n5. HSAB Multiplicative:")
    test_hsab_soft_soft_amplified(); test_hsab_hard_hard_amplified()
    print("\\nIntegration:")
    test_validation_runs(); test_all_predicted()
    test_metrics_reported(); test_new_terms_in_output()
    test_calibration_works(); test_full_report()
    print("\\n\\u2705 All Sprint 36 tests passed! (22/22)")
    print("\\n\\U0001f389 DEEP PHYSICS CALIBRATION COMPLETE\\n")

''')

print("\n\u2705 Sprint 36 files created!\n")
print("Files modified:")
print("  core/physics_integration.py — 5 term upgrades")
print("  core/validation.py — schema + annotated library")
print("  tests/test_sprint36.py — 25 tests\n")