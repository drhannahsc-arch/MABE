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

        # Charge scaling: anionic α=5.0, neutral α=5.5
        _AN = {"O_carboxylate","O_phenolate","O_catecholate","O_hydroxamate",
               "S_thiolate","S_thiosulfate","S_dithiocarbamate"}
        _AN_EL = {"Cl","Br","I"}
        _is_an = (donor_subtypes and i < len(donor_subtypes) and donor_subtypes[i] in _AN) or da in _AN_EL
        charge_factor = -(5.0 if _is_an else 5.5) * (charge**2 - 1) / max(1, len(donors))
        dg_exchange += charge_factor

        # HSAB MULTIPLICATIVE FACTOR [S36]
        # mismatch ∈ [0, ~0.7]: 0 = perfect match, 0.7 = extreme mismatch
        # f_HSAB: perfect match → 1.5, neutral (0.3) → 1.0, extreme → 0.3
        mismatch = abs(softness - ds)
        if mismatch < 0.15:
            # Good match: amplify binding (up to 1.5×)
            f_hsab = 1.0 + 0.3 * max(softness, ds) * (1.0 - mismatch / 0.15)
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

    # Soft-soft saturation: diminishing returns beyond 3 soft donors
    if softness > 0.5:
        _SOFT_D = {"S", "P", "I"}
        _si = [j for j, d in enumerate(donors) if d in _SOFT_D]
        if len(_si) > 3:
            _excess = sum(donor_energies[j] for j in _si[3:] if j < len(donor_energies))
            dg_bind -= _excess * 0.4

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

    # Effective dielectric for inner-sphere: ~4 (Warshel 4-8 range;
    # calibrated against 47 training + 7705 NIST entries)
    k_elec = 1389.4 / 4.0   # = 347.4

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

    effective_fraction = max(0.01, free_ion_frac) if strategy == "free_ion_binding" else \
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

    conf = "high" if pol.softness_continuous > 0 and recognition.hsab_match > 0.7 else \
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



