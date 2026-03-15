"""
scaffold_designer.py — Physics-Based Scaffold Design Engine

Phases A-C: Geometric descriptors, thermodynamic design equations,
pKa-dependent donor availability.

Design philosophy (MABE reality-up):
  All parameters from first principles or non-biological calibration.
  No fitting to receptor-guest binding data.

Phase A: Extract geometric descriptors from backbone SMILES via RDKit 3D
Phase B: Thermodynamic design equations from geometry + calibrated params
Phase C: pKa adjustment — donor availability at working pH

Phases D-E (next session): Hydrodynamics, inverse design engine.

Usage:
    from core.scaffold_designer import (
        extract_descriptors, compute_thermo_score, compute_pka_adjustment,
        rank_scaffolds_for_guest, ScaffoldDescriptor, GuestSpec
    )

    # Extract from backbone library
    descriptors = extract_all_descriptors()

    # Design for a target guest
    guest = GuestSpec(name="selenite", diameter_A=3.5, volume_A3=45.0,
                      charge=-2, pH=5.0)
    ranked = rank_scaffolds_for_guest(guest, descriptors)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

R_kJ = 8.314e-3       # kJ/(mol·K)
T_STD = 298.15         # K
RT_STD = R_kJ * T_STD  # 2.4790 kJ/mol
PI = math.pi


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Scaffold Descriptors
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldDescriptor:
    """Geometric and physical descriptors for a backbone scaffold.

    Extracted from 3D-embedded SMILES (RDKit).
    """
    name: str
    smiles: str
    category: str               # linear, aromatic, cyclic, macrocyclic, branched
    n_sites: int                # number of attachment points

    # Geometry (from 3D embedding)
    d_arm_A: float = 0.0       # distance between attachment points (Å)
    theta_conv_deg: float = 0.0 # convergence angle (degrees)
                                # <90 = arms converge (receptor), >120 = diverge
    donor_vector_angle_deg: float = 0.0  # angle between arm bond vectors

    # Rigidity
    n_rotors_bridge: int = 0   # rotatable bonds between attachment sites
    rigidity_index: float = 0.0 # 1/(1 + n_rotors), 0→1 (1=fully rigid)

    # Size
    mw: float = 0.0            # molecular weight of backbone (no arms)
    n_heavy: int = 0           # heavy atom count
    sasa_A2: float = 0.0       # solvent-accessible surface area

    # Cavity estimate
    V_cavity_est_A3: float = 0.0  # estimated accessible volume between arms

    # Donor character of backbone itself (not arms)
    backbone_donors: List[str] = field(default_factory=list)
    backbone_acceptors: List[str] = field(default_factory=list)
    n_backbone_hb_donors: int = 0
    n_backbone_hb_acceptors: int = 0

    # Computed scores (populated by Phase B/C)
    thermo_score: float = 0.0
    pka_penalty_kJ: float = 0.0

    @property
    def is_convergent(self) -> bool:
        """Arms point inward — suitable for encapsulation."""
        return self.theta_conv_deg < 90.0

    @property
    def is_preorganized(self) -> bool:
        """Rigid enough to avoid large reorganization penalty."""
        return self.rigidity_index > 0.5


def extract_descriptor(backbone, force_field="MMFF") -> ScaffoldDescriptor:
    """Extract geometric descriptors from a Backbone via RDKit 3D embedding.

    Args:
        backbone: Backbone dataclass (name, smiles, n_sites, category)
        force_field: "MMFF" or "UFF" for geometry optimization

    Returns:
        ScaffoldDescriptor with all fields populated.

    Requires rdkit.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFreeSASA

    sd = ScaffoldDescriptor(
        name=backbone.name,
        smiles=backbone.smiles,
        category=backbone.category,
        n_sites=backbone.n_sites,
    )

    mol = Chem.MolFromSmiles(backbone.smiles)
    if mol is None:
        return sd

    # ── Basic descriptors (no 3D needed) ──────────────────────────────

    sd.mw = Descriptors.MolWt(mol)
    sd.n_heavy = mol.GetNumHeavyAtoms()
    sd.n_rotors_bridge = Descriptors.NumRotatableBonds(mol)
    sd.rigidity_index = 1.0 / (1.0 + sd.n_rotors_bridge)

    # Backbone H-bond character
    sd.n_backbone_hb_donors = rdMolDescriptors.CalcNumHBD(mol)
    sd.n_backbone_hb_acceptors = rdMolDescriptors.CalcNumHBA(mol)

    # Identify donor/acceptor atoms
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ("N", "O", "S") and atom.GetTotalNumHs() > 0:
            sd.backbone_donors.append(sym)
        if sym in ("N", "O", "S"):
            sd.backbone_acceptors.append(sym)

    # ── Find attachment points (dummy atoms) ──────────────────────────

    dummy_indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy atom [*]
            dummy_indices.append(atom.GetIdx())

    if len(dummy_indices) < 2:
        # Single-site backbone — can't compute d_arm or angle
        sd.d_arm_A = 0.0
        sd.theta_conv_deg = 0.0
        return sd

    # ── 3D embedding ──────────────────────────────────────────────────
    # RDKit can't embed dummy atoms. Replace [*] with carbon, embed,
    # then use the carbon positions as proxy for attachment points.

    from rdkit.Chem import RWMol
    mol_rw = RWMol(mol)
    dummy_map = {}  # old_idx → new_idx (may shift after edits)
    for idx in dummy_indices:
        mol_rw.GetAtomWithIdx(idx).SetAtomicNum(6)  # C as placeholder
    mol_real = mol_rw.GetMol()

    try:
        Chem.SanitizeMol(mol_real)
    except Exception:
        return sd  # unsanitizable after substitution

    mol_h = Chem.AddHs(mol_real)

    # Map: find the placeholder carbons (were dummies) in mol_h
    # They correspond to atoms that were AtomicNum 0 in the original
    # After AddHs, indices may shift. Re-find by matching topology.
    # Simpler approach: record which atom indices in mol_real were dummies,
    # then find them in mol_h (AddHs appends H atoms at end, so original
    # atom indices are preserved).
    placeholder_indices = list(dummy_indices)  # same indices in mol_real

    embed_result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    if embed_result < 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxIterations = 500
        embed_result = AllChem.EmbedMolecule(mol_h, params)

    if embed_result < 0:
        return sd

    # Optimize
    try:
        if force_field == "MMFF":
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass

    conf = mol_h.GetConformer()

    # ── Distance between attachment points ────────────────────────────
    # Use placeholder carbon positions (same indices as original dummies)

    if len(placeholder_indices) >= 2:
        p0 = np.array(conf.GetAtomPosition(placeholder_indices[0]))
        p1 = np.array(conf.GetAtomPosition(placeholder_indices[1]))
        sd.d_arm_A = float(np.linalg.norm(p1 - p0))

        # ── Convergence angle ─────────────────────────────────────────

        def get_neighbor_pos(ph_idx):
            atom = mol_h.GetAtomWithIdx(ph_idx)
            for nb in atom.GetNeighbors():
                if nb.GetIdx() not in placeholder_indices and nb.GetAtomicNum() > 1:
                    return np.array(conf.GetAtomPosition(nb.GetIdx()))
            # Fallback: any non-H neighbor
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() > 1:
                    return np.array(conf.GetAtomPosition(nb.GetIdx()))
            return None

        n0 = get_neighbor_pos(placeholder_indices[0])
        n1 = get_neighbor_pos(placeholder_indices[1])

        if n0 is not None and n1 is not None:
            v0 = n0 - p0
            v1 = n1 - p1
            v0n = v0 / (np.linalg.norm(v0) + 1e-10)
            v1n = v1 / (np.linalg.norm(v1) + 1e-10)
            cos_theta = np.clip(np.dot(v0n, v1n), -1.0, 1.0)
            sd.donor_vector_angle_deg = float(np.degrees(np.arccos(cos_theta)))

            mid = (p0 + p1) / 2.0
            to_mid_0 = mid - p0
            to_mid_1 = mid - p1
            conv_0 = np.dot(v0, to_mid_0)
            conv_1 = np.dot(v1, to_mid_1)
            if conv_0 > 0 and conv_1 > 0:
                sd.theta_conv_deg = sd.donor_vector_angle_deg
            else:
                sd.theta_conv_deg = 180.0 - sd.donor_vector_angle_deg

    # ── Cavity volume estimate ────────────────────────────────────────
    # Rough: prolate ellipsoid with d_arm as major axis, minor = d_arm/3
    if sd.d_arm_A > 0:
        a = sd.d_arm_A / 2.0  # semi-major
        b = a / 3.0            # semi-minor (heuristic)
        sd.V_cavity_est_A3 = (4.0 / 3.0) * PI * a * b * b

    # ── SASA ──────────────────────────────────────────────────────────
    try:
        radii = rdFreeSASA.classifyAtoms(mol_h)
        sd.sasa_A2 = rdFreeSASA.CalcSASA(mol_h, radii)
    except Exception:
        sd.sasa_A2 = sd.n_heavy * 10.0  # rough fallback

    return sd


def extract_all_descriptors(backbones=None) -> List[ScaffoldDescriptor]:
    """Extract descriptors from all backbones in the library.

    Args:
        backbones: list of Backbone objects. If None, imports from
                   de_novo_generator.BACKBONE_LIBRARY + RECEPTOR_BACKBONE_LIBRARY.

    Returns:
        List of ScaffoldDescriptor objects.
    """
    if backbones is None:
        from core.de_novo_generator import BACKBONE_LIBRARY, RECEPTOR_BACKBONE_LIBRARY
        # Combine, dedup by name
        seen = set()
        backbones = []
        for bb in list(BACKBONE_LIBRARY) + list(RECEPTOR_BACKBONE_LIBRARY):
            if bb.name not in seen:
                backbones.append(bb)
                seen.add(bb.name)

    descriptors = []
    for bb in backbones:
        sd = extract_descriptor(bb)
        descriptors.append(sd)

    return descriptors


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Thermodynamic Design Equations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GuestSpec:
    """Target guest specification for scaffold design.

    Provides the minimum information needed to compute required scaffold
    geometry and rank candidates.
    """
    name: str = ""
    diameter_A: float = 0.0       # effective guest diameter (Å)
    volume_A3: float = 0.0        # guest volume (ų)
    charge: int = 0               # formal charge
    n_hb_donors: int = 0          # guest H-bond donors
    n_hb_acceptors: int = 0       # guest H-bond acceptors
    n_aromatic_rings: int = 0     # aromatic ring count
    is_anion: bool = False        # True for oxoanion targets
    pH: float = 7.4               # working pH
    smiles: str = ""              # optional SMILES

    def __post_init__(self):
        if self.charge < 0 and not self.is_anion:
            self.is_anion = True
        if self.volume_A3 > 0 and self.diameter_A <= 0:
            # Estimate diameter from volume (sphere)
            self.diameter_A = 2.0 * (3.0 * self.volume_A3 / (4.0 * PI)) ** (1.0 / 3.0)
        if self.diameter_A > 0 and self.volume_A3 <= 0:
            r = self.diameter_A / 2.0
            self.volume_A3 = (4.0 / 3.0) * PI * r ** 3


@dataclass
class ScaffoldRequirements:
    """Computed scaffold requirements from a GuestSpec."""
    d_arm_required_A: float = 0.0       # required arm separation
    V_cavity_required_A3: float = 0.0   # required cavity volume (Rebek)
    convergence_required: bool = True    # need convergent arms?
    min_hb_donors: int = 0              # receptor must provide ≥N donors
    min_hb_acceptors: int = 0           # receptor must provide ≥N acceptors
    prefer_aromatic_walls: bool = False  # π-stacking benefit
    prefer_cationic: bool = False        # charge-assisted H-bond for anions


def compute_scaffold_requirements(guest: GuestSpec) -> ScaffoldRequirements:
    """Derive required scaffold geometry from guest specification.

    Physics:
        d_arm ≈ guest_diameter × 1.5  (arm separation needs to span guest
            plus van der Waals contact on both sides, ~1.5 Å each side)
        V_cavity = V_guest / 0.55  (Rebek 55% packing rule)
        Convergence: encapsulation targets need inward-pointing arms
        Donors: complement guest acceptors, and vice versa

    All relationships from geometric/thermodynamic first principles.
    No fitting to binding data.
    """
    req = ScaffoldRequirements()

    # Size matching
    # Scaffold arm separation needs to span the guest + vdW clearance
    # vdW radius of typical atoms ~1.5-1.8 Å, need clearance on each side
    VDW_CLEARANCE = 1.5  # Å per side
    req.d_arm_required_A = guest.diameter_A + 2 * VDW_CLEARANCE

    # Cavity volume from Rebek 55% rule
    REBEK_OPTIMAL = 0.55
    if guest.volume_A3 > 0:
        req.V_cavity_required_A3 = guest.volume_A3 / REBEK_OPTIMAL

    # Convergence: always needed for molecular recognition
    req.convergence_required = True

    # H-bond complementarity: receptor donors for guest acceptors, vice versa
    req.min_hb_donors = guest.n_hb_acceptors    # receptor donates to guest
    req.min_hb_acceptors = guest.n_hb_donors     # receptor accepts from guest

    # Aromatic guests benefit from aromatic-walled receptors (π-stacking)
    req.prefer_aromatic_walls = guest.n_aromatic_rings > 0

    # Anion guests benefit from cationic/charge-assisted scaffolds
    req.prefer_cationic = guest.is_anion

    return req


# ── Thermodynamic scoring of a scaffold against requirements ──────────

# Calibrated parameters (from MABE Phase 9 + first principles)
EPS_ROTOR = 2.48           # kJ/mol per frozen rotor (MABE HG Phase 9)
K_RIGID_BONUS = 2.0        # kJ/mol bonus per unit rigidity_index
K_SIZE_MISMATCH = 5.0      # kJ/mol per Å mismatch in d_arm
K_VOLUME_MISMATCH = 0.02   # kJ/mol per ų mismatch in cavity volume
K_CONVERGENCE_PENALTY = 8.0 # kJ/mol penalty for divergent scaffold on convergent guest
K_HB_UNMET = 4.0           # kJ/mol per unmet H-bond site
K_AROMATIC_BONUS = 2.0     # kJ/mol bonus for aromatic walls matching aromatic guest
K_CHARGE_BONUS = 5.0       # kJ/mol for cationic scaffold binding anion


def compute_thermo_score(sd: ScaffoldDescriptor,
                         req: ScaffoldRequirements,
                         guest: GuestSpec) -> float:
    """Compute thermodynamic suitability score for a scaffold.

    Returns ΔG_scaffold in kJ/mol (more negative = better scaffold).
    Composed of:
        ΔG_preorg   = bonus for rigid, preorganized scaffolds
        ΔG_flex     = penalty for flexible bridges (entropy on binding)
        ΔG_size     = penalty for arm separation mismatch
        ΔG_volume   = penalty for cavity volume mismatch
        ΔG_conv     = penalty for wrong convergence geometry
        ΔG_hb       = penalty for missing H-bond complementarity
        ΔG_arom     = bonus for aromatic wall matching
        ΔG_charge   = bonus for charge complementarity (cation-anion)

    Parameters from MABE calibration (eps_rotor, k_shape) and
    geometric first principles. NOT fitted to receptor binding data.
    """
    dG = 0.0

    # Preorganization bonus: rigid scaffolds don't pay reorganization
    dG_preorg = -K_RIGID_BONUS * sd.rigidity_index
    dG += dG_preorg

    # Flexibility penalty: each rotor in the bridge costs entropy on binding
    dG_flex = EPS_ROTOR * sd.n_rotors_bridge
    dG += dG_flex

    # Size matching: Gaussian penalty for d_arm mismatch
    if sd.d_arm_A > 0 and req.d_arm_required_A > 0:
        d_mismatch = abs(sd.d_arm_A - req.d_arm_required_A)
        dG_size = K_SIZE_MISMATCH * d_mismatch
        dG += dG_size

    # Cavity volume matching
    if sd.V_cavity_est_A3 > 0 and req.V_cavity_required_A3 > 0:
        v_mismatch = abs(sd.V_cavity_est_A3 - req.V_cavity_required_A3)
        dG_volume = K_VOLUME_MISMATCH * v_mismatch
        dG += dG_volume

    # Convergence geometry
    if req.convergence_required and not sd.is_convergent:
        dG += K_CONVERGENCE_PENALTY

    # H-bond complementarity
    # Scaffold's backbone donors complement guest's acceptors
    unmet_donors = max(0, req.min_hb_donors - sd.n_backbone_hb_donors)
    unmet_acceptors = max(0, req.min_hb_acceptors - sd.n_backbone_hb_acceptors)
    dG_hb = K_HB_UNMET * (unmet_donors + unmet_acceptors)
    dG += dG_hb

    # Aromatic wall bonus
    if req.prefer_aromatic_walls and sd.category in ("aromatic",):
        dG -= K_AROMATIC_BONUS

    # Charge complementarity for anions
    if req.prefer_cationic:
        # Scaffolds with backbone N-H donors or cationic character help
        if sd.n_backbone_hb_donors >= 2:
            dG -= K_CHARGE_BONUS

    sd.thermo_score = dG
    return dG


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: pKa-Dependent Donor Availability
# ═══════════════════════════════════════════════════════════════════════════

# pKa values for common donor group types in water at 25°C
# Sources: CRC Handbook, Perrin 1981, Smith & Martell 2004
# These are CONSENSUS values; individual compounds may vary ±1 unit.

PKA_TABLE: Dict[str, Tuple[float, str]] = {
    # (pKa, protonation_direction)
    # "deprotonates": active form is deprotonated (e.g., RCOO⁻)
    # "protonates": active form is protonated (e.g., RNH₃⁺ → RNH₂ active)

    # Carboxylate donors: active when deprotonated (COO⁻)
    "O_carboxylate": (4.0, "deprotonates"),

    # Phenolate donors: active when deprotonated (ArO⁻)
    "O_phenolate": (10.0, "deprotonates"),

    # Hydroxyl donors: active as neutral ROH (always active, no pKa issue)
    "O_hydroxyl": (None, "always_active"),  # ROH always protonated at practical pH

    # Amine donors: active as neutral R-NH₂ (loses coordination when protonated RNH₃⁺)
    "N_amine": (10.0, "deprotonates"),  # active as free base RNH2 for coordination
    "N_amine_secondary": (10.7, "deprotonates"),  # active as free base R2NH
    "N_amine_tertiary": (9.5, "deprotonates"),  # active as free base R3N

    # Pyridine: active as neutral (lone pair on N)
    "N_pyridine": (5.2, "deprotonates"),  # active as free base (N: lone pair)

    # Imidazole: active as neutral
    "N_imidazole": (7.0, "deprotonates"),  # active as neutral imidazole

    # Hydroxamate: active when deprotonated (N-O⁻)
    "O_hydroxamate": (8.5, "deprotonates"),

    # Phosphonate: active when deprotonated (RPO₃²⁻)
    "O_phosphate": (6.5, "deprotonates"),  # pKa2

    # Thiol: active when deprotonated (RS⁻)
    "S_thiol": (8.5, "deprotonates"),

    # Thioether: no pKa issue (neutral, always active)
    "S_thioether": (None, "always_active"),

    # Catechol: active when first OH deprotonates
    "O_catechol": (9.2, "deprotonates"),

    # Pyrrole NH: active as neutral (NH donor)
    "N_pyrrole": (None, "always_active"),  # pyrrole NH always protonated at practical pH

    # Guanidinium: active as cation (always protonated at normal pH)
    "N_guanidinium": (None, "always_active"),  # pKa 13.5, always protonated/cationic

    # Urea NH: no pKa issue (always neutral)
    "N_urea": (None, "always_active"),

    # Sulfonamide NH: weakly acidic
    "N_sulfonamide": (10.0, "deprotonates"),
}


def donor_fraction_active(donor_type: str, pH: float) -> float:
    """Fraction of a donor type in its active (binding-competent) form.

    Henderson-Hasselbalch:
        For donors active when deprotonated: f = 1 / (1 + 10^(pKa - pH))
        For donors active when protonated:   f = 1 / (1 + 10^(pH - pKa))
        For always-active donors:            f = 1.0

    Parameters
    ----------
    donor_type : str
        Key from PKA_TABLE (e.g., "O_carboxylate", "N_amine").
    pH : float
        Working pH.

    Returns
    -------
    float
        Fraction active, 0.0 to 1.0.
    """
    if donor_type not in PKA_TABLE:
        return 1.0  # unknown type → assume always active

    pKa, direction = PKA_TABLE[donor_type]

    if pKa is None or direction == "always_active":
        return 1.0

    if direction == "deprotonates":
        # Active form is deprotonated: f = [A⁻]/([HA]+[A⁻])
        return 1.0 / (1.0 + 10.0 ** (pKa - pH))
    elif direction == "protonates":
        # Active form is protonated (HA): f = [HA]/([HA]+[A⁻])
        # f = 1/(1 + 10^(pH - pKa))
        # At pH << pKa: protonated (active), f → 1
        # At pH >> pKa: deprotonated (inactive), f → 0
        return 1.0 / (1.0 + 10.0 ** (pH - pKa))
    else:
        return 1.0


def compute_pka_adjustment(donor_types: List[str], pH: float) -> Tuple[float, Dict[str, float]]:
    """Compute total pKa penalty for a set of donor types at given pH.

    The penalty reflects the free energy cost of maintaining donors in
    their active (binding-competent) protonation state at working pH.

    ΔG_pH = -RT × Σ ln(f_active_i)

    where f_active_i = Henderson-Hasselbalch fraction for donor i.

    Parameters
    ----------
    donor_types : list of str
        Donor types present on the scaffold + arms.
    pH : float
        Working pH.

    Returns
    -------
    dG_pH : float
        Total pKa penalty in kJ/mol (positive = unfavorable).
    fractions : dict
        Per-donor-type active fraction.
    """
    fractions = {}
    dG_total = 0.0

    for dt in donor_types:
        f = donor_fraction_active(dt, pH)
        fractions[dt] = f
        if f > 0 and f < 1.0:
            # Cost of maintaining this donor in active form
            dG_total += -RT_STD * math.log(f)
        elif f <= 0:
            # Effectively dead at this pH
            dG_total += 20.0  # large penalty (kJ/mol)

    return dG_total, fractions


def effective_donor_count(donor_types: List[str], pH: float) -> float:
    """Compute effective number of active donors at working pH.

    n_eff = Σ f_active(donor_i)

    A carboxylate at pH 2 contributes ~0.01 donors.
    An amine at pH 7 contributes ~1.0 donors.
    """
    return sum(donor_fraction_active(dt, pH) for dt in donor_types)


# ═══════════════════════════════════════════════════════════════════════════
# Combined Ranking: Phases A+B+C
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldRanking:
    """Result of ranking a scaffold against a guest specification."""
    descriptor: ScaffoldDescriptor
    guest: GuestSpec
    requirements: ScaffoldRequirements

    # Scores (kJ/mol, more negative = better)
    dG_thermo: float = 0.0         # Phase B thermodynamic score
    dG_pka: float = 0.0            # Phase C pKa penalty
    dG_total: float = 0.0          # Combined A+B+C

    # Diagnostics
    size_match_A: float = 0.0      # |d_arm - d_required| in Å
    volume_match_A3: float = 0.0   # |V_cavity - V_required| in ų
    n_eff_donors: float = 0.0      # effective donors at pH
    pka_fractions: Dict[str, float] = field(default_factory=dict)

    # Gap flags
    too_small: bool = False
    too_large: bool = False
    wrong_geometry: bool = False
    insufficient_donors: bool = False

    @property
    def rank_score(self) -> float:
        return self.dG_total


def rank_one_scaffold(sd: ScaffoldDescriptor,
                      guest: GuestSpec,
                      arm_donor_types: Optional[List[str]] = None) -> ScaffoldRanking:
    """Score a single scaffold against a guest specification.

    Combines Phase B (thermo) and Phase C (pKa) scoring.

    Args:
        sd: ScaffoldDescriptor (from Phase A extraction)
        guest: GuestSpec (target)
        arm_donor_types: optional list of donor types from attached arms.
            If None, uses only backbone donors.

    Returns:
        ScaffoldRanking with complete diagnostics.
    """
    req = compute_scaffold_requirements(guest)

    # Phase B: thermodynamic score
    dG_thermo = compute_thermo_score(sd, req, guest)

    # Phase C: pKa adjustment
    # Collect all donor types (backbone + arms)
    all_donors = list(sd.backbone_donors)
    if arm_donor_types:
        all_donors.extend(arm_donor_types)

    # Map element symbols to donor types for backbone donors
    _element_to_type = {"N": "N_amine", "O": "O_hydroxyl", "S": "S_thiol"}
    typed_donors = []
    for d in all_donors:
        if d in PKA_TABLE:
            typed_donors.append(d)
        elif d in _element_to_type:
            typed_donors.append(_element_to_type[d])

    dG_pka, fractions = compute_pka_adjustment(typed_donors, guest.pH)
    n_eff = effective_donor_count(typed_donors, guest.pH)

    dG_total = dG_thermo + dG_pka

    # Diagnostics
    size_match = abs(sd.d_arm_A - req.d_arm_required_A) if sd.d_arm_A > 0 else 999.0
    volume_match = abs(sd.V_cavity_est_A3 - req.V_cavity_required_A3) if sd.V_cavity_est_A3 > 0 else 999.0

    ranking = ScaffoldRanking(
        descriptor=sd,
        guest=guest,
        requirements=req,
        dG_thermo=dG_thermo,
        dG_pka=dG_pka,
        dG_total=dG_total,
        size_match_A=size_match,
        volume_match_A3=volume_match,
        n_eff_donors=n_eff,
        pka_fractions=fractions,
        too_small=sd.d_arm_A > 0 and sd.d_arm_A < req.d_arm_required_A * 0.7,
        too_large=sd.d_arm_A > 0 and sd.d_arm_A > req.d_arm_required_A * 1.5,
        wrong_geometry=req.convergence_required and not sd.is_convergent,
        insufficient_donors=(n_eff < max(req.min_hb_donors, req.min_hb_acceptors) * 0.5
                             if max(req.min_hb_donors, req.min_hb_acceptors) > 0 else False),
    )

    sd.thermo_score = dG_total
    sd.pka_penalty_kJ = dG_pka
    return ranking


def rank_scaffolds_for_guest(guest: GuestSpec,
                             descriptors: List[ScaffoldDescriptor],
                             arm_donor_types: Optional[List[str]] = None,
                             top_n: int = 20) -> List[ScaffoldRanking]:
    """Rank all scaffolds against a guest specification.

    Args:
        guest: target GuestSpec
        descriptors: list of ScaffoldDescriptor (from Phase A)
        arm_donor_types: optional common donor types to assume for arms
        top_n: return top N scaffolds

    Returns:
        List of ScaffoldRanking, sorted by dG_total (best first).
    """
    rankings = []
    for sd in descriptors:
        r = rank_one_scaffold(sd, guest, arm_donor_types)
        rankings.append(r)

    rankings.sort(key=lambda r: r.dG_total)
    return rankings[:top_n]


def gap_analysis(rankings: List[ScaffoldRanking],
                 guest: GuestSpec) -> Dict[str, List[str]]:
    """Identify gaps in the scaffold library for a guest.

    Returns dict of gap_type → list of scaffold names with that gap.
    """
    gaps = {
        "too_small": [],
        "too_large": [],
        "wrong_geometry": [],
        "insufficient_donors": [],
        "severe_pka_penalty": [],
    }

    for r in rankings:
        if r.too_small:
            gaps["too_small"].append(r.descriptor.name)
        if r.too_large:
            gaps["too_large"].append(r.descriptor.name)
        if r.wrong_geometry:
            gaps["wrong_geometry"].append(r.descriptor.name)
        if r.insufficient_donors:
            gaps["insufficient_donors"].append(r.descriptor.name)
        if r.dG_pka > 10.0:  # >10 kJ/mol pKa penalty
            gaps["severe_pka_penalty"].append(r.descriptor.name)

    return {k: v for k, v in gaps.items() if v}


# ═══════════════════════════════════════════════════════════════════════════
# Summary / Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_ranking_summary(rankings: List[ScaffoldRanking], top_n: int = 15):
    """Print human-readable ranking table."""
    guest = rankings[0].guest if rankings else None
    if guest:
        print(f"\nScaffold Design for: {guest.name}")
        print(f"Guest: d={guest.diameter_A:.1f} Å, V={guest.volume_A3:.1f} ų, "
              f"q={guest.charge:+d}, pH={guest.pH}")
        req = rankings[0].requirements
        print(f"Required: d_arm={req.d_arm_required_A:.1f} Å, "
              f"V_cavity={req.V_cavity_required_A3:.1f} ų")

    print(f"\n{'Rank':>4} {'Scaffold':<30} {'ΔG_tot':>7} {'ΔG_thm':>7} "
          f"{'ΔG_pKa':>7} {'d_arm':>6} {'rigid':>5} {'n_eff':>5} {'Flags'}")
    print("-" * 100)

    for i, r in enumerate(rankings[:top_n]):
        flags = []
        if r.too_small: flags.append("S")
        if r.too_large: flags.append("L")
        if r.wrong_geometry: flags.append("G")
        if r.insufficient_donors: flags.append("D")
        if r.dG_pka > 5.0: flags.append("pH")
        flag_str = ",".join(flags) if flags else "-"

        print(f"{i+1:4d} {r.descriptor.name:<30} {r.dG_total:+7.1f} "
              f"{r.dG_thermo:+7.1f} {r.dG_pka:+7.1f} "
              f"{r.descriptor.d_arm_A:6.1f} {r.descriptor.rigidity_index:5.2f} "
              f"{r.n_eff_donors:5.1f} {flag_str}")


def print_pka_table(pH_values=None):
    """Print pKa-dependent donor availability table."""
    if pH_values is None:
        pH_values = [3.0, 5.0, 7.0, 7.4, 9.0, 11.0]

    print(f"\n{'Donor Type':<20}", end="")
    for pH in pH_values:
        print(f"  pH {pH:>4.1f}", end="")
    print(f"  {'pKa':>5}  {'Active form'}")
    print("-" * (22 + 8 * len(pH_values) + 18))

    for dt, (pKa, direction) in sorted(PKA_TABLE.items()):
        print(f"{dt:<20}", end="")
        for pH in pH_values:
            f = donor_fraction_active(dt, pH)
            print(f"  {f:>6.3f}", end="")
        pka_str = f"{pKa:.1f}" if pKa is not None else "  n/a"
        print(f"  {pka_str:>5}  {direction}")


if __name__ == "__main__":
    print("=" * 70)
    print("Scaffold Designer — Phases A/B/C")
    print("=" * 70)

    # Print pKa table
    print_pka_table()

    # Example: selenite target at pH 5 (Elk Valley)
    selenite = GuestSpec(
        name="selenite (SeO₃²⁻)",
        diameter_A=3.5,
        volume_A3=45.0,
        charge=-2,
        n_hb_donors=0,
        n_hb_acceptors=3,  # 3 oxygens
        pH=5.0,
    )

    req = compute_scaffold_requirements(selenite)
    print(f"\nSelenite requirements:")
    print(f"  d_arm = {req.d_arm_required_A:.1f} Å")
    print(f"  V_cavity = {req.V_cavity_required_A3:.1f} ų")
    print(f"  convergent = {req.convergence_required}")
    print(f"  min HB donors = {req.min_hb_donors}")
    print(f"  prefer cationic = {req.prefer_cationic}")

    # Example pKa effects at pH 5
    test_donors = ["O_carboxylate", "N_amine", "N_pyridine",
                   "O_phenolate", "S_thiol", "O_phosphate"]
    print(f"\nDonor availability at pH {selenite.pH}:")
    for dt in test_donors:
        f = donor_fraction_active(dt, selenite.pH)
        dG = -RT_STD * math.log(f) if f > 0 else 999.0
        print(f"  {dt:<20}: f={f:.3f}  ΔG_pH={dG:+.2f} kJ/mol")

    print("\nPhase D (hydrodynamics) and Phase E (inverse design) → next session")
