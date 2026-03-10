"""
core/receptor_guest_scorer.py — Physics-Based Receptor-Guest Binding Prediction

Scores a de novo synthetic receptor–guest pair using the same calibrated
physics parameters as the host-guest and protein-ligand scorers, but
without requiring the receptor to be in HOST_DB.

Energy terms (all from existing MABE calibration):
    1. Hydrophobic burial:  γ_flat × k_curvature × buried_SASA
    2. Cavity dehydration:  partial frustrated water release
    3. H-bond network:      matched donor-acceptor pairs × ε_neutral
    4. π-contacts:          aromatic ring overlap × ε_π_stack
    5. Conformational entropy: rotatable bond freezing penalty
    6. Size complementarity: Gaussian match between receptor cavity and guest volume

Parameters sourced from:
    hg_scorer.HG_PARAMS (γ_flat, k_curvature, dg_dehydr_per_A2)
    knowledge/hg_hbond.HBOND_PARAMS (eps_neutral, water_penalty)
    knowledge/hg_pi.PI_PARAMS (eps_pi_stack, eps_ch_pi)
    Conformational: 2.5 kJ/mol per frozen rotor (MABE convention)

Does NOT:
    - Modify unified_scorer_v2 or any existing scorer
    - Add new fitted parameters
    - Require 3D docking (uses 2D/topological descriptors)
"""

import math
from dataclasses import dataclass, field

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS PARAMETERS (from existing MABE calibrations — no new fitting)
# ═══════════════════════════════════════════════════════════════════════════

# From hg_scorer.HG_PARAMS
GAMMA_FLAT = 0.0251          # kJ/(mol·Å²) hydrophobic transfer
K_CURVATURE_CONCAVE = 1.149  # concave amplification
DG_DEHYDR_PER_A2 = -0.0703  # kJ/(mol·Å²) base dehydration
DEHYDR_SYNTHETIC = 1.0       # multiplier for synthetic receptors (between CD=0.644 and CB=3.667)

# From knowledge/hg_hbond.HBOND_PARAMS
EPS_NEUTRAL_HB = -3.0        # kJ/mol per neutral H-bond
WATER_PENALTY_PER_HB = 3.5   # kJ/mol water displacement cost
WATER_DISPLACEMENT = 1.2     # waters displaced per H-bond formed

# From knowledge/hg_pi.PI_PARAMS
EPS_PI_STACK = -4.0           # kJ/mol per π-π stacking pair
EPS_CH_PI = -1.5              # kJ/mol per CH-π contact

# Conformational
DG_PER_FROZEN_ROTOR = 2.5    # kJ/mol per frozen rotatable bond (TΔS at 298K)

# Conversion
LN10_RT = 5.71                # RT·ln(10) at 25°C in kJ/mol


# ═══════════════════════════════════════════════════════════════════════════
# RECEPTOR CAVITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReceptorCavityEstimate:
    """Estimated cavity properties of a synthetic receptor from SMILES."""
    smiles: str
    cavity_sasa_A2: float = 0.0       # estimated internal surface area
    cavity_volume_A3: float = 0.0     # estimated cavity volume
    curvature_class: str = "shallow"  # "concave", "shallow", "flat"
    n_aromatic_walls: int = 0         # aromatic rings available for π-stacking
    n_hb_donors: int = 0             # receptor H-bond donors (point inward)
    n_hb_acceptors: int = 0          # receptor H-bond acceptors
    n_rotatable_bonds: int = 0
    total_sasa_A2: float = 0.0
    mw: float = 0.0
    n_rings: int = 0
    is_macrocyclic: bool = False
    preorganization_score: float = 0.0  # 0-1, higher = more preorganized


def estimate_receptor_cavity(smiles):
    """Estimate cavity properties from receptor SMILES.

    Heuristic model:
    - Receptors with ≥2 aromatic rings + cleft/macrocyclic topology → concave
    - Cavity SASA ≈ fraction of total SASA (depends on topology)
    - Preorganization from ring count vs rotatable bond count
    """
    if not _RDKIT:
        return ReceptorCavityEstimate(smiles=smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ReceptorCavityEstimate(smiles=smiles)

    est = ReceptorCavityEstimate(smiles=smiles)
    est.mw = Descriptors.ExactMolWt(mol)
    est.n_hb_donors = Descriptors.NumHDonors(mol)
    est.n_hb_acceptors = Descriptors.NumHAcceptors(mol)
    est.n_aromatic_walls = Descriptors.NumAromaticRings(mol)
    est.n_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    est.n_rings = Descriptors.RingCount(mol)

    # Macrocyclic check
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) >= 12:
            est.is_macrocyclic = True
            break

    # Total SASA
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol_h)
        # Approximate SASA from MW correlation (faster than FreeSASA)
        est.total_sasa_A2 = est.mw * 1.8  # rough: ~1.8 Å²/Da for drug-like molecules
    except Exception:
        est.total_sasa_A2 = est.mw * 1.8

    # Cavity SASA estimation
    # Macrocyclic: ~40% of SASA faces inward
    # Cleft (≥2 aromatic rings, some rotors): ~25%
    # Open chain: ~15%
    if est.is_macrocyclic:
        cavity_fraction = 0.40
        est.curvature_class = "concave"
    elif est.n_rings >= 3 and est.n_aromatic_walls >= 2:
        cavity_fraction = 0.30
        est.curvature_class = "concave"
    elif est.n_rings >= 2:
        cavity_fraction = 0.20
        est.curvature_class = "shallow"
    else:
        cavity_fraction = 0.10
        est.curvature_class = "flat"

    est.cavity_sasa_A2 = est.total_sasa_A2 * cavity_fraction

    # Cavity volume: rough estimate from cavity SASA assuming spherical
    # V = (4/3)π(A/(4π))^(3/2) = (A^(3/2)) / (6√π)
    if est.cavity_sasa_A2 > 0:
        est.cavity_volume_A3 = est.cavity_sasa_A2 ** 1.5 / (6 * math.sqrt(math.pi))
    else:
        est.cavity_volume_A3 = 0.0

    # Preorganization: high ring count + low rotor count → preorganized
    if est.n_rings > 0:
        flex_ratio = est.n_rotatable_bonds / max(est.n_rings, 1)
        est.preorganization_score = max(0, min(1.0, 1.0 - flex_ratio * 0.2))
    else:
        est.preorganization_score = 0.1

    return est


# ═══════════════════════════════════════════════════════════════════════════
# GUEST PROPERTIES (reuse existing guest_compute)
# ═══════════════════════════════════════════════════════════════════════════

def _guest_props(smiles):
    """Get guest properties needed for scoring."""
    if not _RDKIT:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mw": Descriptors.ExactMolWt(mol),
        "logP": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "n_arom": Descriptors.NumAromaticRings(mol),
        "n_rot": Descriptors.NumRotatableBonds(mol),
        "sasa_nonpolar": Descriptors.ExactMolWt(mol) * 1.2,  # rough nonpolar SASA
        "volume_A3": Descriptors.ExactMolWt(mol) * 0.98,     # rough volume
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReceptorGuestScore:
    """Physics-based binding prediction for a receptor-guest pair."""
    receptor_smiles: str
    guest_smiles: str
    log_Ka_pred: float = 0.0
    dg_total_kJ: float = 0.0

    # Energy decomposition (kJ/mol, negative = favorable)
    dg_hydrophobic: float = 0.0
    dg_dehydration: float = 0.0
    dg_hbond: float = 0.0
    dg_pi: float = 0.0
    dg_conf_entropy: float = 0.0     # positive = unfavorable
    dg_size_match: float = 0.0

    # Receptor properties used
    receptor_cavity_sasa: float = 0.0
    receptor_preorganization: float = 0.0

    # Metadata
    n_hbonds_formed: int = 0
    n_pi_contacts: int = 0
    packing_coefficient: float = 0.0


def score_receptor_guest(
    receptor_smiles: str,
    guest_smiles: str,
) -> ReceptorGuestScore:
    """Score a synthetic receptor–guest binding pair.

    Uses calibrated HG/PL physics parameters without fitting.

    Args:
        receptor_smiles: SMILES of de novo receptor
        guest_smiles: SMILES of target guest

    Returns:
        ReceptorGuestScore with log_Ka_pred and energy decomposition.
    """
    result = ReceptorGuestScore(
        receptor_smiles=receptor_smiles,
        guest_smiles=guest_smiles,
    )

    # Get receptor cavity properties
    rec = estimate_receptor_cavity(receptor_smiles)
    result.receptor_cavity_sasa = rec.cavity_sasa_A2
    result.receptor_preorganization = rec.preorganization_score

    # Get guest properties
    gp = _guest_props(guest_smiles)
    if not gp:
        return result

    guest_np_sasa = gp["sasa_nonpolar"]
    guest_vol = gp["volume_A3"]

    # ── 1. Hydrophobic burial ──
    # Buried SASA = min(receptor cavity SASA, guest nonpolar SASA)
    buried_sasa = min(rec.cavity_sasa_A2, guest_np_sasa)

    gamma = GAMMA_FLAT
    if rec.curvature_class == "concave":
        gamma *= K_CURVATURE_CONCAVE

    result.dg_hydrophobic = -gamma * buried_sasa  # negative = favorable

    # ── 2. Cavity dehydration ──
    # Synthetic receptors release some frustrated water upon guest binding
    # Less than CB (no ureido enclosure) but more than flat surface
    result.dg_dehydration = DG_DEHYDR_PER_A2 * DEHYDR_SYNTHETIC * buried_sasa

    # ── 3. H-bond network ──
    # Count matched donor-acceptor pairs
    # Receptor donors ↔ guest acceptors
    n_hb_rec_don_guest_acc = min(rec.n_hb_donors, gp["hba"])
    # Receptor acceptors ↔ guest donors
    n_hb_rec_acc_guest_don = min(rec.n_hb_acceptors, gp["hbd"])
    n_hb_total = n_hb_rec_don_guest_acc + n_hb_rec_acc_guest_don
    result.n_hbonds_formed = n_hb_total

    # Net H-bond energy: formation − water displacement
    dg_hb_formation = EPS_NEUTRAL_HB * n_hb_total  # negative
    dg_hb_water = WATER_PENALTY_PER_HB * WATER_DISPLACEMENT * n_hb_total  # positive
    # In preorganized receptors, fewer water molecules need displacement
    # (the binding site was already desolvated)
    preorg_discount = 1.0 - rec.preorganization_score * 0.5  # 0.5-1.0
    dg_hb_water *= preorg_discount

    result.dg_hbond = dg_hb_formation + dg_hb_water

    # ── 4. π-contacts ──
    # π-π stacking: min(receptor aromatic walls, guest aromatic rings)
    n_pi_stack = min(rec.n_aromatic_walls, gp["n_arom"])
    result.n_pi_contacts = n_pi_stack
    result.dg_pi = EPS_PI_STACK * n_pi_stack

    # CH-π: guest aliphatic C-H near receptor aromatic (rough estimate)
    if rec.n_aromatic_walls > n_pi_stack:
        # Extra aromatic walls available for CH-π with guest alkyl
        n_ch_pi = min(rec.n_aromatic_walls - n_pi_stack, 2)
        result.dg_pi += EPS_CH_PI * n_ch_pi

    # ── 5. Conformational entropy ──
    # Both receptor and guest lose rotational freedom upon binding
    # Receptor: penalty reduced by preorganization
    receptor_rot_penalty = rec.n_rotatable_bonds * DG_PER_FROZEN_ROTOR
    receptor_rot_penalty *= (1.0 - rec.preorganization_score * 0.7)  # 30-100%

    # Guest: assume ~60% of rotors frozen upon binding (partial insertion)
    guest_rot_penalty = gp["n_rot"] * DG_PER_FROZEN_ROTOR * 0.6

    result.dg_conf_entropy = receptor_rot_penalty + guest_rot_penalty  # positive

    # ── 6. Size complementarity ──
    # Gaussian match: optimal when guest fills ~55% of cavity
    if rec.cavity_volume_A3 > 0 and guest_vol > 0:
        packing = guest_vol / rec.cavity_volume_A3
        result.packing_coefficient = packing
        # Optimal packing ~0.55, σ=0.2
        size_match = math.exp(-((packing - 0.55) ** 2) / (2 * 0.2 ** 2))
        # Size bonus (favorable) or penalty (unfavorable)
        result.dg_size_match = -3.0 * size_match + 1.5  # range: -1.5 to +1.5 kJ/mol

    # ── Total ──
    dg_total = (
        result.dg_hydrophobic
        + result.dg_dehydration
        + result.dg_hbond
        + result.dg_pi
        + result.dg_conf_entropy
        + result.dg_size_match
    )

    result.dg_total_kJ = dg_total
    result.log_Ka_pred = -dg_total / LN10_RT

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BATCH SCORING for de novo generator
# ═══════════════════════════════════════════════════════════════════════════

def score_receptor_batch(
    receptor_smiles_list: list,
    guest_smiles: str,
) -> list:
    """Score a batch of receptors against one guest.

    Returns list[ReceptorGuestScore] sorted by log_Ka descending.
    """
    results = []
    for rec_smi in receptor_smiles_list:
        try:
            score = score_receptor_guest(rec_smi, guest_smiles)
            results.append(score)
        except Exception:
            pass

    results.sort(key=lambda r: r.log_Ka_pred, reverse=True)
    return results
