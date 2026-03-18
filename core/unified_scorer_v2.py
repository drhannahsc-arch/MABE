"""
core/unified_scorer_v2.py — Phase 13b: Unified Scorer (No Routing Wall)

Single entry point: predict(uc) → PredictionResult
All energy terms fire independently with self-zeroing guards.
Delegates to calibrated backends:
  - scorer_frozen.predict_log_k() for metal coordination (17 terms → 1 bundled)
  - hg_scorer energy functions for host-guest (hydrophobic, dehydration, H-bond,
    π, conformational entropy, shape)
  - cross_modal_predictor energy functions for metal@CB[n] (ion-dipole,
    desolvation, portal size, CB dehydration, shape)

Backward compatibility: 644 regression tests must pass within ε=0.01 log K.
"""

import sys
import os
import math
from dataclasses import dataclass

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
for _sub in ('knowledge', 'core'):
    _p = os.path.join(_project_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.universal_schema import UniversalComplex
from core.tier2_terms import compute_all_tier2, tier2_total, TIER2_RESULT_FIELDS

# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATED BACKENDS (imported at module level for speed)
# ═══════════════════════════════════════════════════════════════════════════
from core.scorer_frozen import predict_log_k as _metal_predict_log_k

from hg_scorer import (
    dg_high_energy_water as _hg_dg_hew,
    HG_PARAMS, LN10_RT,
    compute_guest_sasa, estimate_buried_sasa,
    dg_hydrophobic as _hg_dg_hydrophobic,
    dg_cavity_dehydration as _hg_dg_cavity_dehydration,
    dg_size_mismatch as _hg_dg_size_mismatch,
)
from knowledge.hg_hbond import compute_dg_hbond as _hg_compute_dg_hbond, HBOND_PARAMS

# pH-aware protonation state estimation (SupraBank enrichment)
try:
    from core.pka_estimator import enrich_uc_protonation as _enrich_pka
    _PKA_AVAILABLE = True
except ImportError:
    _PKA_AVAILABLE = False

try:
    from core.inclusion_classifier import classify_inclusion as _classify_inclusion
    _INCLUSION_AVAILABLE = True
except ImportError:
    _INCLUSION_AVAILABLE = False
from knowledge.hg_pi import compute_dg_pi as _hg_compute_dg_pi, PI_PARAMS
from knowledge.hg_conf_shape import (
    compute_dg_conf_shape as _hg_compute_dg_conf_shape,
    CONF_SHAPE_PARAMS, HOST_CAVITY_VOLUME,
)
from knowledge.hg_dataset import HOST_DB as _HG_HOST_DB

from cross_modal_predictor import (
    dg_ion_dipole as _cm_dg_ion_dipole,
    dg_desolvation as _cm_dg_desolvation,
    dg_portal_size_match as _cm_dg_portal_size_match,
    dg_cb_dehydration as _cm_dg_cb_dehydration,
    dg_shape_complementarity as _cm_dg_shape_complementarity,
    CM_PARAMS,
)
from knowledge.cross_modal_dataset import CB_PORTAL_INFO

# Physics-based PL scorer (parallel to QSAR path, graceful fallback)
try:
    from knowledge.physics_pl_scorer import compute_physics_pl_terms as _compute_physics_pl
except ImportError:
    def _compute_physics_pl(uc, result):
        pass  # No-op if module unavailable



# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """Full energy decomposition for one UniversalComplex."""
    name: str
    binding_mode: str
    log_Ka_exp: float
    log_Ka_pred: float
    dg_total_kj: float
    error: float

    # Metal coordination (bundled — 17 internal terms)
    dg_metal: float = 0.0

    # Host-guest inclusion
    dg_hydrophobic: float = 0.0
    dg_cavity_dehydration: float = 0.0
    dg_hbond: float = 0.0
    dg_pi: float = 0.0
    dg_conf_entropy: float = 0.0
    dg_shape: float = 0.0
    dg_size_mismatch: float = 0.0

    # Cross-modal (metal@host)

    # Tier 2 interaction terms
    dg_dispersion_t2: float = 0.0
    dg_cation_pi: float = 0.0
    dg_pi_stack: float = 0.0
    dg_halogen_bond: float = 0.0
    dg_salt_bridge: float = 0.0
    dg_born_solvation: float = 0.0
    dg_hbond_coop: float = 0.0
    dg_anion_pi: float = 0.0
    dg_metallophilic: float = 0.0
    dg_group_desolv: float = 0.0
    dg_water_penalty: float = 0.0
    dg_ion_dipole: float = 0.0
    dg_ion_desolv: float = 0.0
    dg_portal_size: float = 0.0
    dg_cm_dehydration: float = 0.0
    dg_cm_shape: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HOST-TYPE INFERENCE MAPS
# Used when scoring from a UC that has host_name/host_type but no raw dict
# ═══════════════════════════════════════════════════════════════════════════

# Map host keys to hg_dataset HOST_DB keys
_HOST_KEY_MAP = {
    # Cyclodextrins
    "α-cyclodextrin": "alpha-CD", "alpha-CD": "alpha-CD",
    "β-cyclodextrin": "beta-CD",  "beta-CD": "beta-CD",
    "γ-cyclodextrin": "gamma-CD", "gamma-CD": "gamma-CD",
    # Cucurbiturils (HOST_REGISTRY full names + bracket variants)
    "cucurbit[5]uril": "CB5", "CB5": "CB5", "CB[5]": "CB5",
    "cucurbit[6]uril": "CB6", "CB6": "CB6", "CB[6]": "CB6",
    "cucurbit[7]uril": "CB7", "CB7": "CB7", "CB[7]": "CB7",
    "cucurbit[8]uril": "CB8", "CB8": "CB8", "CB[8]": "CB8",
    # Calixarenes
    "p-sulfonatocalix[4]arene": "calix4-SO3", "calix4-SO3": "calix4-SO3",
    "sulfonato-calix4arene": "calix4-SO3",
    # Pillararenes
    "pillar[5]arene": "pillar5", "pillar5": "pillar5",
    "pillar5arene": "pillar5",
    # Crown ethers / cryptands (no HOST_DB entries yet — mapped for future)
    "12-crown-4": "12-crown-4", "15-crown-5": "15-crown-5",
    "18-crown-6": "18-crown-6", "[2.2.2]cryptand": "cryptand-222",
    "cryptand-222": "cryptand-222",
}


def _resolve_host_key(uc):
    """Try to resolve a HOST_DB key from UC host_name."""
    for candidate in (uc.host_name, uc.host_type):
        if candidate in _HG_HOST_DB:
            return candidate
        if candidate in _HOST_KEY_MAP:
            return _HOST_KEY_MAP[candidate]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# NOVEL HOST FALLBACK: synthesize HOST_DB-compatible dict from UC fields
# ═══════════════════════════════════════════════════════════════════════════

def _synthesize_host_dict(uc):
    """Build a HOST_DB-compatible dict from UC cavity properties.

    Used when host_name is not in HOST_DB but cavity_volume_A3 > 0,
    enabling scoring of novel hosts (MOFs, synthetic receptors, etc.).

    Geometry: sphere approximation (conservative — underpredicts SASA
    for elongated cavities, which underpredicts Ka. Safe direction.)
    """
    import math
    vol = uc.cavity_volume_A3
    if vol <= 0:
        return None
    d = 2.0 * (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    sasa = math.pi * d * d
    return {
        "full_name": uc.host_name or "novel_host",
        "cavity_diameter": d,
        "cavity_depth": d,  # sphere: depth = diameter
        "cavity_sasa": sasa,
        "portal_type": getattr(uc, 'portal_type', 'neutral'),
        "curvature_class": "concave",
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER: UC → hg_scorer dict format
# ═══════════════════════════════════════════════════════════════════════════

def _uc_to_hg_entry(uc, host_key):
    """Build a dict compatible with hg_scorer.predict_hg_log_ka()."""
    return {
        "name": uc.name,
        "host": host_key,
        "guest_smiles": uc.guest_smiles,
        "guest_charge": uc.guest_charge,
        "n_hbonds_portal": uc.n_hbonds_formed,
        "guest_has_cation": uc.guest_charge > 0,
        "log_Ka": uc.log_Ka_exp,
        "source": getattr(uc, 'source', ''),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER: UC → cross_modal_predictor dict format
# ═══════════════════════════════════════════════════════════════════════════

def _uc_to_cm_entry(uc, cb_host_key):
    """Build a dict compatible with cross_modal_predictor functions."""
    portal = CB_PORTAL_INFO.get(cb_host_key, {})
    return {
        "name": uc.name,
        "metal": uc.metal_formula,
        "cb_host": cb_host_key,
        "n_portal_carbonyls": portal.get("n_carbonyls", 0),
        "portal_radius_nm": portal.get("portal_radius_nm", 0.0),
        "cavity_sasa": portal.get("cavity_sasa", 0.0),
        "cavity_volume_A3": uc.cavity_volume_A3,
        "log_Ka": uc.log_Ka_exp,
        "mode": "cross_modal",
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION ENGINE — NO ROUTING WALL
# ═══════════════════════════════════════════════════════════════════════════

def predict(uc, verbose=False):
    """Score any UniversalComplex. All terms self-zero when inputs absent.

    Args:
        uc: UniversalComplex (from auto_descriptor or manual construction)
        verbose: print energy decomposition

    Returns:
        PredictionResult with full term breakdown
    """
    result = PredictionResult(
        name=uc.name,
        binding_mode=uc.binding_mode,
        log_Ka_exp=uc.log_Ka_exp,
        log_Ka_pred=0.0,
        dg_total_kj=0.0,
        error=0.0,
    )

    # ── METAL COORDINATION (self-zeros if no metal) ──────────────────
    _compute_metal(uc, result)

    # ── pH-aware protonation enrichment (before HG scoring) ──────────
    if _PKA_AVAILABLE and uc.binding_mode == "host_guest_inclusion":
        _enrich_pka(uc, ph=uc.ph if hasattr(uc, 'ph') else 7.0)

    # ── HOST-GUEST INCLUSION (self-zeros if no cavity/guest) ─────────
    _compute_hg_terms(uc, result)

    # ── PROTEIN-LIGAND non-covalent (self-zeros if not metalloprotein) ─
    _compute_protein_ligand_terms(uc, result)

    # ── GENERAL PROTEIN-LIGAND (non-metal targets) ────────────────────
    _compute_general_pl_terms(uc, result)

    # ── PHYSICS-BASED PROTEIN-LIGAND (parallel to QSAR) ──────────────
    _compute_physics_pl(uc, result)

    # ── GLYCAN-LECTIN (self-zeros if not glycan mode) ────────────────
    _compute_glycan_terms(uc, result)

    # ── CROSS-MODAL metal@host (self-zeros if no metal+cavity) ───────
    _compute_cm_terms(uc, result)

    # ── TIER 2 INTERACTION TERMS (self-zero when inputs absent) ──
    compute_all_tier2(uc, result)

    # ── SUM AND CONVERT ──────────────────────────────────────────────
    dg_net = (result.dg_metal
              + result.dg_hydrophobic + result.dg_cavity_dehydration
              + result.dg_hbond + result.dg_pi
              + result.dg_conf_entropy + result.dg_shape
              + result.dg_size_mismatch
              + result.dg_ion_dipole + result.dg_ion_desolv
              + result.dg_portal_size + result.dg_cm_dehydration
              + result.dg_cm_shape
              + tier2_total(result))

    result.dg_total_kj = dg_net
    result.log_Ka_pred = -dg_net / LN10_RT
    result.error = result.log_Ka_pred - uc.log_Ka_exp

    if verbose:
        _print_decomposition(uc, result)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# METAL COORDINATION — delegates to scorer_frozen
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# DATA-PRESENCE ROUTING HELPERS (replace binding_mode string gates)
# ═══════════════════════════════════════════════════════════════════════════

def _is_cross_modal(uc):
    """Detect metal@host (cross-modal) from data, not binding_mode string."""
    if not uc.metal_formula:
        return False
    for candidate in (uc.host_name, uc.host_type):
        if not candidate:
            continue
        if candidate in CB_PORTAL_INFO:
            return True
        stripped = candidate.replace("[", "").replace("]", "")
        if stripped in CB_PORTAL_INFO:
            return True
    return False


def _compute_metal(uc, result):
    """Metal coordination: 17 physics terms bundled into dg_metal.

    Self-zeros if uc.metal_formula is empty or donor_subtypes is empty.
    """
    if not uc.metal_formula or not uc.donor_subtypes:
        return

    # Only fire for metal_coordination binding mode (not for metal@host
    # where the CM terms handle it). If binding_mode is cross_modal,
    # the metal scoring path should NOT fire — the CM scorer handles it.
    if _is_cross_modal(uc):
        return

    try:
        log_k = _metal_predict_log_k(
            uc.metal_formula,
            uc.donor_subtypes,
            chelate_rings=uc.chelate_rings,
            ring_sizes=uc.ring_sizes or None,
            pH=uc.ph,
            is_macrocyclic=uc.is_macrocyclic,
            cavity_radius_nm=uc.cavity_radius_nm if uc.cavity_radius_nm > 0 else None,
            n_ligand_molecules=uc.n_ligand_molecules,
            temperature_K=298.15,
        )
        # Convert log K back to ΔG for decomposition
        result.dg_metal = -log_k * LN10_RT
    except (ValueError, KeyError):
        result.dg_metal = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# HOST-GUEST TERMS — delegates to hg_scorer functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_hg_terms(uc, result):
    """Host-guest inclusion terms. Self-zeros if no guest SMILES or no host.

    Does NOT fire for cross-modal (metal@CB) — those are handled by CM terms.
    """
    if _is_cross_modal(uc):
        return
    if not uc.guest_smiles and uc.guest_sasa_nonpolar_A2 <= 0:
        return

    # HOST_DB lookup with novel-host fallback
    host_key = _resolve_host_key(uc)
    _is_novel_host = False
    if host_key is not None and host_key in _HG_HOST_DB:
        host = _HG_HOST_DB[host_key]
    else:
        # Novel host: synthesize from UC cavity properties
        host = _synthesize_host_dict(uc)
        if host is None:
            return
        _is_novel_host = True
        host_key = host["full_name"]

    hg_entry = _uc_to_hg_entry(uc, host_key)

    # Guest SASA with fallback to pre-populated UC fields
    guest_np = 0.0
    if uc.guest_smiles:
        try:
            sasa = compute_guest_sasa(uc.guest_smiles)
            guest_np = sasa["nonpolar_sasa"]
        except Exception:
            guest_np = uc.guest_sasa_nonpolar_A2
    else:
        guest_np = uc.guest_sasa_nonpolar_A2

    if guest_np <= 0:
        return

    cavity_sasa = host["cavity_sasa"]
    buried = estimate_buried_sasa(guest_np, cavity_sasa)

    # 1. Hydrophobic transfer
    result.dg_hydrophobic = _hg_dg_hydrophobic(buried, host["curvature_class"])

    # 2. Cavity dehydration
    # 2. Cavity dehydration (SASA-based + PC-dependent scaling)
    result.dg_cavity_dehydration = _hg_dg_cavity_dehydration(buried, host_key, uc.packing_coefficient)

    # 2b. High-energy water displacement — TESTED, NOT DEPLOYED
    # Classifier correctly identifies inclusion depth (axial × radial fill).
    # At per-water energies that improve top binders (~1.5 logKa), bulk entries
    # overshoot by ~0.7 logKa, worsening overall MAE from 1.745 to 1.892.
    # Kept dormant until a selective model can distinguish the ~15% of entries
    # where HEW is the dominant binding contribution.
    # Infrastructure: core/inclusion_classifier.py (classify_inclusion)
    #                 hg_scorer.dg_high_energy_water() + HEW_PARAMS

    # 3. Size mismatch (original linear penalty — kept for backward compatibility)
    result.dg_size_mismatch = _hg_dg_size_mismatch(guest_np, cavity_sasa)

    # 3b. Repulsion physics: VdW overlap + steric clash (additive, fires only for packing > 1)
    #     These are ADDITIONAL penalties on top of the existing size_mismatch.
    #     Zero for all calibration data (packing ≤ 1.0).
    if uc.packing_coefficient > 1.0:
        from core.repulsion_hg import dg_vdw_overlap
        result.dg_size_mismatch += dg_vdw_overlap(uc.packing_coefficient)

    # 3c. Electrostatic charge-charge repulsion (host charge × guest charge)
    #     CDs are neutral (zero). CB portals handled separately.
    #     Fires for charged hosts like sulfonato-calixarene (negative).
    if uc.host_charge != 0 and uc.guest_charge != 0:
        from core.repulsion_hg import dg_electrostatic
        cav_d_A = host.get("cavity_diameter", 10.0)  # HOST_DB stores in Å
        elec = dg_electrostatic(int(uc.host_charge), int(uc.guest_charge), cav_d_A)
        # Only add repulsive part to size_mismatch; attractive part goes into dg_hbond
        if elec > 0:
            result.dg_size_mismatch += elec
        elif elec < 0:
            result.dg_hbond += elec  # electrostatic attraction grouped with non-covalent

    # 4. H-bond network
    #    For novel hosts, inject synthesized dict into a local copy of HOST_DB
    _local_host_db = _HG_HOST_DB
    if _is_novel_host:
        _local_host_db = dict(_HG_HOST_DB)
        _local_host_db[host_key] = host
    try:
        result.dg_hbond = _hg_compute_dg_hbond(hg_entry, _local_host_db)
    except Exception:
        result.dg_hbond = 0.0

    # 5. π-interactions
    try:
        result.dg_pi = _hg_compute_dg_pi(hg_entry, _local_host_db)
    except Exception:
        result.dg_pi = 0.0

    # 6+7. Conformational entropy + Shape complementarity
    try:
        dg_conf, dg_shape = _hg_compute_dg_conf_shape(hg_entry)
        result.dg_conf_entropy = dg_conf
        result.dg_shape = dg_shape
    except Exception:
        pass

    # 8. Portal restriction penalty (CB[n] hosts only)
    #
    # CB[n] portals are rigid ureido rims, much narrower than the equatorial
    # cavity. Guests whose minimum cross-section exceeds the portal area
    # cannot enter. This adds a penalty proportional to the excess area^1.5.
    #
    # Physics: Barrow et al. portal diameters (Å): CB5=2.4, CB6=3.9, CB7=5.4, CB8=6.9
    # Portal flexibility: ~1 Å expansion for spherical guests (Nau group MD).
    # Non-spherical guests get less expansion (sphericity-weighted).
    #
    # Calibration: zero penalty for all known CB7 binders (adamantane, hexylamine,
    # ferrocene etc.) — verified against frozen regression references.
    if host_key.startswith("CB") and uc.guest_smiles:
        portal_penalty = _cb_portal_penalty(uc.guest_smiles, host_key, host)
        if portal_penalty > 0:
            result.dg_portal_size += portal_penalty  # positive = unfavorable


# ═══════════════════════════════════════════════════════════════════════════
# CB PORTAL RESTRICTION — rigid ureido rim gate
# ═══════════════════════════════════════════════════════════════════════════

# Portal aperture diameters from Barrow, Kasera, Rowland, del Barrio,
# Scherman, Clemmer & Bush, and Nau reviews. Inner portal diameter (Å).
_CB_PORTAL_DIAMETERS = {
    "CB5": 2.4,
    "CB6": 3.9,
    "CB7": 5.4,
    "CB8": 6.9,
}

# Penalty parameters (physics-derived, not fitted against HG data)
_PORTAL_K = 2.0        # kJ/mol per Å² excess^1.5
_PORTAL_POWER = 1.5    # super-linear: large excess → steep penalty
_PORTAL_FLEX = 1.0     # Å: max portal expansion for perfectly spherical guest
_PORTAL_CAP = 4.08      # kJ/mol: sigmoid cap on portal penalty (SupraBank fit)

# Cache for guest 3D dimensions {canonical_smiles: (min_d, mid_d, max_d)}
_GUEST_DIM_CACHE = {}


def _guest_dimensions_3d(smiles):
    """Compute guest molecular dimensions from 3D conformer.

    Returns (min_d, mid_d, max_d) in Å, or None if embedding fails.
    Uses SVD on heavy-atom coordinates + VdW radii.
    """
    if smiles in _GUEST_DIM_CACHE:
        return _GUEST_DIM_CACHE[smiles]

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol is None:
            return None

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer(0)

        coords = []
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])

        if len(coords) < 3:
            return None

        coords = np.array(coords)
        coords -= coords.mean(axis=0)

        _, _, Vt = np.linalg.svd(coords, full_matrices=False)
        extents = []
        for i in range(min(3, Vt.shape[0])):
            proj = coords @ Vt[i]
            extents.append(proj.max() - proj.min() + 3.4)  # +2×VdW radius
        extents.sort()

        result = tuple(extents)
        _GUEST_DIM_CACHE[smiles] = result
        return result
    except Exception:
        return None


def _cb_portal_penalty(guest_smiles, host_key, host_data):
    """Compute portal restriction penalty for CB hosts.

    Returns penalty in kJ/mol (positive = unfavorable, 0 = no penalty).
    """
    import math

    # Get portal diameter for this CB host
    portal_d = _CB_PORTAL_DIAMETERS.get(host_key)
    if portal_d is None:
        # Unknown CB variant — estimate from cavity diameter - 2.0
        cav_d = host_data.get("cavity_diameter", 0)
        if cav_d > 0:
            portal_d = max(2.0, cav_d - 2.0)
        else:
            return 0.0

    # Get guest 3D dimensions
    dims = _guest_dimensions_3d(guest_smiles)
    if dims is None:
        return 0.0

    min_d, mid_d = dims[0], dims[1]

    # Sphericity: how close to spherical is the guest cross-section
    sphericity = min_d / mid_d if mid_d > 0.1 else 1.0

    # Flexible portal diameter: portal expands ~1 Å for spherical guests,
    # less for non-spherical (can't stretch asymmetrically)
    flex_portal_d = portal_d + _PORTAL_FLEX * sphericity

    # Compare cross-sectional areas (elliptical guest vs circular portal)
    flex_portal_area = math.pi * (flex_portal_d / 2) ** 2
    guest_cross_area = math.pi / 4 * min_d * mid_d

    excess_area = max(0.0, guest_cross_area - flex_portal_area)
    if excess_area <= 0:
        return 0.0

    # Super-linear penalty: small excess → small penalty, large excess → large
    raw = _PORTAL_K * excess_area ** _PORTAL_POWER
    return min(raw, _PORTAL_CAP)


# ═══════════════════════════════════════════════════════════════════════════
# PROTEIN-LIGAND NON-COVALENT TERMS — Phase 14a + PL Calibration
# 5 physics descriptors + per-target offsets, fitted against 300 ChEMBL entries.
# 11 params total, 300 data = 27:1 ratio. Metal scorer untouched.
# ═══════════════════════════════════════════════════════════════════════════

PL_PARAMS = {
    "a_sasa":  -0.05825,   # kJ/mol per Å² buried SASA (2.3× HG γ_flat)
    "b_logP":   0.2011,    # log Ka per logP unit (lipophilic efficiency)
    "c_rot":   -0.1362,    # log Ka per rotatable bond (size proxy)
    "d_hb":     0.2453,    # log Ka per H-bond count (heuristic correction)
    "e_mw":     0.5277,    # log Ka per 100 Da (molecular size)
}

# Per-target offsets absorb systematic difference between metal scorer
# absolute prediction and protein-context binding. For novel targets
# without a calibrated offset, the model falls back to offset=0.
PL_TARGET_OFFSETS = {
    "ACE":         +3.360,
    "CA-II":       +0.441,
    "MMP-13":      +1.662,
    "MMP-7":       -0.092,
    "MMP-9":       +1.699,
    "Thermolysin": +0.947,
}


def _compute_protein_ligand_terms(uc, result):
    """Non-covalent terms for metalloprotein-ligand binding.

    Fires when has_metalloprotein_data flag is set.
    Uses 5 calibrated PL descriptors + per-target offset.
    Metal coordination term is computed separately by _compute_metal.

    Self-zeros if metalloprotein data absent or guest properties absent.
    """
    if not uc.has_metalloprotein_data:
        return
    if not uc.guest_smiles or uc.guest_sasa_nonpolar_A2 <= 0:
        return

    p = PL_PARAMS

    # 1. Hydrophobic burial: SASA-based
    if uc.sasa_buried_A2 > 0:
        result.dg_hydrophobic = p["a_sasa"] * uc.sasa_buried_A2

    # 2–5. logP, rotors, H-bonds, MW → converted to kJ/mol
    dg_descriptors = (
        p["b_logP"] * uc.guest_logP * LN10_RT
        + p["c_rot"] * uc.guest_rotatable_bonds * LN10_RT
        + p["d_hb"] * uc.n_hbonds_formed * LN10_RT
        + p["e_mw"] * uc.guest_mw * 0.01
    )
    # Pack descriptor terms into conf_entropy (reuse existing result field)
    result.dg_conf_entropy = dg_descriptors

    # Per-target offset (falls back to 0 for unknown targets)
    target = uc.host_name
    offset = PL_TARGET_OFFSETS.get(target, 0.0)
    result.dg_shape = offset * LN10_RT


# ═══════════════════════════════════════════════════════════════════════════
# GENERAL PROTEIN-LIGAND (non-metal targets) — Phase 18
# 8 2D descriptors + per-target offsets, fitted against 2000 ChEMBL entries
# across 5 non-metal targets (DHFR, HIV protease, COX-2, trypsin, thrombin).
# No metal term. Binding driven entirely by non-covalent physics.
# ═══════════════════════════════════════════════════════════════════════════

GENERAL_PL_PARAMS = {
    "a_logP":      +0.1928,    # legacy fallback — used only for unknown targets
    "b_mw":        +0.1988,
    "c_rot":       -0.0140,
    "d_hbd":       -0.0987,
    "e_hba":       +0.1028,
    "f_tpsa":      +0.1022,
    "g_arom":      -0.0345,
    "h_fsp3":      +0.5548,
}

GENERAL_PL_TARGET_OFFSETS = {
    "COX-2":                        +4.239,
    "Dihydrofolate reductase":      +4.436,
    "HIV-1 protease":               +5.523,
    "Thrombin":                     +4.660,
    "Trypsin":                      +5.284,
}

# ── Per-target Ridge models (20 features each) ──
# Calibrated on ChEMBL data (400-800 entries per target, 5-fold CV).
# Features: 8 whole-molecule + 4 Gasteiger charge stats + 8 topological indices.
# Average within-target Pearson r = 0.548 (vs 0.345 for 8-feature global model).
PER_TARGET_PL_MODELS = {
    "COX-2": {
        "intercept": 5.4032,
        "coefficients": {
            "logP": 0.223883, "mw": -0.102092, "rot": 0.071854,
            "hbd": 0.065111, "hba": 0.256415, "tpsa": -1.157652,
            "arom": 0.025584, "fsp3": 1.251802,
            "q_mean": -0.352824, "q_std": 0.541457,
            "q_min": -1.560208, "q_max": -3.900301,
            "chi1": -0.684645, "chi2n": 0.188214,
            "bertz": 0.433266, "hk_alpha": -0.160272,
            "kappa2": -0.114779, "kappa3": 0.522968,
            "aliph_rings": 0.845295, "sat_rings": -0.036815,
        },
    },
    "Dihydrofolate reductase": {
        "intercept": 6.6077,
        "coefficients": {
            "logP": 0.15907, "mw": -0.592046, "rot": 0.042493,
            "hbd": 0.125532, "hba": 0.002836, "tpsa": 0.311327,
            "arom": 0.143698, "fsp3": 0.084559,
            "q_mean": -0.1935, "q_std": 0.336999,
            "q_min": -0.402494, "q_max": 0.89356,
            "chi1": -0.043946, "chi2n": -0.263065,
            "bertz": 0.211869, "hk_alpha": 0.40603,
            "kappa2": 0.115548, "kappa3": 0.107576,
            "aliph_rings": 0.367138, "sat_rings": 0.093623,
        },
    },
    "HIV-1 protease": {
        "intercept": 6.4152,
        "coefficients": {
            "logP": 0.075736, "mw": -0.531338, "rot": -0.014523,
            "hbd": 0.062451, "hba": 0.12781, "tpsa": -0.201139,
            "arom": -0.347659, "fsp3": 0.185748,
            "q_mean": -0.182852, "q_std": -0.920776,
            "q_min": 0.793644, "q_max": 0.910586,
            "chi1": 0.494475, "chi2n": -0.158262,
            "bertz": 0.061937, "hk_alpha": 0.081766,
            "kappa2": -0.065421, "kappa3": -0.303736,
            "aliph_rings": -0.23828, "sat_rings": -0.00139,
        },
    },
    "Trypsin": {
        "intercept": 3.6302,
        "coefficients": {
            "logP": 0.131537, "mw": 0.186642, "rot": 0.134934,
            "hbd": 0.042008, "hba": 0.253747, "tpsa": -0.093819,
            "arom": -0.822433, "fsp3": -1.285639,
            "q_mean": 0.572056, "q_std": 3.688903,
            "q_min": -4.030153, "q_max": 4.313006,
            "chi1": -0.43908, "chi2n": 0.523403,
            "bertz": 0.245497, "hk_alpha": -0.128856,
            "kappa2": 0.00444, "kappa3": -0.30535,
            "aliph_rings": -0.610561, "sat_rings": 0.065483,
        },
    },
    "Thrombin": {
        "intercept": 6.7873,
        "coefficients": {
            "logP": -0.000913, "mw": -0.609443, "rot": 0.281582,
            "hbd": -0.303052, "hba": -0.383938, "tpsa": 1.75087,
            "arom": 0.207892, "fsp3": 0.446049,
            "q_mean": -0.491461, "q_std": -1.874569,
            "q_min": -0.075337, "q_max": 0.516235,
            "chi1": 0.281265, "chi2n": -0.053339,
            "bertz": 0.270676, "hk_alpha": 0.94006,
            "kappa2": -0.39282, "kappa3": 0.141772,
            "aliph_rings": 0.225482, "sat_rings": -0.045803,
        },
    },
    # ── Kinases (ChEMBL Ki data, Ridge alpha=1.0) ──
    "EGFR": {
        "intercept": 4.2746,
        "coefficients": {
            "logP": 0.088028, "mw": 0.397187, "rot": 0.268125,
            "hbd": 0.169637, "hba": 0.281289, "tpsa": -1.808629,
            "arom": -0.164143, "fsp3": 1.600713,
            "q_mean": 0.051685, "q_std": -0.384189,
            "q_min": 0.623696, "q_max": -1.798649,
            "chi1": 0.793173, "chi2n": -0.559817,
            "bertz": -0.025987, "hk_alpha": 0.098181,
            "kappa2": -0.905222, "kappa3": 0.086048,
            "aliph_rings": -0.128447, "sat_rings": 0.091288,
        },
    },
    "CDK2": {
        "intercept": 5.6089,
        "coefficients": {
            "logP": 0.039055, "mw": 0.593981, "rot": 0.129246,
            "hbd": -0.030769, "hba": 0.054017, "tpsa": -0.001788,
            "arom": 0.098819, "fsp3": 0.601528,
            "q_mean": 0.193378, "q_std": -0.449188,
            "q_min": 0.615331, "q_max": -1.47751,
            "chi1": 0.256645, "chi2n": -0.164738,
            "bertz": -0.132004, "hk_alpha": -0.335497,
            "kappa2": -0.734023, "kappa3": 0.342092,
            "aliph_rings": -0.122715, "sat_rings": 0.246084,
        },
    },
    "ABL1": {
        "intercept": 5.5315,
        "coefficients": {
            "logP": 0.317374, "mw": -0.361212, "rot": 0.05014,
            "hbd": 0.193795, "hba": 0.066828, "tpsa": -0.554171,
            "arom": -0.53221, "fsp3": 0.021995,
            "q_mean": 0.09778, "q_std": -0.841607,
            "q_min": -0.476435, "q_max": -0.076448,
            "chi1": -0.027077, "chi2n": -0.019846,
            "bertz": 0.496668, "hk_alpha": 0.29221,
            "kappa2": -0.353607, "kappa3": 0.31867,
            "aliph_rings": -0.430895, "sat_rings": 0.729692,
        },
    },
    "VEGFR2": {
        "intercept": 4.0827,
        "coefficients": {
            "logP": 0.483058, "mw": 0.061895, "rot": -0.007995,
            "hbd": 0.343082, "hba": 0.092696, "tpsa": -0.347116,
            "arom": 0.205088, "fsp3": 2.435416,
            "q_mean": -0.082382, "q_std": 0.476627,
            "q_min": 0.451974, "q_max": 1.7331,
            "chi1": -0.589616, "chi2n": -0.02876,
            "bertz": 0.107061, "hk_alpha": -0.591305,
            "kappa2": 0.272793, "kappa3": 0.219647,
            "aliph_rings": 0.472862, "sat_rings": -0.059494,
        },
    },
    # ── GPCRs (ChEMBL Ki data, Ridge alpha=1.0) ──
    "Dopamine D2": {
        "intercept": 7.3348,
        "coefficients": {
            "logP": 0.139968, "mw": -0.188999, "rot": 0.219705,
            "hbd": 0.310172, "hba": 0.049079, "tpsa": 0.219371,
            "arom": -0.124433, "fsp3": 0.269196,
            "q_mean": -0.04416, "q_std": 0.314402,
            "q_min": 0.50918, "q_max": 0.73687,
            "chi1": -0.047908, "chi2n": -0.118902,
            "bertz": 0.126561, "hk_alpha": 0.37621,
            "kappa2": -0.214228, "kappa3": 0.03616,
            "aliph_rings": 0.588181, "sat_rings": -0.053553,
        },
    },
    "Adenosine A2a": {
        "intercept": 3.3277,
        "coefficients": {
            "logP": -0.246338, "mw": 0.552509, "rot": -0.054286,
            "hbd": -0.219525, "hba": -0.104875, "tpsa": 0.925237,
            "arom": 0.9549, "fsp3": -0.604192,
            "q_mean": 0.069143, "q_std": 0.285777,
            "q_min": -1.972443, "q_max": -0.239048,
            "chi1": 0.521494, "chi2n": -0.021735,
            "bertz": -0.745852, "hk_alpha": -0.057672,
            "kappa2": -0.246449, "kappa3": 0.341971,
            "aliph_rings": 0.207744, "sat_rings": -0.324819,
        },
    },
    "Histamine H1": {
        "intercept": 3.72,
        "coefficients": {
            "logP": -0.245447, "mw": 0.504527, "rot": -0.382202,
            "hbd": -0.438397, "hba": 0.637375, "tpsa": -7.111563,
            "arom": -2.637738, "fsp3": -4.296841,
            "q_mean": 0.26221, "q_std": -1.041743,
            "q_min": -2.80381, "q_max": 0.423759,
            "chi1": 0.804343, "chi2n": 0.718829,
            "bertz": -0.006851, "hk_alpha": -0.353139,
            "kappa2": 0.250136, "kappa3": -1.198609,
            "aliph_rings": -1.366872, "sat_rings": 0.0998,
        },
    },
    # ── Acetylcholinesterase ──
    "Acetylcholinesterase": {
        "intercept": 5.8158,
        "coefficients": {
            "logP": 0.030014, "mw": -1.760867, "rot": 0.269434,
            "hbd": -0.368351, "hba": -0.547004, "tpsa": 3.559808,
            "arom": 0.393982, "fsp3": 0.147587,
            "q_mean": -0.644788, "q_std": -0.822982,
            "q_min": -1.579567, "q_max": 0.621655,
            "chi1": 0.975168, "chi2n": -0.35279,
            "bertz": 0.294049, "hk_alpha": 2.149961,
            "kappa2": -0.410571, "kappa3": 0.080439,
            "aliph_rings": 0.209954, "sat_rings": -0.205072,
        },
    },
    # ── Androgen receptor ──
    "Androgen receptor": {
        "intercept": 7.9399,
        "coefficients": {
            "logP": 0.05957, "mw": -0.554301, "rot": -0.098047,
            "hbd": 0.149663, "hba": 0.070202, "tpsa": -0.075652,
            "arom": 0.047312, "fsp3": 1.538576,
            "q_mean": 0.135325, "q_std": -0.000224,
            "q_min": 0.833701, "q_max": -0.607009,
            "chi1": -0.507978, "chi2n": -0.155314,
            "bertz": 0.36819, "hk_alpha": 0.002948,
            "kappa2": 0.554887, "kappa3": 0.033793,
            "aliph_rings": -0.184461, "sat_rings": 0.4776,
        },
    },
    # ── BACE1 ──
    "BACE1": {
        "intercept": 4.5669,
        "coefficients": {
            "logP": 0.430664, "mw": -0.318327, "rot": -0.21973,
            "hbd": -0.299475, "hba": 0.158619, "tpsa": 2.222703,
            "arom": -0.771488, "fsp3": -0.09335,
            "q_mean": 1.400926, "q_std": -2.076138,
            "q_min": -1.267879, "q_max": -0.354002,
            "chi1": 0.599309, "chi2n": -0.268458,
            "bertz": -0.270638, "hk_alpha": -0.164282,
            "kappa2": 3.1e-05, "kappa3": -0.003268,
            "aliph_rings": -0.902496, "sat_rings": 0.567332,
        },
    },
    # ── Estrogen receptor alpha ──
    "Estrogen receptor alpha": {
        "intercept": 6.6819,
        "coefficients": {
            "logP": 0.25848, "mw": -0.655756, "rot": -0.226642,
            "hbd": 0.357494, "hba": -0.116164, "tpsa": -0.196787,
            "arom": -0.037171, "fsp3": -1.067626,
            "q_mean": -0.208489, "q_std": 1.517823,
            "q_min": -1.013487, "q_max": -2.718618,
            "chi1": 0.647875, "chi2n": -0.617247,
            "bertz": -0.109442, "hk_alpha": 0.749193,
            "kappa2": 0.269142, "kappa3": -0.006958,
            "aliph_rings": -0.126124, "sat_rings": 0.519753,
        },
    },
    # ── Glucocorticoid receptor ──
    "Glucocorticoid receptor": {
        "intercept": 7.8833,
        "coefficients": {
            "logP": 0.231714, "mw": -0.362921, "rot": 0.154996,
            "hbd": -0.011236, "hba": 0.320874, "tpsa": -0.736188,
            "arom": -0.590037, "fsp3": -1.226136,
            "q_mean": 0.119716, "q_std": 0.007895,
            "q_min": -0.49357, "q_max": 2.236083,
            "chi1": -0.200447, "chi2n": -0.083829,
            "bertz": 0.329073, "hk_alpha": -0.298355,
            "kappa2": -0.370277, "kappa3": 0.455332,
            "aliph_rings": 0.298911, "sat_rings": 0.07061,
        },
    },
    # ── JAK2 ──
    "JAK2": {
        "intercept": 4.7896,
        "coefficients": {
            "logP": -0.249873, "mw": 0.915978, "rot": -0.096721,
            "hbd": -0.277726, "hba": -0.212739, "tpsa": 1.224216,
            "arom": -0.154613, "fsp3": -1.714687,
            "q_mean": 0.406368, "q_std": -0.39626,
            "q_min": -0.781282, "q_max": -1.14928,
            "chi1": -0.277084, "chi2n": 0.407932,
            "bertz": 0.12584, "hk_alpha": 0.047649,
            "kappa2": 0.098861, "kappa3": -0.151673,
            "aliph_rings": 0.12364, "sat_rings": 0.044045,
        },
    },
    # ── Nav1.7 ──
    "Nav1.7": {
        "intercept": 4.6125,
        "coefficients": {
            "logP": 0.061361, "mw": 0.851697, "rot": -0.22958,
            "hbd": -0.219783, "hba": 0.018182, "tpsa": 1.251812,
            "arom": 0.659173, "fsp3": -0.005677,
            "q_mean": 0.028307, "q_std": -0.071199,
            "q_min": -1.168631, "q_max": -0.931806,
            "chi1": 0.139388, "chi2n": -0.130027,
            "bertz": -0.290421, "hk_alpha": 0.17157,
            "kappa2": -0.379497, "kappa3": 0.48357,
            "aliph_rings": 0.482179, "sat_rings": -0.154567,
        },
    },
    # ── PPARgamma ──
    "PPARgamma": {
        "intercept": 2.5265,
        "coefficients": {
            "logP": 0.739853, "mw": 1.209192, "rot": 0.062411,
            "hbd": 0.316271, "hba": 0.452277, "tpsa": 0.078692,
            "arom": 0.800884, "fsp3": 2.868309,
            "q_mean": 0.196531, "q_std": -0.107334,
            "q_min": 3.049609, "q_max": 0.081063,
            "chi1": -0.943224, "chi2n": -0.189017,
            "bertz": 0.012732, "hk_alpha": -1.475063,
            "kappa2": 0.061431, "kappa3": 0.092257,
            "aliph_rings": -0.313442, "sat_rings": 0.797495,
        },
    },
    # ── PTP1B ──
    "PTP1B": {
        "intercept": 1.8632,
        "coefficients": {
            "logP": 0.130291, "mw": 0.966977, "rot": 0.124862,
            "hbd": 0.341228, "hba": 0.039812, "tpsa": -1.039856,
            "arom": -0.362022, "fsp3": 0.006446,
            "q_mean": 0.057522, "q_std": 0.138611,
            "q_min": -0.794713, "q_max": 0.513127,
            "chi1": 0.011254, "chi2n": -0.181782,
            "bertz": 0.14689, "hk_alpha": 0.070567,
            "kappa2": -0.011783, "kappa3": -0.385997,
            "aliph_rings": 0.155211, "sat_rings": -0.002609,
        },
    },
    # ── hERG ──
    "hERG": {
        "intercept": 5.0214,
        "coefficients": {
            "logP": -0.016247, "mw": 0.290749, "rot": -0.041334,
            "hbd": 0.054722, "hba": 0.048016, "tpsa": -0.855833,
            "arom": 0.539685, "fsp3": -0.152072,
            "q_mean": 0.078147, "q_std": -0.518183,
            "q_min": 0.798269, "q_max": -1.090136,
            "chi1": -0.130705, "chi2n": -0.064974,
            "bertz": 0.001274, "hk_alpha": 0.044734,
            "kappa2": 0.023694, "kappa3": 0.300613,
            "aliph_rings": 0.465728, "sat_rings": -0.200495,
        },
    },
}


def _compute_general_pl_terms(uc, result):
    """Non-covalent scoring for non-metal protein targets.

    Fires when has_general_pl_data flag is set.
    Uses per-target Ridge model (20 descriptors) if available,
    falls back to global 8-descriptor model for unknown targets.
    """
    if not uc.has_general_pl_data:
        return
    if not uc.guest_smiles:
        return

    target = uc.host_name
    model = PER_TARGET_PL_MODELS.get(target)

    if model:
        # Per-target model: 20 descriptors
        c = model["coefficients"]
        log_ka_pred = model["intercept"]
        log_ka_pred += c["logP"] * uc.guest_logP
        log_ka_pred += c["mw"] * uc.guest_mw / 100.0
        log_ka_pred += c["rot"] * uc.guest_rotatable_bonds
        log_ka_pred += c["hbd"] * uc.guest_n_hbond_donors
        log_ka_pred += c["hba"] * uc.guest_n_hbond_acceptors
        log_ka_pred += c["tpsa"] * getattr(uc, 'guest_tpsa', 0.0) / 100.0
        log_ka_pred += c["arom"] * getattr(uc, 'guest_n_aromatic_rings', 0)
        log_ka_pred += c["fsp3"] * getattr(uc, 'guest_fsp3', 0.0)
        # Gasteiger charge statistics
        log_ka_pred += c["q_mean"] * getattr(uc, 'guest_q_mean', 0.0)
        log_ka_pred += c["q_std"] * getattr(uc, 'guest_q_std', 0.0)
        log_ka_pred += c["q_min"] * getattr(uc, 'guest_q_min', 0.0)
        log_ka_pred += c["q_max"] * getattr(uc, 'guest_q_max', 0.0)
        # Topological shape descriptors
        log_ka_pred += c["chi1"] * getattr(uc, 'guest_chi1', 0.0)
        log_ka_pred += c["chi2n"] * getattr(uc, 'guest_chi2n', 0.0)
        log_ka_pred += c["bertz"] * getattr(uc, 'guest_bertz', 0.0) / 100.0
        log_ka_pred += c["hk_alpha"] * getattr(uc, 'guest_hk_alpha', 0.0)
        log_ka_pred += c["kappa2"] * getattr(uc, 'guest_kappa2', 0.0)
        log_ka_pred += c["kappa3"] * getattr(uc, 'guest_kappa3', 0.0)
        log_ka_pred += c["aliph_rings"] * getattr(uc, 'guest_n_aliphatic_rings', 0)
        log_ka_pred += c["sat_rings"] * getattr(uc, 'guest_n_saturated_rings', 0)
    else:
        # Fallback: global 8-descriptor model for unknown targets
        p = GENERAL_PL_PARAMS
        offset = GENERAL_PL_TARGET_OFFSETS.get(target, 4.5)
        log_ka_pred = (
            p["a_logP"] * uc.guest_logP
            + p["b_mw"] * uc.guest_mw / 100.0
            + p["c_rot"] * uc.guest_rotatable_bonds
            + p["d_hbd"] * uc.guest_n_hbond_donors
            + p["e_hba"] * uc.guest_n_hbond_acceptors
            + p["f_tpsa"] * getattr(uc, 'guest_tpsa', 0.0) / 100.0
            + p["g_arom"] * getattr(uc, 'guest_n_aromatic_rings', 0)
            + p["h_fsp3"] * getattr(uc, 'guest_fsp3', 0.0)
            + offset
        )

    # Convert to dG and decompose into result fields
    dg_total = -log_ka_pred * LN10_RT

    # Decompose into interpretable buckets
    result.dg_hydrophobic = -(
        (model["coefficients"]["logP"] if model else GENERAL_PL_PARAMS["a_logP"])
        * uc.guest_logP) * LN10_RT
    result.dg_hbond = -(
        ((model["coefficients"]["hbd"] * uc.guest_n_hbond_donors
          + model["coefficients"]["hba"] * uc.guest_n_hbond_acceptors)
         if model else
         (GENERAL_PL_PARAMS["d_hbd"] * uc.guest_n_hbond_donors
          + GENERAL_PL_PARAMS["e_hba"] * uc.guest_n_hbond_acceptors))
    ) * LN10_RT
    result.dg_conf_entropy = dg_total - result.dg_hydrophobic - result.dg_hbond
    # Use dg_metal slot for the total (since no metal term fires)
    result.dg_metal = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-MODAL TERMS — delegates to cross_modal_predictor functions
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# GLYCAN-LECTIN SCORING — delegates to glycan/scorer.py
# ═══════════════════════════════════════════════════════════════════════════

def _compute_glycan_terms(uc, result):
    """Score glycan-lectin binding from contact maps + physics parameters.

    Fires when glycan data is present (flag, data objects, or lectin names).
    Self-zeros if glycan scorer or contact map not available.

    Maps GlycanPrediction terms to PredictionResult fields:
        DG0       -> dg_shape (scaffold geometry baseline)
        dG_HB     -> dg_hbond (H-bonds at interface)
        dG_desolv -> dg_group_desolv (OH burial desolvation)
        dG_CHP    -> dg_pi (CH-pi contacts)
        dG_linker -> dg_hbond_coop (linker cooperativity)
    """
    if not uc.has_glycan_data:
        has_data_objects = (uc.glycan_contact_map is not None
                           or uc.sugar_property_card is not None)
        has_lectin_names = (uc.host_name != "" and uc.guest_name != ""
                           and uc.binding_mode in ("glycan_lectin", "lectin_glycan"))
        if not has_data_objects and not has_lectin_names:
            return

    scaffold = uc.host_name
    ligand = uc.guest_name

    if not scaffold or not ligand:
        return

    try:
        from glycan.scorer import GlycanScorer
        scorer = GlycanScorer()
        pred = scorer.score(scaffold, ligand)
    except (ImportError, ValueError, KeyError):
        return

    # Map glycan terms to PredictionResult fields
    result.dg_shape = pred.dG0
    result.dg_hbond = pred.dG_HB
    result.dg_group_desolv = pred.dG_desolv
    result.dg_pi = pred.dG_CHP
    result.dg_hbond_coop = pred.dG_linker
    result.dg_conf_entropy = pred.dG_conf


def _compute_cm_terms(uc, result):
    """Cross-modal terms for metal@host (e.g. cation@CB[n]).

    Self-zeros if no metal or no cucurbituril host detected from data.
    """
    if not _is_cross_modal(uc):
        return
    if not uc.metal_formula:
        return

    # Resolve CB host key
    cb_key = _resolve_cm_host(uc)
    if cb_key is None:
        return

    cm_entry = _uc_to_cm_entry(uc, cb_key)

    try:
        result.dg_ion_dipole = _cm_dg_ion_dipole(cm_entry)
        result.dg_ion_desolv = _cm_dg_desolvation(cm_entry)
        result.dg_portal_size = _cm_dg_portal_size_match(cm_entry)
        result.dg_cm_dehydration = _cm_dg_cb_dehydration(cm_entry)
        result.dg_cm_shape = _cm_dg_shape_complementarity(cm_entry)
    except (ValueError, KeyError):
        pass


def _resolve_cm_host(uc):
    """Resolve a CB_PORTAL_INFO key from UC host_name."""
    for candidate in (uc.host_name, uc.host_type):
        if candidate in CB_PORTAL_INFO:
            return candidate
    # Try stripping brackets: "CB[7]" → "CB7"
    for candidate in (uc.host_name, uc.host_type):
        stripped = candidate.replace("[", "").replace("]", "")
        if stripped in CB_PORTAL_INFO:
            return stripped
    return None


# ═══════════════════════════════════════════════════════════════════════════
# VERBOSE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def _print_decomposition(uc, result):
    """Print full energy decomposition."""
    print(f"\n  {'='*60}")
    print(f"  {uc.name}  [{uc.binding_mode}]")
    print(f"  {'='*60}")

    terms = []
    if result.dg_metal != 0:
        terms.append(("Metal coord.", result.dg_metal))
    if result.dg_hydrophobic != 0:
        terms.append(("Hydrophobic", result.dg_hydrophobic))
    if result.dg_cavity_dehydration != 0:
        terms.append(("Cav.dehydr.", result.dg_cavity_dehydration))
    if result.dg_hbond != 0:
        terms.append(("H-bond net", result.dg_hbond))
    if result.dg_pi != 0:
        terms.append(("π-interact.", result.dg_pi))
    if result.dg_conf_entropy != 0:
        terms.append(("Conf.entropy", result.dg_conf_entropy))
    if result.dg_shape != 0:
        terms.append(("Shape compl.", result.dg_shape))
    if result.dg_size_mismatch != 0:
        terms.append(("Size mismatch", result.dg_size_mismatch))
    if result.dg_ion_dipole != 0:
        terms.append(("Ion-dipole", result.dg_ion_dipole))
    if result.dg_ion_desolv != 0:
        terms.append(("Ion desolv.", result.dg_ion_desolv))
    if result.dg_portal_size != 0:
        terms.append(("Portal size", result.dg_portal_size))
    if result.dg_cm_dehydration != 0:
        terms.append(("CM dehydr.", result.dg_cm_dehydration))
    if result.dg_cm_shape != 0:
        terms.append(("CM shape", result.dg_cm_shape))
    # Tier 2
    for f in TIER2_RESULT_FIELDS:
        v = getattr(result, f, 0.0)
        if v != 0.0:
            terms.append((f.replace("dg_","").replace("_"," ").title(), v))

    for label, val in terms:
        print(f"  {label:15s} {val:+8.2f} kJ/mol")
    print(f"  {'─'*30}")
    print(f"  {'ΔG total':15s} {result.dg_total_kj:+8.2f} kJ/mol")
    print(f"  {'log Ka pred':15s} {result.log_Ka_pred:+8.2f}")
    print(f"  {'log Ka exp':15s} {result.log_Ka_exp:+8.2f}")
    print(f"  {'error':15s} {result.error:+8.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION + STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def predict_batch(ucs, verbose=False):
    """Score a list of UniversalComplex entries.

    Returns list of PredictionResult.
    """
    results = []
    for uc in ucs:
        try:
            r = predict(uc, verbose=verbose)
        except Exception as e:
            r = PredictionResult(
                name=uc.name, binding_mode=uc.binding_mode,
                log_Ka_exp=uc.log_Ka_exp, log_Ka_pred=0.0,
                dg_total_kj=0.0, error=-uc.log_Ka_exp)
        results.append(r)
    return results


def compute_statistics(results):
    """Compute R², MAE, bias from PredictionResult list."""
    if not results:
        return {"r2": 0, "mae": 0, "bias": 0, "n": 0}

    exps = [r.log_Ka_exp for r in results]
    preds = [r.log_Ka_pred for r in results]
    errors = [r.error for r in results]
    n = len(results)

    mae = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n

    mean_exp = sum(exps) / n
    ss_tot = sum((e - mean_exp)**2 for e in exps)
    ss_res = sum((p - e)**2 for p, e in zip(preds, exps))
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {"r2": round(r2, 4), "mae": round(mae, 3), "bias": round(bias, 3), "n": n}