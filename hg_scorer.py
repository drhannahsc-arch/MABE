"""
hg_scorer.py — Host-guest binding scorer for MABE Phase 6.

Predicts log Ka for host-guest inclusion complexes.
Primary energy terms:
  1. Hydrophobic transfer: -γ × SASA_buried
  2. Electrostatic ion-dipole (portal charge × guest charge)
  3. H-bond at portal (n_hbonds × ε_hbond)
  4. Size mismatch penalty (guest too large or too small for cavity)

Terms 2-4 are simple corrections; the hydrophobic term is the calibration target.
Metal calibration (scorer_frozen.py) is unaffected — these terms are zero for
metal-ligand entries which have no host cavity.
"""

import math
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, Descriptors

from hg_dataset import HOST_DB
from hg_hbond import compute_dg_hbond, HBOND_PARAMS
from hg_pi import compute_dg_pi, PI_PARAMS
from hg_conf_shape import compute_dg_conf_shape, CONF_SHAPE_PARAMS

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS (fitted by hg_calibrate.py)
# ═══════════════════════════════════════════════════════════════════════════
HG_PARAMS = {
    # Phase 6 core: hydrophobic transfer
    "gamma_flat":       0.0251,   # kJ/(mol·Å²), Eisenberg-McLachlan consensus
    "k_curvature":      1.149,    # concave cavity amplification

    # Cavity dehydration: high-energy water release (Biedermann & Nau 2014)
    "dg_dehydr_per_A2": -0.0703,  # kJ/(mol·Å²) base dehydration
    "dehydr_CB":         3.667,   # CB[n] multiplier (frustrated water)
    "dehydr_CD":         0.644,   # cyclodextrin multiplier
    "dehydr_other":      1.736,   # calixarene/pillararene multiplier

    # Portal corrections RETIRED — replaced by Phase 7 hg_hbond.py
    # Kept at 0.0 for backward compat; all portal physics now in HBOND_PARAMS
    "epsilon_hbond":    0.0,
    "epsilon_cation":   0.0,
    "k_electrostatic":  0.0,

    # Size match
    "k_size_penalty":   0.031,    # kJ/(mol·Å²), oversize penalty
    "k_undersize":      0.0,      # undersize (not significant at this phase)
}

# RT at 298.15 K in kJ/mol
RT_298 = 8.314e-3 * 298.15  # 2.479 kJ/mol
LN10_RT = 2.303 * RT_298     # 5.71 kJ/mol


# ═══════════════════════════════════════════════════════════════════════════
# SASA COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════
_SASA_CACHE = {}


def compute_guest_sasa(smiles: str) -> dict:
    """Compute total and nonpolar SASA for a guest molecule.

    Returns dict with:
        total_sasa: total solvent-accessible surface area (Å²)
        tpsa: topological polar surface area (Å²)
        nonpolar_sasa: total - tpsa (Å²)
    """
    if smiles in _SASA_CACHE:
        return _SASA_CACHE[smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        # Fallback: try without ETKDG
        AllChem.EmbedMolecule(mol_h)

    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass  # Use unoptimized geometry

    radii = rdFreeSASA.classifyAtoms(mol_h)
    total = rdFreeSASA.CalcSASA(mol_h, radii)
    tpsa = Descriptors.TPSA(mol)
    nonpolar = max(0.0, total - tpsa)

    result = {
        "total_sasa": total,
        "tpsa": tpsa,
        "nonpolar_sasa": nonpolar,
    }
    _SASA_CACHE[smiles] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════
# BURIED SASA MODEL
# ═══════════════════════════════════════════════════════════════════════════

def estimate_buried_sasa(guest_nonpolar_sasa: float,
                         cavity_sasa: float) -> float:
    """Estimate the nonpolar SASA buried upon inclusion.

    Simple geometric model: buried = min(guest_nonpolar, cavity_interior).
    If guest is larger than cavity, only cavity-sized portion buries.
    If guest is smaller, all guest nonpolar SASA can bury.

    More sophisticated models would use 3D overlap, but this captures
    the dominant scaling.
    """
    return min(guest_nonpolar_sasa, cavity_sasa)


# ═══════════════════════════════════════════════════════════════════════════
# ENERGY TERMS
# ═══════════════════════════════════════════════════════════════════════════

def dg_hydrophobic(buried_sasa: float, curvature_class: str) -> float:
    """Hydrophobic transfer energy from burying nonpolar surface.

    Returns negative (favorable) kJ/mol.
    """
    gamma = HG_PARAMS["gamma_flat"]
    if curvature_class == "concave":
        gamma *= HG_PARAMS["k_curvature"]
    elif curvature_class == "shallow":
        gamma *= 1.0  # No amplification for shallow cavities

    return -gamma * buried_sasa


def dg_cavity_dehydration(buried_sasa: float, host_key: str) -> float:
    """Extra energy from releasing frustrated/high-energy water from cavity.

    CB[n] cavities contain water molecules that cannot form a full H-bond
    network (Biedermann & Nau). Releasing them provides additional driving
    force beyond classical hydrophobic transfer.

    Returns negative (favorable) kJ/mol.
    """
    base = HG_PARAMS["dg_dehydr_per_A2"]
    if host_key.startswith("CB"):
        mult = HG_PARAMS["dehydr_CB"]
    elif host_key.endswith("-CD"):
        mult = HG_PARAMS["dehydr_CD"]
    else:
        mult = HG_PARAMS["dehydr_other"]

    return base * mult * buried_sasa


def dg_portal_hbond(n_hbonds: int) -> float:
    """H-bond energy at host portal."""
    return HG_PARAMS["epsilon_hbond"] * n_hbonds


def dg_electrostatic_portal(guest_charge: int, portal_type: str,
                            guest_has_cation: bool) -> float:
    """Ion-dipole interaction between charged guest and portal."""
    dg = 0.0
    if guest_charge != 0:
        dg += HG_PARAMS["k_electrostatic"] * abs(guest_charge)

    # Specific cation-carbonyl interaction (CB portals)
    if guest_has_cation and portal_type == "carbonyl":
        dg += HG_PARAMS["epsilon_cation"]

    # Cation-sulfonate for calixarene
    if guest_has_cation and portal_type == "sulfonate":
        dg += HG_PARAMS["epsilon_cation"] * 1.5  # Stronger ion-ion

    return dg


def dg_size_mismatch(guest_nonpolar_sasa: float,
                     cavity_sasa: float) -> float:
    """Penalty for size mismatch between guest and cavity.

    Too large: steric clash / incomplete insertion.
    Too small: rattling / poor van der Waals contact.
    """
    dg = 0.0
    excess = guest_nonpolar_sasa - cavity_sasa
    if excess > 0:
        dg += HG_PARAMS["k_size_penalty"] * excess
    else:
        deficit = cavity_sasa - guest_nonpolar_sasa
        if deficit > cavity_sasa * 0.5:  # >50% empty
            dg += HG_PARAMS["k_undersize"] * deficit

    return dg


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

def predict_hg_log_ka(entry: dict, verbose: bool = False) -> float:
    """Predict log Ka for a host-guest inclusion complex.

    Args:
        entry: dict from HG_DATA with keys: host, guest_smiles,
               guest_charge, n_hbonds_portal, guest_has_cation
        verbose: print term breakdown

    Returns:
        Predicted log Ka
    """
    host_key = entry["host"]
    host = HOST_DB[host_key]
    sasa = compute_guest_sasa(entry["guest_smiles"])

    guest_np = sasa["nonpolar_sasa"]
    cavity_sasa = host["cavity_sasa"]
    buried = estimate_buried_sasa(guest_np, cavity_sasa)

    # Energy terms
    dg_hphob = dg_hydrophobic(buried, host["curvature_class"])
    dg_dehyd = dg_cavity_dehydration(buried, host_key)
    dg_hb_portal = dg_portal_hbond(entry.get("n_hbonds_portal", 0))
    dg_elec = dg_electrostatic_portal(
        entry["guest_charge"], host["portal_type"],
        entry.get("guest_has_cation", False))
    dg_size = dg_size_mismatch(guest_np, cavity_sasa)
    dg_hbond = compute_dg_hbond(entry, HOST_DB, verbose=False)
    dg_pi = compute_dg_pi(entry, HOST_DB, verbose=False)
    dg_conf, dg_shape = compute_dg_conf_shape(entry, verbose=False)

    dg_total = (dg_hphob + dg_dehyd + dg_hb_portal + dg_elec
                + dg_size + dg_hbond + dg_pi + dg_conf + dg_shape)
    log_ka = -dg_total / LN10_RT

    if verbose:
        print(f"  Host: {host_key} (cavity Ø{host['cavity_diameter']}Å, "
              f"SASA_cav={cavity_sasa:.0f}Å²)")
        print(f"  Guest: {entry['guest_smiles']}")
        print(f"  Guest SASA: total={sasa['total_sasa']:.0f} "
              f"nonpolar={guest_np:.0f}Å²")
        print(f"  Buried SASA: {buried:.0f}Å²")
        print(f"  ── Energy terms (kJ/mol) ──")
        print(f"  Hydrophobic:  {dg_hphob:+8.1f}")
        print(f"  Cav.dehydr.:  {dg_dehyd:+8.1f}")
        print(f"  H-bond net:   {dg_hbond:+8.1f}")
        print(f"  π-interact.:  {dg_pi:+8.1f}")
        print(f"  Portal H-bond:{dg_hb_portal:+8.1f}")
        print(f"  Electrostatic:{dg_elec:+8.1f}")
        print(f"  Size mismatch:{dg_size:+8.1f}")
        print(f"  Conf.entropy: {dg_conf:+8.1f}")
        print(f"  Shape compl.: {dg_shape:+8.1f}")
        print(f"  ─────────────────────────")
        print(f"  ΔG total:     {dg_total:+8.1f} kJ/mol")
        print(f"  log Ka:       {log_ka:+8.2f}")

    return log_ka


# ═══════════════════════════════════════════════════════════════════════════
# BATCH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dataset(data: list) -> dict:
    """Evaluate all entries and return statistics."""
    import numpy as np

    preds = []
    exps = []
    for e in data:
        try:
            pred = predict_hg_log_ka(e)
        except Exception:
            pred = 0.0
        preds.append(pred)
        exps.append(e["log_Ka"])

    preds = np.array(preds)
    exps = np.array(exps)
    residuals = preds - exps
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((exps - np.mean(exps))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    worst_idx = np.argmax(np.abs(residuals))

    return {
        "n": len(data),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "max_err": residuals[worst_idx],
        "worst": data[worst_idx]["name"],
        "preds": preds,
        "exps": exps,
    }


if __name__ == "__main__":
    from hg_dataset import HG_DATA

    print("═" * 60)
    print("  MABE Phase 6: Host-Guest Scorer (default params)")
    print("═" * 60)

    stats = evaluate_dataset(HG_DATA)
    print(f"\n  N = {stats['n']} host-guest pairs")
    print(f"  R²   = {stats['r2']:.4f}")
    print(f"  MAE  = {stats['mae']:.2f} log Ka")
    print(f"  RMSE = {stats['rmse']:.2f} log Ka")
    print(f"  Max  = {abs(stats['max_err']):.1f} ({stats['worst']})")

    # Show a few verbose predictions
    print("\n── Sample predictions ──")
    for name in ["bCD+adamantane-COOH", "CB7+adamantane-NH3+", "aCD+1-hexanol"]:
        entry = next(e for e in HG_DATA if e["name"] == name)
        print(f"\n{name} (exp={entry['log_Ka']:.2f}):")
        predict_hg_log_ka(entry, verbose=True)