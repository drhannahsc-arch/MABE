"""
hg_conf_shape.py — Phase 9 (conformational entropy) + Phase 10 (shape complementarity).

Phase 9: Freezing rotatable bonds upon binding costs entropy.
  ΔG_conf = n_frozen × ε_rotor × f_partial

Phase 10: Packing coefficient determines shape complementarity.
  ΔG_shape = -k_shape × exp(-(PC - PC_opt)² / (2σ²))
  PC = V_guest / V_cavity (Rebek's 55% rule)

Both terms zero for metal coordination.
"""

import math
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
CONF_SHAPE_PARAMS = {
    # Phase 9: conformational entropy
    "eps_rotor":        2.5,      # kJ/mol per frozen rotor (Mammen 1998: ~3.4)
    "f_partial":        0.5,      # fraction of rotors actually frozen upon binding

    # Phase 10: shape complementarity
    "k_shape":         -8.0,      # kJ/mol max shape bonus (at optimal PC)
    "PC_optimal":       0.55,     # optimal packing coefficient (Rebek rule)
    "sigma_PC":         0.15,     # width of packing Gaussian
}


# ═══════════════════════════════════════════════════════════════════════════
# HOST CAVITY VOLUMES (Å³, from crystal structures)
# ═══════════════════════════════════════════════════════════════════════════
HOST_CAVITY_VOLUME = {
    "alpha-CD":    174.0,    # Szejtli 1998
    "beta-CD":     262.0,
    "gamma-CD":    427.0,
    "CB6":         164.0,    # Lagona 2005
    "CB7":         279.0,
    "CB8":         479.0,
    "calix4-SO3":  120.0,    # approximate, shallow cone
    "pillar5":     115.0,    # Ogoshi 2012
}


# ═══════════════════════════════════════════════════════════════════════════
# GUEST MOLECULAR PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════
_VOL_CACHE = {}
_ROT_CACHE = {}


def guest_n_rotors(smiles: str) -> int:
    """Count rotatable bonds in guest molecule."""
    if smiles in _ROT_CACHE:
        return _ROT_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    n = rdMolDescriptors.CalcNumRotatableBonds(mol)
    _ROT_CACHE[smiles] = n
    return n


def guest_volume(smiles: str) -> float:
    """Compute guest van der Waals volume in ų."""
    if smiles in _VOL_CACHE:
        return _VOL_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        AllChem.EmbedMolecule(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    try:
        vol = AllChem.ComputeMolVolume(mol)
    except Exception:
        # Fallback: estimate from MW (crude)
        mw = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
        vol = mw * 0.9  # rough Å³/Da for organic molecules
    _VOL_CACHE[smiles] = vol
    return vol


# ═══════════════════════════════════════════════════════════════════════════
# ENERGY TERMS
# ═══════════════════════════════════════════════════════════════════════════

def dg_conformational_entropy(smiles: str) -> float:
    """Conformational entropy penalty from freezing rotatable bonds.

    Returns positive (unfavorable) kJ/mol.
    """
    n_rot = guest_n_rotors(smiles)
    if n_rot == 0:
        return 0.0
    n_frozen = n_rot * CONF_SHAPE_PARAMS["f_partial"]
    return CONF_SHAPE_PARAMS["eps_rotor"] * n_frozen


def dg_shape_complementarity(smiles: str, host_key: str) -> float:
    """Shape complementarity energy from packing coefficient.

    Gaussian centered at PC_optimal. Returns negative (favorable) at
    good packing, zero at poor packing.
    """
    v_cavity = HOST_CAVITY_VOLUME.get(host_key, 0.0)
    if v_cavity <= 0:
        return 0.0
    v_guest = guest_volume(smiles)
    if v_guest <= 0:
        return 0.0

    PC = v_guest / v_cavity

    # Gaussian shape function
    PC_opt = CONF_SHAPE_PARAMS["PC_optimal"]
    sigma = CONF_SHAPE_PARAMS["sigma_PC"]
    f_pack = math.exp(-(PC - PC_opt)**2 / (2 * sigma**2))

    # Active penalty for PC > 0.9 (can't enter)
    if PC > 0.90:
        f_pack = -abs(f_pack) * (PC - 0.90) / 0.10

    return CONF_SHAPE_PARAMS["k_shape"] * f_pack


def compute_dg_conf_shape(entry: dict, verbose: bool = False) -> tuple:
    """Compute both conformational entropy and shape energy.

    Returns (dg_conf, dg_shape) in kJ/mol.
    """
    smiles = entry["guest_smiles"]
    host_key = entry["host"]

    dg_conf = dg_conformational_entropy(smiles)
    dg_shape = dg_shape_complementarity(smiles, host_key)

    if verbose:
        n_rot = guest_n_rotors(smiles)
        v_guest = guest_volume(smiles)
        v_cavity = HOST_CAVITY_VOLUME.get(host_key, 0)
        pc = v_guest / v_cavity if v_cavity > 0 else 0
        print(f"  Rotatable bonds: {n_rot}, frozen: "
              f"{n_rot * CONF_SHAPE_PARAMS['f_partial']:.1f}")
        print(f"  V_guest={v_guest:.0f}ų, V_cavity={v_cavity:.0f}ų, "
              f"PC={pc:.2f}")
        print(f"  ΔG_conf:  {dg_conf:+.1f} kJ/mol")
        print(f"  ΔG_shape: {dg_shape:+.1f} kJ/mol")

    return dg_conf, dg_shape


if __name__ == "__main__":
    from hg_dataset import HG_DATA

    print("Phase 9+10: Conformational entropy + Shape complementarity")
    print("=" * 60)

    # Show range of properties across dataset
    print(f"\n{'Name':30s} {'Rot':>4s} {'V_g':>6s} {'V_cav':>6s} {'PC':>5s} "
          f"{'dG_c':>6s} {'dG_s':>6s}")
    for e in HG_DATA[:30]:
        n_rot = guest_n_rotors(e["guest_smiles"])
        v_g = guest_volume(e["guest_smiles"])
        v_c = HOST_CAVITY_VOLUME.get(e["host"], 0)
        pc = v_g / v_c if v_c > 0 else 0
        dg_c, dg_s = compute_dg_conf_shape(e)
        print(f"  {e['name']:28s} {n_rot:4d} {v_g:6.0f} {v_c:6.0f} {pc:5.2f} "
              f"{dg_c:+6.1f} {dg_s:+6.1f}")