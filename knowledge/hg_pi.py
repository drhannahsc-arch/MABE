"""
hg_pi.py — π-Interaction energy for MABE Phase 8.

Classifies and scores π-contacts at host-guest interfaces:
  1. CH-π: aliphatic C-H → aromatic face (~1-3 kJ/mol)
  2. π-π stacking: aromatic guest face → aromatic host wall (~2-8 kJ/mol)
  3. Cation-π: cationic guest → aromatic host wall (~2-10 kJ/mol in water)

Activated only for hosts with aromatic walls (calixarene, pillararene,
cyclophane). Zero for CDs (no aromatic walls) and CBs (glycoluril walls,
minimal π-character).

Zero for metal coordination (handled by scorer_frozen.py).
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETERS (fitted by calibration)
# ═══════════════════════════════════════════════════════════════════════════
PI_PARAMS = {
    "eps_ch_pi":       -1.5,     # kJ/mol per CH-π contact
    "eps_pi_stack":    -4.0,     # kJ/mol per π-π stacking contact
    "eps_cation_pi":   -5.0,     # kJ/mol per cation-π contact (per ring)
}


# ═══════════════════════════════════════════════════════════════════════════
# HOST π-CHARACTER
# ═══════════════════════════════════════════════════════════════════════════
# Which host families have aromatic walls for π-interactions?
HOST_PI_CHARACTER = {
    "alpha-CD":    "none",       # glucose O-H, no aromatic
    "beta-CD":     "none",
    "gamma-CD":    "none",
    "CB6":         "minimal",    # glycoluril has some π, but weak
    "CB7":         "minimal",
    "CB8":         "minimal",
    "calix4-SO3":  "aromatic",   # 4 phenyl rings = strong π-host
    "pillar5":     "aromatic",   # 5 hydroquinone units = strong π-host
}


def _count_guest_aromatic_rings(smiles: str) -> int:
    """Count aromatic rings in guest."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def _count_guest_aliphatic_ch(smiles: str) -> int:
    """Rough count of aliphatic C-H groups that could contact π-face.

    Uses heavy atom count minus aromatic atoms as proxy.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    n_heavy = mol.GetNumHeavyAtoms()
    n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_hetero = sum(1 for a in mol.GetAtoms()
                   if a.GetAtomicNum() not in (1, 6))
    # Aliphatic carbons ≈ heavy - aromatic - heteroatoms
    n_aliph_c = max(0, n_heavy - n_arom - n_hetero)
    # Each aliphatic C has ~2 H accessible to π-face on average,
    # but only ~1-2 can contact at once (geometric constraint)
    return min(n_aliph_c, 6)  # Cap at 6 contacts


def classify_pi_contacts(entry: dict, host_db: dict) -> dict:
    """Classify π-interactions between host and guest.

    Returns dict with counts:
        n_ch_pi: CH-π contacts (aliphatic guest → aromatic host)
        n_pi_stack: π-π contacts (aromatic guest → aromatic host)
        n_cation_pi: cation-π contacts (cationic guest → aromatic host)
    """
    host_key = entry["host"]
    pi_char = HOST_PI_CHARACTER.get(host_key, "none")
    smiles = entry["guest_smiles"]
    is_cation = entry.get("guest_has_cation", False)
    charge = entry.get("guest_charge", 0)

    n_ch_pi = 0
    n_pi_stack = 0
    n_cation_pi = 0

    if pi_char == "none":
        # CDs: no aromatic walls, no π-interactions
        return {"n_ch_pi": 0, "n_pi_stack": 0, "n_cation_pi": 0}

    if pi_char == "minimal":
        # CBs: very weak, only count 1 contact max
        guest_arom = _count_guest_aromatic_rings(smiles)
        if guest_arom > 0:
            n_pi_stack = 1  # Weak π-π with glycoluril backbone
        return {"n_ch_pi": 0, "n_pi_stack": n_pi_stack, "n_cation_pi": 0}

    # Aromatic host (calixarene, pillararene)
    host_data = host_db[host_key]
    n_host_arene = host_data.get("n_arene", 4)

    # Cation-π: cationic center interacts with aromatic bowl
    if charge > 0:
        # Number of host arene units accessible to the cation
        # Typically 2-4 depending on geometry
        n_cation_pi = min(charge, 2) * min(n_host_arene, 4) // 2

    # π-π stacking: aromatic guest in aromatic host
    guest_arom = _count_guest_aromatic_rings(smiles)
    if guest_arom > 0:
        # Each guest ring can stack with 1-2 host arene walls
        n_pi_stack = min(guest_arom * 2, n_host_arene)

    # CH-π: aliphatic guest in aromatic host
    if guest_arom == 0:
        n_aliph = _count_guest_aliphatic_ch(smiles)
        # Geometric constraint: only a few CH groups face the π-walls
        n_ch_pi = min(n_aliph, n_host_arene)

    return {
        "n_ch_pi": n_ch_pi,
        "n_pi_stack": n_pi_stack,
        "n_cation_pi": n_cation_pi,
    }


def dg_pi_total(pi_counts: dict) -> float:
    """Compute total π-interaction energy in kJ/mol."""
    return (
        pi_counts["n_ch_pi"] * PI_PARAMS["eps_ch_pi"]
        + pi_counts["n_pi_stack"] * PI_PARAMS["eps_pi_stack"]
        + pi_counts["n_cation_pi"] * PI_PARAMS["eps_cation_pi"]
    )


def compute_dg_pi(entry: dict, host_db: dict, verbose: bool = False) -> float:
    """Full pipeline: classify then score π-contacts."""
    counts = classify_pi_contacts(entry, host_db)
    dg = dg_pi_total(counts)

    if verbose:
        print(f"  π-contacts: {counts['n_ch_pi']} CH-π, "
              f"{counts['n_pi_stack']} π-stack, "
              f"{counts['n_cation_pi']} cation-π")
        print(f"  ΔG_π: {dg:+.1f} kJ/mol")

    return dg


if __name__ == "__main__":
    from hg_dataset import HG_DATA, HOST_DB

    print("Phase 8: π-Interaction classification test")
    print("=" * 50)

    for e in HG_DATA:
        counts = classify_pi_contacts(e, HOST_DB)
        total = sum(counts.values())
        if total > 0:
            dg = dg_pi_total(counts)
            print(f"  {e['name']:30s} CH-π={counts['n_ch_pi']} "
                  f"π-π={counts['n_pi_stack']} cat-π={counts['n_cation_pi']} "
                  f"ΔG={dg:+.1f}")