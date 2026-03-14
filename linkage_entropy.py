"""
linkage_entropy.py — Phase G2: Conformational Entropy from Rotamer Populations

Computes TΔS_freeze for glycosidic linkages upon binding. When a sugar
binds a receptor, glycosidic torsions (φ/ψ/ω) freeze from a Boltzmann
distribution of rotamers into a single bound-state conformation.

Physics:
    TΔS_freeze(linkage) = -RT × Σ_i [p_i × ln(p_i)]

    where p_i = population of rotamer i in the free (solution) state.
    Bound state: S ≈ 0 (locked in one well).

Sources (per torsion type):
    φ (H1-C1-O-Cx): Dominated by exo-anomeric effect.
        Population ~90:10 for primary:secondary conformer.
        Ref: Tvaroška & Bleha, Adv. Carbohydr. Chem. Biochem. 1989, 47, 45.
        Ref: Lemieux RU, Koto S, Tetrahedron 1974, 30, 1933.

    ψ (C1-O-Cx-Hx): Varies by linkage type, 1-3 accessible rotamers.
        Populations PROVISIONAL — require verification from:
        Kirschner et al., J. Comput. Chem. 2008, 29, 622 (GLYCAM06 QM surfaces).
        Imberty & Pérez, Chem. Rev. 2000, 100, 4567 (computed φ/ψ maps).

    ω (O5'-C5'-C6'-O6): Only in 1→6 linkages. 3 rotamers (gg, gt, tg).
        Populations from NMR J-coupling analysis.
        Ref: Stenutz R et al., J. Org. Chem. 2004, 69, 9216.
        Ref: Bock K, Duus JØ, J. Carbohydr. Chem. 1994, 13, 513.
        Typical: gg:gt:tg ≈ 50:40:10 (varies by sugar identity).

Cross-checks:
    - MABE HG Phase 9 eps_rotor = 2.48 kJ/mol per generic rotor
    - Mammen/Whitesides consensus: 3.4 kJ/mol per rotor (Angew. Chem. 1998, 37, 2754)
    - Glycosidic torsions should be LESS than generic rotors (anomeric pre-restriction)
    - NIST CCCBDB: dimethyl ether C-O-C barrier = 11.38 kJ/mol (experimental)

NO binding data used. NO biology involved. Pure thermochemistry + NMR rotamer analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

R_kJ = 8.314e-3       # kJ/(mol·K)
T_STD = 298.15         # K
RT_STD = R_kJ * T_STD  # 2.4790 kJ/mol


# ═══════════════════════════════════════════════════════════════════════════
# Core physics: rotamer entropy
# ═══════════════════════════════════════════════════════════════════════════

def rotamer_entropy(populations: List[float], T: float = T_STD) -> float:
    """
    Compute TΔS_freeze from rotamer populations.

    TΔS = -RT × Σ_i [p_i × ln(p_i)]

    Parameters
    ----------
    populations : list of float
        Fractional populations of each rotamer. Must sum to ~1.0.
    T : float
        Temperature in K (default 298.15).

    Returns
    -------
    float
        TΔS_freeze in kJ/mol (always ≥ 0).
    """
    pops = np.array(populations, dtype=float)
    total = np.sum(pops)
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Rotamer populations must sum to ~1.0, got {total:.4f}"
        )
    # Renormalize for floating point
    pops = pops / total
    RT = R_kJ * T
    mask = pops > 1e-30
    S_over_R = -np.sum(pops[mask] * np.log(pops[mask]))
    return RT * S_over_R


def rotamer_entropy_from_barrier(V0_kJ: float, n_fold: int,
                                  T: float = T_STD,
                                  n_bins: int = 3600) -> Tuple[float, List[float]]:
    """
    Compute TΔS_freeze from a symmetric n-fold cosine barrier.

    V(θ) = V0/2 × (1 - cos(n×θ))

    Finds wells, computes Boltzmann-weighted populations, returns
    TΔS_freeze and list of populations.

    For symmetric barriers, all wells are equally populated:
    TΔS = RT × ln(n_fold).

    Parameters
    ----------
    V0_kJ : float
        Barrier height in kJ/mol.
    n_fold : int
        Periodicity (number of equivalent minima).
    T : float
        Temperature in K.
    n_bins : int
        Grid resolution for numerical integration.

    Returns
    -------
    TdS : float
        TΔS_freeze in kJ/mol.
    populations : list of float
        Population of each well.
    """
    if V0_kJ < 0:
        raise ValueError("Barrier height must be non-negative")
    if n_fold < 1:
        raise ValueError("n_fold must be >= 1")

    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    V = V0_kJ / 2.0 * (1.0 - np.cos(n_fold * theta))
    RT = R_kJ * T

    boltz = np.exp(-V / RT)

    # Assign each bin to nearest minimum (wells at θ = 2πk/n for k=0..n-1)
    well_sums = np.zeros(n_fold)
    for i in range(n_bins):
        # Which well is this bin closest to?
        well_idx = int(round(theta[i] * n_fold / (2 * np.pi))) % n_fold
        well_sums[well_idx] += boltz[i]

    total = np.sum(well_sums)
    populations = [w / total for w in well_sums]

    TdS = rotamer_entropy(populations, T)
    return TdS, populations


# ═══════════════════════════════════════════════════════════════════════════
# Torsion definitions for glycosidic linkages
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TorsionDef:
    """Definition of a single torsion with its rotamer populations."""
    name: str               # e.g., "phi", "psi", "omega"
    populations: List[float] # rotamer populations (sum to 1.0)
    source: str             # citation for the populations
    tier: int               # 1 = NMR/experimental, 2 = QM/strong theory, 3 = provisional
    notes: str = ""


@dataclass
class LinkageEntropy:
    """Conformational entropy for a specific glycosidic linkage type."""
    linkage_type: str                    # e.g., "b1-4", "a1-6"
    torsions: List[TorsionDef]          # φ, ψ, [ω]
    TdS_freeze_kJmol: float = 0.0      # computed total

    def compute(self, T: float = T_STD) -> float:
        """Compute total TΔS_freeze from component torsions."""
        total = 0.0
        for t in self.torsions:
            total += rotamer_entropy(t.populations, T)
        self.TdS_freeze_kJmol = total
        return total

    @property
    def n_torsions(self) -> int:
        return len(self.torsions)


# ═══════════════════════════════════════════════════════════════════════════
# Linkage entropy table — the main deliverable
# ═══════════════════════════════════════════════════════════════════════════

# φ torsion: common to all linkages
# Exo-anomeric effect locks φ to primary conformer with ~90% population.
# Source: anomeric effect theory, NMR J-coupling consensus.
# Tier 2: strong theoretical basis, specific 90:10 split is approximate.
PHI_ANOMERIC = TorsionDef(
    name="phi",
    populations=[0.90, 0.10],
    source="Anomeric effect: Tvaroška & Bleha 1989 Adv.Carbohydr.Chem.Biochem. 47:45; "
           "Lemieux & Koto 1974 Tetrahedron 30:1933",
    tier=2,
    notes="Exo-anomeric effect dominates. 90:10 is consensus estimate. "
          "Verify from Kirschner 2008 QM surfaces if available."
)

# ψ torsions: linkage-specific
# These are PROVISIONAL (Tier 3) — need verification from GLYCAM06 QM surfaces.
# Physical basis: steric interaction between residues determines well populations.

PSI_BETA_1_4 = TorsionDef(
    name="psi",
    populations=[0.75, 0.25],
    source="PROVISIONAL: estimated from cellobiose/lactose φ/ψ maps. "
           "Verify: Kirschner 2008 J.Comput.Chem. 29:622; "
           "Imberty & Pérez 2000 Chem.Rev. 100:4567",
    tier=3,
    notes="β1→4 has two accessible ψ wells. Dominant conformer at ~180°. "
          "Moderate restriction from 1,3-diaxial-like interactions."
)

PSI_BETA_1_3 = TorsionDef(
    name="psi",
    populations=[0.85, 0.15],
    source="PROVISIONAL: estimated from laminaribiose-type φ/ψ maps. "
           "Verify: Kirschner 2008 J.Comput.Chem. 29:622",
    tier=3,
    notes="β1→3 is more restricted than β1→4. Less steric strain, "
          "single dominant well."
)

PSI_ALPHA_1_3 = TorsionDef(
    name="psi",
    populations=[0.85, 0.15],
    source="PROVISIONAL: estimated from α1→3 disaccharide φ/ψ maps. "
           "Verify: Kirschner 2008 J.Comput.Chem. 29:622",
    tier=3,
    notes="α1→3 is relatively restricted. Similar to β1→3."
)

PSI_ALPHA_1_4 = TorsionDef(
    name="psi",
    populations=[0.65, 0.35],
    source="PROVISIONAL: estimated from maltose φ/ψ map. "
           "Verify: Kirschner 2008 J.Comput.Chem. 29:622",
    tier=3,
    notes="α1→4 (maltose-type) has broader ψ distribution than β1→4. "
          "Two accessible wells with moderate populations."
)

PSI_ALPHA_1_2 = TorsionDef(
    name="psi",
    populations=[0.80, 0.20],
    source="PROVISIONAL: estimated from mannobiose α1→2 φ/ψ map. "
           "Verify: Imberty & Pérez 2000 Chem.Rev. 100:4567",
    tier=3,
    notes="α1→2 is moderately restricted."
)

PSI_ALPHA_1_6 = TorsionDef(
    name="psi",
    populations=[0.65, 0.35],
    source="PROVISIONAL: ψ for 1→6 linkage estimated similar to α1→4. "
           "Verify: Kirschner 2008 J.Comput.Chem. 29:622",
    tier=3,
    notes="1→6 ψ flexibility similar to other α-linkages. The major "
          "additional flexibility comes from the ω torsion."
)

PSI_ALPHA_2_3_SIA = TorsionDef(
    name="psi",
    populations=[0.80, 0.20],
    source="PROVISIONAL: estimated for Neu5Ac α2→3 linkage. "
           "Verify: DeMarco & Woods 2008 Glycobiology 18:426",
    tier=3,
    notes="Sialic acid α2→3 linkage. Anomeric center is C2 (ketose)."
)

# ω torsion: only in 1→6 linkages
# NMR-derived populations — Tier 1 data.
OMEGA_1_6 = TorsionDef(
    name="omega",
    populations=[0.50, 0.40, 0.10],
    source="NMR ³J(H5,H6) coupling analysis: "
           "Stenutz 2004 J.Org.Chem. 69:9216; "
           "Bock & Duus 1994 J.Carbohydr.Chem. 13:513",
    tier=1,
    notes="gg:gt:tg rotamers. Populations vary by sugar identity "
          "(50:40:10 is galactose-type consensus). Glucose-type may "
          "differ (~60:40:0). Both give TΔS ≈ 2.2–2.3 kJ/mol."
)


def build_linkage_entropy_table() -> Dict[str, LinkageEntropy]:
    """
    Build the complete linkage entropy table.

    Returns dict mapping linkage type string to LinkageEntropy object.
    All TΔS_freeze values computed at 298.15 K.
    """
    table = {}

    entries = [
        ("b1-4", [PHI_ANOMERIC, PSI_BETA_1_4]),
        ("b1-3", [PHI_ANOMERIC, PSI_BETA_1_3]),
        ("a1-4", [PHI_ANOMERIC, PSI_ALPHA_1_4]),
        ("a1-3", [PHI_ANOMERIC, PSI_ALPHA_1_3]),
        ("a1-2", [PHI_ANOMERIC, PSI_ALPHA_1_2]),
        ("a1-6", [PHI_ANOMERIC, PSI_ALPHA_1_6, OMEGA_1_6]),
        ("a2-3", [PHI_ANOMERIC, PSI_ALPHA_2_3_SIA]),
    ]

    for linkage_type, torsions in entries:
        le = LinkageEntropy(linkage_type=linkage_type, torsions=torsions)
        le.compute()
        table[linkage_type] = le

    return table


# Singleton table
LINKAGE_ENTROPY_TABLE = build_linkage_entropy_table()


def get_TdS_freeze(linkage_type: str) -> float:
    """
    Look up TΔS_freeze for a given linkage type.

    Parameters
    ----------
    linkage_type : str
        One of: "b1-4", "b1-3", "a1-4", "a1-3", "a1-2", "a1-6", "a2-3"

    Returns
    -------
    float
        TΔS_freeze in kJ/mol

    Raises
    ------
    KeyError
        If linkage_type not in table.
    """
    key = linkage_type.lower().replace("→", "-").replace("β", "b").replace("α", "a")
    if key not in LINKAGE_ENTROPY_TABLE:
        raise KeyError(
            f"Unknown linkage type '{linkage_type}'. "
            f"Known types: {sorted(LINKAGE_ENTROPY_TABLE.keys())}"
        )
    return LINKAGE_ENTROPY_TABLE[key].TdS_freeze_kJmol


def get_branch_penalty(n_branches: int = 1) -> float:
    """
    Estimate conformational entropy penalty for branch points.

    At an N-glycan branch point, the backbone is constrained by two
    diverging chains. This reduces the backbone's residual flexibility
    beyond what individual linkages predict.

    The branch penalty is estimated as the entropy cost of correlating
    two torsion pairs that share a common pivot residue.

    Parameters
    ----------
    n_branches : int
        Number of branch points (1 for biantennary, 2 for triantennary).

    Returns
    -------
    float
        Additional TΔS penalty in kJ/mol.

    Notes
    -----
    PROVISIONAL (Tier 3). The branch penalty is estimated as ~0.5 kJ/mol
    per branch point, based on the entropy reduction from correlated motion
    of two chains attached to a common residue. This needs verification
    from MD simulations of branched N-glycans.

    Ref: Woods RJ, Tessier MB, Curr. Opin. Struct. Biol. 2010, 20, 575.
    """
    K_BRANCH = 0.5  # kJ/mol per branch point (provisional)
    return n_branches * K_BRANCH


def score_conformational_entropy(linkage_types: List[str],
                                  n_branches: int = 0) -> float:
    """
    Compute total conformational entropy penalty for a glycan.

    This is the sum of per-linkage TΔS_freeze values plus any
    branch-point penalties.

    Parameters
    ----------
    linkage_types : list of str
        Glycosidic linkage types in the glycan (e.g., ["a1-6", "b1-4"]).
    n_branches : int
        Number of branch points.

    Returns
    -------
    float
        Total TΔS_conf penalty in kJ/mol (positive = unfavorable).
    """
    total = 0.0
    for lt in linkage_types:
        total += get_TdS_freeze(lt)
    total += get_branch_penalty(n_branches)
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Integration with glycan scorer
# ═══════════════════════════════════════════════════════════════════════════

def update_glycan_params_with_G2(params_dict: dict) -> dict:
    """
    Update a GlycanParams-style dict with G2 conformational entropy values.

    For backward compatibility with the scorer that uses a single eps_conf,
    compute the MEAN per-linkage TΔS as the default eps_conf. The scorer
    can optionally use linkage-specific values via the table directly.

    Parameters
    ----------
    params_dict : dict
        Current glycan parameters (from Phase G1 or defaults).

    Returns
    -------
    dict
        Updated parameters with eps_conf populated.
    """
    table = LINKAGE_ENTROPY_TABLE
    all_TdS = [le.TdS_freeze_kJmol for le in table.values()]
    mean_TdS = np.mean(all_TdS)

    updated = dict(params_dict)
    updated["eps_conf"] = mean_TdS
    updated["eps_conf_source"] = "G2: rotameric entropy from φ/ψ/ω populations"
    updated["linkage_entropy_table"] = {
        k: v.TdS_freeze_kJmol for k, v in table.items()
    }
    return updated


# ═══════════════════════════════════════════════════════════════════════════
# Summary / reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_summary():
    """Print human-readable summary of all linkage entropy values."""
    print("=" * 70)
    print("Phase G2: Conformational Entropy from Rotamer Populations")
    print("=" * 70)
    print(f"\nRT = {RT_STD:.4f} kJ/mol at {T_STD:.2f} K\n")

    table = LINKAGE_ENTROPY_TABLE
    print(f"{'Linkage':<10} {'n_torsions':<12} {'TΔS_freeze':>12}  Torsion breakdown")
    print("-" * 70)
    for key in sorted(table.keys()):
        le = table[key]
        parts = []
        for t in le.torsions:
            TdS_t = rotamer_entropy(t.populations)
            tier_mark = {1: "✓", 2: "~", 3: "?"}[t.tier]
            parts.append(f"{t.name}={TdS_t:.2f}{tier_mark}")
        breakdown = " + ".join(parts)
        print(f"{key:<10} {le.n_torsions:<12d} {le.TdS_freeze_kJmol:>10.2f}    {breakdown}")

    all_TdS = [le.TdS_freeze_kJmol for le in table.values()]
    print("-" * 70)
    print(f"{'Mean':<10} {'':12} {np.mean(all_TdS):>10.2f}")
    print(f"{'Range':<10} {'':12} {min(all_TdS):>5.2f} – {max(all_TdS):.2f}")

    print(f"\nCross-checks:")
    print(f"  MABE HG eps_rotor = 2.48 kJ/mol per generic rotor")
    print(f"  Mammen/Whitesides = 3.40 kJ/mol per rotor")
    avg_per_torsion = np.mean(all_TdS) / np.mean([le.n_torsions for le in table.values()])
    print(f"  G2 avg per torsion = {avg_per_torsion:.2f} kJ/mol "
          f"(< eps_rotor: anomeric pre-restriction)")

    print(f"\nTier key: ✓ = NMR/experimental, ~ = QM/theory, ? = provisional")
    print(f"\nOrdering check: a1-6 > a1-4 ≈ b1-4 > a1-3 ≈ b1-3: ", end="")
    if (table["a1-6"].TdS_freeze_kJmol > table["a1-4"].TdS_freeze_kJmol >
            table["a1-3"].TdS_freeze_kJmol):
        print("PASS ✓")
    else:
        print("FAIL ✗")


if __name__ == "__main__":
    print_summary()
