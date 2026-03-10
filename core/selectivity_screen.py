"""
core/selectivity_screen.py — Small-Molecule Selectivity Screening

Scores a target guest against a panel of structural interferents through
the same host cavities, computing selectivity ratios per host.

Physics basis:
    - Same unified_scorer_v2 HG terms for all compounds
    - Selectivity ratio = 10^(log_Ka_target - log_Ka_interferent) per host
    - No fitted selectivity parameters — differential binding emerges from
      the same hydrophobic, H-bond, size-mismatch physics applied to
      different guest structures

Entry point:
    screen_selectivity(target_smiles, interferent_panel, hosts)
    → SelectivityResult with per-host, per-interferent ratios

Does NOT:
    - Use any selectivity-specific fitted parameters
    - Assume which host is "best" for selectivity (that's the output)
    - Modify scorer behavior per compound
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# INTERFERENT PANELS (pre-defined for common targets)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Interferent:
    """One structural interferent to test selectivity against."""
    name: str
    smiles: str
    relationship: str   # "parent", "analog", "fragment", "common"
    notes: str = ""


# 6PPD-Q interferent panel
PANEL_6PPD_Q = [
    Interferent(
        "6PPD", "CC(C)CC(NC1=CC=C(NC2=CC=CC=C2)C=C1)C",
        "parent",
        "Parent amine — reduced form of 6PPD-Q, always co-present in tire leachate",
    ),
    Interferent(
        "DPPD-Q", "O=C1C=C(NC2=CC=CC=C2)C(=O)C=C1NC1=CC=CC=C1",
        "analog",
        "Diphenyl-PPD quinone — symmetric analog, different alkyl chain",
    ),
    Interferent(
        "IPPD-Q", "CC(C)NC1=CC(=O)C(=CC1=O)NC2=CC=CC=C2",
        "analog",
        "Isopropyl-PPD quinone — shorter alkyl chain than 6PPD-Q",
    ),
    Interferent(
        "aniline", "NC1=CC=CC=C1",
        "fragment",
        "Aniline — aromatic amine fragment, common environmental contaminant",
    ),
    Interferent(
        "p-benzoquinone", "O=C1C=CC(=O)C=C1",
        "fragment",
        "p-Benzoquinone — quinone core fragment, ubiquitous",
    ),
    Interferent(
        "phenol", "OC1=CC=CC=C1",
        "common",
        "Phenol — common interferent in environmental water monitoring",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HostSelectivity:
    """Selectivity profile for one host against all interferents."""
    host_key: str
    target_log_ka: float

    # Per-interferent results
    interferent_log_kas: dict = field(default_factory=dict)
    # {interferent_name: log_Ka}
    selectivity_ratios: dict = field(default_factory=dict)
    # {interferent_name: ratio}  where ratio = 10^(target - interferent)

    @property
    def worst_ratio(self) -> float:
        """Smallest selectivity ratio (worst case)."""
        if not self.selectivity_ratios:
            return float("inf")
        return min(self.selectivity_ratios.values())

    @property
    def worst_interferent(self) -> str:
        """Name of the interferent with worst (lowest) selectivity."""
        if not self.selectivity_ratios:
            return ""
        return min(self.selectivity_ratios, key=self.selectivity_ratios.get)

    @property
    def is_selective(self) -> bool:
        """True if selectivity ratio > 1 for all interferents."""
        return all(r > 1.0 for r in self.selectivity_ratios.values())

    @property
    def is_highly_selective(self) -> bool:
        """True if selectivity ratio > 10 for all interferents."""
        return all(r > 10.0 for r in self.selectivity_ratios.values())

    @property
    def grade(self) -> str:
        """Selectivity grade."""
        w = self.worst_ratio
        if w > 100:
            return "excellent"
        elif w > 10:
            return "good"
        elif w > 1:
            return "marginal"
        else:
            return "non-selective"


@dataclass
class SelectivityResult:
    """Full selectivity screening result."""
    target_name: str
    target_smiles: str
    interferents: list = field(default_factory=list)  # list[Interferent]

    # Per-host selectivity
    host_selectivity: list = field(default_factory=list)  # list[HostSelectivity]

    # Summary
    best_host: str = ""
    best_host_worst_ratio: float = 0.0
    n_selective_hosts: int = 0         # ratio > 1 for all interferents
    n_highly_selective_hosts: int = 0  # ratio > 10 for all interferents

    # MIP selectivity (if computed)
    mip_selectivity: dict = field(default_factory=dict)
    # {interferent_name: estimated_selectivity_factor}


# ═══════════════════════════════════════════════════════════════════════════
# CORE SCREENING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def screen_selectivity(
    target_smiles: str,
    target_name: str = "",
    interferents: list = None,
    hosts: list = None,
    include_mip: bool = True,
) -> SelectivityResult:
    """Screen target selectivity against interferent panel across all hosts.

    Args:
        target_smiles: SMILES of target molecule
        target_name: display name
        interferents: list[Interferent] or None (uses built-in panel if target
                      matches a known compound)
        hosts: host keys to screen (default: all in HOST_DB)
        include_mip: estimate MIP selectivity from pharmacophore similarity

    Returns:
        SelectivityResult with per-host selectivity ratios.
    """
    from core.small_molecule_target import screen_hosts

    # Auto-select panel if not provided
    if interferents is None:
        interferents = _auto_select_panel(target_smiles)

    result = SelectivityResult(
        target_name=target_name or target_smiles[:40],
        target_smiles=target_smiles,
        interferents=interferents,
    )

    if not interferents:
        return result

    # Score target
    target_results = screen_hosts(target_smiles, hosts=hosts, name=target_name)
    target_by_host = {r.host_key: r for r in target_results}

    # Score each interferent
    interferent_scores = {}  # {name: {host_key: log_Ka}}
    for intf in interferents:
        intf_results = screen_hosts(intf.smiles, hosts=hosts, name=intf.name)
        interferent_scores[intf.name] = {
            r.host_key: r.log_Ka_pred for r in intf_results
        }

    # Compute selectivity per host
    for host_key, target_r in target_by_host.items():
        target_log_ka = target_r.log_Ka_pred
        if target_log_ka == float("-inf"):
            continue

        hs = HostSelectivity(
            host_key=host_key,
            target_log_ka=target_log_ka,
        )

        for intf in interferents:
            intf_log_ka = interferent_scores.get(intf.name, {}).get(host_key, float("-inf"))

            hs.interferent_log_kas[intf.name] = intf_log_ka

            if intf_log_ka == float("-inf"):
                # Interferent can't bind → infinite selectivity
                hs.selectivity_ratios[intf.name] = float("inf")
            else:
                # Selectivity ratio = 10^(target - interferent)
                delta = target_log_ka - intf_log_ka
                hs.selectivity_ratios[intf.name] = 10.0 ** delta

        result.host_selectivity.append(hs)

    # Sort by worst-case selectivity (best host first)
    result.host_selectivity.sort(
        key=lambda h: h.worst_ratio if h.worst_ratio != float("inf") else 1e10,
        reverse=True,
    )

    # Summary stats
    result.n_selective_hosts = sum(
        1 for h in result.host_selectivity if h.is_selective
    )
    result.n_highly_selective_hosts = sum(
        1 for h in result.host_selectivity if h.is_highly_selective
    )

    if result.host_selectivity:
        best = result.host_selectivity[0]
        result.best_host = best.host_key
        result.best_host_worst_ratio = best.worst_ratio

    # MIP selectivity estimate
    if include_mip:
        result.mip_selectivity = _estimate_mip_selectivity(
            target_smiles, interferents
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MIP SELECTIVITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_mip_selectivity(target_smiles, interferents):
    """Estimate MIP selectivity from molecular similarity.

    MIP selectivity is governed by shape + functional group complementarity
    to the imprinted cavity. Molecules that differ in:
      - Volume → poor cavity fit → reduced binding
      - Functional group pattern → H-bond mismatch → reduced binding
      - Shape → steric clash → reduced binding

    We estimate this from RDKit fingerprint similarity (Tanimoto) and
    volume ratio. This is a heuristic — DFT refinement improves it.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, DataStructs

        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            return {}

        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
        target_mw = Descriptors.ExactMolWt(target_mol)

        mip_sel = {}
        for intf in interferents:
            intf_mol = Chem.MolFromSmiles(intf.smiles)
            if intf_mol is None:
                mip_sel[intf.name] = 1.0
                continue

            intf_fp = AllChem.GetMorganFingerprintAsBitVect(intf_mol, 2, nBits=2048)
            intf_mw = Descriptors.ExactMolWt(intf_mol)

            # Tanimoto similarity
            tanimoto = DataStructs.TanimotoSimilarity(target_fp, intf_fp)

            # Volume ratio penalty (MIP cavity is template-sized)
            mw_ratio = intf_mw / target_mw if target_mw > 0 else 1.0
            # Penalty: molecules much larger or smaller than template bind worse
            volume_factor = math.exp(-2.0 * (mw_ratio - 1.0) ** 2)

            # MIP selectivity factor estimate:
            # High similarity → low selectivity (interferent fits the cavity)
            # Low similarity → high selectivity (interferent doesn't fit)
            # Volume mismatch → higher selectivity (can't enter cavity)
            #
            # IF_ratio ≈ 1 / (tanimoto * volume_factor)
            # Clamped to [1, 1000]
            combined = tanimoto * volume_factor
            if combined < 0.01:
                mip_sel[intf.name] = 1000.0
            else:
                mip_sel[intf.name] = min(1000.0, max(1.0, 1.0 / combined))

        return mip_sel

    except ImportError:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# PANEL AUTO-SELECTION
# ═══════════════════════════════════════════════════════════════════════════

# Known panels keyed by canonical SMILES
_KNOWN_PANELS = {}


def _register_panel(target_smiles, panel):
    """Register a pre-defined interferent panel for a target."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(target_smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            _KNOWN_PANELS[canonical] = panel
    except ImportError:
        pass


# Register 6PPD-Q panel
_register_panel(
    "CC(C)CC(NC1=CC(=O)C(=CC1=O)NC2=CC=CC=C2)C",
    PANEL_6PPD_Q,
)


def _auto_select_panel(target_smiles):
    """Auto-select interferent panel if target matches a known compound."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(target_smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            if canonical in _KNOWN_PANELS:
                return _KNOWN_PANELS[canonical]
    except ImportError:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_selectivity(result, verbose=False):
    """Print selectivity screening results."""
    print(f"\n{'═' * 70}")
    print(f"SELECTIVITY: {result.target_name}")
    print(f"{'═' * 70}")
    print(f"  Interferents: {len(result.interferents)}")
    print(f"  Hosts screened: {len(result.host_selectivity)}")
    print(f"  Selective hosts (>1×): {result.n_selective_hosts}")
    print(f"  Highly selective (>10×): {result.n_highly_selective_hosts}")

    if result.best_host:
        print(f"  Best host: {result.best_host} "
              f"(worst ratio: {result.best_host_worst_ratio:.1f}×)")

    # Per-host table
    if result.host_selectivity:
        intf_names = [i.name for i in result.interferents]

        print(f"\n  {'Host':>15s}  {'target':>8s}", end="")
        for n in intf_names:
            print(f"  {n:>12s}", end="")
        print(f"  {'worst':>8s}  {'grade':>10s}")

        print(f"  {'':>15s}  {'log Ka':>8s}", end="")
        for _ in intf_names:
            print(f"  {'ratio':>12s}", end="")
        print()

        for hs in result.host_selectivity:
            print(f"  {hs.host_key:>15s}  {hs.target_log_ka:>8.2f}", end="")
            for n in intf_names:
                r = hs.selectivity_ratios.get(n, 0)
                if r == float("inf"):
                    print(f"  {'∞':>12s}", end="")
                elif r >= 100:
                    print(f"  {r:>11.0f}×", end="")
                else:
                    print(f"  {r:>11.2f}×", end="")
            wr = hs.worst_ratio
            if wr == float("inf"):
                print(f"  {'∞':>8s}", end="")
            else:
                print(f"  {wr:>7.2f}×", end="")
            print(f"  {hs.grade:>10s}")

    # MIP selectivity
    if result.mip_selectivity:
        print(f"\n  MIP estimated selectivity (imprinting factor ratio):")
        for name, sel in sorted(result.mip_selectivity.items(),
                                key=lambda x: x[1], reverse=True):
            rel = next((i.relationship for i in result.interferents
                       if i.name == name), "")
            print(f"    vs {name:20s} ({rel:8s}): {sel:>8.1f}×")
