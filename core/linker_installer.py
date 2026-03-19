"""
core/linker_installer.py — Click Handle Installation on Binder Candidates

Takes a scored binder candidate + attachment site analysis, computationally
installs a click-chemistry handle (PEG-azide, propargylamine, maleimide,
etc.), and re-scores the modified molecule to verify binding is maintained.

This is the "functionalization-readiness" step: transforms a binder-only
molecule into a conjugation-ready binder for scaffold display.

Architecture:
  1. Take top-scoring candidates from Pareto front
  2. Analyze attachment sites (from click_attachment.py)
  3. For each viable site, install a click handle via SMILES surgery
  4. Re-score the modified molecule
  5. Report binding retention (ΔΔG) and final attachability

Linker types:
  - PEG2-azide:      short PEG spacer + terminal azide (for SPAAC with DBCO)
  - PEG4-azide:      longer PEG spacer + terminal azide
  - Propargyl:        terminal alkyne (for CuAAC)
  - Maleimide-PEG2:  maleimide + PEG spacer (for thiol conjugation)
  - NHS-PEG2:        activated ester (for amine conjugation on scaffold)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops


# ═══════════════════════════════════════════════════════════════════════════
# LINKER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LinkerSpec:
    """Specification for a click-chemistry linker."""
    name: str
    click_chemistry: str        # 'SPAAC', 'CuAAC', 'maleimide-thiol', etc.
    smiles_fragment: str        # SMILES to append (with [*] attachment point)
    mw_addition: float          # MW added by linker
    n_peg_units: int = 0
    spacer_length_nm: float = 0.0  # approximate extended length
    compatible_sites: List[str] = field(default_factory=list)  # functional groups that can bear this linker
    installation_smarts: str = ""   # reaction SMARTS for installation


LINKER_LIBRARY = [
    LinkerSpec(
        name="PEG2-azide",
        click_chemistry="SPAAC",
        smiles_fragment="COCCOCCOCCOCN=[N+]=[N-]",
        mw_addition=218.2,
        n_peg_units=2,
        spacer_length_nm=1.5,
        compatible_sites=['primary_amine', 'carboxylic_acid', 'secondary_amine'],
    ),
    LinkerSpec(
        name="PEG4-azide",
        click_chemistry="SPAAC",
        smiles_fragment="COCCOCCOCCOCCOCCOCCOCCOCN=[N+]=[N-]",
        mw_addition=306.4,
        n_peg_units=4,
        spacer_length_nm=2.5,
        compatible_sites=['primary_amine', 'carboxylic_acid'],
    ),
    LinkerSpec(
        name="propargyl",
        click_chemistry="CuAAC",
        smiles_fragment="C#C",
        mw_addition=39.0,
        spacer_length_nm=0.5,
        compatible_sites=['primary_amine', 'carboxylic_acid', 'alcohol', 'secondary_amine'],
    ),
    LinkerSpec(
        name="propargyl-PEG2",
        click_chemistry="CuAAC",
        smiles_fragment="CCOCCOCCOC#C",
        mw_addition=170.2,
        n_peg_units=2,
        spacer_length_nm=1.8,
        compatible_sites=['primary_amine', 'carboxylic_acid'],
    ),
    LinkerSpec(
        name="azidoethyl",
        click_chemistry="SPAAC",
        smiles_fragment="CCN=[N+]=[N-]",
        mw_addition=84.1,
        spacer_length_nm=0.7,
        compatible_sites=['primary_amine', 'carboxylic_acid', 'secondary_amine', 'benzylic_CH2'],
    ),
    LinkerSpec(
        name="aminoethyl-azide",
        click_chemistry="SPAAC",
        smiles_fragment="NCCN=[N+]=[N-]",
        mw_addition=99.1,
        spacer_length_nm=0.8,
        compatible_sites=['carboxylic_acid'],
    ),
]

LINKER_BY_NAME = {l.name: l for l in LINKER_LIBRARY}


# ═══════════════════════════════════════════════════════════════════════════
# HANDLE INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InstalledHandle:
    """Result of installing a click handle on a binder."""
    original_smiles: str
    original_name: str
    modified_smiles: str            # SMILES with handle installed
    modified_valid: bool = False
    linker_name: str = ""
    click_chemistry: str = ""
    site_functional_group: str = ""
    site_atom_idx: int = -1
    mw_original: float = 0.0
    mw_modified: float = 0.0
    spacer_length_nm: float = 0.0
    # Binding retention
    dg_original: float = 0.0
    dg_modified: float = 0.0
    ddg: float = 0.0                # dG_modified - dG_original (positive = weakened)
    binding_retained: bool = False   # |ΔΔG| < threshold
    retention_fraction: float = 0.0  # |dG_modified / dG_original|


def install_handle_on_amine(smiles: str, atom_idx: int, linker: LinkerSpec) -> Optional[str]:
    """Install a linker on a primary amine via amide coupling.

    Chemistry: R-NH2 + linker-COOH → R-NH-CO-linker (amide bond)
    Simplified: replace NH2 with NH-CO-linker fragment.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Simple approach: use reaction SMARTS for amide coupling
    # R-[NH2] + azido-acetic acid → R-NH-C(=O)-CH2-N3
    if linker.name == "azidoethyl":
        rxn_smarts = '[NH2:1]>>[NH:1]C(=O)CN=[N+]=[N-]'
    elif linker.name == "PEG2-azide":
        rxn_smarts = '[NH2:1]>>[NH:1]C(=O)COCCOCCN=[N+]=[N-]'
    elif linker.name == "propargyl":
        rxn_smarts = '[NH2:1]>>[NH:1]CC#C'
    elif linker.name == "propargyl-PEG2":
        rxn_smarts = '[NH2:1]>>[NH:1]C(=O)COCCOC#C'
    elif linker.name == "aminoethyl-azide":
        return None  # can't do amine-to-amine
    elif linker.name == "PEG4-azide":
        rxn_smarts = '[NH2:1]>>[NH:1]C(=O)COCCOCCOCCOCCN=[N+]=[N-]'
    else:
        return None

    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if rxn is None:
        return None

    products = rxn.RunReactants((mol,))
    if not products:
        return None

    # Take first product
    prod = products[0][0]
    try:
        Chem.SanitizeMol(prod)
        return Chem.MolToSmiles(prod)
    except Exception:
        return None


def install_handle_on_cooh(smiles: str, atom_idx: int, linker: LinkerSpec) -> Optional[str]:
    """Install a linker on a carboxylic acid via amide coupling.

    Chemistry: R-COOH + H2N-linker → R-CO-NH-linker
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if linker.name == "aminoethyl-azide":
        rxn_smarts = '[CX3:1](=[OX1])[OX2H1]>>[C:1](=O)NCCN=[N+]=[N-]'
    elif linker.name == "PEG2-azide":
        rxn_smarts = '[CX3:1](=[OX1])[OX2H1]>>[C:1](=O)NCCOCCOCCN=[N+]=[N-]'
    elif linker.name == "propargyl":
        rxn_smarts = '[CX3:1](=[OX1])[OX2H1]>>[C:1](=O)NCC#C'
    elif linker.name == "propargyl-PEG2":
        rxn_smarts = '[CX3:1](=[OX1])[OX2H1]>>[C:1](=O)NCCOCCOC#C'
    else:
        return None

    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if rxn is None:
        return None

    products = rxn.RunReactants((mol,))
    if not products:
        return None

    prod = products[0][0]
    try:
        Chem.SanitizeMol(prod)
        return Chem.MolToSmiles(prod)
    except Exception:
        return None


def install_handle_on_thiol(smiles: str, atom_idx: int, linker: LinkerSpec) -> Optional[str]:
    """Install a linker on a thiol via thiol-maleimide or thioether."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Thiol alkylation: R-SH + BrCH2-linker → R-S-CH2-linker
    if linker.name == "azidoethyl":
        rxn_smarts = '[SH:1]>>[S:1]CCN=[N+]=[N-]'
    elif linker.name == "propargyl":
        rxn_smarts = '[SH:1]>>[S:1]CC#C'
    else:
        return None

    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if rxn is None:
        return None

    products = rxn.RunReactants((mol,))
    if not products:
        return None

    prod = products[0][0]
    try:
        Chem.SanitizeMol(prod)
        return Chem.MolToSmiles(prod)
    except Exception:
        return None


INSTALL_FUNCTIONS = {
    'primary_amine': install_handle_on_amine,
    'secondary_amine': install_handle_on_amine,
    'carboxylic_acid': install_handle_on_cooh,
    'thiol': install_handle_on_thiol,
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN API: INSTALL + RE-SCORE
# ═══════════════════════════════════════════════════════════════════════════

def install_and_rescore(
    smiles: str,
    name: str,
    dg_original: float,
    attachment_analysis=None,
    score_fn=None,
    ddg_threshold: float = 5.0,  # kJ/mol: max acceptable binding loss
) -> List[InstalledHandle]:
    """
    Try all compatible linkers on all viable sites, re-score each.

    Args:
        smiles: original binder SMILES
        name: binder name
        dg_original: original ΔG_bind (kJ/mol)
        attachment_analysis: from click_attachment.analyze_attachment()
        score_fn: callable(smiles) → dG_bind in kJ/mol
        ddg_threshold: max acceptable binding loss (kJ/mol)

    Returns:
        List[InstalledHandle] sorted by binding retention (best first)
    """
    if attachment_analysis is None:
        from core.click_attachment import analyze_attachment
        attachment_analysis = analyze_attachment(smiles, name)

    if score_fn is None:
        # Default: use GalNAc scorer
        from core.galnac_binder_scorer import score_galnac_binder
        def score_fn(s):
            r = score_galnac_binder(s, include_selectivity=False)
            return r.dg_galnac if r.valid else 0.0

    mol_orig = Chem.MolFromSmiles(smiles)
    mw_orig = Chem.Descriptors.MolWt(mol_orig) if mol_orig else 0

    results = []

    for site in attachment_analysis.sites:
        fg = site.functional_group
        install_fn = INSTALL_FUNCTIONS.get(fg)
        if install_fn is None:
            continue

        for linker in LINKER_LIBRARY:
            if fg not in linker.compatible_sites:
                continue

            modified_smi = install_fn(smiles, site.atom_idx, linker)
            if modified_smi is None:
                continue

            # Verify valid molecule
            mod_mol = Chem.MolFromSmiles(modified_smi)
            if mod_mol is None:
                continue

            mw_mod = Chem.Descriptors.MolWt(mod_mol)

            # Re-score
            dg_mod = score_fn(modified_smi)

            ddg = dg_mod - dg_original  # positive = weakened binding
            retained = abs(ddg) < ddg_threshold
            retention = abs(dg_mod / dg_original) if dg_original != 0 else 0

            results.append(InstalledHandle(
                original_smiles=smiles,
                original_name=name,
                modified_smiles=modified_smi,
                modified_valid=True,
                linker_name=linker.name,
                click_chemistry=linker.click_chemistry,
                site_functional_group=fg,
                site_atom_idx=site.atom_idx,
                mw_original=mw_orig,
                mw_modified=mw_mod,
                spacer_length_nm=linker.spacer_length_nm,
                dg_original=dg_original,
                dg_modified=dg_mod,
                ddg=ddg,
                binding_retained=retained,
                retention_fraction=retention,
            ))

    # Sort by binding retention (smallest |ΔΔG| first)
    results.sort(key=lambda h: abs(h.ddg))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING FOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def functionalize_top_candidates(
    candidates: list,  # list of dicts with 'smiles', 'name', 'dg_galnac'
    max_candidates: int = 20,
    ddg_threshold: float = 5.0,
    verbose: bool = True,
) -> List[Dict]:
    """
    Take top candidates from Pareto front, install handles, re-score.

    Returns list of functionalized candidate specs.
    """
    from core.click_attachment import analyze_attachment

    results = []

    for i, cand in enumerate(candidates[:max_candidates]):
        smiles = cand.get('smiles', cand.smiles if hasattr(cand, 'smiles') else '')
        name = cand.get('name', cand.name if hasattr(cand, 'name') else f'candidate_{i}')
        dg = cand.get('dg_galnac', cand.dg_galnac if hasattr(cand, 'dg_galnac') else 0)

        if not smiles or dg == 0:
            continue

        # Analyze attachment
        analysis = analyze_attachment(smiles, name)
        if analysis.n_sites_found == 0:
            if verbose:
                print(f"  {name[:40]:40s}: no attachment sites")
            continue

        # Try all linkers
        handles = install_and_rescore(
            smiles, name, dg,
            attachment_analysis=analysis,
            ddg_threshold=ddg_threshold,
        )

        retained = [h for h in handles if h.binding_retained]

        if verbose:
            n_total = len(handles)
            n_kept = len(retained)
            best_ddg = handles[0].ddg if handles else 0
            best_linker = handles[0].linker_name if handles else "none"
            print(f"  {name[:40]:40s}: {analysis.n_sites_found} sites, "
                  f"{n_total} linker combos, {n_kept} retained, "
                  f"best ΔΔG={best_ddg:+.1f} ({best_linker})")

        if retained:
            best = retained[0]
            results.append({
                'original_name': name,
                'original_smiles': smiles,
                'original_dg': dg,
                'modified_smiles': best.modified_smiles,
                'linker': best.linker_name,
                'click_chemistry': best.click_chemistry,
                'site_type': best.site_functional_group,
                'dg_modified': best.dg_modified,
                'ddg': best.ddg,
                'retention': best.retention_fraction,
                'spacer_nm': best.spacer_length_nm,
                'mw_modified': best.mw_modified,
                'n_options': len(retained),  # how many linker options work
            })

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("LINKER INSTALLATION — TEST PANEL")
    print("=" * 70)

    test_binders = [
        ("triglycine", "NCC(=O)NCC(=O)NCC(=O)O", -25.0),
        ("tryptophan_deriv", "NC(Cc1c[nH]c2ccccc12)C(=O)O", -17.4),
        ("beta-ala-phenylboronic", "NCCC(=O)Oc1ccc(B(O)O)cc1", -32.3),
        ("glycine-phenylboronic", "NCC(=O)Oc1ccc(B(O)O)cc1", -32.3),
        ("bis_urea_xylylene", "O=C(N)NCc1cccc(CNC(N)=O)c1", -22.3),
    ]

    for name, smi, dg in test_binders:
        print(f"\n{'─' * 50}")
        print(f"Binder: {name} (ΔG = {dg:+.1f} kJ/mol)")
        print(f"{'─' * 50}")

        handles = install_and_rescore(smi, name, dg)

        if not handles:
            print("  No viable linker installations found")
            continue

        print(f"  {'Linker':>20s} {'Chemistry':>8s} {'Site':>16s} "
              f"{'ΔG_mod':>7s} {'ΔΔG':>6s} {'Retained':>8s}")
        for h in handles[:5]:
            r = "✓" if h.binding_retained else "✗"
            print(f"  {h.linker_name:>20s} {h.click_chemistry:>8s} "
                  f"{h.site_functional_group:>16s} {h.dg_modified:+7.1f} "
                  f"{h.ddg:+6.1f} {r:>8s}")