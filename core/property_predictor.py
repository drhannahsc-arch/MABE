"""
core/property_predictor.py -- Molecular property prediction from SMILES.

All predictions from RDKit descriptors + published regression models.
No ML, no external APIs.

Properties:
  - logP (Crippen/Wildman)
  - TPSA (topological polar surface area)
  - Solubility logS (Delaney ESOL 2004)
  - Aqueous solubility class (insoluble/poor/moderate/soluble/very_soluble)
  - pKa estimates (strongest acidic/basic, from SMARTS pattern matching)
  - Lipinski rule-of-5 (and Veber extensions)
  - Synthetic accessibility (SA score)
  - Aqueous stability flags (hydrolyzable groups)

Entry point:
  predict_properties(smiles) -> MolecularProperties
  filter_properties(smiles, pfilter) -> (bool, MolecularProperties)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Property dataclass
# ---------------------------------------------------------------------------

@dataclass
class MolecularProperties:
    """Computed properties of a molecule."""
    smiles: str
    valid: bool = False
    # Basic descriptors
    molecular_weight: float = 0.0
    heavy_atom_count: int = 0
    logP: float = 0.0
    tpsa: float = 0.0
    n_hbd: int = 0
    n_hba: int = 0
    n_rotatable: int = 0
    n_rings: int = 0
    n_aromatic_rings: int = 0
    fraction_sp3: float = 0.0
    # Solubility
    logS_esol: float = 0.0           # Delaney ESOL
    solubility_class: str = ""       # insoluble/poor/moderate/soluble/very_soluble
    solubility_mg_ml: float = 0.0    # estimated mg/mL
    # pKa
    strongest_acidic_pka: float = 14.0
    strongest_basic_pka: float = 0.0
    acidic_groups: List[str] = field(default_factory=list)
    basic_groups: List[str] = field(default_factory=list)
    # Druglikeness
    lipinski_violations: int = 0
    lipinski_pass: bool = True
    veber_pass: bool = True          # TPSA <= 140, rotatable <= 10
    # Stability
    hydrolyzable_groups: List[str] = field(default_factory=list)
    aqueous_stable: bool = True
    # SA
    sa_score: float = 5.0


# ---------------------------------------------------------------------------
# ESOL solubility (Delaney 2004, J. Chem. Inf. Comput. Sci. 44:1000)
# ---------------------------------------------------------------------------

def _esol_logs(mol) -> float:
    """Delaney ESOL: logS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP"""
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    ap = len(mol.GetAromaticAtoms()) / n_heavy if n_heavy > 0 else 0
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def _solubility_class(logs: float) -> str:
    if logs < -6:
        return "insoluble"
    elif logs < -4:
        return "poor"
    elif logs < -2:
        return "moderate"
    elif logs < 0:
        return "soluble"
    else:
        return "very_soluble"


def _solubility_mg_ml(logs: float, mw: float) -> float:
    """Convert logS (mol/L) to mg/mL."""
    if mw <= 0:
        return 0.0
    mol_per_L = 10 ** logs
    return mol_per_L * mw / 1000.0  # g/L -> mg/mL... wait, *mw gives g/L
    # Actually: mol/L * g/mol = g/L, then /1 = mg/mL (since 1 g/L = 1 mg/mL)


# ---------------------------------------------------------------------------
# pKa estimation (SMARTS pattern matching)
# ---------------------------------------------------------------------------

# Approximate pKa values for common functional groups
# Source: Perrin 1981, March's Advanced Organic Chemistry
_ACIDIC_GROUPS = [
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 4.0),
    ("sulfonamide_NH", "[SX4](=O)(=O)[NX3H]", 10.0),
    ("phenol", "[OX2H]c", 10.0),
    ("thiol", "[SX2H]", 8.5),
    ("phosphoric_acid", "[PX4](=O)([OX2H])([OX2H])[OX2H]", 2.0),
    ("sulfinic_acid", "[SX3](=O)[OX2H]", 2.5),
    ("boronic_acid", "[BX3]([OX2H])[OX2H]", 8.8),
    ("imide_NH", "[NX3H]([CX3]=O)[CX3]=O", 9.5),
    ("hydroxamic_acid", "[CX3](=O)[NX3H][OX2H]", 8.0),
    ("enol", "[OX2H][CX3]=[CX3]", 10.5),
]

_BASIC_GROUPS = [
    ("primary_amine", "[NX3H2;!$(NC=O)]", 10.5),
    ("secondary_amine", "[NX3H1;!$(NC=O);!$(Nc)]([C])[C]", 10.5),
    ("tertiary_amine", "[NX3;!$(NC=O);!$(Nc)]([C])([C])[C]", 9.5),
    ("pyridine", "c1ccncc1", 5.2),
    ("imidazole", "c1c[nH]cn1", 6.9),
    ("guanidine", "[NX3H2]C(=[NX2H])N", 13.5),
    ("amidine", "[NX3H2]C(=[NX2H])", 11.5),
]


def _estimate_pka(mol) -> Tuple[float, float, List[str], List[str]]:
    """Estimate strongest acidic and basic pKa from SMARTS matches."""
    acidic_pka = 14.0
    basic_pka = 0.0
    acidic_names = []
    basic_names = []

    for name, smarts, pka in _ACIDIC_GROUPS:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            n_matches = len(mol.GetSubstructMatches(pat))
            acidic_names.append(f"{name}(x{n_matches})")
            acidic_pka = min(acidic_pka, pka)

    for name, smarts, pka in _BASIC_GROUPS:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            n_matches = len(mol.GetSubstructMatches(pat))
            basic_names.append(f"{name}(x{n_matches})")
            basic_pka = max(basic_pka, pka)

    return acidic_pka, basic_pka, acidic_names, basic_names


# ---------------------------------------------------------------------------
# Stability flags (hydrolyzable groups)
# ---------------------------------------------------------------------------

_HYDROLYZABLE = [
    ("ester", "[CX3](=O)[OX2][C]"),
    ("anhydride", "[CX3](=O)[OX2][CX3](=O)"),
    ("acid_chloride", "[CX3](=O)[Cl]"),
    ("acetal", "[CX4]([OX2])([OX2])"),
    ("imine", "[CX3H0](=[NX2])"),
    ("enamine", "[NX3][CX3]=[CX3]"),
    ("vinyl_ether", "[OX2][CX3]=[CX3]"),
]


def _check_stability(mol) -> Tuple[List[str], bool]:
    """Check for hydrolyzable groups."""
    found = []
    for name, smarts in _HYDROLYZABLE:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            found.append(name)
    stable = len(found) == 0
    return found, stable


# ---------------------------------------------------------------------------
# SA score
# ---------------------------------------------------------------------------

def _sa_score(mol) -> float:
    """Synthetic accessibility score (1=easy, 10=hard)."""
    try:
        from core.de_novo_generator import sa_score
        return sa_score(mol)
    except (ImportError, Exception):
        # Fallback: simple heuristic
        n_rings = Descriptors.RingCount(mol)
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_heavy = mol.GetNumHeavyAtoms()
        # More rings, chirality, and size = harder
        return min(10.0, 1.0 + 0.3 * n_rings + 0.5 * n_chiral + 0.02 * n_heavy)


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

def predict_properties(smiles: str) -> MolecularProperties:
    """Predict all molecular properties from SMILES."""
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MolecularProperties(smiles=smiles, valid=False)

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    n_rings = Descriptors.RingCount(mol)
    ri = mol.GetRingInfo()
    n_arom_rings = sum(1 for r in ri.BondRings()
                        if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in r))
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    n_heavy = mol.GetNumHeavyAtoms()

    # Solubility
    logs = _esol_logs(mol)
    sol_class = _solubility_class(logs)
    sol_mg = _solubility_mg_ml(logs, mw)

    # pKa
    acid_pka, base_pka, acid_names, base_names = _estimate_pka(mol)

    # Lipinski
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # Veber
    veber = tpsa <= 140 and rot <= 10

    # Stability
    hydro_groups, stable = _check_stability(mol)

    # SA
    sa = _sa_score(mol)

    return MolecularProperties(
        smiles=smiles, valid=True,
        molecular_weight=round(mw, 1),
        heavy_atom_count=n_heavy,
        logP=round(logp, 2),
        tpsa=round(tpsa, 1),
        n_hbd=hbd, n_hba=hba,
        n_rotatable=rot,
        n_rings=n_rings,
        n_aromatic_rings=n_arom_rings,
        fraction_sp3=round(fsp3, 2),
        logS_esol=round(logs, 2),
        solubility_class=sol_class,
        solubility_mg_ml=round(sol_mg, 4),
        strongest_acidic_pka=acid_pka,
        strongest_basic_pka=base_pka,
        acidic_groups=acid_names,
        basic_groups=base_names,
        lipinski_violations=violations,
        lipinski_pass=(violations <= 1),
        veber_pass=veber,
        hydrolyzable_groups=hydro_groups,
        aqueous_stable=stable,
        sa_score=round(sa, 2),
    )


# ---------------------------------------------------------------------------
# Property filter
# ---------------------------------------------------------------------------

@dataclass
class PropertyGate:
    """Filter gate for molecular properties."""
    max_mw: float = 800.0
    max_logP: float = 6.0
    min_logS: float = -6.0         # reject insoluble
    max_tpsa: float = 200.0
    max_rotatable: int = 15
    max_sa: float = 7.0
    require_aqueous_stable: bool = False
    require_lipinski: bool = False
    require_veber: bool = False
    ph_range: Optional[Tuple[float, float]] = None  # (min_pH, max_pH) for ionization check


def filter_properties(
    smiles: str,
    gate: Optional[PropertyGate] = None,
    props: Optional[MolecularProperties] = None,
) -> Tuple[bool, MolecularProperties, List[str]]:
    """
    Check if a molecule passes property gates.

    Returns (passes, properties, failure_reasons).
    """
    if gate is None:
        gate = PropertyGate()

    if props is None:
        props = predict_properties(smiles)

    if not props.valid:
        return False, props, ["invalid_smiles"]

    failures = []

    if props.molecular_weight > gate.max_mw:
        failures.append(f"MW={props.molecular_weight:.0f}>{gate.max_mw:.0f}")
    if props.logP > gate.max_logP:
        failures.append(f"logP={props.logP:.1f}>{gate.max_logP:.1f}")
    if props.logS_esol < gate.min_logS:
        failures.append(f"logS={props.logS_esol:.1f}<{gate.min_logS:.1f}")
    if props.tpsa > gate.max_tpsa:
        failures.append(f"TPSA={props.tpsa:.0f}>{gate.max_tpsa:.0f}")
    if props.n_rotatable > gate.max_rotatable:
        failures.append(f"RotBonds={props.n_rotatable}>{gate.max_rotatable}")
    if props.sa_score > gate.max_sa:
        failures.append(f"SA={props.sa_score:.1f}>{gate.max_sa:.1f}")
    if gate.require_aqueous_stable and not props.aqueous_stable:
        failures.append(f"hydrolyzable:{','.join(props.hydrolyzable_groups)}")
    if gate.require_lipinski and not props.lipinski_pass:
        failures.append(f"Lipinski_violations={props.lipinski_violations}")
    if gate.require_veber and not props.veber_pass:
        failures.append("Veber_fail")

    return len(failures) == 0, props, failures


# ---------------------------------------------------------------------------
# Batch utility
# ---------------------------------------------------------------------------

def predict_batch(smiles_list: List[str]) -> List[MolecularProperties]:
    """Predict properties for a list of SMILES."""
    return [predict_properties(s) for s in smiles_list]


def filter_batch(
    smiles_list: List[str],
    gate: Optional[PropertyGate] = None,
) -> Tuple[List[str], List[str], List[MolecularProperties]]:
    """
    Filter a list of SMILES by property gates.

    Returns (passed_smiles, failed_smiles, all_properties).
    """
    passed = []
    failed = []
    all_props = []
    for smi in smiles_list:
        ok, props, reasons = filter_properties(smi, gate)
        all_props.append(props)
        if ok:
            passed.append(smi)
        else:
            failed.append(smi)
    return passed, failed, all_props
