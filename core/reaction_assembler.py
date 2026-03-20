"""
core/reaction_assembler.py -- Reaction-aware molecular assembly.

Every bond in a generated molecule should correspond to a known synthetic reaction.
This module validates assembly steps and annotates candidates with reaction routes.

Reaction library (SMARTS-based):
  - Amide coupling (amine + carboxylic acid)
  - Suzuki coupling (aryl halide + boronic acid)
  - CuAAC click (azide + terminal alkyne -> 1,2,3-triazole)
  - SPAAC click (azide + strained alkyne, no catalyst)
  - Reductive amination (amine + aldehyde)
  - Schiff base (amine + aldehyde -> imine)
  - Urea formation (amine + isocyanate)
  - Sulfonamide (amine + sulfonyl chloride)
  - Ether formation (Williamson: alkoxide + alkyl halide)
  - Ester formation (alcohol + acid)

Each reaction has:
  - Name and reference
  - Reactant SMARTS (A + B)
  - Product SMARTS (validates the bond in the product)
  - Conditions (catalyst, solvent, temperature)
  - Reliability rating (robust/moderate/challenging)

Entry points:
  identify_bonds(smiles) -> list of BondAnnotation
  validate_synthesis(smiles) -> SynthesisRoute
  annotate_candidate(smiles) -> AnnotatedCandidate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Reaction definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactionDef:
    """A synthetic reaction definition."""
    name: str
    product_smarts: str         # SMARTS pattern that appears in product
    conditions: str             # brief conditions description
    reliability: str            # "robust", "moderate", "challenging"
    reference: str = ""         # literature reference
    catalyst: str = ""
    notes: str = ""


REACTION_LIBRARY: List[ReactionDef] = [
    # Amide bond
    ReactionDef("amide_coupling",
                "[NX3][CX3](=O)[#6]",
                "EDC/HOBt, DIPEA, DMF, RT",
                "robust",
                reference="General peptide coupling",
                catalyst="EDC/HOBt or HATU"),
    # Sulfonamide
    ReactionDef("sulfonamide_formation",
                "[NX3]S(=O)(=O)",
                "Et3N, DCM, 0C to RT",
                "robust",
                reference="Standard sulfonylation"),
    # Urea
    ReactionDef("urea_formation",
                "[NX3]C(=O)[NX3]",
                "CDI or isocyanate + amine, DCM",
                "robust",
                reference="Isocyanate + amine"),
    # Ether (aryl or alkyl)
    ReactionDef("ether_williamson",
                "[OX2]([CX4])[c,C]",
                "K2CO3, DMF, 80C or NaH, THF",
                "robust",
                reference="Williamson ether synthesis"),
    # Ester
    ReactionDef("ester_formation",
                "[CX3](=O)[OX2][C]",
                "DCC/DMAP or Fischer (H2SO4, reflux)",
                "robust",
                reference="Steglich or Fischer esterification"),
    # Suzuki coupling
    ReactionDef("suzuki_coupling",
                "[c]:[c]",  # biaryl bond
                "Pd(PPh3)4, K2CO3, DME/H2O, 80C",
                "robust",
                reference="Miyaura & Suzuki 1995",
                catalyst="Pd(0)"),
    # CuAAC click
    ReactionDef("CuAAC_click",
                "c1cn(nn1)",  # 1,2,3-triazole
                "CuSO4, sodium ascorbate, H2O/tBuOH",
                "robust",
                reference="Sharpless 2002",
                catalyst="Cu(I)"),
    # SPAAC click (Cu-free)
    ReactionDef("SPAAC_click",
                "c1cn(nn1)",  # same triazole product
                "No catalyst, RT, aqueous compatible",
                "robust",
                reference="Bertozzi 2004",
                notes="Requires DBCO or BCN strained alkyne"),
    # Reductive amination
    ReactionDef("reductive_amination",
                "[NX3][CX4]",  # amine-CH2 bond
                "NaBH3CN or NaBH(OAc)3, MeOH/AcOH",
                "robust",
                reference="Standard reductive amination"),
    # Schiff base / imine
    ReactionDef("schiff_base",
                "[NX2]=[CX3]",
                "Molecular sieves, EtOH, reflux",
                "moderate",
                reference="Imine condensation",
                notes="Reversible unless reduced"),
    # Buchwald-Hartwig (C-N on arene)
    ReactionDef("buchwald_hartwig",
                "[NX3]c",  # aryl amine
                "Pd2(dba)3, XPhos, KOtBu, toluene, 100C",
                "moderate",
                reference="Buchwald & Hartwig",
                catalyst="Pd(0)"),
    # Sonogashira (C-C to alkyne)
    ReactionDef("sonogashira",
                "[c]C#C",
                "PdCl2(PPh3)2, CuI, Et3N",
                "moderate",
                reference="Sonogashira 1975",
                catalyst="Pd(0)/Cu(I)"),
    # Thiol-maleimide
    ReactionDef("thiol_maleimide",
                "[SX2][CX4][CX3](=O)",
                "PBS pH 7.0, RT, 1h",
                "robust",
                reference="Standard bioconjugation",
                notes="Michael addition, fast at neutral pH"),
    # Boronate ester (for boronic acid sensors)
    ReactionDef("boronate_ester",
                "[B]([O])([O])",
                "Aqueous, pH 7-9, spontaneous",
                "robust",
                reference="Boronic acid-diol equilibrium",
                notes="Reversible, pH-dependent"),
]


# ---------------------------------------------------------------------------
# Bond annotation
# ---------------------------------------------------------------------------

@dataclass
class BondAnnotation:
    """Annotation of a specific bond in a molecule."""
    bond_idx: int
    begin_atom: int
    end_atom: int
    bond_type: str
    reaction_name: str
    reaction_conditions: str
    reliability: str
    catalyst: str = ""


def identify_reactions(smiles: str) -> List[BondAnnotation]:
    """
    Identify which bonds in a molecule correspond to known reactions.

    Matches product SMARTS from the reaction library against the molecule.
    Returns annotations for each matched bond.
    """
    if not HAS_RDKIT:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    annotations = []
    seen_bonds = set()

    for rxn in REACTION_LIBRARY:
        pat = Chem.MolFromSmarts(rxn.product_smarts)
        if pat is None:
            continue

        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            # Find bonds within the match
            for i, ai in enumerate(match):
                for j, aj in enumerate(match):
                    if i >= j:
                        continue
                    bond = mol.GetBondBetweenAtoms(ai, aj)
                    if bond is not None and bond.GetIdx() not in seen_bonds:
                        seen_bonds.add(bond.GetIdx())
                        annotations.append(BondAnnotation(
                            bond_idx=bond.GetIdx(),
                            begin_atom=ai, end_atom=aj,
                            bond_type=str(bond.GetBondType()),
                            reaction_name=rxn.name,
                            reaction_conditions=rxn.conditions,
                            reliability=rxn.reliability,
                            catalyst=rxn.catalyst,
                        ))

    return annotations


# ---------------------------------------------------------------------------
# Synthesis route
# ---------------------------------------------------------------------------

@dataclass
class SynthesisRoute:
    """Retrosynthetic analysis of a candidate molecule."""
    smiles: str
    valid: bool = False
    n_bonds_total: int = 0
    n_bonds_annotated: int = 0
    annotation_coverage: float = 0.0  # fraction of non-ring bonds annotated
    reactions_used: List[str] = field(default_factory=list)
    annotations: List[BondAnnotation] = field(default_factory=list)
    max_reliability: str = "robust"     # worst reliability in route
    requires_catalyst: List[str] = field(default_factory=list)
    route_summary: str = ""


def validate_synthesis(smiles: str) -> SynthesisRoute:
    """
    Validate that a molecule can be assembled from known reactions.

    Computes annotation coverage: what fraction of non-ring, non-aromatic
    bonds correspond to known synthetic reactions.
    """
    if not HAS_RDKIT:
        return SynthesisRoute(smiles=smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return SynthesisRoute(smiles=smiles)

    # Count non-ring, non-aromatic bonds (these are the "assembly" bonds)
    assembly_bonds = set()
    ri = mol.GetRingInfo()
    ring_bonds = set()
    for ring in ri.BondRings():
        ring_bonds.update(ring)

    for bond in mol.GetBonds():
        if bond.GetIdx() not in ring_bonds and not bond.GetIsAromatic():
            assembly_bonds.add(bond.GetIdx())

    n_assembly = len(assembly_bonds)

    annotations = identify_reactions(smiles)

    # How many assembly bonds are annotated?
    annotated_assembly = set()
    for ann in annotations:
        if ann.bond_idx in assembly_bonds:
            annotated_assembly.add(ann.bond_idx)

    coverage = len(annotated_assembly) / n_assembly if n_assembly > 0 else 1.0

    reactions = sorted(set(a.reaction_name for a in annotations))
    catalysts = sorted(set(a.catalyst for a in annotations if a.catalyst))

    # Worst reliability
    reliabilities = [a.reliability for a in annotations]
    if "challenging" in reliabilities:
        worst = "challenging"
    elif "moderate" in reliabilities:
        worst = "moderate"
    else:
        worst = "robust"

    # Summary
    steps = []
    for rxn_name in reactions:
        rxn_anns = [a for a in annotations if a.reaction_name == rxn_name]
        steps.append(f"{rxn_name} (x{len(rxn_anns)})")
    summary = " -> ".join(steps) if steps else "no recognized reactions"

    return SynthesisRoute(
        smiles=smiles, valid=True,
        n_bonds_total=mol.GetNumBonds(),
        n_bonds_annotated=len(annotated_assembly),
        annotation_coverage=round(coverage, 2),
        reactions_used=reactions,
        annotations=annotations,
        max_reliability=worst,
        requires_catalyst=catalysts,
        route_summary=summary,
    )


# ---------------------------------------------------------------------------
# Annotated candidate (combines properties + synthesis)
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedCandidate:
    """A fully annotated candidate molecule."""
    smiles: str
    properties: Optional[object] = None  # MolecularProperties
    synthesis: Optional[SynthesisRoute] = None
    score: Optional[object] = None       # BinderScore or ScoredCandidate
    passes_property_gate: bool = True
    passes_synthesis_check: bool = True
    property_failures: List[str] = field(default_factory=list)


def annotate_candidate(
    smiles: str,
    property_gate=None,
    min_synthesis_coverage: float = 0.3,
) -> AnnotatedCandidate:
    """
    Fully annotate a candidate: properties + synthesis route.
    """
    from core.property_predictor import predict_properties, filter_properties

    props = predict_properties(smiles)
    passes_prop = True
    prop_failures = []
    if property_gate is not None:
        passes_prop, _, prop_failures = filter_properties(smiles, property_gate, props)

    synth = validate_synthesis(smiles)
    passes_synth = synth.annotation_coverage >= min_synthesis_coverage

    return AnnotatedCandidate(
        smiles=smiles,
        properties=props,
        synthesis=synth,
        passes_property_gate=passes_prop,
        passes_synthesis_check=passes_synth,
        property_failures=prop_failures,
    )
