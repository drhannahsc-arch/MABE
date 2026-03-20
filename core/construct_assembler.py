"""
core/construct_assembler.py -- Modular construct assembly pipeline.

Assembles complete functional constructs from modular components:
  Recognition element (binder) + Linker + Click handle + Solid support

Pipeline:
  1. ConstructSpec defines the modules
  2. Assembler joins them via SMILES
  3. Property predictor validates the construct
  4. Reaction assembler annotates the synthesis route

Entry point:
  assemble_construct(spec) -> AssembledConstruct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Component library
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkerDef:
    """A linker module."""
    name: str
    smiles: str          # SMILES with [1*] (recognition end) and [2*] (click end)
    length_A: float      # estimated extended length in Angstroms
    n_peg_units: int     # PEG repeat count (0 for non-PEG)
    hydrophilic: bool
    notes: str = ""


@dataclass(frozen=True)
class ClickHandleDef:
    """A click chemistry handle for bioconjugation."""
    name: str
    smiles: str          # SMILES with [1*] (linker end)
    partner: str         # what it reacts with on the bead
    reaction: str        # reaction type
    catalyst_free: bool
    biocompatible: bool
    notes: str = ""


@dataclass(frozen=True)
class SupportDef:
    """A solid support definition (not a SMILES — describes the bead)."""
    name: str
    material: str
    functional_group: str   # what's on the surface
    diameter_um: float
    magnetic: bool
    notes: str = ""


LINKER_LIBRARY: List[LinkerDef] = [
    LinkerDef("PEG2", "[1*]OCCO[2*]", 7.0, 2, True, "Short PEG"),
    LinkerDef("PEG4", "[1*]OCCOCCOCCO[2*]", 14.0, 4, True, "Standard PEG spacer"),
    LinkerDef("PEG8", "[1*]OCCOCCOCCOCCOCCOCCOCCO[2*]", 28.0, 8, True, "Long PEG"),
    LinkerDef("C6-alkyl", "[1*]CCCCCC[2*]", 7.5, 0, False, "Hydrophobic spacer"),
    LinkerDef("C3-alkyl", "[1*]CCC[2*]", 3.8, 0, False, "Short hydrophobic"),
    LinkerDef("aminocaproic", "[1*]NCCCCCC(=O)[2*]", 8.0, 0, False, "Amide-linked C6"),
    LinkerDef("PEG4-amide", "[1*]NC(=O)COCCOCCOCCO[2*]", 16.0, 4, True,
              "Amide + PEG4, common in bioconjugation"),
    LinkerDef("none", "[1*][2*]", 0.0, 0, True, "Direct connection"),
]

CLICK_HANDLE_LIBRARY: List[ClickHandleDef] = [
    ClickHandleDef("azide", "[1*]N=[N+]=[N-]", "DBCO", "SPAAC", True, True,
                    "Cu-free click, most common"),
    ClickHandleDef("terminal-alkyne", "[1*]C#C", "azide", "CuAAC", False, False,
                    "Requires Cu(I) catalyst"),
    ClickHandleDef("thiol", "[1*]S", "maleimide", "thiol-maleimide", True, True,
                    "Fast at pH 7"),
    ClickHandleDef("NHS-ester", "[1*]C(=O)ON1CCC(=O)C1=O", "amine", "NHS-amine", True, True,
                    "Reacts with surface amines"),
    ClickHandleDef("tetrazine", "[1*]c1nnc(nn1)c1ccccc1", "TCO", "iEDDA", True, True,
                    "Fastest bioorthogonal click"),
    ClickHandleDef("biotin", "[1*]CCCCNC(=O)[C@@H]1CS[C@H]2NC(=O)N[C@@H]12",
                    "streptavidin", "biotin-SA", True, True,
                    "Non-covalent, Kd~10^-15 M"),
]

SUPPORT_LIBRARY: List[SupportDef] = [
    SupportDef("DBCO-Fe3O4", "Fe3O4", "DBCO", 1.0, True,
               "Commercial magnetic beads with DBCO surface"),
    SupportDef("NH2-Fe3O4", "Fe3O4", "NH2", 1.0, True,
               "Amine-coated magnetic beads"),
    SupportDef("maleimide-Fe3O4", "Fe3O4", "maleimide", 1.0, True,
               "Maleimide-coated magnetic beads"),
    SupportDef("streptavidin-Fe3O4", "Fe3O4", "streptavidin", 1.0, True,
               "SA-coated magnetic beads for biotin conjugation"),
    SupportDef("DBCO-agarose", "agarose", "DBCO", 50.0, False,
               "Agarose resin with DBCO, for column pulldown"),
    SupportDef("NHS-glass", "glass", "NHS", 0.0, False,
               "Glass slide for microarray / SPR"),
    SupportDef("DBCO-polystyrene", "polystyrene", "DBCO", 10.0, False,
               "PS beads with DBCO for flow cytometry"),
]


# ---------------------------------------------------------------------------
# Construct specification
# ---------------------------------------------------------------------------

@dataclass
class ConstructSpec:
    """Specification for a modular construct."""
    recognition_smiles: str        # binder SMILES (must have one [*] for linker attachment)
    recognition_name: str = ""
    linker: str = "PEG4"           # name from LINKER_LIBRARY
    click_handle: str = "azide"    # name from CLICK_HANDLE_LIBRARY
    support: str = "DBCO-Fe3O4"    # name from SUPPORT_LIBRARY
    target: str = ""               # what the construct is designed to capture


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class AssembledConstruct:
    """Result of construct assembly."""
    spec: ConstructSpec
    # Assembled molecule (recognition + linker + click handle)
    soluble_smiles: str = ""       # the part in solution (before bead attachment)
    soluble_valid: bool = False
    # Properties
    molecular_weight: float = 0.0
    logP: float = 0.0
    solubility_class: str = ""
    # Synthesis route
    n_reactions: int = 0
    reactions: List[str] = field(default_factory=list)
    synthesis_coverage: float = 0.0
    # Click compatibility
    click_compatible: bool = False
    click_reaction: str = ""
    # Support info
    support_name: str = ""
    support_magnetic: bool = False
    # Readout options
    readout_options: List[str] = field(default_factory=list)
    # Errors
    errors: List[str] = field(default_factory=list)


def _get_linker(name: str) -> Optional[LinkerDef]:
    for l in LINKER_LIBRARY:
        if l.name == name:
            return l
    return None


def _get_click(name: str) -> Optional[ClickHandleDef]:
    for c in CLICK_HANDLE_LIBRARY:
        if c.name == name:
            return c
    return None


def _get_support(name: str) -> Optional[SupportDef]:
    for s in SUPPORT_LIBRARY:
        if s.name == name:
            return s
    return None


def _join_smiles(smi_a: str, smi_b: str) -> Optional[str]:
    """Join two SMILES at their [*] dummy atoms."""
    if not HAS_RDKIT:
        return None

    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None

    # Find dummy atoms
    def _find_dummy(mol, isotope=0):
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                if isotope == 0 or a.GetIsotope() == isotope:
                    return a.GetIdx()
        return None

    # For joining: use [2*] on A with [1*] on B
    # If no isotope labels, use any dummy
    dummy_a = _find_dummy(mol_a, 2)
    if dummy_a is None:
        dummy_a = _find_dummy(mol_a, 0)
    if dummy_a is None:
        # Try any dummy
        dummy_a = _find_dummy(mol_a)

    dummy_b = _find_dummy(mol_b, 1)
    if dummy_b is None:
        dummy_b = _find_dummy(mol_b, 0)
    if dummy_b is None:
        dummy_b = _find_dummy(mol_b)

    if dummy_a is None or dummy_b is None:
        return None

    # Get neighbors
    nbr_a = None
    for n in mol_a.GetAtomWithIdx(dummy_a).GetNeighbors():
        nbr_a = n.GetIdx()
        break
    nbr_b = None
    for n in mol_b.GetAtomWithIdx(dummy_b).GetNeighbors():
        nbr_b = n.GetIdx()
        break

    if nbr_a is None or nbr_b is None:
        return None

    # Combine
    combo = Chem.RWMol(Chem.CombineMols(mol_a, mol_b))
    nbr_b_shifted = nbr_b + mol_a.GetNumAtoms()
    dummy_b_shifted = dummy_b + mol_a.GetNumAtoms()

    combo.AddBond(nbr_a, nbr_b_shifted, Chem.BondType.SINGLE)

    # Remove dummies (highest index first)
    to_remove = sorted([dummy_a, dummy_b_shifted], reverse=True)
    for idx in to_remove:
        combo.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(combo)
        return Chem.MolToSmiles(combo)
    except Exception:
        return None


def assemble_construct(spec: ConstructSpec) -> AssembledConstruct:
    """
    Assemble a modular construct from specification.

    Joins: recognition + linker + click handle
    Then validates properties and synthesis route.
    The support (bead) is not part of the SMILES — it's metadata.
    """
    result = AssembledConstruct(spec=spec)

    linker = _get_linker(spec.linker)
    click = _get_click(spec.click_handle)
    support = _get_support(spec.support)

    if linker is None:
        result.errors.append(f"Unknown linker: {spec.linker}")
        return result
    if click is None:
        result.errors.append(f"Unknown click handle: {spec.click_handle}")
        return result
    if support is None:
        result.errors.append(f"Unknown support: {spec.support}")
        return result

    # Step 1: Join recognition + linker
    rec_linker = _join_smiles(spec.recognition_smiles, linker.smiles)
    if rec_linker is None:
        # Try with [1*] label on recognition
        rec_smi = spec.recognition_smiles
        if "*" not in rec_smi:
            result.errors.append("Recognition SMILES has no attachment point [*]")
            return result
        result.errors.append("Failed to join recognition + linker")
        return result

    # Step 2: Join (recognition+linker) + click handle
    soluble = _join_smiles(rec_linker, click.smiles)
    if soluble is None:
        # Linker output might not have a dummy; try direct join
        soluble = rec_linker  # fallback: just rec+linker without click
        result.errors.append("Click handle attachment failed, using rec+linker only")

    result.soluble_smiles = soluble or ""

    # Validate
    if soluble:
        mol = Chem.MolFromSmiles(soluble)
        result.soluble_valid = mol is not None

    # Step 3: Properties
    if result.soluble_valid:
        from core.property_predictor import predict_properties
        props = predict_properties(result.soluble_smiles)
        result.molecular_weight = props.molecular_weight
        result.logP = props.logP
        result.solubility_class = props.solubility_class

    # Step 4: Synthesis route
    if result.soluble_valid:
        from core.reaction_assembler import validate_synthesis
        synth = validate_synthesis(result.soluble_smiles)
        result.n_reactions = len(synth.reactions_used)
        result.reactions = synth.reactions_used
        result.synthesis_coverage = synth.annotation_coverage

    # Step 5: Click compatibility
    result.click_compatible = (click.partner == support.functional_group)
    result.click_reaction = click.reaction if result.click_compatible else "INCOMPATIBLE"
    result.support_name = support.name
    result.support_magnetic = support.magnetic

    # Step 6: Readout options
    readouts = []
    if support.magnetic:
        readouts.extend(["magnetic_pulldown", "magnetic_separation"])
    readouts.append("LC-MS")
    readouts.append("fluorescence")
    if support.material == "glass":
        readouts.append("SPR")
        readouts.append("microarray")
    if support.material == "polystyrene":
        readouts.append("flow_cytometry")
    result.readout_options = readouts

    return result


# ---------------------------------------------------------------------------
# Utility: list compatible click+support pairs
# ---------------------------------------------------------------------------

def compatible_pairs() -> List[Tuple[str, str, str]]:
    """Return all compatible (click_handle, support, reaction) triples."""
    pairs = []
    for click in CLICK_HANDLE_LIBRARY:
        for support in SUPPORT_LIBRARY:
            if click.partner == support.functional_group:
                pairs.append((click.name, support.name, click.reaction))
    return pairs


def suggest_construct(
    recognition_smiles: str,
    target: str = "",
    prefer_magnetic: bool = True,
    prefer_catalyst_free: bool = True,
) -> ConstructSpec:
    """
    Suggest a construct specification given a recognition element.

    Picks linker, click, and support based on preferences.
    """
    # Default linker: PEG4 (good balance of length and solubility)
    linker = "PEG4"

    # Pick click handle
    if prefer_catalyst_free:
        click = "azide"  # SPAAC with DBCO
    else:
        click = "terminal-alkyne"  # CuAAC

    # Pick support
    if prefer_magnetic:
        if click == "azide":
            support = "DBCO-Fe3O4"
        elif click == "thiol":
            support = "maleimide-Fe3O4"
        else:
            support = "NH2-Fe3O4"
    else:
        support = "DBCO-agarose"

    return ConstructSpec(
        recognition_smiles=recognition_smiles,
        recognition_name=target,
        linker=linker,
        click_handle=click,
        support=support,
        target=target,
    )
