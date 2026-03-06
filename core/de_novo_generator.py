"""
core/de_novo_generator.py — De Novo Molecule Generation for MABE

Physics-first combinatorial enumeration of novel binders.

Strategy:
  1. BACKBONE_LIBRARY: skeletal scaffolds with labeled attachment points ([*])
  2. ARM_LIBRARY: donor-group fragments with attachment point ([*])
  3. Enumerate: attach arms to backbones via RDKit reaction SMARTS
  4. Filter: RDKit validity, duplicate removal, size/property gates
  5. Score: feed through auto_descriptor → unified_scorer_v2
  6. Rank: composite of predicted log Ka, synthetic accessibility, novelty

Entry points:
  generate_candidates(target_metal, ...) → GenerationResult
  generate_and_screen(target_metal, interferents, ...) → GenerationResult
  generate_for_host(host_key, ...) → GenerationResult

No ML. No fitted parameters against target data. Pure combinatorial
chemistry + physics scoring.
"""

import sys
import os
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from itertools import product as cartesian_product

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from core.design_engine_v2 import (
    score_one, ScoredCandidate, RankingResult, _grade
)


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC ACCESSIBILITY SCORE (SA_Score)
# Ertl & Schuffenhauer, J. Cheminf. 1:8 (2009) — simplified version
# Uses fragment complexity + ring penalty + stereo penalty + size penalty
# Lower = easier to synthesize. Range ~1 (trivial) to ~10 (very hard).
# ═══════════════════════════════════════════════════════════════════════════

def sa_score(mol):
    """Compute synthetic accessibility score (1-10, lower=easier).

    Simplified Ertl SA_Score using RDKit descriptors.
    Not the full fragment-contribution model (requires pre-built
    fragment database), but captures the main drivers:
    ring complexity, stereocenter count, sp3 fraction, molecular size.
    """
    if mol is None:
        return 10.0

    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms == 0:
        return 10.0

    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_rotatable = Descriptors.NumRotatableBonds(mol)

    # Macrocycle penalty: rings > 8 atoms are hard to close
    macro_penalty = 0.0
    for ring in ring_info.AtomRings():
        if len(ring) > 8:
            macro_penalty += 1.5

    # Size penalty (quadratic above 40 heavy atoms)
    size_penalty = max(0, (n_atoms - 40) * 0.05) if n_atoms > 40 else 0.0

    # Complexity score
    complexity = (
        1.0                                    # baseline
        + n_rings * 0.3                        # each ring adds difficulty
        + n_spiro * 1.0                        # spiro centers are hard
        + n_bridgehead * 1.0                   # bridgeheads are hard
        + n_stereo * 0.5                       # stereocenters need control
        + macro_penalty
        + size_penalty
        + max(0, n_rotatable - 10) * 0.1       # very flexible = harder to purify
    )

    # Clamp to 1-10
    return max(1.0, min(10.0, complexity))


def sa_score_smiles(smiles):
    """SA score from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return sa_score(mol)


# ═══════════════════════════════════════════════════════════════════════════
# FRAGMENT LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════
#
# Backbones: molecular skeletons with dummy atom attachment points [*]
# Arms: donor-containing fragments with one dummy atom [*]
#
# Assembly: replace [*] on backbone with [*] on arm → form bond
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Backbone:
    """A skeletal scaffold with attachment points."""
    name: str
    smiles: str           # SMILES with [*] or [1*],[2*],... attachment points
    n_sites: int          # Number of attachment points
    category: str         # "linear", "branched", "cyclic", "macrocyclic"
    notes: str = ""


@dataclass
class Arm:
    """A donor-group fragment with one attachment point."""
    name: str
    smiles: str           # SMILES with [*] attachment point
    donor_subtypes: list  # What donor types this arm provides
    donor_element: str    # Primary donor element (O, N, S, P)
    hardness: str         # "hard", "borderline", "soft"
    category: str = ""


# ── BACKBONES ─────────────────────────────────────────────────────────────
# [*] marks where arms attach. Multi-site backbones use [1*], [2*], etc.
# These are all real, synthetically accessible scaffolds.

BACKBONE_LIBRARY = [
    # ── LINEAR DIAMINES (2 sites) ────────────────────────────────────────
    Backbone("ethylenediamine", "[1*]NCC[2*]", 2, "linear",
             "5-membered chelate rings, workhorse"),
    Backbone("propylenediamine", "[1*]NCCC[2*]", 2, "linear",
             "6-membered chelate rings"),
    Backbone("1,2-diaminocyclohexane", "[1*]N[C@@H]1CCCC[C@H]1[2*]", 2, "linear",
             "Rigid 5-membered ring, chiral"),
    Backbone("dithioether-linear", "[1*]CSCCSC[2*]", 2, "linear",
             "S-donor podand, soft-metal selective"),

    # ── TRIAMINES / BRANCHED (3 sites) ───────────────────────────────────
    Backbone("diethylenetriamine", "[1*]NCC([3*])NCC[2*]", 3, "branched",
             "Central N bears arm, dien scaffold"),
    Backbone("tris(2-aminoethyl)amine", "[1*]NCC([2*])NCC[3*]", 3, "branched",
             "Tripodal N4, tren scaffold"),
    Backbone("tris-amine", "[1*]N(CC[2*])CC[3*]", 3, "branched",
             "Generic tripodal amine, NTA-like core"),

    # ── AROMATIC (2-3 sites) ─────────────────────────────────────────────
    Backbone("2,6-disubstituted-pyridine", "[1*]c1cccc([2*])n1", 2, "aromatic",
             "Pyridine core provides N_pyridine + 2 arm sites"),
    Backbone("1,2-disubstituted-benzene", "[1*]c1ccccc1[2*]", 2, "aromatic",
             "ortho-substituted ring, close approach"),
    Backbone("1,3,5-trisubstituted-benzene", "[1*]c1cc([2*])cc([3*])c1", 3, "aromatic",
             "Mesitylene-type tripodal"),
    Backbone("4,4'-bipy-disubstituted", "[1*]c1ccnc(-c2cc([2*])ccn2)c1", 2, "aromatic",
             "4,4'-bipyridine with arm sites, two N_pyridine built in"),
    Backbone("biphenol", "Oc1ccc(-c2ccc(O)c([2*])c2)cc1[1*]", 2, "aromatic",
             "Biphenol scaffold, phenolate O donors + 2 arm sites"),
    Backbone("2,6-pyridinedicarboxamide", "[1*]NC(=O)c1cccc(C(=O)N[2*])n1", 2, "aromatic",
             "Pyridine-bis(amide), tridentate N3 core"),

    # ── SALEN / SCHIFF BASE (2 sites) ───────────────────────────────────
    Backbone("salen-en", "[1*]N=Cc1ccc(O)cc1Oc1cc(C=N[2*])ccc1O", 2, "aromatic",
             "Salen: 2×(imine+phenolate), N2O2 tetradentate, workhorse"),
    Backbone("salen-en-v2", "[1*]/N=C/c1cc(O)ccc1-c1ccc(O)cc1/C=N/[2*]", 2, "aromatic",
             "Salen variant: biphenyl-linked salicylaldimine"),

    # ── ETHER BACKBONES ──────────────────────────────────────────────────
    Backbone("diethylene-glycol", "[1*]COCCOCC[2*]", 2, "linear",
             "Ether oxygens + 2 arm sites, podand-like"),
    Backbone("triethylene-glycol", "[1*]COCCOCCOC[2*]", 2, "linear",
             "3 ether oxygens + 2 arm sites"),

    # ── SINGLE SITE ──────────────────────────────────────────────────────
    Backbone("methyl", "[1*]C", 1, "linear",
             "Simplest linker, monodentate arm"),
    Backbone("ethyl", "[1*]CC", 1, "linear",
             "Short spacer"),
    Backbone("nitrilotriacetic-core", "[1*]N(CC(=O)O)CC(=O)O", 1, "linear",
             "NTA backbone: 1 arm site + built-in N(CH2COOH)2"),

    # ── AMINO ACID / PEPTOID BACKBONES ───────────────────────────────────
    Backbone("glycine-backbone", "[1*]NCC(=O)O", 1, "linear",
             "Glycinate with arm at N"),
    Backbone("beta-alanine-backbone", "[1*]NCCC(=O)O", 1, "linear",
             "β-amino acid, 6-membered chelate possible"),
    Backbone("diglycine", "[1*]NCC(=O)NCC(=O)[2*]", 2, "linear",
             "Peptoid backbone, 2 arm sites, amide O donors"),

    # ── MACROCYCLIC CORES ────────────────────────────────────────────────
    Backbone("cyclam-core", "[1*]N1CCNCCNCC[2*]NCC1", 2, "macrocyclic",
             "14-membered N4 macrocycle, 2 functionalizable N"),
    Backbone("cyclen-core", "[1*]N1CCN([2*])CCN([3*])CC1", 3, "macrocyclic",
             "12-membered N4, three N bear arms (DOTA-like)"),
    Backbone("tacn", "C1CN([1*])CCN([2*])CCN1[3*]", 3, "macrocyclic",
             "1,4,7-triazacyclononane, 9-membered N3, NOTA-like"),
    Backbone("dioxocyclam", "[1*]N1CCOC(=O)CN([2*])CCOC(=O)C1", 2, "macrocyclic",
             "Dioxo-cyclam, N2O2 mixed macrocycle"),
    Backbone("9aneS3-core", "[1*]SCCSCC[2*]SCC", 2, "macrocyclic",
             "9-ane-S3 thioether macrocycle, soft metal selective"),
]


# ── DONOR ARMS ────────────────────────────────────────────────────────────
# Each arm connects via the [*] to a backbone site.
# Provides one or more donor atoms to the metal.

ARM_LIBRARY = [
    # ── CARBOXYLATE (hard O) ─────────────────────────────────────────────
    Arm("acetic-acid", "[*]CC(=O)O", ["O_carboxylate"], "O", "hard",
        "aminocarboxylate"),
    Arm("propionic-acid", "[*]CCC(=O)O", ["O_carboxylate"], "O", "hard",
        "aminocarboxylate"),

    # ── HYDROXYL (hard O) ────────────────────────────────────────────────
    Arm("ethanol", "[*]CCO", ["O_hydroxyl"], "O", "hard", "alcohol"),
    Arm("phenol", "[*]c1ccc(O)cc1", ["O_phenolate"], "O", "hard", "phenol"),

    # ── HYDROXAMATE (hard O, bidentate) ──────────────────────────────────
    Arm("acetohydroxamate", "[*]CC(=O)NO", ["O_hydroxamate", "O_hydroxamate"],
        "O", "hard", "hydroxamate"),

    # ── CATECHOL (hard O, bidentate) ─────────────────────────────────────
    Arm("catechol", "[*]c1ccc(O)c(O)c1", ["O_catecholate", "O_catecholate"],
        "O", "hard", "catechol"),

    # ── PHOSPHONATE (hard O) ─────────────────────────────────────────────
    Arm("phosphonate", "[*]CP(=O)(O)O", ["O_phosphate"], "O", "hard",
        "phosphonate"),
    Arm("bisphosphonate", "[*]C(P(=O)(O)O)P(=O)(O)O",
        ["O_phosphate", "O_phosphate"], "O", "hard",
        "bisphosphonate"),

    # ── HYDROXYPYRIDINONE (hard O, bidentate, Fe3+-selective) ────────────
    Arm("hydroxypyridinone", "[*]c1cc(=O)c(O)cn1C",
        ["O_hydroxamate", "O_hydroxamate"], "O", "hard",
        "pyridinone"),

    # ── 8-HYDROXYQUINOLINE (hard/borderline, bidentate N,O) ─────────────
    Arm("8-hydroxyquinolinyl", "[*]c1ccc2cccc(O)c2n1",
        ["N_pyridine", "O_phenolate"], "N", "borderline",
        "quinoline"),

    # ── SQUARAMIDE (hard O) ──────────────────────────────────────────────
    Arm("squaramide", "[*]NC1=C(O)C(=O)C1=O",
        ["O_hydroxyl", "O_hydroxyl"], "O", "hard",
        "squaramide"),

    # ── PYRIDINE (borderline N) ──────────────────────────────────────────
    Arm("2-pyridylmethyl", "[*]Cc1ccccn1", ["N_pyridine"], "N", "borderline",
        "pyridine"),
    Arm("2-pyridyl", "[*]c1ccccn1", ["N_pyridine"], "N", "borderline",
        "pyridine"),
    Arm("picolylamine", "[*]NCc1ccccn1", ["N_amine", "N_pyridine"],
        "N", "borderline", "pyridine"),

    # ── IMIDAZOLE (borderline N) ─────────────────────────────────────────
    Arm("imidazolylmethyl", "[*]Cc1cnc[nH]1", ["N_imidazole"], "N", "borderline",
        "imidazole"),
    Arm("benzimidazolyl", "[*]c1nc2ccccc2[nH]1", ["N_imidazole"], "N",
        "borderline", "benzimidazole"),

    # ── AMINE (borderline N) ─────────────────────────────────────────────
    Arm("aminomethyl", "[*]CN", ["N_amine"], "N", "borderline", "amine"),
    Arm("aminoethyl", "[*]CCN", ["N_amine"], "N", "borderline", "amine"),

    # ── IMINE / SCHIFF BASE (borderline N) ───────────────────────────────
    Arm("salicylaldimine", "[*]N=Cc1ccccc1O",
        ["N_imine", "O_phenolate"], "N", "borderline", "salen"),

    # ── OXIME (borderline N,O) ───────────────────────────────────────────
    Arm("oxime-simple", "[*]C=NO", ["N_imine", "O_hydroxyl"], "N",
        "borderline", "oxime"),
    Arm("amidoxime", "[*]C(=NO)N", ["N_imine", "O_hydroxyl", "N_amine"],
        "N", "borderline", "amidoxime"),

    # ── HYDRAZIDE (borderline N,O) ───────────────────────────────────────
    Arm("hydrazide", "[*]C(=O)NN", ["O_carboxylate", "N_amine"], "N",
        "borderline", "hydrazide"),

    # ── NITRILE (borderline N) ───────────────────────────────────────────
    Arm("nitrile", "[*]CC#N", ["N_nitrile"], "N", "borderline", "nitrile"),

    # ── CARBAMOYL / AMIDE (borderline N) ─────────────────────────────────
    Arm("carbamoylmethyl", "[*]CC(=O)N", ["N_amide"], "N", "borderline",
        "amide"),

    # ── THIOL (soft S) ───────────────────────────────────────────────────
    Arm("thiol-methyl", "[*]CS", ["S_thiolate"], "S", "soft", "thiol"),
    Arm("thiol-ethyl", "[*]CCS", ["S_thiolate"], "S", "soft", "thiol"),

    # ── THIOETHER (soft S) ───────────────────────────────────────────────
    Arm("thioether-methyl", "[*]CSC", ["S_thioether"], "S", "soft", "thioether"),

    # ── DITHIOCARBAMATE (soft S, bidentate) ──────────────────────────────
    Arm("dithiocarbamate-NMe", "[*]N(C)C(=S)S",
        ["S_thiolate", "S_thiolate"], "S", "soft",
        "dithiocarbamate"),
    Arm("dithiocarbamate-NH", "[*]NC(=S)S",
        ["S_thiolate", "S_thiolate"], "S", "soft",
        "dithiocarbamate"),

    # ── THIOSEMICARBAZONE (soft S, borderline N) ─────────────────────────
    Arm("thiosemicarbazone", "[*]C=NNC(=S)N",
        ["N_imine", "S_thiolate"], "S", "soft",
        "thiosemicarbazone"),

    # ── THIOUREA (soft S) ────────────────────────────────────────────────
    Arm("thiourea", "[*]NC(=S)N", ["S_thiolate", "N_amine"], "S", "soft",
        "thiourea"),

    # ── AMINOTHIADIAZOLE (soft S, borderline N) ──────────────────────────
    Arm("aminothiadiazole", "[*]c1nnc(N)s1",
        ["N_pyridine", "S_thioether"], "S", "soft",
        "thiadiazole"),

    # ── PHOSPHINE (soft P) ───────────────────────────────────────────────
    Arm("diphenylphosphino", "[*]CP(c1ccccc1)c1ccccc1", ["P_phosphine"],
        "P", "soft", "phosphine"),

    # ── SULFONATE (hard O) ───────────────────────────────────────────────
    Arm("sulfonate", "[*]CS(=O)(=O)O", ["O_sulfonate"], "O", "hard",
        "sulfonate"),

    # ── HYDROGEN CAP ─────────────────────────────────────────────────────
    Arm("H-cap", "[*][H]", [], "", "", "cap"),
]


# ═══════════════════════════════════════════════════════════════════════════
# BIOISOSTERE TABLE
# ═══════════════════════════════════════════════════════════════════════════
#
# Arms grouped by equivalent donor function. Arms within a group can
# substitute for each other — same primary coordination mode, different
# scaffold chemistry. Grouped by primary donor signature.
#
# Each group: (group_name, [arm_name, ...])
# ═══════════════════════════════════════════════════════════════════════════

BIOISOSTERE_GROUPS = [
    # ── Monodentate O-acid (hard, single O_carboxylate/phosphate/sulfonate) ──
    ("mono_O_acid", [
        "acetic-acid", "propionic-acid", "phosphonate", "sulfonate",
    ]),

    # ── Bidentate O,O chelate (hard, catechol/hydroxamate equivalents) ──
    ("bident_OO_chelate", [
        "catechol", "acetohydroxamate", "hydroxypyridinone",
        "bisphosphonate", "squaramide",
    ]),

    # ── Monodentate N aromatic (borderline, pyridine/imidazole type) ──
    ("mono_N_aromatic", [
        "2-pyridyl", "2-pyridylmethyl", "imidazolylmethyl",
        "benzimidazolyl",
    ]),

    # ── Monodentate N amine (borderline, aliphatic amine) ──
    ("mono_N_amine", [
        "aminomethyl", "aminoethyl",
    ]),

    # ── Bidentate N,O (borderline, mixed imine/phenolate) ──
    ("bident_NO_mixed", [
        "salicylaldimine", "8-hydroxyquinolinyl", "oxime-simple",
        "hydrazide", "picolylamine",
    ]),

    # ── Monodentate S (soft, thiol/thioether) ──
    ("mono_S", [
        "thiol-methyl", "thiol-ethyl", "thioether-methyl",
    ]),

    # ── Bidentate S,S (soft, dithio chelate) ──
    ("bident_SS_chelate", [
        "dithiocarbamate-NMe", "dithiocarbamate-NH",
    ]),

    # ── Bidentate N,S (soft/borderline, thiosemicarbazone type) ──
    ("bident_NS_mixed", [
        "thiosemicarbazone", "thiourea", "aminothiadiazole",
    ]),

    # ── Monodentate O hydroxyl/phenol ──
    ("mono_O_hydroxyl", [
        "ethanol", "phenol",
    ]),

    # ── Weak/special ──
    ("mono_N_weak", [
        "nitrile", "carbamoylmethyl",
    ]),
]

# Build lookup: arm_name → group_name
_ARM_TO_GROUP = {}
_GROUP_TO_ARMS = {}
for _gname, _members in BIOISOSTERE_GROUPS:
    _GROUP_TO_ARMS[_gname] = _members
    for _mname in _members:
        _ARM_TO_GROUP[_mname] = _gname

# Build lookup: arm_name → Arm object
_ARM_BY_NAME = {a.name: a for a in ARM_LIBRARY}


def get_bioisosteres(arm_name):
    """Return list of Arm objects that are bioisosteric replacements.

    Excludes the input arm itself.
    """
    group = _ARM_TO_GROUP.get(arm_name)
    if not group:
        return []
    return [_ARM_BY_NAME[n] for n in _GROUP_TO_ARMS[group]
            if n != arm_name and n in _ARM_BY_NAME]


# ═══════════════════════════════════════════════════════════════════════════
# DONOR SIGNATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _donor_signature(smiles, metal=None):
    """Extract sorted donor subtype tuple from a SMILES via auto_descriptor.

    Returns (tuple_of_sorted_subtypes, denticity) or (None, 0) on failure.
    """
    try:
        from core.auto_descriptor import from_smiles
        uc = from_smiles(smiles, metal=metal)
        if uc and uc.donor_subtypes:
            sig = tuple(sorted(uc.donor_subtypes))
            return sig, uc.denticity
    except Exception:
        pass
    return None, 0


def _arm_donor_signature(arms):
    """Get combined sorted donor subtype tuple from a list of Arm objects."""
    all_donors = []
    for arm in arms:
        all_donors.extend(arm.donor_subtypes)
    return tuple(sorted(all_donors))


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD HOPPING
# ═══════════════════════════════════════════════════════════════════════════

def scaffold_hop(smiles, metal=None, host=None, pH=7.4,
                 max_candidates=100, max_scored=30):
    """Keep the donor set, swap the backbone.

    Takes a known molecule, extracts its donor signature, then finds
    backbone + arm combinations that reproduce the same donor set on
    a different scaffold.

    Args:
        smiles: input molecule SMILES
        metal: target metal for scoring (optional)
        host: target host for scoring (optional)
        pH: working pH
        max_candidates: max to enumerate
        max_scored: max to score

    Returns:
        list of (smiles, backbone_name, arm_names, sa_score) tuples
    """
    target_sig, target_dent = _donor_signature(smiles, metal=metal)
    if target_sig is None:
        return []

    # Find all backbone + arm combos that match the donor signature
    from itertools import product as cartesian_product

    results = []
    seen = set()
    input_canonical = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            input_canonical = Chem.MolToSmiles(mol)
    except Exception:
        pass

    # Filter arms by HSAB if metal specified
    if metal:
        compatible_arms = [a for a in ARM_LIBRARY
                           if hsab_compatible(metal, a) and a.name != "H-cap"]
    else:
        compatible_arms = [a for a in ARM_LIBRARY if a.name != "H-cap"]

    for bb in BACKBONE_LIBRARY:
        arm_combos = cartesian_product(compatible_arms, repeat=bb.n_sites)
        for arm_tuple in arm_combos:
            if len(results) >= max_candidates:
                break

            # Check if this arm combo reproduces the target donor signature
            combo_sig = _arm_donor_signature(list(arm_tuple))
            if combo_sig != target_sig:
                continue

            out_smi, mol = assemble(bb, list(arm_tuple))
            if out_smi is None:
                continue

            # Skip if it's the same molecule
            if out_smi == input_canonical:
                continue

            if out_smi in seen:
                continue
            seen.add(out_smi)

            if not passes_filter(out_smi, mol):
                continue

            sa = sa_score(mol)
            arm_names = [a.name for a in arm_tuple]
            results.append((out_smi, bb.name, arm_names, sa))

        if len(results) >= max_candidates:
            break

    return results[:max_scored]


# ═══════════════════════════════════════════════════════════════════════════
# BIOISOSTERIC REPLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

def bioisosteric_replace(smiles, metal=None, host=None, pH=7.4,
                         max_candidates=100, max_scored=30):
    """Keep the backbone concept, swap arms with bioisosteric equivalents.

    Strategy: try to decompose the input molecule into a known backbone
    + arms pattern. For each arm position, substitute bioisosteric
    alternatives and re-assemble. Score all variants.

    Since exact decomposition of arbitrary SMILES into backbone+arms is
    hard, we use a simpler approach: extract the donor signature, then
    for each backbone, find arm combos where at least one arm differs
    from the "closest match" combo but stays within the same bioisostere
    group.

    Args:
        smiles: input molecule SMILES
        metal, host, pH: for scoring
        max_candidates, max_scored: limits

    Returns:
        list of (smiles, backbone_name, arm_names, sa_score) tuples
    """
    target_sig, target_dent = _donor_signature(smiles, metal=metal)
    if target_sig is None:
        return []

    from itertools import product as cartesian_product

    # For each donor subtype in the signature, find which bioisostere
    # group(s) contain arms providing that subtype
    def _arms_for_subtype(subtype):
        """Find arms whose donor_subtypes include this subtype."""
        matches = []
        for arm in ARM_LIBRARY:
            if arm.name == "H-cap":
                continue
            if subtype in arm.donor_subtypes:
                matches.append(arm)
        return matches

    # Build per-position arm options: for each donor in the target sig,
    # find arms that provide it AND their bioisosteric alternatives
    target_donors = list(target_sig)

    # Group consecutive identical donors (e.g., 4x O_carboxylate for EDTA)
    # and find arm combos that cover them
    results = []
    seen = set()
    input_canonical = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            input_canonical = Chem.MolToSmiles(mol)
    except Exception:
        pass

    # Strategy: enumerate backbone × arm combos where the arm combo's
    # donor signature has the same LENGTH as target and each donor is
    # from the same bioisostere group as the corresponding target donor
    def _donor_group(subtype):
        """Map a donor subtype to its bioisostere group."""
        for arm in ARM_LIBRARY:
            if subtype in arm.donor_subtypes:
                grp = _ARM_TO_GROUP.get(arm.name)
                if grp:
                    return grp
        return subtype  # fallback: use subtype as its own group

    target_groups = tuple(sorted(_donor_group(d) for d in target_donors))

    if metal:
        compatible_arms = [a for a in ARM_LIBRARY
                           if hsab_compatible(metal, a) and a.name != "H-cap"]
    else:
        compatible_arms = [a for a in ARM_LIBRARY if a.name != "H-cap"]

    for bb in BACKBONE_LIBRARY:
        arm_combos = cartesian_product(compatible_arms, repeat=bb.n_sites)
        for arm_tuple in arm_combos:
            if len(results) >= max_candidates:
                break

            # Check: same number of total donors
            combo_donors = []
            for arm in arm_tuple:
                combo_donors.extend(arm.donor_subtypes)
            if len(combo_donors) != len(target_donors):
                continue

            # Check: each donor maps to same bioisostere group
            combo_groups = tuple(sorted(_donor_group(d) for d in combo_donors))
            if combo_groups != target_groups:
                continue

            # Must differ from exact target signature (that's scaffold_hop)
            combo_sig = tuple(sorted(combo_donors))
            if combo_sig == target_sig:
                continue

            out_smi, mol = assemble(bb, list(arm_tuple))
            if out_smi is None or out_smi == input_canonical:
                continue

            if out_smi in seen:
                continue
            seen.add(out_smi)

            if not passes_filter(out_smi, mol):
                continue

            sa = sa_score(mol)
            arm_names = [a.name for a in arm_tuple]
            results.append((out_smi, bb.name, arm_names, sa))

        if len(results) >= max_candidates:
            break

    return results[:max_scored]


# ═══════════════════════════════════════════════════════════════════════════
# SCORE SCAFFOLD HOPS / BIOISOSTERIC REPLACEMENTS
# ═══════════════════════════════════════════════════════════════════════════

def hop_and_score(smiles, metal=None, host=None, pH=7.4,
                  mode="both", max_candidates=100, max_scored=20,
                  interferents=None, sa_penalty_weight=0.3):
    """Scaffold-hop and/or bioisosteric-replace, then score all variants.

    Args:
        smiles: seed molecule SMILES
        metal: target metal (optional)
        host: target host (optional)
        pH: working pH
        mode: "hop", "bioisostere", or "both"
        max_candidates: max to enumerate per strategy
        max_scored: max to score total
        interferents: optional list for selectivity screening
        sa_penalty_weight: SA penalty in composite

    Returns:
        GenerationResult with scored variants
    """
    t0 = time.time()
    raw = []

    if mode in ("hop", "both"):
        hops = scaffold_hop(smiles, metal=metal, host=host, pH=pH,
                            max_candidates=max_candidates,
                            max_scored=max_candidates)
        raw.extend(hops)

    if mode in ("bioisostere", "both"):
        bios = bioisosteric_replace(smiles, metal=metal, host=host, pH=pH,
                                     max_candidates=max_candidates,
                                     max_scored=max_candidates)
        raw.extend(bios)

    # Dedup
    seen = set()
    unique = []
    for item in raw:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)

    # Sort by SA, take top N for scoring
    unique.sort(key=lambda x: x[3])
    to_score = unique[:max_scored]

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smi, bb_name, arm_names, sa in to_score:
        try:
            sc = score_one(smi, metal=metal, host=host, pH=pH, name=smi[:40])
            gc = GeneratedCandidate(
                smiles=smi,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=sc.log_Ka_pred,
                dg_total_kj=sc.dg_total_kj,
                prediction=sc.prediction,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                novel=smi not in known,
            )

            # Selectivity if requested
            if interferents and metal:
                for intf in interferents:
                    try:
                        intf_sc = score_one(smi, metal=intf, pH=pH)
                        gc.interferent_scores[intf] = intf_sc.log_Ka_pred
                        gc.selectivity_gaps[intf] = (gc.log_Ka_pred
                                                      - intf_sc.log_Ka_pred)
                    except Exception:
                        gc.interferent_scores[intf] = 0.0
                        gc.selectivity_gaps[intf] = gc.log_Ka_pred

                if gc.selectivity_gaps:
                    gc.worst_interferent = min(gc.selectivity_gaps,
                                               key=gc.selectivity_gaps.get)
                    gc.min_gap = gc.selectivity_gaps[gc.worst_interferent]
                gc.grade = _grade(gc.min_gap)
                gc.composite_score = gc.min_gap - sa_penalty_weight * sa
            else:
                gc.composite_score = gc.log_Ka_pred - sa_penalty_weight * sa

            candidates.append(gc)
        except Exception as e:
            errors.append((smi, str(e)))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return GenerationResult(
        target=metal or host or "unknown",
        mode=f"scaffold_{mode}",
        interferents=list(interferents) if interferents else [],
        candidates=candidates,
        n_enumerated=len(raw),
        n_valid=len(unique),
        n_unique=len(unique),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _replace_dummy(mol, dummy_idx, arm_smiles):
    """Replace a dummy atom [*] in mol with an arm fragment.

    Returns a new molecule with the dummy replaced by the arm's
    non-dummy portion, bonded at the dummy's neighbor.
    """
    arm_mol = Chem.MolFromSmiles(arm_smiles)
    if arm_mol is None:
        return None

    # Find dummy atoms
    mol_dummy = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            iso = atom.GetIsotope()
            if iso == dummy_idx or (dummy_idx == 0 and iso == 0):
                mol_dummy = atom.GetIdx()
                break
    if mol_dummy is None:
        return None

    arm_dummy = None
    for atom in arm_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            arm_dummy = atom.GetIdx()
            break
    if arm_dummy is None:
        return None

    # Use RWMol combine + bond formation
    combo = Chem.RWMol(Chem.CombineMols(mol, arm_mol))

    # Get neighbors of dummies
    mol_neighbor = None
    for n in combo.GetAtomWithIdx(mol_dummy).GetNeighbors():
        mol_neighbor = n.GetIdx()
        break

    arm_dummy_shifted = arm_dummy + mol.GetNumAtoms()
    arm_neighbor = None
    for n in combo.GetAtomWithIdx(arm_dummy_shifted).GetNeighbors():
        arm_neighbor = n.GetIdx()
        break

    if mol_neighbor is None or arm_neighbor is None:
        return None

    # Add bond between neighbors
    combo.AddBond(mol_neighbor, arm_neighbor, Chem.BondType.SINGLE)

    # Remove dummy atoms (higher index first)
    to_remove = sorted([mol_dummy, arm_dummy_shifted], reverse=True)
    for idx in to_remove:
        combo.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(combo)
        return combo.GetMol()
    except Exception:
        return None


def assemble(backbone, arms):
    """Assemble a molecule from a backbone and a list of arms.

    Args:
        backbone: Backbone object
        arms: list[Arm], length must match backbone.n_sites

    Returns:
        (smiles, mol) or (None, None) on failure
    """
    if len(arms) != backbone.n_sites:
        return None, None

    mol = Chem.MolFromSmiles(backbone.smiles)
    if mol is None:
        return None, None

    # Replace dummies in order: [1*], [2*], [3*], ...
    # If backbone uses isotope labels, match them.
    for i, arm in enumerate(arms):
        isotope = i + 1  # [1*], [2*], [3*]
        mol = _replace_dummy(mol, isotope, arm.smiles)
        if mol is None:
            return None, None

    try:
        smiles = Chem.MolToSmiles(mol)
        # Re-parse for canonical form
        mol2 = Chem.MolFromSmiles(smiles)
        if mol2 is None:
            return None, None
        smiles = Chem.MolToSmiles(mol2)
        return smiles, mol2
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# PROPERTY FILTERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PropertyFilter:
    """Gates for generated molecules."""
    max_heavy_atoms: int = 60
    max_mw: float = 800.0
    min_mw: float = 80.0
    max_rotatable: int = 15
    max_sa_score: float = 7.0        # reject very hard-to-synthesize
    require_donors: bool = True       # must have at least 1 donor atom


def passes_filter(smiles, mol, pfilter=None):
    """Check if a molecule passes property filters."""
    if pfilter is None:
        pfilter = PropertyFilter()
    if mol is None:
        return False

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > pfilter.max_heavy_atoms:
        return False

    mw = Descriptors.MolWt(mol)
    if mw > pfilter.max_mw or mw < pfilter.min_mw:
        return False

    n_rot = Descriptors.NumRotatableBonds(mol)
    if n_rot > pfilter.max_rotatable:
        return False

    if pfilter.max_sa_score < 10.0:
        sa = sa_score(mol)
        if sa > pfilter.max_sa_score:
            return False

    if pfilter.require_donors:
        # Check for N, O, S, P donor atoms
        has_donor = False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in (7, 8, 15, 16):  # N, O, P, S
                has_donor = True
                break
        if not has_donor:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# HSAB PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════

# Metal → preferred donor hardness
_METAL_HARDNESS = {
    # Hard metals prefer hard donors (O, N-amine)
    "Li+": "hard", "Na+": "hard", "K+": "hard", "Mg2+": "hard",
    "Ca2+": "hard", "Sr2+": "hard", "Ba2+": "hard",
    "Al3+": "hard", "Fe3+": "hard", "Cr3+": "hard", "La3+": "hard",
    "Ce3+": "hard", "Gd3+": "hard", "Zr4+": "hard",
    # Borderline
    "Fe2+": "borderline", "Co2+": "borderline", "Ni2+": "borderline",
    "Cu2+": "borderline", "Zn2+": "borderline", "Mn2+": "borderline",
    "Pb2+": "borderline", "Cd2+": "borderline",
    # Soft metals prefer soft donors (S, P)
    "Cu+": "soft", "Ag+": "soft", "Au+": "soft", "Au3+": "soft",
    "Hg2+": "soft", "Pd2+": "soft", "Pt2+": "soft",
}

# Compatibility: does this arm's hardness match the metal?
# borderline metals accept everything; hard/soft accept matching + borderline
_COMPAT = {
    ("hard", "hard"): True,
    ("hard", "borderline"): True,
    ("hard", "soft"): False,
    ("borderline", "hard"): True,
    ("borderline", "borderline"): True,
    ("borderline", "soft"): True,
    ("soft", "hard"): False,
    ("soft", "borderline"): True,
    ("soft", "soft"): True,
}


def hsab_compatible(metal, arm):
    """Check if an arm's donor hardness is compatible with a metal."""
    metal_h = _METAL_HARDNESS.get(metal, "borderline")
    arm_h = arm.hardness
    if arm_h == "" or arm_h == "cap":
        return True  # H-cap is always compatible
    return _COMPAT.get((metal_h, arm_h), True)


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedCandidate(ScoredCandidate):
    """Extends ScoredCandidate with generation metadata."""
    backbone_name: str = ""
    arm_names: list = field(default_factory=list)
    sa_score_val: float = 10.0
    composite_score: float = 0.0      # log_Ka - SA_penalty
    novel: bool = True                # Not in existing ligand library


@dataclass
class GenerationResult:
    """Output of a de novo generation run."""
    target: str
    mode: str                              # "metal", "host_guest", "selectivity"
    interferents: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    n_enumerated: int = 0                  # Total assembled
    n_valid: int = 0                       # Passed filters
    n_unique: int = 0                      # After dedup
    n_scored: int = 0                      # Successfully scored
    n_failed: int = 0
    elapsed_s: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def best(self):
        return self.candidates[0] if self.candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# NOVELTY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def _known_smiles_set():
    """Get canonical SMILES of all molecules in LIGAND_DB."""
    try:
        from knowledge.ligand_library import LIGAND_DB
        known = set()
        for lig in LIGAND_DB:
            mol = Chem.MolFromSmiles(lig.smiles)
            if mol:
                known.add(Chem.MolToSmiles(mol))
        return known
    except ImportError:
        return set()


# ═══════════════════════════════════════════════════════════════════════════
# CORE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def enumerate_molecules(metal=None, host=None, max_candidates=500,
                        backbones=None, arms=None,
                        pfilter=None, hsab_filter=True):
    """Enumerate molecules by combinatorial backbone × arm assembly.

    Args:
        metal: target metal (for HSAB filtering), e.g. "Cu2+"
        host: target host (for host-guest mode), e.g. "beta-CD"
        max_candidates: stop after this many valid unique molecules
        backbones: override backbone library (default: BACKBONE_LIBRARY)
        arms: override arm library (default: ARM_LIBRARY)
        pfilter: PropertyFilter instance
        hsab_filter: apply HSAB compatibility filter

    Returns:
        list of (smiles, backbone_name, arm_names, sa_score) tuples
    """
    if backbones is None:
        backbones = BACKBONE_LIBRARY
    if arms is None:
        arms = ARM_LIBRARY
    if pfilter is None:
        pfilter = PropertyFilter()

    # For host-guest mode, relax donor requirement
    if host and not metal:
        pfilter.require_donors = False

    # Pre-filter arms by HSAB if metal specified
    if metal and hsab_filter:
        compatible_arms = [a for a in arms if hsab_compatible(metal, a)]
    else:
        compatible_arms = list(arms)

    if not compatible_arms:
        compatible_arms = list(arms)  # fallback: use all

    seen = set()
    results = []

    for bb in backbones:
        # Generate all arm combinations for this backbone
        arm_combos = cartesian_product(compatible_arms, repeat=bb.n_sites)

        for arm_tuple in arm_combos:
            if len(results) >= max_candidates:
                return results

            smiles, mol = assemble(bb, list(arm_tuple))
            if smiles is None:
                continue

            # Dedup by canonical SMILES
            if smiles in seen:
                continue
            seen.add(smiles)

            # Property filter
            if not passes_filter(smiles, mol, pfilter):
                continue

            sa = sa_score(mol)
            arm_names = [a.name for a in arm_tuple]
            results.append((smiles, bb.name, arm_names, sa))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API: GENERATE + SCORE
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidates(target_metal, pH=7.4, max_candidates=200,
                        max_scored=50, pfilter=None, hsab_filter=True,
                        sa_penalty_weight=0.3):
    """Generate and score novel chelators for a target metal.

    Args:
        target_metal: e.g. "Cu2+", "Pb2+", "Fe3+"
        pH: working pH
        max_candidates: max molecules to enumerate
        max_scored: max to attempt scoring (top by SA)
        pfilter: PropertyFilter
        hsab_filter: apply HSAB pre-filter
        sa_penalty_weight: weight of SA penalty in composite score

    Returns:
        GenerationResult with scored, ranked candidates
    """
    t0 = time.time()

    # Step 1: Enumerate
    raw = enumerate_molecules(
        metal=target_metal, max_candidates=max_candidates,
        pfilter=pfilter, hsab_filter=hsab_filter
    )
    n_enumerated = len(raw)

    # Step 2: Sort by SA score (easiest first), take top N for scoring
    raw.sort(key=lambda x: x[3])  # sort by sa_score ascending
    to_score = raw[:max_scored]

    # Step 3: Score each through unified scorer
    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa in to_score:
        try:
            sc = score_one(smiles, metal=target_metal, pH=pH, name=smiles[:40])
            gc = GeneratedCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=sc.log_Ka_pred,
                dg_total_kj=sc.dg_total_kj,
                prediction=sc.prediction,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                composite_score=sc.log_Ka_pred - sa_penalty_weight * sa,
                novel=smiles not in known,
            )
            candidates.append(gc)
        except Exception as e:
            errors.append((smiles, str(e)))

    # Step 4: Rank by composite score
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return GenerationResult(
        target=target_metal,
        mode="metal",
        candidates=candidates,
        n_enumerated=n_enumerated,
        n_valid=n_enumerated,
        n_unique=n_enumerated,
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


def generate_and_screen(target_metal, interferents, pH=7.4,
                        max_candidates=200, max_scored=50,
                        pfilter=None, hsab_filter=True,
                        sa_penalty_weight=0.3):
    """Generate novel chelators and screen for selectivity.

    Same as generate_candidates but also scores each candidate
    against interferent metals and ranks by selectivity gap.

    Args:
        target_metal: e.g. "Pb2+"
        interferents: e.g. ["Ca2+", "Mg2+", "Fe3+"]
        pH, max_candidates, max_scored, pfilter, hsab_filter: as above
        sa_penalty_weight: weight of SA penalty

    Returns:
        GenerationResult with selectivity data populated
    """
    t0 = time.time()

    # Step 1: Enumerate
    raw = enumerate_molecules(
        metal=target_metal, max_candidates=max_candidates,
        pfilter=pfilter, hsab_filter=hsab_filter
    )
    raw.sort(key=lambda x: x[3])
    to_score = raw[:max_scored]

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa in to_score:
        try:
            # Score target
            sc = score_one(smiles, metal=target_metal, pH=pH, name=smiles[:40])

            gc = GeneratedCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=sc.log_Ka_pred,
                dg_total_kj=sc.dg_total_kj,
                prediction=sc.prediction,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                novel=smiles not in known,
            )

            # Score interferents
            for intf in interferents:
                try:
                    intf_sc = score_one(smiles, metal=intf, pH=pH)
                    gc.interferent_scores[intf] = intf_sc.log_Ka_pred
                    gc.selectivity_gaps[intf] = (gc.log_Ka_pred
                                                  - intf_sc.log_Ka_pred)
                except Exception:
                    gc.interferent_scores[intf] = 0.0
                    gc.selectivity_gaps[intf] = gc.log_Ka_pred

            if gc.selectivity_gaps:
                gc.worst_interferent = min(gc.selectivity_gaps,
                                           key=gc.selectivity_gaps.get)
                gc.min_gap = gc.selectivity_gaps[gc.worst_interferent]
            else:
                gc.min_gap = gc.log_Ka_pred
            gc.grade = _grade(gc.min_gap)

            # Composite: selectivity gap - SA penalty
            gc.composite_score = gc.min_gap - sa_penalty_weight * sa
            candidates.append(gc)

        except Exception as e:
            errors.append((smiles, str(e)))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return GenerationResult(
        target=target_metal,
        mode="selectivity",
        interferents=list(interferents),
        candidates=candidates,
        n_enumerated=len(raw),
        n_valid=len(raw),
        n_unique=len(raw),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


def generate_for_host(host_key, max_candidates=200, max_scored=50,
                      pfilter=None, sa_penalty_weight=0.3):
    """Generate novel guests for a synthetic host.

    Args:
        host_key: HOST_REGISTRY key, e.g. "beta-CD", "CB7"
        max_candidates, max_scored, pfilter: as above

    Returns:
        GenerationResult
    """
    t0 = time.time()

    if pfilter is None:
        pfilter = PropertyFilter()
    pfilter.require_donors = False  # guests don't need donor atoms
    pfilter.max_heavy_atoms = 30   # guests must fit in host cavity

    raw = enumerate_molecules(
        host=host_key, max_candidates=max_candidates,
        pfilter=pfilter, hsab_filter=False
    )
    raw.sort(key=lambda x: x[3])
    to_score = raw[:max_scored]

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa in to_score:
        try:
            sc = score_one(smiles, host=host_key, name=smiles[:40])
            gc = GeneratedCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=sc.log_Ka_pred,
                dg_total_kj=sc.dg_total_kj,
                prediction=sc.prediction,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                composite_score=sc.log_Ka_pred - sa_penalty_weight * sa,
                novel=smiles not in known,
            )
            candidates.append(gc)
        except Exception as e:
            errors.append((smiles, str(e)))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return GenerationResult(
        target=host_key,
        mode="host_guest",
        candidates=candidates,
        n_enumerated=len(raw),
        n_valid=len(raw),
        n_unique=len(raw),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_generation(result, top_n=20, verbose=False):
    """Pretty-print generation results."""
    print()
    print(f"  MABE De Novo Generator — {result.mode} generation")
    print(f"  Target: {result.target}")
    if result.interferents:
        print(f"  Interferents: {', '.join(result.interferents)}")
    print(f"  Enumerated: {result.n_enumerated} → "
          f"Scored: {result.n_scored} ({result.n_failed} failed) "
          f"in {result.elapsed_s:.1f}s")
    print()

    if not result.candidates:
        print("  No candidates generated.")
        return

    shown = result.candidates[:top_n]

    if result.mode == "selectivity":
        print(f"  {'#':>3s}  {'Grade':5s}  {'logKa':>6s}  {'MinGap':>7s}  "
              f"{'SA':>4s}  {'Comp':>6s}  {'Novel':5s}  SMILES")
        print(f"  {'─'*80}")
        for c in shown:
            print(f"  {c.rank:3d}  {c.grade:^5s}  {c.log_Ka_pred:+6.1f}  "
                  f"{c.min_gap:+7.1f}  {c.sa_score_val:4.1f}  "
                  f"{c.composite_score:+6.1f}  {'Y' if c.novel else 'N':^5s}  "
                  f"{c.smiles[:40]}")
    else:
        print(f"  {'#':>3s}  {'logKa':>7s}  {'SA':>4s}  {'Comp':>6s}  "
              f"{'Novel':5s}  SMILES")
        print(f"  {'─'*65}")
        for c in shown:
            print(f"  {c.rank:3d}  {c.log_Ka_pred:+7.2f}  {c.sa_score_val:4.1f}  "
                  f"{c.composite_score:+6.1f}  {'Y' if c.novel else 'N':^5s}  "
                  f"{c.smiles[:40]}")

    if verbose and result.best:
        b = result.best
        print()
        print(f"  ── Best: {b.name} ──")
        print(f"  SMILES: {b.smiles}")
        print(f"  Backbone: {b.backbone_name}")
        print(f"  Arms: {', '.join(b.arm_names)}")
        r = b.prediction
        for attr in ['dg_metal', 'dg_hydrophobic', 'dg_hbond',
                     'dg_conf_entropy', 'dg_shape', 'dg_cavity_dehydration',
                     'dg_ion_dipole']:
            val = getattr(r, attr, 0.0)
            if abs(val) > 0.01:
                print(f"    {attr:25s} {val:+8.2f} kJ/mol")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MABE De Novo Generator — Self-Test")
    print("=" * 70)

    # Test 1: Enumerate only (no scoring)
    print("\n--- Test 1: Enumeration ---")
    raw = enumerate_molecules(metal="Cu2+", max_candidates=50)
    print(f"  Enumerated {len(raw)} molecules for Cu2+")
    for smi, bb, arms, sa in raw[:5]:
        print(f"    SA={sa:.1f}  {bb:25s}  {smi[:50]}")

    # Test 2: Generate + score for Cu2+
    print("\n--- Test 2: Generate for Cu2+ ---")
    r = generate_candidates("Cu2+", max_candidates=100, max_scored=20)
    print_generation(r, top_n=10, verbose=True)

    # Test 3: Selectivity screen Pb2+ vs Ca2+/Mg2+
    print("\n--- Test 3: Selectivity screen Pb2+ ---")
    r2 = generate_and_screen("Pb2+", ["Ca2+", "Mg2+"],
                              max_candidates=100, max_scored=20, pH=5.0)
    print_generation(r2, top_n=10)

    # Test 4: Host-guest generation
    print("\n--- Test 4: Generate guests for beta-CD ---")
    r3 = generate_for_host("beta-CD", max_candidates=100, max_scored=20)
    print_generation(r3, top_n=10, verbose=True)

    print("\n  All self-tests completed.")
