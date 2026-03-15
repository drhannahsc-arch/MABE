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
# NOVEL HOST SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NovelHostSpec:
    """Specification for a novel cavity host not in HOST_REGISTRY.

    Provides the minimum properties needed for the unified scorer's
    novel host fallback path (see _synthesize_host_dict in unified_scorer_v2).

    Args:
        name: host identifier (e.g. "HKUST-1", "Cage-A")
        cavity_volume_A3: accessible cavity volume in ų
        host_charge: formal charge on host framework (default 0)
        n_hbonds_host: number of H-bond donor/acceptor sites at cavity portal
        portal_type: "neutral", "hydroxyl", "carbonyl", "amine" (affects H-bond model)
        host_type: classification string (e.g. "MOF", "cage", "synthetic_receptor")
        max_guest_volume_A3: optional; if set, enforces Rebek packing constraint.
            If 0, auto-computed as cavity_volume * 0.65 (upper Rebek bound).
    """
    name: str = "novel_host"
    cavity_volume_A3: float = 0.0
    host_charge: int = 0
    n_hbonds_host: int = 0
    portal_type: str = "neutral"
    host_type: str = "novel_cavity"
    max_guest_volume_A3: float = 0.0

    def __post_init__(self):
        if self.max_guest_volume_A3 <= 0 and self.cavity_volume_A3 > 0:
            # Rebek 55% rule: optimal packing 0.55, allow up to 0.65
            self.max_guest_volume_A3 = self.cavity_volume_A3 * 0.65


def _score_for_novel_host(smiles, host_spec, pH=7.4, name=None):
    """Score a guest SMILES against a novel host specification.

    Builds a UniversalComplex with cavity properties set directly,
    bypassing HOST_REGISTRY lookup. The unified scorer's novel host
    fallback (_synthesize_host_dict) handles the rest.

    Args:
        smiles: guest SMILES
        host_spec: NovelHostSpec instance
        pH: working pH
        name: optional label

    Returns:
        ScoredCandidate
    """
    from core.auto_descriptor import from_smiles, compute_guest_properties
    from core.unified_scorer_v2 import predict

    # Build UC from SMILES (no host= arg, so _apply_host won't fire)
    uc = from_smiles(smiles, pH=pH)

    # Manually apply novel host properties
    uc.host_name = host_spec.name
    uc.host_type = host_spec.host_type
    uc.binding_mode = "host_guest_inclusion"
    uc.cavity_volume_A3 = host_spec.cavity_volume_A3
    uc.host_charge = host_spec.host_charge

    # Estimate portal H-bonds from host spec
    uc.n_hbonds_formed = host_spec.n_hbonds_host

    # Packing coefficient
    if uc.guest_volume_A3 > 0 and host_spec.cavity_volume_A3 > 0:
        uc.packing_coefficient = uc.guest_volume_A3 / host_spec.cavity_volume_A3

    # Name
    uc.name = name or f"{smiles[:30]}@{host_spec.name}"

    r = predict(uc)
    return ScoredCandidate(
        smiles=smiles,
        name=uc.name,
        log_Ka_pred=r.log_Ka_pred,
        dg_total_kj=r.dg_total_kj,
        prediction=r,
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

    # Heteroatom diversity: more types of heteroatoms = harder synthesis
    het_types = set()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ("C", "H"):
            het_types.add(sym)
    n_het_types = len(het_types)

    # Functional group complexity from SMILES patterns
    smiles_for_fg = Chem.MolToSmiles(mol)
    fg_penalty = 0.0
    if "P(=O)" in smiles_for_fg or "P=O" in smiles_for_fg:
        fg_penalty += 0.6         # phosphonate/phosphine oxide
    if "C(=S)" in smiles_for_fg or "C=S" in smiles_for_fg:
        fg_penalty += 0.4         # thioamide/dithiocarbamate
    if "S(=O)(=O)" in smiles_for_fg:
        fg_penalty += 0.5         # sulfonamide/sulfonate
    if "N=C" in smiles_for_fg or "C=N" in smiles_for_fg:
        fg_penalty += 0.2         # imine/oxime
    if "B(O)" in smiles_for_fg:
        fg_penalty += 0.5         # boronic acid
    if "N=N" in smiles_for_fg:
        fg_penalty += 0.3         # hydrazide/azo

    # MW-based baseline (heavier molecules generally harder to make)
    mw = Descriptors.MolWt(mol)
    mw_term = mw / 200.0  # ~1.0 for MW=200, ~1.5 for MW=300

    # Complexity score
    complexity = (
        mw_term                                # MW-scaled baseline (not flat 1.0)
        + n_rings * 0.3                        # each ring adds difficulty
        + n_spiro * 1.0                        # spiro centers are hard
        + n_bridgehead * 1.0                   # bridgeheads are hard
        + n_stereo * 0.5                       # stereocenters need control
        + macro_penalty
        + size_penalty
        + max(0, n_rotatable - 10) * 0.1       # very flexible = harder to purify
        + max(0, n_het_types - 2) * 0.3        # heteroatom diversity penalty
        + fg_penalty                           # functional group complexity
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

    # ── CROWN ETHERS (macrocyclic O-donors) ───────────────────────────
    Backbone("12-crown-4", "[1*]OCCOCCOC[2*]", 2, "macrocyclic",
             "12-crown-4 open form, O4 donor set, Li+/Na+ selective"),
    Backbone("15-crown-5-open", "[1*]OCCOCCOCCOC[2*]", 2, "macrocyclic",
             "15-crown-5 open, O5 donor set, Na+/K+ selective"),
    Backbone("18-crown-6-open", "[1*]OCCOCCOCCOCCOC[2*]", 2, "macrocyclic",
             "18-crown-6 open, O6 donor set, K+/Ba2+ selective"),

    # ── CRYPTAND ──────────────────────────────────────────────────────
    Backbone("cryptand-222-open", "[1*]N(CCOCCOCCOC[2*])CCOCC", 2, "macrocyclic",
             "[2.2.2] cryptand open form, 3D cage, extreme alkali selectivity"),

    # ── HETEROCYCLIC N-DONORS ─────────────────────────────────────────
    Backbone("terpyridine", "[1*]c1cc(-c2cccc(-c3cc([2*])ccn3)n2)ccn1", 2, "aromatic",
             "2,2':6',2''-terpyridine, tridentate N3, Ru/Fe/Co"),
    Backbone("phenanthroline", "[1*]c1cc2ccc3cc([2*])cnc3c2nc1", 2, "aromatic",
             "1,10-phenanthroline disubst, rigid N2 chelator"),
    Backbone("triazine", "[1*]c1nc([2*])nc([3*])n1", 3, "aromatic",
             "1,3,5-triazine trisubstituted, C3-symmetric platform"),
    Backbone("bipyrazole", "[1*]c1cc(-c2cc([2*])n[nH]2)[nH]n1", 2, "aromatic",
             "3,3'-bipyrazole, N4 donor set, pH-switchable"),

    # ── PEPTIDIC SCAFFOLDS ────────────────────────────────────────────
    Backbone("tripeptide", "[1*]NCC(=O)NCC(=O)NCC(=O)[2*]", 2, "linear",
             "Gly-Gly-Gly backbone, amide N/O donors, metal-peptide"),
    Backbone("DKP", "[1*]C1NC(=O)C([2*])NC1=O", 2, "macrocyclic",
             "Diketopiperazine, cyclic dipeptide, rigid N2O2 platform"),
    Backbone("cyclic-tripeptide", "[1*]C1NC(=O)C([2*])NC(=O)C([3*])NC1=O", 3, "macrocyclic",
             "Cyclic tripeptide scaffold, 3 diversification sites"),

    # ── POLYAMINOCARBOXYLATE CORES ────────────────────────────────────
    Backbone("BPA", "[1*]N(Cc1ccccn1)Cc1cccc([2*])n1", 2, "aromatic",
             "Bis(2-pyridylmethyl)amine, N3 + 2 pendant sites"),
    Backbone("IDA-core", "[1*]N(CC(=O)O)CC(=O)O", 1, "linear",
             "Iminodiacetic acid, N1O4 chelator core (NTA half)"),
    Backbone("DTPA-open", "[1*]N(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)[2*]", 2, "linear",
             "DTPA-type open chain, N3O6, octadentate Gd3+ chelator"),
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

    # ── SOFT DONOR ARMS (added for Pb/Hg/Cd selectivity) ────────────────
    Arm("thioether-propyl", "[*]CCCS", ["S_thioether"], "S", "soft",
        "thioether, propyl spacer"),
    Arm("thioacetate", "[*]CC(=S)O", ["S_thioamide", "O_carboxylate"], "S", "soft",
        "thioacetate, bidentate S/O"),
    Arm("mercaptoacetate", "[*]SCC(=O)O", ["S_thiol", "O_carboxylate"], "S", "soft",
        "mercaptoacetic acid arm, S+O bidentate"),
    Arm("pyridine-2-thiol", "[*]c1cccc(S)n1", ["N_pyridine", "S_thiol"], "S", "borderline",
        "NS bidentate, borderline-soft"),
    Arm("thiadiazole-thiol", "[*]c1nnc(S)s1", ["S_thiol", "N_imine"], "S", "soft",
        "1,3,4-thiadiazole-2-thiol, strong soft donor"),
        # ── RECEPTOR-SPECIFIC ARMS (aromatic panels, H-bond arrays) ────────────
    Arm("indolyl", "[*]c1c[nH]c2ccccc12",
        ["N_pyrrole"], "N", "borderline",
        "Indole: NH donor + large aromatic surface, tryptophan mimic"),
    Arm("pyrrole", "[*]c1ccc[nH]1",
        ["N_pyrrole"], "N", "borderline",
        "Pyrrole: NH donor for anion binding, calixpyrrole unit"),
    Arm("naphthyl", "[*]c1cccc2ccccc12",
        [], "C", "borderline",
        "Naphthyl: large aromatic panel for pi-stacking / hydrophobic wall"),
    Arm("anthracenyl", "[*]c1ccc2cc3ccccc3cc2c1",
        [], "C", "borderline",
        "Anthracenyl: large pi surface, fluorescent reporter + stacking"),
    Arm("urea-NH2", "[*]NC(=O)N",
        ["N_amine", "O_carbonyl"], "N", "hard",
        "Terminal urea: 2 NH donors + C=O acceptor, anion binding arm"),
    Arm("guanidinium", "[*]NC(=N)N",
        ["N_amine"], "N", "hard",
        "Guanidinium: cationic, charge-assisted H-bond donor for oxoanions"),
    Arm("sulfonamide-NH", "[*]NS(=O)(=O)C",
        ["N_amine"], "N", "borderline",
        "Sulfonamide: acidic NH donor (pKa ~10), anion recognition"),
    Arm("nitrophenyl", "[*]c1ccc([N+](=O)[O-])cc1",
        [], "C", "borderline",
        "p-Nitrophenyl: electron-poor aromatic, CT stacking with electron-rich guests"),
        Arm("H-cap", "[*][H]", [], "", "", "cap"),

    # ── PEPTIDIC SIDE CHAINS ──────────────────────────────────────────
    Arm("cysteinyl", "[*]CC(N)C(=O)O",
        ["S_thiolate", "N_amine", "O_carboxylate"], "S", "soft",
        "cysteine"),
    Arm("seryl-OH", "[*]CO", ["O_hydroxyl"], "O", "hard", "serine"),
    Arm("threonyl-OH", "[*]C(C)O", ["O_hydroxyl"], "O", "hard", "threonine"),
    Arm("tyrosyl", "[*]Cc1ccc(O)cc1", ["O_phenolate"], "O", "hard",
        "tyrosine"),

    # ── BORONIC ACIDS (saccharide recognition) ────────────────────────
    Arm("phenylboronic", "[*]c1ccc(B(O)O)cc1", ["O_hydroxyl"], "B", "hard",
        "boronic_acid"),
    Arm("methylboronic", "[*]CB(O)O", ["O_hydroxyl"], "B", "hard",
        "boronic_acid"),

    # ── ANION RECOGNITION ─────────────────────────────────────────────
    Arm("urea", "[*]NC(=O)N", ["N_amine"], "N", "hard", "urea"),
    Arm("guanidinium", "[*]NC(=N)N", ["N_amine"], "N", "hard",
        "guanidinium"),
    Arm("sulfonamide", "[*]NS(=O)(=O)C", ["N_amine"], "N", "borderline",
        "sulfonamide"),

    # ── ETHER DONORS ──────────────────────────────────────────────────
    Arm("methoxy", "[*]COC", ["O_ether"], "O", "hard", "ether"),
    Arm("methoxyethyl", "[*]CCOC", ["O_ether"], "O", "hard", "ether"),
    Arm("ethoxyethyl", "[*]CCOCC", ["O_ether"], "O", "hard", "ether"),

    # ── CARBENE / ISOCYANIDE (soft C donors) ──────────────────────────
    Arm("NHC-imidazolyl", "[*]c1nccn1C", ["C_carbene"], "C", "soft",
        "NHC"),
    Arm("isocyanide", "[*]C[N+]#[C-]", ["C_isocyanide"], "C", "soft",
        "isocyanide"),

    # ── N-HETEROCYCLIC DONORS ─────────────────────────────────────────
    Arm("tetrazolyl", "[*]c1nnn[nH]1", ["N_pyridine"], "N", "borderline",
        "tetrazole"),
    Arm("pyrazolyl", "[*]c1cc[nH]n1", ["N_pyridine"], "N", "borderline",
        "pyrazole"),
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
        "tetrazolyl",   # carboxylate bioisostere (pKa-matched)
    ]),

    # ── Bidentate O,O chelate (hard, catechol/hydroxamate equivalents) ──
    ("bident_OO_chelate", [
        "catechol", "acetohydroxamate", "hydroxypyridinone",
        "bisphosphonate", "squaramide",
    ]),

    # ── Monodentate N aromatic (borderline, pyridine/imidazole type) ──
    ("mono_N_aromatic", [
        "2-pyridyl", "2-pyridylmethyl", "imidazolylmethyl",
        "benzimidazolyl", "pyrazolyl",
    ]),

    # ── Monodentate N amine (borderline, aliphatic amine) ──
    ("mono_N_amine", [
        "aminomethyl", "aminoethyl",
    ]),

    # ── Bidentate N,O (borderline, mixed imine/phenolate) ──
    ("bident_NO_mixed", [
        "salicylaldimine", "8-hydroxyquinolinyl", "oxime-simple",
        "hydrazide", "picolylamine", "amidoxime",
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
        "ethanol", "phenol", "seryl-OH", "threonyl-OH", "tyrosyl",
    ]),

    # ── Weak/special ──
    ("mono_N_weak", [
        "nitrile", "carbamoylmethyl",
    ]),

    # ── Boronic acids (saccharide recognition) ──
    ("boronic_acid", [
        "phenylboronic", "methylboronic",
    ]),

    # ── Anion recognition (H-bond donors) ──
    ("anion_donor", [
        "urea", "guanidinium", "sulfonamide",
    ]),

    # ── Monodentate O ether (hard, crown-type coordination) ──
    ("mono_O_ether", [
        "methoxy", "methoxyethyl", "ethoxyethyl",
    ]),

    # ── Soft C donors (organometallic) ──
    ("mono_C_soft", [
        "NHC-imidazolyl", "isocyanide",
    ]),

    # ── Polydentate peptidic (multi-donor amino acid side chain) ──
    ("peptidic_multi", [
        "cysteinyl",   # S,N,O tridentate
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
    # Pareto fields (populated when ranking_mode="pareto")
    pareto_front_idx: int = -1        # 0 = Pareto-optimal, -1 = not ranked
    pareto_rank: int = 0              # Combined rank (front + crowding)
    pareto_crowding: float = 0.0      # Crowding distance
    pareto_objectives: dict = field(default_factory=dict)  # name → value


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
    ranking_mode: str = "composite"        # "composite" or "pareto"
    pareto_result: object = None           # ParetoResult when mode=pareto

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

    # Round-robin across backbones to ensure scaffold diversity.
    # Each backbone gets a budget, then we cycle until max_candidates.
    import random as _rng
    _rng.seed(42)

    # Build combo iterators per backbone
    bb_combos = {}
    for bb in backbones:
        combos = list(cartesian_product(compatible_arms, repeat=bb.n_sites))
        _rng.shuffle(combos)
        bb_combos[bb.name] = (bb, combos, 0)  # (backbone, shuffled combos, index)

    # Round-robin: take N from each backbone per round
    PER_ROUND = max(3, max_candidates // (len(backbones) * 5))
    active_bbs = list(bb_combos.keys())
    _rng.shuffle(active_bbs)

    while len(results) < max_candidates and active_bbs:
        next_active = []
        for bb_name in active_bbs:
            if len(results) >= max_candidates:
                break
            bb, combos, idx = bb_combos[bb_name]
            added = 0
            while idx < len(combos) and added < PER_ROUND:
                arm_tuple = combos[idx]
                idx += 1

                smiles, mol = assemble(bb, list(arm_tuple))
                if smiles is None:
                    continue

                if smiles in seen:
                    continue
                seen.add(smiles)

                if not passes_filter(smiles, mol, pfilter):
                    continue

                sa = sa_score(mol)
                arm_names = [a.name for a in arm_tuple]
                results.append((smiles, bb.name, arm_names, sa))
                added += 1

            bb_combos[bb_name] = (bb, combos, idx)
            if idx < len(combos):
                next_active.append(bb_name)
        active_bbs = next_active

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API: GENERATE + SCORE
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidates(target_metal, pH=7.4, max_candidates=200,
                        max_scored=50, pfilter=None, hsab_filter=True,
                        sa_penalty_weight=0.3, ranking_mode="composite",
                        objectives=None):
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

    # Step 4: Rank
    _pareto_result = None
    if ranking_mode == "pareto" and candidates:
        from core.pareto import pareto_rank as _pareto_rank, AFFINITY_SA_OBJECTIVES
        _objs = objectives if objectives is not None else AFFINITY_SA_OBJECTIVES
        _pareto_result = _pareto_rank(candidates, objectives=_objs)
        for pc in _pareto_result.candidates:
            c = candidates[pc.index]
            c.pareto_front_idx = pc.front
            c.pareto_rank = pc.pareto_rank
            c.pareto_crowding = pc.crowding
            c.pareto_objectives = pc.objectives
        candidates.sort(key=lambda c: c.pareto_rank if c.pareto_rank > 0 else 9999)
    else:
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
        ranking_mode=ranking_mode,
        pareto_result=_pareto_result,
    )


def generate_and_screen(target_metal, interferents, pH=7.4,
                        max_candidates=200, max_scored=50,
                        pfilter=None, hsab_filter=True,
                        sa_penalty_weight=0.3,
                        ranking_mode="composite",
                        objectives=None):
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

    # Ranking
    _pareto_result = None
    if ranking_mode == "pareto" and candidates:
        from core.pareto import pareto_rank as _pareto_rank
        _pareto_result = _pareto_rank(candidates, objectives=objectives)
        for pc in _pareto_result.candidates:
            c = candidates[pc.index]
            c.pareto_front_idx = pc.front
            c.pareto_rank = pc.pareto_rank
            c.pareto_crowding = pc.crowding
            c.pareto_objectives = pc.objectives
        # Sort by Pareto rank (front ASC, crowding DESC)
        candidates.sort(key=lambda c: c.pareto_rank if c.pareto_rank > 0 else 9999)
    else:
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
        ranking_mode=ranking_mode,
        pareto_result=_pareto_result,
    )


def generate_for_host(host_key, max_candidates=200, max_scored=50,
                      pfilter=None, sa_penalty_weight=0.3,
                      ranking_mode="composite", objectives=None):
    """Generate novel guests for a host — known or novel.

    Args:
        host_key: HOST_REGISTRY key (str, e.g. "beta-CD", "CB7")
                  OR NovelHostSpec instance for MOFs, cages, etc.
        max_candidates: max molecules to enumerate
        max_scored: max to score (top by SA)
        pfilter: PropertyFilter instance
        sa_penalty_weight: weight for SA penalty in composite score
        ranking_mode: "composite" (default) or "pareto"
        objectives: custom Objective list for Pareto mode

    Returns:
        GenerationResult
    """
    t0 = time.time()
    _is_novel = isinstance(host_key, NovelHostSpec)
    host_spec = host_key if _is_novel else None
    host_name = host_spec.name if _is_novel else host_key

    if pfilter is None:
        pfilter = PropertyFilter()
    pfilter.require_donors = False  # guests don't need donor atoms

    # Size constraint from cavity volume
    if _is_novel and host_spec.max_guest_volume_A3 > 0:
        # Estimate max heavy atoms from volume: ~10 ų per heavy atom (rough)
        max_ha = max(10, int(host_spec.max_guest_volume_A3 / 10.0))
        pfilter.max_heavy_atoms = min(pfilter.max_heavy_atoms or 50, max_ha)
    else:
        pfilter.max_heavy_atoms = 30  # default for known hosts

    # enumerate_molecules only uses host for relaxing donor filter
    raw = enumerate_molecules(
        host=host_name, max_candidates=max_candidates,
        pfilter=pfilter, hsab_filter=False
    )
    raw.sort(key=lambda x: x[3])
    to_score = raw[:max_scored]

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa in to_score:
        try:
            if _is_novel:
                sc = _score_for_novel_host(smiles, host_spec, name=smiles[:40])
            else:
                sc = score_one(smiles, host=host_name, name=smiles[:40])

            # Guest volume check against cavity (Rebek constraint)
            if _is_novel and host_spec.max_guest_volume_A3 > 0:
                gv = sc.prediction.dg_metal  # wrong — need guest_volume
                # Get guest volume from the UC that was scored
                # prediction is PredictionResult which doesn't carry guest_volume
                # Instead, compute from SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    try:
                        gv_est = AllChem.ComputeMolVolume(mol)
                    except Exception:
                        gv_est = Descriptors.HeavyAtomCount(mol) * 10.0
                    if gv_est > host_spec.max_guest_volume_A3:
                        continue  # too large for cavity

            gc = GeneratedCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}@{host_name}",
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

    # Ranking
    _pareto_result = None
    if ranking_mode == "pareto" and candidates:
        from core.pareto import pareto_rank as _pareto_rank, AFFINITY_SA_OBJECTIVES
        _objs = objectives if objectives is not None else AFFINITY_SA_OBJECTIVES
        _pareto_result = _pareto_rank(candidates, objectives=_objs)
        for pc in _pareto_result.candidates:
            c = candidates[pc.index]
            c.pareto_front_idx = pc.front
            c.pareto_rank = pc.pareto_rank
            c.pareto_crowding = pc.crowding
            c.pareto_objectives = pc.objectives
        candidates.sort(key=lambda c: c.pareto_rank if c.pareto_rank > 0 else 9999)
    else:
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

    for i, c in enumerate(candidates):
        c.rank = i + 1

    return GenerationResult(
        target=host_name,
        mode="host_guest",
        candidates=candidates,
        n_enumerated=len(raw),
        n_valid=len(raw),
        n_unique=len(raw),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
        ranking_mode=ranking_mode,
        pareto_result=_pareto_result,
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
# MUTATIONAL SCANNING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MutationResult:
    """Result of mutating one arm position."""
    position: int                  # Which arm index was varied
    original_arm: str              # Name of original arm at this position
    variants: list = field(default_factory=list)   # list[GeneratedCandidate]
    best_variant: object = None    # Best-scoring variant at this position
    sensitivity: float = 0.0       # max log Ka delta from original


@dataclass
class ScanResult:
    """Full mutational scan output."""
    seed_smiles: str
    seed_score: float              # log Ka of original
    target: str
    positions: list = field(default_factory=list)  # list[MutationResult]
    most_sensitive_position: int = -1
    max_sensitivity: float = 0.0
    best_overall: object = None    # Best candidate across all positions
    elapsed_s: float = 0.0


def mutational_scan(smiles, metal=None, host=None, pH=7.4,
                    backbone_name=None, arm_names=None,
                    interferents=None, sa_penalty_weight=0.3,
                    max_variants_per_position=15):
    """Systematically vary one arm at a time, keeping all others fixed.

    If backbone_name + arm_names are not provided, attempts to find the
    closest matching backbone+arm decomposition by scoring all combos
    against the input SMILES. If that fails, uses a brute-force approach:
    enumerate 2-site backbones with all arm pairs, find the one that
    produces the input SMILES.

    Args:
        smiles: seed molecule SMILES
        metal: target metal for scoring
        host: target host for scoring
        pH: working pH
        backbone_name: known backbone (skip decomposition)
        arm_names: known arm list (skip decomposition)
        interferents: for selectivity scoring
        sa_penalty_weight: SA penalty weight
        max_variants_per_position: max arms to try per position

    Returns:
        ScanResult with per-position sensitivity analysis
    """
    t0 = time.time()

    # Score the seed
    try:
        seed_sc = score_one(smiles, metal=metal, host=host, pH=pH,
                            name="seed")
        seed_score = seed_sc.log_Ka_pred
    except Exception:
        seed_score = 0.0

    # Resolve backbone + arms
    bb = None
    arms_resolved = None

    if backbone_name and arm_names:
        # User provided decomposition
        bb_matches = [b for b in BACKBONE_LIBRARY if b.name == backbone_name]
        if bb_matches:
            bb = bb_matches[0]
            arms_resolved = [_ARM_BY_NAME.get(n) for n in arm_names]
            if None in arms_resolved:
                arms_resolved = None

    if bb is None or arms_resolved is None:
        # Try to find the backbone+arm combo that reproduces this SMILES
        bb, arms_resolved = _decompose_smiles(smiles, metal)

    if bb is None or arms_resolved is None:
        # Can't decompose — return empty scan
        return ScanResult(
            seed_smiles=smiles, seed_score=seed_score,
            target=metal or host or "unknown",
            elapsed_s=time.time() - t0,
        )

    # Filter compatible arms
    if metal:
        compat_arms = [a for a in ARM_LIBRARY
                       if hsab_compatible(metal, a) and a.name != "H-cap"]
    else:
        compat_arms = [a for a in ARM_LIBRARY if a.name != "H-cap"]

    positions = []
    all_variants = []

    for pos_idx in range(bb.n_sites):
        original_arm = arms_resolved[pos_idx]
        pos_variants = []

        # Try each compatible arm at this position, keep others fixed
        for alt_arm in compat_arms[:max_variants_per_position]:
            if alt_arm.name == original_arm.name:
                continue

            test_arms = list(arms_resolved)
            test_arms[pos_idx] = alt_arm
            out_smi, mol = assemble(bb, test_arms)
            if out_smi is None:
                continue
            if not passes_filter(out_smi, mol):
                continue

            try:
                sc = score_one(out_smi, metal=metal, host=host, pH=pH,
                               name=f"pos{pos_idx}:{alt_arm.name}")
                sa = sa_score(mol)
                gc = GeneratedCandidate(
                    smiles=out_smi,
                    name=f"{bb.name}[pos{pos_idx}→{alt_arm.name}]",
                    log_Ka_pred=sc.log_Ka_pred,
                    dg_total_kj=sc.dg_total_kj,
                    prediction=sc.prediction,
                    backbone_name=bb.name,
                    arm_names=[a.name for a in test_arms],
                    sa_score_val=sa,
                    composite_score=sc.log_Ka_pred - sa_penalty_weight * sa,
                )

                if interferents and metal:
                    for intf in interferents:
                        try:
                            intf_sc = score_one(out_smi, metal=intf, pH=pH)
                            gc.interferent_scores[intf] = intf_sc.log_Ka_pred
                            gc.selectivity_gaps[intf] = (gc.log_Ka_pred
                                                          - intf_sc.log_Ka_pred)
                        except Exception:
                            pass
                    if gc.selectivity_gaps:
                        gc.worst_interferent = min(gc.selectivity_gaps,
                                                   key=gc.selectivity_gaps.get)
                        gc.min_gap = gc.selectivity_gaps[gc.worst_interferent]
                        gc.grade = _grade(gc.min_gap)

                pos_variants.append(gc)
            except Exception:
                continue

        pos_variants.sort(key=lambda c: c.log_Ka_pred, reverse=True)
        best = pos_variants[0] if pos_variants else None
        sensitivity = abs(best.log_Ka_pred - seed_score) if best else 0.0

        mr = MutationResult(
            position=pos_idx,
            original_arm=original_arm.name,
            variants=pos_variants,
            best_variant=best,
            sensitivity=sensitivity,
        )
        positions.append(mr)
        all_variants.extend(pos_variants)

    # Find most sensitive position
    most_sens_idx = -1
    max_sens = 0.0
    for mr in positions:
        if mr.sensitivity > max_sens:
            max_sens = mr.sensitivity
            most_sens_idx = mr.position

    # Find best overall
    all_variants.sort(key=lambda c: c.log_Ka_pred, reverse=True)
    best_overall = all_variants[0] if all_variants else None

    return ScanResult(
        seed_smiles=smiles,
        seed_score=seed_score,
        target=metal or host or "unknown",
        positions=positions,
        most_sensitive_position=most_sens_idx,
        max_sensitivity=max_sens,
        best_overall=best_overall,
        elapsed_s=time.time() - t0,
    )


def _decompose_smiles(smiles, metal=None):
    """Try to find backbone + arm combo that produces this SMILES.

    Brute force: try all backbone × arm combos, check if output matches.
    Returns (backbone, [arm, ...]) or (None, None).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        target_canonical = Chem.MolToSmiles(mol)
    except Exception:
        return None, None

    arms_no_cap = [a for a in ARM_LIBRARY if a.name != "H-cap"]

    for bb in BACKBONE_LIBRARY:
        if bb.n_sites > 2:
            continue  # skip 3-site for decomposition speed
        from itertools import product as cartesian_product
        for arm_tuple in cartesian_product(arms_no_cap, repeat=bb.n_sites):
            out_smi, _ = assemble(bb, list(arm_tuple))
            if out_smi == target_canonical:
                return bb, list(arm_tuple)

    return None, None


def print_scan(scan):
    """Pretty-print a mutational scan result."""
    print()
    print(f"  MABE Mutational Scan")
    print(f"  Seed: {scan.seed_smiles[:50]}  (logKa = {scan.seed_score:+.2f})")
    print(f"  Target: {scan.target}")
    print(f"  Time: {scan.elapsed_s:.1f}s")
    print()

    if not scan.positions:
        print("  Could not decompose seed molecule.")
        return

    for mr in scan.positions:
        print(f"  Position {mr.position}: {mr.original_arm}")
        print(f"    Sensitivity: {mr.sensitivity:.2f} logKa units")
        print(f"    Variants tested: {len(mr.variants)}")
        if mr.best_variant:
            b = mr.best_variant
            delta = b.log_Ka_pred - scan.seed_score
            print(f"    Best: {b.arm_names[mr.position]:20s} "
                  f"logKa={b.log_Ka_pred:+.2f} (Δ={delta:+.2f})")
        print()

    if scan.best_overall:
        b = scan.best_overall
        print(f"  ── Best overall: logKa={b.log_Ka_pred:+.2f} ──")
        print(f"  {b.smiles[:60]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRAINED GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def constrained_generate(metal=None, host=None, pH=7.4,
                         required_donors=None, forbidden_donors=None,
                         required_hardness=None,
                         min_denticity=None, max_denticity=None,
                         require_macrocyclic=False,
                         require_category=None,
                         max_candidates=200, max_scored=50,
                         interferents=None, sa_penalty_weight=0.3,
                         hsab_filter=True):
    """Generate candidates with explicit donor/scaffold constraints.

    Args:
        metal: target metal for scoring
        host: target host for scoring
        pH: working pH
        required_donors: list of donor subtypes that MUST be present
            e.g. ["S_thiolate", "N_amine"]
        forbidden_donors: list of donor subtypes that must NOT appear
            e.g. ["O_carboxylate"]
        required_hardness: if set, only arms of this hardness
            e.g. "soft" for Hg2+ soft-donor-only design
        min_denticity: minimum total donor count
        max_denticity: maximum total donor count
        require_macrocyclic: only use macrocyclic backbones
        require_category: backbone category filter, e.g. "aromatic"
        max_candidates, max_scored: limits
        interferents: for selectivity screening
        sa_penalty_weight: SA penalty
        hsab_filter: apply HSAB pre-filter

    Returns:
        GenerationResult with constrained candidates
    """
    t0 = time.time()

    # Filter backbones
    bbs = list(BACKBONE_LIBRARY)
    if require_macrocyclic:
        bbs = [b for b in bbs if b.category == "macrocyclic"]
    if require_category:
        bbs = [b for b in bbs if b.category == require_category]

    # Filter arms
    arms = [a for a in ARM_LIBRARY if a.name != "H-cap"]
    if required_hardness:
        arms = [a for a in arms if a.hardness == required_hardness]
    if forbidden_donors:
        forbidden_set = set(forbidden_donors)
        arms = [a for a in arms
                if not any(d in forbidden_set for d in a.donor_subtypes)]
    if metal and hsab_filter:
        arms = [a for a in arms if hsab_compatible(metal, a)]

    if not arms:
        return GenerationResult(
            target=metal or host or "unknown",
            mode="constrained",
            elapsed_s=time.time() - t0,
        )

    # Enumerate with constraints
    pfilter = PropertyFilter()
    if host and not metal:
        pfilter.require_donors = False

    from itertools import product as cartesian_product

    seen = set()
    raw = []

    for bb in bbs:
        for arm_tuple in cartesian_product(arms, repeat=bb.n_sites):
            if len(raw) >= max_candidates:
                break

            # Check donor constraints
            all_donors = []
            for arm in arm_tuple:
                all_donors.extend(arm.donor_subtypes)

            # Required donors check
            if required_donors:
                donor_set = set(all_donors)
                if not all(d in donor_set for d in required_donors):
                    continue

            # Denticity bounds
            total_dent = len(all_donors)
            if min_denticity and total_dent < min_denticity:
                continue
            if max_denticity and total_dent > max_denticity:
                continue

            out_smi, mol = assemble(bb, list(arm_tuple))
            if out_smi is None:
                continue
            if out_smi in seen:
                continue
            seen.add(out_smi)
            if not passes_filter(out_smi, mol, pfilter):
                continue

            sa = sa_score(mol)
            arm_names = [a.name for a in arm_tuple]
            raw.append((out_smi, bb.name, arm_names, sa))

        if len(raw) >= max_candidates:
            break

    # Score
    raw.sort(key=lambda x: x[3])
    to_score = raw[:max_scored]

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
        mode="constrained",
        interferents=list(interferents) if interferents else [],
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
# 3D-SCORED GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_and_score_3d(target_metal, donor_subtypes=None,
                          geometry="auto", pH=7.4,
                          max_candidates=100, max_scored=20,
                          n_conformers=20, interferents=None,
                          sa_penalty_weight=0.3, hsab_filter=True,
                          weight_2d=0.5, weight_3d=0.5):
    """Generate candidates AND rank by 3D binding geometry quality.

    Combines: enumerate → 2D score → 3D conformer assessment → blended ranking.

    Args:
        target_metal:    e.g. "Cu2+"
        donor_subtypes:  expected donors for ideal pocket (e.g. ["N_amine"]*4).
                         If None, auto-inferred per candidate.
        geometry:        coordination geometry for ideal pocket
        pH:              working pH
        max_candidates:  max molecules to enumerate
        max_scored:      max to carry through scoring
        n_conformers:    conformers per molecule for 3D
        interferents:    competitor metals for selectivity
        sa_penalty_weight: SA penalty weight
        hsab_filter:     apply HSAB pre-filter
        weight_2d:       weight for 2D log Ka in composite
        weight_3d:       weight for 3D fidelity in composite

    Returns:
        GenerationResult with candidates sorted by composite_3d
    """
    # Phase 1: Standard 2D generation
    if interferents:
        result = generate_and_screen(
            target_metal, interferents, pH=pH,
            max_candidates=max_candidates, max_scored=max_scored,
            sa_penalty_weight=sa_penalty_weight, hsab_filter=hsab_filter)
    else:
        result = generate_candidates(
            target_metal, pH=pH,
            max_candidates=max_candidates, max_scored=max_scored,
            sa_penalty_weight=sa_penalty_weight, hsab_filter=hsab_filter)

    if not result.candidates:
        return result

    # Phase 2: 3D reranking
    from core.design_engine_v2 import rerank_3d
    result.candidates = rerank_3d(
        result.candidates, target_metal,
        donor_subtypes=donor_subtypes,
        geometry=geometry,
        n_conformers=n_conformers,
        weight_2d=weight_2d,
        weight_3d=weight_3d,
    )

    result.mode = result.mode + "_3d" if result.mode else "3d"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# DE NOVO RECEPTOR GENERATION FOR SMALL-MOLECULE GUESTS
# ═══════════════════════════════════════════════════════════════════════════
#
# Inverts the standard generate_candidates flow:
#   Standard: generate ligand → score against metal/host
#   Receptor: generate receptor → score complementarity to guest
#
# Uses existing ARM_LIBRARY + new receptor-oriented backbones.
# Arm selection is driven by guest pharmacophore: guest acceptors → donor
# arms, guest donors → acceptor arms, guest aromatics → aromatic arms.
# ═══════════════════════════════════════════════════════════════════════════

RECEPTOR_BACKBONE_LIBRARY = [
    # ── MOLECULAR TWEEZERS (2 sites) ─────────────────────────────────────
    # Two arms presented in a cleft geometry — good for flat aromatic guests
    Backbone("glycoluril-clip", "[1*]N1C(=O)NC2(N(C1=O)C1([2*])NC(=O)N1)CC2",
             2, "cyclic",
             "Glycoluril clip: U-shaped cavity, binds flat aromatics"),
    Backbone("xanthene-tweezer", "[1*]c1ccc2c(c1)Oc1cc([2*])ccc1C2",
             2, "aromatic",
             "Xanthene scaffold: 120° cleft, preorganized"),
    Backbone("Troeger-base", "[1*]CN1CC2=CC=CC(=C2N(C1)C[2*])C",
             2, "cyclic",
             "Tröger's base: V-shaped cleft, chiral, rigid"),
    Backbone("dibenzofuran-tweezer", "[1*]c1ccc2c(c1)oc1cc([2*])ccc12",
             2, "aromatic",
             "Dibenzofuran: rigid 150° cleft, aromatic walls"),

    # ── CLEFT / U-SHAPED (2-3 sites) ────────────────────────────────────
    Backbone("isophthalamide", "[1*]NC(=O)c1cccc(C(=O)N[2*])c1",
             2, "aromatic",
             "Isophthalamide cleft: convergent NH donors, anion/quinone binding"),
    Backbone("pyridine-2,6-diamide", "[1*]NC(=O)c1cccc(C(=O)N[2*])n1",
             2, "aromatic",
             "Pyridine-2,6-dicarboxamide: N+2×NH convergent, Hamilton receptor"),
    Backbone("urea-cleft", "[1*]NC(=O)NC(=O)N[2*]",
             2, "linear",
             "Bis-urea cleft: 4 NH donors, strong H-bond donor array"),
    Backbone("squaramide-cleft", "[1*]NC1=C(N[2*])C(=O)C1=O",
             2, "cyclic",
             "Squaramide: acidic NH + C=O array, quinone-complementary"),

    # ── MACROCYCLIC HOSTS (2-3 sites) ────────────────────────────────────
    Backbone("calix4-functionalized",
             "[1*]c1cc(CC2CC([2*])CC(CC3CC([1*])CC(C1)C3)C2)cc(C)c1O",
             2, "macrocyclic",
             "Calix[4]arene: upper-rim functionalization, deep cavity"),
    Backbone("pillar5-functionalized",
             "[1*]c1cc(Cc2cc([2*])cc(C)c2OC)c(OC)c1",
             2, "macrocyclic",
             "Pillar[5]arene-derived: tubular cavity"),

    # ── TRIPODAL RECEPTORS (3 sites) ─────────────────────────────────────
    Backbone("tris-amide-tripod",
             "[1*]NC(=O)CN(CC(=O)N[2*])CC(=O)N[3*]",
             3, "branched",
             "Tris(carboxamide)amine: 3 convergent NH + 3 C=O, cage-like"),
    Backbone("1,3,5-triamide-benzene",
             "[1*]NC(=O)c1cc(C(=O)N[2*])cc(C(=O)N[3*])c1",
             3, "aromatic",
             "Trimesic triamide: flat receptor, 3 NH convergent"),
    Backbone("CTV-core",
             "[1*]c1cc2c(cc1OC)CC(c1cc([2*])c(OC)cc1C1)c(cc([3*])c(OC)c3)C1Cc3c2",
             3, "macrocyclic",
             "Cyclotriveratrylene: bowl-shaped, 3 arm sites"),

    # ── EXPANDED CAVITY (2 sites, larger guests) ─────────────────────────
    Backbone("naphtho-tweezer",
             "[1*]c1ccc2cc3ccc([2*])cc3cc2c1",
             2, "aromatic",
             "Naphthalene-spaced tweezer: larger cleft for bigger guests"),
    Backbone("biphenyl-cleft",
             "[1*]c1ccc(-c2ccc([2*])cc2)cc1",
             2, "aromatic",
             "Biphenyl: rotatable but can converge arms"),
    # ── SOFT-METAL SELECTIVE (added for Pb/Hg/Cd selectivity) ────────────
    Backbone("bis-thioether-en", "[1*]SCCSC[2*]", 2, "linear",
             "S2N0 podand, soft-metal selective via thioether donors"),
    Backbone("NS2-triamine", "[1*]NCCSCCS[2*]", 2, "linear",
             "NS2 mixed donor, borderline-soft selectivity"),
    Backbone("dithiol-propyl", "[1*]SCCCS[2*]", 2, "linear",
             "S2 dithiol, highly soft-selective"),
    Backbone("pyridine-2-thiol-6-subst", "[1*]c1cccc(S)n1", 1, "aromatic",
             "Pyridine-thiol NS donor, borderline-soft"),
    Backbone("thiophene-2,5-disubst", "[1*]c1ccc([2*])s1", 2, "aromatic",
             "Thiophene S-donor platform, soft metal selective"),
    Backbone("thioether-crown-S3", "[1*]SCCSCCSC[2*]", 2, "macrocyclic",
             "Trithia macrocyclic motif, high Pb/Hg selectivity"),
    Backbone("NS2-macrocycle", "[1*]N1CCSCCSCC1[2*]", 2, "macrocyclic",
             "NS2 macrocycle, Pb²⁺ / Cd²⁺ selective"),
    # ── DEEP CLEFTS & AROMATIC BOXES ─────────────────────────────────────
    Backbone("diphenylmethane-cleft",
             "[1*]c1ccc(Cc2ccc([2*])cc2)cc1",
             2, "aromatic",
             "Diphenylmethane: hinged aromatic cleft, ~120 deg angle. "
             "Ref: Zimmerman HJ. Chem. Rev. 1997, 97, 1681"),
    Backbone("fluorene-platform",
             "[1*]c1ccc2c(c1)Cc1cc([2*])ccc1-2",
             2, "aromatic",
             "Fluorene: rigid 9H-fluorene platform, coplanar arms. "
             "Ref: Cram DJ. Science 1988, 240, 760"),
    Backbone("carbazole-cleft",
             "[1*]c1ccc2c(c1)[nH]c1cc([2*])ccc12",
             2, "aromatic",
             "Carbazole: NH donor at hinge + two aromatic walls. "
             "Ref: Etter MC. JACS 1990, 112, 8415"),
    Backbone("acridine-cleft",
             "[1*]c1ccc2cc3ccc([2*])cc3nc2c1",
             2, "aromatic",
             "Acridine: N-heterocycle hinge, fluorescent, intercalator geometry. "
             "Ref: Albert A. The Acridines, 2nd ed., Arnold 1966"),
    Backbone("terphenyl-spacer",
             "[1*]c1ccc(-c2ccc(-c3ccc([2*])cc3)cc2)cc1",
             2, "aromatic",
             "Terphenyl: extended linear spacer, ~15 A arm separation. "
             "Ref: Hamilton AD. Chem. Rev. 1997, 97, 1669"),
    Backbone("diphenylamine-cleft",
             "[1*]c1ccc(Nc2ccc([2*])cc2)cc1",
             2, "aromatic",
             "Diphenylamine: NH at hinge, electron-rich walls, ~120 deg. "
             "Ref: Anslyn EV. JACS 2005, 127, 15566"),
    Backbone("bis-naphthyl-methane",
             "[1*]c1ccc2ccccc2c1Cc1c([2*])ccc2ccccc12",
             2, "aromatic",
             "Bis(naphthyl)methane: large aromatic cleft for PAH guests. "
             "Ref: Klärner FG. Angew. Chem. Int. Ed. 2001, 40, 3635"),

    # ── ANION-BINDING SCAFFOLDS ──────────────────────────────────────────
    Backbone("thiourea-cleft",
             "[1*]NC(=S)NC(=S)N[2*]",
             2, "linear",
             "Bis-thiourea: 4 NH donors, strong anion/oxoanion binding. "
             "Ref: Gale PA. Chem. Commun. 2005, 3761"),
    Backbone("bis-amidopyridine",
             "[1*]NC(=O)c1cccc(C(=O)N[2*])n1",
             2, "aromatic",
             "2,6-bis(amido)pyridine: Hamilton-type receptor, 3 convergent H-bond "
             "donors for barbiturate/carboxylate guests. "
             "Ref: Hamilton AD. JACS 1988, 110, 1318"),
    Backbone("bis-sulfonamide-arene",
             "[1*]NS(=O)(=O)c1cccc(S(=O)(=O)N[2*])c1",
             2, "aromatic",
             "Bis-sulfonamide: acidic NH donors, anion cleft. "
             "Ref: Gale PA. Coord. Chem. Rev. 2003, 240, 191"),
    Backbone("bis-pyrrole-methane",
             "[1*]c1ccc([nH]1)Cc1ccc([2*])[nH]1",
             2, "aromatic",
             "Dipyrromethane: calix[4]pyrrole precursor, 2 NH donors converging. "
             "Ref: Sessler JL. Angew. Chem. Int. Ed. 1996, 35, 2380"),
    Backbone("guanidinium-cleft",
             "[1*]NC(=N)NC(=N)N[2*]",
             2, "linear",
             "Bis-guanidinium: cationic H-bond donor array, strong oxoanion binding. "
             "Ref: Schmidtchen FP. Chem. Rev. 1997, 97, 1609"),

    # ── CAPSULE-FORMING & 3D ENCAPSULATION ───────────────────────────────
    Backbone("tris-urea-tripod",
             "[1*]NC(=O)CCN(CCNC(=O)[2*])CCNC(=O)[3*]",
             3, "branched",
             "Tris(ureido)amine: 3 convergent urea NH pairs, cage-like H-bond "
             "capsule when dimerized. Ref: Rebek J. JACS 1996, 118, 2545"),
    Backbone("triamino-cyclohexane",
             "[1*]N[C@H]1C[C@@H](N[2*])C[C@H](N[3*])C1",
             3, "cyclic",
             "cis,cis-1,3,5-triaminocyclohexane: preorganized tripodal receptor, "
             "C3v symmetry. Ref: Steed JW. Supramol. Chem. 2000, 12, 129"),
    Backbone("benzene-tricarboxamide-extended",
             "[1*]CNC(=O)c1cc(C(=O)NC[2*])cc(C(=O)NC[3*])c1",
             3, "aromatic",
             "Extended trimesic triamide: methylene spacers give deeper cavity. "
             "Ref: Meijer EW. Chem. Rev. 2001, 101, 3893"),
]


# ═══════════════════════════════════════════════════════════════════════════
# PHARMACOPHORE-DRIVEN ARM SELECTION
# ═══════════════════════════════════════════════════════════════════════════

# Map guest feature types to complementary arm categories from ARM_LIBRARY.
# Guest acceptor → we need donor arms (provide H to guest's lone pairs)
# Guest donor → we need acceptor arms (accept H from guest's NH/OH)
# Guest aromatic → we need aromatic arms (π-stacking)

_COMPLEMENTARY_ARM_MAP = {
    # guest feature type → list of (arm_name, match_score) pairs
    "hb_acceptor": [
        # Arms that DONATE H-bonds to guest acceptors (C=O, N:)
        ("catechol", 0.9),          # two OH donors, strong with quinones
        ("phenol", 0.8),            # phenol OH donor
        ("ethanol", 0.6),           # aliphatic OH
        ("squaramide", 0.85),       # acidic NH + OH
        ("amidoxime", 0.7),         # N-OH + NH₂ donors
        ("acetohydroxamate", 0.75), # N-OH donor
        ("acetic-acid", 0.7),       # COOH donor
        ("thiourea", 0.65),         # NH donors
        ("hydrazide", 0.6),         # NH donor
    ],
    "hb_donor": [
        # Arms that ACCEPT H-bonds from guest donors (NH, OH)
        ("2-pyridyl", 0.85),            # pyridine N lone pair
        ("2-pyridylmethyl", 0.8),       # pyridine N lone pair
        ("imidazolylmethyl", 0.75),     # imidazole N lone pair
        ("benzimidazolyl", 0.7),        # benzimidazole N
        ("carbamoylmethyl", 0.6),       # amide C=O acceptor
        ("8-hydroxyquinolinyl", 0.7),   # N + O acceptors
        ("nitrile", 0.5),               # weak N acceptor
    ],
    "aromatic": [
        # Arms with aromatic surfaces for π-stacking
        ("phenol", 0.7),                # phenyl ring
        ("2-pyridyl", 0.75),            # pyridine ring
        ("catechol", 0.8),              # catechol ring (electron-rich → CT with quinone)
        ("8-hydroxyquinolinyl", 0.85),  # extended aromatic, best π-stacking
        ("benzimidazolyl", 0.8),        # fused aromatic
        ("salicylaldimine", 0.7),       # phenyl + imine
        ("aminothiadiazole", 0.65),     # heteroaromatic
    ],
    "hydrophobic": [
        # Arms with hydrophobic character
        ("thioether-methyl", 0.5),
        ("thiol-ethyl", 0.4),
        ("naphthyl", 0.85),            # large hydrophobic aromatic wall
        ("anthracenyl", 0.9),          # largest aromatic panel
        ("H-cap", 0.3),               # leave site unfunctionalized
    ],
    "anion": [
        # Arms that bind anions (carboxylate, phosphate, halide)
        ("pyrrole", 0.85),             # NH donor, calixpyrrole-like
        ("indolyl", 0.8),              # indole NH donor
        ("urea-NH2", 0.9),            # strong NH donor array
        ("guanidinium", 0.95),         # charge-assisted, best for oxoanions
        ("sulfonamide-NH", 0.75),      # acidic NH
        ("squaramide", 0.85),          # acidic NH + planar
        ("thiourea", 0.8),             # 2x NH donors
    ],
    "electron_poor_aromatic": [
        # Arms for charge-transfer stacking with electron-rich guests
        ("nitrophenyl", 0.9),          # strong electron-withdrawing
        ("sulfonamide-NH", 0.5),       # weakly electron-poor ring
        ("2-pyridyl", 0.6),            # moderately electron-poor
    ],
    "electron_rich_aromatic": [
        # Arms for CT stacking with electron-poor guests (quinones, NDI)
        ("catechol", 0.9),             # electron-rich, strong CT with quinones
        ("indolyl", 0.85),             # electron-rich heterocycle
        ("phenol", 0.7),               # moderately electron-rich
        ("naphthyl", 0.6),             # neutral aromatic
    ],
}


def _select_receptor_arms(pharmacophore, max_arms_per_type=5):
    """Select arms from ARM_LIBRARY that complement a guest pharmacophore.

    Returns list of (Arm, match_score, matched_guest_feature) tuples,
    sorted by match_score descending.
    """
    arm_scores = {}  # arm_name → (best_score, feature_type)

    for feat in pharmacophore.features:
        ft = feat.feature_type
        if ft not in _COMPLEMENTARY_ARM_MAP:
            continue
        for arm_name, score in _COMPLEMENTARY_ARM_MAP[ft]:
            # Boost for strong guest features
            if feat.strength == "strong":
                score *= 1.15
            # Quinone-specific boost for electron-rich aromatics
            if ft == "aromatic" and feat.subtype == "aromatic_6ring":
                if arm_name in ("catechol", "8-hydroxyquinolinyl"):
                    score *= 1.1  # electron-rich → charge-transfer with quinone

            if arm_name not in arm_scores or score > arm_scores[arm_name][0]:
                arm_scores[arm_name] = (score, ft)

    # Resolve to Arm objects
    results = []
    for arm_name, (score, ft) in sorted(arm_scores.items(), key=lambda x: x[1][0], reverse=True):
        if arm_name in _ARM_BY_NAME:
            results.append((_ARM_BY_NAME[arm_name], score, ft))

    return results[:max_arms_per_type * 3]  # cap total


def _score_receptor_complementarity(receptor_smiles, guest_pharmacophore):
    """Score how well a generated receptor complements the guest pharmacophore.

    Returns a complementarity score (higher = better match).

    Scoring axes:
    1. Donor/acceptor count match: receptor provides what guest needs
    2. Aromatic surface: receptor provides π-stacking area
    3. Size compatibility: receptor not too small for guest
    4. Preorganization bonus: rigid scaffolds score higher

    All physics-derived, no fitted parameters.
    """
    mol = Chem.MolFromSmiles(receptor_smiles)
    if mol is None:
        return 0.0

    score = 0.0

    # ── H-bond complementarity ──
    # Count receptor's HBD and HBA
    rec_hbd = Descriptors.NumHDonors(mol)
    rec_hba = Descriptors.NumHAcceptors(mol)

    # Guest's acceptors need receptor donors
    guest_acc = guest_pharmacophore.n_hb_acceptors
    hbd_match = min(rec_hbd, guest_acc) / max(guest_acc, 1)
    score += hbd_match * 3.0  # 3 points for full H-bond donor coverage

    # Guest's donors need receptor acceptors
    guest_don = guest_pharmacophore.n_hb_donors
    hba_match = min(rec_hba, guest_don) / max(guest_don, 1)
    score += hba_match * 2.0  # 2 points for H-bond acceptor coverage

    # ── Aromatic complementarity ──
    rec_arom_rings = Descriptors.NumAromaticRings(mol)
    guest_arom = guest_pharmacophore.n_aromatic_rings
    if guest_arom > 0 and rec_arom_rings > 0:
        score += min(rec_arom_rings, guest_arom * 2) * 1.0  # π-stacking potential

    # ── Size check ──
    rec_mw = Descriptors.ExactMolWt(mol)
    guest_mw = guest_pharmacophore.mw
    # Receptor should be at least 60% of guest MW for meaningful wrapping
    if rec_mw >= guest_mw * 0.6:
        score += 1.0
    # But not absurdly large (>5× guest)
    if rec_mw > guest_mw * 5.0:
        score -= 1.0

    # ── Preorganization bonus ──
    n_rot = Descriptors.NumRotatableBonds(mol)
    n_rings = Descriptors.RingCount(mol)
    # Rigid receptors bind more tightly (less conformational entropy loss)
    if n_rings >= 2 and n_rot < 5:
        score += 1.5  # preorganized
    elif n_rings >= 1:
        score += 0.5

    return max(0.0, score)


@dataclass
class ReceptorCandidate(GeneratedCandidate):
    """A de novo receptor candidate for a guest molecule."""
    complementarity_score: float = 0.0
    matched_features: list = field(default_factory=list)  # which guest features are covered


def generate_for_guest(
    guest_smiles,
    guest_name="",
    max_candidates=500,
    max_scored=50,
    sa_penalty_weight=0.3,
    pfilter=None,
    include_standard_backbones=True,
):
    """Generate de novo receptor molecules for a small-molecule guest.

    Strategy:
      1. Analyze guest pharmacophore
      2. Select arms complementary to guest features
      3. Enumerate receptor candidates from receptor backbones × arms
      4. Score by pharmacophore complementarity + SA penalty
      5. Return ranked candidates

    Args:
        guest_smiles: SMILES of the target guest molecule
        guest_name: display name (e.g. "6PPD-quinone")
        max_candidates: max molecules to enumerate
        max_scored: max to score (top by SA, then complement)
        sa_penalty_weight: weight of SA penalty in composite
        pfilter: PropertyFilter override
        include_standard_backbones: also use standard aromatic backbones
            from BACKBONE_LIBRARY (some work well as receptor scaffolds)

    Returns:
        GenerationResult with scored ReceptorCandidate objects
    """
    from core.small_molecule_target import analyze_guest

    t0 = time.time()

    # Step 1: Pharmacophore analysis
    pharmacophore = analyze_guest(guest_smiles, name=guest_name)

    # Step 2: Select complementary arms
    arm_selections = _select_receptor_arms(pharmacophore)
    receptor_arms = [arm for arm, _, _ in arm_selections]

    if not receptor_arms:
        # Fallback: use all arms
        receptor_arms = list(ARM_LIBRARY)

    # Step 3: Build backbone set
    backbones = list(RECEPTOR_BACKBONE_LIBRARY)
    if include_standard_backbones:
        # Include aromatic and branched standard backbones
        # (many are usable as receptor scaffolds)
        for bb in BACKBONE_LIBRARY:
            if bb.category in ("aromatic", "branched", "macrocyclic"):
                backbones.append(bb)

    # Step 4: Property filter
    if pfilter is None:
        pfilter = PropertyFilter()
    pfilter.require_donors = False  # receptors don't need metal donors
    pfilter.max_heavy_atoms = 60   # receptors can be larger than guests
    pfilter.min_heavy_atoms = 10   # but not trivially small

    # Step 5: Enumerate
    raw = enumerate_molecules(
        metal=None, host=None,
        max_candidates=max_candidates,
        backbones=backbones,
        arms=receptor_arms,
        pfilter=pfilter,
        hsab_filter=False,
    )
    n_enumerated = len(raw)

    # Step 6: Pre-score by complementarity, then take top N
    pre_scored = []
    for smiles, bb_name, arm_names, sa in raw:
        comp = _score_receptor_complementarity(smiles, pharmacophore)
        pre_scored.append((smiles, bb_name, arm_names, sa, comp))

    # Sort by complementarity (descending), take top
    pre_scored.sort(key=lambda x: x[4], reverse=True)
    to_score = pre_scored[:max_scored]

    # Step 7: Physics-based scoring through receptor_guest_scorer
    from core.receptor_guest_scorer import score_receptor_guest

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa, comp in to_score:
        try:
            # Score receptor-guest binding with calibrated physics
            rg_score = score_receptor_guest(smiles, guest_smiles)
            physics_log_ka = rg_score.log_Ka_pred
            physics_dg = rg_score.dg_total_kJ

            # Composite: physics log Ka minus SA penalty
            composite = physics_log_ka - sa_penalty_weight * sa

            gc = ReceptorCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=physics_log_ka,
                dg_total_kj=physics_dg,
                prediction=None,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                composite_score=composite,
                novel=smiles not in known,
                complementarity_score=comp,
            )
            candidates.append(gc)
        except Exception as e:
            errors.append((smiles, str(e)))

    # Step 8: Rank
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return GenerationResult(
        target=guest_name or guest_smiles[:40],
        mode="receptor",
        candidates=candidates,
        n_enumerated=n_enumerated,
        n_valid=n_enumerated,
        n_unique=n_enumerated,
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=time.time() - t0,
        errors=errors,
    )


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

# ═══════════════════════════════════════════════════════════════════════════
# NOVEL HOST LIBRARY CONVENIENCE WRAPPERS
# Added by bootstrap_novel_host_library.py
# ═══════════════════════════════════════════════════════════════════════════

def generate_for_known_host(host_name, **kwargs):
    """Generate guests for a named host from the novel host library.

    Convenience wrapper: looks up host_name in novel_host_library,
    then calls generate_for_host() with the resulting NovelHostSpec.

    Args:
        host_name: str - host name or alias (e.g. "HKUST-1", "Cu-BTC", "CC3")
        **kwargs: passed to generate_for_host()

    Returns:
        GenerationResult

    Example:
        r = generate_for_known_host("HKUST-1", max_candidates=200)
        r = generate_for_known_host("Cu-BTC", ranking_mode="pareto")
    """
    from core.novel_host_library import get_host
    spec = get_host(host_name)
    return generate_for_host(spec, **kwargs)


def list_available_hosts(host_type=None):
    """List all available novel hosts for guest generation.

    Args:
        host_type: optional filter - "MOF", "cage", "zeolite", "synthetic_receptor"

    Returns:
        list of (name, cavity_volume, host_type) tuples
    """
    from core.novel_host_library import list_hosts
    entries = list_hosts(host_type=host_type)
    return [(e.spec.name, e.spec.cavity_volume_A3, e.category) for e in entries]

