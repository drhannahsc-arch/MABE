"""
donor_enumerator.py — Combinatorial donor set generator for MABE design engine.

Enumerates chemically feasible coordination environments from the 18+ donor
subtypes, constrained by:
  - Metal coordination number (CN) range
  - Denticity and chelate ring formation rules
  - Chemical feasibility filters (no impossible combinations)
  - HSAB pre-filtering (skip donor sets that are obviously mismatched)

Usage:
    from donor_enumerator import enumerate_donor_sets, DonorSet
    candidates = enumerate_donor_sets("Pb2+", pH=5.0, max_candidates=500)
"""

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Optional

from core.scorer_frozen import METAL_DB, DONOR_SOFTNESS, DONOR_PKA, SUBTYPE_EXCHANGE


# ═══════════════════════════════════════════════════════════════════════════
# DONOR SET REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DonorSet:
    """A candidate coordination environment for a metal ion."""
    donor_subtypes: list[str]           # e.g. ["N_amine", "N_amine", "O_carboxylate", ...]
    chelate_rings: int = 0              # Number of chelate rings (total)
    denticity: int = 1                  # Denticity of the (conceptual) ligand
    is_macrocyclic: bool = False        # Crown ether, cryptand, etc.
    cavity_radius_nm: Optional[float] = None  # Cavity radius for size-match (nm)
    ring_sizes: list[int] = field(default_factory=list)  # 5 or 6 per ring (Hancock)
    n_ligand_molecules: int = 1         # Number of separate ligand molecules
    archetype: str = ""                 # Human-readable description
    feasibility_notes: list[str] = field(default_factory=list)

    @property
    def cn(self) -> int:
        return len(self.donor_subtypes)

    @property
    def donor_elements(self) -> list[str]:
        return [s.split("_")[0] for s in self.donor_subtypes]

    @property
    def signature(self) -> str:
        """Canonical string for deduplication."""
        return "|".join(sorted(self.donor_subtypes))

    def __repr__(self):
        donors_str = ", ".join(self.donor_subtypes)
        return (f"DonorSet(CN={self.cn}, [{donors_str}], "
                f"rings={self.chelate_rings}, dent={self.denticity})")


# ═══════════════════════════════════════════════════════════════════════════
# FEASIBILITY RULES
# ═══════════════════════════════════════════════════════════════════════════

# Maximum count of each donor subtype in a single ligand (chemical feasibility)
MAX_DONOR_COUNT = {
    "O_catecholate": 6,       # Tris(catecholate) like enterobactin
    "O_hydroxamate": 6,       # Tris(hydroxamate) like desferrioxamine
    "O_carboxylate": 6,       # EDTA has 4, DTPA has 5
    "O_phenolate": 4,         # Salen-type
    "O_hydroxyl": 4,
    "O_ether": 8,             # Crown ethers up to 8
    "O_phosphate": 4,
    "O_sulfonate": 2,
    "N_amine": 6,             # Hexaammine or tris(en)
    "N_imine": 6,             # Tris(bipyridine) equivalent
    "N_pyridine": 6,          # Tris(bipyridine)
    "N_imidazole": 4,         # His-tag type
    "N_nitrile": 4,
    "N_amide": 4,
    "S_thiolate": 4,          # Metallothionein-like
    "S_thioether": 6,         # Crown thioether
    "S_thiosulfate": 3,
    "S_dithiocarbamate": 4,
    "P_phosphine": 4,
    "Cl_chloride": 4,
    "Br_bromide": 4,
    "I_iodide": 4,
    # Non-carbon-bias additions
    "Se_selenolate": 4,    # Analogous to S_thiolate
    "Se_selenoether": 6,   # Selenacrown analogues
    "Se_selenourea": 4,
    "Te_tellurolate": 4,
    "Te_telluroether": 6,
    "As_arsine": 4,
    "As_arsenite": 3,      # Tridentate as As(O)₃ donor
    "Sb_stibine": 2,       # Rare; mainly bidentate
    "P_phosphonate": 4,
    "P_phosphite": 4,
    "F_fluoride": 6,       # Fluoride bridges in high-valent complexes
    "C_cyanide": 6,        # Hexacyanometallate
    "C_carbonyl": 6,       # Hexacarbonyl
    "N_oxime": 4,
    "N_hydrazine": 4,
    "N_aromatic": 6,
    "N_thioamide": 4,
    "S_sulfoxide": 4,
    "S_thioamide": 4,
    "O_carbonyl": 8,
    "O_phosphonate": 4,
}

# Donor subtypes grouped by chemical compatibility
# Donors within a group can coexist in one ligand
COMPATIBLE_GROUPS = [
    # EDTA-type: amines + carboxylates
    {"N_amine", "O_carboxylate"},
    # Salen-type: imines + phenolates
    {"N_imine", "O_phenolate"},
    # Bipyridine-type: pyridines
    {"N_pyridine"},
    # Catecholate-type: pure catechol
    {"O_catecholate"},
    # Hydroxamate-type: pure hydroxamate
    {"O_hydroxamate"},
    # Crown ether-type: ethers
    {"O_ether"},
    # Thiolate-type: thiolates + amines (cysteine-like)
    {"S_thiolate", "N_amine"},
    # Dithiocarbamate-type
    {"S_dithiocarbamate"},
    # Mixed N/O: amines + hydroxamate/phenolate
    {"N_amine", "O_hydroxamate"},
    {"N_amine", "O_phenolate"},
    {"N_amine", "O_catecholate"},
    # Phosphine-type
    {"P_phosphine"},
    # Peptide-type: amide + amine + carboxylate + imidazole
    {"N_amide", "N_amine", "O_carboxylate", "N_imidazole"},
    # Thioether crown + amine
    {"S_thioether", "N_amine"},
    # Pyridine + carboxylate (picolinic acid type)
    {"N_pyridine", "O_carboxylate"},
    # Imidazole + carboxylate (histidine type)
    {"N_imidazole", "O_carboxylate"},
    # ── Non-carbon-bias groups ────────────────────────────────────────────
    # Selenolate + amine (Se/N mixed bidentate for soft metals)
    {"Se_selenolate", "N_amine"},
    # Selenolate + imine (Schiff-base Se/N)
    {"Se_selenolate", "N_imine"},
    # Pure selenolate (metallothionein Se analogue)
    {"Se_selenolate"},
    # Selenoether crown (crown ether with Se replacing O)
    {"Se_selenoether"},
    # Mixed thiolate + selenolate (gradient chalcogenide chelator)
    {"S_thiolate", "Se_selenolate"},
    # Phosphine + selenolate (transition metal P/Se bidentate)
    {"P_phosphine", "Se_selenolate"},
    # Phosphine + thiolate (soft metal P/S bidentate)
    {"P_phosphine", "S_thiolate"},
    # Phosphine + amine (mixed P/N)
    {"P_phosphine", "N_amine"},
    # Arsine + phosphine (mixed pnictogen)
    {"As_arsine", "P_phosphine"},
    # Cyanide (hexacyanometallate-type donors)
    {"C_cyanide"},
    # Fluoride-selective hard donors
    {"F_fluoride", "O_carboxylate"},
    {"F_fluoride"},
    # Oxime + amine (dimethylglyoxime-type)
    {"N_oxime", "N_amine"},
    {"N_oxime"},
]

# Chelate ring rules: which donor pairs can form 5-membered chelate rings
# (donor_a, donor_b) → True if they can be adjacent in a chelate ring
CHELATE_PAIRS = {
    ("N_amine", "N_amine"),             # Ethylenediamine
    ("N_amine", "O_carboxylate"),       # Glycinate
    ("N_amine", "S_thiolate"),          # Cysteine
    ("N_imine", "O_phenolate"),         # Salen
    ("N_imine", "N_imine"),             # Bipyridine backbone
    ("N_pyridine", "N_pyridine"),       # Bipyridine
    ("O_catecholate", "O_catecholate"), # Catecholate bidentate
    ("O_hydroxamate", "O_hydroxamate"), # Hydroxamate bidentate
    ("O_carboxylate", "O_hydroxyl"),    # Citrate-type
    ("O_carboxylate", "O_carboxylate"), # Malonate-type (6-ring but counts)
    ("S_dithiocarbamate", "S_dithiocarbamate"),  # DTC bidentate
    ("N_amine", "N_pyridine"),          # Picolylamine
    ("N_amine", "N_imidazole"),         # Histamine-type
    ("N_amine", "O_hydroxamate"),       # DFOB-type
    ("N_pyridine", "O_carboxylate"),    # Picolinate
    ("N_imidazole", "O_carboxylate"),   # Histidine
    ("S_thioether", "N_amine"),         # Met-type
    ("O_ether", "O_ether"),             # Crown ether
    ("P_phosphine", "P_phosphine"),     # Bisphosphine
    # ── Non-carbon-bias chelate pairs ────────────────────────────────────
    # Se-containing chelate pairs
    ("Se_selenolate", "N_amine"),        # Selenocysteine-type 5-ring
    ("Se_selenolate", "N_imine"),        # Se/N Schiff-base chelate
    ("Se_selenolate", "Se_selenolate"),  # Diselenolate bidentate
    ("Se_selenoether", "Se_selenoether"),# Selenacrown ether
    ("S_thiolate", "Se_selenolate"),     # Mixed S/Se bidentate
    ("Se_selenolate", "O_carboxylate"),  # Se-carboxylate 5-ring
    # P-containing chelate pairs
    ("P_phosphine", "N_amine"),          # Aminophosphine 5-ring
    ("P_phosphine", "S_thiolate"),       # Phosphino-thiolate
    ("P_phosphine", "Se_selenolate"),    # Phosphino-selenolate
    ("P_phosphine", "O_carboxylate"),    # Phosphinocarboxylate
    ("P_phosphite", "P_phosphite"),      # Bisphosphite
    ("As_arsine", "As_arsine"),          # Bis(arsine)
    ("As_arsine", "P_phosphine"),        # As/P mixed bidentate
    ("As_arsine", "N_amine"),            # Arsino-amine
    # C-donor pairs
    ("C_cyanide", "N_amine"),            # Cyanide + amine (ambidentate)
    # Oxime chelate pairs
    ("N_oxime", "N_oxime"),              # Bis-oxime (DMG-type)
    ("N_oxime", "N_amine"),              # Oxime-amine
    # Te (rare; only for extreme soft metals)
    ("Te_tellurolate", "Te_tellurolate"),
    ("Te_tellurolate", "N_amine"),
}


# ═══════════════════════════════════════════════════════════════════════════
# ARCHETYPE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

# Pre-defined ligand archetypes that are known to be synthetically accessible
ARCHETYPES = [
    # EDTA family
    DonorSet(["N_amine","N_amine","O_carboxylate","O_carboxylate",
              "O_carboxylate","O_carboxylate"],
             chelate_rings=5, denticity=6, is_macrocyclic=False,
             ring_sizes=[5,5,5,5,5], n_ligand_molecules=1,
             archetype="EDTA-type (N₂O₄)"),
    DonorSet(["N_amine","N_amine","N_amine","O_carboxylate","O_carboxylate",
              "O_carboxylate","O_carboxylate","O_carboxylate"],
             chelate_rings=8, denticity=8, is_macrocyclic=False,
             ring_sizes=[5]*8, n_ligand_molecules=1,
             archetype="DTPA-type (N₃O₅)"),
    # Salen family
    DonorSet(["N_imine","N_imine","O_phenolate","O_phenolate"],
             chelate_rings=2, denticity=4, is_macrocyclic=False,
             ring_sizes=[5,6], n_ligand_molecules=1,
             archetype="Salen-type (N₂O₂)"),
    # Bipyridine family
    DonorSet(["N_pyridine","N_pyridine"],
             chelate_rings=1, denticity=2, is_macrocyclic=False,
             ring_sizes=[5], n_ligand_molecules=1,
             archetype="Bipyridine (monodentate pair)"),
    DonorSet(["N_pyridine","N_pyridine","N_pyridine","N_pyridine",
              "N_pyridine","N_pyridine"],
             chelate_rings=3, denticity=6, is_macrocyclic=False,
             ring_sizes=[5,5,5], n_ligand_molecules=3,
             archetype="Tris(bipyridine) (3 ligands)"),
    # Catecholate family
    DonorSet(["O_catecholate","O_catecholate"],
             chelate_rings=1, denticity=2, is_macrocyclic=False,
             ring_sizes=[5], n_ligand_molecules=1,
             archetype="Catecholate (bidentate)"),
    DonorSet(["O_catecholate","O_catecholate","O_catecholate",
              "O_catecholate","O_catecholate","O_catecholate"],
             chelate_rings=3, denticity=6, is_macrocyclic=False,
             ring_sizes=[5,5,5], n_ligand_molecules=3,
             archetype="Tris(catecholate) siderophore"),
    # Hydroxamate family
    DonorSet(["O_hydroxamate","O_hydroxamate","O_hydroxamate",
              "O_hydroxamate","O_hydroxamate","O_hydroxamate"],
             chelate_rings=3, denticity=6, is_macrocyclic=False,
             ring_sizes=[5,5,5], n_ligand_molecules=3,
             archetype="Tris(hydroxamate) siderophore"),
    # Crown ethers (with cavity radii from Pedersen/Izatt data)
    DonorSet(["O_ether"]*4, chelate_rings=4, denticity=4, is_macrocyclic=True,
             cavity_radius_nm=0.060, ring_sizes=[5,5,5,5], n_ligand_molecules=1,
             archetype="12-crown-4"),
    DonorSet(["O_ether"]*5, chelate_rings=5, denticity=5, is_macrocyclic=True,
             cavity_radius_nm=0.092, ring_sizes=[5,5,5,5,5], n_ligand_molecules=1,
             archetype="15-crown-5"),
    DonorSet(["O_ether"]*6, chelate_rings=6, denticity=6, is_macrocyclic=True,
             cavity_radius_nm=0.140, ring_sizes=[5,5,5,5,5,5], n_ligand_molecules=1,
             archetype="18-crown-6"),
    # Thiolate
    DonorSet(["S_thiolate","S_thiolate"],
             chelate_rings=0, denticity=2, n_ligand_molecules=2,
             archetype="Bis(thiolate)"),
    DonorSet(["S_thiolate","S_thiolate","S_thiolate","S_thiolate"],
             chelate_rings=0, denticity=4, n_ligand_molecules=4,
             archetype="Tetra(thiolate) — metallothionein-like"),
    # Cysteine-type (S + N + O)
    DonorSet(["S_thiolate","N_amine","O_carboxylate"],
             chelate_rings=2, denticity=3, ring_sizes=[5,5], n_ligand_molecules=1,
             archetype="Cysteine-type (SNO)"),
    # DTC
    DonorSet(["S_dithiocarbamate","S_dithiocarbamate"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Dithiocarbamate (bidentate)"),
    DonorSet(["S_dithiocarbamate","S_dithiocarbamate",
              "S_dithiocarbamate","S_dithiocarbamate"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Bis(dithiocarbamate)"),
    # Ethylenediamine family
    DonorSet(["N_amine","N_amine"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Ethylenediamine (en)"),
    DonorSet(["N_amine","N_amine","N_amine","N_amine",
              "N_amine","N_amine"],
             chelate_rings=3, denticity=6, ring_sizes=[5,5,5], n_ligand_molecules=3,
             archetype="Tris(ethylenediamine)"),
    # Glycinate family
    DonorSet(["N_amine","O_carboxylate"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Glycinate-type"),
    DonorSet(["N_amine","O_carboxylate","N_amine","O_carboxylate"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Bis(glycinate)"),
    # Picolinate family
    DonorSet(["N_pyridine","O_carboxylate"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Picolinate-type"),
    DonorSet(["N_pyridine","O_carboxylate","N_pyridine","O_carboxylate",
              "N_pyridine","O_carboxylate"],
             chelate_rings=3, denticity=6, ring_sizes=[5,5,5], n_ligand_molecules=3,
             archetype="Tris(picolinate)"),
    # Phosphine
    DonorSet(["P_phosphine","P_phosphine"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Bisphosphine (soft metal binder)"),
    # Imidazole family
    DonorSet(["N_imidazole","N_imidazole","N_imidazole","N_imidazole"],
             chelate_rings=0, denticity=4, n_ligand_molecules=4,
             archetype="His-tag type (4× imidazole)"),
    # Mixed O-hard chelators
    DonorSet(["O_carboxylate","O_carboxylate","O_hydroxyl"],
             chelate_rings=2, denticity=3, ring_sizes=[5,6], n_ligand_molecules=1,
             archetype="Citrate-type (O₃)"),
    DonorSet(["O_carboxylate","O_hydroxamate","N_amine"],
             chelate_rings=2, denticity=3, ring_sizes=[5,5], n_ligand_molecules=1,
             archetype="Mixed hard chelator (NHO₂)"),
    # ── Non-carbon-bias archetypes ────────────────────────────────────────
    # Selenolate: Se/N bidentate (selenocysteine analogue)
    DonorSet(["Se_selenolate","N_amine"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Selenocysteine-type (SeN)"),
    DonorSet(["Se_selenolate","Se_selenolate"],
             chelate_rings=0, denticity=2, n_ligand_molecules=2,
             archetype="Bis(selenolate) — soft metal capture"),
    DonorSet(["Se_selenolate","Se_selenolate","Se_selenolate","Se_selenolate"],
             chelate_rings=0, denticity=4, n_ligand_molecules=4,
             archetype="Tetra(selenolate) — Hg2+/Au+/Pd2+"),
    DonorSet(["Se_selenolate","N_amine","Se_selenolate","N_amine"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Bis(selenocysteine)-type (Se₂N₂)"),
    # Selenoether crowns (analogous to thiacrowns; Ibers & Holm 1980)
    DonorSet(["Se_selenoether"]*4,
             chelate_rings=4, denticity=4, is_macrocyclic=True,
             cavity_radius_nm=0.065, ring_sizes=[5,5,5,5], n_ligand_molecules=1,
             archetype="Selenacrown-12 ([12]aneS₄ Se-analogue)"),
    DonorSet(["Se_selenoether"]*6,
             chelate_rings=6, denticity=6, is_macrocyclic=True,
             cavity_radius_nm=0.145, ring_sizes=[5]*6, n_ligand_molecules=1,
             archetype="Selenacrown-18 (large selenacrown)"),
    # Mixed thiacrown: S + Se for graded soft selectivity
    DonorSet(["S_thioether","Se_selenoether","S_thioether","Se_selenoether"],
             chelate_rings=4, denticity=4, is_macrocyclic=True,
             cavity_radius_nm=0.075, ring_sizes=[5]*4, n_ligand_molecules=1,
             archetype="Mixed [12]aneS₂Se₂ thia-selenacrown"),
    # Phosphine binders: key for precious metals (Au, Pt, Pd, Rh)
    DonorSet(["P_phosphine","P_phosphine","P_phosphine","P_phosphine"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Bis(bisphosphine) — Pd2+/Pt2+/Rh3+"),
    DonorSet(["P_phosphine","S_thiolate"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Phosphino-thiolate (P/S bidentate)"),
    DonorSet(["P_phosphine","Se_selenolate"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Phosphino-selenolate (P/Se bidentate)"),
    # Arsine binders
    DonorSet(["As_arsine","As_arsine"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Bis(arsine) (DIARS-type)"),
    DonorSet(["As_arsine","As_arsine","As_arsine","As_arsine"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Tetra(arsine) — Pd2+/Pt2+"),
    # Cyanide complexes
    DonorSet(["C_cyanide","C_cyanide","C_cyanide","C_cyanide",
              "C_cyanide","C_cyanide"],
             chelate_rings=0, denticity=6, n_ligand_molecules=6,
             archetype="Hexacyanometallate [M(CN)₆]⁴⁻ — Fe2+/Fe3+"),
    DonorSet(["C_cyanide","C_cyanide","C_cyanide","C_cyanide"],
             chelate_rings=0, denticity=4, n_ligand_molecules=4,
             archetype="Tetracyanometallate [M(CN)₄]²⁻ — Ni2+/Pd2+"),
    # Oxime: DMG-type (dimethylglyoxime — classic Ni2+ chelator)
    DonorSet(["N_oxime","N_oxime"],
             chelate_rings=1, denticity=2, ring_sizes=[5], n_ligand_molecules=1,
             archetype="Dimethylglyoxime-type (N₂ bis-oxime)"),
    DonorSet(["N_oxime","N_oxime","N_oxime","N_oxime"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Bis(dimethylglyoximate) — Ni2+/Co2+"),
    # Fluoride-selective hard chelators (Al3+, Zr4+, Th4+)
    DonorSet(["F_fluoride","F_fluoride","F_fluoride","F_fluoride"],
             chelate_rings=0, denticity=4, n_ligand_molecules=4,
             archetype="Tetrafluoride coordination — Al3+/Zr4+"),
    DonorSet(["F_fluoride","O_carboxylate","F_fluoride","O_carboxylate"],
             chelate_rings=2, denticity=4, ring_sizes=[5,5], n_ligand_molecules=2,
             archetype="Fluorocarboxylate chelate — hard metal selective"),
    # Thioether crowns (already in ARCHETYPES via crown ethers; Se versions above)
    # Adding S-crowns explicitly for completeness
    DonorSet(["S_thioether"]*4,
             chelate_rings=4, denticity=4, is_macrocyclic=True,
             cavity_radius_nm=0.065, ring_sizes=[5,5,5,5], n_ligand_molecules=1,
             archetype="[12]aneS₄ thiacrown"),
    DonorSet(["S_thioether"]*6,
             chelate_rings=6, denticity=6, is_macrocyclic=True,
             cavity_radius_nm=0.145, ring_sizes=[5]*6, n_ligand_molecules=1,
             archetype="[18]aneS₆ large thiacrown"),
]


# ═══════════════════════════════════════════════════════════════════════════
# FEASIBILITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════

def _check_feasibility(donor_set: DonorSet) -> list[str]:
    """Check chemical feasibility of a donor set. Returns list of issues."""
    issues = []

    # Check max donor counts
    from collections import Counter
    counts = Counter(donor_set.donor_subtypes)
    for subtype, count in counts.items():
        max_c = MAX_DONOR_COUNT.get(subtype, 4)
        if count > max_c:
            issues.append(f"Too many {subtype}: {count} > max {max_c}")

    # Check chelate ring plausibility
    if donor_set.chelate_rings > 0:
        n = donor_set.cn
        max_rings = n - 1  # Can't have more rings than donors - 1
        if donor_set.chelate_rings > max_rings:
            issues.append(f"Too many chelate rings: {donor_set.chelate_rings} "
                          f"> max {max_rings} for CN={n}")

    return issues


def _estimate_chelate_rings(donor_subtypes: list[str]) -> int:
    """Estimate maximum chelate rings from donor subtype combination."""
    n = len(donor_subtypes)
    if n < 2:
        return 0

    # Count how many adjacent pairs can form chelate rings
    rings = 0
    used = [False] * n
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if used[j]:
                continue
            pair = (donor_subtypes[i], donor_subtypes[j])
            pair_rev = (donor_subtypes[j], donor_subtypes[i])
            if pair in CHELATE_PAIRS or pair_rev in CHELATE_PAIRS:
                rings += 1
                used[i] = True
                used[j] = True
                break

    return rings


def _estimate_denticity(donor_subtypes: list[str], chelate_rings: int) -> int:
    """Estimate ligand denticity from donor count and chelate rings."""
    # If all donors in one chelate system: denticity = CN
    # If separate molecules: denticity = 1 or 2
    n = len(donor_subtypes)
    if chelate_rings >= n - 1:
        return n  # Fully chelated = one ligand
    elif chelate_rings > 0:
        return min(n, chelate_rings + 1)
    else:
        return 1  # No chelation = monodentate


# ═══════════════════════════════════════════════════════════════════════════
# ENUMERATOR
# ═══════════════════════════════════════════════════════════════════════════

# Core donor subtypes to enumerate from (excludes exotic/rare ones)
CORE_SUBTYPES = [
    "O_carboxylate", "O_phenolate", "O_hydroxamate", "O_catecholate",
    "O_ether", "O_hydroxyl",
    "N_amine", "N_imine", "N_pyridine", "N_imidazole",
    "S_thiolate", "S_thioether", "S_dithiocarbamate",
    "P_phosphine",
]

# ── HSAB-stratified donor pools ──────────────────────────────────────────
# Selected by HSAB softness for targeted enumeration.
# Soft pool: used when target metal has hsab_softness > 0.60
# Hard pool: used when target metal has hsab_softness < 0.25
# Borderline pool: 0.25–0.60 (augments CORE_SUBTYPES)
# Sources: noncarbonbias_donors.py; Pearson 1988; NIST SRD 46

SOFT_METAL_SUBTYPES = [
    # Strong soft donors — chalcogenides and pnictogens
    "S_thiolate", "S_dithiocarbamate", "S_thioether", "S_sulfoxide",
    "Se_selenolate", "Se_selenoether",
    "Te_tellurolate",                        # Extreme soft: Hg, Au only
    "P_phosphine", "P_phosphite",
    "As_arsine",
    "I_iodide", "Br_bromide",
    "C_cyanide",                             # C-end: Fe2+, Au+, Pd2+
    "N_imine", "N_aromatic",                 # Moderate soft N donors
]

HARD_METAL_SUBTYPES = [
    # Strong hard donors — oxyanions, fluoride, hard N
    "O_carboxylate", "O_phenolate", "O_hydroxamate", "O_catecholate",
    "O_phosphonate", "O_hydroxyl",
    "F_fluoride",                            # Al3+, Th4+, Zr4+, Be2+
    "N_amine", "N_imidazole",
]

BORDERLINE_SUBTYPES = [
    # N/S/P that work across the borderline zone (Fe2+/3+, Co2+, Ni2+, Cu2+)
    "N_amine", "N_imine", "N_pyridine", "N_imidazole",
    "S_thiolate", "S_dithiocarbamate",
    "O_carboxylate", "O_phenolate",
    "P_phosphine",
]

# Donor subtype → element (first token before "_")
def _donor_element(subtype: str) -> str:
    return subtype.split("_")[0] if "_" in subtype else subtype


def _get_subtypes_for_metal(
    metal_formula: str,
    allowed_subtypes=None,
    ph: float = 7.0,
) -> list:
    """Return HSAB-appropriate donor subtypes for a metal.

    If allowed_subtypes given, use those (filtered by pH).
    Otherwise pick from HSAB-stratified pools.
    """
    from core.scorer_frozen import METAL_DB
    metal = METAL_DB.get(metal_formula)
    if metal is None:
        return CORE_SUBTYPES

    softness = metal.hsab_softness

    if allowed_subtypes:
        pool = allowed_subtypes
    elif softness > 0.60:
        pool = list(set(SOFT_METAL_SUBTYPES + CORE_SUBTYPES))
    elif softness < 0.25:
        pool = list(set(HARD_METAL_SUBTYPES + CORE_SUBTYPES))
    else:
        pool = list(set(BORDERLINE_SUBTYPES + CORE_SUBTYPES))

    return pool


def _ph_available(subtype: str, pH: float) -> bool:
    """Is this donor subtype significantly available at the given pH?"""
    pka = DONOR_PKA.get(subtype, 99.0)
    if pka >= 50:
        return True  # Non-protonable donor (ethers, thioethers, phosphines)
    if pka <= 0:
        return True  # Always deprotonated (strong acids)
    # Available if pH is within 2 units of pKa (at least 1% deprotonated)
    return pH >= pka - 2.0


def enumerate_donor_sets(
    metal_formula: str,
    pH: float = 7.0,
    max_candidates: int = 500,
    include_archetypes: bool = True,
    include_combinatorial: bool = True,
    allowed_subtypes: Optional[list[str]] = None,
) -> list[DonorSet]:
    """Enumerate chemically feasible donor sets for a target metal.

    Args:
        metal_formula: Target metal (must be in METAL_DB)
        pH: Operating pH (filters out protonated donors)
        max_candidates: Maximum number of candidates to return
        include_archetypes: Include pre-defined ligand archetypes
        include_combinatorial: Include combinatorially generated sets
        allowed_subtypes: Restrict to these subtypes (None = all core)

    Returns:
        List of DonorSet candidates, sorted by estimated binding strength
    """
    metal = METAL_DB.get(metal_formula)
    if metal is None:
        raise ValueError(f"Unknown metal: {metal_formula}")

    cn_min, cn_max = metal.cn_range
    candidates = {}  # signature → DonorSet (dedup)

    # Filter subtypes by pH availability
    subtypes = _get_subtypes_for_metal(metal_formula, allowed_subtypes, pH)
    available = [s for s in subtypes if _ph_available(s, pH)]
    if not available:
        return []

    # ── Phase 1: Archetypes ──────────────────────────────────────────
    if include_archetypes:
        for arch in ARCHETYPES:
            # Check CN compatibility
            if not (cn_min <= arch.cn <= cn_max):
                continue
            # Check all donors are pH-available
            if not all(_ph_available(s, pH) for s in arch.donor_subtypes):
                continue
            # Feasibility check
            issues = _check_feasibility(arch)
            if issues:
                continue
            sig = arch.signature
            if sig not in candidates:
                candidates[sig] = arch

    # ── Phase 2: Combinatorial enumeration ───────────────────────────
    if include_combinatorial:
        for cn in range(cn_min, min(cn_max + 1, 9)):  # Cap at 8 to limit combinatorics
            for combo in combinations_with_replacement(available, cn):
                subtypes_list = list(combo)

                # Quick feasibility: check max counts
                from collections import Counter
                counts = Counter(subtypes_list)
                skip = False
                for st, c in counts.items():
                    if c > MAX_DONOR_COUNT.get(st, 4):
                        skip = True
                        break
                if skip:
                    continue

                # Estimate chelate rings and denticity
                rings = _estimate_chelate_rings(subtypes_list)
                dent = _estimate_denticity(subtypes_list, rings)
                n_lig = max(1, cn - rings) if rings > 0 else cn

                ds = DonorSet(
                    donor_subtypes=subtypes_list,
                    chelate_rings=rings,
                    denticity=dent,
                    is_macrocyclic=False,
                    ring_sizes=[5] * rings if rings > 0 else [],
                    n_ligand_molecules=n_lig,
                )

                sig = ds.signature
                if sig not in candidates:
                    # Prefer higher chelation (fewer ligand molecules)
                    candidates[sig] = ds

                if len(candidates) >= max_candidates * 3:  # Allow 3× for post-filtering
                    break
            if len(candidates) >= max_candidates * 3:
                break

    # ── Phase 3: Sort by estimated binding quality ───────────────────
    # Quick heuristic: sum of exchange energies + chelate bonus
    def _score_estimate(ds: DonorSet) -> float:
        score = sum(SUBTYPE_EXCHANGE.get(s, -5.0) for s in ds.donor_subtypes)
        score += -10.0 * ds.chelate_rings  # Chelate bonus
        if ds.is_macrocyclic:
            score += -10.0
        return score  # More negative = stronger binding

    result = sorted(candidates.values(), key=_score_estimate)
    return result[:max_candidates]


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Donor Enumerator — Self-Test ===\n")

    # Pb2+ at pH 5
    candidates = enumerate_donor_sets("Pb2+", pH=5.0, max_candidates=20)
    print(f"Pb2+ (pH 5): {len(candidates)} candidates")
    for i, ds in enumerate(candidates[:10]):
        print(f"  {i+1:2d}. CN={ds.cn} {ds.donor_subtypes} "
              f"rings={ds.chelate_rings} arch='{ds.archetype}'")

    print()

    # Hg2+ at pH 7 (soft metal — should favor S donors)
    candidates = enumerate_donor_sets("Hg2+", pH=7.0, max_candidates=20)
    print(f"Hg2+ (pH 7): {len(candidates)} candidates")
    for i, ds in enumerate(candidates[:10]):
        print(f"  {i+1:2d}. CN={ds.cn} {ds.donor_subtypes} "
              f"rings={ds.chelate_rings} arch='{ds.archetype}'")

    print()

    # K+ at pH 7 (crown ether target)
    candidates = enumerate_donor_sets("K+", pH=7.0, max_candidates=10)
    print(f"K+ (pH 7): {len(candidates)} candidates")
    for i, ds in enumerate(candidates[:10]):
        print(f"  {i+1:2d}. CN={ds.cn} {ds.donor_subtypes} "
              f"rings={ds.chelate_rings} macro={ds.is_macrocyclic} "
              f"arch='{ds.archetype}'")

    print("\nSelf-test complete.")
    # ═══════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY SHIM (F0 — remove after generative_integration refactor)
# ═══════════════════════════════════════════════════════════════════════════
def enumerate_donor_arrangements(coord_env, working_ph=7.0, max_candidates=4, **kwargs):
    """Deprecated: shim mapping old API to enumerate_donor_sets.

    coord_env is expected to have .target_formula.
    """
    metal = getattr(coord_env, 'target_formula', '??')
    return enumerate_donor_sets(metal, pH=working_ph, max_candidates=max_candidates)