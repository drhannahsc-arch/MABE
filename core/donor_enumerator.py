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

from scorer_frozen import METAL_DB, DONOR_SOFTNESS, DONOR_PKA, SUBTYPE_EXCHANGE


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
    subtypes = allowed_subtypes or CORE_SUBTYPES
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