"""
design_engine.py — MABE Generative Design Engine.

Orchestrates the full design pipeline:
  1. Enumerate donor sets for target metal
  2. Score each candidate for target AND all interferents
  3. Rank by selectivity gap (target log K - worst interferent log K)
  4. Apply feasibility filters
  5. Return ranked candidates with full scoring breakdown

Usage:
    from design_engine import design_binder, DesignResult
    result = design_binder(
        target="Pb2+",
        interferents=["Ca2+", "Mg2+", "Fe3+"],
        pH=5.0,
        top_n=10
    )
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from scorer_frozen import (
    predict_log_k, predict_selectivity, METAL_DB, MetalProperties
)
from donor_enumerator import (
    enumerate_donor_sets, DonorSet, ARCHETYPES
)


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateScore:
    """Scoring result for a single donor set candidate."""
    donor_set: DonorSet
    target_log_k: float
    interferent_log_ks: dict          # metal → log K
    selectivity_gaps: dict            # metal → (target_log_k - interferent_log_k)
    min_gap: float                    # Worst-case selectivity gap
    worst_interferent: str            # Metal with smallest selectivity gap
    selectivity_grade: str            # A/B/C/D/F based on min_gap
    rank: int = 0

    @property
    def summary(self) -> str:
        donors = ", ".join(self.donor_set.donor_subtypes)
        return (f"[{self.donor_set.archetype or 'custom'}] "
                f"CN={self.donor_set.cn} ({donors}) "
                f"log K={self.target_log_k:.1f} "
                f"gap={self.min_gap:+.1f} vs {self.worst_interferent} "
                f"grade={self.selectivity_grade}")


@dataclass
class DesignResult:
    """Complete output of a design run."""
    target_metal: str
    interferents: list[str]
    pH: float
    n_enumerated: int
    n_scored: int
    candidates: list[CandidateScore]
    elapsed_seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[CandidateScore]:
        return self.candidates[0] if self.candidates else None


# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════

def _grade_selectivity(min_gap: float) -> str:
    """Grade selectivity based on worst-case gap.

    Gap is in log K units: positive = selective for target.
    """
    if min_gap >= 5.0:
        return "A"   # Excellent: >100,000× selectivity
    elif min_gap >= 3.0:
        return "B"   # Good: >1,000× selectivity
    elif min_gap >= 1.0:
        return "C"   # Moderate: >10× selectivity
    elif min_gap >= 0.0:
        return "D"   # Marginal: slight preference
    else:
        return "F"   # Poor: interferent binds more strongly


# ═══════════════════════════════════════════════════════════════════════════
# METAL PARSING
# ═══════════════════════════════════════════════════════════════════════════

# Common name → formula mapping
METAL_ALIASES = {
    # Names
    "lead": "Pb2+", "pb": "Pb2+", "pb2+": "Pb2+", "pb(ii)": "Pb2+",
    "copper": "Cu2+", "cu": "Cu2+", "cu2+": "Cu2+", "cu(ii)": "Cu2+",
    "copper(i)": "Cu+", "cu+": "Cu+", "cu(i)": "Cu+",
    "zinc": "Zn2+", "zn": "Zn2+", "zn2+": "Zn2+", "zn(ii)": "Zn2+",
    "nickel": "Ni2+", "ni": "Ni2+", "ni2+": "Ni2+", "ni(ii)": "Ni2+",
    "iron": "Fe3+", "fe": "Fe3+", "fe3+": "Fe3+", "fe(iii)": "Fe3+",
    "iron(ii)": "Fe2+", "fe2+": "Fe2+", "fe(ii)": "Fe2+",
    "calcium": "Ca2+", "ca": "Ca2+", "ca2+": "Ca2+", "ca(ii)": "Ca2+",
    "magnesium": "Mg2+", "mg": "Mg2+", "mg2+": "Mg2+", "mg(ii)": "Mg2+",
    "manganese": "Mn2+", "mn": "Mn2+", "mn2+": "Mn2+", "mn(ii)": "Mn2+",
    "mercury": "Hg2+", "hg": "Hg2+", "hg2+": "Hg2+", "hg(ii)": "Hg2+",
    "cadmium": "Cd2+", "cd": "Cd2+", "cd2+": "Cd2+", "cd(ii)": "Cd2+",
    "silver": "Ag+", "ag": "Ag+", "ag+": "Ag+", "ag(i)": "Ag+",
    "gold": "Au3+", "au": "Au3+", "au3+": "Au3+", "au(iii)": "Au3+",
    "gold(i)": "Au+", "au+": "Au+", "au(i)": "Au+",
    "chromium": "Cr3+", "cr": "Cr3+", "cr3+": "Cr3+", "cr(iii)": "Cr3+",
    "cobalt": "Co2+", "co": "Co2+", "co2+": "Co2+", "co(ii)": "Co2+",
    "aluminum": "Al3+", "al": "Al3+", "al3+": "Al3+", "al(iii)": "Al3+",
    "aluminium": "Al3+",
    "platinum": "Pt2+", "pt": "Pt2+", "pt2+": "Pt2+", "pt(ii)": "Pt2+",
    "palladium": "Pd2+", "pd": "Pd2+", "pd2+": "Pd2+", "pd(ii)": "Pd2+",
    "uranium": "UO2_2+", "uranyl": "UO2_2+", "uo2": "UO2_2+",
    "gadolinium": "Gd3+", "gd": "Gd3+", "gd3+": "Gd3+",
    "cerium": "Ce3+", "ce": "Ce3+", "ce3+": "Ce3+",
    "neodymium": "Nd3+", "nd": "Nd3+", "nd3+": "Nd3+",
    "sodium": "Na+", "na": "Na+", "na+": "Na+",
    "potassium": "K+", "k": "K+", "k+": "K+",
    "barium": "Ba2+", "ba": "Ba2+", "ba2+": "Ba2+",
    "strontium": "Sr2+", "sr": "Sr2+", "sr2+": "Sr2+",
    "lithium": "Li+", "li": "Li+", "li+": "Li+",
    "tin": "Sn2+", "sn": "Sn2+", "sn2+": "Sn2+",
    "bismuth": "Bi3+", "bi": "Bi3+", "bi3+": "Bi3+",
    "thallium": "Tl+", "tl": "Tl+", "tl+": "Tl+",
    "indium": "In3+", "in": "In3+", "in3+": "In3+",
}


def resolve_metal(name: str) -> str:
    """Resolve a metal name/formula to canonical METAL_DB key."""
    # Try direct match first
    if name in METAL_DB:
        return name
    # Try alias
    key = name.lower().strip()
    if key in METAL_ALIASES:
        return METAL_ALIASES[key]
    # Try with common charge suffixes
    raise ValueError(f"Unknown metal: '{name}'. Try formula like 'Pb2+' or "
                     f"name like 'lead'. Available: {sorted(METAL_DB.keys())}")


# ═══════════════════════════════════════════════════════════════════════════
# MATRIX PRESETS
# ═══════════════════════════════════════════════════════════════════════════

# Common interferent sets for different water matrices
MATRIX_PRESETS = {
    "mine_water": ["Ca2+", "Mg2+", "Fe3+", "Mn2+", "Zn2+", "Cu2+"],
    "mine_amd": ["Ca2+", "Mg2+", "Fe3+", "Fe2+", "Mn2+", "Zn2+",
                 "Cu2+", "Ni2+", "Al3+"],
    "drinking_water": ["Ca2+", "Mg2+", "Na+", "K+", "Fe2+", "Mn2+"],
    "seawater": ["Na+", "Mg2+", "Ca2+", "K+", "Sr2+"],
    "blood": ["Ca2+", "Mg2+", "Na+", "K+", "Fe2+", "Zn2+", "Cu2+"],
    "industrial": ["Ca2+", "Mg2+", "Fe3+", "Cu2+", "Zn2+", "Ni2+",
                   "Cr3+", "Cd2+"],
}


def parse_matrix(matrix_str: str) -> tuple:
    """Parse matrix specification string into (pH, interferents).

    Examples:
        "pH 5, Ca, Mg, Fe" → (5.0, ["Ca2+", "Mg2+", "Fe3+"])
        "mine_water pH 4.5" → (4.5, [...mine_water interferents...])
        "pH 7" → (7.0, [])
    """
    parts = [p.strip() for p in matrix_str.replace(",", " ").split()]

    pH = 7.0
    interferents = []

    i = 0
    while i < len(parts):
        p = parts[i].lower()
        if p == "ph" and i + 1 < len(parts):
            try:
                pH = float(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif p.startswith("ph") and len(p) > 2:
            try:
                pH = float(p[2:])
                i += 1
                continue
            except ValueError:
                pass

        # Check for matrix preset
        if p in MATRIX_PRESETS:
            interferents.extend(MATRIX_PRESETS[p])
            i += 1
            continue

        # Try to resolve as metal
        try:
            metal = resolve_metal(p)
            if metal not in interferents:
                interferents.append(metal)
        except ValueError:
            pass  # Skip unknown tokens
        i += 1

    return pH, interferents


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DESIGN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def design_binder(
    target: str,
    interferents: Optional[list[str]] = None,
    matrix: Optional[str] = None,
    pH: float = 7.0,
    top_n: int = 10,
    max_enumerate: int = 500,
    min_target_log_k: float = -999.0,
    allowed_subtypes: Optional[list[str]] = None,
) -> DesignResult:
    """Design selective binders for a target metal.

    Args:
        target: Target metal (name or formula)
        interferents: List of interferent metals (names or formulas)
        matrix: Matrix string (e.g., "pH 5, Ca, Mg, Fe" or "mine_water")
        pH: Operating pH (overridden by matrix string if present)
        top_n: Number of top candidates to return
        max_enumerate: Maximum candidates to enumerate
        min_target_log_k: Minimum target log K to include
        allowed_subtypes: Restrict donor subtypes

    Returns:
        DesignResult with ranked candidates
    """
    t0 = time.time()
    notes = []

    # ── Resolve target ───────────────────────────────────────────────
    target_formula = resolve_metal(target)
    target_metal = METAL_DB[target_formula]

    # ── Parse matrix / interferents ──────────────────────────────────
    if matrix:
        mat_pH, mat_interferents = parse_matrix(matrix)
        pH = mat_pH
        intf_formulas = mat_interferents
    elif interferents:
        intf_formulas = [resolve_metal(m) for m in interferents]
    else:
        intf_formulas = []

    # Remove target from interferents if accidentally included
    intf_formulas = [m for m in intf_formulas if m != target_formula]

    notes.append(f"Target: {target_formula} ({target_metal.name})")
    notes.append(f"pH: {pH}")
    notes.append(f"Interferents: {', '.join(intf_formulas) or 'none'}")

    # ── Enumerate donor sets ─────────────────────────────────────────
    candidates = enumerate_donor_sets(
        target_formula, pH=pH,
        max_candidates=max_enumerate,
        allowed_subtypes=allowed_subtypes,
    )
    n_enumerated = len(candidates)
    notes.append(f"Enumerated: {n_enumerated} candidate donor sets")

    # ── Score all candidates ─────────────────────────────────────────
    scored = []
    for ds in candidates:
        try:
            result = predict_selectivity(
                target_formula, intf_formulas,
                ds.donor_subtypes,
                chelate_rings=ds.chelate_rings,
                ring_sizes=ds.ring_sizes if ds.ring_sizes else None,
                pH=pH,
                is_macrocyclic=ds.is_macrocyclic,
                cavity_radius_nm=ds.cavity_radius_nm,
                n_ligand_molecules=ds.n_ligand_molecules,
            )

            target_lk = result["target_log_k"]

            # Filter by minimum target binding
            if target_lk < min_target_log_k:
                continue

            grade = _grade_selectivity(result["min_gap"])

            scored.append(CandidateScore(
                donor_set=ds,
                target_log_k=target_lk,
                interferent_log_ks=result["interferent_log_ks"],
                selectivity_gaps=result["selectivity_gaps"],
                min_gap=result["min_gap"],
                worst_interferent=result["worst_interferent"] or "none",
                selectivity_grade=grade,
            ))
        except Exception as e:
            notes.append(f"Scoring failed for {ds}: {e}")

    # ── Rank by selectivity ──────────────────────────────────────────
    # Primary: min_gap (higher = more selective)
    # Secondary: target_log_k (higher = stronger binding)
    scored.sort(key=lambda c: (c.min_gap, c.target_log_k), reverse=True)

    # Assign ranks
    for i, c in enumerate(scored):
        c.rank = i + 1

    elapsed = time.time() - t0
    notes.append(f"Scored: {len(scored)} candidates in {elapsed:.1f}s")

    return DesignResult(
        target_metal=target_formula,
        interferents=intf_formulas,
        pH=pH,
        n_enumerated=n_enumerated,
        n_scored=len(scored),
        candidates=scored[:top_n],
        elapsed_seconds=elapsed,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_design_result(result: DesignResult, verbose: bool = False):
    """Pretty-print design results."""
    print()
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  MABE GENERATIVE DESIGN ENGINE — Results                   ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Target:       {result.target_metal:44s} ║")
    intf_str = ", ".join(result.interferents) or "none"
    if len(intf_str) > 44:
        intf_str = intf_str[:41] + "..."
    print(f"  ║  Interferents: {intf_str:44s} ║")
    print(f"  ║  pH:           {result.pH:<44.1f} ║")
    print(f"  ║  Enumerated:   {result.n_enumerated:<44d} ║")
    print(f"  ║  Scored:       {result.n_scored:<44d} ║")
    print(f"  ║  Time:         {result.elapsed_seconds:<44.1f} ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    print()

    if not result.candidates:
        print("  No viable candidates found.")
        return

    # Header
    print(f"  {'#':>3s}  {'Grade':5s}  {'log K':>6s}  {'Min Gap':>8s}  "
          f"{'Worst':>6s}  {'CN':>3s}  Donor Set")
    print(f"  {'─'*80}")

    for c in result.candidates:
        donors_short = _compact_donors(c.donor_set.donor_subtypes)
        arch = f" [{c.donor_set.archetype}]" if c.donor_set.archetype else ""
        print(f"  {c.rank:3d}  {c.selectivity_grade:^5s}  "
              f"{c.target_log_k:+6.1f}  {c.min_gap:+8.1f}  "
              f"{c.worst_interferent:>6s}  {c.donor_set.cn:3d}  "
              f"{donors_short}{arch}")

    if verbose and result.candidates:
        print()
        print(f"  ── Top Candidate Detail ──")
        best = result.candidates[0]
        print(f"  Donors: {best.donor_set.donor_subtypes}")
        print(f"  Chelate rings: {best.donor_set.chelate_rings}")
        print(f"  Macrocyclic: {best.donor_set.is_macrocyclic}")
        print(f"  Ligand molecules: {best.donor_set.n_ligand_molecules}")
        print(f"  Target {result.target_metal}: log K = {best.target_log_k:.2f}")
        for m in result.interferents:
            lk = best.interferent_log_ks.get(m, 0)
            gap = best.selectivity_gaps.get(m, 0)
            print(f"  vs {m:6s}: log K = {lk:+.2f}  "
                  f"gap = {gap:+.2f} ({10**gap:.0f}×)")

    print()


def _compact_donors(subtypes: list[str]) -> str:
    """Compact donor list for display: N_amine×2, O_carboxylate×4."""
    from collections import Counter
    counts = Counter(subtypes)
    parts = []
    for st, n in sorted(counts.items()):
        short = st.split("_")[1] if "_" in st else st
        el = st.split("_")[0]
        if n == 1:
            parts.append(f"{el}-{short}")
        else:
            parts.append(f"{el}-{short}×{n}")
    return " + ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== MABE Design Engine — Self-Test ===")

    # Test 1: Pb2+ from pH 5 mine water
    result = design_binder(
        target="Pb(II)",
        matrix="pH 5, Ca, Mg, Fe, Zn, Cu, Mn",
        top_n=10,
        max_enumerate=200,
    )
    print_design_result(result, verbose=True)

    # Test 2: Hg2+ selective over Pb, Cd, Zn
    result = design_binder(
        target="Hg2+",
        interferents=["Pb2+", "Cd2+", "Zn2+", "Cu2+"],
        pH=7.0,
        top_n=5,
    )
    print_design_result(result)

    # Test 3: Au3+ from industrial water
    result = design_binder(
        target="Au(III)",
        matrix="industrial pH 2",
        top_n=5,
    )
    print_design_result(result)