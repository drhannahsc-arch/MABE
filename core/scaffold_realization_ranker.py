"""
core/scaffold_realization_ranker.py — Scaffold Selection for Immune Display

Scores binder candidates against material scaffold options for
multivalent immune-engaging construct assembly.

Scaffolds scored on 7 criteria:
  1. Spacing fidelity — can this scaffold achieve target inter-binder spacing?
  2. Attachment compatibility — binder handle chemistry vs scaffold conjugation
  3. Valency match — can scaffold display optimal copy number?
  4. Immune accessibility — are binders accessible to BCR/TCR/DC receptors?
  5. Manufacturability — synthetic complexity, scale, reproducibility
  6. Biocompatibility — scaffold immunogenicity, toxicity, clearance
  7. Cost — materials + assembly + purification per dose

Informed by:
  Veneziano et al. 2020, Nat. Nanotech. — BCR activation peaks at ≥5 copies, ~22 nm
  Shaw et al. 2014 — EphA2 receptor clustering: 40 nm > 100 nm
  DoriVac (Shih lab) — DBCO-azide click on DNA origami square block

No fitted parameters. Scoring is rule-based from published design constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# CONJUGATION CHEMISTRY
# ═══════════════════════════════════════════════════════════════════════════

class ConjugationChemistry(Enum):
    """Available conjugation strategies."""
    DBCO_AZIDE = "DBCO-azide click"         # strain-promoted, no catalyst
    CUTHP_CLICK = "CuAAC click"             # copper-catalyzed alkyne-azide
    NHS_AMINE = "NHS-amine"                 # lysine/amine coupling
    MALEIMIDE_THIOL = "maleimide-thiol"     # cysteine/thiol coupling
    EDC_CARBOXYL = "EDC/NHS carboxyl"       # carbodiimide
    TETRAZINE_TCO = "tetrazine-TCO"         # inverse electron demand DA
    SILANE = "silane functionalization"       # for silica surfaces
    PSM_CLICK = "post-synthetic modification" # for MOF linkers


# ═══════════════════════════════════════════════════════════════════════════
# SCAFFOLD DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldSpec:
    """Complete specification of a display scaffold."""
    name: str
    category: str                           # 'origami', 'cage', 'dendrimer', etc.
    # Geometry
    min_spacing_nm: float = 0.0             # minimum achievable inter-site distance
    max_spacing_nm: float = 0.0             # maximum achievable
    spacing_programmable: bool = False       # can spacing be tuned continuously?
    min_valency: int = 1
    max_valency: int = 1
    particle_diameter_nm: float = 0.0       # overall construct size
    # Chemistry
    conjugation_options: List[ConjugationChemistry] = field(default_factory=list)
    co_display_possible: bool = False        # can carry adjuvant + binder on same particle?
    # Manufacturability
    synthetic_steps: int = 1                 # approximate steps to assemble
    scale_kg_feasible: bool = False          # can produce at >1 kg scale?
    storage_stability_days: float = 1.0      # RT shelf life estimate
    batch_reproducibility: float = 0.5       # 0-1, how reproducible
    # Biocompatibility
    scaffold_immunogenicity: float = 0.5     # 0=inert, 1=highly immunogenic
    biodegradable: bool = False
    clearance_route: str = ""                # renal, hepatic, RES
    fda_precedent: bool = False
    # Cost
    est_cost_per_mg_usd: float = 100.0       # scaffold material cost
    # References
    key_reference: str = ""


SCAFFOLD_DATABASE = [
    # ── DNA Origami ──────────────────────────────────────────────────
    ScaffoldSpec(
        name="DNA origami icosahedron",
        category="origami",
        min_spacing_nm=3.5, max_spacing_nm=80.0, spacing_programmable=True,
        min_valency=1, max_valency=60,
        particle_diameter_nm=60.0,
        conjugation_options=[ConjugationChemistry.DBCO_AZIDE,
                             ConjugationChemistry.NHS_AMINE],
        co_display_possible=True,
        synthetic_steps=3,  # staple anneal + purification + conjugation
        scale_kg_feasible=False,
        storage_stability_days=30,  # -20C lyophilized
        batch_reproducibility=0.9,
        scaffold_immunogenicity=0.05,  # DoriVac: zero scaffold-specific Ab
        biodegradable=True,
        clearance_route="nuclease degradation → renal",
        fda_precedent=False,
        est_cost_per_mg_usd=500.0,
        key_reference="Veneziano 2020 Nat.Nanotech; DoriVac 2024 Nat.Biomed.Eng",
    ),
    ScaffoldSpec(
        name="DNA origami square block (DoriVac)",
        category="origami",
        min_spacing_nm=3.5, max_spacing_nm=20.0, spacing_programmable=True,
        min_valency=1, max_valency=36,
        particle_diameter_nm=30.0,
        conjugation_options=[ConjugationChemistry.DBCO_AZIDE],
        co_display_possible=True,  # antigen one face, CpG other face
        synthetic_steps=3,
        scale_kg_feasible=False,
        storage_stability_days=90,  # DoriVac stability data
        batch_reproducibility=0.9,
        scaffold_immunogenicity=0.05,
        biodegradable=True,
        clearance_route="nuclease degradation → renal",
        fda_precedent=False,
        est_cost_per_mg_usd=500.0,
        key_reference="DoriVac 2024/2026",
    ),

    # ── Coordination Cages ───────────────────────────────────────────
    ScaffoldSpec(
        name="Pd₁₂L₂₄ Fujita cage",
        category="coordination_cage",
        min_spacing_nm=1.5, max_spacing_nm=4.0, spacing_programmable=False,
        min_valency=12, max_valency=24,
        particle_diameter_nm=5.0,
        conjugation_options=[ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.CUTHP_CLICK],
        co_display_possible=True,  # exo + endo functionalization
        synthetic_steps=2,  # self-assembly + functionalization
        scale_kg_feasible=False,
        storage_stability_days=180,
        batch_reproducibility=0.7,
        scaffold_immunogenicity=0.3,  # Pd toxicity concern
        biodegradable=False,
        clearance_route="RES uptake",
        fda_precedent=False,
        est_cost_per_mg_usd=200.0,
        key_reference="Fujita group, multiple JACS/Nature",
    ),
    ScaffoldSpec(
        name="Fe₄L₆ Nitschke cage",
        category="coordination_cage",
        min_spacing_nm=1.0, max_spacing_nm=2.5, spacing_programmable=False,
        min_valency=4, max_valency=6,
        particle_diameter_nm=2.5,
        conjugation_options=[ConjugationChemistry.NHS_AMINE],
        co_display_possible=False,  # too small for dual display
        synthetic_steps=2,
        scale_kg_feasible=True,  # subcomponent self-assembly is scalable
        storage_stability_days=365,
        batch_reproducibility=0.8,
        scaffold_immunogenicity=0.2,  # Fe is biocompatible
        biodegradable=True,
        clearance_route="Fe recycling + renal",
        fda_precedent=False,
        est_cost_per_mg_usd=50.0,
        key_reference="Nitschke group, Nature Chem / Acc. Chem. Res.",
    ),

    # ── Dendrimers ───────────────────────────────────────────────────
    ScaffoldSpec(
        name="PAMAM G5 dendrimer",
        category="dendrimer",
        min_spacing_nm=1.5, max_spacing_nm=3.0, spacing_programmable=False,
        min_valency=32, max_valency=128,
        particle_diameter_nm=5.4,
        conjugation_options=[ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.MALEIMIDE_THIOL,
                             ConjugationChemistry.CUTHP_CLICK],
        co_display_possible=True,
        synthetic_steps=1,  # commercial + conjugation
        scale_kg_feasible=True,
        storage_stability_days=365,
        batch_reproducibility=0.7,  # PDI ~1.02-1.05 for high-G
        scaffold_immunogenicity=0.15,
        biodegradable=False,
        clearance_route="renal (G4-5) / RES (G6+)",
        fda_precedent=True,  # Vivagel (SPL7013) approved
        est_cost_per_mg_usd=5.0,
        key_reference="Chauhan 2020; Vivagel",
    ),
    ScaffoldSpec(
        name="PAMAM G7 dendrimer",
        category="dendrimer",
        min_spacing_nm=2.5, max_spacing_nm=4.0, spacing_programmable=False,
        min_valency=128, max_valency=512,
        particle_diameter_nm=8.1,
        conjugation_options=[ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.MALEIMIDE_THIOL],
        co_display_possible=True,
        synthetic_steps=1,
        scale_kg_feasible=True,
        storage_stability_days=365,
        batch_reproducibility=0.5,  # high-G dendrimers have defects
        scaffold_immunogenicity=0.2,
        biodegradable=False,
        clearance_route="RES uptake",
        fda_precedent=True,
        est_cost_per_mg_usd=15.0,
        key_reference="Tomalia; Chauhan 2020",
    ),

    # ── MOF Nanoparticles ────────────────────────────────────────────
    ScaffoldSpec(
        name="UiO-66-NH₂ nanoparticle",
        category="mof",
        min_spacing_nm=1.2, max_spacing_nm=1.2, spacing_programmable=False,
        min_valency=50, max_valency=500,  # surface sites
        particle_diameter_nm=100.0,
        conjugation_options=[ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.CUTHP_CLICK,
                             ConjugationChemistry.PSM_CLICK],
        co_display_possible=True,
        synthetic_steps=2,  # solvothermal + PSM
        scale_kg_feasible=True,
        storage_stability_days=365,
        batch_reproducibility=0.6,
        scaffold_immunogenicity=0.15,  # Zr is biocompatible
        biodegradable=True,  # phosphate buffer degrades UiO-66
        clearance_route="degradation → renal (Zr) + hepatic (linker)",
        fda_precedent=False,
        est_cost_per_mg_usd=1.0,
        key_reference="Horcajada 2012 Chem.Rev.",
    ),

    # ── Mesoporous Silica ────────────────────────────────────────────
    ScaffoldSpec(
        name="MCM-41 mesoporous silica NP",
        category="silica",
        min_spacing_nm=3.8, max_spacing_nm=3.8, spacing_programmable=False,
        min_valency=50, max_valency=500,
        particle_diameter_nm=100.0,
        conjugation_options=[ConjugationChemistry.SILANE,
                             ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.CUTHP_CLICK],
        co_display_possible=True,
        synthetic_steps=2,  # Stöber synthesis + functionalization
        scale_kg_feasible=True,
        storage_stability_days=365,
        batch_reproducibility=0.8,
        scaffold_immunogenicity=0.1,
        biodegradable=True,  # dissolves in biological media over weeks
        clearance_route="degradation → renal (silicic acid)",
        fda_precedent=True,  # Cornell dots (C-dots) FDA IND
        est_cost_per_mg_usd=0.5,
        key_reference="Lu 2009; Phillips 2014 (C-dots)",
    ),

    # ── Polymer NPs ──────────────────────────────────────────────────
    ScaffoldSpec(
        name="PLGA nanoparticle",
        category="polymer",
        min_spacing_nm=5.0, max_spacing_nm=20.0, spacing_programmable=False,
        min_valency=20, max_valency=200,
        particle_diameter_nm=150.0,
        conjugation_options=[ConjugationChemistry.EDC_CARBOXYL,
                             ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.MALEIMIDE_THIOL],
        co_display_possible=True,
        synthetic_steps=2,  # nanoprecipitation + conjugation
        scale_kg_feasible=True,
        storage_stability_days=180,
        batch_reproducibility=0.7,
        scaffold_immunogenicity=0.05,
        biodegradable=True,
        clearance_route="hydrolysis → renal (lactic/glycolic acid)",
        fda_precedent=True,  # multiple approved products
        est_cost_per_mg_usd=0.1,
        key_reference="Peer 2007; multiple FDA-approved PLGA products",
    ),

    # ── ROMP Polymer ─────────────────────────────────────────────────
    ScaffoldSpec(
        name="ROMP polynorbornene scaffold",
        category="polymer",
        min_spacing_nm=0.5, max_spacing_nm=5.0, spacing_programmable=True,
        min_valency=5, max_valency=100,
        particle_diameter_nm=10.0,  # extended chain
        conjugation_options=[ConjugationChemistry.NHS_AMINE,
                             ConjugationChemistry.CUTHP_CLICK],
        co_display_possible=True,
        synthetic_steps=2,  # ROMP + end-capping
        scale_kg_feasible=True,
        storage_stability_days=365,
        batch_reproducibility=0.85,  # living polymerization = tight PDI
        scaffold_immunogenicity=0.1,
        biodegradable=False,
        clearance_route="renal (low MW) / RES (high MW)",
        fda_precedent=False,
        est_cost_per_mg_usd=2.0,
        key_reference="Barz/Kiessling; Grubbs catalyst",
    ),
]

SCAFFOLD_BY_NAME = {s.name: s for s in SCAFFOLD_DATABASE}


# ═══════════════════════════════════════════════════════════════════════════
# BINDER-SCAFFOLD COMPATIBILITY SCORING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffoldScore:
    """Score for a specific binder-scaffold pairing."""
    scaffold_name: str
    category: str
    # Individual criteria (0-1, higher = better)
    spacing_fidelity: float = 0.0
    attachment_compat: float = 0.0
    valency_match: float = 0.0
    immune_accessibility: float = 0.0
    manufacturability: float = 0.0
    biocompatibility: float = 0.0
    cost_score: float = 0.0
    # Composite
    composite: float = 0.0
    # Notes
    rationale: str = ""
    conjugation_strategy: str = ""
    predicted_valency: int = 0
    predicted_spacing_nm: float = 0.0


def _score_spacing(scaffold: ScaffoldSpec, target_min_nm: float = 5.0,
                    target_max_nm: float = 25.0) -> Tuple[float, float]:
    """Score spacing fidelity. Returns (score, predicted_spacing)."""
    # Veneziano optimal: 5-25 nm for BCR activation
    if scaffold.spacing_programmable:
        # Can hit any target in range
        if scaffold.min_spacing_nm <= target_min_nm and scaffold.max_spacing_nm >= target_max_nm:
            return 1.0, (target_min_nm + target_max_nm) / 2
        elif scaffold.max_spacing_nm >= target_min_nm:
            overlap = min(scaffold.max_spacing_nm, target_max_nm) - max(scaffold.min_spacing_nm, target_min_nm)
            target_range = target_max_nm - target_min_nm
            return max(0, overlap / target_range), min(scaffold.max_spacing_nm, target_max_nm)
        else:
            return 0.0, scaffold.max_spacing_nm
    else:
        # Fixed spacing — how close to optimal?
        mid_target = (target_min_nm + target_max_nm) / 2
        best_scaffold = (scaffold.min_spacing_nm + scaffold.max_spacing_nm) / 2
        distance = abs(best_scaffold - mid_target)
        half_range = (target_max_nm - target_min_nm) / 2
        if distance <= half_range:
            return 1.0 - (distance / half_range) * 0.5, best_scaffold
        else:
            return max(0, 0.5 - (distance - half_range) / 20.0), best_scaffold


def _score_attachment(scaffold: ScaffoldSpec, binder_mw: float,
                       binder_has_amine: bool, binder_has_aromatic: bool) -> Tuple[float, str]:
    """Score attachment compatibility. Returns (score, best_chemistry)."""
    # Check which conjugation chemistries the binder could support
    # All small molecules can have handles installed; score based on
    # number of available options and how clean the chemistry is
    n_options = len(scaffold.conjugation_options)
    if n_options == 0:
        return 0.0, "none"

    # DBCO-azide is best (no catalyst, bioorthogonal)
    if ConjugationChemistry.DBCO_AZIDE in scaffold.conjugation_options:
        return min(1.0, 0.7 + 0.1 * n_options), "DBCO-azide"
    # Tetrazine-TCO is also excellent
    elif ConjugationChemistry.TETRAZINE_TCO in scaffold.conjugation_options:
        return min(1.0, 0.65 + 0.1 * n_options), "tetrazine-TCO"
    # CuAAC requires catalyst (Cu toxicity concern)
    elif ConjugationChemistry.CUTHP_CLICK in scaffold.conjugation_options:
        return min(1.0, 0.5 + 0.1 * n_options), "CuAAC"
    # NHS/EDC are standard but less clean
    elif ConjugationChemistry.NHS_AMINE in scaffold.conjugation_options:
        return min(1.0, 0.4 + 0.1 * n_options), "NHS-amine"
    else:
        return 0.3, scaffold.conjugation_options[0].value


def _score_valency(scaffold: ScaffoldSpec, target_valency: int = 10) -> Tuple[float, int]:
    """Score valency match. Returns (score, predicted_valency)."""
    if scaffold.min_valency <= target_valency <= scaffold.max_valency:
        return 1.0, target_valency
    elif target_valency < scaffold.min_valency:
        # Scaffold forces higher valency than needed — partial loading possible
        return 0.7, scaffold.min_valency
    else:
        # Can't achieve target valency
        return max(0, 1.0 - (target_valency - scaffold.max_valency) / 50), scaffold.max_valency


def _score_immune_access(scaffold: ScaffoldSpec, binder_mw: float) -> float:
    """Score immune accessibility — are displayed binders reachable by immune receptors?"""
    # BCR is ~10 nm; binders must protrude from scaffold surface
    # Smaller scaffolds → less steric occlusion
    # DNA origami is best (binders on staple overhangs, fully exposed)
    if scaffold.category == "origami":
        return 0.95
    elif scaffold.category in ("dendrimer", "polymer"):
        # Surface binders are accessible but packed
        return 0.7 if binder_mw < 500 else 0.5
    elif scaffold.category in ("mof", "silica"):
        # Surface only; pore binders are inaccessible
        return 0.6
    elif scaffold.category == "coordination_cage":
        # Small cages — exo sites exposed, endo hidden
        return 0.5
    return 0.5


def _score_manufacturability(scaffold: ScaffoldSpec) -> float:
    """Score manufacturability (0-1)."""
    score = 0.0
    if scaffold.scale_kg_feasible:
        score += 0.3
    score += max(0, (1.0 - scaffold.synthetic_steps / 10.0)) * 0.3
    score += scaffold.batch_reproducibility * 0.2
    score += min(1.0, scaffold.storage_stability_days / 365.0) * 0.2
    return score


def _score_biocompat(scaffold: ScaffoldSpec) -> float:
    """Score biocompatibility (0-1)."""
    score = 0.0
    score += (1.0 - scaffold.scaffold_immunogenicity) * 0.3
    if scaffold.biodegradable:
        score += 0.25
    if scaffold.fda_precedent:
        score += 0.25
    if scaffold.clearance_route:
        score += 0.2
    return min(1.0, score)


def _score_cost(scaffold: ScaffoldSpec) -> float:
    """Score cost (0-1, higher = cheaper)."""
    # Log-scale: $0.1/mg = 1.0, $1000/mg = 0.0
    import math
    if scaffold.est_cost_per_mg_usd <= 0:
        return 0.5
    log_cost = math.log10(scaffold.est_cost_per_mg_usd)
    # Map: log10(0.1)=-1 → 1.0, log10(1000)=3 → 0.0
    return max(0.0, min(1.0, 1.0 - (log_cost + 1.0) / 4.0))


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def rank_scaffolds(
    binder_mw: float = 350.0,
    binder_n_hbd: int = 4,
    binder_n_aromatic: int = 1,
    target_valency: int = 10,
    target_spacing_min_nm: float = 5.0,
    target_spacing_max_nm: float = 25.0,
    weights: Optional[Dict[str, float]] = None,
) -> List[ScaffoldScore]:
    """
    Rank all scaffold options for a given binder profile.

    Args:
        binder_mw: molecular weight of the binder
        binder_n_hbd: H-bond donors on binder (for handle compatibility)
        binder_n_aromatic: aromatic rings on binder
        target_valency: desired copies per construct
        target_spacing_min_nm: minimum inter-binder distance
        target_spacing_max_nm: maximum inter-binder distance
        weights: optional {criterion: weight} override

    Returns:
        List[ScaffoldScore] sorted by composite (best first)
    """
    if weights is None:
        weights = {
            'spacing': 0.20,
            'attachment': 0.15,
            'valency': 0.10,
            'immune_access': 0.15,
            'manufacturability': 0.15,
            'biocompatibility': 0.15,
            'cost': 0.10,
        }

    results = []
    for scaffold in SCAFFOLD_DATABASE:
        ss = ScaffoldScore(
            scaffold_name=scaffold.name,
            category=scaffold.category,
        )

        # Score each criterion
        ss.spacing_fidelity, ss.predicted_spacing_nm = _score_spacing(
            scaffold, target_spacing_min_nm, target_spacing_max_nm)
        ss.attachment_compat, ss.conjugation_strategy = _score_attachment(
            scaffold, binder_mw, binder_n_hbd > 0, binder_n_aromatic > 0)
        ss.valency_match, ss.predicted_valency = _score_valency(
            scaffold, target_valency)
        ss.immune_accessibility = _score_immune_access(scaffold, binder_mw)
        ss.manufacturability = _score_manufacturability(scaffold)
        ss.biocompatibility = _score_biocompat(scaffold)
        ss.cost_score = _score_cost(scaffold)

        # Composite
        ss.composite = (
            weights['spacing'] * ss.spacing_fidelity +
            weights['attachment'] * ss.attachment_compat +
            weights['valency'] * ss.valency_match +
            weights['immune_access'] * ss.immune_accessibility +
            weights['manufacturability'] * ss.manufacturability +
            weights['biocompatibility'] * ss.biocompatibility +
            weights['cost'] * ss.cost_score
        )

        # Rationale
        strengths = []
        if ss.spacing_fidelity > 0.8:
            strengths.append("optimal spacing")
        if ss.attachment_compat > 0.7:
            strengths.append(f"clean conjugation ({ss.conjugation_strategy})")
        if ss.immune_accessibility > 0.8:
            strengths.append("high immune accessibility")
        if ss.biocompatibility > 0.8:
            strengths.append("strong biocompatibility")
        if ss.cost_score > 0.7:
            strengths.append("low cost")
        ss.rationale = "; ".join(strengths) if strengths else "no standout advantage"

        results.append(ss)

    results.sort(key=lambda s: -s.composite)
    return results


def rank_scaffolds_for_binder(binder_features) -> List[ScaffoldScore]:
    """Convenience: score scaffolds from a ReceptorFeatures object."""
    return rank_scaffolds(
        binder_mw=binder_features.mw,
        binder_n_hbd=binder_features.n_hbd,
        binder_n_aromatic=binder_features.n_aromatic_rings,
    )


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def scaffold_comparison_table(results: List[ScaffoldScore]) -> str:
    """Format scaffold ranking as a readable table."""
    lines = []
    header = (f"{'Rank':>4s} {'Scaffold':35s} {'Composite':>9s} "
              f"{'Space':>6s} {'Attach':>6s} {'Valen':>6s} "
              f"{'Immun':>6s} {'Manuf':>6s} {'Biocm':>6s} {'Cost':>6s} "
              f"  Conjugation")
    lines.append(header)
    lines.append("─" * len(header))
    for i, s in enumerate(results):
        lines.append(
            f"{i+1:4d} {s.scaffold_name:35s} {s.composite:9.3f} "
            f"{s.spacing_fidelity:6.2f} {s.attachment_compat:6.2f} "
            f"{s.valency_match:6.2f} {s.immune_accessibility:6.2f} "
            f"{s.manufacturability:6.2f} {s.biocompatibility:6.2f} "
            f"{s.cost_score:6.2f}   {s.conjugation_strategy}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 90)
    print("SCAFFOLD REALIZATION RANKING — GalNAc Binder (MW~350, 4 HBD, 1 aromatic)")
    print("=" * 90)
    print()

    results = rank_scaffolds(
        binder_mw=350, binder_n_hbd=4, binder_n_aromatic=1,
        target_valency=10,
        target_spacing_min_nm=5.0, target_spacing_max_nm=25.0,
    )

    print(scaffold_comparison_table(results))

    print()
    print("─── RATIONALE ───")
    for i, s in enumerate(results):
        print(f"  #{i+1} {s.scaffold_name}: {s.rationale}")

    # Repeat for different binder profile: larger boronic acid receptor
    print()
    print("=" * 90)
    print("SCAFFOLD RANKING — Large Boronic Acid Receptor (MW~500, 6 HBD, 2 aromatic)")
    print("=" * 90)
    print()
    results2 = rank_scaffolds(
        binder_mw=500, binder_n_hbd=6, binder_n_aromatic=2,
        target_valency=10,
    )
    print(scaffold_comparison_table(results2))