"""
glycan/pulldown_selector.py -- Cell type -> lectin -> sugar -> linker pipeline.

Given a target cell type and bead diameter, returns ranked pulldown designs:
  1. Look up cell-surface lectins for that cell type
  2. Map lectins to MABE scorer proxies where available
  3. Score sugars, pick best binder
  4. Find CANDIDATE click sites
  5. Design linker + multivalent geometry
  6. Rank by composite score

Usage:
    recs = recommend_pulldown("macrophage_m2", bead_diameter_nm=50)
    for r in recs:
        print(r.summary)

Cell-lectin data from published immunology (DOI-referenced per entry).
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from glycan.click_site_predictor import get_attachment_sites, CANDIDATE
from glycan.bead_linker_design import (
    design_pulldown, LinkerDesign, MultivalentEstimate,
    estimate_multivalent_enhancement,
    is_oligosaccharide, design_pulldown_reducing_end,
    recommend_peg_reducing_end,
)

# Lazy import scorer to avoid circular deps if scorer isn't deployed
_SCORER = None

def _get_scorer():
    global _SCORER
    if _SCORER is None:
        from glycan.scorer import GlycanScorer
        _SCORER = GlycanScorer()
    return _SCORER


# ── Cell type -> lectin database ────────────────────────────────────────
# Each entry: lectin name, sugar preference, estimated density (per cell),
# MABE scorer proxy (None if no proxy), confidence, literature source.
#
# Density: from quantitative flow cytometry / radiolabeled ligand studies.
# None = not quantified in literature.
#
# Confidence levels for scorer proxy:
#   HIGH   = direct structural match (same protein or same CRD fold + pharmacophore)
#   MEDIUM = same sugar preference, different fold (proxy approximation)
#   LOW    = qualitative only (no scorer proxy)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class LectinEntry:
    lectin: str
    sugar_preference: str          # preferred monosaccharide class
    sugar_for_scorer: Optional[str]  # exact ligand name in scorer contact maps
    density_per_cell: Optional[int]
    scorer_proxy: Optional[str]    # MABE scaffold name, or None
    proxy_confidence: str          # HIGH | MEDIUM | LOW
    source: str                    # DOI or short citation


CELL_LECTIN_DB: dict[str, list[LectinEntry]] = {
    # ── T cells ─────────────────────────────────────────────────────
    "t_cell": [
        LectinEntry(
            lectin="Galectin-3",
            sugar_preference="Gal/LacNAc",
            sugar_for_scorer="Gal",
            density_per_cell=None,
            scorer_proxy="Gal3",
            proxy_confidence="HIGH",
            source="Stillman 2006 J Immunol 176:778",
        ),
    ],
    "t_cell_activated": [
        LectinEntry(
            lectin="Galectin-3",
            sugar_preference="Gal/LacNAc",
            sugar_for_scorer="Gal",
            density_per_cell=None,
            scorer_proxy="Gal3",
            proxy_confidence="HIGH",
            source="Stillman 2006 J Immunol 176:778",
        ),
        LectinEntry(
            lectin="CD62L (L-selectin)",
            sugar_preference="sialyl-Lewis-x",
            sugar_for_scorer=None,
            density_per_cell=70000,
            scorer_proxy=None,
            proxy_confidence="LOW",
            source="Kansas 1993 Blood 82:3435",
        ),
    ],

    # ── Macrophages ─────────────────────────────────────────────────
    "macrophage_m2": [
        LectinEntry(
            lectin="Mannose receptor (CD206)",
            sugar_preference="Man",
            sugar_for_scorer="Man",
            density_per_cell=50000,
            scorer_proxy="ConA",
            proxy_confidence="MEDIUM",
            source="Stahl 1978 PNAS 75:1399; Taylor 2005 Annu Rev Immunol 23:901",
        ),
        LectinEntry(
            lectin="Galectin-3",
            sugar_preference="Gal/LacNAc",
            sugar_for_scorer="Gal",
            density_per_cell=None,
            scorer_proxy="Gal3",
            proxy_confidence="HIGH",
            source="Sato 2002 J Biol Chem 277:20830",
        ),
    ],
    "macrophage_m1": [
        LectinEntry(
            lectin="Galectin-3",
            sugar_preference="Gal/LacNAc",
            sugar_for_scorer="Gal",
            density_per_cell=None,
            scorer_proxy="Gal3",
            proxy_confidence="HIGH",
            source="Sato 2002 J Biol Chem 277:20830",
        ),
    ],

    # ── Dendritic cells ─────────────────────────────────────────────
    "dendritic_cell": [
        LectinEntry(
            lectin="DC-SIGN (CD209)",
            sugar_preference="Man/Fuc",
            sugar_for_scorer="Man",
            density_per_cell=10000,
            scorer_proxy="ConA",
            proxy_confidence="MEDIUM",
            source="Geijtenbeek 2000 Cell 100:575",
        ),
        LectinEntry(
            lectin="Mannose receptor (CD206)",
            sugar_preference="Man",
            sugar_for_scorer="Man",
            density_per_cell=30000,
            scorer_proxy="ConA",
            proxy_confidence="MEDIUM",
            source="Sallusto 1995 J Exp Med 182:389",
        ),
    ],

    # ── NK cells ────────────────────────────────────────────────────
    "nk_cell": [
        LectinEntry(
            lectin="NKR-P1",
            sugar_preference="GlcNAc",
            sugar_for_scorer="GlcNAc",
            density_per_cell=None,
            scorer_proxy="WGA",
            proxy_confidence="MEDIUM",
            source="Bezouska 1994 Nature 372:150",
        ),
    ],

    # ── B cells ─────────────────────────────────────────────────────
    "b_cell": [
        LectinEntry(
            lectin="CD22 (Siglec-2)",
            sugar_preference="sialic acid (Neu5Ac)",
            sugar_for_scorer=None,
            density_per_cell=20000,
            scorer_proxy=None,
            proxy_confidence="LOW",
            source="Tedder 2005 Annu Rev Immunol 23:515",
        ),
    ],

    # ── Neutrophils ─────────────────────────────────────────────────
    "neutrophil": [
        LectinEntry(
            lectin="L-selectin (CD62L)",
            sugar_preference="sialyl-Lewis-x",
            sugar_for_scorer=None,
            density_per_cell=None,
            scorer_proxy=None,
            proxy_confidence="LOW",
            source="Kansas 1993 Blood 82:3435",
        ),
    ],

    # ── Hepatocytes ─────────────────────────────────────────────────
    "hepatocyte": [
        LectinEntry(
            lectin="ASGPR (asialoglycoprotein receptor)",
            sugar_preference="Gal/GalNAc",
            sugar_for_scorer="Gal",
            density_per_cell=500000,
            scorer_proxy="PNA",
            proxy_confidence="MEDIUM",
            source="Stockert 1980 JBC 255:3830",
        ),
    ],

    # ── Tumor (epithelial) ──────────────────────────────────────────
    "tumor_epithelial": [
        LectinEntry(
            lectin="Galectin-1",
            sugar_preference="LacNAc/Gal",
            sugar_for_scorer="Gal",
            density_per_cell=None,
            scorer_proxy="Gal3",
            proxy_confidence="MEDIUM",
            source="Camby 2006 Glycobiology 16:137R",
        ),
        LectinEntry(
            lectin="Galectin-3",
            sugar_preference="LacNAc/Gal",
            sugar_for_scorer="Gal",
            density_per_cell=None,
            scorer_proxy="Gal3",
            proxy_confidence="HIGH",
            source="Liu & Rabinovich 2005 Nat Rev Cancer 5:29",
        ),
    ],
}

# Aliases for common naming variants
_ALIASES = {
    "t cell": "t_cell",
    "t-cell": "t_cell",
    "tcell": "t_cell",
    "activated t cell": "t_cell_activated",
    "macrophage": "macrophage_m2",
    "m2 macrophage": "macrophage_m2",
    "m1 macrophage": "macrophage_m1",
    "dc": "dendritic_cell",
    "dendritic": "dendritic_cell",
    "nk": "nk_cell",
    "nk cell": "nk_cell",
    "natural killer": "nk_cell",
    "b cell": "b_cell",
    "b-cell": "b_cell",
    "hepatocyte": "hepatocyte",
    "liver": "hepatocyte",
    "tumor": "tumor_epithelial",
    "cancer": "tumor_epithelial",
    "epithelial tumor": "tumor_epithelial",
}


def _resolve_cell_type(cell_type: str) -> str:
    """Resolve cell type name to canonical key."""
    key = cell_type.strip().lower().replace("_", " ")
    if key in _ALIASES:
        return _ALIASES[key]
    # Try direct match
    canonical = cell_type.strip().lower().replace(" ", "_")
    if canonical in CELL_LECTIN_DB:
        return canonical
    raise ValueError(
        f"Unknown cell type: '{cell_type}'. "
        f"Available: {sorted(CELL_LECTIN_DB.keys())}"
    )


# ── Recommendation dataclass ───────────────────────────────────────────

@dataclass
class PulldownRecommendation:
    cell_type: str
    lectin: str
    lectin_density: Optional[int]
    scorer_proxy: Optional[str]
    proxy_confidence: str
    sugar: str
    dG_pred: Optional[float]
    position: str
    linker: Optional[LinkerDesign]
    multivalent: Optional[MultivalentEstimate]
    composite_score: float
    confidence: str
    notes: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def summary(self) -> str:
        parts = [f"{self.cell_type} -> {self.lectin} -> {self.sugar}@{self.position}"]
        if self.dG_pred is not None:
            parts.append(f"dG={self.dG_pred:.1f} kJ/mol")
        if self.linker and self.linker.feasible:
            parts.append(f"PEG_{self.linker.peg_n_recommended}")
            parts.append(f"entropy={self.linker.ddG_entropy_kJ:.2f} kJ/mol")
        if self.multivalent:
            parts.append(f"enhancement=10^{self.multivalent.enhancement_log10:.1f}")
        parts.append(f"[{self.confidence}]")
        if self.linker and self.linker.warning:
            parts.append(f"WARNING: {self.linker.warning}")
        return " | ".join(parts)


# ── Main pipeline ──────────────────────────────────────────────────────

def recommend_pulldown(
    cell_type: str,
    bead_diameter_nm: float = 50.0,
    sugar_spacing_nm: float = 5.0,
    max_ligands_per_lectin: int = 3,
) -> list[PulldownRecommendation]:
    """Full pipeline: cell type -> ranked pulldown designs.

    Scores ALL ligands per scaffold (mono + oligosaccharide) and picks
    the best binders. Oligosaccharides use reducing-end C1 attachment;
    monosaccharides use position-resolved click site analysis.

    Args:
        cell_type: target cell type (e.g., "macrophage_m2", "t_cell")
        bead_diameter_nm: magnetic bead diameter in nm
        sugar_spacing_nm: sugar spacing on bead surface in nm
        max_ligands_per_lectin: max ligands to report per lectin (top N by dG)

    Returns:
        List of PulldownRecommendation sorted by composite_score (descending).
    """
    canonical = _resolve_cell_type(cell_type)
    lectins = CELL_LECTIN_DB[canonical]
    scorer = _get_scorer()

    recommendations = []

    for entry in lectins:
        if entry.scorer_proxy is not None:
            # Full quantitative path: score ALL ligands in this scaffold
            try:
                all_preds = scorer.score_scaffold(entry.scorer_proxy)
            except (ValueError, KeyError):
                all_preds = []

            if not all_preds:
                # Fallback: try single sugar if specified
                if entry.sugar_for_scorer:
                    try:
                        pred = scorer.score(entry.scorer_proxy, entry.sugar_for_scorer)
                        all_preds = [pred]
                    except (ValueError, KeyError):
                        pass

            # Sort by predicted dG (most negative = strongest binder)
            all_preds.sort(key=lambda p: p.dG_pred)

            # Take top N ligands
            top_preds = all_preds[:max_ligands_per_lectin]

            for pred in top_preds:
                dG_pred = pred.dG_pred
                ligand_name = pred.ligand

                if is_oligosaccharide(ligand_name):
                    # Oligosaccharide: reducing-end C1 attachment
                    recs_for_ligand = _build_oligo_recs(
                        canonical, entry, ligand_name, dG_pred,
                        bead_diameter_nm, sugar_spacing_nm,
                    )
                else:
                    # Monosaccharide: position-resolved click sites
                    recs_for_ligand = _build_mono_recs(
                        canonical, entry, ligand_name, dG_pred,
                        bead_diameter_nm, sugar_spacing_nm,
                    )

                recommendations.extend(recs_for_ligand)
        else:
            # Qualitative path: no scorer proxy
            mv = estimate_multivalent_enhancement(
                bead_diameter_nm, sugar_spacing_nm,
                receptor_density_per_um2=_density_to_per_um2(entry.density_per_cell),
            )
            composite = 0.1

            recommendations.append(PulldownRecommendation(
                cell_type=canonical,
                lectin=entry.lectin,
                lectin_density=entry.density_per_cell,
                scorer_proxy=None,
                proxy_confidence="LOW",
                sugar=entry.sugar_preference,
                dG_pred=None,
                position="C1 (default anomeric)",
                linker=None,
                multivalent=mv,
                composite_score=composite,
                confidence="LOW",
                notes=["No MABE scorer proxy -- qualitative sugar preference only",
                       "C1 anomeric is universally the safest attachment site"],
                source=entry.source,
            ))

    # Sort by composite score descending
    recommendations.sort(key=lambda r: r.composite_score, reverse=True)
    return recommendations


def _build_oligo_recs(
    cell_type: str,
    entry: LectinEntry,
    ligand: str,
    dG_pred: float,
    bead_diameter_nm: float,
    sugar_spacing_nm: float,
) -> list[PulldownRecommendation]:
    """Build recommendations for an oligosaccharide ligand (reducing-end attachment)."""
    try:
        pd = design_pulldown_reducing_end(
            entry.scorer_proxy, ligand, bead_diameter_nm, sugar_spacing_nm,
            receptor_density_per_um2=_density_to_per_um2(entry.density_per_cell),
        )
        linker = pd.linker
        mv = pd.multivalent
    except (ValueError, KeyError):
        linker = None
        mv = None

    composite = _compute_composite(dG_pred, linker, mv, entry.proxy_confidence)

    notes = [
        f"Proxy: {entry.scorer_proxy} models {entry.lectin}",
        f"Oligosaccharide: reducing-end C1 attachment (neoglycoconjugate)",
    ]
    if entry.proxy_confidence == "MEDIUM":
        notes.append("Proxy is approximate (same sugar preference, different fold)")

    return [PulldownRecommendation(
        cell_type=cell_type,
        lectin=entry.lectin,
        lectin_density=entry.density_per_cell,
        scorer_proxy=entry.scorer_proxy,
        proxy_confidence=entry.proxy_confidence,
        sugar=ligand,
        dG_pred=dG_pred,
        position="C1_reducing",
        linker=linker,
        multivalent=mv,
        composite_score=composite,
        confidence=entry.proxy_confidence,
        notes=notes,
        source=entry.source,
    )]


def _build_mono_recs(
    cell_type: str,
    entry: LectinEntry,
    ligand: str,
    dG_pred: float,
    bead_diameter_nm: float,
    sugar_spacing_nm: float,
) -> list[PulldownRecommendation]:
    """Build recommendations for a monosaccharide ligand (position-resolved click sites)."""
    recs = []

    # Get CANDIDATE positions
    try:
        sites = get_attachment_sites(entry.scorer_proxy, ligand)
    except ValueError:
        sites = []

    if sites:
        for site in sites:
            pos = site.position
            try:
                pd = design_pulldown(
                    entry.scorer_proxy, ligand, pos,
                    bead_diameter_nm, sugar_spacing_nm,
                    receptor_density_per_um2=_density_to_per_um2(entry.density_per_cell),
                )
                linker = pd.linker
                mv = pd.multivalent
            except (ValueError, KeyError):
                linker = None
                mv = None

            composite = _compute_composite(dG_pred, linker, mv, entry.proxy_confidence)

            notes = [f"Proxy: {entry.scorer_proxy} models {entry.lectin}"]
            if entry.proxy_confidence == "MEDIUM":
                notes.append("Proxy is approximate (same sugar preference, different fold)")

            recs.append(PulldownRecommendation(
                cell_type=cell_type,
                lectin=entry.lectin,
                lectin_density=entry.density_per_cell,
                scorer_proxy=entry.scorer_proxy,
                proxy_confidence=entry.proxy_confidence,
                sugar=ligand,
                dG_pred=dG_pred,
                position=pos,
                linker=linker,
                multivalent=mv,
                composite_score=composite,
                confidence=entry.proxy_confidence,
                notes=notes,
                source=entry.source,
            ))
    else:
        # No position contacts for this mono at this scaffold -- use default C1
        composite = _compute_composite(dG_pred, None, None, entry.proxy_confidence)
        recs.append(PulldownRecommendation(
            cell_type=cell_type,
            lectin=entry.lectin,
            lectin_density=entry.density_per_cell,
            scorer_proxy=entry.scorer_proxy,
            proxy_confidence=entry.proxy_confidence,
            sugar=ligand,
            dG_pred=dG_pred,
            position="C1 (default)",
            linker=None,
            multivalent=None,
            composite_score=composite,
            confidence=entry.proxy_confidence,
            notes=[f"No position contacts mapped for {ligand} at {entry.scorer_proxy}; default C1"],
            source=entry.source,
        ))

    return recs

    # Sort by composite score descending
    recommendations.sort(key=lambda r: r.composite_score, reverse=True)
    return recommendations


def _density_to_per_um2(density_per_cell: Optional[int]) -> float:
    """Convert receptor copies per cell to per um^2.

    Assumes typical lymphocyte surface area ~300 um^2.
    If density unknown, return default 1000/um^2.
    """
    if density_per_cell is None:
        return 1000.0
    # Lymphocyte SA ~ 300 um^2; macrophage ~ 1000 um^2; hepatocyte ~ 1500 um^2
    # Use 500 um^2 as a reasonable middle estimate
    cell_surface_um2 = 500.0
    return density_per_cell / cell_surface_um2


def _compute_composite(
    dG_pred: Optional[float],
    linker: Optional[LinkerDesign],
    mv: Optional[MultivalentEstimate],
    proxy_confidence: str,
) -> float:
    """Composite score for ranking pulldown designs.

    Higher = better pulldown candidate.
    Components:
      - Binding strength: |dG_pred| (stronger binding = higher)
      - Feasibility: 1.0 if feasible, 0.1 if not
      - Multivalent boost: 10^(enhancement_log10 / 6) normalized to [1, 10]
      - Confidence weight: HIGH=1.0, MEDIUM=0.7, LOW=0.3
    """
    # Binding strength
    if dG_pred is not None:
        strength = abs(dG_pred)  # kJ/mol, higher = stronger
    else:
        strength = 5.0  # minimal default for qualitative entries

    # Feasibility
    if linker is not None and linker.feasible:
        feasibility = 1.0
    elif linker is not None and not linker.feasible:
        feasibility = 0.1
    else:
        feasibility = 0.5  # unknown

    # Multivalent boost
    if mv is not None and mv.enhancement_log10 > 0:
        mv_boost = 1.0 + mv.enhancement_log10  # linear in log scale
    else:
        mv_boost = 1.0

    # Confidence weight
    conf_weight = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.3}.get(proxy_confidence, 0.3)

    return round(strength * feasibility * mv_boost * conf_weight, 2)


# ── Convenience ─────────────────────────────────────────────────────────

def list_cell_types() -> list[str]:
    """Return all available cell types."""
    return sorted(CELL_LECTIN_DB.keys())


def list_aliases() -> dict[str, str]:
    """Return alias -> canonical mapping."""
    return dict(_ALIASES)
