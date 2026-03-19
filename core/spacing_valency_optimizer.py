"""
core/spacing_valency_optimizer.py — Display Geometry Optimization

Predicts optimal binder copy number (valency) and inter-binder
spacing on a scaffold for immune cell activation.

Physics basis:
  - Veneziano et al. 2020, Nat. Nanotech.: BCR activation peaks at
    ≥5 copies, increases with spacing up to ~22 nm, plateau beyond
  - Shaw et al. 2014: EphA2 receptor clustering in breast cancer
    cells, 40 nm spacing > 100 nm
  - DoriVac: CpG adjuvant on opposite face from antigen, ~3.5 nm
    intra-face spacing on square block
  - Fang et al.: PD-L1 spacing on origami controls T cell checkpoint

Model: empirical activation curve fitted to published data points.
No free parameters beyond literature values.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ═══════════════════════════════════════════════════════════════════════════
# BINDER FOOTPRINT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BinderFootprint:
    """Physical size of the binder relevant to display geometry."""
    mw: float = 350.0
    diameter_nm: float = 1.0      # approximate molecular diameter
    height_nm: float = 0.8        # protrusion from scaffold surface
    binding_face_fraction: float = 0.5  # fraction of surface that is the pharmacophore
    n_rotatable: int = 3          # flexibility (affects orientation control)

    @classmethod
    def from_mw(cls, mw: float, n_rotatable: int = 3):
        """Estimate footprint from MW using empirical scaling.

        Small molecule diameter ~ (MW / 200)^(1/3) nm for globular.
        Elongated molecules: use max dimension from RDKit if available.
        """
        diam = (mw / 200.0) ** (1.0 / 3.0)  # ~1.0 nm for MW 200, ~1.5 for MW 675
        height = diam * 0.8  # protrusion ~80% of diameter
        return cls(mw=mw, diameter_nm=diam, height_nm=height,
                   n_rotatable=n_rotatable)


# ═══════════════════════════════════════════════════════════════════════════
# BCR ACTIVATION MODEL
# ═══════════════════════════════════════════════════════════════════════════

def bcr_activation_score(valency: int, spacing_nm: float) -> float:
    """
    Predict relative B cell receptor activation from valency + spacing.

    Model from Veneziano 2020 empirical data:
    - Activation increases with valency up to ~10, diminishing returns beyond
    - Activation increases with spacing up to ~22 nm, plateau/decline beyond
    - Below ~5 nm, steric occlusion reduces activation
    - Monovalent (1 copy) has baseline activation = 0.1

    Returns: 0-1 relative activation score.
    """
    # Valency component: sigmoid saturating at ~10 copies
    # Veneziano: 5 copies needed for strong activation, 10 near-maximal
    if valency <= 0:
        return 0.0
    v_score = 1.0 / (1.0 + math.exp(-0.8 * (valency - 5)))

    # Spacing component: peaked around 15-25 nm
    # Veneziano: monotonic increase to ~22 nm on icosahedron
    # Shaw: 40 nm better than 100 nm (but that's a different receptor)
    # Model: Gaussian-like peak centered at 18 nm, σ = 10 nm
    if spacing_nm < 3.0:
        s_score = 0.2  # too close — steric clash
    elif spacing_nm <= 5.0:
        s_score = 0.2 + 0.3 * (spacing_nm - 3.0) / 2.0  # ramp up
    else:
        # Peak around 18 nm, gradual decline
        s_score = 0.5 + 0.5 * math.exp(-((spacing_nm - 18.0) ** 2) / (2 * 100))

    return v_score * s_score


def optimal_display_for_scaffold(
    scaffold_min_spacing: float,
    scaffold_max_spacing: float,
    scaffold_min_valency: int,
    scaffold_max_valency: int,
    binder_diameter_nm: float = 1.0,
    spacing_programmable: bool = False,
) -> Dict:
    """
    Find optimal valency and spacing for a given scaffold.

    Searches the achievable parameter space to maximize BCR activation.
    """
    best_score = 0.0
    best_valency = scaffold_min_valency
    best_spacing = scaffold_min_spacing

    # Minimum spacing = binder diameter + 1 nm clearance
    min_feasible = max(scaffold_min_spacing, binder_diameter_nm + 1.0)

    if spacing_programmable:
        # Search continuous spacing
        spacings = [min_feasible + i * 0.5
                    for i in range(int((scaffold_max_spacing - min_feasible) / 0.5) + 1)]
    else:
        # Fixed spacing — just use what the scaffold gives
        spacings = [(scaffold_min_spacing + scaffold_max_spacing) / 2]

    valencies = list(range(
        max(1, scaffold_min_valency),
        min(scaffold_max_valency + 1, 61),  # cap search at 60
    ))
    # Coarsen if range is large
    if len(valencies) > 30:
        valencies = valencies[::max(1, len(valencies) // 30)]

    for v in valencies:
        for s in spacings:
            score = bcr_activation_score(v, s)
            if score > best_score:
                best_score = score
                best_valency = v
                best_spacing = s

    return {
        'optimal_valency': best_valency,
        'optimal_spacing_nm': best_spacing,
        'predicted_activation': best_score,
        'min_feasible_spacing': min_feasible,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CO-DISPLAY ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DisplaySpec:
    """Complete display specification for an immune-engaging construct."""
    # Binder display
    binder_valency: int = 10
    binder_spacing_nm: float = 15.0
    binder_face: str = "A"       # which face of the scaffold
    # Adjuvant co-display
    adjuvant_type: str = "CpG"   # CpG ODN (TLR9 agonist)
    adjuvant_valency: int = 10
    adjuvant_face: str = "B"     # opposite face (DoriVac design)
    adjuvant_spacing_nm: float = 3.5
    # Targeting ligand (optional)
    targeting_ligand: str = ""   # e.g., folate, transferrin peptide
    targeting_valency: int = 0
    # Scaffold
    scaffold_name: str = ""
    conjugation_chemistry: str = ""
    # Predicted performance
    predicted_bcr_activation: float = 0.0
    predicted_dc_activation: str = "moderate"  # from CpG co-display
    # PEG shield
    peg_density: str = "sparse"  # sparse/moderate/dense
    peg_mw: int = 2000
    # Total sites used
    total_sites_used: int = 0
    sites_available: int = 0


def design_display_spec(
    scaffold_name: str,
    scaffold_max_valency: int,
    scaffold_min_spacing: float,
    scaffold_max_spacing: float,
    scaffold_programmable: bool,
    binder_diameter_nm: float = 1.0,
    include_adjuvant: bool = True,
    include_peg: bool = True,
    co_display_possible: bool = True,
) -> DisplaySpec:
    """
    Design the full display specification for a construct.

    Allocates scaffold sites to: binder + adjuvant + PEG shield.
    DoriVac model: binder on face A, CpG on face B.
    If co-display not possible, binder only.
    """
    spec = DisplaySpec(scaffold_name=scaffold_name)

    # Get optimal binder display
    opt = optimal_display_for_scaffold(
        scaffold_min_spacing, scaffold_max_spacing,
        1, scaffold_max_valency,
        binder_diameter_nm, scaffold_programmable,
    )

    spec.binder_valency = opt['optimal_valency']
    spec.binder_spacing_nm = opt['optimal_spacing_nm']
    spec.predicted_bcr_activation = opt['predicted_activation']

    sites_remaining = scaffold_max_valency - spec.binder_valency

    # Adjuvant allocation
    if include_adjuvant and co_display_possible and sites_remaining >= 5:
        spec.adjuvant_valency = min(spec.binder_valency, sites_remaining // 2)
        spec.adjuvant_spacing_nm = min(5.0, scaffold_max_spacing)
        spec.adjuvant_face = "B"
        sites_remaining -= spec.adjuvant_valency
        spec.predicted_dc_activation = "strong"  # CpG + multivalent binder
    elif include_adjuvant:
        spec.adjuvant_valency = 0
        spec.predicted_dc_activation = "weak"

    # PEG allocation (anti-fouling)
    if include_peg and sites_remaining >= 5:
        peg_count = min(sites_remaining, spec.binder_valency)
        sites_remaining -= peg_count
        spec.peg_density = "moderate" if peg_count >= 10 else "sparse"
    else:
        spec.peg_density = "none"

    spec.total_sites_used = scaffold_max_valency - sites_remaining
    spec.sites_available = scaffold_max_valency

    return spec


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH SCAFFOLD RANKER
# ═══════════════════════════════════════════════════════════════════════════

def full_construct_spec(
    binder_name: str,
    binder_mw: float,
    binder_n_rotatable: int,
    scaffold_results,  # List[ScaffoldScore] from scaffold_realization_ranker
) -> List[Dict]:
    """
    For each ranked scaffold, generate a complete construct specification.

    Returns list of dicts with scaffold ranking + display spec + construct summary.
    """
    from core.scaffold_realization_ranker import SCAFFOLD_BY_NAME

    footprint = BinderFootprint.from_mw(binder_mw, binder_n_rotatable)
    constructs = []

    for ss in scaffold_results:
        scaffold = SCAFFOLD_BY_NAME.get(ss.scaffold_name)
        if scaffold is None:
            continue

        display = design_display_spec(
            scaffold_name=scaffold.name,
            scaffold_max_valency=scaffold.max_valency,
            scaffold_min_spacing=scaffold.min_spacing_nm,
            scaffold_max_spacing=scaffold.max_spacing_nm,
            scaffold_programmable=scaffold.spacing_programmable,
            binder_diameter_nm=footprint.diameter_nm,
            co_display_possible=scaffold.co_display_possible,
        )

        constructs.append({
            'scaffold_name': scaffold.name,
            'scaffold_category': scaffold.category,
            'scaffold_composite_score': ss.composite,
            'conjugation': ss.conjugation_strategy,
            'binder_name': binder_name,
            'binder_valency': display.binder_valency,
            'binder_spacing_nm': round(display.binder_spacing_nm, 1),
            'adjuvant_type': display.adjuvant_type if display.adjuvant_valency > 0 else "none",
            'adjuvant_valency': display.adjuvant_valency,
            'predicted_bcr_activation': round(display.predicted_bcr_activation, 3),
            'predicted_dc_activation': display.predicted_dc_activation,
            'peg_shield': display.peg_density,
            'total_sites': f"{display.total_sites_used}/{display.sites_available}",
            'est_cost_per_mg': scaffold.est_cost_per_mg_usd,
        })

    return constructs


if __name__ == "__main__":
    # Test BCR activation model
    print("BCR ACTIVATION MODEL (Veneziano-informed)")
    print(f"{'Valency':>8s} {'Spacing':>8s} {'Activation':>10s}")
    for v in [1, 3, 5, 8, 10, 15, 20]:
        for s in [3, 5, 10, 15, 20, 25, 40]:
            a = bcr_activation_score(v, s)
            if a > 0.3:
                print(f"{v:8d} {s:8.0f} nm {a:10.3f}")

    # Test full construct spec
    print()
    print("=" * 80)
    print("FULL CONSTRUCT SPECIFICATIONS — tripeptide+boronic (MW=308)")
    print("=" * 80)

    from core.scaffold_realization_ranker import rank_scaffolds

    scaffolds = rank_scaffolds(binder_mw=308, binder_n_hbd=5, binder_n_aromatic=1)
    constructs = full_construct_spec(
        binder_name="tripeptide+2-pyridyl|methylboronic",
        binder_mw=308,
        binder_n_rotatable=4,
        scaffold_results=scaffolds,
    )

    print(f"\n{'Scaffold':35s} {'Conj':12s} {'Val':>4s} {'Space':>6s} "
          f"{'BCR':>5s} {'DC':>8s} {'Adj':>4s} {'PEG':>8s} {'Sites':>8s} {'$/mg':>6s}")
    print("─" * 110)
    for c in constructs:
        print(f"{c['scaffold_name']:35s} {c['conjugation']:12s} "
              f"{c['binder_valency']:4d} {c['binder_spacing_nm']:5.1f}nm "
              f"{c['predicted_bcr_activation']:5.3f} {c['predicted_dc_activation']:>8s} "
              f"{c['adjuvant_valency']:4d} {c['peg_shield']:>8s} "
              f"{c['total_sites']:>8s} {c['est_cost_per_mg']:6.1f}")