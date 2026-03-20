"""
core/demand_generator.py -- Unified physics-demand-driven molecule generation.

The target defines what the binder/chelator/receptor needs. The demand vector
narrows the combinatorial space before enumeration. The scorer ranks what's left.

Modality-specific demand functions:
  - metal_demand(metal) -> DemandVector: HSAB + denticity + charge
  - host_guest_demand(host) -> DemandVector: cavity + polarity + HB
  - glycan_demand(sugar) -> DemandVector: CH-pi + HBD strategy + boronic

All produce the same DemandVector and feed into the same generation pipeline.

Entry point:
  generate_from_demand(mode, target, ...) -> GenerationResult

Wires into existing de_novo_generator via:
  backbones = grammar_backbones(pfilter from demand)
  arms = grammar_arms(filtered by demand)
  enumerate_molecules(backbones=backbones, arms=arms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.ring_enumerator import (
    PhysicsFilter, grammar_backbones, grammar_arms,
    get_decorators, DECORATOR_LIBRARY,
)

# Re-export for convenience
from core.ring_enumerator import (
    get_catalog, get_ring_system, list_ring_systems,
    enumerate_decorated, enumerate_all_decorated,
    enumerate_physics_filtered,
    RingSystem, DecoratedScaffold, Decorator,
)


# ---------------------------------------------------------------------------
# Demand vector (unified across modalities)
# ---------------------------------------------------------------------------

@dataclass
class DemandVector:
    """Physics-derived demands for any MABE modality."""
    mode: str                      # "metal", "host_guest", "glycan", "receptor"
    target: str                    # target identifier
    # Scaffold demands
    min_aromatic_atoms: int = 0
    max_aromatic_atoms: int = 999
    prefer_large_aromatic: bool = False
    min_rings: int = 1
    max_rings: int = 4
    scaffold_categories: Optional[List[str]] = None
    require_NH: bool = False
    # Decorator demands
    decorator_categories: Optional[List[str]] = None
    required_hardness: Optional[str] = None    # HSAB filter for metals
    required_donor_element: Optional[str] = None
    min_hbd_per_arm: int = 0
    require_boronic: bool = False
    hbd_strategy: str = "any"      # "none", "minimal", "bimodal", "saturate", "any"
    # Size demands
    target_volume_A3: float = 0.0
    max_mw: float = 800.0
    max_heavy_atoms: int = 60
    # Metadata
    selectivity_axis: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Metal demand
# ---------------------------------------------------------------------------

_METAL_HARDNESS = {
    "Li+": "hard", "Na+": "hard", "K+": "hard", "Mg2+": "hard",
    "Ca2+": "hard", "Sr2+": "hard", "Ba2+": "hard",
    "Al3+": "hard", "Fe3+": "hard", "Cr3+": "hard", "La3+": "hard",
    "Zr4+": "hard",
    "Fe2+": "borderline", "Co2+": "borderline", "Ni2+": "borderline",
    "Cu2+": "borderline", "Zn2+": "borderline", "Mn2+": "borderline",
    "Pb2+": "borderline", "Cd2+": "borderline",
    "Cu+": "soft", "Ag+": "soft", "Au+": "soft", "Au3+": "soft",
    "Hg2+": "soft", "Pd2+": "soft", "Pt2+": "soft",
}


def metal_demand(metal: str) -> DemandVector:
    """
    Derive demand vector for a metal chelator.

    Physics logic:
    - HSAB: hard metals want O/N donors, soft metals want S/P donors
    - Charge: higher charge = more donor sites needed
    - Ring preference: N-heterocycles provide direct coordination (pyridine, imidazole)
    - Soft metals benefit from S-heterocycles (thiophene, benzothiazole)
    """
    hardness = _METAL_HARDNESS.get(metal, "borderline")

    # Parse charge from metal string
    charge = 0
    for ch in metal:
        if ch == "+":
            charge += 1
        elif ch.isdigit() and "+" in metal:
            charge = int(ch)

    # HSAB-driven decorator selection
    if hardness == "hard":
        req_hardness = "hard"
        donor_elem = "O"
        dec_cats = ["HBD", "HBA", "donor"]
        scaffold_cats = ["carbocyclic", "N-hetero"]
        notes = "Hard metal: O-donors preferred (carboxylate, hydroxamate, catechol)"
    elif hardness == "soft":
        req_hardness = "soft"
        donor_elem = "S"
        dec_cats = ["donor"]
        scaffold_cats = ["S-hetero", "carbocyclic"]
        notes = "Soft metal: S-donors preferred (thiol, thioether)"
    else:
        req_hardness = None  # borderline accepts everything
        donor_elem = None
        dec_cats = ["HBD", "HBA", "donor"]
        scaffold_cats = ["N-hetero", "carbocyclic"]
        notes = "Borderline metal: mixed N/O/S donors"

    # Ring demand: N-heterocycles can coordinate directly
    if hardness in ("borderline", "hard"):
        min_arom = 5  # at least one heterocyclic ring
    else:
        min_arom = 0

    # Denticity demand from charge
    min_rings_needed = max(1, charge)  # rough: higher charge needs more donor sites

    return DemandVector(
        mode="metal", target=metal,
        min_aromatic_atoms=min_arom,
        min_rings=1, max_rings=3,
        scaffold_categories=scaffold_cats,
        require_NH=(hardness == "borderline"),
        decorator_categories=dec_cats,
        required_hardness=req_hardness,
        required_donor_element=donor_elem,
        hbd_strategy="any",
        max_mw=600.0,  # chelators are typically smaller
        max_heavy_atoms=45,
        selectivity_axis=f"HSAB ({hardness})",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Host-guest demand
# ---------------------------------------------------------------------------

_HOST_VOLUMES = {
    "alpha-CD": 174.0,
    "beta-CD": 262.0,
    "gamma-CD": 427.0,
    "CB[7]": 279.0,
    "CB[8]": 479.0,
    "pillar[5]arene": 120.0,
}


def host_guest_demand(host_key: str, cavity_volume: float = 0.0) -> DemandVector:
    """
    Derive demand vector for a guest targeting a specific host cavity.

    Physics logic:
    - Rebek 55% rule: guest volume ~55% of cavity volume
    - Hydrophobic cavity = hydrophobic guest (aromatic, aliphatic)
    - Polar portal = H-bond capable guest
    - Cavity depth determines max guest length
    """
    if cavity_volume <= 0:
        cavity_volume = _HOST_VOLUMES.get(host_key, 262.0)

    # Rebek: optimal guest volume = 55% of cavity
    optimal_guest_vol = cavity_volume * 0.55
    max_guest_vol = cavity_volume * 0.65

    # Guest MW estimate from volume (~1 Da per A3 for organic)
    est_mw = min(optimal_guest_vol * 1.2, 500.0)

    # Aromatic guests pack better in aromatic-lined cavities (CD, pillar)
    prefer_aromatic = cavity_volume < 350  # smaller cavities prefer flat guests

    return DemandVector(
        mode="host_guest", target=host_key,
        min_aromatic_atoms=6 if prefer_aromatic else 0,
        prefer_large_aromatic=False,
        min_rings=1, max_rings=2,
        decorator_categories=["hydrophobic", "HBA", "HBD"],
        hbd_strategy="minimal",
        target_volume_A3=optimal_guest_vol,
        max_mw=est_mw,
        max_heavy_atoms=int(est_mw / 12),  # rough heavy atom estimate
        selectivity_axis="cavity shape + Rebek packing",
        notes=f"Cavity {cavity_volume} A3, optimal guest {optimal_guest_vol:.0f} A3",
    )


# ---------------------------------------------------------------------------
# Glycan demand (delegates to glycan module if available)
# ---------------------------------------------------------------------------

def glycan_demand(sugar: str) -> DemandVector:
    """Derive demand for glycan binder. Uses glycan-specific sugar descriptors."""
    try:
        from glycan.demand_grammar import compute_demand as _glycan_demand
        gd = _glycan_demand(sugar)
        return DemandVector(
            mode="glycan", target=sugar,
            min_aromatic_atoms=gd.min_aromatic_atoms,
            prefer_large_aromatic=gd.prefer_large_aromatic,
            min_rings=1 if gd.min_aromatic_atoms < 10 else 2,
            max_rings=4,
            require_boronic=gd.boronic_preferred,
            hbd_strategy=gd.hbd_strategy,
            target_volume_A3=gd.target_volume_A3,
            selectivity_axis=gd.selectivity_axis,
            notes=gd.notes,
        )
    except ImportError:
        # Fallback if glycan module not available
        return DemandVector(mode="glycan", target=sugar)


# ---------------------------------------------------------------------------
# Unified generation pipeline
# ---------------------------------------------------------------------------

def generate_from_demand(
    mode: str,
    target: str,
    demand: Optional[DemandVector] = None,
    max_candidates: int = 200,
    max_scored: int = 50,
    use_existing_pipeline: bool = True,
    **kwargs,
):
    """
    Unified demand-driven generation.

    Args:
        mode: "metal", "host_guest", "glycan"
        target: target identifier (metal ion, host key, sugar name)
        demand: pre-computed DemandVector (auto-computed if None)
        max_candidates: max molecules to enumerate
        max_scored: max to score (for metal/host paths that use unified_scorer)
        use_existing_pipeline: if True, feed grammar into existing enumerate_molecules
        **kwargs: passed to modality-specific generator

    Returns:
        GenerationResult from existing pipeline
    """
    from core.de_novo_generator import (
        enumerate_molecules, generate_candidates,
        generate_for_host, GenerationResult,
    )

    # Compute demand if not provided
    if demand is None:
        if mode == "metal":
            demand = metal_demand(target)
        elif mode == "host_guest":
            demand = host_guest_demand(target, **kwargs)
        elif mode == "glycan":
            demand = glycan_demand(target)
        else:
            demand = DemandVector(mode=mode, target=target)

    # Build physics filter from demand
    pf = PhysicsFilter(
        min_aromatic_atoms=demand.min_aromatic_atoms,
        max_aromatic_atoms=demand.max_aromatic_atoms,
        require_large_aromatic=demand.prefer_large_aromatic,
        min_rings=demand.min_rings,
        max_rings=demand.max_rings,
        categories=demand.scaffold_categories,
        require_NH_donor=demand.require_NH,
    )

    # Generate backbones from ring grammar
    backbones = grammar_backbones(
        n_sites=2, pfilter=pf, max_per_system=10, max_total=200,
    )

    if not backbones:
        # Relax filter
        pf.min_aromatic_atoms = 0
        pf.require_large_aromatic = False
        pf.min_rings = 1
        pf.categories = None
        backbones = grammar_backbones(
            n_sites=2, pfilter=pf, max_per_system=10, max_total=200,
        )

    # Generate arms from decorator library
    arms = grammar_arms(
        categories=demand.decorator_categories,
        hardness=demand.required_hardness,
        donor_element=demand.required_donor_element,
    )

    if not arms:
        # Fallback: all decorators
        arms = grammar_arms()

    if not use_existing_pipeline:
        # Return raw components for custom pipelines (e.g., glycan)
        return {"backbones": backbones, "arms": arms, "demand": demand}

    # Feed into existing pipeline
    if mode == "metal":
        # Use enumerate_molecules with grammar backbones/arms
        from core.de_novo_generator import PropertyFilter as PF
        pfilter = PF(max_mw=demand.max_mw, max_heavy_atoms=demand.max_heavy_atoms)

        raw = enumerate_molecules(
            metal=target,
            backbones=backbones,
            arms=arms,
            max_candidates=max_candidates,
            pfilter=pfilter,
            hsab_filter=True,
        )

        # Score through existing metal pipeline
        return _score_metal_candidates(raw, target, max_scored, demand)

    elif mode == "host_guest":
        from core.de_novo_generator import PropertyFilter as PF
        pfilter = PF(max_mw=demand.max_mw, max_heavy_atoms=demand.max_heavy_atoms)

        raw = enumerate_molecules(
            host=target,
            backbones=backbones,
            arms=arms,
            max_candidates=max_candidates,
            pfilter=pfilter,
            hsab_filter=False,
        )
        return _score_host_candidates(raw, target, max_scored, demand)

    elif mode == "glycan":
        # Glycan has its own scorer; delegate to glycan module
        try:
            from glycan.demand_grammar import generate_from_demand as _glycan_gen
            return _glycan_gen(target=target, max_candidates=max_candidates)
        except ImportError:
            return GenerationResult(
                target=target, mode="glycan",
                errors=["glycan module not available"],
            )


def _score_metal_candidates(raw, metal, max_scored, demand):
    """Score enumerated candidates through design_engine_v2 for metal binding."""
    from core.de_novo_generator import (
        GeneratedCandidate, GenerationResult, _known_smiles_set,
    )
    from core.design_engine_v2 import score_one

    t0 = time.time()
    raw.sort(key=lambda x: x[3])  # sort by SA
    to_score = raw[:max_scored]

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa in to_score:
        try:
            sc = score_one(smiles, metal=metal, name=smiles[:40])
            gc = GeneratedCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=sc.log_Ka_pred,
                dg_total_kj=sc.dg_total_kj,
                prediction=sc.prediction,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                composite_score=sc.log_Ka_pred - 0.3 * sa,
                novel=smiles not in known,
            )
            candidates.append(gc)
        except Exception as e:
            errors.append(str(e))

    candidates.sort(key=lambda c: -c.composite_score)
    elapsed = time.time() - t0

    return GenerationResult(
        target=metal, mode="metal",
        candidates=candidates,
        n_enumerated=len(raw),
        n_valid=len(raw),
        n_unique=len(raw),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=elapsed,
        errors=errors[:5],
    )


def _score_host_candidates(raw, host, max_scored, demand):
    """Score enumerated candidates through design_engine_v2 for host-guest binding."""
    from core.de_novo_generator import (
        GeneratedCandidate, GenerationResult, _known_smiles_set,
    )
    from core.design_engine_v2 import score_one

    t0 = time.time()
    raw.sort(key=lambda x: x[3])
    to_score = raw[:max_scored]

    known = _known_smiles_set()
    candidates = []
    errors = []

    for smiles, bb_name, arm_names, sa in to_score:
        try:
            sc = score_one(smiles, host=host, name=smiles[:40])
            gc = GeneratedCandidate(
                smiles=smiles,
                name=f"{bb_name}+{'|'.join(arm_names)}",
                log_Ka_pred=sc.log_Ka_pred,
                dg_total_kj=sc.dg_total_kj,
                prediction=sc.prediction,
                backbone_name=bb_name,
                arm_names=arm_names,
                sa_score_val=sa,
                composite_score=sc.log_Ka_pred - 0.3 * sa,
                novel=smiles not in known,
            )
            candidates.append(gc)
        except Exception as e:
            errors.append(str(e))

    candidates.sort(key=lambda c: -c.composite_score)
    elapsed = time.time() - t0

    return GenerationResult(
        target=host, mode="host_guest",
        candidates=candidates,
        n_enumerated=len(raw),
        n_valid=len(raw),
        n_unique=len(raw),
        n_scored=len(candidates),
        n_failed=len(errors),
        elapsed_s=elapsed,
        errors=errors[:5],
    )
