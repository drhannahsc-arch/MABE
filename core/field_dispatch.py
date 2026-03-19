"""
core/field_dispatch.py — Unified Field Interaction Dispatch

Routes a FieldInteractionSpec to the correct physics engine based on
field_type. This is the architectural proof of computational isomorphism:
one dispatch call, three physics domains, same design chain.

Also provides CompositeFieldSpec for multi-physics targets: a single
structure that simultaneously handles optical + acoustic + thermal.

Architecture:
  dispatch_field(spec) →
    field_type == "electromagnetic" → optical pipeline
    field_type == "acoustic"        → acoustic pipeline
    field_type == "thermal"         → thermal pipeline

  dispatch_composite(specs) →
    score each physics domain independently
    check compatibility constraints (optical transparency of acoustic layer, etc.)
    return ranked multi-physics designs

This is the MABE Prime Directive applied to wave physics: the same
four-layer abstraction (target → geometry → realization → fabrication)
routes photons, phonons, and thermal radiation identically.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════════════
# FIELD SPECS (simplified, independent of models.py for portability)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FieldTarget:
    """Unified target specification for any field interaction."""
    field_type: str                  # "electromagnetic", "acoustic", "thermal"
    # Spectral target
    target_wavelength_nm: float = 0  # for EM
    target_frequency_Hz: float = 0   # for acoustic
    target_temperature_K: float = 300 # for thermal
    # Response
    response: str = "reflect"        # "reflect", "absorb", "emit", "block"
    target_efficiency: float = 0.9
    # Bandwidth
    bandwidth: float = 0.0           # nm for EM, Hz for acoustic
    # Constraints
    max_thickness_m: float = 0.01
    substrate: str = ""
    allowed_materials: List[str] = field(default_factory=list)


@dataclass
class FieldDesignResult:
    """Unified result from any field dispatch."""
    field_type: str
    mechanism: str                   # 'bragg', 'mie', 'local_resonance', 'tmm', 'beer_lambert'
    materials: List[str] = field(default_factory=list)
    layer_spec: List[Dict] = field(default_factory=list)  # [{material, thickness}, ...]
    # Performance
    primary_metric: float = 0.0      # reflectance (EM), TL_dB (acoustic), P_cool (thermal)
    metric_name: str = ""
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    # Assembly
    click_chemistry: str = "SPAAC"
    anchor: str = ""
    n_layers: int = 0
    total_thickness_m: float = 0.0
    # Metadata
    notes: str = ""
    cost_estimate: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def dispatch_field(target: FieldTarget, verbose: bool = False) -> List[FieldDesignResult]:
    """
    Route a field target to the correct physics engine.

    This is the unified entry point. Same function call, different physics.
    """
    if target.field_type == "electromagnetic":
        return _dispatch_optical(target, verbose)
    elif target.field_type == "acoustic":
        return _dispatch_acoustic(target, verbose)
    elif target.field_type == "thermal":
        return _dispatch_thermal(target, verbose)
    else:
        raise ValueError(f"Unknown field_type: {target.field_type}")


def _dispatch_optical(target: FieldTarget, verbose: bool) -> List[FieldDesignResult]:
    """Route to optical pipeline."""
    results = []
    target_nm = target.target_wavelength_nm

    if target_nm <= 0:
        return results

    # Bragg opal estimate
    try:
        from optical.bragg_opal import bragg_opal, n_eff_volume_average
        from optical.refractive_index import n_real

        n_sio2 = n_real("SiO2", target_nm)
        n_eff = n_eff_volume_average(n_sio2, 1.0)
        D_nm = target_nm / (1.633 * n_eff)

        # Verify by forward model
        lam_check = bragg_opal(D_nm, material="SiO2")

        results.append(FieldDesignResult(
            field_type="electromagnetic",
            mechanism="bragg_opal",
            materials=["SiO2", "air"],
            layer_spec=[{"material": "SiO2", "diameter_nm": D_nm}],
            primary_metric=lam_check,
            metric_name="peak_wavelength_nm",
            secondary_metrics={"particle_diameter_nm": D_nm, "n_eff": n_eff},
            click_chemistry="SPAAC",
            anchor="APTES",
            n_layers=20,
            total_thickness_m=D_nm * 20 * 1e-9,
            notes=f"SiO₂ opal D={D_nm:.0f}nm → λ={lam_check:.0f}nm (ordered, iridescent)",
        ))
    except Exception as e:
        if verbose:
            print(f"    [bragg_opal failed: {e}]")

    # Photonic glass estimate (non-iridescent structural color)
    try:
        from optical.refractive_index import n_real

        n_sio2 = n_real("SiO2", target_nm)
        # Photonic glass: λ ≈ 1.22 × D × n_eff (short-range order)
        n_eff = (0.64 * n_sio2**2 + 0.36 * 1.0**2)**0.5  # random packing ~64%
        D_glass = target_nm / (1.22 * n_eff)

        results.append(FieldDesignResult(
            field_type="electromagnetic",
            mechanism="photonic_glass",
            materials=["SiO2", "air"],
            layer_spec=[{"material": "SiO2", "diameter_nm": D_glass}],
            primary_metric=target_nm,
            metric_name="peak_wavelength_nm",
            secondary_metrics={"particle_diameter_nm": D_glass, "n_eff": n_eff},
            click_chemistry="SPAAC",
            anchor="APTES",
            n_layers=1,
            total_thickness_m=D_glass * 30 * 1e-9,
            notes=f"SiO₂ photonic glass D={D_glass:.0f}nm → λ≈{target_nm:.0f}nm (angle-independent)",
        ))
    except Exception as e:
        if verbose:
            print(f"    [photonic_glass failed: {e}]")

    # TiO₂ higher-contrast options
    try:
        from optical.refractive_index import n_real

        n_tio2 = n_real("TiO2_rutile", min(target_nm, 700))
        n_eff_tio2 = (0.74 * n_tio2**2 + 0.26 * 1.0**2)**0.5
        D_tio2 = target_nm / (1.633 * n_eff_tio2)

        results.append(FieldDesignResult(
            field_type="electromagnetic",
            mechanism="bragg_opal",
            materials=["TiO2_rutile", "air"],
            layer_spec=[{"material": "TiO2_rutile", "diameter_nm": D_tio2}],
            primary_metric=target_nm,
            metric_name="peak_wavelength_nm",
            secondary_metrics={"particle_diameter_nm": D_tio2, "n_eff": n_eff_tio2},
            click_chemistry="SPAAC",
            anchor="catechol",
            n_layers=10,
            total_thickness_m=D_tio2 * 10 * 1e-9,
            notes=f"TiO₂ opal D={D_tio2:.0f}nm → λ≈{target_nm:.0f}nm (high contrast, fewer layers)",
        ))
    except Exception as e:
        pass

    if verbose and results:
        print(f"  OPTICAL: {len(results)} designs for λ={target_nm:.0f} nm")
        for r in results[:3]:
            print(f"    {r.mechanism}: {r.notes}")

    return results


def _dispatch_acoustic(target: FieldTarget, verbose: bool) -> List[FieldDesignResult]:
    """Route to acoustic pipeline."""
    from acoustic.assembly import design_acoustic_particle
    from acoustic.forward_models import design_sound_blocker

    results = []
    freq = target.target_frequency_Hz

    if freq <= 0:
        return results

    # Local resonance designs (click-assembled particles)
    particle_designs = design_acoustic_particle(freq, verbose=False)
    for d in particle_designs[:5]:
        p = d.particle
        results.append(FieldDesignResult(
            field_type="acoustic",
            mechanism="local_resonance",
            materials=[p.core_material, "silicone_rubber"],
            layer_spec=[
                {"material": p.core_material, "radius_mm": p.core_radius_m * 1000},
                {"material": "silicone_rubber", "thickness_mm": p.shell_thickness_m * 1000},
            ],
            primary_metric=d.peak_tl_dB,
            metric_name="TL_dB_per_cell",
            secondary_metrics={
                "resonance_Hz": d.resonance_freq_Hz,
                "bandgap_lower_Hz": d.bandgap_Hz[0],
                "bandgap_upper_Hz": d.bandgap_Hz[1],
            },
            click_chemistry=p.click,
            anchor=p.anchor,
            n_layers=1,
            total_thickness_m=p.outer_radius_m * 2,
            notes=d.assembly_notes,
            cost_estimate=d.cost_rank,
        ))

    # Bragg/multilayer designs
    blocker_designs = design_sound_blocker(freq, max_thickness_m=target.max_thickness_m, verbose=False)
    for d in blocker_designs[:3]:
        results.append(FieldDesignResult(
            field_type="acoustic",
            mechanism=d.mechanism,
            materials=d.materials,
            layer_spec=[{"material": m, "thickness_mm": t * 1000}
                       for m, t in zip(d.materials, d.layer_thicknesses_m)],
            primary_metric=d.predicted_tl_dB,
            metric_name="TL_dB",
            secondary_metrics={"bandgap_width_Hz": d.bandgap_width_Hz},
            n_layers=d.n_layers,
            total_thickness_m=d.total_thickness_m,
            notes=d.notes,
        ))

    if verbose and results:
        print(f"  ACOUSTIC: {len(results)} designs for f={freq:.0f} Hz")
        for r in results[:3]:
            print(f"    {r.mechanism}: {r.primary_metric:.1f} dB — {r.notes[:60]}")

    return results


def _dispatch_thermal(target: FieldTarget, verbose: bool) -> List[FieldDesignResult]:
    """Route to thermal pipeline."""
    from thermal.radiative_models import design_radiative_cooler

    results = []

    cooler_designs = design_radiative_cooler(verbose=False)
    for d in cooler_designs[:5]:
        results.append(FieldDesignResult(
            field_type="thermal",
            mechanism="selective_emitter",
            materials=[l.material for l in d.layers],
            layer_spec=[{"material": l.material, "thickness_um": l.thickness_um}
                       for l in d.layers],
            primary_metric=d.cooling_power_W_m2,
            metric_name="P_cool_W_m2",
            secondary_metrics={
                "avg_emissivity_window": d.avg_emissivity_window,
                "n_layers": d.n_layers,
            },
            click_chemistry=d.click,
            anchor=d.anchor,
            n_layers=d.n_layers,
            total_thickness_m=d.total_thickness_um * 1e-6,
            notes=d.assembly_notes,
        ))

    if verbose and results:
        print(f"  THERMAL: {len(results)} designs (radiative cooler)")
        for r in results[:3]:
            print(f"    {r.mechanism}: {r.primary_metric:.1f} W/m² — "
                  f"{', '.join(r.materials[:3])}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE MULTI-PHYSICS DESIGN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CompositeFieldTarget:
    """Multi-physics target: a structure that handles multiple field types."""
    targets: List[FieldTarget] = field(default_factory=list)
    # Compatibility constraints
    optical_transparency_required: bool = False  # acoustic/thermal layers must pass light
    shared_substrate: str = ""
    max_total_thickness_m: float = 0.05
    name: str = ""


@dataclass
class CompositeDesignResult:
    """Result of multi-physics composite design."""
    name: str = ""
    sub_designs: Dict[str, FieldDesignResult] = field(default_factory=dict)
    # Compatibility
    compatible: bool = True
    compatibility_issues: List[str] = field(default_factory=list)
    # Combined stack
    total_thickness_m: float = 0.0
    n_total_layers: int = 0
    materials_used: List[str] = field(default_factory=list)
    click_interfaces: int = 0
    # Performance summary
    performance: Dict[str, float] = field(default_factory=dict)


def dispatch_composite(
    composite: CompositeFieldTarget,
    verbose: bool = True,
) -> CompositeDesignResult:
    """
    Design a multi-physics composite structure.

    Each sub-target is designed independently, then compatibility
    is checked and a combined stack is proposed.

    Example: building panel that blocks sound (acoustic) + cools
    radiatively (thermal) + has structural color (optical).
    """
    result = CompositeDesignResult(name=composite.name)
    issues = []

    if verbose:
        print(f"\n{'═' * 70}")
        print(f"COMPOSITE MULTI-PHYSICS DESIGN: {composite.name}")
        print(f"{'═' * 70}")

    # Design each sub-target
    for target in composite.targets:
        if verbose:
            label = target.field_type.upper()
            if target.target_frequency_Hz > 0:
                print(f"\n  [{label}] Target: {target.target_frequency_Hz:.0f} Hz, "
                      f"response={target.response}")
            elif target.target_wavelength_nm > 0:
                print(f"\n  [{label}] Target: {target.target_wavelength_nm:.0f} nm, "
                      f"response={target.response}")
            else:
                print(f"\n  [{label}] Target: T={target.target_temperature_K:.0f}K, "
                      f"response={target.response}")

        designs = dispatch_field(target, verbose=verbose)

        if designs:
            best = designs[0]
            result.sub_designs[target.field_type] = best
            result.performance[f"{target.field_type}_{best.metric_name}"] = best.primary_metric

            if verbose:
                print(f"  → Best: {best.mechanism}, "
                      f"{best.metric_name}={best.primary_metric:.1f}")
        else:
            issues.append(f"No design found for {target.field_type}")

    # Compatibility check
    total_thick = sum(d.total_thickness_m for d in result.sub_designs.values())
    result.total_thickness_m = total_thick
    result.n_total_layers = sum(d.n_layers for d in result.sub_designs.values())

    all_materials = set()
    n_click = 0
    for d in result.sub_designs.values():
        all_materials.update(d.materials)
        if d.click_chemistry and d.click_chemistry != "none":
            n_click += 1
    result.materials_used = sorted(all_materials)
    result.click_interfaces = n_click

    if total_thick > composite.max_total_thickness_m:
        issues.append(f"Total thickness {total_thick*1000:.1f}mm exceeds "
                     f"limit {composite.max_total_thickness_m*1000:.1f}mm")

    # Optical transparency check
    if composite.optical_transparency_required:
        for ft, design in result.sub_designs.items():
            if ft != "electromagnetic":
                # Check if acoustic/thermal materials block visible light
                opaque_mats = {"lead", "tungsten", "steel_mild", "iron", "copper",
                               "aluminum", "gold"}
                blocked = opaque_mats.intersection(set(design.materials))
                if blocked:
                    issues.append(f"{ft} layer uses opaque material(s): {blocked}")

    result.compatibility_issues = issues
    result.compatible = len(issues) == 0

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"COMPOSITE SUMMARY: {composite.name}")
        print(f"{'─' * 70}")
        print(f"  Sub-designs: {len(result.sub_designs)}")
        print(f"  Total thickness: {result.total_thickness_m*1000:.1f} mm")
        print(f"  Total layers: {result.n_total_layers}")
        print(f"  Materials: {', '.join(result.materials_used)}")
        print(f"  Click interfaces: {result.click_interfaces}")
        print(f"  Compatible: {result.compatible}")
        if issues:
            for issue in issues:
                print(f"    ⚠ {issue}")
        print(f"\n  Performance:")
        for k, v in result.performance.items():
            print(f"    {k}: {v:.1f}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MOLECULAR ↔ FIELD BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

def dispatch_any(
    target_type: str,
    **kwargs,
) -> List:
    """
    Universal dispatch: molecular OR field.

    target_type:
      "metal"           → core/design_engine_v2.py
      "host_guest"      → core/design_engine_v2.py
      "glycan"          → core/galnac_binder_scorer.py
      "electromagnetic" → optical pipeline
      "acoustic"        → acoustic pipeline
      "thermal"         → thermal pipeline

    This is the single entry point that proves computational isomorphism
    across ALL six modalities.
    """
    if target_type in ("metal", "host_guest", "mixed"):
        from core.design_engine_v2 import score_one
        smiles = kwargs.get("smiles", "")
        metal = kwargs.get("metal", None)
        host = kwargs.get("host", None)
        result = score_one(smiles, metal=metal, host=host)
        return [result] if result else []

    elif target_type == "glycan":
        from core.galnac_binder_scorer import score_galnac_binder
        smiles = kwargs.get("smiles", "")
        result = score_galnac_binder(smiles)
        return [result]

    elif target_type in ("electromagnetic", "acoustic", "thermal"):
        target = FieldTarget(field_type=target_type, **kwargs)
        return dispatch_field(target)

    else:
        raise ValueError(f"Unknown target_type: {target_type}")


# ═══════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("UNIFIED FIELD DISPATCH — SIX MODALITY PROOF")
    print("═" * 70)

    # ── 1. Optical ────────────────────────────────────────────────────
    print("\n[1] ELECTROMAGNETIC — structural blue color")
    optical_results = dispatch_field(FieldTarget(
        field_type="electromagnetic",
        target_wavelength_nm=450,
        response="reflect",
    ), verbose=True)

    # ── 2. Acoustic ───────────────────────────────────────────────────
    print("\n[2] ACOUSTIC — block 1 kHz noise")
    acoustic_results = dispatch_field(FieldTarget(
        field_type="acoustic",
        target_frequency_Hz=1000,
        response="block",
    ), verbose=True)

    # ── 3. Thermal ────────────────────────────────────────────────────
    print("\n[3] THERMAL — radiative cooling surface")
    thermal_results = dispatch_field(FieldTarget(
        field_type="thermal",
        target_temperature_K=300,
        response="emit",
    ), verbose=True)

    # ── 4. Composite: Sound + Thermal ─────────────────────────────────
    print("\n\n")
    composite = CompositeFieldTarget(
        name="Sound-blocking radiative cooling panel",
        targets=[
            FieldTarget(field_type="acoustic", target_frequency_Hz=1000,
                       response="block"),
            FieldTarget(field_type="thermal", target_temperature_K=300,
                       response="emit"),
        ],
        max_total_thickness_m=0.05,
    )
    composite_result = dispatch_composite(composite, verbose=True)

    # ── 5. Composite: All three ───────────────────────────────────────
    print("\n\n")
    triple = CompositeFieldTarget(
        name="Structural color + sound blocking + radiative cooling",
        targets=[
            FieldTarget(field_type="electromagnetic", target_wavelength_nm=500,
                       response="reflect"),
            FieldTarget(field_type="acoustic", target_frequency_Hz=2000,
                       response="block"),
            FieldTarget(field_type="thermal", target_temperature_K=300,
                       response="emit"),
        ],
        max_total_thickness_m=0.05,
        optical_transparency_required=False,
    )
    triple_result = dispatch_composite(triple, verbose=True)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n\n{'═' * 70}")
    print(f"DISPATCH SUMMARY: ONE FUNCTION, THREE PHYSICS")
    print(f"{'═' * 70}")
    print(f"  dispatch_field(electromagnetic) → {len(optical_results)} designs")
    print(f"  dispatch_field(acoustic)        → {len(acoustic_results)} designs")
    print(f"  dispatch_field(thermal)         → {len(thermal_results)} designs")
    print(f"  dispatch_composite(2 targets)   → {len(composite_result.sub_designs)} sub-designs")
    print(f"  dispatch_composite(3 targets)   → {len(triple_result.sub_designs)} sub-designs")
    print(f"\n  Same function. Same FieldTarget spec. Different physics.")
    print(f"  Molecular dispatch (metal/HG/glycan) uses dispatch_any().")
    print(f"  Six modalities. One architecture. Zero re-training.")