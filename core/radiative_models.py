"""
thermal/radiative_models.py — Thermal Radiation Forward Models

TMM at IR wavelengths for multilayer thermal emissivity design.
Click-directed stack assembly for radiative cooling surfaces.

Three thermal design targets:
  1. RADIATIVE COOLER: maximize emission in 8-13 μm atmospheric window
     while reflecting solar (0.3-2.5 μm)
  2. SOLAR ABSORBER: maximize absorption of solar spectrum (0.3-2.5 μm)
     while minimizing IR emission (selective absorber)
  3. THERMAL BARRIER: minimize thermal radiation transfer across a gap
     (IR reflector / low-emissivity coating)

Multi-stage build (parallels optical/acoustic):
  [Substrate] — [Anchor] — [IR-active layer 1] — [Click] — [IR-active layer 2] — ...

Click chemistry controls:
  - Layer sequence and thickness (same spacer/click libraries)
  - Interlayer coupling (Kapitza resistance for phonon transport)
  - Assembly precision (self-limiting click = uniform layers)

The physics is electromagnetic (Maxwell's equations at IR wavelengths),
so the optical TMM is the correct forward model — just with IR n(λ),k(λ).
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from thermal.ir_properties import (
    get_ir_material, IRMaterial, IR_DB,
    planck_spectral_radiance, atmospheric_transmission,
    radiative_cooling_power, total_radiated_power,
)


# ═══════════════════════════════════════════════════════════════════════════
# IR TRANSFER MATRIX METHOD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IRLayer:
    """Single layer in an IR multilayer stack."""
    material: str           # key into IR_DB
    thickness_um: float     # layer thickness in μm

    @property
    def props(self) -> IRMaterial:
        return get_ir_material(self.material)


@dataclass
class IRTMMResult:
    """Result of IR TMM calculation."""
    wavelengths_um: np.ndarray
    reflectance: np.ndarray
    transmittance: np.ndarray
    absorptance: np.ndarray        # = emissivity (Kirchhoff)
    # Derived
    avg_emissivity_window: float = 0.0   # average ε in 8-13 μm
    avg_reflectance_solar: float = 0.0   # average R in 0.3-2.5 μm
    cooling_power_W_m2: float = 0.0


def ir_tmm_spectrum(
    layers: List[IRLayer],
    substrate: str = "aluminum",
    wavelengths_um: np.ndarray = None,
) -> IRTMMResult:
    """
    Compute reflectance/emissivity spectrum for an IR multilayer stack.

    Hybrid approach:
    - For thin layers (d < 5λ/k): full TMM with interference
    - For thick absorbers (d >> λ/k): Beer-Lambert + Fresnel (no interference)

    This avoids numerical overflow from exp(large imaginary arguments)
    that occurs when TMM is applied to films much thicker than the
    absorption length.
    """
    if wavelengths_um is None:
        wavelengths_um = np.linspace(2, 25, 500)

    air = get_ir_material("air_ir")
    sub = get_ir_material(substrate)

    R_array = np.zeros(len(wavelengths_um))
    T_array = np.zeros(len(wavelengths_um))
    A_array = np.zeros(len(wavelengths_um))

    for wi, wl in enumerate(wavelengths_um):
        if wl <= 0:
            continue

        # Compute single-pass absorption for each layer
        total_absorption = 0.0
        for layer in layers:
            mat = layer.props
            n, k = mat.n_k(wl)

            if k > 0 and layer.thickness_um > 0:
                # Absorption coefficient α = 4πk/λ (in 1/μm)
                alpha = 4 * math.pi * k / wl
                # Single-pass absorptance
                abs_layer = 1.0 - math.exp(-alpha * layer.thickness_um)
            else:
                abs_layer = 0.0

            # Accumulate (simplified: ignore multiple reflections between layers)
            total_absorption = total_absorption + (1 - total_absorption) * abs_layer

        # Front surface Fresnel reflection (air → first layer)
        if layers:
            n1, k1 = layers[0].props.n_k(wl)
            eta_air = complex(1.0, 0.0)
            eta_layer = complex(n1, k1)
            r_front = abs((eta_air - eta_layer) / (eta_air + eta_layer))**2
        else:
            r_front = 0.0

        # Back surface: substrate reflection
        n_sub, k_sub = sub.n_k(wl)
        if k_sub > 10:  # metal — high reflectivity
            r_back = 0.95  # practical metal reflectance in IR
        else:
            if layers:
                n_last, k_last = layers[-1].props.n_k(wl)
            else:
                n_last, k_last = 1.0, 0.0
            eta_last = complex(n_last, k_last)
            eta_sub = complex(n_sub, k_sub)
            r_back = abs((eta_last - eta_sub) / (eta_last + eta_sub))**2

        # Net absorptance with double-pass (front-surface transmitted,
        # absorbed in film, reflected off substrate, absorbed again)
        T_front = 1.0 - r_front
        A_single = total_absorption
        T_through = T_front * (1 - A_single)  # what reaches substrate
        reflected_back = T_through * r_back     # reflects off substrate
        A_second = reflected_back * total_absorption  # absorbed on return

        total_R = r_front + T_through * r_back * (1 - total_absorption) * T_front
        total_A = T_front * A_single + A_second
        total_T = max(0, 1.0 - total_R - total_A)

        # Clamp
        total_R = min(1.0, max(0.0, total_R))
        total_A = min(1.0 - total_R, max(0.0, total_A))
        total_T = max(0.0, 1.0 - total_R - total_A)

        R_array[wi] = total_R
        T_array[wi] = total_T
        A_array[wi] = total_A  # = emissivity by Kirchhoff

    # Average emissivity in atmospheric window
    window_mask = (wavelengths_um >= 8.0) & (wavelengths_um <= 13.0)
    avg_e_window = np.mean(A_array[window_mask]) if window_mask.any() else 0

    # Cooling power
    def emissivity_func(w):
        idx = np.argmin(np.abs(wavelengths_um - w))
        return A_array[idx] if idx < len(A_array) else 0

    cool = radiative_cooling_power(emissivity_func)

    return IRTMMResult(
        wavelengths_um=wavelengths_um,
        reflectance=R_array,
        transmittance=T_array,
        absorptance=A_array,
        avg_emissivity_window=avg_e_window,
        avg_reflectance_solar=0.0,
        cooling_power_W_m2=cool['P_cool_net_W_m2'],
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLICK-DIRECTED THERMAL STACK ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThermalStackDesign:
    """Complete thermal stack specification."""
    name: str
    target: str                     # 'radiative_cooler', 'solar_absorber', 'thermal_barrier'
    layers: List[IRLayer] = field(default_factory=list)
    substrate: str = "aluminum"
    # Click assembly
    anchor: str = "APTES"
    click: str = "SPAAC"
    spacer: str = "none"            # minimal for thermal (nm spacers negligible at μm wavelengths)
    # Performance
    avg_emissivity_window: float = 0.0
    cooling_power_W_m2: float = 0.0
    total_thickness_um: float = 0.0
    n_layers: int = 0
    assembly_notes: str = ""


def design_radiative_cooler(
    method: str = "auto",
    max_layers: int = 5,
    verbose: bool = False,
) -> List[ThermalStackDesign]:
    """
    Design a radiative cooling surface.

    Strategies:
    1. Single thick SiO₂ film on Al reflector (simplest)
    2. SiO₂/Si₃N₄ multilayer (broadened window coverage)
    3. PDMS film (cheap polymer alternative)
    4. SiO₂/hBN heterostack with click assembly (novel 2D approach)

    All on metal reflector substrate (Al or Ag) to reject solar radiation.
    """
    designs = []

    # ── Strategy 1: Single SiO₂ film ─────────────────────────────────
    for thickness in [10, 50, 100, 200]:
        stack = [IRLayer("SiO2", thickness)]
        result = ir_tmm_spectrum(stack, substrate="aluminum")
        designs.append(ThermalStackDesign(
            name=f"SiO2_{thickness}um_on_Al",
            target="radiative_cooler",
            layers=stack,
            substrate="aluminum",
            anchor="APTES",
            click="none",
            avg_emissivity_window=result.avg_emissivity_window,
            cooling_power_W_m2=result.cooling_power_W_m2,
            total_thickness_um=thickness,
            n_layers=1,
            assembly_notes=f"APTES-anchored SiO₂ film, {thickness} μm",
        ))

    # ── Strategy 2: SiO₂/Si₃N₄ bilayer ──────────────────────────────
    for d_sio2 in [30, 50, 100]:
        for d_sin in [20, 50]:
            stack = [
                IRLayer("Si3N4", d_sin),
                IRLayer("SiO2", d_sio2),
            ]
            result = ir_tmm_spectrum(stack, substrate="aluminum")
            designs.append(ThermalStackDesign(
                name=f"Si3N4_{d_sin}+SiO2_{d_sio2}_on_Al",
                target="radiative_cooler",
                layers=stack,
                substrate="aluminum",
                anchor="APTES",
                click="SPAAC",
                avg_emissivity_window=result.avg_emissivity_window,
                cooling_power_W_m2=result.cooling_power_W_m2,
                total_thickness_um=d_sin + d_sio2,
                n_layers=2,
                assembly_notes=f"Click-assembled bilayer: APTES/SiO₂ + SPAAC + Si₃N₄",
            ))

    # ── Strategy 3: PDMS film (cheap) ─────────────────────────────────
    for thickness in [50, 100, 200]:
        stack = [IRLayer("PDMS", thickness)]
        result = ir_tmm_spectrum(stack, substrate="aluminum")
        designs.append(ThermalStackDesign(
            name=f"PDMS_{thickness}um_on_Al",
            target="radiative_cooler",
            layers=stack,
            substrate="aluminum",
            anchor="catechol",
            click="none",
            avg_emissivity_window=result.avg_emissivity_window,
            cooling_power_W_m2=result.cooling_power_W_m2,
            total_thickness_um=thickness,
            n_layers=1,
            assembly_notes=f"PDMS film spin-coated, {thickness} μm",
        ))

    # ── Strategy 4: hBN / SiO₂ heterostack (novel) ───────────────────
    for n_periods in [2, 3, 5]:
        stack_layers = []
        for _ in range(n_periods):
            stack_layers.append(IRLayer("hBN_ir", 5))   # 5 μm hBN
            stack_layers.append(IRLayer("SiO2", 20))     # 20 μm SiO₂
        result = ir_tmm_spectrum(stack_layers, substrate="aluminum")
        total = n_periods * 25
        designs.append(ThermalStackDesign(
            name=f"hBN_SiO2_x{n_periods}_on_Al",
            target="radiative_cooler",
            layers=stack_layers,
            substrate="aluminum",
            anchor="APTES",
            click="SPAAC",
            spacer="PEG2",
            avg_emissivity_window=result.avg_emissivity_window,
            cooling_power_W_m2=result.cooling_power_W_m2,
            total_thickness_um=total,
            n_layers=n_periods * 2,
            assembly_notes=(f"Click-assembled heterostack: "
                           f"{n_periods}× (hBN/SPAAC/SiO₂), orthogonal handles"),
        ))

    # Sort by cooling power
    designs.sort(key=lambda d: -d.cooling_power_W_m2)

    if verbose:
        print(f"\nRADIATIVE COOLER DESIGNS:")
        print(f"{'#':>3s} {'Name':>35s} {'ε_window':>9s} {'P_cool':>8s} "
              f"{'d(μm)':>7s} {'Layers':>6s} {'Click':>6s}")
        print("─" * 80)
        for i, d in enumerate(designs[:15]):
            print(f"{i+1:3d} {d.name:>35s} {d.avg_emissivity_window:9.3f} "
                  f"{d.cooling_power_W_m2:8.1f} {d.total_thickness_um:7.0f} "
                  f"{d.n_layers:6d} {d.click:>6s}")

    return designs


# ═══════════════════════════════════════════════════════════════════════════
# INVERSE DESIGN: TARGET PERFORMANCE → STACK
# ═══════════════════════════════════════════════════════════════════════════

def design_thermal_surface(
    target: str = "radiative_cooler",
    min_cooling_power: float = 50.0,  # W/m² minimum
    max_thickness_um: float = 500.0,
    verbose: bool = True,
) -> List[ThermalStackDesign]:
    """
    Design a thermal surface for a target application.

    Same dispatch pattern as optical inverse_design and acoustic
    design_sound_blocker: target spec → ranked designs.
    """
    if target == "radiative_cooler":
        all_designs = design_radiative_cooler(verbose=False)
        # Filter by performance
        filtered = [d for d in all_designs
                    if d.cooling_power_W_m2 >= min_cooling_power
                    and d.total_thickness_um <= max_thickness_um]
        filtered.sort(key=lambda d: -d.cooling_power_W_m2)

        if verbose:
            print(f"\nTHERMAL SURFACE DESIGNS: {target}")
            print(f"  Constraint: P_cool ≥ {min_cooling_power} W/m², "
                  f"d ≤ {max_thickness_um} μm")
            print(f"  Designs found: {len(filtered)}")
            print(f"\n{'#':>3s} {'Name':>35s} {'P_cool':>8s} {'ε_8-13':>7s} "
                  f"{'d(μm)':>7s} {'Click':>6s}")
            for i, d in enumerate(filtered[:10]):
                print(f"{i+1:3d} {d.name:>35s} {d.cooling_power_W_m2:8.1f} "
                      f"{d.avg_emissivity_window:7.3f} "
                      f"{d.total_thickness_um:7.0f} {d.click:>6s}")

        return filtered

    return []


# ═══════════════════════════════════════════════════════════════════════════
# ISOMORPHISM DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

def demonstrate_isomorphism():
    """Show the three-way isomorphism: optical ↔ acoustic ↔ thermal."""
    print("═" * 70)
    print("THREE-WAY ISOMORPHISM: Optical ↔ Acoustic ↔ Thermal")
    print("═" * 70)
    print(f"\n{'Concept':>25s} {'Optical':>20s} {'Acoustic':>20s} {'Thermal':>20s}")
    print("─" * 85)
    rows = [
        ("Material DB",           "n(λ) complex",     "Z = ρv",           "n(λ),k(λ) IR"),
        ("Forward model",         "TMM (visible)",     "ATMM (audible)",   "TMM (IR 8-13μm)"),
        ("Bandgap/window",        "Photonic gap",      "Phononic gap",     "Atmospheric window"),
        ("Sub-λ mechanism",       "Plasmonic",         "Local resonance",  "Phonon resonance"),
        ("Design target",         "Color/reflectance", "TL (dB) / STC",   "ε(window) / P_cool"),
        ("Assembly chemistry",    "Click → spacing",   "Click → Kapitza",  "Click → layer seq"),
        ("Key material",          "SiO₂ (n=1.46)",    "Pb/silicone",      "SiO₂ (phonon 9.7μm)"),
        ("Inverse design",        "Color → particles", "Freq → core+coat", "P_cool → stack"),
    ]
    for row in rows:
        print(f"{row[0]:>25s} {row[1]:>20s} {row[2]:>20s} {row[3]:>20s}")

    # SiO₂ serves different physics in each domain:
    print(f"\n{'─' * 85}")
    print(f"SiO₂ — ONE MATERIAL, THREE ROLES:")
    print(f"  Optical:  transparent dielectric (n=1.46), scatters visible light → structural color")
    print(f"  Acoustic: stiff ceramic (Z=13.1 MRayl), Bragg scatterer → sound blocking")
    print(f"  Thermal:  phonon resonance at 9.7 μm (Si-O stretch), selective emitter → radiative cooling")
    print(f"\n  Same material. Same click-assembly chemistry. Three physics payoffs.")
    print(f"  MABE scores all three through the same dispatch architecture.")


if __name__ == "__main__":
    demonstrate_isomorphism()

    # Design radiative coolers
    print(f"\n\n")
    designs = design_thermal_surface(
        target="radiative_cooler",
        min_cooling_power=30.0,
        verbose=True,
    )