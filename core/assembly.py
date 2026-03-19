"""
acoustic/assembly.py — Click-Directed Acoustic Particle Assembly

Mirrors optical/click_linker.py: the same anchor → spacer → click chain
that controls optical shell thickness and inter-particle spacing also
controls acoustic properties:

  Optical:  shell n(λ) → Mie scattering → color
  Acoustic: shell G, Z → spring constant K → resonance frequency

The click chemistry linker IS the soft spring in acoustic local resonance.
Same chemistry, different physics, same design pipeline.

Multi-stage build:
  [Heavy Core] — [Anchor] — [Spacer] — [Click] ··· [Click] — [Spacer] — [Anchor] — [Heavy Core]

Core: high-Z material (Pb, W, steel, BaTiO₃) → provides inertial mass
Shell: anchor + spacer + click → provides soft "spring" for resonance
Gap:  2 × (L_anchor + L_spacer + L_click_half) → inter-particle spacing
Assembly: ordered (Bragg) or disordered (broadband scattering)

Key insight: the click bond controls Kapitza resistance at the interface.
Covalent triazole (CuAAC) has lower R_K → more phonon transmission.
Van der Waals gap has higher R_K → more phonon scattering.
This makes assembly chemistry a tunable phonon engineering variable.

Reuses anchor/spacer/click databases from optical/click_linker.py.
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from acoustic.impedance_db import get_material, AcousticMaterial


# ═══════════════════════════════════════════════════════════════════════════
# ACOUSTIC SHELL PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShellAcousticProperties:
    """Acoustic properties of the anchor+spacer+click shell layer."""
    total_thickness_nm: float = 0.0
    density_kg_m3: float = 1100.0       # organic shell ~1.1 g/cm³
    shear_modulus_Pa: float = 1e6       # soft shell ~1 MPa
    v_longitudinal_m_s: float = 1500.0
    Z_MRayl: float = 1.65
    kapitza_R_m2K_W: float = 1e-8       # interface thermal resistance


# Shell material properties by component type
# Shear modulus is the key acoustic parameter — determines spring constant
SHELL_SHEAR_MODULUS = {
    # Anchors (thin, rigid molecular monolayer)
    'silane':               1e9,    # rigid covalent Si-O-Si
    'catechol_chelation':   5e8,    # stiff bidentate
    'P-O-M chelation':     5e8,
    'Au-S covalent':       2e8,    # thiol SAM, moderate
    'C-C covalent':        1e9,    # diazonium, rigid
    'pi-pi stacking':      5e7,    # non-covalent, soft
    'copolymer incorporation': 1e8,

    # Spacers
    'PEG': 1e6,                     # very soft, hydrated polymer chain
    'alkyl': 5e7,                   # stiffer, van der Waals packed chain

    # Click products
    'triazole_SPAAC': 5e8,         # 1,2,3-triazole, rigid heterocycle
    'triazole_CuAAC': 5e8,         # same product
    'thioether': 1e8,              # thiol-maleimide product, moderate
    'pyridazine': 5e8,             # IEDDA product, rigid
}

# Kapitza resistance by click bond type (m²K/W)
# Covalent bonds transmit phonons better → lower R_K
KAPITZA_R_BY_CLICK = {
    'SPAAC':            5e-9,   # covalent triazole, good phonon coupling
    'CuAAC':            5e-9,   # same triazole product
    'thiol_maleimide':  8e-9,   # thioether, slightly softer
    'IEDDA':            4e-9,   # rigid pyridazine, very good coupling
    'van_der_waals':    5e-8,   # no covalent bond — 10× higher resistance
    'none':             1e-7,   # air gap between particles
}


# ═══════════════════════════════════════════════════════════════════════════
# ACOUSTIC PARTICLE SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AcousticParticleSpec:
    """Full specification of a core-shell acoustic particle."""
    # Core
    core_material: str
    core_radius_m: float
    core_density: float = 0.0
    core_Z_MRayl: float = 0.0

    # Shell (from click chain)
    anchor: str = "APTES"
    spacer: str = "PEG4"
    click: str = "SPAAC"
    shell_thickness_m: float = 0.0
    shell_props: Optional[ShellAcousticProperties] = None

    # Assembly
    matrix_material: str = "epoxy"
    filling_fraction: float = 0.4
    assembly_type: str = "disordered"     # 'ordered_fcc', 'disordered', 'linear'

    # Derived
    outer_radius_m: float = 0.0
    effective_diameter_m: float = 0.0     # includes inter-particle gap
    gap_nm: float = 0.0

    # Predicted performance
    resonance_freq_Hz: float = 0.0
    bandgap_lower_Hz: float = 0.0
    bandgap_upper_Hz: float = 0.0
    peak_tl_dB: float = 0.0


def compute_shell_properties(
    anchor_type: str = "silane",
    spacer_type: str = "PEG",
    click_type: str = "SPAAC",
    anchor_thickness_nm: float = 0.8,
    spacer_thickness_nm: float = 1.4,
    click_half_nm: float = 0.9,
) -> ShellAcousticProperties:
    """
    Compute effective acoustic properties of the shell layer.

    Shell is a composite of anchor + spacer + click layers.
    Effective shear modulus: series model (softest layer dominates).
    """
    total_nm = anchor_thickness_nm + spacer_thickness_nm + click_half_nm

    if total_nm == 0:
        return ShellAcousticProperties()

    # Shear modulus: series average (reciprocal mean — soft layer dominates)
    G_anchor = SHELL_SHEAR_MODULUS.get(anchor_type, 1e8)
    G_spacer = SHELL_SHEAR_MODULUS.get(spacer_type, 1e6)
    G_click = SHELL_SHEAR_MODULUS.get(f'triazole_{click_type}', 5e8)

    # Weight by thickness fraction
    f_a = anchor_thickness_nm / total_nm
    f_s = spacer_thickness_nm / total_nm
    f_c = click_half_nm / total_nm

    # Series model: 1/G_eff = Σ fᵢ/Gᵢ
    inv_G = f_a / G_anchor + f_s / G_spacer + f_c / G_click
    G_eff = 1.0 / inv_G if inv_G > 0 else 1e6

    # Density: volume-weighted average (~organic, ~1100 kg/m³)
    rho_shell = 1100.0

    # Sound speed from G and density: v_T = √(G/ρ), v_L ≈ v_T × 2 for polymers
    v_T = math.sqrt(G_eff / rho_shell)
    v_L = v_T * 2.0  # approximate for soft organics

    # Impedance
    Z = rho_shell * v_L

    # Kapitza resistance at the click interface
    R_K = KAPITZA_R_BY_CLICK.get(click_type, 1e-8)

    return ShellAcousticProperties(
        total_thickness_nm=total_nm,
        density_kg_m3=rho_shell,
        shear_modulus_Pa=G_eff,
        v_longitudinal_m_s=v_L,
        Z_MRayl=Z / 1e6,
        kapitza_R_m2K_W=R_K,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PARTICLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

# Anchor thickness library (nm) — reusing optical values
ANCHOR_THICKNESS = {
    "APTES": 0.8, "MPTMS": 0.7, "catechol": 0.6,
    "phosphonate": 0.5, "thiol": 0.3, "diazonium": 0.5,
    "pyrene": 0.35, "direct_azide": 0.0,
}

SPACER_THICKNESS = {
    "none": 0.0, "PEG2": 0.7, "PEG4": 1.4, "PEG8": 2.8,
    "PEG12": 4.2, "alkyl_C6": 0.8, "alkyl_C11": 1.4,
}

CLICK_HALF_LENGTH = {
    "SPAAC": 0.9, "CuAAC": 0.4, "thiol_maleimide": 0.3, "IEDDA": 1.0,
}

# Core material → compatible anchors (same as optical)
CORE_ANCHOR_COMPAT = {
    "lead":             ["catechol", "phosphonate"],
    "tungsten":         ["catechol", "phosphonate"],
    "steel_mild":       ["catechol", "phosphonate", "APTES"],
    "barium_titanate":  ["catechol", "phosphonate", "APTES"],
    "silicon":          ["APTES", "MPTMS"],
    "silica_fused":     ["APTES", "MPTMS"],
    "gold":             ["thiol"],
    "iron":             ["catechol", "phosphonate"],
    "titanium":         ["catechol", "phosphonate", "APTES"],
    "alumina":          ["catechol", "phosphonate", "APTES"],
    "PZT":              ["catechol", "phosphonate", "APTES"],
}


def build_acoustic_particle(
    core_material: str,
    core_radius_m: float,
    coating_material: str = "silicone_rubber",
    coating_thickness_m: float = 0.0,   # 0 = auto-calculate
    anchor: str = "auto",
    spacer: str = "PEG4",
    click: str = "SPAAC",
    matrix_material: str = "epoxy",
    filling_fraction: float = 0.4,
    target_freq_Hz: float = 0.0,        # if set, auto-size coating
) -> AcousticParticleSpec:
    """
    Build a complete acoustic core-shell particle specification.

    Architecture:
      [Heavy Core] — [click interface] — [Bulk Soft Coating] — [click interface] — [Matrix]

    The BULK COATING (silicone rubber, PDMS, etc.) provides the spring constant
    that determines resonance frequency. Coating thickness ~ mm for audible.

    The CLICK CHEMISTRY is at the core-coating and coating-matrix interfaces.
    It's nanometer-scale — negligible for acoustic wavelength, but controls:
      - Interfacial adhesion (delamination resistance)
      - Kapitza resistance (phonon transmission at interface)
      - Assembly precision (self-limiting click = uniform coating)

    If target_freq_Hz > 0 and coating_thickness_m == 0, auto-calculates
    coating thickness to hit the target resonance.
    """
    core = get_material(core_material)
    coating = get_material(coating_material)

    # Auto-select anchor
    if anchor == "auto":
        compatible = CORE_ANCHOR_COMPAT.get(core_material, ["catechol"])
        anchor = compatible[0]

    # Auto-calculate coating thickness from target frequency
    if target_freq_Hz > 0 and coating_thickness_m == 0:
        # f_res = (1/2π)√(K/m), K = 4πGr(R/d), m = (4/3)πr³ρ
        # Solving for d: d = G × r × R / (m × (2πf)²)
        # Simplified: d ≈ v_coating / (2π f_target) for order-of-magnitude
        if coating.v_transverse_m_s > 0:
            v_coat = coating.v_transverse_m_s
        else:
            v_coat = coating.v_longitudinal_m_s * 0.1
        # Better estimate from resonance formula
        m_core = (4/3) * math.pi * core_radius_m**3 * core.density_kg_m3
        G_coat = coating.density_kg_m3 * v_coat**2
        omega_target = 2 * math.pi * target_freq_Hz
        # K_needed = m × ω²
        K_needed = m_core * omega_target**2
        # K = 4π G r (R/d), solve for d with R ≈ r + d
        # Iterate: start with d = r/3
        d_est = core_radius_m / 3
        for _ in range(5):
            R_est = core_radius_m + d_est
            K_est = 4 * math.pi * G_coat * core_radius_m * (R_est / d_est) if d_est > 0 else 1e20
            d_est = 4 * math.pi * G_coat * core_radius_m * (core_radius_m + d_est) / K_needed
            d_est = max(1e-4, min(d_est, core_radius_m * 3))
        coating_thickness_m = d_est

    if coating_thickness_m == 0:
        coating_thickness_m = core_radius_m * 0.3  # default: 30% of core

    # Click interface properties (thin, at core-coating boundary)
    anchor_nm = ANCHOR_THICKNESS.get(anchor, 0.6)
    spacer_nm = SPACER_THICKNESS.get(spacer, 1.4)
    click_nm = CLICK_HALF_LENGTH.get(click, 0.5)

    shell_click = compute_shell_properties(
        anchor_type=anchor.split("_")[0] if "_" in anchor else
                    ("silane" if anchor in ("APTES", "MPTMS") else
                     "catechol_chelation" if anchor == "catechol" else
                     "P-O-M chelation" if anchor == "phosphonate" else
                     "Au-S covalent" if anchor == "thiol" else "silane"),
        spacer_type="PEG" if "PEG" in spacer else
                    "alkyl" if "alkyl" in spacer else "PEG",
        click_type=click,
        anchor_thickness_nm=anchor_nm,
        spacer_thickness_nm=spacer_nm,
        click_half_nm=click_nm,
    )

    outer_r = core_radius_m + coating_thickness_m
    gap_nm = 2 * (anchor_nm + spacer_nm + click_nm)  # click interface gap only

    spec = AcousticParticleSpec(
        core_material=core_material,
        core_radius_m=core_radius_m,
        core_density=core.density_kg_m3,
        core_Z_MRayl=core.Z_MRayl,
        anchor=anchor,
        spacer=spacer,
        click=click,
        shell_thickness_m=coating_thickness_m,
        shell_props=shell_click,
        matrix_material=matrix_material,
        filling_fraction=filling_fraction,
        outer_radius_m=outer_r,
        effective_diameter_m=2 * outer_r,
        gap_nm=gap_nm,
    )

    # Predict resonance using local_resonance model with BULK coating
    from acoustic.forward_models import local_resonance
    lr = local_resonance(
        core_material=core_material,
        coating_material=coating_material,
        core_radius_m=core_radius_m,
        coating_thickness_m=coating_thickness_m,
        matrix_material=matrix_material,
        filling_fraction=filling_fraction,
    )
    spec.resonance_freq_Hz = lr.resonance_freq_Hz
    spec.bandgap_lower_Hz = lr.bandgap_lower_Hz
    spec.bandgap_upper_Hz = lr.bandgap_upper_Hz
    spec.peak_tl_dB = lr.peak_attenuation_dB_per_cell

    return spec


# ═══════════════════════════════════════════════════════════════════════════
# INVERSE DESIGN: TARGET FREQUENCY → PARTICLE SPEC
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AcousticDesign:
    """Complete design for a sound-blocking particle system."""
    particle: AcousticParticleSpec
    resonance_freq_Hz: float
    freq_error_pct: float          # |f_pred - f_target| / f_target
    bandgap_Hz: Tuple[float, float] = (0, 0)
    peak_tl_dB: float = 0.0
    total_thickness_mm: float = 0.0
    assembly_notes: str = ""
    cost_rank: str = ""            # "low", "medium", "high"


def design_acoustic_particle(
    target_freq_Hz: float,
    max_particle_radius_m: float = 0.02,
    core_options: Optional[List[str]] = None,
    coating_options: Optional[List[str]] = None,
    spacer_options: Optional[List[str]] = None,
    click_options: Optional[List[str]] = None,
    matrix: str = "epoxy",
    verbose: bool = False,
) -> List[AcousticDesign]:
    """
    Design a click-assembled acoustic particle for a target frequency.

    Searches core × coating × size × click combinations.
    Coating is the BULK soft layer (silicone, PDMS, rubber).
    Click is the INTERFACE chemistry (controls Kapitza R, assembly precision).
    """
    if core_options is None:
        core_options = ["lead", "tungsten", "steel_mild", "barium_titanate",
                        "iron", "copper"]
    if coating_options is None:
        coating_options = ["silicone_rubber", "PDMS", "natural_rubber",
                           "polyurethane", "PVDF"]
    if spacer_options is None:
        spacer_options = ["PEG4"]
    if click_options is None:
        click_options = ["SPAAC", "CuAAC"]

    designs = []

    for core_mat in core_options:
        for coat_mat in coating_options:
            for click in click_options:
                # Scan core radius
                for r_mm in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
                    r_m = r_mm * 1e-3
                    if r_m > max_particle_radius_m:
                        continue

                    try:
                        spec = build_acoustic_particle(
                            core_material=core_mat,
                            core_radius_m=r_m,
                            coating_material=coat_mat,
                            click=click,
                            matrix_material=matrix,
                            target_freq_Hz=target_freq_Hz,
                        )
                    except (KeyError, Exception):
                        continue

                    if spec.resonance_freq_Hz <= 0:
                        continue

                    f_err = abs(spec.resonance_freq_Hz - target_freq_Hz) / target_freq_Hz
                    if f_err > 0.5:
                        continue

                    cost = "low"
                    if core_mat in ("gold", "platinum", "tungsten"):
                        cost = "high"
                    elif core_mat in ("barium_titanate", "copper"):
                        cost = "medium"

                    designs.append(AcousticDesign(
                        particle=spec,
                        resonance_freq_Hz=spec.resonance_freq_Hz,
                        freq_error_pct=f_err * 100,
                        bandgap_Hz=(spec.bandgap_lower_Hz, spec.bandgap_upper_Hz),
                        peak_tl_dB=spec.peak_tl_dB,
                        total_thickness_mm=spec.outer_radius_m * 2 * 1000,
                        assembly_notes=(f"{core_mat} r={r_mm}mm + "
                                       f"{coat_mat} coat + {click} interface → {matrix}"),
                        cost_rank=cost,
                    ))

    # Sort by frequency accuracy, then by TL
    designs.sort(key=lambda d: (d.freq_error_pct, -d.peak_tl_dB))

    if verbose and designs:
        print(f"\nACOUSTIC PARTICLE DESIGNS for {target_freq_Hz:.0f} Hz:")
        print(f"{'#':>3s} {'Core':>10s} {'Coating':>15s} {'r(mm)':>6s} "
              f"{'Click':>6s} {'f_res':>8s} {'err%':>5s} {'TL':>5s} "
              f"{'d_coat(mm)':>10s} {'Cost':>6s}")
        for i, d in enumerate(designs[:15]):
            p = d.particle
            coat_mm = p.shell_thickness_m * 1000
            print(f"{i+1:3d} {p.core_material:>10s} "
                  f"{'silicone_rubber':>15s} {p.core_radius_m*1000:6.1f} "
                  f"{p.click:>6s} "
                  f"{d.resonance_freq_Hz:8.0f} {d.freq_error_pct:5.1f} "
                  f"{d.peak_tl_dB:5.1f} {coat_mm:10.2f} {d.cost_rank:>6s}")

    return designs


# ═══════════════════════════════════════════════════════════════════════════
# ISOMORPHISM: OPTICAL ↔ ACOUSTIC DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

def demonstrate_isomorphism():
    """
    Show that the same design chain works for both optical and acoustic:

    Optical: target color → particle size → click shell → assembly → reflectance
    Acoustic: target freq → particle size → click shell → assembly → transmission loss

    Same anchor/spacer/click databases. Same assembly logic.
    Different physics forward model.
    """
    print("═" * 70)
    print("COMPUTATIONAL ISOMORPHISM: Optical ↔ Acoustic")
    print("═" * 70)

    # Same particle, two physics:
    # Silica core, 200nm radius, PEG4 spacer, SPAAC click
    print("\nSame particle design, two physics domains:")
    print(f"  Core: silica_fused, r = 200 nm")
    print(f"  Shell: APTES anchor + PEG4 spacer + SPAAC click")

    # Optical properties
    shell_opt = compute_shell_properties("silane", "PEG", "SPAAC", 0.8, 1.4, 0.9)
    gap_nm_opt = 2 * (0.8 + 1.4 + 0.9)
    d_eff_nm = 400 + gap_nm_opt
    print(f"\n  OPTICAL:")
    print(f"    Shell n_eff ≈ 1.46 (organic)")
    print(f"    Gap: {gap_nm_opt:.1f} nm → d_eff = {d_eff_nm:.0f} nm")
    print(f"    → Photonic glass peak λ ≈ {d_eff_nm * 1.633 * 1.35:.0f} nm")

    # Acoustic properties
    print(f"\n  ACOUSTIC:")
    print(f"    Shell G_eff = {shell_opt.shear_modulus_Pa:.1e} Pa")
    print(f"    Shell v_L = {shell_opt.v_longitudinal_m_s:.0f} m/s")
    print(f"    Shell Z = {shell_opt.Z_MRayl:.2f} MRayl")
    print(f"    Kapitza R = {shell_opt.kapitza_R_m2K_W:.1e} m²K/W")

    # Now show a real acoustic design
    print(f"\n" + "─" * 70)
    print(f"Lead core + silicone rubber coating + click interface:")

    spec = build_acoustic_particle(
        "lead", 0.005,  # 5mm radius
        coating_material="silicone_rubber",
        spacer="PEG4", click="SPAAC", matrix_material="epoxy",
        target_freq_Hz=1000,
    )
    print(f"  Core: lead, r = 5 mm, Z = {spec.core_Z_MRayl:.1f} MRayl")
    print(f"  Coating: silicone_rubber, d = {spec.shell_thickness_m*1000:.2f} mm")
    print(f"  Interface: catechol + PEG4 + SPAAC click")
    print(f"  Kapitza R at interface: {spec.shell_props.kapitza_R_m2K_W:.1e} m²K/W")
    print(f"  Resonance: {spec.resonance_freq_Hz:.0f} Hz")
    print(f"  Bandgap: {spec.bandgap_lower_Hz:.0f} – {spec.bandgap_upper_Hz:.0f} Hz")
    print(f"  Peak TL: {spec.peak_tl_dB:.1f} dB/cell")

    # Compare click chemistries — same core+coating, different interface
    print(f"\n" + "─" * 70)
    print(f"CLICK CHEMISTRY COMPARISON (same core+coating, different interface):")
    print(f"  Core: lead 5mm, Coating: silicone_rubber, Target: 1 kHz")
    print(f"{'Click':>15s} {'Kapitza R':>12s} {'f_res':>8s} {'Bandgap':>18s} {'TL/cell':>8s}")
    for click_name in ["SPAAC", "CuAAC", "thiol_maleimide", "IEDDA"]:
        sp = build_acoustic_particle("lead", 0.005,
                                      coating_material="silicone_rubber",
                                      click=click_name, target_freq_Hz=1000)
        rk = KAPITZA_R_BY_CLICK.get(click_name, 1e-8)
        gap = f"{sp.bandgap_lower_Hz:.0f}-{sp.bandgap_upper_Hz:.0f}"
        print(f"{click_name:>15s} {rk:12.1e} {sp.resonance_freq_Hz:8.0f} "
              f"{gap:>18s} {sp.peak_tl_dB:8.1f}")


if __name__ == "__main__":
    demonstrate_isomorphism()

    # Design API test
    print(f"\n\n")
    for target in [200, 1000, 5000]:
        designs = design_acoustic_particle(
            target_freq_Hz=target,
            verbose=True,
        )