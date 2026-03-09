"""
optical/multi_shell.py — Multi-Shell Compositor

Stacks multiple click-attached optical layers on a commodity particle core.
Each layer is independently addressable via orthogonal click chemistry —
SPAAC at interface 1, CuAAC at interface 2, thiol-maleimide at interface 3,
IEDDA at interface 4. Up to 4 independent shells on a single core.

Architecture:
  core (SiO₂, TiO₂, PS, ...) → shell 1 → shell 2 → shell 3 → ...
        interface 1    interface 2    interface 3

Each shell contributes:
  Δn(λ): refractive index modification (index shells, chromophore bulk n)
  Δk(λ): wavelength-selective absorption (chromophores, d-d, CT bands)
  Δr:    physical thickness → changes effective diameter → shifts S(q)

The composite particle's optical response is computed by recursive
Bohren-Huffman effective medium (inside→out), then fed through
Mie → S(q) → R(λ) → CIE.

Design modes:
  1. Spectral shaping: chromophore absorber + structural resonance
  2. Index engineering: porous low-n + dense high-n for peak control
  3. Absorption + index: simultaneous Δk and Δn from different layers
  4. Red problem attack: low-n inner shell + selective absorber outer shell

Click orthogonality is validated — incompatible stacks are rejected.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from optical.refractive_index import n_complex, n_real
from optical.core_shell_mie import _effective_n_core_shell
from optical.mie_scattering import mie_efficiencies
from optical.structure_factor import structure_factor_PY
from optical.cie_color import (
    spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab, XYZ_to_sRGB, cie_delta_E,
)
from optical.shell_library import shell_optical_properties, CHROMOPHORES, INDEX_SHELLS


# ═══════════════════════════════════════════════════════════════════════════
# CLICK ORTHOGONALITY
# ═══════════════════════════════════════════════════════════════════════════

# Functional group pairs for each click reaction
CLICK_CHEMISTRY = {
    "SPAAC":             {"A": "azide",     "B": "DBCO",      "catalyst": "none"},
    "CuAAC":             {"A": "azide",     "B": "alkyne",    "catalyst": "Cu(I)"},
    "thiol_maleimide":   {"A": "thiol",     "B": "maleimide", "catalyst": "none"},
    "IEDDA":             {"A": "tetrazine", "B": "TCO",       "catalyst": "none"},
}

# Groups that cross-react (non-orthogonal)
_CROSS_REACTIVE = {
    ("azide", "azide"),       # SPAAC and CuAAC both consume azide
    ("thiol", "maleimide"),   # thiol reacts with both maleimide AND azide at high conc
}


def check_orthogonality(click_sequence):
    """Validate that a sequence of click reactions is orthogonal.

    Each interface uses a different click pair. The functional groups
    from one reaction must not interfere with adjacent reactions.

    Args:
        click_sequence: List of click chemistry names for each interface.
                        Length = number of shells.

    Returns:
        (bool, str): (is_orthogonal, reason_if_not)
    """
    if len(click_sequence) <= 1:
        return True, ""

    # Check for duplicate click types
    used = set()
    for i, click in enumerate(click_sequence):
        if click in used:
            return False, (f"Interface {i+1} reuses '{click}' — "
                          f"same click pair cannot be used at two interfaces")
        used.add(click)

    # Check that no two adjacent reactions share a reactive group
    for i in range(len(click_sequence) - 1):
        c1 = CLICK_CHEMISTRY.get(click_sequence[i], {})
        c2 = CLICK_CHEMISTRY.get(click_sequence[i + 1], {})
        groups_1 = {c1.get("A", ""), c1.get("B", "")}
        groups_2 = {c2.get("A", ""), c2.get("B", "")}
        shared = groups_1 & groups_2 - {""}
        if shared:
            return False, (f"Interfaces {i+1} and {i+2} share group(s) {shared} — "
                          f"'{click_sequence[i]}' and '{click_sequence[i+1]}' "
                          f"are not orthogonal")

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════
# SHELL LAYER SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShellLayer:
    """Specification for one shell layer in a multi-shell particle."""
    shell_type: str           # Key in shell_library (chromophore or index shell)
    thickness_nm: float       # Physical thickness of this layer
    click: str = "SPAAC"     # Click chemistry used to attach this layer
    coverage_nm2: float = 3.0 # Sites per nm² (for chromophore loading)
    notes: str = ""

    # Filled by compositor
    n_array: Optional[np.ndarray] = field(default=None, repr=False)
    k_array: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class MultiShellParticle:
    """Complete multi-shell particle specification."""
    core_material: str
    core_diameter_nm: float
    shells: list  # List of ShellLayer, inner to outer

    # Computed
    total_diameter_nm: float = 0.0
    n_effective: complex = 0 + 0j  # at reference wavelength
    orthogonal: bool = True
    orthogonality_note: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# RECURSIVE EFFECTIVE MEDIUM
# ═══════════════════════════════════════════════════════════════════════════

def _recursive_effective_n(core_n, core_radius, shell_layers, wavelength_nm):
    """Recursively compute effective n for a multi-layer sphere.

    Starts from the core and wraps each shell layer using
    Bohren-Huffman quasistatic polarizability.

    Args:
        core_n: Complex refractive index of core at this wavelength
        core_radius: Core radius in nm
        shell_layers: List of (n_shell_complex, thickness_nm) tuples, inner→outer
        wavelength_nm: Current wavelength

    Returns:
        complex: Effective refractive index of the composite particle
        float: Total outer radius
    """
    n_inner = core_n
    r_inner = core_radius

    for n_shell, t_shell in shell_layers:
        r_outer = r_inner + t_shell
        if t_shell < 0.01:
            # Negligible layer — skip
            r_inner = r_outer
            continue

        n_inner = _effective_n_core_shell(n_inner, n_shell, r_inner, r_outer)
        r_inner = r_outer

    return n_inner, r_inner


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-SHELL PHOTONIC GLASS
# ═══════════════════════════════════════════════════════════════════════════

def multi_shell_reflectance(
    core_material: str,
    core_diameter_nm: float,
    shells: list,
    packing_fraction: float = 0.55,
    n_medium: float = 1.0,
    wavelengths_nm: np.ndarray = None,
    absorber_fraction: float = 0.0,
    absorber_material: str = "carbon",
):
    """Compute photonic glass reflectance for a multi-shell particle.

    Args:
        core_material: Core material name
        core_diameter_nm: Core diameter (nm)
        shells: List of ShellLayer objects (inner → outer)
        packing_fraction: Volume fraction φ
        n_medium: Interstitial medium n
        wavelengths_nm: Wavelength array
        absorber_fraction: External absorber volume fraction
        absorber_material: External absorber material

    Returns:
        dict with R, peak_nm, CIE_xy, Lab, sRGB, particle spec
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 201)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    # ── Validate click orthogonality ─────────────────────────────────────
    click_seq = [s.click for s in shells]
    orthogonal, note = check_orthogonality(click_seq)

    # ── Compute shell optical properties ─────────────────────────────────
    for shell in shells:
        if shell.n_array is None:
            props = shell_optical_properties(
                shell.shell_type, wavelengths_nm,
                surface_coverage_nm2=shell.coverage_nm2,
                shell_thickness_nm=shell.thickness_nm,
            )
            shell.n_array = props["n_shell"]
            shell.k_array = props["k_shell"]

    # ── Total particle dimensions ────────────────────────────────────────
    total_shell_thickness = sum(s.thickness_nm for s in shells)
    d_total = core_diameter_nm + 2 * total_shell_thickness
    r_core = core_diameter_nm / 2

    # ── Compute R(λ) ─────────────────────────────────────────────────────
    R = np.zeros(len(wavelengths_nm))

    for i, lam in enumerate(wavelengths_nm):
        # Core index
        n_core = n_complex(core_material, lam)

        # Build shell layer stack for this wavelength
        layer_stack = []
        for shell in shells:
            n_s = complex(shell.n_array[i], shell.k_array[i])
            layer_stack.append((n_s, shell.thickness_nm))

        # Recursive effective medium → single effective sphere
        n_eff_particle, r_outer = _recursive_effective_n(
            n_core, r_core, layer_stack, lam
        )

        # Effective film index for S(q)
        n_core_real = n_real(core_material, lam)
        n_eff_film = math.sqrt(
            packing_fraction * n_core_real**2 +
            (1 - packing_fraction) * n_medium**2
        )

        # Backscattering wavevector
        q_back = 4 * math.pi * n_eff_film / lam

        # Structure factor using total particle diameter
        Sq = structure_factor_PY(q_back, d_total, packing_fraction)

        # Mie on the effective sphere
        eff = mie_efficiencies(d_total, n_eff_particle, n_medium, lam)
        Q_back = eff["Q_back"]
        C_back = (math.pi / 4) * d_total**2 * Q_back

        # Absorber attenuation
        atten = 1.0
        if absorber_fraction > 0:
            n_abs = n_complex(absorber_material, lam)
            k_abs = absorber_fraction * abs(n_abs.imag)
            L = 10 * d_total
            atten = math.exp(-4 * math.pi * k_abs * L / lam)

        R[i] = C_back * Sq * atten

    # Normalize
    Rmax = R.max()
    if Rmax > 0:
        R = R / Rmax

    # ── CIE colorimetry ──────────────────────────────────────────────────
    X, Y, Z = spectrum_to_XYZ(R, wavelengths_nm)
    x, y, Y_lum = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)
    srgb = XYZ_to_sRGB(X, Y, Z)
    peak_nm = float(wavelengths_nm[np.argmax(R)])

    particle = MultiShellParticle(
        core_material=core_material,
        core_diameter_nm=core_diameter_nm,
        shells=shells,
        total_diameter_nm=d_total,
        orthogonal=orthogonal,
        orthogonality_note=note,
    )

    return {
        "R": R,
        "wavelengths_nm": wavelengths_nm,
        "peak_nm": peak_nm,
        "CIE_xy": (x, y),
        "Lab": Lab,
        "sRGB": srgb,
        "particle": particle,
        "orthogonal": orthogonal,
        "orthogonality_note": note,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN PRESETS
# ═══════════════════════════════════════════════════════════════════════════

def red_problem_attack(core_diameter_nm=300, porosity=0.30):
    """Low-n porous shell + red-selective absorber for red structural color.

    Strategy: porous silica shell reduces effective n → redshifts S(q) peak
    without Mie blue contamination. CuPc absorber on outer shell selectively
    removes residual blue backscatter.

    Args:
        core_diameter_nm: Core diameter (larger → redder structural peak)
        porosity: Porous shell porosity (higher → lower n → more redshift)

    Returns:
        dict: Multi-shell reflectance result
    """
    porosity_key = f"porous_silica_{int(porosity*100)}"
    if porosity_key not in INDEX_SHELLS:
        porosity_key = "porous_silica_30"

    shells = [
        ShellLayer(porosity_key, 8.0, click="SPAAC",
                   notes="Low-n inner shell — reduces Mie blue bias"),
        ShellLayer("CuPc", 2.0, click="thiol_maleimide", coverage_nm2=1.0,
                   notes="Red absorber — suppresses residual blue backscatter"),
    ]
    return multi_shell_reflectance("SiO2", core_diameter_nm, shells)


def high_saturation_green(core_diameter_nm=225):
    """High-saturation green via complementary red+blue absorption.

    Strategy: Green structural peak (D≈225nm SiO₂). Inner shell absorbs
    red tail (CuPc, λ_max=678nm). Outer shell absorbs blue tail (TPP,
    λ_max=419nm). Result: sharper green with suppressed tails.
    """
    shells = [
        ShellLayer("CuPc", 2.0, click="SPAAC", coverage_nm2=1.5,
                   notes="Absorbs red tail of structural peak"),
        ShellLayer("TPP_freebase", 2.0, click="thiol_maleimide",
                   coverage_nm2=1.5,
                   notes="Absorbs blue tail via Soret band"),
    ]
    return multi_shell_reflectance("SiO2", core_diameter_nm, shells)


def warm_gold(core_diameter_nm=240):
    """Warm gold/amber via structural green + blue absorption.

    Strategy: Green-yellow structural peak + strong blue absorber
    on inner shell → warm gold appearance.
    """
    shells = [
        ShellLayer("TPP_freebase", 3.0, click="SPAAC", coverage_nm2=2.0,
                   notes="Soret band absorbs blue → warms the structural green"),
        ShellLayer("polydopamine", 5.0, click="thiol_maleimide",
                   notes="Broadband weak absorber + high-n shell → saturation boost"),
    ]
    return multi_shell_reflectance("SiO2", core_diameter_nm, shells)


def index_sandwich(core_diameter_nm=225):
    """Low-n + high-n sandwich for maximum spectral contrast.

    Strategy: porous inner shell (n=1.29) + TiO₂ outer shell (n=2.3).
    The huge Δn between layers creates strong interference effects.
    """
    shells = [
        ShellLayer("porous_silica_30", 5.0, click="SPAAC",
                   notes="Low-n inner layer (n≈1.29)"),
        ShellLayer("TiO2_solgel", 3.0, click="thiol_maleimide",
                   notes="High-n outer layer (n≈2.3)"),
    ]
    return multi_shell_reflectance("SiO2", core_diameter_nm, shells)


def triple_stack(core_diameter_nm=250):
    """Three-layer stack: index + absorber + protection.

    Demonstrates 3-interface orthogonal chemistry:
    SPAAC → CuAAC → thiol-maleimide.
    """
    shells = [
        ShellLayer("porous_silica_30", 5.0, click="SPAAC",
                   notes="Low-n index tuning layer"),
        ShellLayer("CuPc", 2.0, click="thiol_maleimide", coverage_nm2=2.0,
                   notes="Selective red absorption"),
        ShellLayer("PMMA_brush", 5.0, click="IEDDA",
                   notes="Protective polymer brush (mechanical + chemical)"),
    ]
    return multi_shell_reflectance("SiO2", core_diameter_nm, shells)


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def compare_designs(designs, bare_diameter_nm=225, bare_material="SiO2"):
    """Compare multiple multi-shell designs against bare reference.

    Args:
        designs: Dict of {name: result_dict} from multi_shell_reflectance
        bare_diameter_nm: Reference bare particle diameter
        bare_material: Reference core material

    Returns:
        Comparison table as list of dicts
    """
    from optical.structural_dye import bare_core_reflectance, chromaticity_shift

    bare = bare_core_reflectance(bare_diameter_nm, bare_material)
    table = []

    for name, result in designs.items():
        shift = chromaticity_shift(bare, result)
        n_shells = len(result["particle"].shells)
        d_total = result["particle"].total_diameter_nm
        orth = "✓" if result["orthogonal"] else "✗"

        table.append({
            "name": name,
            "n_shells": n_shells,
            "d_total_nm": d_total,
            "peak_nm": result["peak_nm"],
            "delta_peak_nm": shift["delta_peak_nm"],
            "delta_xy": shift["delta_xy"],
            "delta_E_Lab": shift["delta_E_Lab"],
            "orthogonal": orth,
            "CIE_xy": result["CIE_xy"],
            "Lab": result["Lab"],
        })

    return {"bare": bare, "designs": table}


def print_comparison(comp):
    """Pretty-print a design comparison table."""
    bare = comp["bare"]
    print(f"  Bare reference: peak={bare['peak_nm']:.0f}nm, "
          f"CIE xy=({bare['CIE_xy'][0]:.4f}, {bare['CIE_xy'][1]:.4f})")
    print()
    print(f"  {'Design':<22} {'#':>2} {'D_tot':>6} {'peak':>6} "
          f"{'Δλ':>5} {'Δxy':>7} {'ΔE':>5} {'orth':>4}")
    print("  " + "-" * 62)
    for d in comp["designs"]:
        print(f"  {d['name']:<22} {d['n_shells']:>2} {d['d_total_nm']:>5.0f}nm "
              f"{d['peak_nm']:>5.0f}nm {d['delta_peak_nm']:>+4.0f}nm "
              f"{d['delta_xy']:>6.4f} {d['delta_E_Lab']:>5.1f} {d['orthogonal']:>4}")


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("  MABE Multi-Shell Compositor")
    print("  " + "=" * 30)
    print()

    designs = {
        "red_attack": red_problem_attack(),
        "hi_sat_green": high_saturation_green(),
        "warm_gold": warm_gold(),
        "index_sandwich": index_sandwich(),
        "triple_stack": triple_stack(),
    }

    comp = compare_designs(designs)
    print_comparison(comp)
