"""
optical/click_linker.py — Module 10: Click Chemistry Linker Model

Models the full attachment chain from particle surface to interparticle
crosslink, including material-specific anchor chemistry:

  [Particle Core] — [Anchor] — [Spacer] — [Click Group A] ··· [Click Group B] — [Spacer] — [Anchor] — [Particle Core]

Total interparticle surface-to-surface gap:
  L_gap = 2 × (L_anchor + L_spacer + L_click_half)

This modifies:
  1. Mie scattering: core-shell with shell = anchor + spacer + click
     (optical shell thickness per particle)
  2. Structure factor: d_eff = D_core + L_gap
     (effective diameter for PY S(q))
  3. Effective packing: φ_eff = φ_actual × (D_core/d_eff)³
     (fewer particles fit when they have thick shells)

Material-specific anchor chemistry:
  SiO2:        silane (APTES, MPTMS)         — covalent Si-O-Si
  TiO2:        catechol or phosphonate        — bidentate surface chelation
  Fe2O3:       catechol or phosphonate        — same as TiO2 (oxide surface)
  Carbon:      diazonium or pyrene-π-stacking — covalent C-C or π-π
  Au:          thiol SAM                      — Au-S bond
  Polystyrene: surface azide (copolymer)      — direct from polymerization

Spacer options:
  PEG_n (n=2,4,8,12):  hydrophilic, flexible, anti-fouling
  alkyl_Cn (n=6,11):   hydrophobic, rigid SAM-like
  none:                 direct anchor → click (shortest chain)

Click chemistry pairs:
  SPAAC:  azide + DBCO (strain-promoted, no catalyst, biocompatible)
  CuAAC:  azide + terminal alkyne (Cu-catalyzed, short linker)
  thiol-maleimide:  thiol + maleimide (fast, but hydrolysis-sensitive)
  IEDDA:  tetrazine + trans-cyclooctene (fastest, but bulky)

References:
  Hermanson, Bioconjugate Techniques, 3rd ed. (2013)
  Love et al. Chem. Rev. 105:1103 (2005) — thiol SAMs on Au
  Lassenberger et al. Langmuir 32:4259 (2016) — catechol anchors on Fe3O4
  Awsiuk et al. Langmuir 35:16756 (2019) — silane on SiO2
  Blakey et al. Soft Matter 9:2613 (2013) — diazonium on carbon
"""

import sys
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from optical.refractive_index import n_complex, n_real
from optical.core_shell_mie import mie_coated_efficiencies
from optical.mie_scattering import mie_efficiencies
from optical.structure_factor import structure_factor_PY
from optical.cie_color import spectrum_to_XYZ, XYZ_to_xyY, XYZ_to_Lab


# ═══════════════════════════════════════════════════════════════════════════
# ANCHOR CHEMISTRY DATABASE
# ═══════════════════════════════════════════════════════════════════════════
# Each anchor: length (nm), refractive index, compatible core materials

@dataclass
class Anchor:
    name: str
    length_nm: float          # Surface-to-terminal-group distance
    n: float                  # Refractive index of anchor layer
    compatible_cores: list    # Which core materials this works on
    chemistry: str            # Bond type
    notes: str = ""


ANCHORS = {
    # Silane coupling agents — covalent Si-O-Si to oxide surfaces
    "APTES": Anchor("APTES", 0.8, 1.46,
                    ["SiO2", "TiO2_rutile", "TiO2_anatase", "Al2O3", "ZnO"],
                    "silane", "3-aminopropyltriethoxysilane, most common"),
    "MPTMS": Anchor("MPTMS", 0.7, 1.46,
                    ["SiO2", "TiO2_rutile", "TiO2_anatase"],
                    "silane", "3-mercaptopropyltrimethoxysilane, thiol-terminated"),

    # Catechol/dopamine — bidentate chelation to oxide surfaces
    "catechol": Anchor("catechol", 0.6, 1.65,
                       ["TiO2_rutile", "TiO2_anatase", "Fe2O3", "ZnO", "SiO2"],
                       "catechol_chelation",
                       "Dopamine or DOPA anchor, universal oxide adhesion"),
    "phosphonate": Anchor("phosphonate", 0.5, 1.50,
                          ["TiO2_rutile", "TiO2_anatase", "Fe2O3", "ZnO",
                           "Al2O3", "SiO2"],
                          "P-O-M chelation",
                          "Alkylphosphonate, very stable on TiO2/Fe2O3"),

    # Thiol SAM — Au-S bond
    "thiol": Anchor("thiol", 0.3, 1.45,
                    ["Au", "gold"],
                    "Au-S covalent",
                    "Thiol self-assembled monolayer"),

    # Diazonium — covalent C-C to carbon surfaces
    "diazonium": Anchor("diazonium", 0.5, 1.60,
                        ["carbon"],
                        "C-C covalent",
                        "Electrochemical or spontaneous grafting"),

    # Pyrene π-stacking — non-covalent to carbon/graphene
    "pyrene": Anchor("pyrene", 0.35, 1.70,
                     ["carbon"],
                     "pi-pi stacking",
                     "Non-covalent, reversible, preserves sp² network"),

    # Direct surface azide — no anchor needed (built into polymer)
    "direct_azide": Anchor("direct_azide", 0.0, 1.0,
                           ["polystyrene", "PS"],
                           "copolymer incorporation",
                           "Azide-functionalized comonomer during polymerization"),
}

# ═══════════════════════════════════════════════════════════════════════════
# SPACER DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Spacer:
    name: str
    length_nm: float      # End-to-end distance (extended conformation)
    n: float              # Refractive index
    hydrophilic: bool
    notes: str = ""


SPACERS = {
    "none":     Spacer("none", 0.0, 1.0, True, "Direct anchor→click, shortest"),
    "PEG2":     Spacer("PEG2", 0.7, 1.47, True, "~8 atoms, flexible"),
    "PEG4":     Spacer("PEG4", 1.4, 1.47, True, "~16 atoms, standard"),
    "PEG8":     Spacer("PEG8", 2.8, 1.47, True, "~32 atoms, long flexible"),
    "PEG12":    Spacer("PEG12", 4.2, 1.47, True, "~48 atoms, very long"),
    "alkyl_C6": Spacer("alkyl_C6", 0.8, 1.45, False, "Hexyl chain, rigid"),
    "alkyl_C11":Spacer("alkyl_C11", 1.4, 1.45, False, "Undecyl chain, SAM-like"),
}

# ═══════════════════════════════════════════════════════════════════════════
# CLICK CHEMISTRY DATABASE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClickPair:
    name: str
    group_A: str          # Functional group on particle A
    group_B: str          # Functional group on particle B (or surface)
    half_length_nm: float # Length contributed per side of the click junction
    n: float              # Refractive index of click product
    catalyst: str         # Required catalyst ("none", "Cu(I)", etc.)
    rate: str             # "fast", "moderate", "slow"
    notes: str = ""


CLICK_PAIRS = {
    "SPAAC": ClickPair("SPAAC", "azide", "DBCO", 0.9, 1.46,
                        "none", "moderate",
                        "Strain-promoted, no catalyst, biocompatible"),
    "CuAAC": ClickPair("CuAAC", "azide", "alkyne", 0.4, 1.46,
                        "Cu(I)/THPTA", "fast",
                        "Copper-catalyzed, shortest triazole product"),
    "thiol_maleimide": ClickPair("thiol_maleimide", "thiol", "maleimide",
                                  0.3, 1.46, "none", "fast",
                                  "Very fast, but maleimide hydrolysis at pH>7"),
    "IEDDA": ClickPair("IEDDA", "tetrazine", "TCO", 1.0, 1.48,
                        "none", "very fast",
                        "Inverse electron-demand Diels-Alder, bulky groups"),
}


# ═══════════════════════════════════════════════════════════════════════════
# CHAIN LENGTH CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttachmentChain:
    """Full specification of one particle's attachment chain."""
    core_material: str
    anchor: str
    spacer: str
    click: str

    # Computed
    L_anchor_nm: float = 0.0
    L_spacer_nm: float = 0.0
    L_click_half_nm: float = 0.0
    L_total_per_side_nm: float = 0.0  # total shell thickness per particle
    n_shell_effective: float = 1.46    # volume-weighted average n of shell
    compatible: bool = True
    incompatibility_reason: str = ""


def compute_chain(core_material, anchor="APTES", spacer="PEG4",
                  click="SPAAC"):
    """Compute full attachment chain dimensions.

    Args:
        core_material: particle core material
        anchor: anchor chemistry name
        spacer: spacer name
        click: click chemistry name

    Returns:
        AttachmentChain with computed lengths
    """
    chain = AttachmentChain(
        core_material=core_material,
        anchor=anchor,
        spacer=spacer,
        click=click,
    )

    # Look up components
    anc = ANCHORS.get(anchor)
    spc = SPACERS.get(spacer)
    clk = CLICK_PAIRS.get(click)

    if anc is None:
        chain.compatible = False
        chain.incompatibility_reason = f"Unknown anchor: {anchor}"
        return chain
    if spc is None:
        chain.compatible = False
        chain.incompatibility_reason = f"Unknown spacer: {spacer}"
        return chain
    if clk is None:
        chain.compatible = False
        chain.incompatibility_reason = f"Unknown click: {click}"
        return chain

    # Check core compatibility
    if core_material not in anc.compatible_cores:
        chain.compatible = False
        chain.incompatibility_reason = (
            f"{anchor} not compatible with {core_material}. "
            f"Compatible: {anc.compatible_cores}")
        return chain

    chain.L_anchor_nm = anc.length_nm
    chain.L_spacer_nm = spc.length_nm
    chain.L_click_half_nm = clk.half_length_nm
    chain.L_total_per_side_nm = anc.length_nm + spc.length_nm + clk.half_length_nm

    # Volume-weighted effective n for the shell
    # (anchor layer near surface, spacer in middle, click at exterior)
    L_total = chain.L_total_per_side_nm
    if L_total > 0:
        chain.n_shell_effective = (
            anc.length_nm * anc.n +
            spc.length_nm * spc.n +
            clk.half_length_nm * clk.n
        ) / L_total
    else:
        chain.n_shell_effective = 1.0

    return chain


def interparticle_gap(chain_A, chain_B=None):
    """Compute surface-to-surface gap between two functionalized particles.

    If chain_B is None, assumes both particles have the same chain.

    Returns gap in nm.
    """
    if chain_B is None:
        chain_B = chain_A
    return chain_A.L_total_per_side_nm + chain_B.L_total_per_side_nm


def effective_diameter(core_diameter_nm, chain):
    """Effective diameter for structure factor (core + 2 × shell)."""
    return core_diameter_nm + 2 * chain.L_total_per_side_nm


def effective_packing(phi_actual, core_diameter_nm, chain):
    """Effective packing fraction adjusted for shell volume.

    Shell-occupied volume reduces the number of particles that fit,
    so φ_eff = φ_actual × (D_core/D_eff)³
    """
    D_eff = effective_diameter(core_diameter_nm, chain)
    if D_eff <= 0:
        return phi_actual
    return phi_actual * (core_diameter_nm / D_eff)**3


# ═══════════════════════════════════════════════════════════════════════════
# RECOMMEND ATTACHMENT FOR A MATERIAL
# ═══════════════════════════════════════════════════════════════════════════

def recommend_attachment(core_material, click="SPAAC",
                         prefer_short=False):
    """Find compatible anchor+spacer combinations for a core material.

    Returns list of AttachmentChain sorted by total length.
    """
    results = []
    spacer_choices = ["none", "PEG2", "PEG4"] if prefer_short else list(SPACERS.keys())

    for anchor_name, anc in ANCHORS.items():
        if core_material not in anc.compatible_cores:
            continue
        for spacer_name in spacer_choices:
            chain = compute_chain(core_material, anchor_name, spacer_name, click)
            if chain.compatible:
                results.append(chain)

    results.sort(key=lambda c: c.L_total_per_side_nm)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONALIZED PHOTONIC GLASS — FULL FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════════════

def functionalized_photonic_glass(
    core_diameter_nm: float,
    core_material: str,
    anchor: str,
    spacer: str,
    click: str,
    n_medium: float,
    packing_fraction: float,
    wavelengths_nm: np.ndarray,
    absorber_fraction: float = 0.0,
    absorber_material: str = "carbon",
) -> dict:
    """Full photonic glass model including click linker effects.

    Combines:
      - Core-shell Mie (M4) with shell = attachment chain
      - Modified structure factor (M5) with d_eff
      - Absorber attenuation

    Args:
        core_diameter_nm:  Core particle diameter
        core_material:     Core material
        anchor:            Anchor chemistry
        spacer:            Spacer type
        click:             Click chemistry
        n_medium:          Medium refractive index
        packing_fraction:  Actual packing fraction
        wavelengths_nm:    Wavelength array
        absorber_fraction: Volume fraction of absorber
        absorber_material: Absorber material

    Returns:
        dict with:
          R: reflectance spectrum (normalized)
          chain: AttachmentChain
          D_eff: effective diameter
          phi_eff: effective packing fraction
          peak_nm: peak wavelength
          peak_shift_nm: shift vs bare particles
          cie_xy, Lab: predicted color
    """
    chain = compute_chain(core_material, anchor, spacer, click)
    if not chain.compatible:
        raise ValueError(chain.incompatibility_reason)

    D_eff = effective_diameter(core_diameter_nm, chain)
    phi_eff = effective_packing(packing_fraction, core_diameter_nm, chain)
    shell_t = chain.L_total_per_side_nm
    n_shell = chain.n_shell_effective

    R = np.zeros(len(wavelengths_nm))
    r_core = core_diameter_nm / 2
    r_total = r_core + shell_t

    for i, lam in enumerate(wavelengths_nm):
        n_core_r = n_real(core_material, lam)

        # Effective index for composite (uses effective diameter)
        n_eff = math.sqrt(phi_eff * n_core_r**2
                          + (1 - phi_eff) * n_medium**2)

        # Backscattering wavevector
        q_back = 4 * math.pi * n_eff / lam

        # Structure factor with effective diameter
        Sq = structure_factor_PY(q_back, D_eff, phi_eff)

        # Core-shell Mie backscattering
        n_core_c = n_complex(core_material, lam)
        n_sh = complex(n_shell, 0.0)

        if shell_t > 0.1:  # meaningful shell
            try:
                eff = mie_coated_efficiencies(r_core, r_total,
                                               n_core_c, n_sh,
                                               n_medium, lam)
                Qback = eff["Q_back"]
            except Exception:
                eff_bare = mie_efficiencies(core_diameter_nm, n_core_c,
                                            n_medium, lam)
                Qback = eff_bare["Q_back"]
        else:
            eff_bare = mie_efficiencies(core_diameter_nm, n_core_c,
                                        n_medium, lam)
            Qback = eff_bare["Q_back"]

        C_back = (math.pi / 4) * D_eff**2 * Qback

        # Absorber
        k_abs = 0.0
        if absorber_fraction > 0:
            n_abs_c = n_complex(absorber_material, lam)
            k_abs = absorber_fraction * n_abs_c.imag
        L_film = 10 * D_eff
        attenuation = math.exp(-4 * math.pi * k_abs * L_film / lam)

        R[i] = C_back * Sq * attenuation

    Rmax = R.max()
    R_norm = R / Rmax if Rmax > 0 else R

    # Peak wavelength
    peak_nm = float(wavelengths_nm[np.argmax(R_norm)])

    # Bare particle peak for comparison
    from optical.photonic_glass import photonic_glass_peak_wavelength
    bare_peak = photonic_glass_peak_wavelength(core_diameter_nm, core_material,
                                                n_medium, packing_fraction)

    # CIE color
    X, Y, Z = spectrum_to_XYZ(R_norm, wavelengths_nm)
    x, y, _ = XYZ_to_xyY(X, Y, Z)
    Lab = XYZ_to_Lab(X, Y, Z)

    return {
        "R": R_norm,
        "chain": chain,
        "D_eff_nm": round(D_eff, 1),
        "phi_eff": round(phi_eff, 4),
        "shell_thickness_nm": round(shell_t, 2),
        "n_shell": round(n_shell, 3),
        "peak_nm": peak_nm,
        "bare_peak_nm": bare_peak,
        "peak_shift_nm": round(peak_nm - bare_peak, 1),
        "cie_xy": (round(x, 4), round(y, 4)),
        "Lab": tuple(round(v, 1) for v in Lab),
    }


# ═══════════════════════════════════════════════════════════════════════════
# COMPARE LINKER CHEMISTRIES FOR A GIVEN CORE
# ═══════════════════════════════════════════════════════════════════════════

def compare_linkers(core_diameter_nm, core_material, n_medium=1.0,
                    packing_fraction=0.55, click="SPAAC",
                    wavelengths_nm=None):
    """Compare all compatible attachment chains for a given core material.

    Returns list of dicts sorted by peak shift (smallest first).
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(380, 780, 81)

    chains = recommend_attachment(core_material, click=click)
    results = []

    for chain in chains:
        try:
            r = functionalized_photonic_glass(
                core_diameter_nm, core_material,
                chain.anchor, chain.spacer, chain.click,
                n_medium, packing_fraction, wavelengths_nm)
            r["anchor"] = chain.anchor
            r["spacer"] = chain.spacer
            results.append(r)
        except Exception:
            continue

    results.sort(key=lambda r: abs(r["peak_shift_nm"]))
    return results


def print_linker_comparison(results):
    """Pretty-print linker comparison."""
    print()
    print(f"  MABE Click Linker Comparison")
    print(f"  {'Anchor':15s}  {'Spacer':10s}  {'Shell':6s}  "
          f"{'D_eff':6s}  {'Peak':5s}  {'Shift':6s}  CIE xy")
    print(f"  {'─'*75}")
    for r in results:
        print(f"  {r['anchor']:15s}  {r['spacer']:10s}  "
              f"{r['shell_thickness_nm']:5.1f}nm  "
              f"{r['D_eff_nm']:5.0f}nm  "
              f"{r['peak_nm']:5.0f}  {r['peak_shift_nm']:+5.1f}  "
              f"({r['cie_xy'][0]:.3f},{r['cie_xy'][1]:.3f})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MABE Click Chemistry Linker Model — Self-Test")
    print("=" * 70)

    lam = np.linspace(380, 780, 81)

    # Test 1: SiO2 with SPAAC
    print("\n--- SiO2 (200nm) linker comparison ---")
    r = compare_linkers(200, "SiO2", click="SPAAC", wavelengths_nm=lam)
    print_linker_comparison(r)

    # Test 2: TiO2 with SPAAC
    print("--- TiO2 (200nm) linker comparison ---")
    r2 = compare_linkers(200, "TiO2_rutile", click="SPAAC", wavelengths_nm=lam)
    print_linker_comparison(r2)

    # Test 3: Fe2O3 (iron oxide) with SPAAC
    print("--- Fe2O3 (200nm) linker comparison ---")
    r3 = compare_linkers(200, "Fe2O3", click="SPAAC", wavelengths_nm=lam)
    print_linker_comparison(r3)

    # Test 4: Carbon with SPAAC
    print("--- Carbon (200nm) linker comparison ---")
    r4 = compare_linkers(200, "carbon", click="SPAAC", wavelengths_nm=lam)
    print_linker_comparison(r4)

    # Test 5: Recommend attachment for each material
    print("--- Recommended attachments ---")
    for mat in ["SiO2", "TiO2_rutile", "Fe2O3", "carbon", "Au", "polystyrene"]:
        chains = recommend_attachment(mat, prefer_short=True)
        if chains:
            best = chains[0]
            print(f"  {mat:15s}: {best.anchor} + {best.spacer} + {best.click} "
                  f"= {best.L_total_per_side_nm:.1f} nm/side")
        else:
            print(f"  {mat:15s}: no compatible attachment found")
