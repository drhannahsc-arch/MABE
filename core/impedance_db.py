"""
acoustic/impedance_db.py — Acoustic Material Property Database

Analog to optical/refractive_index.py. Provides density (ρ), longitudinal
sound speed (v_L), transverse sound speed (v_T), and acoustic impedance
(Z = ρ × v) for materials used in phononic design.

All values at 20°C / 293K unless noted.

Sources:
  - CRC Handbook of Chemistry and Physics (Tier 1)
  - ASM International materials database
  - Kinsler & Frey "Fundamentals of Acoustics" (Tier 1)
  - Materials Project elastic tensors (Tier 2 for novel materials)
"""

from dataclasses import dataclass
from typing import Optional, Dict
import math


@dataclass
class AcousticMaterial:
    """Complete acoustic properties for a material."""
    name: str
    density_kg_m3: float              # ρ
    v_longitudinal_m_s: float         # v_L (compressional wave speed)
    v_transverse_m_s: float = 0.0     # v_T (shear wave speed; 0 for fluids)
    category: str = ""                # 'metal', 'ceramic', 'polymer', 'fluid', '2D', etc.
    source: str = ""

    @property
    def Z_longitudinal(self) -> float:
        """Longitudinal acoustic impedance (Pa·s/m = Rayl)."""
        return self.density_kg_m3 * self.v_longitudinal_m_s

    @property
    def Z_MRayl(self) -> float:
        """Impedance in MRayl (10⁶ Rayl) — common engineering unit."""
        return self.Z_longitudinal / 1e6

    @property
    def bulk_modulus_GPa(self) -> float:
        """Bulk modulus from density and sound speeds."""
        K = self.density_kg_m3 * (self.v_longitudinal_m_s**2
                                   - 4/3 * self.v_transverse_m_s**2)
        return K / 1e9

    @property
    def youngs_modulus_GPa(self) -> float:
        """Young's modulus (if v_T available)."""
        if self.v_transverse_m_s == 0:
            return 0.0
        vl = self.v_longitudinal_m_s
        vt = self.v_transverse_m_s
        rho = self.density_kg_m3
        # E = ρ v_T² (3v_L² - 4v_T²) / (v_L² - v_T²)
        E = rho * vt**2 * (3*vl**2 - 4*vt**2) / (vl**2 - vt**2)
        return E / 1e9

    @property
    def poisson_ratio(self) -> float:
        """Poisson's ratio from sound speed ratio."""
        if self.v_transverse_m_s == 0:
            return 0.5  # fluid
        r = self.v_longitudinal_m_s / self.v_transverse_m_s
        return (r**2 - 2) / (2 * (r**2 - 1))


# ═══════════════════════════════════════════════════════════════════════════
# MATERIAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════

ACOUSTIC_DB: Dict[str, AcousticMaterial] = {}

def _add(name, rho, vl, vt=0.0, category="", source="CRC"):
    ACOUSTIC_DB[name] = AcousticMaterial(name, rho, vl, vt, category, source)

# ── Metals ────────────────────────────────────────────────────────────
_add("steel_mild",      7850, 5960, 3235, "metal", "CRC")
_add("steel_stainless", 7900, 5790, 3100, "metal", "CRC")
_add("aluminum",        2700, 6420, 3040, "metal", "CRC")
_add("copper",          8960, 4760, 2325, "metal", "CRC")
_add("tungsten",       19300, 5220, 2890, "metal", "CRC")
_add("lead",           11340, 2160, 700,  "metal", "CRC")
_add("gold",           19300, 3240, 1200, "metal", "CRC")
_add("titanium",        4510, 6070, 3125, "metal", "CRC")
_add("iron",            7870, 5950, 3240, "metal", "CRC")
_add("zinc",            7130, 4210, 2440, "metal", "CRC")
_add("nickel",          8900, 5630, 2960, "metal", "CRC")
_add("platinum",       21450, 3260, 1730, "metal", "CRC")

# ── Ceramics / Oxides ─────────────────────────────────────────────────
_add("silica_fused",    2200, 5968, 3764, "ceramic", "CRC")  # SiO2
_add("alumina",         3950, 10520, 6040,"ceramic", "CRC")  # Al2O3
_add("silicon",         2330, 8433, 5843, "ceramic", "CRC")
_add("germanium",       5323, 5400, 3250, "ceramic", "CRC")
_add("glass_pyrex",     2230, 5640, 3280, "ceramic", "CRC")
_add("glass_soda_lime", 2500, 5900, 3400, "ceramic", "CRC")
_add("zirconia",        6000, 7040, 3640, "ceramic", "ASM")  # ZrO2
_add("barium_titanate", 5700, 5540, 2730, "ceramic", "Kinsler") # BaTiO3
_add("PZT",             7600, 4350, 1750, "ceramic", "Kinsler") # lead zirconate titanate
_add("silicon_nitride", 3200, 11000, 6250,"ceramic", "ASM")  # Si3N4
_add("silicon_carbide", 3200, 12000, 7700,"ceramic", "ASM")  # SiC

# ── Polymers ──────────────────────────────────────────────────────────
_add("PLGA",            1250, 2300, 1100, "polymer", "est.")
_add("PMMA",            1190, 2690, 1340, "polymer", "CRC")
_add("polystyrene",     1050, 2350, 1120, "polymer", "CRC")
_add("polyethylene_HD", 960,  2430, 950,  "polymer", "CRC")
_add("polyethylene_LD", 920,  1950, 540,  "polymer", "CRC")
_add("nylon_6",         1140, 2620, 1070, "polymer", "CRC")
_add("PDMS",            1030, 1030, 0,    "polymer", "Kinsler")  # silicone rubber
_add("silicone_rubber", 1100, 1000, 0,    "polymer", "Liu 2000")
_add("epoxy",           1200, 2600, 1100, "polymer", "CRC")
_add("polyurethane",    1100, 1700, 0,    "polymer", "est.")
_add("PVDF",            1780, 2600, 1100, "polymer", "Kinsler")
_add("natural_rubber",  930,  1500, 0,    "polymer", "CRC")
_add("melamine_foam",   9,    60,   0,    "polymer", "est.")  # acoustic foam

# ── Fluids ────────────────────────────────────────────────────────────
_add("air",              1.21, 343,  0,   "fluid", "CRC")
_add("water",            998,  1480, 0,   "fluid", "CRC")
_add("seawater",        1025, 1533, 0,    "fluid", "CRC")
_add("glycerol",        1260, 1920, 0,    "fluid", "CRC")

# ── Porous / Composite ───────────────────────────────────────────────
_add("concrete",        2300, 3100, 1700, "composite", "CRC")
_add("wood_oak",        700,  3850, 1550, "composite", "CRC")  # along grain
_add("wood_pine",       500,  3500, 1300, "composite", "CRC")
_add("plasterboard",    800,  2700, 1200, "composite", "est.")
_add("fiberglass",      100,  340,  0,    "composite", "est.")  # bulk insulation
_add("rockwool",        60,   340,  0,    "composite", "est.")
_add("cork",            120,  500,  0,    "composite", "CRC")

# ── 2D Materials (novel) ─────────────────────────────────────────────
_add("Ti3C2_MXene",     3700, 8800, 5100, "2D", "Materials Project DFT")
_add("hBN",             2100, 13400, 7800,"2D", "Materials Project")
_add("graphene_film",   2267, 17800, 12200, "2D", "Balandin 2011")
_add("MoS2",            5060, 7000, 4000, "2D", "Materials Project")
_add("WS2",             7500, 4500, 2600, "2D", "Materials Project")

# ── MOFs ──────────────────────────────────────────────────────────────
_add("UiO_66",          1300, 2800, 1400, "MOF", "est. from elastic tensors")
_add("ZIF_8",           960,  2500, 1200, "MOF", "est.")
_add("HKUST_1",         900,  2200, 1000, "MOF", "est.")


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP API
# ═══════════════════════════════════════════════════════════════════════════

def get_material(name: str) -> AcousticMaterial:
    """Look up acoustic properties by material name."""
    if name not in ACOUSTIC_DB:
        raise KeyError(f"Unknown material: {name}. "
                       f"Available: {sorted(ACOUSTIC_DB.keys())}")
    return ACOUSTIC_DB[name]


def impedance_contrast(mat1: str, mat2: str) -> float:
    """Compute impedance contrast ratio Z_high / Z_low."""
    z1 = get_material(mat1).Z_longitudinal
    z2 = get_material(mat2).Z_longitudinal
    return max(z1, z2) / min(z1, z2)


def wavelength_in_material(material: str, frequency_Hz: float) -> float:
    """Acoustic wavelength (m) in a material at given frequency."""
    mat = get_material(material)
    return mat.v_longitudinal_m_s / frequency_Hz


def reflection_coefficient(mat1: str, mat2: str) -> float:
    """Normal-incidence pressure reflection coefficient at interface."""
    z1 = get_material(mat1).Z_longitudinal
    z2 = get_material(mat2).Z_longitudinal
    return (z2 - z1) / (z2 + z1)


def transmission_loss_dB(mat1: str, mat2: str) -> float:
    """Normal-incidence transmission loss (dB) at a single interface."""
    z1 = get_material(mat1).Z_longitudinal
    z2 = get_material(mat2).Z_longitudinal
    # TL = 10 log10 [(Z1 + Z2)² / (4 Z1 Z2)]
    return 10 * math.log10((z1 + z2)**2 / (4 * z1 * z2))


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def quarter_wave_thickness(material: str, frequency_Hz: float) -> float:
    """Quarter-wave layer thickness (m) for impedance matching."""
    lam = wavelength_in_material(material, frequency_Hz)
    return lam / 4


def bragg_frequency(material1: str, material2: str,
                    d1_m: float, d2_m: float) -> float:
    """Center frequency of Bragg bandgap for 1D phononic crystal.

    f_Bragg = 1 / (2 × (d1/v1 + d2/v2))
    """
    v1 = get_material(material1).v_longitudinal_m_s
    v2 = get_material(material2).v_longitudinal_m_s
    period = d1_m / v1 + d2_m / v2
    return 1.0 / (2 * period)


def bragg_gap_width(material1: str, material2: str) -> float:
    """Fractional bandgap width Δf/f for 1D phononic crystal.

    Δf/f ≈ (2/π) × arcsin((Z₂−Z₁)/(Z₂+Z₁))

    Equal to 0 for no contrast, approaches 1 for extreme contrast.
    """
    z1 = get_material(material1).Z_longitudinal
    z2 = get_material(material2).Z_longitudinal
    return (2 / math.pi) * math.asin(abs(z2 - z1) / (z2 + z1))


def rank_material_pairs_for_bandgap(
    target_frequency_Hz: float,
    categories: Optional[list] = None,
    min_gap_fraction: float = 0.1,
) -> list:
    """Rank all material pairs by bandgap width at target frequency.

    Returns list of (mat1, mat2, gap_fraction, d1_m, d2_m, f_bragg) sorted
    by gap width descending.
    """
    if categories is None:
        categories = list(set(m.category for m in ACOUSTIC_DB.values()))

    mats = [m for m in ACOUSTIC_DB.values() if m.category in categories]
    pairs = []

    for i, m1 in enumerate(mats):
        for m2 in mats[i+1:]:
            gap = bragg_gap_width(m1.name, m2.name)
            if gap < min_gap_fraction:
                continue

            # Quarter-wave thicknesses at target frequency
            d1 = m1.v_longitudinal_m_s / (4 * target_frequency_Hz)
            d2 = m2.v_longitudinal_m_s / (4 * target_frequency_Hz)

            # Actual Bragg frequency for these thicknesses
            f_bragg = bragg_frequency(m1.name, m2.name, d1, d2)

            pairs.append((m1.name, m2.name, gap, d1, d2, f_bragg))

    pairs.sort(key=lambda x: -x[2])
    return pairs


if __name__ == "__main__":
    print("ACOUSTIC IMPEDANCE DATABASE")
    print(f"{'Material':>25s} {'ρ(kg/m³)':>10s} {'v_L(m/s)':>10s} "
          f"{'Z(MRayl)':>10s} {'Category':>12s}")
    print("─" * 70)
    for name, mat in sorted(ACOUSTIC_DB.items(),
                             key=lambda x: -x[1].Z_longitudinal):
        print(f"{name:>25s} {mat.density_kg_m3:10.0f} "
              f"{mat.v_longitudinal_m_s:10.0f} {mat.Z_MRayl:10.2f} "
              f"{mat.category:>12s}")

    # Top bandgap pairs for 1 kHz (audible)
    print(f"\nTOP BANDGAP PAIRS at 1 kHz:")
    pairs = rank_material_pairs_for_bandgap(1000)
    print(f"{'Mat1':>20s} {'Mat2':>20s} {'Gap%':>6s} {'d1(mm)':>8s} {'d2(mm)':>8s}")
    for m1, m2, gap, d1, d2, fb in pairs[:15]:
        print(f"{m1:>20s} {m2:>20s} {gap*100:5.1f}% "
              f"{d1*1000:8.2f} {d2*1000:8.2f}")

    # Impedance contrast examples
    print(f"\nIMPEDANCE CONTRAST EXAMPLES:")
    for m1, m2 in [("air", "steel_mild"), ("air", "water"),
                    ("lead", "silicone_rubber"), ("tungsten", "PDMS"),
                    ("steel_mild", "epoxy"), ("alumina", "PDMS")]:
        r = impedance_contrast(m1, m2)
        tl = transmission_loss_dB(m1, m2)
        print(f"  {m1:>15s} / {m2:<15s}: Z-ratio={r:8.1f}x  TL={tl:5.1f} dB")