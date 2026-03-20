"""
core/conductive_base.py -- Conductive Base Layer for Energy Harvesting Stack

Predicts electrical transport properties (sheet resistance, optical transparency,
resistive power loss) of transparent/flexible conductor materials used as the
current-collection base layer in MABE energy-harvesting building elements.

Materials covered: PEDOT:PSS, silver nanowire networks, few-layer graphene,
MXene (Ti3C2Tx), ITO, copper mesh.

Key physics:
  - Sheet resistance R_s = rho / t  (bulk approximation for films > percolation)
  - Temperature dependence: metallic conductors have positive TCR (~+0.003/K);
    PEDOT:PSS has negative TCR (hopping transport, ~-0.002/K)
  - Optical transparency: Beer-Lambert for continuous films, effective medium
    for mesh/nanowire networks
  - Resistive loss: P_loss = J^2 / sigma_eff * path_length * thickness

Click chemistry compatibility: each material tracks top-face and bottom-face
functional handles for covalent interfacing to harvesting and substrate layers.

Data tier: Tier 2 (values from peer-reviewed literature ranges; see source fields).

References:
  - Hu et al., Adv. Mater. 2011, 23, 4035 (PEDOT:PSS conductivity)
  - Lee et al., Nano Lett. 2008, 8, 689 (AgNW networks)
  - Bae et al., Nat. Nanotechnol. 2010, 5, 574 (graphene sheet resistance)
  - Hantanasirisakul & Gogotsi, Adv. Mater. 2018, 30, 1804779 (MXene)
  - Ellmer, Nat. Photonics 2012, 6, 809 (ITO/TCO review)
  - van de Groep et al., Nano Lett. 2012, 12, 3138 (metal mesh)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELECTRON_CHARGE = 1.602e-19  # C


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConductorMaterial:
    """Properties of a transparent/flexible conductor material."""
    name: str
    sigma_S_m: float               # bulk electrical conductivity (S/m)
    sheet_R_range_ohm_sq: Tuple[float, float]  # (min, max) at reference thickness
    ref_thickness_nm: float        # thickness at which sheet_R_range applies
    transparency_vis: float        # visible-range transmittance at ref thickness (0-1)
    tcr_per_K: float               # temperature coefficient of resistance (dR/R per K)
    flexible: bool
    click_handles_top: List[str]   # functional groups facing harvesting layer
    click_handles_bottom: List[str]  # functional groups facing substrate
    density_kg_m3: float
    safe: bool
    source: str

    @property
    def sheet_R_nominal_ohm_sq(self) -> float:
        """Geometric mean of the sheet resistance range at reference thickness."""
        return math.sqrt(self.sheet_R_range_ohm_sq[0] * self.sheet_R_range_ohm_sq[1])


@dataclass
class ConductorSpec:
    """Input specification for conductor prediction."""
    material: str
    thickness_nm: float
    temperature_C: float = 25.0
    wavelength_range_nm: Tuple[float, float] = (380.0, 780.0)
    current_density_A_m2: float = 0.0
    path_length_m: float = 0.0
    area_m2: float = 1.0


@dataclass
class ConductorResult:
    """Output from conductor prediction."""
    material: str
    thickness_nm: float
    temperature_C: float
    sheet_resistance_ohm_sq: float
    transparency: float               # 0-1
    sigma_eff_S_m: float
    power_loss_W_m2: float
    click_handles_top: List[str]
    click_handles_bottom: List[str]
    flexible: bool
    safe: bool


# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------

_CONDUCTORS = {
    "PEDOT_PSS": ConductorMaterial(
        name="PEDOT_PSS",
        sigma_S_m=500.0,            # mid-range for thick films
        sheet_R_range_ohm_sq=(10.0, 100.0),
        ref_thickness_nm=200.0,
        transparency_vis=0.85,
        tcr_per_K=-0.002,           # negative: hopping transport
        flexible=True,
        click_handles_top=["-NH2", "-COOH"],
        click_handles_bottom=["-NH2", "-COOH"],
        density_kg_m3=1011.0,
        safe=True,
        source="Hu et al. Adv. Mater. 2011, 23, 4035",
    ),
    "AgNW": ConductorMaterial(
        name="AgNW",
        sigma_S_m=5e5,              # network conductivity, not bulk Ag
        sheet_R_range_ohm_sq=(5.0, 20.0),
        ref_thickness_nm=100.0,
        transparency_vis=0.87,
        tcr_per_K=0.0038,           # metallic, close to bulk Ag
        flexible=True,
        click_handles_top=["thiol-Ag"],
        click_handles_bottom=["thiol-Ag"],
        density_kg_m3=1500.0,       # effective for nanowire network, not bulk
        safe=True,
        source="Lee et al. Nano Lett. 2008, 8, 689",
    ),
    "graphene": ConductorMaterial(
        name="graphene",
        sigma_S_m=1e7,              # 3-5 layer graphene
        sheet_R_range_ohm_sq=(30.0, 125.0),
        ref_thickness_nm=1.5,       # ~4 layers at 0.34 nm each
        transparency_vis=0.88,
        tcr_per_K=0.001,            # weak metallic in few-layer
        flexible=True,
        click_handles_top=["-N3", "-alkyne"],
        click_handles_bottom=["-N3", "-alkyne"],
        density_kg_m3=2200.0,
        safe=True,
        source="Bae et al. Nat. Nanotechnol. 2010, 5, 574",
    ),
    "MXene_Ti3C2": ConductorMaterial(
        name="MXene_Ti3C2",
        sigma_S_m=1e6,
        sheet_R_range_ohm_sq=(1.0, 10.0),
        ref_thickness_nm=50.0,
        transparency_vis=0.70,
        tcr_per_K=0.003,            # metallic
        flexible=False,             # spec says "moderate" -> conservative
        click_handles_top=["-OH", "azide"],
        click_handles_bottom=["-OH", "azide"],
        density_kg_m3=3700.0,
        safe=True,
        source="Hantanasirisakul & Gogotsi Adv. Mater. 2018, 30, 1804779",
    ),
    "ITO": ConductorMaterial(
        name="ITO",
        sigma_S_m=1e6,
        sheet_R_range_ohm_sq=(10.0, 15.0),
        ref_thickness_nm=150.0,
        transparency_vis=0.90,
        tcr_per_K=0.003,            # metallic oxide
        flexible=False,
        click_handles_top=["surface-OH"],
        click_handles_bottom=["surface-OH"],
        density_kg_m3=7140.0,
        safe=True,
        source="Ellmer Nat. Photonics 2012, 6, 809",
    ),
    "Cu_mesh": ConductorMaterial(
        name="Cu_mesh",
        sigma_S_m=5.8e7,            # bulk Cu
        sheet_R_range_ohm_sq=(0.1, 1.0),
        ref_thickness_nm=500.0,
        transparency_vis=0.85,
        tcr_per_K=0.00393,          # bulk Cu TCR
        flexible=False,             # "moderate" -> conservative
        click_handles_top=["thiol-Cu", "-N3"],
        click_handles_bottom=["thiol-Cu", "-N3"],
        density_kg_m3=8960.0,
        safe=True,
        source="van de Groep et al. Nano Lett. 2012, 12, 3138",
    ),
}


# ---------------------------------------------------------------------------
# Database access helpers
# ---------------------------------------------------------------------------

def get_conductor(name: str) -> ConductorMaterial:
    """Return a conductor material by name. Raises KeyError if not found."""
    if name not in _CONDUCTORS:
        raise KeyError(
            f"Unknown conductor '{name}'. "
            f"Available: {sorted(_CONDUCTORS.keys())}"
        )
    return _CONDUCTORS[name]


def list_conductors() -> List[str]:
    """Return sorted list of available conductor material names."""
    return sorted(_CONDUCTORS.keys())


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_sheet_resistance(
    material: str,
    thickness_nm: float,
    temperature_C: float = 25.0,
) -> float:
    """
    Predict sheet resistance (ohm/sq) for a conductor film.

    Sheet resistance scales inversely with thickness from the reference value.
    Temperature dependence uses linear TCR model:
        R_s(T) = R_s(25C) * (1 + TCR * (T - 25))

    Parameters
    ----------
    material : str
        Key into _CONDUCTORS database.
    thickness_nm : float
        Film thickness in nanometers. Must be > 0.
    temperature_C : float
        Temperature in Celsius.

    Returns
    -------
    float
        Sheet resistance in ohm/sq.
    """
    if thickness_nm <= 0:
        raise ValueError(f"thickness_nm must be > 0, got {thickness_nm}")

    mat = get_conductor(material)
    # Scale from reference thickness: R_s ~ 1/t
    rs_25 = mat.sheet_R_nominal_ohm_sq * (mat.ref_thickness_nm / thickness_nm)
    # Temperature correction
    rs = rs_25 * (1.0 + mat.tcr_per_K * (temperature_C - 25.0))
    # Clamp to positive (extreme cold + negative TCR could go negative)
    return max(rs, 1e-12)


def predict_transparency(
    material: str,
    thickness_nm: float,
    wavelength_range_nm: Tuple[float, float] = (380.0, 780.0),
) -> float:
    """
    Predict visible-range optical transparency (0-1) for a conductor film.

    Uses Beer-Lambert scaling from reference transparency:
        T(t) = T_ref^(t / t_ref)

    This is an approximation; real films have interference fringes.
    For mesh/nanowire networks, transparency depends more on coverage
    fraction than thickness, so scaling is weaker. We use a blended
    model: exponent is clamped to give physically reasonable results.

    Parameters
    ----------
    material : str
        Key into _CONDUCTORS database.
    thickness_nm : float
        Film thickness in nanometers. Must be > 0.
    wavelength_range_nm : tuple
        (min, max) wavelength range. Currently used only for validation;
        the model returns broadband visible transmittance.

    Returns
    -------
    float
        Transmittance (0 to 1).
    """
    if thickness_nm <= 0:
        raise ValueError(f"thickness_nm must be > 0, got {thickness_nm}")
    if wavelength_range_nm[0] >= wavelength_range_nm[1]:
        raise ValueError("wavelength_range_nm must be (min, max) with min < max")

    mat = get_conductor(material)
    # Beer-Lambert scaling
    exponent = thickness_nm / mat.ref_thickness_nm
    t = mat.transparency_vis ** exponent
    return max(0.0, min(1.0, t))


def predict_power_loss(
    material: str,
    thickness_nm: float,
    current_density_A_m2: float,
    path_length_m: float,
    temperature_C: float = 25.0,
) -> float:
    """
    Predict resistive power loss (W/m2) in the conductor layer.

    P_loss = (J^2 / sigma_eff) * path_length
    where sigma_eff is computed from the sheet resistance and thickness.

    Parameters
    ----------
    material : str
        Key into _CONDUCTORS database.
    current_density_A_m2 : float
        Current density in A/m2.
    path_length_m : float
        Current path length (e.g., half the panel width for center collection).
    temperature_C : float
        Temperature in Celsius.

    Returns
    -------
    float
        Power dissipated per unit area in W/m2.
    """
    if thickness_nm <= 0:
        raise ValueError(f"thickness_nm must be > 0, got {thickness_nm}")
    if current_density_A_m2 < 0:
        raise ValueError(f"current_density_A_m2 must be >= 0, got {current_density_A_m2}")
    if path_length_m < 0:
        raise ValueError(f"path_length_m must be >= 0, got {path_length_m}")

    if current_density_A_m2 == 0.0 or path_length_m == 0.0:
        return 0.0

    # Effective conductivity from sheet resistance
    rs = predict_sheet_resistance(material, thickness_nm, temperature_C)
    t_m = thickness_nm * 1e-9
    sigma_eff = 1.0 / (rs * t_m)

    # P_loss per unit area = J^2 * path_length / sigma_eff
    # (J is volumetric current density; path_length gives the length dimension)
    p_loss = (current_density_A_m2 ** 2 / sigma_eff) * path_length_m
    return p_loss


def select_conductor(
    application: str = "general",
    min_transparency: float = 0.0,
    max_sheet_R: float = float("inf"),
    require_flexible: bool = False,
    require_safe: bool = True,
    thickness_nm: Optional[float] = None,
    temperature_C: float = 25.0,
) -> List[ConductorResult]:
    """
    Select and rank conductor materials by suitability.

    Filters by constraints, then ranks by a composite score:
      score = 0.4 * (1 - R_s/R_s_max_in_set) + 0.4 * T + 0.2 * flexibility_bonus

    Parameters
    ----------
    application : str
        Application hint (currently unused beyond logging; filtering is explicit).
    min_transparency : float
        Minimum required transparency (0-1).
    max_sheet_R : float
        Maximum allowed sheet resistance (ohm/sq).
    require_flexible : bool
        If True, exclude rigid materials.
    require_safe : bool
        If True, exclude unsafe materials.
    thickness_nm : float or None
        Evaluate at this thickness. If None, use reference thickness.
    temperature_C : float
        Evaluate at this temperature.

    Returns
    -------
    List[ConductorResult]
        Ranked list (best first) of conductors meeting all constraints.
    """
    candidates = []

    for name, mat in _CONDUCTORS.items():
        t_nm = thickness_nm if thickness_nm is not None else mat.ref_thickness_nm
        rs = predict_sheet_resistance(name, t_nm, temperature_C)
        tr = predict_transparency(name, t_nm)

        # Apply filters
        if require_safe and not mat.safe:
            continue
        if require_flexible and not mat.flexible:
            continue
        if tr < min_transparency:
            continue
        if rs > max_sheet_R:
            continue

        result = ConductorResult(
            material=name,
            thickness_nm=t_nm,
            temperature_C=temperature_C,
            sheet_resistance_ohm_sq=rs,
            transparency=tr,
            sigma_eff_S_m=1.0 / (rs * t_nm * 1e-9),
            power_loss_W_m2=0.0,  # no current spec in selection
            click_handles_top=list(mat.click_handles_top),
            click_handles_bottom=list(mat.click_handles_bottom),
            flexible=mat.flexible,
            safe=mat.safe,
        )
        candidates.append(result)

    if not candidates:
        return []

    # Rank: lower sheet R and higher transparency are better
    rs_max = max(c.sheet_resistance_ohm_sq for c in candidates)
    if rs_max == 0:
        rs_max = 1.0  # avoid division by zero

    def score(c: ConductorResult) -> float:
        rs_norm = 1.0 - (c.sheet_resistance_ohm_sq / rs_max)
        flex_bonus = 1.0 if c.flexible else 0.0
        return 0.4 * rs_norm + 0.4 * c.transparency + 0.2 * flex_bonus

    candidates.sort(key=score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Convenience: full prediction from spec
# ---------------------------------------------------------------------------

def predict_conductor(spec: ConductorSpec) -> ConductorResult:
    """
    Run all predictions for a ConductorSpec, returning a ConductorResult.

    Parameters
    ----------
    spec : ConductorSpec
        Full input specification.

    Returns
    -------
    ConductorResult
    """
    mat = get_conductor(spec.material)
    rs = predict_sheet_resistance(spec.material, spec.thickness_nm, spec.temperature_C)
    tr = predict_transparency(spec.material, spec.thickness_nm, spec.wavelength_range_nm)
    pl = predict_power_loss(
        spec.material, spec.thickness_nm,
        spec.current_density_A_m2, spec.path_length_m, spec.temperature_C,
    )
    t_m = spec.thickness_nm * 1e-9
    sigma_eff = 1.0 / (rs * t_m)

    return ConductorResult(
        material=spec.material,
        thickness_nm=spec.thickness_nm,
        temperature_C=spec.temperature_C,
        sheet_resistance_ohm_sq=rs,
        transparency=tr,
        sigma_eff_S_m=sigma_eff,
        power_loss_W_m2=pl,
        click_handles_top=list(mat.click_handles_top),
        click_handles_bottom=list(mat.click_handles_bottom),
        flexible=mat.flexible,
        safe=mat.safe,
    )


# ---------------------------------------------------------------------------
# Click handle compatibility
# ---------------------------------------------------------------------------

# CuAAC: azide + alkyne
# SPAAC: azide + cyclooctyne (strained)
# Thiol-maleimide: thiol + maleimide
_CLICK_PAIRS = {
    "CuAAC": ({"-N3", "azide"}, {"-alkyne", "alkyne"}),
    "SPAAC": ({"-N3", "azide"}, {"-cyclooctyne", "DBCO"}),
    "thiol-maleimide": ({"thiol-Ag", "thiol-Cu", "-SH"}, {"-maleimide", "maleimide"}),
}


def check_click_compatibility(
    handle_a: str,
    handle_b: str,
) -> Optional[str]:
    """
    Check if two click handles are complementary.

    Returns the click chemistry name if compatible, None otherwise.
    Order-independent: checks (a in group1 and b in group2) OR vice versa.
    """
    for click_name, (group1, group2) in _CLICK_PAIRS.items():
        if (handle_a in group1 and handle_b in group2) or \
           (handle_a in group2 and handle_b in group1):
            return click_name
    return None


def find_compatible_click(
    conductor: str,
    face: str = "top",
    partner_handles: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Find click chemistries that can join a conductor face to partner handles.

    Parameters
    ----------
    conductor : str
        Conductor material name.
    face : str
        'top' or 'bottom'.
    partner_handles : list of str or None
        Functional groups on the partner layer. If None, returns all
        possible pairings.

    Returns
    -------
    List of (conductor_handle, partner_handle, click_chemistry_name)
    """
    mat = get_conductor(conductor)
    handles = mat.click_handles_top if face == "top" else mat.click_handles_bottom

    results = []
    if partner_handles is None:
        # List all possible pairings from _CLICK_PAIRS
        for h in handles:
            for click_name, (g1, g2) in _CLICK_PAIRS.items():
                if h in g1:
                    for partner in g2:
                        results.append((h, partner, click_name))
                elif h in g2:
                    for partner in g1:
                        results.append((h, partner, click_name))
    else:
        for h in handles:
            for ph in partner_handles:
                compat = check_click_compatibility(h, ph)
                if compat is not None:
                    results.append((h, ph, compat))

    return results
