"""
core/click_interface.py -- Click Interface Electrical & Thermal Model

Models the electrical contact resistance and thermal Kapitza resistance
of covalent click-chemistry bonds used to join layers in the MABE
energy-harvesting stack.

Each click bond type does triple duty:
  1. Covalent structural attachment
  2. Kapitza thermal barrier (phonon impedance mismatch)
  3. Electrical contact (conjugated or non-conjugated pathways)

Bond types covered:
  - SPAAC triazole (strain-promoted azide-alkyne)
  - CuAAC triazole (Cu-catalyzed azide-alkyne)
  - Thiol-maleimide (thioether linkage)
  - Diels-Alder adduct (thermally reversible)
  - van der Waals gap (non-covalent reference)

Data tier: Tier 2 (values from literature ranges on molecular junctions
and self-assembled monolayers; see source fields).

References:
  - Cui et al., Science 2017, 355, 1192 (single-molecule conductance)
  - Venkataraman et al., Nano Lett. 2006, 6, 458 (molecular junctions)
  - Wang et al., ACS Nano 2011, 5, 3645 (triazole conductance)
  - Losego et al., Nat. Mater. 2012, 11, 502 (SAM Kapitza resistance)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClickBond:
    """Properties of a click-chemistry interface bond."""
    name: str
    bond_type: str                   # chemistry name
    R_contact_ohm_cm2: float         # electrical contact resistance (ohm*cm2)
    R_kapitza_m2K_W: float           # thermal interface resistance (m2*K/W)
    conjugated: bool                 # pi-conjugated pathway?
    reversible: bool                 # thermally reversible?
    source: str
    notes: str = ""


@dataclass
class ClickInterfaceSpec:
    """Input specification for an interface in the stack."""
    bond_type: str
    area_m2: float = 1.0


@dataclass
class InterfaceResult:
    """Prediction for a single interface."""
    bond_type: str
    electrical_R_ohm: float          # total resistance for given area
    thermal_R_kapitza_m2K_W: float
    area_m2: float


@dataclass
class StackLossResult:
    """Power loss across all interfaces in a stack."""
    total_power_loss_W: float
    total_electrical_R_ohm: float
    per_interface: List[InterfaceResult]


# ---------------------------------------------------------------------------
# Bond database
# ---------------------------------------------------------------------------

_CLICK_BONDS = {
    "SPAAC": ClickBond(
        name="SPAAC triazole",
        bond_type="SPAAC",
        R_contact_ohm_cm2=1e-4,
        R_kapitza_m2K_W=5e-8,
        conjugated=True,
        reversible=False,
        source="Wang et al. ACS Nano 2011, 5, 3645",
        notes="Conjugated triazole ring; moderate conductance",
    ),
    "CuAAC": ClickBond(
        name="CuAAC triazole",
        bond_type="CuAAC",
        R_contact_ohm_cm2=1e-5,
        R_kapitza_m2K_W=5e-8,
        conjugated=True,
        reversible=False,
        source="Wang et al. ACS Nano 2011, 5, 3645",
        notes="Slightly more conjugated than SPAAC; lower contact R",
    ),
    "thiol-maleimide": ClickBond(
        name="Thiol-maleimide thioether",
        bond_type="thiol-maleimide",
        R_contact_ohm_cm2=1e-5,
        R_kapitza_m2K_W=3e-8,
        conjugated=False,
        reversible=False,
        source="Cui et al. Science 2017, 355, 1192",
        notes="Non-conjugated but short bond; low R",
    ),
    "Diels-Alder": ClickBond(
        name="Diels-Alder adduct",
        bond_type="Diels-Alder",
        R_contact_ohm_cm2=1e-3,
        R_kapitza_m2K_W=8e-8,
        conjugated=False,
        reversible=True,
        source="Venkataraman et al. Nano Lett. 2006, 6, 458",
        notes="Saturated adduct; poor electrical; thermally reversible ~120C",
    ),
    "van_der_Waals": ClickBond(
        name="van der Waals gap",
        bond_type="van_der_Waals",
        R_contact_ohm_cm2=1e-2,
        R_kapitza_m2K_W=1e-6,
        conjugated=False,
        reversible=True,
        source="Losego et al. Nat. Mater. 2012, 11, 502",
        notes="No covalent bond; reference for non-bonded interface",
    ),
}


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

def get_click_bond(bond_type: str) -> ClickBond:
    """Return a ClickBond by type name. Raises KeyError if not found."""
    if bond_type not in _CLICK_BONDS:
        raise KeyError(
            f"Unknown bond type '{bond_type}'. "
            f"Available: {sorted(_CLICK_BONDS.keys())}"
        )
    return _CLICK_BONDS[bond_type]


def list_click_bonds() -> List[str]:
    """Return sorted list of available bond type names."""
    return sorted(_CLICK_BONDS.keys())


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def predict_interface_resistance(
    bond_type: str,
    area_m2: float = 1.0,
) -> float:
    """
    Predict electrical resistance (ohm) of a click interface.

    R = R_contact (ohm*cm2) / area (cm2)

    Parameters
    ----------
    bond_type : str
        Key into _CLICK_BONDS.
    area_m2 : float
        Interface area in m2. Must be > 0.

    Returns
    -------
    float
        Electrical resistance in ohm.
    """
    if area_m2 <= 0:
        raise ValueError(f"area_m2 must be > 0, got {area_m2}")

    bond = get_click_bond(bond_type)
    area_cm2 = area_m2 * 1e4  # m2 to cm2
    return bond.R_contact_ohm_cm2 / area_cm2


def predict_stack_loss(
    interfaces: List[str],
    current_A: float,
    area_m2: float = 1.0,
) -> StackLossResult:
    """
    Predict total resistive power loss across all interfaces in a stack.

    P_loss = I^2 * R_total

    Parameters
    ----------
    interfaces : list of str
        Bond type names for each interface, bottom to top.
    current_A : float
        Total current through the stack (A). Must be >= 0.
    area_m2 : float
        Interface area (assumed same for all). Must be > 0.

    Returns
    -------
    StackLossResult
    """
    if current_A < 0:
        raise ValueError(f"current_A must be >= 0, got {current_A}")
    if area_m2 <= 0:
        raise ValueError(f"area_m2 must be > 0, got {area_m2}")

    per_interface = []
    total_R = 0.0

    for bt in interfaces:
        bond = get_click_bond(bt)
        r_elec = predict_interface_resistance(bt, area_m2)
        total_R += r_elec
        per_interface.append(InterfaceResult(
            bond_type=bt,
            electrical_R_ohm=r_elec,
            thermal_R_kapitza_m2K_W=bond.R_kapitza_m2K_W,
            area_m2=area_m2,
        ))

    total_loss = current_A ** 2 * total_R

    return StackLossResult(
        total_power_loss_W=total_loss,
        total_electrical_R_ohm=total_R,
        per_interface=per_interface,
    )


def default_harvesting_stack() -> List[str]:
    """
    Return the default interface bond sequence for the energy harvesting stack.

    From spec:
      [Structural Color] --SPAAC-- [Harvesting] --CuAAC-- [Conductor] --thiol-maleimide-- [Substrate]
    """
    return ["SPAAC", "CuAAC", "thiol-maleimide"]
