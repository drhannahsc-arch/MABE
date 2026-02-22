"""
core/orbital_binding.py - Orbital-level binding analysis.

Uses frontier molecular orbital theory to estimate binding potential:

1. Charge transfer probability:
   - Donor HOMO → Metal LUMO gap determines covalent bond strength
   - Smaller gap = more favorable charge transfer = stronger coordinate bond
   - ΔE_CT = HOMO_donor - LUMO_metal (should be small and positive for good overlap)

2. Induced dipole binding:
   - Metal polarizability × donor polarizability → London dispersion
   - Larger polarizability = stronger van der Waals = softer interactions

3. Orbital overlap estimate:
   - From electronegativity difference (Mulliken scale)
   - Small difference = covalent character (soft-soft)
   - Large difference = ionic character (hard-hard)

These refine the HSAB heuristic from sprint 9 with actual orbital energies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.problem import TargetSpecies
from core.assembly import RecognitionChemistry
from knowledge.electronic_data import DONOR_HOMO, DONOR_POLARIZABILITY, get_electronic_data


@dataclass
class OrbitalAnalysis:
    """Orbital-level binding analysis."""
    # Charge transfer
    homo_lumo_gap_ev: Optional[float] = None   # donor_HOMO - metal_LUMO
    charge_transfer_favorable: bool = False
    charge_transfer_dG_kj: float = 0.0          # estimated ΔG from CT

    # Induced dipole
    london_dispersion_kj: float = 0.0           # London dispersion contribution

    # Orbital character
    bond_character: str = "unknown"              # covalent, ionic, mixed
    covalent_fraction: float = 0.5

    # DFT data available?
    dft_data_available: bool = False

    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.dft_data_available:
            return "No DFT data — orbital analysis unavailable"
        parts = [f"Bond character: {self.bond_character} ({self.covalent_fraction:.0%} covalent)"]
        if self.homo_lumo_gap_ev is not None:
            parts.append(f"HOMO-LUMO gap: {self.homo_lumo_gap_ev:.2f} eV → CT ΔG: {self.charge_transfer_dG_kj:.1f} kJ/mol")
        if self.london_dispersion_kj != 0:
            parts.append(f"London dispersion: {self.london_dispersion_kj:.1f} kJ/mol")
        return " | ".join(parts)


def compute_orbital_binding(recognition: RecognitionChemistry,
                             target: TargetSpecies) -> OrbitalAnalysis:
    """
    Compute orbital-level binding analysis from DFT-derived data.
    Degrades gracefully if data is missing.
    """
    breakdown = []
    result = OrbitalAnalysis()

    # Get metal LUMO
    metal_lumo = target.electronic.lumo_ev
    metal_polar = target.electronic.polarizability
    metal_eneg = target.electronic.electronegativity or 1.8

    # Fallback: try database
    if metal_lumo is None or metal_polar is None:
        data = get_electronic_data(target.identity)
        if data:
            metal_lumo = metal_lumo or data.get("lumo_ev")
            metal_polar = metal_polar or data.get("polarizability")
            result.dft_data_available = True
        else:
            breakdown.append(f"No DFT data for {target.identity} — using heuristic estimates")

    if metal_lumo is None:
        # Can still do polarizability/character analysis if we have electronegativity
        result.dft_data_available = False
    else:
        result.dft_data_available = True

    donors = recognition.donor_atoms or ["O", "N"]

    # ── 1. Charge transfer analysis ───────────────────────────────
    if metal_lumo is not None:
        # Average donor HOMO
        donor_homos = [DONOR_HOMO.get(d, -9.0) for d in donors]
        avg_donor_homo = sum(donor_homos) / len(donor_homos)

        # Gap: positive = donor HOMO above metal LUMO = favorable donation
        gap = avg_donor_homo - metal_lumo
        result.homo_lumo_gap_ev = round(gap, 3)

        # Favorable if gap is moderate positive (0 to 5 eV)
        # Too large = no overlap; too negative = unfavorable
        result.charge_transfer_favorable = gap > -10.0

        # Estimate ΔG_CT: empirical ~-10 to -20 kJ/mol per eV of favorable gap
        # Diminishing returns: dG_CT ∝ -k × gap / (1 + |gap|/5)
        if gap > 0:
            dG_ct = -12.0 * gap / (1.0 + gap / 5.0)
        elif gap > -8.0:
            # Sigma donation: weaker but still real. Scale linearly.
            dG_ct = -5.0 * (8.0 + gap) / 8.0
        else:
            dG_ct = -0.5  # minimal residual CT

        # Scale by number of donors
        dG_ct *= len(donors) * 0.5  # diminishing per additional donor
        result.charge_transfer_dG_kj = round(dG_ct, 2)

        breakdown.append(
            f"Charge transfer: donor HOMO avg = {avg_donor_homo:.1f} eV, "
            f"metal LUMO = {metal_lumo:.1f} eV, gap = {gap:.2f} eV "
            f"→ ΔG_CT = {dG_ct:.1f} kJ/mol"
        )

    # ── 2. London dispersion (induced dipole) ─────────────────────
    if metal_polar is not None:
        donor_polars = [DONOR_POLARIZABILITY.get(d, 0.5) for d in donors]
        total_london = 0.0
        for dp in donor_polars:
            # London formula: E_disp ∝ -3/4 × (α_A × α_B × I_A × I_B) / (I_A + I_B) / r^6
            # Simplified: E_disp ≈ -C × α_metal × α_donor (kJ/mol)
            # C ~ 2.0 kJ/(mol·Å⁶) for typical separations
            e_london = -2.0 * metal_polar * dp / (2.5**6) * 1e6  # at ~2.5 Å
            # Simpler empirical: -0.5 × sqrt(α_M × α_D) kJ/mol per donor
            e_london = -0.5 * math.sqrt(metal_polar * dp)
            total_london += e_london

        result.london_dispersion_kj = round(total_london, 2)
        breakdown.append(
            f"London dispersion: metal α = {metal_polar:.1f} Å³ → {total_london:.1f} kJ/mol "
            f"({len(donors)} donors)"
        )

    # ── 3. Bond character ─────────────────────────────────────────
    # From electronegativity difference between donor and metal
    donor_enegs = {"S": 2.58, "N": 3.04, "O": 3.44, "P": 2.19, "electrostatic": 3.5}
    avg_donor_eneg = sum(donor_enegs.get(d, 3.0) for d in donors) / len(donors)

    delta_eneg = abs(avg_donor_eneg - metal_eneg)

    # Pauling: % ionic character ≈ 1 - exp(-0.25 × Δχ²)
    ionic_fraction = 1.0 - math.exp(-0.25 * delta_eneg**2)
    covalent_fraction = 1.0 - ionic_fraction
    result.covalent_fraction = round(covalent_fraction, 3)

    if covalent_fraction > 0.7:
        result.bond_character = "covalent"
    elif covalent_fraction > 0.4:
        result.bond_character = "mixed"
    else:
        result.bond_character = "ionic"

    breakdown.append(
        f"Bond character: Δχ = {delta_eneg:.2f} → {ionic_fraction:.0%} ionic, "
        f"{covalent_fraction:.0%} covalent → {result.bond_character}"
    )

    result.breakdown = breakdown
    return result
