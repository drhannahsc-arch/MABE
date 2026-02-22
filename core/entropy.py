"""
core/entropy.py - Entropy decomposition and Gibbs-Helmholtz temperature prediction.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from knowledge.entropy_data import (
    ENTHALPY_FRACTIONS, DONOR_ENTROPY, WATER_RELEASE_ENTROPY,
    DONOR_HEAT_CAPACITY,
)
from core.thermodynamics import BindingThermodynamics, R_GAS


@dataclass
class EntropyContribution:
    """ΔH/ΔS decomposition of a single ΔG term."""
    name: str
    dG: float = 0.0
    dH: float = 0.0
    minus_TdS: float = 0.0
    dS: float = 0.0       # J/(mol·K)
    note: str = ""


@dataclass
class EntropyDecomposition:
    """Full ΔH / -TΔS decomposition of binding thermodynamics."""
    temperature_ref_k: float = 298.15
    dH_total: float = 0.0
    dS_total: float = 0.0           # J/(mol·K)
    minus_TdS_total: float = 0.0    # kJ/mol
    dG_ref: float = 0.0
    contributions: list[EntropyContribution] = field(default_factory=list)
    dCp_total: float = 0.0          # J/(mol·K)

    dG_at_4C: Optional[float] = None
    dG_at_37C: Optional[float] = None
    dG_at_60C: Optional[float] = None
    dG_at_target: Optional[float] = None
    target_temp_c: Optional[float] = None

    enthalpy_entropy_compensation: str = ""
    temperature_sensitivity: str = ""
    breakdown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"ΔG = {self.dG_ref:+.1f} kJ/mol = ΔH ({self.dH_total:+.1f}) "
            f"- TΔS ({self.minus_TdS_total:+.1f})",
            f"  ΔS = {self.dS_total:+.1f} J/(mol·K) | ΔCp = {self.dCp_total:+.1f} J/(mol·K)",
        ]
        if self.dG_at_4C is not None:
            parts.append(
                f"  ΔG(4°C) = {self.dG_at_4C:+.1f} | ΔG(37°C) = {self.dG_at_37C:+.1f} | "
                f"ΔG(60°C) = {self.dG_at_60C:+.1f} kJ/mol"
            )
        if self.enthalpy_entropy_compensation:
            parts.append(f"  {self.enthalpy_entropy_compensation}")
        if self.temperature_sensitivity:
            parts.append(f"  {self.temperature_sensitivity}")
        return "\n".join(parts)


def gibbs_helmholtz(dH: float, dS_jmolk: float, T_target_K: float,
                      dCp: float = 0.0, T_ref_K: float = 298.15) -> float:
    """
    ΔG(T) = ΔH - T × ΔS + ΔCp × [(T - T_ref) - T × ln(T/T_ref)]
    dH in kJ/mol, dS in J/(mol·K), dCp in J/(mol·K).
    """
    dS_kj = dS_jmolk / 1000.0
    dCp_kj = dCp / 1000.0
    dG = dH - T_target_K * dS_kj
    if abs(dCp_kj) > 1e-6 and T_target_K > 0 and T_ref_K > 0:
        dG += dCp_kj * ((T_target_K - T_ref_K) - T_target_K * math.log(T_target_K / T_ref_K))
    return dG


def decompose_thermodynamics(thermo: BindingThermodynamics,
                                donors: list[str] = None,
                                n_donors: int = 0,
                                is_macrocyclic: bool = False,
                                actual_temp_c: float = None,
                                ) -> EntropyDecomposition:
    """Decompose ΔG_net into ΔH and -TΔS; predict ΔG at other temperatures."""
    T_ref = thermo.temperature_k
    result = EntropyDecomposition(temperature_ref_k=T_ref)
    breakdown = []
    contributions = []
    donors = donors or []
    if n_donors == 0:
        n_donors = len(donors)

    terms = {
        "dG_bind": thermo.dG_bind,
        "dG_desolv": thermo.dG_desolv,
        "dG_chelate": thermo.dG_chelate,
        "dG_preorg": thermo.dG_preorg,
        "dG_electrostatic": thermo.dG_electrostatic,
    }

    total_dH = 0.0
    total_minus_TdS = 0.0

    for name, dG in terms.items():
        if abs(dG) < 0.01:
            continue
        f_H = ENTHALPY_FRACTIONS.get(name, 0.5)
        dH = dG * f_H
        minus_TdS = dG * (1.0 - f_H)
        dS = -minus_TdS / (T_ref / 1000.0) if T_ref > 0 else 0.0
        total_dH += dH
        total_minus_TdS += minus_TdS
        contributions.append(EntropyContribution(
            name=name, dG=round(dG, 2), dH=round(dH, 2),
            minus_TdS=round(minus_TdS, 2), dS=round(dS, 1),
        ))
        breakdown.append(
            f"{name}: ΔG={dG:+.1f} → ΔH={dH:+.1f} + (-TΔS)={minus_TdS:+.1f}"
        )

    # Donor-specific entropy
    donor_dS = sum(DONOR_ENTROPY.get(d, -10.0) for d in donors)
    water_dS = n_donors * WATER_RELEASE_ENTROPY
    net_coord_dS = donor_dS + water_dS

    if abs(net_coord_dS) > 1.0:
        minus_TdS_coord = -(net_coord_dS / 1000.0) * T_ref
        total_minus_TdS += minus_TdS_coord
        contributions.append(EntropyContribution(
            name="coordination_entropy", dG=round(minus_TdS_coord, 2),
            dH=0.0, minus_TdS=round(minus_TdS_coord, 2),
            dS=round(net_coord_dS, 1),
            note=f"Donor entropy + water release for {n_donors} donors",
        ))
        breakdown.append(
            f"Coordination entropy: ΔS={net_coord_dS:+.0f} J/(mol·K) → -TΔS={minus_TdS_coord:+.1f}"
        )

    # Heat capacity
    dCp = sum(DONOR_HEAT_CAPACITY.get(d, 0.0) for d in donors)
    result.dCp_total = round(dCp, 1)

    # Totals
    result.dH_total = round(total_dH, 2)
    result.minus_TdS_total = round(total_minus_TdS, 2)
    total_dS = -total_minus_TdS / (T_ref / 1000.0) if T_ref > 0 else 0.0
    result.dS_total = round(total_dS, 1)
    result.dG_ref = round(total_dH + total_minus_TdS, 2)
    result.contributions = contributions

    # Gibbs-Helmholtz predictions
    for temp_c, attr in [(4.0, "dG_at_4C"), (37.0, "dG_at_37C"), (60.0, "dG_at_60C")]:
        T_K = temp_c + 273.15
        dG_T = gibbs_helmholtz(total_dH, total_dS, T_K, dCp, T_ref)
        setattr(result, attr, round(dG_T, 2))

    if actual_temp_c is not None:
        T_actual = actual_temp_c + 273.15
        dG_actual = gibbs_helmholtz(total_dH, total_dS, T_actual, dCp, T_ref)
        result.dG_at_target = round(dG_actual, 2)
        result.target_temp_c = actual_temp_c
        breakdown.append(f"ΔG at {actual_temp_c:.0f}°C = {dG_actual:+.1f} kJ/mol")

    # Compensation analysis
    if abs(total_dH) > 1.0 and abs(total_minus_TdS) > 1.0:
        if total_dH < 0 and total_minus_TdS > 0:
            result.enthalpy_entropy_compensation = (
                "Enthalpy-entropy COMPENSATION: favorable ΔH partially cancelled by unfavorable -TΔS"
            )
        elif total_dH > 0 and total_minus_TdS < 0:
            result.enthalpy_entropy_compensation = (
                "Entropy-driven binding: unfavorable ΔH overcome by favorable entropy"
            )
        elif total_dH < 0 and total_minus_TdS < 0:
            result.enthalpy_entropy_compensation = (
                "Enthalpy AND entropy favorable — strong binding"
            )
        else:
            result.enthalpy_entropy_compensation = (
                "Both ΔH and -TΔS unfavorable — weak/no binding expected"
            )

    if abs(total_dS) > 100:
        result.temperature_sensitivity = "HIGH temperature sensitivity (large ΔS)"
    elif abs(total_dS) > 30:
        result.temperature_sensitivity = "Moderate temperature sensitivity"
    else:
        result.temperature_sensitivity = "Low temperature sensitivity (enthalpy-dominated)"

    result.breakdown = breakdown
    return result
