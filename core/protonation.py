"""
core/protonation.py - pH-dependent donor availability analysis.

This module computes:
1. Per-donor fraction available at working pH
2. Effective denticity (how many donors are actually functional)
3. pH-corrected ΔG_bind (protonated donors contribute nothing)
4. Metal-assisted deprotonation (shifts pKa when metal is present)

The key insight: at pH 3.5 mine water, a hexadentate EDTA-type chelator
with 4 carboxylate O and 2 amine N becomes effectively mono/bidentate
because the amines are 100% protonated and carboxylates ~90% protonated.

Meanwhile, a dithiocarbamate (pKa 3.5) is 50% active at pH 3.5,
and with metal-assisted deprotonation it may be 80%+ active.

This is why EDTA fails at pH < 4 despite having log K = 18 for lead.
The log K was measured at pH 7-10 where all donors are available.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from core.assembly import RecognitionChemistry
from core.problem import Problem
from knowledge.pka_data import (
    fraction_deprotonated, effective_pka, classify_donor_group,
    get_donor_pka, DONOR_PKA, METAL_ASSISTED_PKA_SHIFT,
)


@dataclass
class DonorAvailability:
    """Protonation state analysis for a single donor atom."""
    atom: str                          # O, N, S, P
    functional_group: str              # e.g. "carboxylate", "thiolate"
    intrinsic_pka: float              # pKa without metal present
    effective_pka: float              # pKa with metal-assisted shift
    fraction_available: float          # 0-1, at working pH
    fraction_without_metal: float      # for comparison
    is_protonated: bool               # True if >50% protonated
    is_dead: bool                     # True if <5% available


@dataclass
class ProtonationProfile:
    """Complete protonation analysis for a recognition element."""
    donors: list[DonorAvailability] = field(default_factory=list)
    
    # Summary
    nominal_denticity: int = 0        # how many donors the binder claims
    effective_denticity: float = 0.0  # sum of fractions available
    denticity_loss: float = 0.0       # nominal - effective
    
    # Correction factors for thermodynamics
    dG_protonation_penalty: float = 0.0  # additional ΔG cost (positive = unfavorable)
    chelate_effect_correction: float = 0.0  # loss of chelate effect from reduced denticity
    
    # Overall
    ph: float = 7.0
    fraction_total_available: float = 1.0  # average across all donors
    
    breakdown: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        parts = [
            f"pH {self.ph:.1f}: effective denticity = {self.effective_denticity:.1f} / {self.nominal_denticity}",
        ]
        for d in self.donors:
            status = "DEAD" if d.is_dead else ("weak" if d.is_protonated else "active")
            shift = d.intrinsic_pka - d.effective_pka
            parts.append(
                f"  {d.atom} ({d.functional_group}): pKa {d.intrinsic_pka:.1f} → "
                f"{d.effective_pka:.1f} (metal shift -{shift:.1f}) → "
                f"{d.fraction_available:.0%} available [{status}]"
            )
        if self.denticity_loss > 0.5:
            parts.append(f"  ⚠ Lost {self.denticity_loss:.1f} effective donors to protonation")
        parts.append(f"  ΔG_protonation = +{self.dG_protonation_penalty:.1f} kJ/mol")
        return "\n".join(parts)


def compute_protonation(recognition: RecognitionChemistry,
                         problem: Problem) -> ProtonationProfile:
    """
    Compute pH-dependent donor availability for a recognition element.
    """
    target = problem.target
    ph = problem.matrix.ph or 7.0
    metal_hsab = target.electronic.hardness_softness or "borderline"
    metal_charge = abs(target.charge) if target.charge else 2.0
    
    # Map donor HSAB character
    donor_hsab_map = {"soft": "soft", "hard": "hard", "borderline": "borderline"}
    donor_hsab = donor_hsab_map.get(recognition.donor_type or "borderline", "borderline")
    
    structure_context = recognition.structure or ""
    donors = recognition.donor_atoms or []
    
    breakdown = []
    donor_profiles = []
    effective_n = 0.0
    
    for atom in donors:
        intrinsic_pka, func_group = get_donor_pka(atom, structure_context)
        
        # Metal-assisted shift
        eff_pka = effective_pka(intrinsic_pka, donor_hsab, metal_hsab, metal_charge)
        
        # Fraction available at working pH
        frac = fraction_deprotonated(eff_pka, ph)
        frac_no_metal = fraction_deprotonated(intrinsic_pka, ph)
        
        is_protonated = frac < 0.5
        is_dead = frac < 0.05
        
        effective_n += frac
        
        donor_profiles.append(DonorAvailability(
            atom=atom,
            functional_group=func_group,
            intrinsic_pka=intrinsic_pka,
            effective_pka=round(eff_pka, 2),
            fraction_available=round(frac, 4),
            fraction_without_metal=round(frac_no_metal, 4),
            is_protonated=is_protonated,
            is_dead=is_dead,
        ))
    
    nominal = len(donors)
    effective_n = round(effective_n, 2)
    loss = nominal - effective_n
    
    # ΔG penalty for protonation
    # Each protonated donor: compete with proton for binding site
    # ΔG_protonation ≈ RT × ln(10) × (pKa - pH) per protonated donor
    # This is the free energy cost of displacing the proton
    RT_ln10 = 5.71  # RT × ln(10) at 298K in kJ/mol
    dG_prot = 0.0
    for d in donor_profiles:
        if d.effective_pka > ph:
            # Cost of proton displacement: favorable for binding but costs energy
            dG_prot += RT_ln10 * max(0, d.effective_pka - ph) * 0.3
            # Factor of 0.3: not full thermodynamic penalty because metal binding
            # energy partially compensates. This is the RESIDUAL penalty.
    
    # Chelate effect correction: reduced effective denticity means fewer ring closures
    effective_int = max(1, round(effective_n))
    chelate_correction = 0.0
    if nominal > 1 and effective_int < nominal:
        # Lost ring closures: each costs ~6 kJ/mol of chelate stabilization
        lost_rings = (nominal - 1) - max(0, effective_int - 1)
        chelate_correction = 6.0 * lost_rings  # positive = penalty
    
    frac_total = effective_n / nominal if nominal > 0 else 0.0
    
    breakdown.append(f"pH {ph:.1f}: {nominal} donors → {effective_n:.1f} effective")
    if loss > 0.5:
        breakdown.append(f"Lost {loss:.1f} donors to protonation")
    
    return ProtonationProfile(
        donors=donor_profiles,
        nominal_denticity=nominal,
        effective_denticity=effective_n,
        denticity_loss=loss,
        dG_protonation_penalty=round(dG_prot, 2),
        chelate_effect_correction=round(chelate_correction, 2),
        ph=ph,
        fraction_total_available=round(frac_total, 4),
        breakdown=breakdown,
    )
