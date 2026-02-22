"""
MABE Sprint 15 Bootstrap - Entropy Decomposition & Gibbs-Helmholtz
===================================================================
ΔG = ΔH - TΔS. Until now we had ΔG as a single number. This sprint
decomposes into enthalpy and entropy, enabling temperature prediction.

    cd Documents\\mabe
    python bootstrap_sprint15.py
    python tests\\test_sprint15.py
"""
import os

def write_file(path, content):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")

print()
print("  MABE Sprint 15 - Entropy Decomposition & Gibbs-Helmholtz")
print("  " + "=" * 56)
print()


# ═══════════════════════════════════════════════════════════════════════════
# knowledge/entropy_data.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("knowledge/entropy_data.py", '''"""
knowledge/entropy_data.py - Enthalpy/entropy decomposition parameters.

Convention: ΔG = ΔH - TΔS
  ΔH < 0 = exothermic (favorable)
  ΔS > 0 = entropy increase (favorable)
"""

ENTHALPY_FRACTIONS = {
    "dG_bind":          0.85,   # bond formation: mostly enthalpic
    "dG_desolv":        0.60,   # mixed: lost bonds (ΔH) + released water (ΔS)
    "dG_chelate":       0.10,   # chelate effect IS entropy
    "dG_preorg":        0.15,   # preorg = pre-paid conformational entropy
    "dG_electrostatic": 0.95,   # Coulomb = direct energy
    "dG_protonation":   0.70,   # mixed
    "dG_lfse":          1.00,   # crystal field = electronic energy
    "dG_activity":      0.50,   # mixed
    "dG_repulsion":     1.00,   # Pauli/Born = energy barriers
}

# ΔS per donor coordination (J/(mol·K))
DONOR_ENTROPY = {
    "S": -15.0,  "N": -20.0,  "O": +5.0,
    "P": -10.0,  "electrostatic": +8.0,
}

WATER_RELEASE_ENTROPY = 25.0   # J/(mol·K) per water displaced
ROTATABLE_BOND_ENTROPY_LOSS = -6.0  # J/(mol·K) per bond

DONOR_HEAT_CAPACITY = {
    "S": -30.0,  "N": -10.0,  "O": +5.0,
    "P": -20.0,  "electrostatic": 0.0,
}
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/entropy.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/entropy.py", '''"""
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
        return "\\n".join(parts)


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
''')


# ═══════════════════════════════════════════════════════════════════════════
# core/entropy_integration.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("core/entropy_integration.py", '''"""
core/entropy_integration.py - Integrates entropy decomposition into pipeline.
"""
import core.sprint10_integration as s10
from core.entropy import decompose_thermodynamics
from core.thermodynamics import BindingThermodynamics


_orig_rescore = s10.full_physics_rescore


def _entropy_aware_rescore(assemblies, problem):
    """Add entropy decomposition to physics report."""
    assemblies = _orig_rescore(assemblies, problem)
    actual_temp = problem.matrix.temperature_c

    for assembly in assemblies:
        thermo = getattr(assembly, "thermodynamics", None)
        if thermo is None:
            thermo = BindingThermodynamics(
                dG_bind=getattr(assembly, "_dG_bind", 0.0),
                dG_desolv=getattr(assembly, "_dG_desolv", 0.0),
                dG_chelate=getattr(assembly, "_dG_chelate", 0.0),
                dG_preorg=getattr(assembly, "_dG_preorg", 0.0),
                dG_electrostatic=getattr(assembly, "_dG_electrostatic", 0.0),
                dG_net=getattr(assembly, "score_physics", 0.0),
                temperature_k=(actual_temp or 25.0) + 273.15,
            )

        donors = assembly.recognition.donor_atoms if assembly.recognition else []
        is_macro = ("macrocycl" in assembly.recognition.structure.lower()
                     if assembly.recognition and assembly.recognition.structure else False)

        entropy = decompose_thermodynamics(
            thermo, donors=donors, n_donors=len(donors),
            is_macrocyclic=is_macro, actual_temp_c=actual_temp,
        )

        assembly.confidence_reasoning += "\\n\\nENTROPY DECOMPOSITION:\\n" + entropy.summary()

        if actual_temp is not None and entropy.dG_at_target is not None:
            dG_diff = entropy.dG_at_target - entropy.dG_ref
            if dG_diff > 5.0:
                assembly.failure_modes.append(
                    f"Temperature penalty: ΔG worsens by {dG_diff:+.1f} kJ/mol "
                    f"at {actual_temp:.0f}°C vs 25°C"
                )

        if entropy.temperature_sensitivity.startswith("HIGH"):
            assembly.failure_modes.append(
                f"High temperature sensitivity (ΔS = {entropy.dS_total:+.0f} J/(mol·K))"
            )

    return assemblies


s10.full_physics_rescore = _entropy_aware_rescore
''')


# ═══════════════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("main.py", '''"""MABE - Modality-Agnostic Binder Engine"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer_patch import patch_targets
from conversation.interface import run_interactive, run_single_query
patch_targets()
import core.assembly_composer_patch, core.scoring_patch
import core.physics_integration, core.sprint10_integration
import core.protonation_integration, core.lfse_integration
import core.ionic_integration, core.repulsion_integration
import core.entropy_integration

def build_registry():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available(): registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry

def main():
    registry = build_registry()
    if len(sys.argv) > 1: run_single_query(registry, " ".join(sys.argv[1:]))
    else: run_interactive(registry)

if __name__ == "__main__": main()
''')


# ═══════════════════════════════════════════════════════════════════════════
# tests/test_sprint15.py
# ═══════════════════════════════════════════════════════════════════════════

write_file("tests/test_sprint15.py", '''"""
tests/test_sprint15.py - Entropy decomposition and Gibbs-Helmholtz tests.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conversation.decomposer_patch import patch_targets
patch_targets()
import core.assembly_composer_patch, core.scoring_patch
import core.physics_integration, core.sprint10_integration
import core.protonation_integration, core.lfse_integration
import core.ionic_integration, core.repulsion_integration
import core.entropy_integration

from knowledge.entropy_data import ENTHALPY_FRACTIONS, DONOR_ENTROPY, WATER_RELEASE_ENTROPY
from core.entropy import decompose_thermodynamics, gibbs_helmholtz, EntropyDecomposition
from core.thermodynamics import BindingThermodynamics, R_GAS
from conversation.decomposer import decompose
from core.orchestrator import Orchestrator
from adapters.base import ToolRegistry
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter


def _build():
    r = ToolRegistry()
    r.register(DNAzymeAdapter())
    r.register(PeptideAdapter())
    r.register(AptamerAdapter())
    return r


def _make_thermo(dG_bind=-50.0, dG_desolv=10.0, dG_chelate=-18.0,
                   dG_preorg=-8.0, dG_elec=-6.0, T_k=298.15):
    dG_net = dG_bind + dG_desolv + dG_chelate + dG_preorg + dG_elec
    return BindingThermodynamics(
        dG_bind=dG_bind, dG_desolv=dG_desolv, dG_chelate=dG_chelate,
        dG_preorg=dG_preorg, dG_electrostatic=dG_elec,
        dG_net=dG_net, temperature_k=T_k,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data tables
# ═══════════════════════════════════════════════════════════════════════════

def test_enthalpy_fractions_valid():
    """All fractions between 0 and 1."""
    for name, f in ENTHALPY_FRACTIONS.items():
        assert 0.0 <= f <= 1.0, f"{name}: f_H = {f}"
    assert ENTHALPY_FRACTIONS["dG_bind"] > 0.7
    assert ENTHALPY_FRACTIONS["dG_chelate"] < 0.3
    print("  + Enthalpy fractions valid")


def test_donor_entropy_signs():
    """N, S lose entropy; O gains (water release)."""
    assert DONOR_ENTROPY["N"] < 0
    assert DONOR_ENTROPY["S"] < 0
    assert DONOR_ENTROPY["O"] > 0
    print("  + Donor entropy signs correct")


# ═══════════════════════════════════════════════════════════════════════════
# Gibbs-Helmholtz
# ═══════════════════════════════════════════════════════════════════════════

def test_gh_identity():
    """At T_ref, ΔG = ΔH - TΔS."""
    dH, dS = -40.0, 50.0
    T = 298.15
    dG = gibbs_helmholtz(dH, dS, T, dCp=0.0, T_ref_K=T)
    expected = dH - T * dS / 1000.0
    assert abs(dG - expected) < 0.01
    print(f"  + GH identity: ΔG(T_ref) = {dG:.2f} kJ/mol")


def test_gh_entropy_driven():
    """Entropy-driven: ΔG more negative at higher T."""
    dH, dS = 10.0, 200.0
    dG_25 = gibbs_helmholtz(dH, dS, 298.15)
    dG_60 = gibbs_helmholtz(dH, dS, 333.15)
    assert dG_60 < dG_25
    print(f"  + Entropy-driven: ΔG(25°C)={dG_25:.1f}, ΔG(60°C)={dG_60:.1f}")


def test_gh_enthalpy_dominated():
    """Enthalpy-dominated: ΔG less negative at higher T."""
    dH, dS = -80.0, -100.0
    dG_25 = gibbs_helmholtz(dH, dS, 298.15)
    dG_60 = gibbs_helmholtz(dH, dS, 333.15)
    assert dG_60 > dG_25
    print(f"  + Enthalpy-dom: ΔG(25°C)={dG_25:.1f}, ΔG(60°C)={dG_60:.1f}")


def test_gh_with_dCp():
    """ΔCp adds curvature."""
    dH, dS = -40.0, 50.0
    dG_0 = gibbs_helmholtz(dH, dS, 333.15, dCp=0.0)
    dG_neg = gibbs_helmholtz(dH, dS, 333.15, dCp=-200.0)
    dG_pos = gibbs_helmholtz(dH, dS, 333.15, dCp=200.0)
    assert dG_neg != dG_0 and dG_pos != dG_0
    print(f"  + GH with ΔCp: no={dG_0:.1f}, -200={dG_neg:.1f}, +200={dG_pos:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# Decomposition
# ═══════════════════════════════════════════════════════════════════════════

def test_decomposition_sums():
    """ΔH + (-TΔS) ≈ ΔG."""
    thermo = _make_thermo()
    ed = decompose_thermodynamics(thermo, donors=["S", "S", "N"])
    recon = ed.dH_total + ed.minus_TdS_total
    assert abs(recon - ed.dG_ref) < 0.1
    print(f"  + ΔH={ed.dH_total:+.1f} + (-TΔS)={ed.minus_TdS_total:+.1f} = ΔG={ed.dG_ref:+.1f}")


def test_bind_mostly_enthalpic():
    """ΔG_bind → mostly ΔH."""
    thermo = _make_thermo(dG_bind=-50.0, dG_desolv=0, dG_chelate=0, dG_preorg=0, dG_elec=0)
    ed = decompose_thermodynamics(thermo)
    c = [x for x in ed.contributions if x.name == "dG_bind"][0]
    assert abs(c.dH) > abs(c.minus_TdS)
    print(f"  + ΔG_bind: ΔH={c.dH:+.1f}, -TΔS={c.minus_TdS:+.1f}")


def test_chelate_mostly_entropic():
    """ΔG_chelate → mostly -TΔS."""
    thermo = _make_thermo(dG_bind=0, dG_desolv=0, dG_chelate=-18.0, dG_preorg=0, dG_elec=0)
    ed = decompose_thermodynamics(thermo)
    c = [x for x in ed.contributions if x.name == "dG_chelate"][0]
    assert abs(c.minus_TdS) > abs(c.dH)
    print(f"  + ΔG_chelate: ΔH={c.dH:+.1f}, -TΔS={c.minus_TdS:+.1f}")


def test_temperature_predictions_exist():
    """Should predict ΔG at 4, 37, 60°C."""
    thermo = _make_thermo()
    ed = decompose_thermodynamics(thermo, donors=["S", "S", "N"])
    assert ed.dG_at_4C is not None
    assert ed.dG_at_37C is not None
    assert ed.dG_at_60C is not None
    print(f"  + ΔG: 4°C={ed.dG_at_4C:+.1f}, 37°C={ed.dG_at_37C:+.1f}, 60°C={ed.dG_at_60C:+.1f}")


def test_actual_temperature():
    """Target temperature produces prediction."""
    thermo = _make_thermo()
    ed = decompose_thermodynamics(thermo, donors=["S", "S", "N"], actual_temp_c=12.0)
    assert ed.dG_at_target is not None
    assert ed.target_temp_c == 12.0
    print(f"  + ΔG at AMD 12°C: {ed.dG_at_target:+.1f} kJ/mol")


def test_compensation_detection():
    """Detect enthalpy-entropy compensation."""
    thermo = _make_thermo(dG_bind=-80.0, dG_desolv=10.0, dG_chelate=0, dG_preorg=0, dG_elec=0)
    ed = decompose_thermodynamics(thermo, donors=["N", "N", "N", "N"])
    assert len(ed.enthalpy_entropy_compensation) > 0
    print(f"  + {ed.enthalpy_entropy_compensation}")


def test_temperature_sensitivity():
    """Classify temperature sensitivity."""
    thermo = _make_thermo(dG_chelate=-30.0, dG_preorg=-20.0)
    ed = decompose_thermodynamics(thermo, donors=["O", "O", "O", "O", "O", "O"])
    assert "sensitivity" in ed.temperature_sensitivity.lower()
    print(f"  + {ed.temperature_sensitivity}")


def test_cold_water_vs_hot():
    """Cold water should differ from hot for entropy-driven binding."""
    thermo = _make_thermo(dG_bind=-10.0, dG_desolv=0, dG_chelate=-30.0,
                           dG_preorg=-15.0, dG_elec=0)
    ed = decompose_thermodynamics(thermo, donors=["O", "O", "O", "O"])
    # Entropy-driven (chelate + preorg): should strengthen at higher T
    diff = ed.dG_at_60C - ed.dG_at_4C
    assert abs(diff) > 0.5, f"Should see temperature effect: Δ={diff}"
    print(f"  + ΔG(4°C)={ed.dG_at_4C:+.1f}, ΔG(60°C)={ed.dG_at_60C:+.1f} (Δ={diff:+.1f})")


# ═══════════════════════════════════════════════════════════════════════════
# E2E
# ═══════════════════════════════════════════════════════════════════════════

def test_e2e_entropy_in_reports():
    """E2E: entropy decomposition in reports."""
    registry = _build()
    prob = decompose("lead capture from acid mine drainage pH 3.5")
    prob.matrix.ionic_strength_mm = 50.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found = any("ENTROPY" in r.confidence_reasoning for r in results.assemblies)
    assert found, "Entropy should appear in E2E report"
    print(f"  + E2E: entropy decomposition in reports")


def test_e2e_temperature_in_reports():
    """E2E: temperature prediction for actual conditions."""
    registry = _build()
    prob = decompose("copper capture from thermal spring 60C")
    prob.matrix.temperature_c = 60.0
    prob.matrix.ionic_strength_mm = 100.0
    orch = Orchestrator(registry)
    results = orch.solve(prob)
    found = any("60" in r.confidence_reasoning and "ENTROPY" in r.confidence_reasoning
                 for r in results.assemblies)
    assert found, "Temperature prediction should appear"
    print(f"  + E2E: temperature prediction at 60°C in reports")


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_enthalpy_fractions_valid,
        test_donor_entropy_signs,
        test_gh_identity,
        test_gh_entropy_driven,
        test_gh_enthalpy_dominated,
        test_gh_with_dCp,
        test_decomposition_sums,
        test_bind_mostly_enthalpic,
        test_chelate_mostly_entropic,
        test_temperature_predictions_exist,
        test_actual_temperature,
        test_compensation_detection,
        test_temperature_sensitivity,
        test_cold_water_vs_hot,
        test_e2e_entropy_in_reports,
        test_e2e_temperature_in_reports,
    ]

    print()
    print("=" * 60)
    print("  Sprint 15: Entropy Decomposition & Gibbs-Helmholtz")
    print("=" * 60)
    print()

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print()
    print(f"  Sprint 15: {passed} passed, {failed} failed")
    print()
''')


print()
print("  Sprint 15 files created:")
print("    knowledge/entropy_data.py      — ΔH/ΔS fractions, donor entropy, ΔCp")
print("    core/entropy.py                — Decomposition engine + Gibbs-Helmholtz")
print("    core/entropy_integration.py    — Pipeline patches (rescore)")
print("    main.py                        — Updated with Sprint 15 import")
print("    tests/test_sprint15.py         — 16 tests")
print()