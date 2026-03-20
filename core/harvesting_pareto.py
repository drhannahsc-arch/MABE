"""
core/harvesting_pareto.py -- 4-Objective Pareto Optimizer for Energy Harvesting

Extends the 3-objective multi-physics Pareto optimization (color, thermal,
acoustic) with a 4th power objective for energy harvesting building elements.

Uses the same non-domination logic as core/multiphysics_pareto.py but with
4 objectives: (color_delta_E, -R_value, -NRC, -power_W_m2).

Grid search over design variables:
  - Particle: diameter, material, volume fraction, absorber fraction, thickness
  - PV material, PV thickness
  - TE material
  - Piezo material

The optimizer evaluates each combination and extracts the 4-D Pareto front.

Phase 5 of the Energy Harvesting module.
Does NOT modify core/multiphysics_pareto.py (zero regression).
Depends on: Phase 2 (energy_harvesting), Phase 4 (active_element_profiles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.energy_harvesting import (
    predict_pv_power,
    predict_teg_power,
    predict_piezo_power,
    predict_acoustic_harvest,
    predict_total_harvest,
    EnvironmentSpec,
    HarvestingSpec,
    PowerBudget,
    list_pv_materials,
    list_te_materials,
    list_piezo_materials,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HarvestDesignPoint:
    """One evaluated design in the 4-objective space."""
    # Harvesting parameters
    pv_material: str
    pv_thickness_nm: float
    te_material: str
    piezo_material: str

    # Predictions
    power_budget: Optional[PowerBudget] = None

    # Objectives (all minimization: negate to maximize)
    obj_color_delta_e: float = 999.0        # ΔE* from target
    obj_thermal_neg_R: float = 0.0          # -R_value (minimize = maximize R)
    obj_acoustic_neg_nrc: float = 0.0       # -NRC (minimize = maximize NRC)
    obj_power_neg_W_m2: float = 0.0         # -power (minimize = maximize power)

    # Pareto status
    is_pareto: bool = False
    dominated_by: int = 0

    @property
    def objectives(self) -> Tuple[float, float, float, float]:
        return (self.obj_color_delta_e, self.obj_thermal_neg_R,
                self.obj_acoustic_neg_nrc, self.obj_power_neg_W_m2)

    def summary(self) -> str:
        lines = [
            f"PV={self.pv_material} t={self.pv_thickness_nm:.0f}nm "
            f"TE={self.te_material} Piezo={self.piezo_material}",
        ]
        if self.power_budget:
            pb = self.power_budget
            lines.append(f"  Power: PV={pb.pv_W_m2:.3f} TEG={pb.teg_W_m2:.3f} "
                         f"Piezo={pb.piezo_W_m2:.2e} Total={pb.total_W_m2:.3f} W/m2")
            lines.append(f"  Daily: {pb.daily_kWh_m2:.5f} kWh/m2")
        lines.append(f"  Objectives: dE={self.obj_color_delta_e:.1f} "
                     f"-R={self.obj_thermal_neg_R:.4f} "
                     f"-NRC={self.obj_acoustic_neg_nrc:.3f} "
                     f"-P={self.obj_power_neg_W_m2:.3f}")
        if self.is_pareto:
            lines.append(f"  * PARETO OPTIMAL")
        return "\n".join(lines)


@dataclass
class HarvestParetoResult:
    """Complete 4-D Pareto optimization output."""
    all_designs: List[HarvestDesignPoint]
    pareto_front: List[HarvestDesignPoint]
    recommended: Optional[HarvestDesignPoint] = None
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    @property
    def n_designs(self) -> int:
        return len(self.all_designs)

    @property
    def n_pareto(self) -> int:
        return len(self.pareto_front)

    def summary(self) -> str:
        lines = [
            f"Harvest Pareto Result: {self.n_designs} designs, {self.n_pareto} on front",
            f"Weights: color={self.weights[0]:.2f} thermal={self.weights[1]:.2f} "
            f"acoustic={self.weights[2]:.2f} power={self.weights[3]:.2f}",
        ]
        if self.recommended:
            lines.append("Recommended:")
            lines.append(self.recommended.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pareto dominance (4-D)
# ---------------------------------------------------------------------------

def _dominates_4d(a: HarvestDesignPoint, b: HarvestDesignPoint) -> bool:
    """Does design `a` dominate `b`? (all objectives <=, at least one <)"""
    objs_a = a.objectives
    objs_b = b.objectives
    all_leq = all(oa <= ob for oa, ob in zip(objs_a, objs_b))
    any_lt = any(oa < ob for oa, ob in zip(objs_a, objs_b))
    return all_leq and any_lt


def extract_pareto_front_4d(
    designs: List[HarvestDesignPoint],
) -> List[HarvestDesignPoint]:
    """Extract non-dominated designs from a 4-objective set."""
    n = len(designs)
    for i in range(n):
        designs[i].dominated_by = 0
        designs[i].is_pareto = True

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates_4d(designs[j], designs[i]):
                designs[i].dominated_by += 1
                designs[i].is_pareto = False

    return [d for d in designs if d.is_pareto]


# ---------------------------------------------------------------------------
# Weighted balanced recommendation
# ---------------------------------------------------------------------------

def _weighted_score(
    design: HarvestDesignPoint,
    weights: Tuple[float, float, float, float],
    ranges: Tuple[float, float, float, float],
) -> float:
    """
    Compute weighted normalized score (lower is better).

    Each objective is normalized to [0, 1] by dividing by the range
    in the current population, then weighted.
    """
    objs = design.objectives
    score = 0.0
    for obj, w, r in zip(objs, weights, ranges):
        norm = obj / r if r > 0 else 0.0
        score += w * norm
    return score


# ---------------------------------------------------------------------------
# Grid search optimizer
# ---------------------------------------------------------------------------

def optimize_harvesting(
    environment: Optional[EnvironmentSpec] = None,
    transmitted_fraction: float = 0.5,
    pv_materials: Optional[List[str]] = None,
    pv_thicknesses_nm: Optional[List[float]] = None,
    te_materials: Optional[List[str]] = None,
    piezo_materials: Optional[List[str]] = None,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    obj_color_delta_e: float = 5.0,
    obj_thermal_neg_R: float = -0.01,
    obj_acoustic_neg_nrc: float = -0.5,
) -> HarvestParetoResult:
    """
    Grid search over harvesting material combinations.

    The color, thermal, and acoustic objectives are passed in as fixed values
    (from an upstream multi-physics optimization or profile). The power objective
    is computed by the harvesting forward models for each combination.

    Parameters
    ----------
    environment : EnvironmentSpec or None
        Environmental conditions.
    transmitted_fraction : float
        Fraction of solar spectrum reaching PV layer.
    pv_materials : list of str or None
        PV materials to search. Defaults to all.
    pv_thicknesses_nm : list of float or None
        PV thicknesses to search. Defaults to [100, 300, 500].
    te_materials : list of str or None
        TE materials to search. Defaults to all.
    piezo_materials : list of str or None
        Piezo materials to search. Defaults to all.
    weights : 4-tuple
        (color, thermal, acoustic, power) weights for recommendation.
    obj_color_delta_e : float
        Fixed color objective (from upstream optimization).
    obj_thermal_neg_R : float
        Fixed thermal objective (from upstream optimization).
    obj_acoustic_neg_nrc : float
        Fixed acoustic objective (from upstream optimization).

    Returns
    -------
    HarvestParetoResult
    """
    env = environment or EnvironmentSpec()

    if pv_materials is None:
        pv_materials = list_pv_materials()
    if pv_thicknesses_nm is None:
        pv_thicknesses_nm = [100.0, 300.0, 500.0]
    if te_materials is None:
        te_materials = list_te_materials()
    if piezo_materials is None:
        piezo_materials = list_piezo_materials()

    designs: List[HarvestDesignPoint] = []

    for pv_mat in pv_materials:
        for pv_t in pv_thicknesses_nm:
            for te_mat in te_materials:
                for piezo_mat in piezo_materials:
                    h_spec = HarvestingSpec(
                        pv_material=pv_mat,
                        pv_thickness_nm=pv_t,
                        te_material=te_mat,
                        piezo_material=piezo_mat,
                        area_m2=1.0,
                    )
                    pb = predict_total_harvest(
                        harvesting_spec=h_spec,
                        environment_spec=env,
                        transmitted_fraction=transmitted_fraction,
                    )
                    dp = HarvestDesignPoint(
                        pv_material=pv_mat,
                        pv_thickness_nm=pv_t,
                        te_material=te_mat,
                        piezo_material=piezo_mat,
                        power_budget=pb,
                        obj_color_delta_e=obj_color_delta_e,
                        obj_thermal_neg_R=obj_thermal_neg_R,
                        obj_acoustic_neg_nrc=obj_acoustic_neg_nrc,
                        obj_power_neg_W_m2=-pb.total_W_m2,
                    )
                    designs.append(dp)

    # Extract Pareto front
    front = extract_pareto_front_4d(designs)

    # Weighted recommendation from Pareto front
    recommended = None
    if front:
        # Compute ranges for normalization
        all_objs = [d.objectives for d in designs]
        ranges = tuple(
            max(abs(max(o[i] for o in all_objs) - min(o[i] for o in all_objs)), 1e-12)
            for i in range(4)
        )
        best_score = float("inf")
        for d in front:
            s = _weighted_score(d, weights, ranges)
            if s < best_score:
                best_score = s
                recommended = d

    return HarvestParetoResult(
        all_designs=designs,
        pareto_front=front,
        recommended=recommended,
        weights=weights,
    )
