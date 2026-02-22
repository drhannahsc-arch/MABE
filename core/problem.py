"""
core/problem.py - MABE internal representation of molecular design problems.

Everything here describes physics, not chemistry categories.
A Problem is a question about energy landscapes, not about binders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ElectronicDescription:
    """Electronic structure relevant to molecular interactions."""
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    polarizability: Optional[float] = None
    electronegativity: Optional[float] = None
    hardness_softness: Optional[str] = None
    donor_atoms: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class HydrationDescription:
    """How the target interacts with water."""
    hydrated_radius_angstrom: Optional[float] = None
    dehydration_energy_kj_mol: Optional[float] = None
    coordination_number_water: Optional[int] = None
    notes: str = ""


@dataclass
class RedoxState:
    """One possible oxidation state and its properties."""
    oxidation_state: int
    formula: str
    stable_ph_range: Optional[tuple[float, float]] = None
    standard_potential_v: Optional[float] = None
    notes: str = ""


@dataclass
class MagneticDescription:
    """Magnetic properties - determines field response when captured."""
    type: str = "diamagnetic"
    susceptibility: Optional[float] = None
    unpaired_electrons: int = 0


@dataclass
class SizeDescription:
    """Size at different levels."""
    ionic_radius_angstrom: Optional[float] = None
    hydrated_radius_angstrom: Optional[float] = None
    vdw_radius_angstrom: Optional[float] = None
    molecular_weight: Optional[float] = None


@dataclass
class TargetSpecies:
    """A molecular species described by its physics, not its name."""
    identity: str
    formula: str
    charge: float
    geometry: str
    electronic: ElectronicDescription = field(default_factory=ElectronicDescription)
    hydration: HydrationDescription = field(default_factory=HydrationDescription)
    redox_states: list[RedoxState] = field(default_factory=list)
    magnetic: MagneticDescription = field(default_factory=MagneticDescription)
    size: SizeDescription = field(default_factory=SizeDescription)
    notes: str = ""

    def summary(self) -> str:
        parts = [f"{self.identity} ({self.formula}), charge {self.charge:+.0f}"]
        if self.geometry:
            parts.append(self.geometry)
        if self.electronic.hardness_softness:
            parts.append(f"HSAB: {self.electronic.hardness_softness}")
        if self.size.ionic_radius_angstrom:
            parts.append(f"r={self.size.ionic_radius_angstrom} A")
        return ", ".join(parts)


@dataclass
class CompetingSpecies:
    """Something else in the matrix that could interfere."""
    identity: str
    formula: str
    concentration_mm: float
    charge: float = 0.0
    notes: str = ""


@dataclass
class Matrix:
    """The full physical/chemical environment."""
    description: str = ""
    ph: Optional[float] = None
    temperature_c: float = 25.0
    ionic_strength_mm: Optional[float] = None
    redox_potential_mv: Optional[float] = None
    competing_species: list[CompetingSpecies] = field(default_factory=list)
    flow_rate_l_min: Optional[float] = None
    pressure_atm: float = 1.0
    notes: str = ""


@dataclass
class Outcome:
    """What energy landscape trajectory the user wants. Open-ended, not an enum."""
    description: str
    reversible: Optional[bool] = None
    trigger: Optional[str] = None
    product: Optional[str] = None
    destination: Optional[str] = None
    notes: str = ""


@dataclass
class Exclusion:
    """Something that must NOT happen. Repulsion is half the design."""
    description: str
    reason: str = ""


@dataclass
class Constraints:
    """Real-world limits. Encodes values: cost, accessibility, waste, reusability."""
    max_cost_per_unit: Optional[str] = None
    required_reusability_cycles: Optional[int] = None
    no_environmental_release: bool = True
    available_equipment: list[str] = field(default_factory=list)
    scale: str = "lab"
    exclusions: list[Exclusion] = field(default_factory=list)
    notes: str = ""


@dataclass
class Problem:
    """A molecular design problem expressed in physics terms. MABE's core unit."""
    target: TargetSpecies
    matrix: Matrix
    desired_outcome: Outcome
    constraints: Constraints = field(default_factory=Constraints)
    original_query: str = ""
    assumptions_made: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Target: {self.target.summary()}",
            f"Matrix: {self.matrix.description or 'unspecified'}",
            f"Desired outcome: {self.desired_outcome.description}",
        ]
        if self.constraints.exclusions:
            excl = ", ".join(e.description for e in self.constraints.exclusions)
            lines.append(f"Must NOT: {excl}")
        if self.assumptions_made:
            lines.append(f"Assumptions: {'; '.join(self.assumptions_made)}")
        return "\n".join(lines)
