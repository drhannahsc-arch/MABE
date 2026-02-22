"""
conversation/decomposer.py - Translates natural language into a Problem.
Sprint 1: Simple keyword matching. The point is proving the pipeline.
"""

from __future__ import annotations

from core.problem import (
    Problem, TargetSpecies, ElectronicDescription, HydrationDescription,
    RedoxState, MagneticDescription, SizeDescription,
    Matrix, CompetingSpecies, Outcome, Constraints, Exclusion,
)

KNOWN_TARGETS = {
    "selenite": TargetSpecies(
        identity="selenite", formula="SeO3(2-)", charge=-2.0,
        geometry="trigonal pyramidal",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=2.55, donor_atoms=["O"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.8, dehydration_energy_kj_mol=1080.0),
        redox_states=[
            RedoxState(6, "SeO4(2-)"), RedoxState(4, "SeO3(2-)"),
            RedoxState(0, "Se(0)", notes="elemental"), RedoxState(-2, "Se(2-)"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=2.39, molecular_weight=126.96),
    ),
    "lead": TargetSpecies(
        identity="lead", formula="Pb(2+)", charge=2.0,
        geometry="variable 4-8 coordinate",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=2.33, donor_atoms=["O","N","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.01, dehydration_energy_kj_mol=1481.0, coordination_number_water=9),
        redox_states=[RedoxState(2, "Pb(2+)"), RedoxState(0, "Pb(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.19, hydrated_radius_angstrom=4.01, molecular_weight=207.2),
    ),
    "nickel": TargetSpecies(
        identity="nickel", formula="Ni(2+)", charge=2.0,
        geometry="octahedral (preferred), square planar",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.91, donor_atoms=["N","O","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.04, dehydration_energy_kj_mol=2106.0, coordination_number_water=6),
        redox_states=[RedoxState(2, "Ni(2+)"), RedoxState(0, "Ni(0)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=2),
        size=SizeDescription(ionic_radius_angstrom=0.69, hydrated_radius_angstrom=4.04, molecular_weight=58.69),
    ),
    "gold": TargetSpecies(
        identity="gold", formula="Au(3+)", charge=3.0,
        geometry="square planar",
        electronic=ElectronicDescription(hardness_softness="soft", electronegativity=2.54, donor_atoms=["S","P","C"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.5, dehydration_energy_kj_mol=4690.0),
        redox_states=[RedoxState(3, "Au(3+)"), RedoxState(1, "Au(+)"), RedoxState(0, "Au(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=0.85, molecular_weight=196.97),
    ),
    "copper": TargetSpecies(
        identity="copper", formula="Cu(2+)", charge=2.0,
        geometry="Jahn-Teller distorted octahedral",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.90, donor_atoms=["N","O","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.19, dehydration_energy_kj_mol=2100.0, coordination_number_water=6),
        redox_states=[RedoxState(2, "Cu(2+)"), RedoxState(1, "Cu(+)"), RedoxState(0, "Cu(0)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=1),
        size=SizeDescription(ionic_radius_angstrom=0.73, hydrated_radius_angstrom=4.19, molecular_weight=63.55),
    ),
}

KNOWN_MATRICES = {
    "mine": Matrix(
        description="Acid mine drainage - typical BC/Canadian mine site",
        ph=3.5, temperature_c=12.0, ionic_strength_mm=50.0, redox_potential_mv=400.0,
        competing_species=[
            CompetingSpecies("sulfate", "SO4(2-)", 500.0, -2.0),
            CompetingSpecies("calcium", "Ca(2+)", 200.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 100.0, 2.0),
            CompetingSpecies("iron", "Fe(3+)", 50.0, 3.0),
        ],
    ),
    "river": Matrix(
        description="Freshwater river", ph=7.2, temperature_c=15.0, ionic_strength_mm=5.0,
        competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 40.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 15.0, 2.0),
        ],
    ),
    "ocean": Matrix(
        description="Seawater", ph=8.1, temperature_c=18.0, ionic_strength_mm=700.0,
        competing_species=[
            CompetingSpecies("sodium", "Na(+)", 468000.0, 1.0),
            CompetingSpecies("chloride", "Cl(-)", 546000.0, -1.0),
            CompetingSpecies("magnesium", "Mg(2+)", 52800.0, 2.0),
        ],
    ),
}


def decompose(user_input: str) -> Problem:
    text = user_input.lower().strip()
    assumptions = []

    target = None
    for name, species in KNOWN_TARGETS.items():
        if name in text:
            target = species
            break

    if target is None:
        target = TargetSpecies(identity="unknown target", formula="?", charge=0.0, geometry="unknown")
        assumptions.append(f"Could not identify a specific target in: '{user_input}'. Using generic target.")

    matrix = Matrix()
    for keyword, known_matrix in KNOWN_MATRICES.items():
        if keyword in text:
            matrix = known_matrix
            break

    if not matrix.description:
        matrix.description = "unspecified matrix"
        assumptions.append("No matrix specified - using default conditions (pH 7, 25C, freshwater)")
        matrix.ph = 7.0
        matrix.temperature_c = 25.0
        matrix.ionic_strength_mm = 10.0

    outcome_desc = "capture target"
    constraints = Constraints()

    if "release" in text or "feedstock" in text or "recover" in text:
        outcome_desc = "capture and release as clean feedstock"
        constraints.required_reusability_cycles = 20
    elif "transform" in text or "convert" in text:
        outcome_desc = "capture and transform to useful product"
    elif "place" in text or "deposit" in text or "chip" in text:
        outcome_desc = "selective extraction and precision placement"
    elif "remove" in text or "clean" in text:
        outcome_desc = "remove from environment"
        constraints.required_reusability_cycles = 50

    outcome = Outcome(description=outcome_desc)
    constraints.no_environmental_release = True
    constraints.exclusions.append(
        Exclusion("No nanomaterial release to environment", "MABE default - public good")
    )

    return Problem(
        target=target, matrix=matrix, desired_outcome=outcome,
        constraints=constraints, original_query=user_input, assumptions_made=assumptions,
    )
