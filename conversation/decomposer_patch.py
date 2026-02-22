"""
conversation/decomposer_patch.py - Adds more targets to the decomposer.
Sprint 3: mercury, silver, uranium, cerium to support DNAzyme library.
Import and call patch_targets() to extend KNOWN_TARGETS.
"""

from core.problem import (
    TargetSpecies, ElectronicDescription, HydrationDescription,
    RedoxState, MagneticDescription, SizeDescription,
)

ADDITIONAL_TARGETS = {
    "mercury": TargetSpecies(
        identity="mercury", formula="Hg(2+)", charge=2.0,
        geometry="linear to tetrahedral",
        electronic=ElectronicDescription(hardness_softness="soft", electronegativity=2.00, donor_atoms=["S","N"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.1, dehydration_energy_kj_mol=1824.0),
        redox_states=[RedoxState(2, "Hg(2+)"), RedoxState(1, "Hg2(2+)"), RedoxState(0, "Hg(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.02, molecular_weight=200.59),
    ),
    "silver": TargetSpecies(
        identity="silver", formula="Ag(+)", charge=1.0,
        geometry="linear",
        electronic=ElectronicDescription(hardness_softness="soft", electronegativity=1.93, donor_atoms=["S","N"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.41, dehydration_energy_kj_mol=473.0),
        redox_states=[RedoxState(1, "Ag(+)"), RedoxState(0, "Ag(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.15, molecular_weight=107.87),
    ),
    "uranium": TargetSpecies(
        identity="uranium", formula="UO2(2+)", charge=2.0,
        geometry="pentagonal bipyramidal",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.38, donor_atoms=["O"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.2),
        redox_states=[RedoxState(6, "UO2(2+)"), RedoxState(4, "U(4+)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=2),
        size=SizeDescription(ionic_radius_angstrom=0.73, molecular_weight=270.03),
        notes="Uranyl - linear O=U=O with equatorial coordination",
    ),
    "cerium": TargetSpecies(
        identity="cerium", formula="Ce(3+)", charge=3.0,
        geometry="variable 8-9 coordinate",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.12, donor_atoms=["O"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.5, coordination_number_water=9),
        redox_states=[RedoxState(3, "Ce(3+)"), RedoxState(4, "Ce(4+)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=1),
        size=SizeDescription(ionic_radius_angstrom=1.01, molecular_weight=140.12),
        notes="Lanthanide - representative of all rare earth elements",
    ),
    "arsenic": TargetSpecies(
        identity="arsenic", formula="AsO4(3-)", charge=-3.0,
        geometry="tetrahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=2.18, donor_atoms=["O"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.7),
        redox_states=[RedoxState(5, "AsO4(3-)"), RedoxState(3, "AsO3(3-)"), RedoxState(0, "As(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=0.46, molecular_weight=138.92),
    ),
    "cadmium": TargetSpecies(
        identity="cadmium", formula="Cd(2+)", charge=2.0,
        geometry="tetrahedral to octahedral",
        electronic=ElectronicDescription(hardness_softness="soft", electronegativity=1.69, donor_atoms=["S","N","O"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.26, dehydration_energy_kj_mol=1807.0, coordination_number_water=6),
        redox_states=[RedoxState(2, "Cd(2+)"), RedoxState(0, "Cd(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=0.95, molecular_weight=112.41),
    ),
    "zinc": TargetSpecies(
        identity="zinc", formula="Zn(2+)", charge=2.0,
        geometry="tetrahedral (preferred)",
        electronic=ElectronicDescription(hardness_softness="borderline", electronegativity=1.65, donor_atoms=["N","O","S"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.30, dehydration_energy_kj_mol=2046.0, coordination_number_water=6),
        redox_states=[RedoxState(2, "Zn(2+)"), RedoxState(0, "Zn(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=0.74, molecular_weight=65.38),
    ),
    "iron": TargetSpecies(
        identity="iron", formula="Fe(3+)", charge=3.0,
        geometry="octahedral",
        electronic=ElectronicDescription(hardness_softness="hard", electronegativity=1.83, donor_atoms=["O","N"]),
        hydration=HydrationDescription(hydrated_radius_angstrom=4.57, dehydration_energy_kj_mol=4430.0, coordination_number_water=6),
        redox_states=[RedoxState(3, "Fe(3+)"), RedoxState(2, "Fe(2+)"), RedoxState(0, "Fe(0)")],
        magnetic=MagneticDescription(type="paramagnetic", unpaired_electrons=5),
        size=SizeDescription(ionic_radius_angstrom=0.65, molecular_weight=55.85),
    ),
}


def patch_targets():
    """Add these targets to the decomposer's KNOWN_TARGETS."""
    from conversation.decomposer import KNOWN_TARGETS
    KNOWN_TARGETS.update(ADDITIONAL_TARGETS)
