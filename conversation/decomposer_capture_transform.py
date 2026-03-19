"""
conversation/decomposer_capture_transform.py — Capture-Transform Extensions

Extends the decomposer with capture-transform targets and matrices.
Also provides the adapter registration function.

Usage:
    from conversation.decomposer_capture_transform import (
        extend_decomposer,
        register_capture_transform,
    )

    # Add capture-transform targets to the decomposer
    extend_decomposer()

    # Register the adapter with a ToolRegistry
    register_capture_transform(registry)
"""

from __future__ import annotations

from core.problem import (
    TargetSpecies, ElectronicDescription, HydrationDescription,
    RedoxState, MagneticDescription, SizeDescription,
    Matrix, CompetingSpecies,
)


# ═══════════════════════════════════════════════════════════════════════════
# Capture-Transform Targets
# ═══════════════════════════════════════════════════════════════════════════

CAPTURE_TRANSFORM_TARGETS = {
    "carbon dioxide": TargetSpecies(
        identity="carbon dioxide", formula="CO2", charge=0.0,
        geometry="linear",
        electronic=ElectronicDescription(
            hardness_softness="hard", electronegativity=3.44,
            donor_atoms=["O"],
            notes="Lewis acid. Electrophilic carbon attacked by nucleophiles (amines, OH⁻).",
        ),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=1.65,
            dehydration_energy_kj_mol=0.0,
            notes="Kinetic diameter 3.3 Å. Hydration to HCO₃⁻ is rate-limiting (uncatalyzed).",
        ),
        redox_states=[
            RedoxState(4, "CO₂", notes="Oxidized carbon"),
            RedoxState(2, "CO / HCOOH", notes="Reduced — formate, CO"),
            RedoxState(0, "C", notes="Elemental carbon"),
            RedoxState(-4, "CH₄", notes="Fully reduced — methane"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(molecular_weight=44.01),
        notes="Primary DAC target. Mineralization to CaCO₃ is thermodynamically favorable.",
    ),
    "co2": None,  # alias — filled below

    "phosphate": TargetSpecies(
        identity="phosphate", formula="PO4(3-)", charge=-3.0,
        geometry="tetrahedral",
        electronic=ElectronicDescription(
            hardness_softness="hard",
            donor_atoms=["O"],
            notes="Hard oxyanion. pH-dependent speciation: H₃PO₄/H₂PO₄⁻/HPO₄²⁻/PO₄³⁻.",
        ),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=3.4,
            dehydration_energy_kj_mol=2765.0,
        ),
        redox_states=[RedoxState(-3, "PO₄³⁻")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=2.38, molecular_weight=94.97),
        notes="Critical nutrient. Recovery from wastewater as struvite/HAp = circular economy.",
    ),

    "ammonia": TargetSpecies(
        identity="ammonia", formula="NH3", charge=0.0,
        geometry="trigonal pyramidal",
        electronic=ElectronicDescription(
            hardness_softness="hard",
            donor_atoms=["N"],
            notes="Lewis base. Lone pair on N. pKa(NH₄⁺) = 9.25.",
        ),
        hydration=HydrationDescription(hydrated_radius_angstrom=1.5),
        redox_states=[
            RedoxState(-3, "NH₃/NH₄⁺"),
            RedoxState(0, "N₂"),
            RedoxState(+3, "NO₂⁻"),
            RedoxState(+5, "NO₃⁻"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(molecular_weight=17.03),
    ),

    "nitrate": TargetSpecies(
        identity="nitrate", formula="NO3(-)", charge=-1.0,
        geometry="trigonal planar",
        electronic=ElectronicDescription(
            hardness_softness="hard",
            donor_atoms=["O"],
        ),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.35),
        redox_states=[
            RedoxState(5, "NO₃⁻"),
            RedoxState(3, "NO₂⁻"),
            RedoxState(0, "N₂"),
            RedoxState(-3, "NH₃/NH₄⁺", notes="8e⁻ reduction"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.79, molecular_weight=62.00),
        notes="Agricultural runoff pollutant. Photocatalytic or ZVI reduction to NH₄⁺.",
    ),

    "fluoride": TargetSpecies(
        identity="fluoride", formula="F(-)", charge=-1.0,
        geometry="spherical",
        electronic=ElectronicDescription(
            hardness_softness="hard",
            electronegativity=3.98,
        ),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=3.52,
            dehydration_energy_kj_mol=505.0,
        ),
        redox_states=[RedoxState(-1, "F⁻")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.33, molecular_weight=19.0),
        notes="Dental/skeletal fluorosis at >1.5 mg/L. CaF₂ precipitation is spontaneous.",
    ),

    "sulfur dioxide": TargetSpecies(
        identity="sulfur dioxide", formula="SO2", charge=0.0,
        geometry="bent",
        electronic=ElectronicDescription(
            hardness_softness="borderline",
            donor_atoms=["O", "S"],
        ),
        hydration=HydrationDescription(hydrated_radius_angstrom=2.0),
        redox_states=[
            RedoxState(4, "SO₂"),
            RedoxState(6, "SO₄²⁻"),
            RedoxState(-2, "S²⁻"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(molecular_weight=64.07),
        notes="Flue gas pollutant. CaO/MgO surface → gypsum spontaneously.",
    ),

    "arsenic": TargetSpecies(
        identity="arsenic", formula="H2AsO4(-)", charge=-1.0,
        geometry="tetrahedral",
        electronic=ElectronicDescription(
            hardness_softness="hard",
            donor_atoms=["O"],
        ),
        hydration=HydrationDescription(hydrated_radius_angstrom=3.5),
        redox_states=[
            RedoxState(5, "H₂AsO₄⁻ / HAsO₄²⁻"),
            RedoxState(3, "H₃AsO₃"),
            RedoxState(0, "As(0)"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=2.48, molecular_weight=141.94),
        notes="Groundwater toxin. Fe(III) surface → scorodite (FeAsO₄·2H₂O) spontaneously.",
    ),

    "cadmium": TargetSpecies(
        identity="cadmium", formula="Cd(2+)", charge=2.0,
        geometry="octahedral (preferred), tetrahedral",
        electronic=ElectronicDescription(
            hardness_softness="soft",
            electronegativity=1.69,
            donor_atoms=["S", "N", "O"],
        ),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=4.26,
            dehydration_energy_kj_mol=1807.0,
            coordination_number_water=6,
        ),
        redox_states=[RedoxState(2, "Cd²⁺"), RedoxState(0, "Cd(0)")],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=0.95, molecular_weight=112.41),
        notes="Battery/mining contaminant. CdS (Ksp=10⁻²⁷) = semiconductor precursor.",
    ),

    "mercury": TargetSpecies(
        identity="mercury", formula="Hg(2+)", charge=2.0,
        geometry="linear (preferred), tetrahedral",
        electronic=ElectronicDescription(
            hardness_softness="soft",
            electronegativity=2.00,
            donor_atoms=["S", "Se"],
        ),
        hydration=HydrationDescription(
            hydrated_radius_angstrom=4.10,
            dehydration_energy_kj_mol=1824.0,
        ),
        redox_states=[
            RedoxState(2, "Hg²⁺"),
            RedoxState(1, "Hg₂²⁺", notes="mercurous dimer"),
            RedoxState(0, "Hg(0)", notes="liquid metal"),
        ],
        magnetic=MagneticDescription(type="diamagnetic"),
        size=SizeDescription(ionic_radius_angstrom=1.02, molecular_weight=200.59),
        notes="Extreme toxin. HgS (Ksp=10⁻⁵²) = permanent sequestration.",
    ),
}

# Fill alias
CAPTURE_TRANSFORM_TARGETS["co2"] = CAPTURE_TRANSFORM_TARGETS["carbon dioxide"]


# ═══════════════════════════════════════════════════════════════════════════
# Capture-Transform Matrices
# ═══════════════════════════════════════════════════════════════════════════

CAPTURE_TRANSFORM_MATRICES = {
    "wastewater": Matrix(
        description="Municipal/agricultural wastewater",
        ph=7.5, temperature_c=20.0, ionic_strength_mm=20.0,
        competing_species=[
            CompetingSpecies("ammonium", "NH4(+)", 5.0, 1.0),
            CompetingSpecies("phosphate", "PO4(3-)", 0.5, -3.0),
            CompetingSpecies("calcium", "Ca(2+)", 3.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 2.0, 2.0),
            CompetingSpecies("sulfate", "SO4(2-)", 50.0, -2.0),
        ],
    ),
    "agricultural": None,  # alias
    "seawater": Matrix(
        description="Seawater",
        ph=8.1, temperature_c=18.0, ionic_strength_mm=700.0,
        competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 10.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 53.0, 2.0),
            CompetingSpecies("sodium", "Na(+)", 468.0, 1.0),
            CompetingSpecies("chloride", "Cl(-)", 546.0, -1.0),
            CompetingSpecies("sulfate", "SO4(2-)", 28.0, -2.0),
        ],
    ),
    "hard water": Matrix(
        description="Hard water (limestone aquifer)",
        ph=7.8, temperature_c=15.0, ionic_strength_mm=10.0,
        competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 2.5, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 1.5, 2.0),
            CompetingSpecies("bicarbonate", "HCO3(-)", 3.0, -1.0),
        ],
    ),
    "groundwater": Matrix(
        description="Contaminated groundwater",
        ph=6.8, temperature_c=12.0, ionic_strength_mm=5.0,
        competing_species=[
            CompetingSpecies("calcium", "Ca(2+)", 1.5, 2.0),
            CompetingSpecies("iron", "Fe(3+)", 0.5, 3.0),
        ],
    ),
    "flue gas": Matrix(
        description="Coal/gas flue gas scrubbing water",
        ph=5.5, temperature_c=50.0, ionic_strength_mm=30.0,
        competing_species=[
            CompetingSpecies("sulfate", "SO4(2-)", 100.0, -2.0),
            CompetingSpecies("chloride", "Cl(-)", 20.0, -1.0),
        ],
    ),
    "air": Matrix(
        description="Ambient air (direct air capture)",
        ph=7.0, temperature_c=20.0, ionic_strength_mm=0.0,
        notes="CO₂ at ~420 ppm. Humidity-dependent. No competing ions.",
    ),
    "tailings": Matrix(
        description="Mine tailings pond water",
        ph=4.5, temperature_c=15.0, ionic_strength_mm=100.0,
        competing_species=[
            CompetingSpecies("sulfate", "SO4(2-)", 500.0, -2.0),
            CompetingSpecies("calcium", "Ca(2+)", 200.0, 2.0),
            CompetingSpecies("magnesium", "Mg(2+)", 100.0, 2.0),
            CompetingSpecies("iron", "Fe(3+)", 50.0, 3.0),
            CompetingSpecies("manganese", "Mn(2+)", 10.0, 2.0),
        ],
    ),
}

# Alias
CAPTURE_TRANSFORM_MATRICES["agricultural"] = CAPTURE_TRANSFORM_MATRICES["wastewater"]


# ═══════════════════════════════════════════════════════════════════════════
# Extension function
# ═══════════════════════════════════════════════════════════════════════════

def extend_decomposer():
    """Add capture-transform targets and matrices to the decomposer.

    Imports and modifies the decomposer's KNOWN_TARGETS and KNOWN_MATRICES
    dictionaries in-place. Safe to call multiple times (idempotent).
    """
    from conversation.decomposer import KNOWN_TARGETS, KNOWN_MATRICES

    for key, target in CAPTURE_TRANSFORM_TARGETS.items():
        if target is not None and key not in KNOWN_TARGETS:
            KNOWN_TARGETS[key] = target

    for key, matrix in CAPTURE_TRANSFORM_MATRICES.items():
        if matrix is not None and key not in KNOWN_MATRICES:
            KNOWN_MATRICES[key] = matrix


# ═══════════════════════════════════════════════════════════════════════════
# Adapter registration
# ═══════════════════════════════════════════════════════════════════════════

def register_capture_transform(registry=None):
    """Register the CaptureTransformAdapter with a ToolRegistry.

    If no registry provided, uses the global registry from adapters.base.
    """
    if registry is None:
        from adapters.base import registry as global_registry
        registry = global_registry

    from adapters.capture_transform_adapter import CaptureTransformAdapter
    adapter = CaptureTransformAdapter()
    registry.register(adapter)
    return adapter


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: do both
# ═══════════════════════════════════════════════════════════════════════════

def setup_capture_transform(registry=None):
    """One-call setup: extend decomposer + register adapter."""
    extend_decomposer()
    return register_capture_transform(registry)
