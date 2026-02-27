"""
Porphyrin & Phthalocyanine Knowledge Base.

Metalloporphyrin properties from CCDC structures and Kadish/Smith/Guilard
Handbook of Porphyrin Science. Substituent effects from Hammett correlations.
Metal-N bond distances from crystallographic averages.

Data sources:
    - Kadish KM, Smith KM, Guilard R (2010) Handbook of Porphyrin Science
    - Fleischer EB (1970) Acc. Chem. Res. — metalloporphyrin structures
    - Walker FA, Simonis U (2006) Encyclopedia of Inorganic Chemistry
    - BackSolve Phase 1-5 metal coordination calibration (R²=0.908)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


# ─────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PorphyrinCore:
    """A porphyrin macrocycle variant."""

    name: str
    abbreviation: str
    core_type: str                     # "porphyrin", "phthalocyanine", "corrole", "porphyrazine"

    # ── Cavity geometry (crystallographic) ──
    n_pyrrole_N: int                   # 4 for porphyrin, 4 for Pc, 3 for corrole
    core_hole_radius_A: float          # center-to-N distance
    ideal_MN_bond_A: float             # unstrained M-N distance
    cavity_symmetry: str               # "D4h", "C4v", "C2v"

    # ── Electronics ──
    base_pKa1: float                   # first protonation of free base
    base_pKa2: float                   # second protonation
    electron_richness: str             # "electron-rich", "neutral", "electron-poor"

    # ── Production ──
    commercial: bool
    cost_per_gram_usd: float
    common_suppliers: list[str]
    synthesis_route: str               # "Lindsey", "Adler-Longo", "Rothemund"


@dataclass(frozen=True)
class MetalPorphyrinEntry:
    """Known metal-porphyrin binding data."""

    metal: str
    charge: int
    ionic_radius_A: float              # Shannon radius, CN=4 where available
    preferred_coordination: int        # 4, 5, or 6
    spin_state: str                    # "low-spin", "high-spin", "diamagnetic"

    # ── Structural data ──
    typical_MN_bond_A: float           # crystallographic average
    in_plane: bool                     # True if metal sits in porphyrin plane
    displacement_A: float              # out-of-plane displacement if not in-plane
    dome_distortion: bool              # porphyrin ring domes around large metal

    # ── Thermodynamic ──
    logK_metalation: float             # equilibrium constant for metalation
    dG_metalation_kJ_mol: float
    kinetic_class: str                 # "labile", "intermediate", "inert"

    # ── Axial ligand preference ──
    axial_ligands_typical: list[str]   # common 5th/6th ligands
    max_axial_count: int               # 0, 1, or 2

    # ── LFSE contribution ──
    d_electrons: int
    lfse_kJ_mol: float                 # ligand field stabilization in porph


@dataclass(frozen=True)
class MesoSubstituent:
    """Meso-position substituent and its electronic/steric effect."""

    name: str
    abbreviation: str
    hammett_sigma: float               # Hammett σ_para — electron effect
    steric_A_value: float              # A-value (kcal/mol), steric bulk
    effect_on_metalation: str          # "accelerates", "retards", "neutral"
    effect_on_redox: str               # "lowers E1/2", "raises E1/2", "neutral"
    solubility_effect: str             # "hydrophilic", "hydrophobic", "neutral"
    commercial: bool
    conjugation_handle: bool           # can attach to scaffold


@dataclass(frozen=True)
class AxialLigand:
    """Axial ligand for 5- or 6-coordinate metalloporphyrins."""

    name: str
    abbreviation: str
    donor_atom: str                    # "N", "O", "S"
    donor_subtype: str                 # "N_imine", "N_amine", "O_water", etc.
    binding_strength: str              # "strong", "moderate", "weak"
    typical_logK_axial: float          # axial binding constant
    trans_influence: str               # "strong", "moderate", "weak"
    labile: bool                       # easily displaced


# ─────────────────────────────────────────────
# Porphyrin cores
# ─────────────────────────────────────────────

TPP = PorphyrinCore(
    name="meso-Tetraphenylporphyrin",
    abbreviation="TPP",
    core_type="porphyrin",
    n_pyrrole_N=4,
    core_hole_radius_A=2.01,
    ideal_MN_bond_A=2.01,
    cavity_symmetry="D4h",
    base_pKa1=4.2,
    base_pKa2=2.0,
    electron_richness="neutral",
    commercial=True,
    cost_per_gram_usd=5.00,
    common_suppliers=["Sigma-Aldrich", "TCI", "Frontier Scientific"],
    synthesis_route="Adler-Longo (pyrrole + benzaldehyde, propionic acid reflux)",
)

OEP = PorphyrinCore(
    name="Octaethylporphyrin",
    abbreviation="OEP",
    core_type="porphyrin",
    n_pyrrole_N=4,
    core_hole_radius_A=2.03,
    ideal_MN_bond_A=2.03,
    cavity_symmetry="D4h",
    base_pKa1=5.0,
    base_pKa2=3.5,
    electron_richness="electron-rich",
    commercial=True,
    cost_per_gram_usd=15.00,
    common_suppliers=["Sigma-Aldrich", "Frontier Scientific"],
    synthesis_route="Fischer (monopyrrole condensation)",
)

TPFPP = PorphyrinCore(
    name="meso-Tetrakis(pentafluorophenyl)porphyrin",
    abbreviation="TPFPP",
    core_type="porphyrin",
    n_pyrrole_N=4,
    core_hole_radius_A=1.99,
    ideal_MN_bond_A=1.99,
    cavity_symmetry="D4h",
    base_pKa1=2.0,
    base_pKa2=0.5,
    electron_richness="electron-poor",
    commercial=True,
    cost_per_gram_usd=25.00,
    common_suppliers=["Sigma-Aldrich", "Frontier Scientific"],
    synthesis_route="Lindsey (BF3·Et2O catalyst, DDQ oxidation)",
)

TCPP = PorphyrinCore(
    name="meso-Tetrakis(4-carboxyphenyl)porphyrin",
    abbreviation="TCPP",
    core_type="porphyrin",
    n_pyrrole_N=4,
    core_hole_radius_A=2.01,
    ideal_MN_bond_A=2.01,
    cavity_symmetry="D4h",
    base_pKa1=4.0,
    base_pKa2=1.8,
    electron_richness="neutral",
    commercial=True,
    cost_per_gram_usd=20.00,
    common_suppliers=["Sigma-Aldrich", "TCI", "Frontier Scientific"],
    synthesis_route="Adler-Longo (pyrrole + 4-carboxybenzaldehyde)",
)

TAPP = PorphyrinCore(
    name="meso-Tetrakis(4-aminophenyl)porphyrin",
    abbreviation="TAPP",
    core_type="porphyrin",
    n_pyrrole_N=4,
    core_hole_radius_A=2.02,
    ideal_MN_bond_A=2.02,
    cavity_symmetry="D4h",
    base_pKa1=5.5,
    base_pKa2=3.0,
    electron_richness="electron-rich",
    commercial=True,
    cost_per_gram_usd=30.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    synthesis_route="Adler-Longo then nitro reduction (SnCl2/HCl)",
)

PC = PorphyrinCore(
    name="Phthalocyanine",
    abbreviation="Pc",
    core_type="phthalocyanine",
    n_pyrrole_N=4,
    core_hole_radius_A=1.92,  # slightly smaller cavity than porphyrin
    ideal_MN_bond_A=1.92,
    cavity_symmetry="D4h",
    base_pKa1=3.0,
    base_pKa2=1.0,
    electron_richness="electron-rich",
    commercial=True,
    cost_per_gram_usd=8.00,
    common_suppliers=["Sigma-Aldrich", "TCI"],
    synthesis_route="Phthalonitrile cyclotetramerization (DBU, high-T)",
)


ALL_CORES: dict[str, PorphyrinCore] = {
    "TPP": TPP,
    "OEP": OEP,
    "TPFPP": TPFPP,
    "TCPP": TCPP,
    "TAPP": TAPP,
    "Pc": PC,
}


# ─────────────────────────────────────────────
# Metal-porphyrin database
# ─────────────────────────────────────────────

METAL_PORPH_DB: dict[str, MetalPorphyrinEntry] = {
    "Cu2+": MetalPorphyrinEntry(
        metal="Cu2+", charge=2, ionic_radius_A=0.57,
        preferred_coordination=4, spin_state="low-spin",
        typical_MN_bond_A=1.98, in_plane=True, displacement_A=0.0,
        dome_distortion=False,
        logK_metalation=18.0, dG_metalation_kJ_mol=-102.7,
        kinetic_class="labile",
        axial_ligands_typical=["pyridine"], max_axial_count=1,
        d_electrons=9, lfse_kJ_mol=-90.0,
    ),
    "Zn2+": MetalPorphyrinEntry(
        metal="Zn2+", charge=2, ionic_radius_A=0.60,
        preferred_coordination=5, spin_state="diamagnetic",
        typical_MN_bond_A=2.04, in_plane=True, displacement_A=0.05,
        dome_distortion=False,
        logK_metalation=14.5, dG_metalation_kJ_mol=-82.7,
        kinetic_class="labile",
        axial_ligands_typical=["pyridine", "imidazole", "water"],
        max_axial_count=1,
        d_electrons=10, lfse_kJ_mol=0.0,
    ),
    "Fe2+": MetalPorphyrinEntry(
        metal="Fe2+", charge=2, ionic_radius_A=0.61,
        preferred_coordination=6, spin_state="high-spin",
        typical_MN_bond_A=2.07, in_plane=False, displacement_A=0.30,
        dome_distortion=True,
        logK_metalation=12.0, dG_metalation_kJ_mol=-68.5,
        kinetic_class="intermediate",
        axial_ligands_typical=["imidazole", "pyridine", "CO", "O2", "NO"],
        max_axial_count=2,
        d_electrons=6, lfse_kJ_mol=-50.0,
    ),
    "Fe3+": MetalPorphyrinEntry(
        metal="Fe3+", charge=3, ionic_radius_A=0.55,
        preferred_coordination=6, spin_state="high-spin",
        typical_MN_bond_A=2.04, in_plane=True, displacement_A=0.10,
        dome_distortion=False,
        logK_metalation=22.0, dG_metalation_kJ_mol=-125.5,
        kinetic_class="inert",
        axial_ligands_typical=["Cl-", "OH-", "imidazole", "pyridine"],
        max_axial_count=2,
        d_electrons=5, lfse_kJ_mol=0.0,  # d5 HS: zero LFSE
    ),
    "Mn2+": MetalPorphyrinEntry(
        metal="Mn2+", charge=2, ionic_radius_A=0.66,
        preferred_coordination=6, spin_state="high-spin",
        typical_MN_bond_A=2.13, in_plane=False, displacement_A=0.25,
        dome_distortion=True,
        logK_metalation=8.0, dG_metalation_kJ_mol=-45.6,
        kinetic_class="labile",
        axial_ligands_typical=["Cl-", "water", "pyridine"],
        max_axial_count=2,
        d_electrons=5, lfse_kJ_mol=0.0,
    ),
    "Mn3+": MetalPorphyrinEntry(
        metal="Mn3+", charge=3, ionic_radius_A=0.58,
        preferred_coordination=6, spin_state="high-spin",
        typical_MN_bond_A=2.01, in_plane=True, displacement_A=0.05,
        dome_distortion=False,
        logK_metalation=20.0, dG_metalation_kJ_mol=-114.1,
        kinetic_class="intermediate",
        axial_ligands_typical=["Cl-", "OAc-", "pyridine"],
        max_axial_count=2,
        d_electrons=4, lfse_kJ_mol=-60.0,
    ),
    "Co2+": MetalPorphyrinEntry(
        metal="Co2+", charge=2, ionic_radius_A=0.58,
        preferred_coordination=5, spin_state="low-spin",
        typical_MN_bond_A=1.95, in_plane=True, displacement_A=0.0,
        dome_distortion=False,
        logK_metalation=16.0, dG_metalation_kJ_mol=-91.3,
        kinetic_class="intermediate",
        axial_ligands_typical=["pyridine", "imidazole", "CN-"],
        max_axial_count=1,
        d_electrons=7, lfse_kJ_mol=-80.0,
    ),
    "Co3+": MetalPorphyrinEntry(
        metal="Co3+", charge=3, ionic_radius_A=0.55,
        preferred_coordination=6, spin_state="low-spin",
        typical_MN_bond_A=1.92, in_plane=True, displacement_A=0.0,
        dome_distortion=False,
        logK_metalation=25.0, dG_metalation_kJ_mol=-142.6,
        kinetic_class="inert",
        axial_ligands_typical=["CN-", "imidazole", "pyridine"],
        max_axial_count=2,
        d_electrons=6, lfse_kJ_mol=-120.0,
    ),
    "Ni2+": MetalPorphyrinEntry(
        metal="Ni2+", charge=2, ionic_radius_A=0.55,
        preferred_coordination=4, spin_state="low-spin",
        typical_MN_bond_A=1.93, in_plane=True, displacement_A=0.0,
        dome_distortion=False,
        logK_metalation=16.5, dG_metalation_kJ_mol=-94.1,
        kinetic_class="intermediate",
        axial_ligands_typical=[], max_axial_count=0,
        d_electrons=8, lfse_kJ_mol=-100.0,
    ),
    "Pd2+": MetalPorphyrinEntry(
        metal="Pd2+", charge=2, ionic_radius_A=0.64,
        preferred_coordination=4, spin_state="low-spin",
        typical_MN_bond_A=2.01, in_plane=True, displacement_A=0.0,
        dome_distortion=False,
        logK_metalation=24.0, dG_metalation_kJ_mol=-136.9,
        kinetic_class="inert",
        axial_ligands_typical=[], max_axial_count=0,
        d_electrons=8, lfse_kJ_mol=-150.0,
    ),
    "Pb2+": MetalPorphyrinEntry(
        metal="Pb2+", charge=2, ionic_radius_A=0.98,
        preferred_coordination=4, spin_state="diamagnetic",
        typical_MN_bond_A=2.34, in_plane=False, displacement_A=1.00,
        dome_distortion=True,
        logK_metalation=6.0, dG_metalation_kJ_mol=-34.2,
        kinetic_class="labile",
        axial_ligands_typical=[], max_axial_count=0,
        d_electrons=0, lfse_kJ_mol=0.0,
    ),
    "Mg2+": MetalPorphyrinEntry(
        metal="Mg2+", charge=2, ionic_radius_A=0.57,
        preferred_coordination=6, spin_state="diamagnetic",
        typical_MN_bond_A=2.07, in_plane=True, displacement_A=0.0,
        dome_distortion=False,
        logK_metalation=10.0, dG_metalation_kJ_mol=-57.1,
        kinetic_class="labile",
        axial_ligands_typical=["water", "pyridine"],
        max_axial_count=2,
        d_electrons=0, lfse_kJ_mol=0.0,
    ),
}


# ─────────────────────────────────────────────
# Meso-substituent library
# ─────────────────────────────────────────────

MESO_SUBSTITUENTS: dict[str, MesoSubstituent] = {
    "phenyl": MesoSubstituent(
        name="Phenyl", abbreviation="Ph",
        hammett_sigma=0.0, steric_A_value=3.0,
        effect_on_metalation="neutral", effect_on_redox="neutral",
        solubility_effect="hydrophobic",
        commercial=True, conjugation_handle=False,
    ),
    "4-carboxyphenyl": MesoSubstituent(
        name="4-Carboxyphenyl", abbreviation="4-COOH-Ph",
        hammett_sigma=0.45, steric_A_value=3.0,
        effect_on_metalation="retards", effect_on_redox="raises E1/2",
        solubility_effect="hydrophilic",
        commercial=True, conjugation_handle=True,
    ),
    "4-aminophenyl": MesoSubstituent(
        name="4-Aminophenyl", abbreviation="4-NH2-Ph",
        hammett_sigma=-0.66, steric_A_value=3.0,
        effect_on_metalation="accelerates", effect_on_redox="lowers E1/2",
        solubility_effect="hydrophilic",
        commercial=True, conjugation_handle=True,
    ),
    "pentafluorophenyl": MesoSubstituent(
        name="Pentafluorophenyl", abbreviation="C6F5",
        hammett_sigma=0.27, steric_A_value=3.5,
        effect_on_metalation="retards", effect_on_redox="raises E1/2",
        solubility_effect="hydrophobic",
        commercial=True, conjugation_handle=False,
    ),
    "4-pyridyl": MesoSubstituent(
        name="4-Pyridyl", abbreviation="4-Py",
        hammett_sigma=0.44, steric_A_value=2.8,
        effect_on_metalation="retards", effect_on_redox="raises E1/2",
        solubility_effect="hydrophilic",
        commercial=True, conjugation_handle=True,
    ),
    "mesityl": MesoSubstituent(
        name="Mesityl (2,4,6-trimethylphenyl)", abbreviation="Mes",
        hammett_sigma=-0.07, steric_A_value=5.0,
        effect_on_metalation="neutral", effect_on_redox="neutral",
        solubility_effect="hydrophobic",
        commercial=True, conjugation_handle=False,
    ),
    "4-hydroxyphenyl": MesoSubstituent(
        name="4-Hydroxyphenyl", abbreviation="4-OH-Ph",
        hammett_sigma=-0.37, steric_A_value=3.0,
        effect_on_metalation="accelerates", effect_on_redox="lowers E1/2",
        solubility_effect="hydrophilic",
        commercial=True, conjugation_handle=True,
    ),
}


# ─────────────────────────────────────────────
# Axial ligand library
# ─────────────────────────────────────────────

AXIAL_LIGANDS: dict[str, AxialLigand] = {
    "imidazole": AxialLigand(
        name="Imidazole", abbreviation="Im",
        donor_atom="N", donor_subtype="N_imine",
        binding_strength="strong", typical_logK_axial=3.5,
        trans_influence="moderate", labile=False,
    ),
    "pyridine": AxialLigand(
        name="Pyridine", abbreviation="Py",
        donor_atom="N", donor_subtype="N_imine",
        binding_strength="moderate", typical_logK_axial=2.8,
        trans_influence="moderate", labile=True,
    ),
    "chloride": AxialLigand(
        name="Chloride", abbreviation="Cl-",
        donor_atom="Cl", donor_subtype="halide",
        binding_strength="moderate", typical_logK_axial=2.5,
        trans_influence="weak", labile=True,
    ),
    "hydroxide": AxialLigand(
        name="Hydroxide", abbreviation="OH-",
        donor_atom="O", donor_subtype="O_hydroxide",
        binding_strength="moderate", typical_logK_axial=3.0,
        trans_influence="moderate", labile=True,
    ),
    "water": AxialLigand(
        name="Water", abbreviation="H2O",
        donor_atom="O", donor_subtype="O_water",
        binding_strength="weak", typical_logK_axial=0.5,
        trans_influence="weak", labile=True,
    ),
    "cyanide": AxialLigand(
        name="Cyanide", abbreviation="CN-",
        donor_atom="C", donor_subtype="C_cyanide",
        binding_strength="strong", typical_logK_axial=5.0,
        trans_influence="strong", labile=False,
    ),
    "CO": AxialLigand(
        name="Carbon monoxide", abbreviation="CO",
        donor_atom="C", donor_subtype="C_carbonyl",
        binding_strength="strong", typical_logK_axial=4.5,
        trans_influence="strong", labile=False,
    ),
    "thiolate": AxialLigand(
        name="Thiolate", abbreviation="RS-",
        donor_atom="S", donor_subtype="S_thiolate",
        binding_strength="strong", typical_logK_axial=4.0,
        trans_influence="strong", labile=False,
    ),
}


# ─────────────────────────────────────────────
# BackSolve calibrated parameters for metal-porphyrin
# ─────────────────────────────────────────────

BACKSOLVE_METAL_PARAMS = {
    # From Phase 1-5 joint calibration (R²=0.908, MAE=2.35)
    "N_pyrrole_exchange_kJ_mol": -45.0,    # per M-N_pyrrole bond
    "lfse_scale": 0.85,                     # scaling of free-ion LFSE
    "macrocyclic_stab_per_N": 3.5,          # kJ/mol per donor (preorganization)
    "size_match_sigma_A": 0.15,             # Gaussian width for M-N match
    "size_match_penalty_kJ": 25.0,          # max penalty for mismatch
    "axial_base_kJ": -15.0,                # per axial ligand
    "hammett_rho": 8.0,                    # ρ for metalation ΔG (kJ/mol per σ): +σ → +ΔG (less favorable)
}


# ─────────────────────────────────────────────
# Physics functions
# ─────────────────────────────────────────────

def metal_porphyrin_size_match(
    metal_ionic_radius_A: float,
    core_hole_radius_A: float,
) -> float:
    """
    Size-match score for metal in porphyrin core hole.

    Porphyrin core hole is ~2.01 Å (center-to-N).
    Ideal M-N bond ≈ core_hole_radius.
    Returns 0.0–1.0.
    """
    sigma = BACKSOLVE_METAL_PARAMS["size_match_sigma_A"]
    # Compare M-N bond (≈ ionic_radius + ~1.40 Å for N) to core hole
    # Simplified: direct comparison of ionic radius to ideal
    ideal_radius = core_hole_radius_A - 1.40  # N covalent radius ~1.40 for M-N
    delta = metal_ionic_radius_A - ideal_radius
    return math.exp(-0.5 * (delta / sigma) ** 2)


def predict_metalation_dG(
    metal: MetalPorphyrinEntry,
    core: PorphyrinCore,
    meso_sigma: float = 0.0,
    n_axial: int = 0,
) -> float:
    """
    Predict metalation ΔG from BackSolve parameters.

    Terms:
        1. M-N exchange (4 bonds × per-bond energy)
        2. LFSE contribution (scaled)
        3. Macrocyclic stabilization
        4. Size-match penalty
        5. Substituent effect (Hammett)
        6. Axial ligand stabilization
    """
    params = BACKSOLVE_METAL_PARAMS

    # M-N exchange
    dG_exchange = core.n_pyrrole_N * params["N_pyrrole_exchange_kJ_mol"]

    # LFSE
    dG_lfse = metal.lfse_kJ_mol * params["lfse_scale"]

    # Macrocyclic
    dG_macro = -core.n_pyrrole_N * params["macrocyclic_stab_per_N"]

    # Size match penalty
    sm = metal_porphyrin_size_match(metal.ionic_radius_A, core.core_hole_radius_A)
    dG_size = params["size_match_penalty_kJ"] * (1.0 - sm)

    # Hammett substituent effect
    dG_hammett = params["hammett_rho"] * meso_sigma

    # Axial
    dG_axial = n_axial * params["axial_base_kJ"]

    return dG_exchange + dG_lfse + dG_macro + dG_size + dG_hammett + dG_axial


def select_core_for_metal(metal_symbol: str) -> list[tuple[PorphyrinCore, float, str]]:
    """
    Rank porphyrin cores for a target metal.

    Returns (core, size_match_score, rationale) sorted by suitability.
    """
    if metal_symbol not in METAL_PORPH_DB:
        return []

    metal = METAL_PORPH_DB[metal_symbol]
    results = []

    for core in ALL_CORES.values():
        sm = metal_porphyrin_size_match(metal.ionic_radius_A, core.core_hole_radius_A)

        parts = [f"size-match={sm:.2f}"]

        # Electron-poor cores stabilize electron-rich metals (high d-count)
        if metal.d_electrons >= 8 and core.electron_richness == "electron-poor":
            parts.append("electron-poor core matches d8+ metal")
        if metal.d_electrons <= 4 and core.electron_richness == "electron-rich":
            parts.append("electron-rich core stabilizes low-d metal")

        # Phthalocyanine has smaller hole — better for small metals
        if core.core_type == "phthalocyanine" and metal.ionic_radius_A < 0.58:
            parts.append("Pc smaller cavity suits small ion")

        results.append((core, sm, "; ".join(parts)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
