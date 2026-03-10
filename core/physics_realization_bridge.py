"""
core/physics_realization_bridge.py — F3: Physics → Realization Bridge

Connects the calibrated physics engine (unified_scorer_v2) with the
realization engine (13 material system adapters + ranker).

Gap closed:
    Before F3: physics predicts binding energy, realization ranks materials,
    but nothing connects them. The realization engine uses heuristic scores
    while calibrated physics predictions sit unused.

    After F3: every material system's design is scored by the *calibrated*
    physics engine. The bridge converts adapter designs → UniversalComplex
    → unified_scorer_v2.predict() → calibrated log K.

Entry point:
    end_to_end_design(target, conditions, competitors, application)
    → DesignResult with ≥3 ranked material systems, each with calibrated
      log K and selectivity ratios.

Does NOT:
    - Add new physics parameters
    - Add new calibration data
    - Modify unified_scorer_v2
    - Modify existing realization adapters
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
for _sub in ('knowledge', 'core'):
    _p = os.path.join(_project_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.universal_schema import UniversalComplex
from core.unified_scorer_v2 import predict as unified_predict, PredictionResult
from core.scorer_frozen import METAL_DB


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialDesignScore:
    """Physics-calibrated score for one material system's design."""

    material_system: str
    adapter_id: str

    # Physics predictions (from unified_scorer_v2)
    predicted_log_k: float
    prediction_result: Optional[PredictionResult] = None

    # Selectivity (physics-derived)
    selectivity_ratios: dict = field(default_factory=dict)
    # {competitor_formula: ratio} where ratio = 10^(log_K_target - log_K_competitor)

    # Realization metadata
    realization_feasible: bool = True
    infeasibility_reason: str = ""
    physics_fidelity: float = 0.0  # from realization deviation scoring

    # Implementation scores (from realization engine)
    synthetic_accessibility: float = 0.0
    cost_score: float = 0.0
    scalability_score: float = 0.0

    # Composite
    composite_score: float = 0.0

    @property
    def is_selective(self) -> bool:
        """True if target binds stronger than all competitors."""
        return all(r > 1.0 for r in self.selectivity_ratios.values())

    @property
    def worst_selectivity(self) -> float:
        """Smallest selectivity ratio (worst case)."""
        if not self.selectivity_ratios:
            return float('inf')
        return min(self.selectivity_ratios.values())


@dataclass
class DesignResult:
    """End-to-end design output with ranked material systems."""

    target: str  # e.g. "Pb2+"
    conditions: dict = field(default_factory=dict)
    competitors: list = field(default_factory=list)
    application: str = "research"

    # Ranked material designs (best first)
    ranked_designs: list = field(default_factory=list)  # list[MaterialDesignScore]

    # Pipeline metadata
    n_materials_evaluated: int = 0
    n_materials_feasible: int = 0
    pipeline_complete: bool = False

    @property
    def best_design(self) -> Optional[MaterialDesignScore]:
        feasible = [d for d in self.ranked_designs if d.realization_feasible]
        return feasible[0] if feasible else None


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER DESIGN → UNIVERSAL COMPLEX CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

# Default donor subtypes by material system and donor atom.
# These are the canonical donor subtypes each material presents.
# Expanded for non-carbon-bias full-periodic-table coverage.
_MATERIAL_DONOR_DEFAULTS = {
    "planar_coordination_ring": {
        "N": "N_imine",
        "O": "O_phenolate",
        "S": "S_thioether",
        "Se": "Se_selenoether",
        "P": "P_phosphine",
    },
    "cyclic_encapsulant": {
        "N": "N_amine",
        "O": "O_carbonyl",
        "S": "S_thioether",
        "Se": "Se_selenoether",
    },
    "periodic_lattice_node": {
        "N": "N_aromatic",
        "O": "O_carboxylate",
        "S": "S_thiolate",
        "Se": "Se_selenolate",
        "P": "P_phosphonate",
    },
    "folded_polypeptide": {
        "N": "N_amide",
        "O": "O_carboxylate",
        "S": "S_thiolate",
    },
    "emergent_coordination_cage": {
        "N": "N_aromatic",
        "O": "O_carboxylate",
        "S": "S_thioether",
        "Se": "Se_selenoether",
        "P": "P_phosphine",
    },
    # Crown ether variants — O_carbonyl captures preorganized lone-pair
    # donor strength better than O_ether (calibrated on simple ethers)
    "crown_ether": {
        "N": "N_amine",
        "O": "O_carbonyl",
        "S": "S_thioether",
        "Se": "Se_selenoether",  # Selenacrown ethers — soft metal selectivity
    },
    "cryptand": {
        "N": "N_amine",
        "O": "O_carbonyl",
        "S": "S_thioether",
        "Se": "Se_selenoether",
        "P": "P_phosphine",
    },
    # Other adapter types
    "porphyrin": {
        "N": "N_pyrrole",
        "O": "O_carboxylate",
        "S": "S_thiolate",
    },
    "cyclodextrin": {
        "N": "N_amine",
        "O": "O_hydroxyl",
        "S": "S_thioether",
    },
    "lignin": {
        "N": "N_amine",
        "O": "O_phenolate",
        "S": "S_thiolate",
    },
    # ── Non-carbon-bias material systems ──────────────────────────────────
    # Chalcogenide frameworks: MOFs / cages built from Se/Te linkers
    # Relevant for Hg2+, Ag+, Au+, Pd2+ remediation / precious metal recovery
    "chalcogenide_mof": {
        "S": "S_thiolate",
        "Se": "Se_selenolate",
        "Te": "Te_tellurolate",
        "N": "N_aromatic",
        "O": "O_carboxylate",
    },
    # Phosphine-functionalized surfaces / resins (gold/platinum capture)
    "phosphine_resin": {
        "P": "P_phosphine",
        "S": "S_thioether",
        "N": "N_amine",
        "O": "O_hydroxyl",
    },
    # Arsenic/antimony coordination cages (rare; mainly Sb2S3-type materials)
    "pnictogen_cage": {
        "As": "As_arsine",
        "Sb": "Sb_stibine",
        "S": "S_thiolate",
        "Se": "Se_selenolate",
    },
    # Fluoride-selective hosts (Al3+, Th4+, Zr4+ chemistry)
    "fluoride_host": {
        "F": "F_fluoride",
        "O": "O_carboxylate",
        "N": "N_amine",
    },
    # Carbonyl / cyanide-presenting frameworks (Fe/Ni/Ru capture)
    "carbonyl_framework": {
        "C": "C_carbonyl",
        "N": "N_aromatic",
        "O": "O_carboxylate",
    },
    "cyanide_bridged_framework": {
        "C": "C_cyanide",
        "N": "N_imine",
        "Fe": "C_cyanide",  # Prussian blue type: C-end to Fe2+
    },
    # Zeolite / inorganic oxide surfaces (Al3+, Si4+ framework, O donors only)
    "zeolite_framework": {
        "O": "O_oxo",
        "N": "N_amine",
        "F": "F_fluoride",
    },
    # Silica / oxide surface (surface silanol groups)
    "silica_surface": {
        "O": "O_hydroxyl",
        "N": "N_amine",
        "S": "S_thiolate",
        "P": "P_phosphonate",
    },
}


def _resolve_donor_subtypes(
    donor_atoms: list,
    material_system: str,
) -> list:
    """Map raw donor atoms (N, O, S, Se, Te, P, As, F, C...) to calibrated subtypes."""
    defaults = _MATERIAL_DONOR_DEFAULTS.get(material_system, {})
    fallback = {
        "N": "N_amine",
        "O": "O_carboxylate",
        "S": "S_thioether",
        "Se": "Se_selenoether",
        "Te": "Te_telluroether",
        "P": "P_phosphine",
        "As": "As_arsine",
        "Sb": "Sb_stibine",
        "F": "F_fluoride",
        "C": "C_cyanide",
        "Cl": "Cl_chloride",
        "Br": "Br_bromide",
        "I": "I_iodide",
    }
    subtypes = []
    for atom in donor_atoms:
        if '_' in str(atom):
            # Already a subtype
            subtypes.append(atom)
        else:
            subtypes.append(defaults.get(atom, fallback.get(atom, f"{atom}_generic")))
    return subtypes


def material_design_to_uc(
    metal_formula: str,
    material_system: str,
    donor_atoms: list,
    coordination_number: int = 0,
    is_macrocyclic: bool = False,
    cavity_radius_nm: float = 0.0,
    chelate_rings: int = 0,
    ring_sizes: list = None,
    n_ligand_molecules: int = 1,
    pH: float = 7.0,
    temperature_C: float = 25.0,
    ionic_strength_M: float = 0.1,
    host_name: str = "",
    donor_subtypes: list = None,
) -> UniversalComplex:
    """Convert a material adapter's design parameters into a UniversalComplex.

    This is the key bridge function. It takes what the realization engine
    knows about a design and converts it into the format the calibrated
    physics engine expects.

    Args:
        metal_formula: Target metal ion (e.g. "Pb2+")
        material_system: Material system ID (e.g. "cyclic_encapsulant")
        donor_atoms: List of donor atom types ["O", "O", "O", "O", "N", "N"]
        coordination_number: Number of coordination bonds (0 = len(donor_atoms))
        is_macrocyclic: True for macrocycles
        cavity_radius_nm: Macrocycle cavity radius in nm
        chelate_rings: Number of chelate rings
        ring_sizes: Ring sizes [5, 5, 6] etc
        n_ligand_molecules: Number of separate ligand molecules
        pH: Solution pH
        temperature_C: Temperature in Celsius
        ionic_strength_M: Ionic strength
        host_name: Human-readable host name
        donor_subtypes: Explicit donor subtypes (overrides auto-resolution)

    Returns:
        UniversalComplex ready for unified_scorer_v2.predict()
    """
    metal = METAL_DB.get(metal_formula)
    if metal is None:
        raise ValueError(f"Unknown metal: {metal_formula}")

    if coordination_number == 0:
        coordination_number = len(donor_atoms)

    if donor_subtypes is None:
        donor_subtypes = _resolve_donor_subtypes(donor_atoms, material_system)

    # Determine donor type classification (HSAB)
    soft_donors = {"S", "Se", "P"}
    hard_donors = {"O", "F"}
    atom_set = set(donor_atoms)
    if atom_set <= soft_donors:
        donor_type = "soft"
    elif atom_set <= hard_donors:
        donor_type = "hard"
    elif atom_set & soft_donors and atom_set & hard_donors:
        donor_type = "mixed"
    else:
        donor_type = "borderline"

    # Infer chelate ring count if not provided
    if chelate_rings == 0 and n_ligand_molecules == 1 and len(donor_atoms) > 1:
        # Polydentate single ligand → likely has chelate rings
        chelate_rings = max(0, len(donor_atoms) - 1)

    if ring_sizes is None and chelate_rings > 0:
        ring_sizes = [5] * chelate_rings  # default 5-membered rings

    return UniversalComplex(
        name=f"{metal_formula}@{host_name or material_system}",
        binding_mode="metal_coordination",
        log_Ka_exp=0.0,  # no experimental value — this is a prediction
        temperature_C=temperature_C,
        ionic_strength_M=ionic_strength_M,
        ph=pH,
        host_name=host_name or material_system,
        host_type="macrocycle" if is_macrocyclic else "chelator",
        host_charge=0,
        cavity_radius_nm=cavity_radius_nm,
        is_macrocyclic=is_macrocyclic,
        metal_formula=metal_formula,
        metal_charge=metal.charge,
        metal_d_electrons=metal.d_electrons,
        donor_atoms=donor_atoms,
        donor_subtypes=donor_subtypes,
        chelate_rings=chelate_rings,
        ring_sizes=ring_sizes or [],
        denticity=coordination_number,
        n_ligand_molecules=n_ligand_molecules,
        donor_type=donor_type,
        geometry="octahedral" if coordination_number >= 6 else "tetrahedral",
        source="design",
        notes=f"Bridge-generated UC for {material_system} design",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_selectivity(
    target_log_k: float,
    competitor_formulas: list,
    donor_atoms: list,
    material_system: str,
    is_macrocyclic: bool = False,
    cavity_radius_nm: float = 0.0,
    chelate_rings: int = 0,
    ring_sizes: list = None,
    n_ligand_molecules: int = 1,
    pH: float = 7.0,
    donor_subtypes: list = None,
) -> dict:
    """Compute selectivity ratios for target vs each competitor.

    Uses the SAME donor configuration for all ions — this is a fixed-design
    selectivity calculation (how well does THIS pocket discriminate?).

    Returns:
        {competitor_formula: selectivity_ratio} where ratio = 10^(ΔlogK)
    """
    ratios = {}
    for comp_formula in competitor_formulas:
        if comp_formula not in METAL_DB:
            ratios[comp_formula] = float('nan')
            continue

        comp_uc = material_design_to_uc(
            metal_formula=comp_formula,
            material_system=material_system,
            donor_atoms=donor_atoms,
            is_macrocyclic=is_macrocyclic,
            cavity_radius_nm=cavity_radius_nm,
            chelate_rings=chelate_rings,
            ring_sizes=ring_sizes,
            n_ligand_molecules=n_ligand_molecules,
            pH=pH,
            donor_subtypes=donor_subtypes,
        )
        comp_result = unified_predict(comp_uc)
        comp_log_k = comp_result.log_Ka_pred
        delta = target_log_k - comp_log_k
        ratios[comp_formula] = 10 ** delta

    return ratios


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION-WEIGHTED COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════════════════════

# Weight profiles: physics fidelity is always dominant.
# Implementation weights shift with application context.
_APPLICATION_WEIGHTS = {
    "remediation": {
        "physics": 0.50,
        "selectivity": 0.20,
        "cost": 0.15,
        "scalability": 0.10,
        "synthetic_accessibility": 0.05,
    },
    "research": {
        "physics": 0.60,
        "selectivity": 0.15,
        "cost": 0.05,
        "scalability": 0.05,
        "synthetic_accessibility": 0.15,
    },
    "diagnostic": {
        "physics": 0.55,
        "selectivity": 0.25,
        "cost": 0.10,
        "scalability": 0.05,
        "synthetic_accessibility": 0.05,
    },
    "separation": {
        "physics": 0.45,
        "selectivity": 0.25,
        "cost": 0.10,
        "scalability": 0.15,
        "synthetic_accessibility": 0.05,
    },
}


def _compute_composite(
    predicted_log_k: float,
    selectivity_ratios: dict,
    synthetic_accessibility: float,
    cost_score: float,
    scalability_score: float,
    application: str,
) -> float:
    """Compute application-weighted composite score.

    All sub-scores normalized to 0-1 range. Physics dominates.
    """
    weights = _APPLICATION_WEIGHTS.get(application, _APPLICATION_WEIGHTS["research"])

    # Physics score: log K normalized. Higher = better.
    # Typical metal-ligand log K range: 0-25
    physics_score = min(1.0, max(0.0, predicted_log_k / 25.0))

    # Selectivity score: worst-case ratio, capped
    if selectivity_ratios:
        worst = min(selectivity_ratios.values())
        if math.isnan(worst):
            sel_score = 0.0
        else:
            # ratio of 100 → score 1.0, ratio of 1 → score 0.0
            sel_score = min(1.0, max(0.0, math.log10(max(worst, 0.01)) / 2.0))
    else:
        sel_score = 0.5  # no competitors specified

    composite = (
        weights["physics"] * physics_score
        + weights["selectivity"] * sel_score
        + weights["cost"] * cost_score
        + weights["scalability"] * scalability_score
        + weights["synthetic_accessibility"] * synthetic_accessibility
    )
    return composite


# ═══════════════════════════════════════════════════════════════════════════
# MATERIAL SYSTEM DESIGN CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Each material system provides a default coordination design for a given
# target metal. These are the "canonical" designs — what the adapter would
# produce for each target.

def _default_designs_for_metal(metal_formula: str, pH: float = 7.0) -> list:
    """Generate default design configurations for each material system.

    Returns list of dicts with keys needed for material_design_to_uc().
    Each dict represents one material system's best attempt at binding
    the target metal.
    """
    metal = METAL_DB.get(metal_formula)
    if metal is None:
        return []

    charge = metal.charge
    softness = metal.hsab_softness
    r_pm = metal.ionic_radius_pm
    r_nm = r_pm / 1000.0

    # HSAB-informed donor selection
    if softness > 0.6:  # soft metal
        preferred_donors = ["S", "N", "S", "N"]
        preferred_subtypes = ["S_thioether", "N_aromatic", "S_thioether", "N_aromatic"]
    elif softness > 0.3:  # borderline
        preferred_donors = ["N", "N", "O", "O"]
        preferred_subtypes = ["N_amine", "N_amine", "O_carboxylate", "O_carboxylate"]
    else:  # hard metal
        preferred_donors = ["O", "O", "O", "O"]
        preferred_subtypes = ["O_carboxylate", "O_carboxylate", "O_hydroxyl", "O_hydroxyl"]

    # Extend to 6-coordinate
    donors_6 = (preferred_donors * 2)[:6]
    subtypes_6 = (preferred_subtypes * 2)[:6]

    designs = []

    # 1. Planar coordination ring (porphyrin-like): 4 N donors
    designs.append({
        "material_system": "planar_coordination_ring",
        "donor_atoms": ["N", "N", "N", "N"],
        "donor_subtypes": ["N_pyrrole", "N_pyrrole", "N_pyrrole", "N_pyrrole"],
        "coordination_number": 4,
        "is_macrocyclic": True,
        "cavity_radius_nm": 0.002,  # ~2 pm porphyrin core
        "chelate_rings": 4,
        "ring_sizes": [5, 5, 5, 5],
        "n_ligand_molecules": 1,
        "host_name": "porphyrin",
        "sa_score": 0.6,
        "cost_score": 0.4,
        "scalability_score": 0.5,
    })

    # 2. Cyclic encapsulant (crown ether): O donors for hard, mixed for borderline
    crown_donors = ["O"] * 6 if softness < 0.3 else ["O", "O", "O", "N", "N", "O"]
    crown_subtypes = (
        ["O_carbonyl"] * 6 if softness < 0.3
        else ["O_carbonyl", "O_carbonyl", "O_carbonyl", "N_amine", "N_amine", "O_carbonyl"]
    )
    designs.append({
        "material_system": "cyclic_encapsulant",
        "donor_atoms": crown_donors,
        "donor_subtypes": crown_subtypes,
        "coordination_number": 6,
        "is_macrocyclic": True,
        "cavity_radius_nm": r_nm * 1.2,  # slightly larger than ion
        "chelate_rings": 6,
        "ring_sizes": [5] * 6,
        "n_ligand_molecules": 1,
        "host_name": "crown_ether",
        "sa_score": 0.8,
        "cost_score": 0.7,
        "scalability_score": 0.8,
    })

    # 3. Periodic lattice node (MOF-like): carboxylate O donors
    designs.append({
        "material_system": "periodic_lattice_node",
        "donor_atoms": ["O", "O", "O", "O", "O", "O"],
        "donor_subtypes": ["O_carboxylate"] * 6,
        "coordination_number": 6,
        "is_macrocyclic": False,
        "cavity_radius_nm": 0.0,
        "chelate_rings": 0,
        "ring_sizes": [],
        "n_ligand_molecules": 6,
        "host_name": "MOF_node",
        "sa_score": 0.5,
        "cost_score": 0.6,
        "scalability_score": 0.7,
    })

    # 4. Folded polypeptide: HSAB-informed mixed donors
    designs.append({
        "material_system": "folded_polypeptide",
        "donor_atoms": donors_6,
        "donor_subtypes": subtypes_6,
        "coordination_number": 6,
        "is_macrocyclic": False,
        "cavity_radius_nm": 0.0,
        "chelate_rings": 2,
        "ring_sizes": [5, 5],
        "n_ligand_molecules": 1,
        "host_name": "designed_peptide",
        "sa_score": 0.4,
        "cost_score": 0.3,
        "scalability_score": 0.3,
    })

    # 5. Emergent coordination cage: mixed N/O
    designs.append({
        "material_system": "emergent_coordination_cage",
        "donor_atoms": ["N", "N", "N", "O", "O", "O"],
        "donor_subtypes": ["N_aromatic", "N_aromatic", "N_aromatic",
                           "O_carboxylate", "O_carboxylate", "O_carboxylate"],
        "coordination_number": 6,
        "is_macrocyclic": False,
        "cavity_radius_nm": 0.0,
        "chelate_rings": 3,
        "ring_sizes": [5, 5, 5],
        "n_ligand_molecules": 2,
        "host_name": "coordination_cage",
        "sa_score": 0.3,
        "cost_score": 0.2,
        "scalability_score": 0.4,
    })

    # 6. EDTA-like chelator (baseline): mixed N/O, high chelation
    designs.append({
        "material_system": "cyclic_encapsulant",
        "donor_atoms": ["N", "N", "O", "O", "O", "O"],
        "donor_subtypes": ["N_amine", "N_amine", "O_carboxylate",
                           "O_carboxylate", "O_carboxylate", "O_carboxylate"],
        "coordination_number": 6,
        "is_macrocyclic": False,
        "cavity_radius_nm": 0.0,
        "chelate_rings": 5,
        "ring_sizes": [5, 5, 5, 5, 5],
        "n_ligand_molecules": 1,
        "host_name": "EDTA_like_chelator",
        "sa_score": 0.9,
        "cost_score": 0.9,
        "scalability_score": 0.9,
    })

    # 7. Functionalized lignin (for remediation): phenolate O donors
    designs.append({
        "material_system": "lignin",
        "donor_atoms": ["O", "O", "O", "O"],
        "donor_subtypes": ["O_phenolate", "O_phenolate", "O_carboxylate", "O_carboxylate"],
        "coordination_number": 4,
        "is_macrocyclic": False,
        "cavity_radius_nm": 0.0,
        "chelate_rings": 1,
        "ring_sizes": [6],
        "n_ligand_molecules": 2,
        "host_name": "functionalized_lignin",
        "sa_score": 0.7,
        "cost_score": 0.9,
        "scalability_score": 0.9,
    })

    return designs


# ═══════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def end_to_end_design(
    target: str,
    conditions: dict = None,
    competitors: list = None,
    application: str = "research",
) -> DesignResult:
    """Full pipeline: target → ranked material systems with calibrated predictions.

    Args:
        target: Metal formula (e.g. "Pb2+")
        conditions: {"pH": 6.0, "matrix": "mine_water", "temperature_C": 25.0}
        competitors: ["Ca2+", "Mg2+", "Fe3+"] — ions that must be excluded
        application: "remediation" | "research" | "diagnostic" | "separation"

    Returns:
        DesignResult with ranked_designs, each carrying calibrated log K
        and selectivity ratios.
    """
    if conditions is None:
        conditions = {}
    if competitors is None:
        competitors = []

    pH = conditions.get("pH", 7.0)
    temperature_C = conditions.get("temperature_C", 25.0)
    ionic_strength_M = conditions.get("ionic_strength_M", 0.1)

    result = DesignResult(
        target=target,
        conditions=conditions,
        competitors=competitors,
        application=application,
    )

    # Validate target
    if target not in METAL_DB:
        result.pipeline_complete = True
        return result

    # Generate default designs for each material system
    designs = _default_designs_for_metal(target, pH=pH)
    result.n_materials_evaluated = len(designs)

    scored_designs = []

    for design_config in designs:
        ms = design_config["material_system"]
        sa_score = design_config.pop("sa_score", 0.5)
        cost_score = design_config.pop("cost_score", 0.5)
        scalability_score = design_config.pop("scalability_score", 0.5)

        # Build UniversalComplex from design
        uc = material_design_to_uc(
            metal_formula=target,
            pH=pH,
            temperature_C=temperature_C,
            ionic_strength_M=ionic_strength_M,
            **design_config,
        )

        # Score with calibrated physics
        pred = unified_predict(uc)
        predicted_log_k = pred.log_Ka_pred

        # Compute selectivity against competitors
        selectivity = {}
        if competitors:
            selectivity = compute_selectivity(
                target_log_k=predicted_log_k,
                competitor_formulas=competitors,
                donor_atoms=design_config["donor_atoms"],
                material_system=ms,
                is_macrocyclic=design_config.get("is_macrocyclic", False),
                cavity_radius_nm=design_config.get("cavity_radius_nm", 0.0),
                chelate_rings=design_config.get("chelate_rings", 0),
                ring_sizes=design_config.get("ring_sizes"),
                n_ligand_molecules=design_config.get("n_ligand_molecules", 1),
                pH=pH,
                donor_subtypes=design_config.get("donor_subtypes"),
            )

        # Compute composite score
        composite = _compute_composite(
            predicted_log_k=predicted_log_k,
            selectivity_ratios=selectivity,
            synthetic_accessibility=sa_score,
            cost_score=cost_score,
            scalability_score=scalability_score,
            application=application,
        )

        mds = MaterialDesignScore(
            material_system=ms,
            adapter_id=design_config.get("host_name", ms),
            predicted_log_k=predicted_log_k,
            prediction_result=pred,
            selectivity_ratios=selectivity,
            realization_feasible=True,
            physics_fidelity=min(1.0, max(0.0, predicted_log_k / 25.0)),
            synthetic_accessibility=sa_score,
            cost_score=cost_score,
            scalability_score=scalability_score,
            composite_score=composite,
        )
        scored_designs.append(mds)

    # Sort by composite score (descending)
    scored_designs.sort(key=lambda d: d.composite_score, reverse=True)
    result.ranked_designs = scored_designs
    result.n_materials_feasible = sum(1 for d in scored_designs if d.realization_feasible)
    result.pipeline_complete = True

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SMALL-MOLECULE GUEST DESIGN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
# Entry point for "I have a molecule, design me a binder"
# Runs: SMILES → pharmacophore → host screening + MIP design + pocket spec
# Does NOT modify the metal-ion pathway above.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GuestDesignResult:
    """End-to-end design output for a small-molecule guest target."""

    guest_smiles: str
    guest_name: str = ""

    # Layer 1: pharmacophore analysis
    pharmacophore: object = None           # GuestPharmacophore

    # Layer 2: ideal pocket geometry
    pocket_spec: object = None             # InteractionGeometrySpec

    # Host-guest screening (scored via unified_scorer_v2)
    host_screen: list = field(default_factory=list)  # list[HostScreenResult]

    # MIP design
    mip_design: object = None              # MIPDesign

    # De novo receptor generation
    de_novo_result: object = None          # GenerationResult from de_novo_generator

    # Selectivity screening
    selectivity_result: object = None      # SelectivityResult from selectivity_screen

    # DNA origami tertiary binding
    dna_origami_design: object = None      # DNAOrigamiPocketDesign

    # Summary
    top_host: str = ""
    top_host_log_ka: float = 0.0
    n_hosts_screened: int = 0
    n_hosts_feasible: int = 0
    pipeline_complete: bool = False

    # Conditions
    conditions: dict = field(default_factory=dict)
    application: str = "research"


def design_for_guest(
    smiles: str,
    name: str = "",
    conditions: dict = None,
    application: str = "research",
    exclude_species: list = None,
    include_mip: bool = True,
    include_de_novo: bool = True,
    include_selectivity: bool = True,
    include_dna_origami: bool = True,
    de_novo_max_candidates: int = 300,
    de_novo_max_scored: int = 30,
    prefer_electroactive: bool = False,
    require_click: bool = False,
    hosts: list = None,
) -> GuestDesignResult:
    """Full pipeline: guest SMILES → ranked binder designs across modalities.

    Args:
        smiles: Guest molecule SMILES
        name: Display name (e.g. "6PPD-quinone")
        conditions: {"pH": 7.0, "temperature_C": 25.0, "matrix": "stormwater"}
        application: "research" | "remediation" | "diagnostic" | "separation"
        exclude_species: Species that must not bind (selectivity targets)
        include_mip: Whether to run MIP monomer selection
        prefer_electroactive: Prioritize electrochemical MIP for sensor use
        require_click: Require click-chemistry deployable designs
        hosts: Specific host keys to screen (default: all in HOST_DB)

    Returns:
        GuestDesignResult with pharmacophore, pocket spec, host rankings,
        and MIP design.
    """
    if conditions is None:
        conditions = {}

    result = GuestDesignResult(
        guest_smiles=smiles,
        guest_name=name,
        conditions=conditions,
        application=application,
    )

    # ── Step 1: Pharmacophore analysis ──
    from core.small_molecule_target import analyze_guest, guest_to_pocket_spec
    pharma = analyze_guest(smiles, name=name)
    result.pharmacophore = pharma

    # ── Step 2: Generate ideal pocket spec (Layer 2) ──
    pH = conditions.get("pH", 7.0)
    spec = guest_to_pocket_spec(
        pharma,
        application=application,
        pH=pH,
        exclude_species=exclude_species or [],
    )
    result.pocket_spec = spec

    # ── Step 3: Screen existing host cavities via unified_scorer_v2 ──
    from core.small_molecule_target import screen_hosts
    host_results = screen_hosts(smiles, hosts=hosts, name=name)
    result.host_screen = host_results
    result.n_hosts_screened = len(host_results)

    # Filter feasible hosts (packing OK, no error)
    feasible = [
        h for h in host_results
        if h.feasibility_note == "" and h.log_Ka_pred > 0
    ]
    result.n_hosts_feasible = len(feasible)

    if feasible:
        result.top_host = feasible[0].host_key
        result.top_host_log_ka = feasible[0].log_Ka_pred

    # ── Step 4: MIP design ──
    if include_mip:
        from adapters.mip_adapter import select_monomers_for_guest
        mip = select_monomers_for_guest(
            guest_smiles=smiles,
            guest_name=name,
            pharmacophore=pharma,
            prefer_electroactive=prefer_electroactive,
            require_click=require_click,
        )
        result.mip_design = mip

    # ── Step 5: De novo receptor generation ──
    if include_de_novo:
        from core.de_novo_generator import generate_for_guest as _gen_for_guest
        de_novo = _gen_for_guest(
            guest_smiles=smiles,
            guest_name=name,
            max_candidates=de_novo_max_candidates,
            max_scored=de_novo_max_scored,
        )
        result.de_novo_result = de_novo

    # ── Step 6: Selectivity screening ──
    if include_selectivity:
        from core.selectivity_screen import screen_selectivity, Interferent
        # Build interferent list from exclude_species if provided
        custom_interferents = None
        if exclude_species:
            custom_interferents = [
                Interferent(name=sp, smiles=sp, relationship="custom")
                if not sp.startswith("[") and "=" in sp or "c" in sp  # looks like SMILES
                else Interferent(name=sp, smiles=sp, relationship="custom")
                for sp in exclude_species
            ]
        sel = screen_selectivity(
            target_smiles=smiles,
            target_name=name,
            interferents=custom_interferents,  # None → auto-detect panel
            hosts=hosts,
        )
        result.selectivity_result = sel

    # ── Step 7: DNA origami tertiary binding design ──
    if include_dna_origami:
        from core.dna_origami_pocket import design_dna_origami_pocket
        dna_design = design_dna_origami_pocket(
            spec=spec,
            guest_smiles=smiles,
            guest_name=name,
            guest_volume_A3=pharma.volume_A3,
            guest_max_dim_A=pharma.max_dimension_A,
        )
        result.dna_origami_design = dna_design

    result.pipeline_complete = True
    return result