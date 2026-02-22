"""
core/generative_physics_adapter.py - Sprint 17: Generative -> Physics Pipeline Adapter

Maps GenerativeBinderAssembly objects from Sprint 16 into the dataclass
shapes expected by the Sprint 9-15 physics pipeline (RecognitionChemistry,
StructuralConstraint, InteriorDesign, Problem), runs full_physics_rescore(),
and merges physics scores back into the assembly.

Also provides design_and_score() — the unified entry point that runs
generative design and physics scoring in one call.
"""
from dataclasses import dataclass, field
from typing import Optional
import math

from core.coordination_generator import (
    METAL_HSAB_SOFTNESS, METAL_D_ELECTRONS, SPECIAL_EFFECTS,
    _IONIC_RADII,
)
from core.generative_integration import generative_design


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE DATACLASS SHAPES (matching Sprints 1-8 exactly)
# These mirror what bootstrap_sprint1 through bootstrap_sprint8 create.
# In the real codebase these are imported from core.problem, core.assembly, etc.
# Here we define compatible versions so Sprint 17 tests can run standalone.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RecognitionChemistry:
    """From Sprint 2+: what chemical groups do the binding."""
    name: str
    type: str              # "chelator", "dnazyme", "peptide", "aptamer", "generative"
    donor_atoms: list      # ["N", "N", "O", "O"]
    donor_type: str        # "hard", "borderline", "soft", "mixed"
    structure: str = ""    # SMILES, sequence, or description
    denticity: int = 0
    stability_constant_log: float = 0.0
    hsab_match: float = 0.0
    chelate_rings: int = 0
    notes: str = ""

@dataclass
class StructuralConstraint:
    """From Sprint 6+: what scaffold positions the recognition chemistry."""
    name: str
    type: str              # "free", "dna_origami", "MOF", "zeolite", "MIP", etc.
    geometry: str          # "icosahedral", "tetrahedral", "channel", "cavity", etc.
    pore_size_nm: float = 0.0
    interior_volume_nm3: float = 0.0
    backbone_type: str = ""
    ph_range: tuple = (0.0, 14.0)
    temp_range_c: tuple = (-20.0, 100.0)
    cost_relative: float = 1.0
    notes: str = ""

@dataclass
class InteriorDesign:
    """From Sprint 8: how recognition elements are arranged inside scaffold."""
    description: str
    num_binding_sites: int = 1
    site_spacing_nm: float = 0.0
    cooperativity: str = "none"  # "none", "positive", "negative"
    self_binding: bool = False   # Scaffold IS the binder (zeolite, MIP, LDH)
    notes: str = ""

@dataclass
class ReleaseCondition:
    """How to release captured target."""
    mechanism: str         # "pH_shift", "competitor", "thermal", "electrochemical"
    trigger: str = ""      # "pH < 3", "EDTA 10mM", "T > 60C"
    reversible: bool = True
    notes: str = ""

@dataclass
class TargetSpecies:
    """Target ion/molecule description for physics pipeline."""
    identity: str
    formula: str
    charge: int = 2
    ionic_radius_pm: float = 80.0
    hydrated_radius_nm: float = 0.2
    hsab_softness: float = 0.3
    d_electrons: int = 0
    coordination_number: int = 6
    preferred_geometry: str = "octahedral"

@dataclass
class Matrix:
    """Deployment matrix conditions."""
    description: str = ""
    ph: float = 7.0
    temperature_c: float = 25.0
    ionic_strength_mm: float = 10.0
    competing_species: list = field(default_factory=list)

@dataclass
class Problem:
    """Complete problem specification for physics pipeline."""
    target: TargetSpecies = None
    matrix: Matrix = None
    desired_outcome: str = "capture_release"
    constraints: dict = field(default_factory=dict)

@dataclass
class BindingThermodynamics:
    """Output of compute_thermodynamics()."""
    dg_bind_kj: float = 0.0
    dg_desolv_kj: float = 0.0
    dg_preorg_kj: float = 0.0
    dg_chelate_kj: float = 0.0
    dg_electrostatic_kj: float = 0.0
    dg_protonation_kj: float = 0.0
    dg_lfse_kj: float = 0.0
    dg_activity_kj: float = 0.0
    dg_screening_kj: float = 0.0
    dg_repulsion_kj: float = 0.0
    dg_net_kj: float = 0.0
    predicted_kd_um: float = 0.0
    confidence: str = "moderate"

@dataclass
class BinderAssembly:
    """Complete assembly as expected by the physics pipeline."""
    recognition: RecognitionChemistry = None
    structure: StructuralConstraint = None
    interior: InteriorDesign = None
    release: ReleaseCondition = None
    confidence_reasoning: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    # Physics results (populated by scoring)
    thermodynamics: BindingThermodynamics = None
    physics_score: float = 0.0
    temperature_prediction: dict = field(default_factory=dict)
    design_source: str = "catalog"
    is_novel: bool = False
    name: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER: GenerativeBinderAssembly -> BinderAssembly
# ═══════════════════════════════════════════════════════════════════════════

def adapt_generative_to_pipeline(gen_assembly, problem=None):
    """Convert a GenerativeBinderAssembly into a BinderAssembly
    compatible with the Sprint 9-15 physics pipeline.

    Args:
        gen_assembly: GenerativeBinderAssembly from Sprint 16
        problem: Optional Problem for context

    Returns:
        BinderAssembly ready for compute_thermodynamics() and full_physics_rescore()
    """
    # Map recognition chemistry
    recognition = RecognitionChemistry(
        name=gen_assembly.name,
        type="generative",
        donor_atoms=gen_assembly.donor_atoms,
        donor_type=gen_assembly.donor_type,
        structure=f"Generative: {', '.join(gen_assembly.donor_groups)}",
        denticity=gen_assembly.effective_denticity,
        stability_constant_log=_estimate_stability_constant(gen_assembly),
        hsab_match=gen_assembly.hsab_match_score,
        chelate_rings=gen_assembly.chelate_rings,
        notes=gen_assembly.rationale[:200] if gen_assembly.rationale else "",
    )

    # Map structural constraint
    self_binding = gen_assembly.scaffold_backbone in ("silica", "organic") and \
                   gen_assembly.scaffold_type in ("MIP", "zeolite_Y", "zeolite_ZSM5", "LDH")

    structure = StructuralConstraint(
        name=gen_assembly.scaffold_type,
        type=_map_scaffold_type(gen_assembly.scaffold_type),
        geometry=gen_assembly.geometry,
        pore_size_nm=gen_assembly.pore_diameter_nm,
        interior_volume_nm3=gen_assembly.interior_volume_nm3,
        backbone_type=gen_assembly.scaffold_backbone,
        ph_range=gen_assembly.ph_working_range,
        cost_relative=gen_assembly.cost_relative,
    )

    # Map interior design
    interior = InteriorDesign(
        description=f"{gen_assembly.effective_denticity}-dentate "
                    f"{gen_assembly.donor_type} pocket in {gen_assembly.geometry}",
        num_binding_sites=1,  # Generative designs are per-site
        site_spacing_nm=0.0,
        cooperativity="none",
        self_binding=self_binding,
    )

    # Default release
    release = _infer_release(gen_assembly)

    assembly = BinderAssembly(
        recognition=recognition,
        structure=structure,
        interior=interior,
        release=release,
        confidence_reasoning=gen_assembly.confidence_reasoning,
        failure_modes=gen_assembly.failure_modes,
        design_source="generative_coordination_engine",
        is_novel=True,
        name=gen_assembly.name,
    )

    return assembly


def _estimate_stability_constant(gen_assembly):
    """Estimate log K from HSAB match, denticity, and chelate rings.

    Uses Irving-Williams + chelate effect heuristics:
    - Base logK from HSAB match (~2-4 for good match)
    - +1.5 per chelate ring (chelate effect)
    - +0.5 per additional donor atom beyond 2
    - Bonus for d8 square planar (LFSE)
    """
    base = 2.0 + gen_assembly.hsab_match_score * 3.0
    chelate_bonus = gen_assembly.chelate_rings * 1.5
    denticity_bonus = max(0, (gen_assembly.effective_denticity - 2)) * 0.5
    lfse_bonus = gen_assembly.lfse_stabilization_kj * 0.005  # Scale down
    return round(base + chelate_bonus + denticity_bonus + lfse_bonus, 1)


def _map_scaffold_type(scaffold_type):
    """Map scaffold_type string to the type categories used in Sprint 6+."""
    mapping = {
        "dna_origami_icosahedron": "dna_origami",
        "dna_origami_tetrahedron": "dna_origami",
        "zeolite_Y": "zeolite",
        "zeolite_ZSM5": "zeolite",
        "MOF_UiO66": "MOF",
        "MOF_MIL101": "MOF",
        "MIP": "MIP",
        "LDH": "LDH",
        "mesoporous_silica_MCM41": "mesoporous_silica",
        "COF": "COF",
        "coordination_cage": "coordination_cage",
        "carbon_nanotube": "carbon_nanotube",
        "dendrimer_PAMAM_G4": "dendrimer",
    }
    return mapping.get(scaffold_type, scaffold_type)


def _infer_release(gen_assembly):
    """Infer release mechanism from scaffold type."""
    scaffold = gen_assembly.scaffold_type
    if "zeolite" in scaffold or "LDH" in scaffold:
        return ReleaseCondition("competitor", "NaCl 1M or EDTA 10mM", True,
                                "Ion exchange reversal")
    elif "MIP" in scaffold:
        return ReleaseCondition("pH_shift", "pH < 2 or pH > 12", True,
                                "Cavity distortion at extreme pH")
    elif "MOF" in scaffold:
        return ReleaseCondition("competitor", "EDTA 10mM", True,
                                "Competitive displacement")
    elif "dna_origami" in scaffold:
        return ReleaseCondition("competitor", "EDTA wash", True,
                                "Strip metals, cage reusable")
    else:
        return ReleaseCondition("pH_shift", "pH shift", True)


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS SCORING ENGINE (self-contained, mirrors Sprint 9-15 logic)
# In the full codebase, this calls the monkey-patched compute_thermodynamics()
# and full_physics_rescore(). Here we implement the scoring directly so
# Sprint 17 tests can run standalone.
# ═══════════════════════════════════════════════════════════════════════════

def compute_thermodynamics_standalone(recognition, structure, interior, problem):
    """Compute the full 10-term deltaG using Sprint 9-15 physics.

    This is a self-contained implementation that mirrors the monkey-patched
    chain. In the full codebase, you'd call:
        import core.thermodynamics as _thermo_mod
        _thermo_mod.compute_thermodynamics(recognition, structure, interior, problem)
    which would chain through all integration patches.
    """
    target = problem.target if problem else None
    matrix = problem.matrix if problem else None
    ph = matrix.ph if matrix else 7.0
    temp_c = matrix.temperature_c if matrix else 25.0
    temp_k = temp_c + 273.15
    ionic_mm = matrix.ionic_strength_mm if matrix else 10.0

    # === Term 1: dG_bind (HSAB + donor energies) ===
    # Sprint 9: donor atom energies matched to target HSAB
    target_softness = target.hsab_softness if target else 0.3
    donor_softness_map = {"O": 0.15, "N": 0.40, "S": 0.80, "P": 0.75}
    donor_energies = []
    for da in recognition.donor_atoms:
        ds = donor_softness_map.get(da, 0.3)
        mismatch = abs(target_softness - ds)
        energy = -20.0 * (1.0 - mismatch)  # kJ/mol per donor
        donor_energies.append(energy)
    dg_bind = sum(donor_energies)

    # === Term 2: dG_desolv (8% per water displaced) ===
    # Sprint 9: desolvation penalty from hydration shell disruption
    cn = target.coordination_number if target else 6
    waters_displaced = min(len(recognition.donor_atoms), cn)
    dg_desolv = waters_displaced * 8.0  # +8 kJ/mol per water (penalty)

    # === Term 3: dG_preorg (structure-dependent) ===
    # Sprint 9: preorganization bonus from scaffold rigidity
    preorg_map = {
        "dna_origami": -5.0, "zeolite": -8.0, "MOF": -7.0,
        "MIP": -10.0, "COF": -6.0, "coordination_cage": -9.0,
        "mesoporous_silica": -3.0, "dendrimer": -2.0,
        "carbon_nanotube": -1.0, "LDH": -4.0, "free": 0.0,
    }
    scaffold_key = structure.type if structure else "free"
    dg_preorg = preorg_map.get(scaffold_key, -2.0)
    if interior and interior.self_binding:
        dg_preorg -= 5.0  # Extra preorg for self-binding scaffolds

    # === Term 4: dG_chelate (-6 kJ/mol per ring) ===
    # Sprint 9: chelate effect — entropic advantage of ring closure
    dg_chelate = -6.0 * recognition.chelate_rings

    # === Term 5: dG_electrostatic ===
    # Sprint 9: charge-charge interaction
    target_charge = target.charge if target else 2
    donor_charge = sum(-1 if da in ("O", "S") else 0 for da in recognition.donor_atoms)
    dg_electrostatic = -2.5 * abs(target_charge) * abs(donor_charge) / max(1, len(recognition.donor_atoms))

    # === Term 6: dG_protonation (Sprint 11) ===
    # Penalty for donors that must deprotonate at working pH
    # Uses effective pKa: imine/pyridine N (~5), amine N (~10.5),
    # carboxylate O (~4.5), phenolate O (~10), thiol S (~8.3)
    # The recognition.notes or structure field may hint at ligand type;
    # we use the donor_type to pick effective pKa.
    # Imine/aromatic N donors (pyridine, imidazole, bipyridyl) have pKa 4-7
    # Amine N donors have pKa 10-11
    # Chelator/generative designs mostly use imine N, not amine N
    pka_defaults = {"O": 4.5, "S": 8.3, "P": 6.5}
    # For N: distinguish aromatic/imine (low pKa) from amine (high pKa)
    # Heuristic: generative designs use coordination N (low pKa ~5-7)
    # Catalog amine chelators use high pKa N
    n_pka = 6.0 if recognition.type == "generative" else 10.5
    dg_protonation = 0.0
    for da in recognition.donor_atoms:
        if da == "N":
            pka = n_pka
        else:
            pka = pka_defaults.get(da, 7.0)
        if ph < pka:  # Donor is protonated → must lose proton
            penalty = 5.7 * (pka - ph)  # RT * ln(10) * delta_pKa
            dg_protonation += min(penalty, 20.0)  # Cap at 20 kJ/mol

    # === Term 7: dG_LFSE (Sprint 12) ===
    # LFSE stabilization for the coordination geometry
    d_electrons = target.d_electrons if target else 0
    lfse_map = {8: {"square_planar": -25.9, "octahedral": -14.4, "tetrahedral": -4.5},
                9: {"tetragonal_elongated": -12.7, "square_planar": -18.0, "octahedral": -9.6},
                6: {"octahedral": -4.8, "tetrahedral": -2.7},
                7: {"octahedral": -9.6, "tetrahedral": -5.3},
                3: {"octahedral": -14.4, "tetrahedral": -3.2}}
    geom_key = recognition.donor_type  # Approximate — real pipeline uses actual geometry
    dg_lfse = 0.0
    if d_electrons in lfse_map:
        for geom, val in lfse_map[d_electrons].items():
            if geom in structure.geometry or structure.geometry in geom:
                dg_lfse = val
                break
        if dg_lfse == 0.0:  # Default to first available
            dg_lfse = list(lfse_map[d_electrons].values())[0]

    # === Term 8: dG_activity (Sprint 13) ===
    # Ionic strength correction via Davies equation
    I = ionic_mm / 1000.0  # Convert to mol/L
    if I > 0 and target_charge != 0:
        sqrt_I = math.sqrt(I)
        log_gamma = -0.509 * target_charge**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)
        dg_activity = -5.71 * log_gamma  # RT * ln(10) at 25°C
    else:
        dg_activity = 0.0

    # === Term 9: dG_screening (Sprint 13) ===
    # Debye screening reduces electrostatic attraction
    if I > 0:
        kappa = 3.29 * sqrt_I  # nm^-1 at 25°C
        screening_factor = math.exp(-kappa * 0.3)  # 0.3 nm contact distance
        dg_screening = dg_electrostatic * (1 - screening_factor)  # Reduces electrostatic
    else:
        dg_screening = 0.0

    # === Term 10: dG_repulsion (Sprint 14) ===
    # Steric exclusion: hydrated ion vs pore
    dg_repulsion = 0.0
    if structure and structure.pore_size_nm > 0:
        hr = target.hydrated_radius_nm if target else 0.2
        if hr * 2 > structure.pore_size_nm * 0.9:  # Tight fit
            squeeze = (hr * 2 - structure.pore_size_nm * 0.9)
            dg_repulsion = 10.0 * max(0, squeeze / 0.1)  # Penalty per 0.1 nm squeeze

    # === Sum ===
    dg_net = (dg_bind + dg_desolv + dg_preorg + dg_chelate + dg_electrostatic
              + dg_protonation + dg_lfse + dg_activity + dg_screening + dg_repulsion)

    # Convert to Kd: dG = RT ln(Kd) → Kd = exp(dG/RT)
    R = 8.314e-3  # kJ/(mol·K)
    kd_m = math.exp(dg_net / (R * temp_k))
    kd_um = kd_m * 1e6

    # Confidence from HSAB match
    conf = "high" if recognition.hsab_match > 0.8 else \
           "moderate" if recognition.hsab_match > 0.5 else "low"

    return BindingThermodynamics(
        dg_bind_kj=round(dg_bind, 2),
        dg_desolv_kj=round(dg_desolv, 2),
        dg_preorg_kj=round(dg_preorg, 2),
        dg_chelate_kj=round(dg_chelate, 2),
        dg_electrostatic_kj=round(dg_electrostatic, 2),
        dg_protonation_kj=round(dg_protonation, 2),
        dg_lfse_kj=round(dg_lfse, 2),
        dg_activity_kj=round(dg_activity, 2),
        dg_screening_kj=round(dg_screening, 2),
        dg_repulsion_kj=round(dg_repulsion, 2),
        dg_net_kj=round(dg_net, 2),
        predicted_kd_um=round(kd_um, 4) if kd_um < 1e6 else round(kd_um, 0),
        confidence=conf,
    )


def temperature_prediction(thermo, base_temp_c=25.0):
    """Gibbs-Helmholtz temperature prediction at 4°C, 37°C, 60°C."""
    R = 8.314e-3
    T_base = base_temp_c + 273.15
    # Approximate: 85% enthalpic for binding, 90% entropic for chelate
    dh_bind = thermo.dg_bind_kj * 0.85
    ds_bind = -(thermo.dg_bind_kj - dh_bind) / T_base
    dh_chelate = thermo.dg_chelate_kj * 0.10  # 90% entropic
    ds_chelate = -(thermo.dg_chelate_kj - dh_chelate) / T_base
    dh_total = dh_bind + dh_chelate + thermo.dg_desolv_kj * 0.5
    ds_total = ds_bind + ds_chelate

    predictions = {}
    for t_c in [4, 25, 37, 60]:
        t_k = t_c + 273.15
        dg_at_t = dh_total - t_k * ds_total
        # Add non-temperature terms unchanged
        dg_at_t += (thermo.dg_preorg_kj + thermo.dg_electrostatic_kj
                     + thermo.dg_protonation_kj + thermo.dg_lfse_kj
                     + thermo.dg_activity_kj + thermo.dg_screening_kj
                     + thermo.dg_repulsion_kj)
        kd_at_t = math.exp(dg_at_t / (R * t_k)) * 1e6
        predictions[f"{t_c}C"] = {
            "dg_kj": round(dg_at_t, 2),
            "kd_um": round(kd_at_t, 4) if kd_at_t < 1e6 else round(kd_at_t, 0),
        }
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# SCORE: Run physics on an adapted assembly
# ═══════════════════════════════════════════════════════════════════════════

def score_assembly(assembly, problem):
    """Run full physics scoring on a BinderAssembly.

    In the real codebase this calls the monkey-patched chain.
    Here it calls the standalone implementation.
    """
    thermo = compute_thermodynamics_standalone(
        assembly.recognition, assembly.structure, assembly.interior, problem)
    assembly.thermodynamics = thermo
    assembly.temperature_prediction = temperature_prediction(thermo,
        problem.matrix.temperature_c if problem and problem.matrix else 25.0)

    # Physics score: more negative dG = better binding
    # Normalize to 0-1 where 1 = excellent binding
    dg = thermo.dg_net_kj
    if dg < -100:
        assembly.physics_score = 1.0
    elif dg < 0:
        assembly.physics_score = round(-dg / 100.0, 3)
    else:
        assembly.physics_score = 0.0

    return assembly


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT: design_and_score()
# ═══════════════════════════════════════════════════════════════════════════

def design_and_score(
    target_identity, target_formula, charge=2,
    d_electrons=None, hsab_softness=None, ionic_radius_pm=None,
    hydrated_radius_nm=0.2,
    working_ph=7.0, working_temp_c=25.0, ionic_strength_mm=10.0,
    competing_species=None,
    max_results=10,
):
    """Full pipeline: generative design + physics scoring.

    This is the top-level function that does everything:
    1. Generate novel binder architectures from target physics
    2. Adapt each to the physics pipeline format
    3. Score via 10-term deltaG + temperature prediction
    4. Rank by predicted binding energy

    Returns list of scored BinderAssembly objects.
    """
    # Resolve defaults
    if d_electrons is None:
        d_electrons = METAL_D_ELECTRONS.get(target_formula, 0)
    if hsab_softness is None:
        hsab_softness = METAL_HSAB_SOFTNESS.get(target_formula, 0.3)
    if ionic_radius_pm is None:
        ionic_radius_pm = _IONIC_RADII.get(target_formula, 80.0)

    # Build problem
    target = TargetSpecies(
        identity=target_identity, formula=target_formula, charge=charge,
        ionic_radius_pm=ionic_radius_pm, hydrated_radius_nm=hydrated_radius_nm,
        hsab_softness=hsab_softness, d_electrons=d_electrons,
    )
    matrix = Matrix(
        ph=working_ph, temperature_c=working_temp_c,
        ionic_strength_mm=ionic_strength_mm,
        competing_species=competing_species or [],
    )
    problem = Problem(target=target, matrix=matrix)

    # Step 1: Generative design
    gen_assemblies = generative_design(
        target_identity, target_formula, charge,
        d_electrons, hsab_softness, ionic_radius_pm,
        hydrated_radius_nm, working_ph, working_temp_c, ionic_strength_mm,
    )

    # Step 2: Adapt and score
    scored = []
    for gen in gen_assemblies:
        assembly = adapt_generative_to_pipeline(gen, problem)
        score_assembly(assembly, problem)
        scored.append(assembly)

    # Step 3: Rank by physics score (more negative dG = better)
    scored.sort(key=lambda a: a.thermodynamics.dg_net_kj)

    return scored[:max_results]


def print_scored_results(assemblies, top_n=5):
    """Pretty-print design + physics results."""
    print(f"\n{'='*75}")
    print(f"  GENERATIVE DESIGN + PHYSICS SCORING — {len(assemblies)} designs")
    print(f"{'='*75}")

    for i, a in enumerate(assemblies[:top_n]):
        t = a.thermodynamics
        print(f"\n  #{i+1}: {a.name}")
        print(f"  {'-'*65}")
        print(f"  Recognition: {a.recognition.type} | {', '.join(a.recognition.donor_atoms)} "
              f"({a.recognition.donor_type})")
        print(f"  Scaffold:    {a.structure.name} ({a.structure.backbone_type})")
        print(f"  Denticity:   {a.recognition.denticity} | "
              f"Chelate rings: {a.recognition.chelate_rings} | "
              f"logK est: {a.recognition.stability_constant_log}")
        print(f"  Release:     {a.release.mechanism} ({a.release.trigger})")
        print(f"")
        print(f"  === THERMODYNAMICS (10-term deltaG) ===")
        print(f"  dG_bind:          {t.dg_bind_kj:+8.2f} kJ/mol")
        print(f"  dG_desolv:        {t.dg_desolv_kj:+8.2f} kJ/mol")
        print(f"  dG_preorg:        {t.dg_preorg_kj:+8.2f} kJ/mol")
        print(f"  dG_chelate:       {t.dg_chelate_kj:+8.2f} kJ/mol")
        print(f"  dG_electrostatic: {t.dg_electrostatic_kj:+8.2f} kJ/mol")
        print(f"  dG_protonation:   {t.dg_protonation_kj:+8.2f} kJ/mol")
        print(f"  dG_LFSE:          {t.dg_lfse_kj:+8.2f} kJ/mol")
        print(f"  dG_activity:      {t.dg_activity_kj:+8.2f} kJ/mol")
        print(f"  dG_screening:     {t.dg_screening_kj:+8.2f} kJ/mol")
        print(f"  dG_repulsion:     {t.dg_repulsion_kj:+8.2f} kJ/mol")
        print(f"  ────────────────────────────────")
        print(f"  dG_net:           {t.dg_net_kj:+8.2f} kJ/mol")
        print(f"  Predicted Kd:     {t.predicted_kd_um:.4f} uM")
        print(f"  Confidence:       {t.confidence}")
        print(f"  Physics score:    {a.physics_score:.3f}")

        if a.temperature_prediction:
            print(f"\n  Temperature prediction:")
            for temp, vals in a.temperature_prediction.items():
                print(f"    {temp}: dG={vals['dg_kj']:+.2f} kJ/mol, Kd={vals['kd_um']:.4f} uM")

        if a.confidence_reasoning:
            print(f"\n  Confidence:")
            for c in a.confidence_reasoning:
                print(f"    + {c}")
        if a.failure_modes:
            print(f"  Failure modes:")
            for f in a.failure_modes:
                print(f"    ! {f}")

    print(f"\n{'='*75}")
    if len(assemblies) > top_n:
        print(f"  ({len(assemblies) - top_n} more not shown)")
    print()


