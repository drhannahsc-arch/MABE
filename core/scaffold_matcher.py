"""
core/scaffold_matcher.py - Generative Coordination Engine, Layer 3.
From DonorArrangement -> GenerativeBinderAssembly candidates.
"""
from dataclasses import dataclass, field
from typing import Optional
from knowledge.scaffold_geometry import SCAFFOLD_LIBRARY, get_scaffolds_for_conditions

@dataclass
class ScaffoldMatch:
    scaffold_type: str
    backbone_type: str
    spacing_achievable: bool
    spacing_match_nm: float
    capacity_sufficient: bool
    condition_compatible: bool
    geometry_compatible: bool
    fit_score: float
    cost_relative: float
    scalability: str
    notes: str = ""

@dataclass
class GenerativeBinderAssembly:
    name: str
    design_source: str = "generative_coordination_engine"
    recognition_type: str = ""
    donor_atoms: list = field(default_factory=list)
    donor_groups: list = field(default_factory=list)
    donor_type: str = ""
    effective_denticity: int = 0
    chelate_rings: int = 0
    scaffold_type: str = ""
    scaffold_backbone: str = ""
    pore_diameter_nm: float = 0.0
    interior_volume_nm3: float = 0.0
    target_formula: str = ""
    coordination_number: int = 0
    geometry: str = ""
    hsab_match_score: float = 0.0
    lfse_stabilization_kj: float = 0.0
    geometry_preference_kj: float = 0.0
    ph_working_range: tuple = (0.0, 14.0)
    donor_arrangement_rank: int = 0
    scaffold_match_score: float = 0.0
    synthetic_feasibility: float = 0.0
    cost_relative: float = 1.0
    rationale: str = ""
    confidence_reasoning: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    special_constraints: list = field(default_factory=list)
    is_novel: bool = True

_GEOM_COMPAT = {
    "octahedral": ["octahedral_cage","supercage_octahedral","large_cage","arbitrary_interior"],
    "tetrahedral": ["tetrahedral_cage","tetrahedral_vertices","arbitrary_interior","templated_cavity"],
    "square_planar": ["square_grid","arbitrary_interior","interlayer_planar","templated_cavity"],
    "linear": ["channel_linear","arbitrary_interior","cylindrical_pore_linear","cylindrical_surface"],
    "tetragonal_elongated": ["octahedral_cage","large_cage","arbitrary_interior","templated_cavity"],
    "hemidirected_seesaw": ["arbitrary_interior","templated_cavity","interior_pockets"],
    "hemidirected_octahedral": ["arbitrary_interior","templated_cavity","large_cage"],
    "pentagonal_bipyramidal_equatorial": ["large_cage","arbitrary_interior"],
}

def _geometry_compatible(required, available):
    return any(a in _GEOM_COMPAT.get(required, []) for a in available)

def match_scaffolds(donor_arrangement, coord_env, working_ph=7.0,
                    working_temp_c=25.0, ionic_strength_mm=10.0,
                    hydrated_radius_nm=0.2, max_matches=5):
    matches = []
    for sc in SCAFFOLD_LIBRARY:
        ph_ok = sc.ph_range[0] <= working_ph <= sc.ph_range[1]
        temp_ok = sc.temp_range_c[0] <= working_temp_c <= sc.temp_range_c[1]
        is_ok = ionic_strength_mm <= sc.ionic_strength_max_mm
        if not (ph_ok and temp_ok and is_ok): continue

        req = donor_arrangement.required_spacing_nm
        spacing_ok = sc.min_donor_spacing_nm <= req <= sc.max_donor_spacing_nm
        sp_match = 0.0
        if spacing_ok:
            sp_match = 1.0 if sc.spacing_precision_nm <= donor_arrangement.spacing_tolerance_nm else \
                max(0.0, 1.0 - (sc.spacing_precision_nm - donor_arrangement.spacing_tolerance_nm) / req)

        cap_ok = sc.max_interior_donors >= len(donor_arrangement.positioned_donors)
        geom_ok = (donor_arrangement.geometry in sc.achievable_geometries
                   or "arbitrary_interior" in sc.achievable_geometries
                   or _geometry_compatible(donor_arrangement.geometry, sc.achievable_geometries))
        pore_ok = sc.pore_diameter_nm >= hydrated_radius_nm * 2

        score = 0.0
        if spacing_ok: score += sp_match * 0.35
        if cap_ok: score += 0.20
        if geom_ok: score += 0.25
        if pore_ok: score += 0.10
        score += max(0, 0.10 * (1.0 - min(sc.cost_relative, 50.0) / 50.0))

        if score > 0.2:
            matches.append(ScaffoldMatch(sc.scaffold_type, sc.backbone_type,
                spacing_ok, abs(req - (sc.min_donor_spacing_nm + sc.max_donor_spacing_nm) / 2),
                cap_ok, True, geom_ok, round(score, 3), sc.cost_relative, sc.scalability, sc.notes))

    matches.sort(key=lambda m: m.fit_score, reverse=True)
    return matches[:max_matches]

def assemble_generative_binders(coord_env, donor_arrangement, scaffold_matches,
                                 target_identity=""):
    assemblies = []
    for match in scaffold_matches:
        da_list = [p.donor_atom for p in donor_arrangement.positioned_donors]
        unique_da = set(da_list)
        if unique_da == {"S"}: dt = "soft"
        elif unique_da == {"O"}: dt = "hard"
        elif unique_da == {"N"}: dt = "borderline"
        else: dt = "mixed"
        dg = list(set(p.ligand_name for p in donor_arrangement.positioned_donors))
        dd = "+".join(sorted(set(p.ligand_name for p in donor_arrangement.positioned_donors)))
        name = f"gen_{target_identity}_{dd}@{match.scaffold_type}"

        conf = []
        if coord_env.lfse_stabilization_kj > 50:
            conf.append(f"Strong LFSE ({coord_env.lfse_stabilization_kj:.0f} kJ/mol) for {coord_env.geometry}")
        if donor_arrangement.hsab_match_score > 0.8:
            conf.append(f"Excellent HSAB match ({donor_arrangement.hsab_match_score:.2f})")
        if match.fit_score > 0.7:
            conf.append(f"Strong scaffold fit ({match.fit_score:.2f})")
        if donor_arrangement.chelate_rings > 0:
            conf.append(f"{donor_arrangement.chelate_rings} chelate ring(s)")

        fails = []
        if not match.spacing_achievable:
            fails.append(f"Scaffold may not achieve {donor_arrangement.required_spacing_nm:.2f} nm spacing")
        if donor_arrangement.ph_working_range[0] > 5.0:
            fails.append(f"Donors require pH > {donor_arrangement.ph_working_range[0]:.1f}")
        if match.cost_relative > 10.0:
            fails.append(f"High cost ({match.cost_relative:.0f}x baseline)")

        sd = next((s for s in SCAFFOLD_LIBRARY if s.scaffold_type == match.scaffold_type), None)
        assemblies.append(GenerativeBinderAssembly(
            name=name, recognition_type="generative_coordination_pocket",
            donor_atoms=da_list, donor_groups=dg, donor_type=dt,
            effective_denticity=donor_arrangement.effective_denticity,
            chelate_rings=donor_arrangement.chelate_rings,
            scaffold_type=match.scaffold_type, scaffold_backbone=match.backbone_type,
            pore_diameter_nm=sd.pore_diameter_nm if sd else 0.0,
            interior_volume_nm3=sd.interior_volume_nm3 if sd else 0.0,
            target_formula=coord_env.target_formula,
            coordination_number=coord_env.coordination_number,
            geometry=coord_env.geometry,
            hsab_match_score=donor_arrangement.hsab_match_score,
            lfse_stabilization_kj=coord_env.lfse_stabilization_kj,
            geometry_preference_kj=coord_env.geometry_preference_kj,
            ph_working_range=donor_arrangement.ph_working_range,
            donor_arrangement_rank=donor_arrangement.rank,
            scaffold_match_score=match.fit_score,
            synthetic_feasibility=donor_arrangement.synthetic_feasibility,
            cost_relative=match.cost_relative,
            rationale=f"Generative design for {coord_env.target_formula}: {coord_env.rationale} "
                      f"Donors: {donor_arrangement.rationale} Scaffold: {match.scaffold_type}",
            confidence_reasoning=conf, failure_modes=fails,
            special_constraints=coord_env.special_constraints))
    return assemblies


