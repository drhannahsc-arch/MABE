"""
knowledge/scaffold_geometry.py - Scaffold positioning capabilities database.
"""
from dataclasses import dataclass, field

@dataclass
class ScaffoldPositioning:
    scaffold_type: str
    backbone_type: str
    min_donor_spacing_nm: float
    max_donor_spacing_nm: float
    spacing_precision_nm: float
    max_interior_donors: int
    max_exterior_donors: int
    achievable_geometries: list
    geometry_precision: str
    interior_volume_nm3: float
    pore_diameter_nm: float
    ph_range: tuple
    temp_range_c: tuple
    ionic_strength_max_mm: float
    synthesis_complexity: str
    cost_relative: float
    scalability: str
    notes: str = ""

SCAFFOLD_LIBRARY = [
    ScaffoldPositioning("dna_origami_icosahedron", "DNA",
        3.4, 40.0, 0.34, 60, 140,
        ["icosahedral_vertices","face_centers","edge_midpoints","arbitrary_interior"],
        "nm", 15000.0, 8.0, (5.0, 9.0), (4.0, 45.0), 500.0,
        "complex", 50.0, "lab",
        "Precise sub-nm positioning via staple overhangs. DNA depurinates below pH 5."),
    ScaffoldPositioning("dna_origami_tetrahedron", "DNA",
        3.4, 20.0, 0.34, 12, 24,
        ["tetrahedral_vertices","edge_midpoints","face_centers"],
        "nm", 500.0, 5.0, (5.0, 9.0), (4.0, 50.0), 500.0,
        "moderate", 20.0, "lab",
        "Smaller, simpler DNA nanostructure."),
    ScaffoldPositioning("zeolite_Y", "silica",
        0.3, 1.3, 0.05, 48, 0,
        ["tetrahedral_cage","sodalite_cage","supercage_octahedral"],
        "atomic", 0.85, 0.74, (3.0, 12.0), (-20.0, 700.0), 10000.0,
        "moderate", 0.5, "industrial",
        "Si-O bonds (452 kJ/mol) > C-C (346 kJ/mol). Inherent cation exchange."),
    ScaffoldPositioning("zeolite_ZSM5", "silica",
        0.3, 0.56, 0.05, 12, 0,
        ["channel_linear","channel_sinusoidal","intersection"],
        "atomic", 0.35, 0.56, (2.0, 12.0), (-20.0, 800.0), 10000.0,
        "moderate", 0.3, "industrial",
        "Shape-selective via channel dimensions."),
    ScaffoldPositioning("MOF_UiO66", "metal-organic",
        0.6, 1.2, 0.1, 12, 0,
        ["octahedral_cage","tetrahedral_cage"],
        "atomic", 0.55, 0.60, (1.0, 12.0), (-20.0, 400.0), 5000.0,
        "moderate", 2.0, "pilot",
        "Zr-oxo nodes extremely stable. Linker functionalization adds donors."),
    ScaffoldPositioning("MOF_MIL101", "metal-organic",
        0.8, 3.4, 0.2, 24, 0,
        ["large_cage","small_cage"],
        "nm", 20.6, 1.6, (1.0, 10.0), (-20.0, 300.0), 5000.0,
        "moderate", 3.0, "pilot",
        "Huge pores accommodate large hydrated ions."),
    ScaffoldPositioning("MIP", "organic",
        0.2, 2.0, 0.1, 8, 0,
        ["templated_cavity"],
        "atomic", 0.5, 0.3, (1.0, 13.0), (-20.0, 150.0), 10000.0,
        "moderate", 1.0, "industrial",
        "Cavity IS the binder. Templated around target ion."),
    ScaffoldPositioning("LDH", "inorganic_layered",
        0.3, 1.0, 0.1, 0, 100,
        ["interlayer_planar"],
        "nm", 0.0, 0.76, (4.0, 12.0), (-20.0, 300.0), 5000.0,
        "simple", 0.3, "industrial",
        "Anion binding via charged interlayers."),
    ScaffoldPositioning("mesoporous_silica_MCM41", "silica",
        1.0, 100.0, 1.0, 200, 100,
        ["cylindrical_pore_linear"],
        "nm", 500.0, 3.5, (2.0, 10.0), (-20.0, 500.0), 10000.0,
        "moderate", 1.0, "industrial",
        "High surface area (>1000 m2/g). Functionalize with organosilanes."),
    ScaffoldPositioning("COF", "organic",
        0.5, 3.0, 0.2, 16, 0,
        ["hexagonal_channel","kagome","square_grid"],
        "atomic", 2.0, 1.5, (1.0, 14.0), (-20.0, 400.0), 10000.0,
        "complex", 5.0, "lab",
        "All-covalent framework. Extremely chemically stable."),
    ScaffoldPositioning("coordination_cage", "metal-organic",
        0.3, 3.0, 0.05, 12, 24,
        ["M2L4_lantern","M4L6_tetrahedron","M6L4_octahedron","M8L6_cube","M12L24_cuboctahedron"],
        "atomic", 1.5, 0.8, (3.0, 11.0), (0.0, 80.0), 1000.0,
        "complex", 10.0, "lab",
        "Self-assembled discrete cages. Precise cavity size."),
    ScaffoldPositioning("carbon_nanotube", "carbon",
        0.2, 1000.0, 0.5, 0, 500,
        ["cylindrical_surface"],
        "nm", 10.0, 1.0, (1.0, 14.0), (-200.0, 400.0), 10000.0,
        "moderate", 3.0, "pilot",
        "Exterior functionalization. Electroactive."),
    ScaffoldPositioning("dendrimer_PAMAM_G4", "organic",
        0.5, 4.5, 0.5, 30, 64,
        ["spherical_shell","interior_pockets"],
        "nm", 20.0, 1.0, (2.0, 12.0), (-20.0, 80.0), 5000.0,
        "moderate", 15.0, "lab",
        "Multivalent. Terminal group modification gives N, O, or S donors."),
]

def get_scaffolds_for_conditions(ph, temp_c, ionic_strength_mm=0.0):
    return [s for s in SCAFFOLD_LIBRARY
        if s.ph_range[0] <= ph <= s.ph_range[1]
        and s.temp_range_c[0] <= temp_c <= s.temp_range_c[1]
        and ionic_strength_mm <= s.ionic_strength_max_mm]

def get_scaffolds_with_precision(required_precision_nm):
    return [s for s in SCAFFOLD_LIBRARY if s.spacing_precision_nm <= required_precision_nm]

def get_scaffolds_fitting_ion(hydrated_radius_nm):
    return [s for s in SCAFFOLD_LIBRARY
        if s.pore_diameter_nm >= hydrated_radius_nm * 2 or s.pore_diameter_nm == 0]


