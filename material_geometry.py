"""
realization_ranker/geometric_fidelity/material_geometry.py

Physics-derived structural parameters for each material system.
Every constant here traces to a measurable physical property.

Sources:
  - DNA: 10.5 bp/turn, 0.34 nm/bp rise, 2.25 nm diameter (Watson-Crick B-form)
  - Proteins: Ramachandran constraints, amino acid volumes (Creighton, Proteins)
  - MOFs: Node geometries from published crystal structures
  - Crown ethers: Pedersen (1967), Izatt (1991) cavity size compilations
  - Porphyrins: 4N core ~2.0 Å radius (Scheidt & Lee, 2000)
"""

# ═══════════════════════════════════════════════════════════════════════════
# Natural cavity size ranges (nm) — what each material can form without strain
# ═══════════════════════════════════════════════════════════════════════════

NATURAL_CAVITY_RANGE_NM = {
    "small_molecule":     (0.05, 0.8),     # Limited by molecular size
    "chelator":           (0.05, 0.5),     # Coordination sphere
    "porphyrin":          (0.18, 0.22),    # Central cavity 1.8–2.2 Å, very constrained
    "crown_ether":        (0.06, 0.46),    # 12-crown-4 (0.6Å) to 24-crown-8 (4.6Å)
    "peptide":            (0.2, 2.0),      # Cyclic peptides, constrained scaffolds
    "protein":            (0.3, 5.0),      # From tight metal sites to large clefts
    "antibody_CDR":       (0.5, 3.0),      # CDR loop pockets
    "aptamer":            (0.5, 3.0),      # Folded nucleic acid pockets
    "dnazyme":            (0.5, 2.0),
    "DNA_origami":        (2.0, 50.0),     # Interior of designed cages
    "MOF":                (0.3, 5.0),      # Pore diameters
    "crystal":            (0.1, 2.0),      # Lattice interstitial sites
    "ion_exchange_resin": (0.2, 1.0),      # Functional group pockets
}

# ═══════════════════════════════════════════════════════════════════════════
# Maximum simultaneous donor atoms — constrained by material topology
# ═══════════════════════════════════════════════════════════════════════════

MAX_SIMULTANEOUS_DONORS = {
    "small_molecule":     8,     # DTPA = 8 donors (3N + 5O)
    "chelator":           8,     # Large polydentate chelators
    "porphyrin":          4,     # 4 pyrrolic N in core (+ 2 axial possible)
    "crown_ether":        8,     # 24-crown-8 = 8 oxygens
    "peptide":            6,     # Constrained by ring size
    "protein":            10,    # Multiple residues can converge on binding site
    "antibody_CDR":       8,     # CDR loop residues
    "aptamer":            6,     # Loop/bulge donors
    "dnazyme":            6,
    "DNA_origami":        20,    # Multiple staple-positioned functional groups
    "MOF":                12,    # Node + linker donors combined
    "crystal":            12,    # Lattice site coordination
    "ion_exchange_resin": 4,     # Typically mono/bidentate functional groups
}

# ═══════════════════════════════════════════════════════════════════════════
# Positional precision (nm) — how accurately can donors be placed?
# ═══════════════════════════════════════════════════════════════════════════

POSITIONAL_PRECISION_NM = {
    "small_molecule":     0.01,   # Bond lengths known to 0.01 Å
    "chelator":           0.01,
    "porphyrin":          0.005,  # Rigid macrocycle, very precise
    "crown_ether":        0.02,   # Slightly flexible ring
    "peptide":            0.05,   # Rotamer flexibility
    "protein":            0.1,    # B-factors, loop mobility
    "antibody_CDR":       0.1,
    "aptamer":            0.2,    # Folding uncertainty
    "dnazyme":            0.2,
    "DNA_origami":        1.5,    # Thermal fluctuations of staple positions
    "MOF":                0.005,  # Crystallographic precision
    "crystal":            0.005,
    "ion_exchange_resin": 0.5,    # Amorphous, statistical
}

# ═══════════════════════════════════════════════════════════════════════════
# DNA origami structural constants — B-form DNA parameters
# ═══════════════════════════════════════════════════════════════════════════

DNA_BP_RISE_NM = 0.34              # nm per base pair
DNA_HELIX_DIAMETER_NM = 2.25       # nm
DNA_BP_PER_TURN = 10.5             # base pairs per helical turn
DNA_PERSISTENCE_LENGTH_NM = 50.0   # nm (dsDNA)
DNA_CROSSOVER_SPACING_BP = 16      # typical crossover spacing in origami

# Staple-positioned functional group precision
DNA_STAPLE_POSITION_SIGMA_NM = 1.5  # ± 1-2 nm, use 1.5 as σ

# ═══════════════════════════════════════════════════════════════════════════
# Protein structural constants
# ═══════════════════════════════════════════════════════════════════════════

PROTEIN_CA_CA_DISTANCE_NM = 0.38       # Cα-Cα distance along backbone
PROTEIN_EXTENDED_SPAN_PER_RESIDUE = 0.38   # nm, fully extended
# Random coil: d ≈ 0.38 × sqrt(n) nm

# Side chain donor reach from Cα (nm, approximate maximum)
SIDECHAIN_DONOR_REACH_NM = {
    "Asp": 0.3,    # Oδ from Cα
    "Glu": 0.4,    # Oε from Cα
    "His": 0.4,    # Nε2 from Cα
    "Cys": 0.3,    # Sγ from Cα
    "Ser": 0.25,   # Oγ from Cα
    "Thr": 0.25,   # Oγ1 from Cα
    "Tyr": 0.6,    # OH from Cα
    "Met": 0.5,    # Sδ from Cα
    "Lys": 0.65,   # Nζ from Cα
    "Arg": 0.7,    # NH from Cα
}

# Map donor subtypes to amino acids that provide them
DONOR_TO_RESIDUE = {
    "O_carboxylate": ["Asp", "Glu"],
    "O_hydroxyl":    ["Ser", "Thr", "Tyr"],
    "O_carbonyl":    ["Asn", "Gln"],   # Backbone also provides this
    "N_imidazole":   ["His"],
    "N_amine":       ["Lys"],
    "N_amide":       ["Asn", "Gln"],
    "S_thiolate":    ["Cys"],
    "S_thioether":   ["Met"],
}

# ═══════════════════════════════════════════════════════════════════════════
# Crown ether cavity diameters (nm) — Pedersen/Izatt compilations
# ═══════════════════════════════════════════════════════════════════════════

CROWN_ETHER_CAVITIES_NM = {
    "12-crown-4": 0.12,
    "15-crown-5": 0.17,
    "18-crown-6": 0.26,
    "21-crown-7": 0.34,
    "24-crown-8": 0.46,
}

# ═══════════════════════════════════════════════════════════════════════════
# MOF node geometry archetypes — from published crystal structures
# ═══════════════════════════════════════════════════════════════════════════

MOF_NODE_GEOMETRIES = {
    # SBU: (coordination_number, geometry_label, typical_pore_nm)
    "Zr6_oxo_cluster":    (12, "cuboctahedral",   0.6),   # UiO-66 family
    "Zn4O_cluster":       (6,  "octahedral",      1.2),   # MOF-5 / IRMOF family
    "Cu2_paddlewheel":    (4,  "square_planar",   0.9),   # HKUST-1
    "Fe3_oxo_cluster":    (6,  "trigonal_prism",  0.8),   # MIL-88
    "Al_octahedral":      (6,  "octahedral",      0.7),   # MIL-53
    "Cr_octahedral":      (6,  "octahedral",      0.7),   # MIL-101
}