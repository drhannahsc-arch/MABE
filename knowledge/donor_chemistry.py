"""
knowledge/donor_chemistry.py - Donor atom and ligand properties database.
Maps abstract donor specifications to concrete chemical implementations.
"""

from dataclasses import dataclass, field

@dataclass
class DonorAtomProperties:
    symbol: str
    hsab_class: str
    typical_bond_length_A: dict
    field_strength: str
    charge_preference: str

@dataclass
class LigandTemplate:
    name: str
    donor_atom: str
    donor_count: int
    hsab_softness: float
    pka_coordinating: float
    typical_bond_length_A: float
    field_strength_dq: float
    charge_when_bound: int
    steric_bulk: str
    synthetic_accessibility: float
    ph_stable_range: tuple
    functional_group: str
    smarts: str
    notes: str = ""

DONOR_ATOMS = {
    "N": DonorAtomProperties("N", "borderline",
        {"divalent_first_row": 2.05, "divalent_second_row": 2.20,
         "divalent_third_row": 2.25, "trivalent": 2.00},
        "moderate_to_strong", "neutral"),
    "O": DonorAtomProperties("O", "hard",
        {"divalent_first_row": 2.10, "divalent_second_row": 2.30,
         "divalent_third_row": 2.35, "trivalent": 1.95},
        "weak_to_moderate", "anionic"),
    "S": DonorAtomProperties("S", "soft",
        {"divalent_first_row": 2.30, "divalent_second_row": 2.45,
         "divalent_third_row": 2.50, "trivalent": 2.25},
        "moderate", "anionic"),
    "P": DonorAtomProperties("P", "soft",
        {"divalent_first_row": 2.35, "divalent_second_row": 2.50,
         "divalent_third_row": 2.55, "trivalent": 2.30},
        "strong", "neutral"),
}

LIGAND_TEMPLATES = [
    LigandTemplate("imidazole", "N", 1, 0.45, 14.5, 2.00, 1.25, 0,
        "moderate", 0.9, (2.0, 12.0), "imine", "[nR1]1cc[nH]c1",
        "Histidine sidechain analog. Strong field, borderline soft."),
    LigandTemplate("pyridine", "N", 1, 0.40, 5.25, 2.02, 1.25, 0,
        "moderate", 0.95, (1.0, 13.0), "imine", "n1ccccc1",
        "Classic borderline N donor."),
    LigandTemplate("primary_amine", "N", 1, 0.35, 10.5, 2.05, 1.25, 0,
        "minimal", 0.95, (0.0, 14.0), "amine", "[NH2]",
        "Must deprotonate to coordinate. pKa ~10.5."),
    LigandTemplate("tertiary_amine", "N", 1, 0.30, 10.0, 2.10, 1.15, 0,
        "moderate", 0.85, (0.0, 14.0), "amine", "[N;X3;H0]",
        "Backbone amine in polyamines like en, dien, trien."),
    LigandTemplate("bipyridyl", "N", 2, 0.45, 4.3, 2.00, 1.33, 0,
        "moderate", 0.80, (1.0, 13.0), "imine", "c1ccnc(-c2ccccn2)c1",
        "Bidentate N,N donor. Strong field."),
    LigandTemplate("hydroxamate", "N", 1, 0.25, 8.5, 2.05, 1.0, -1,
        "minimal", 0.75, (3.0, 12.0), "hydroxamate", "[OH]NC=O",
        "Hard donor, strong for Fe3+. Siderophore motif."),
    LigandTemplate("carboxylate", "O", 2, 0.15, 4.5, 2.10, 0.85, -1,
        "minimal", 0.95, (5.0, 14.0), "carboxylate", "[O-]C=O",
        "Hard O,O bidentate. EDTA/DTPA backbone. Requires pH > pKa."),
    LigandTemplate("phenolate", "O", 1, 0.20, 10.0, 1.95, 0.90, -1,
        "moderate", 0.85, (7.0, 14.0), "phenol", "[O-]c1ccccc1",
        "Harder than thiolate but softer than carboxylate."),
    LigandTemplate("catechol", "O", 2, 0.25, 9.2, 2.00, 0.95, -2,
        "moderate", 0.75, (6.0, 14.0), "diol", "[OH]c1ccccc1[OH]",
        "Bidentate O,O. Very strong for Fe3+, Ti4+. Siderophore motif."),
    LigandTemplate("phosphonate", "O", 2, 0.10, 6.5, 2.15, 0.80, -2,
        "moderate", 0.70, (4.0, 14.0), "phosphonate", "[O-]P([O-])=O",
        "Very hard donor. Good for UO2, lanthanides, actinides."),
    LigandTemplate("crown_ether_O", "O", 1, 0.15, 99.0, 2.30, 0.60, 0,
        "bulky", 0.60, (0.0, 14.0), "ether", "[OD2]([CH2])[CH2]",
        "Neutral O donor. Size-selective. 18-crown-6 for K+."),
    LigandTemplate("water", "O", 1, 0.10, 15.7, 2.10, 1.0, 0,
        "minimal", 1.0, (0.0, 14.0), "aqua", "[OH2]",
        "Reference ligand. Displaced during coordination."),
    LigandTemplate("thiolate", "S", 1, 0.85, 8.3, 2.30, 1.0, -1,
        "minimal", 0.85, (6.0, 14.0), "thiol", "[S-]",
        "Classic soft donor. Cysteine sidechain. Strong for Au, Hg, Ag, Cd."),
    LigandTemplate("thioether", "S", 1, 0.70, 99.0, 2.35, 0.90, 0,
        "moderate", 0.80, (0.0, 14.0), "thioether", "[SD2]([CH2])[CH2]",
        "Neutral soft donor. Methionine sidechain."),
    LigandTemplate("dithiocarbamate", "S", 2, 0.80, 3.0, 2.35, 0.95, -1,
        "moderate", 0.70, (4.0, 12.0), "dithiocarbamate", "[S-]C(=S)N",
        "Bidentate S,S donor. Very strong for soft metals."),
    LigandTemplate("thiourea", "S", 1, 0.75, 99.0, 2.40, 0.85, 0,
        "moderate", 0.85, (1.0, 13.0), "thioamide", "NC(=S)N",
        "Neutral S donor. Gold leaching agent."),
    LigandTemplate("iminodiacetate", "N", 1, 0.25, 2.5, 2.05, 1.0, -2,
        "moderate", 0.85, (3.0, 13.0), "aminocarboxylate",
        "[NH](CC([O-])=O)CC([O-])=O",
        "NTA/IDA motif. Tridentate. EDTA-type chelator backbone."),
    LigandTemplate("salicylaldehyde_imine", "N", 1, 0.35, 12.0, 2.00, 1.15, -1,
        "moderate", 0.80, (4.0, 13.0), "imine", "N=Cc1ccccc1[O-]",
        "Salen/saloph motif. Bidentate N,O. Strong planar field for d8."),
]

def get_ligands_by_donor(donor_atom):
    return [t for t in LIGAND_TEMPLATES if t.donor_atom == donor_atom]

def get_ligands_by_hsab(target_softness, tolerance=0.25):
    return [t for t in LIGAND_TEMPLATES if abs(t.hsab_softness - target_softness) <= tolerance]

def get_ligands_stable_at_ph(ph):
    return [t for t in LIGAND_TEMPLATES if t.ph_stable_range[0] <= ph <= t.ph_stable_range[1]]

def get_ligands_matching(donor_atom, ph, target_softness=None, tolerance=0.25):
    results = []
    for t in LIGAND_TEMPLATES:
        if t.donor_atom != donor_atom:
            continue
        if not (t.ph_stable_range[0] <= ph <= t.ph_stable_range[1]):
            continue
        if target_softness is not None:
            if abs(t.hsab_softness - target_softness) > tolerance:
                continue
        results.append(t)
    return results


