"""
knowledge/pocket_analyzer.py — Extract protein-side binding descriptors from PDB

Parses a PDB file (or text), identifies the binding pocket by distance
cutoff from ligand/HETATM atoms, and extracts descriptors that fill the
missing protein-side terms in the physics PL scorer.

Descriptors extracted:
  - Counter-charges: Asp/Glu COO⁻, Lys NH3+, Arg guanidinium+, His+, metals
  - Pocket waters: count + H-bond quality classification
  - H-bond donors/acceptors on protein side
  - Pocket SASA estimate (residue count × avg SASA)
  - Aromatic residues (Trp, Tyr, Phe) for π-stacking inventory
  - Metal ions in pocket
  - Net pocket charge
  - Estimated pocket desolvation energy

No BioPython dependency — pure Python PDB parsing.
Handles standard PDB format (ATOM/HETATM/TER records).
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# PDB PARSING (pure Python)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PDBAtom:
    """Minimal PDB atom representation."""
    serial: int
    name: str           # atom name, e.g. " CA ", " OD1"
    resname: str        # residue name, e.g. "ASP", "HOH"
    chain: str
    resseq: int         # residue sequence number
    x: float
    y: float
    z: float
    element: str        # element symbol
    is_hetatm: bool     # True for HETATM records
    bfactor: float = 0.0


def parse_pdb(pdb_text):
    """Parse PDB text into list of PDBAtom objects.

    Handles ATOM and HETATM records in standard PDB format.
    """
    atoms = []
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue

        try:
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resseq = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            bfactor = float(line[60:66].strip()) if len(line) >= 66 else 0.0

            # Element: columns 77-78, or infer from atom name
            if len(line) >= 78:
                element = line[76:78].strip()
            else:
                element = name[0] if name else "C"
                # Fix: "CA" atom name → element "C", not "CA"
                if element in ("1", "2", "3"):
                    element = name[1] if len(name) > 1 else "C"

            atoms.append(PDBAtom(
                serial=serial, name=name, resname=resname,
                chain=chain, resseq=resseq,
                x=x, y=y, z=z, element=element,
                is_hetatm=(record == "HETATM"),
                bfactor=bfactor,
            ))
        except (ValueError, IndexError):
            continue

    return atoms


# ═══════════════════════════════════════════════════════════════════════════
# RESIDUE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# Charged residues at pH 7.4
POSITIVE_RESIDUES = {"LYS", "ARG"}  # His is ~10% protonated at pH 7.4
NEGATIVE_RESIDUES = {"ASP", "GLU"}
AROMATIC_RESIDUES = {"TRP", "TYR", "PHE", "HIS"}

# H-bond donor atoms by residue (sidechain only, backbone handled separately)
SIDECHAIN_HBD = {
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
    "ASN": ["ND2"], "GLN": ["NE2"],
    "LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"],
    "HIS": ["ND1", "NE2"],
    "TRP": ["NE1"],
    "CYS": ["SG"],
}

# H-bond acceptor atoms by residue
SIDECHAIN_HBA = {
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
    "ASN": ["OD1"], "GLN": ["OE1"],
    "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"],
    "MET": ["SD"],
    "CYS": ["SG"],
}

# Common metal ions in PDB
METAL_ELEMENTS = {"ZN", "FE", "CU", "MN", "MG", "CA", "CO", "NI", "NA", "K"}
METAL_CHARGES = {
    "ZN": 2, "FE": 2, "CU": 2, "MN": 2, "MG": 2,
    "CA": 2, "CO": 2, "NI": 2, "NA": 1, "K": 1,
}

# Standard water residue names
WATER_NAMES = {"HOH", "WAT", "H2O", "DOD"}


# ═══════════════════════════════════════════════════════════════════════════
# POCKET ANALYSIS RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PocketDescriptors:
    """Protein-side binding descriptors extracted from PDB pocket."""

    # ── Identification ──
    n_pocket_residues: int = 0
    pocket_residue_names: list = field(default_factory=list)

    # ── Charges ──
    n_positive_residues: int = 0     # Lys, Arg in pocket
    n_negative_residues: int = 0     # Asp, Glu in pocket
    net_charge: int = 0              # positive - negative + metal charges
    counter_charge_pairs: int = 0    # min(positive, negative) salt bridge capacity

    # ── Metals ──
    metal_ions: list = field(default_factory=list)  # list of (element, charge)
    metal_charge_total: int = 0

    # ── Waters ──
    n_waters: int = 0                # crystallographic waters in pocket
    n_waters_unhappy: int = 0        # < 2 protein H-bonds, displaceable
    n_waters_happy: int = 0          # >= 3 protein H-bonds, conserved
    water_displacement_kJ: float = 0.0  # estimated ΔG of displacing all

    # ── H-bond inventory ──
    n_hbd_sidechain: int = 0         # sidechain H-bond donors
    n_hba_sidechain: int = 0         # sidechain H-bond acceptors
    n_hbd_backbone: int = 0          # backbone NH in pocket
    n_hba_backbone: int = 0          # backbone C=O in pocket
    n_hbd_total: int = 0
    n_hba_total: int = 0

    # ── Aromatics ──
    n_trp: int = 0
    n_tyr: int = 0
    n_phe: int = 0
    n_his: int = 0
    n_aromatic_total: int = 0

    # ── Geometry ──
    pocket_sasa_estimate_A2: float = 0.0  # rough estimate from residue count
    burial_fraction_estimate: float = 0.6  # default, refined if possible

    # ── Energetics (estimated) ──
    pocket_desolv_kJ: float = 0.0    # protein-side desolvation
    preorganization_kJ: float = 0.0  # pocket rigidity bonus


# ═══════════════════════════════════════════════════════════════════════════
# POCKET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class PocketAnalyzer:
    """Extract binding pocket descriptors from a PDB structure.

    Usage:
        analyzer = PocketAnalyzer.from_file("2WEJ.pdb")
        # or
        analyzer = PocketAnalyzer.from_text(pdb_text)

        descriptors = analyzer.analyze_pocket(
            ligand_resname="DRZ",  # dorzolamide 3-letter code
            cutoff_A=6.0,
        )
    """

    def __init__(self, atoms):
        """Initialize with list of PDBAtom objects."""
        self.atoms = atoms

    @classmethod
    def from_text(cls, pdb_text):
        return cls(parse_pdb(pdb_text))

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, "r") as f:
            return cls.from_text(f.read())

    def _distance(self, a1, a2):
        return math.sqrt((a1.x-a2.x)**2 + (a1.y-a2.y)**2 + (a1.z-a2.z)**2)

    def _get_ligand_atoms(self, ligand_resname=None):
        """Get ligand atoms (HETATM, non-water, non-metal)."""
        ligand = []
        for a in self.atoms:
            if not a.is_hetatm:
                continue
            if a.resname in WATER_NAMES:
                continue
            if a.element.upper() in METAL_ELEMENTS:
                continue
            if ligand_resname and a.resname != ligand_resname:
                continue
            ligand.append(a)
        return ligand

    def _get_pocket_residues(self, ligand_atoms, cutoff_A=6.0):
        """Find protein residues within cutoff of any ligand atom."""
        pocket_resids = set()
        protein_atoms = [a for a in self.atoms if not a.is_hetatm]

        for pa in protein_atoms:
            if pa.element.upper() == "H":
                continue
            for la in ligand_atoms:
                if self._distance(pa, la) <= cutoff_A:
                    pocket_resids.add((pa.chain, pa.resseq, pa.resname))
                    break

        return pocket_resids

    def _get_pocket_waters(self, ligand_atoms, cutoff_A=5.0):
        """Find water molecules within cutoff of ligand."""
        waters = []
        water_atoms = [a for a in self.atoms if a.resname in WATER_NAMES
                       and a.name in ("O", "OW")]

        for wa in water_atoms:
            for la in ligand_atoms:
                if self._distance(wa, la) <= cutoff_A:
                    waters.append(wa)
                    break

        return waters

    def _get_pocket_metals(self, ligand_atoms, cutoff_A=6.0):
        """Find metal ions within cutoff of ligand."""
        metals = []
        for a in self.atoms:
            if a.element.upper() not in METAL_ELEMENTS:
                continue
            for la in ligand_atoms:
                if self._distance(a, la) <= cutoff_A:
                    el = a.element.upper()
                    charge = METAL_CHARGES.get(el, 2)
                    metals.append((el, charge))
                    break
        return metals

    def _classify_water(self, water_atom, protein_atoms, hb_cutoff=3.2):
        """Classify a water as unhappy/neutral/happy by protein H-bonds."""
        n_hb = 0
        for pa in protein_atoms:
            if pa.element.upper() in ("N", "O", "S"):
                d = self._distance(water_atom, pa)
                if d <= hb_cutoff:
                    n_hb += 1
        if n_hb <= 1:
            return "unhappy"
        elif n_hb >= 3:
            return "happy"
        return "neutral"

    def _count_backbone_hb(self, pocket_resids):
        """Count backbone NH (donor) and C=O (acceptor) in pocket."""
        n_hbd = 0
        n_hba = 0
        for a in self.atoms:
            if a.is_hetatm:
                continue
            key = (a.chain, a.resseq, a.resname)
            if key not in pocket_resids:
                continue
            if a.name == "N":
                n_hbd += 1   # backbone NH
            if a.name == "O":
                n_hba += 1   # backbone C=O
        return n_hbd, n_hba

    def analyze_pocket(self, ligand_resname=None, cutoff_A=6.0,
                       water_cutoff_A=5.0):
        """Full pocket analysis.

        Args:
            ligand_resname: 3-letter code of ligand (None = auto-detect first HETATM)
            cutoff_A: distance cutoff for pocket residues
            water_cutoff_A: distance cutoff for pocket waters

        Returns:
            PocketDescriptors dataclass
        """
        result = PocketDescriptors()

        # Identify ligand
        ligand_atoms = self._get_ligand_atoms(ligand_resname)
        if not ligand_atoms:
            return result

        # Pocket residues
        pocket_resids = self._get_pocket_residues(ligand_atoms, cutoff_A)
        result.n_pocket_residues = len(pocket_resids)
        result.pocket_residue_names = [r[2] for r in pocket_resids]

        # ── Charges ──
        for _, _, resname in pocket_resids:
            if resname in POSITIVE_RESIDUES:
                result.n_positive_residues += 1
            if resname in NEGATIVE_RESIDUES:
                result.n_negative_residues += 1

        # Metals
        metals = self._get_pocket_metals(ligand_atoms, cutoff_A)
        result.metal_ions = metals
        result.metal_charge_total = sum(ch for _, ch in metals)

        # Net charge
        result.net_charge = (result.n_positive_residues
                             - result.n_negative_residues
                             + result.metal_charge_total)

        # Counter-charge pairs (capacity for salt bridges)
        total_pos = result.n_positive_residues + result.metal_charge_total
        total_neg = result.n_negative_residues
        result.counter_charge_pairs = min(total_pos, total_neg)

        # ── Waters ──
        pocket_waters = self._get_pocket_waters(ligand_atoms, water_cutoff_A)
        result.n_waters = len(pocket_waters)

        protein_atoms = [a for a in self.atoms if not a.is_hetatm]
        for wa in pocket_waters:
            cls = self._classify_water(wa, protein_atoms)
            if cls == "unhappy":
                result.n_waters_unhappy += 1
            elif cls == "happy":
                result.n_waters_happy += 1

        # Water displacement energy estimate
        # Unhappy: -5 kJ/mol (favorable to displace)
        # Neutral: 0 kJ/mol
        # Happy: +5 kJ/mol (costly to displace)
        n_neutral = result.n_waters - result.n_waters_unhappy - result.n_waters_happy
        result.water_displacement_kJ = (
            result.n_waters_unhappy * (-5.0)
            + n_neutral * 0.0
            + result.n_waters_happy * (+5.0)
        )

        # ── H-bond inventory ──
        # Sidechain
        for _, _, resname in pocket_resids:
            donors = SIDECHAIN_HBD.get(resname, [])
            acceptors = SIDECHAIN_HBA.get(resname, [])
            result.n_hbd_sidechain += len(donors)
            result.n_hba_sidechain += len(acceptors)

        # Backbone
        bb_hbd, bb_hba = self._count_backbone_hb(pocket_resids)
        result.n_hbd_backbone = bb_hbd
        result.n_hba_backbone = bb_hba
        result.n_hbd_total = result.n_hbd_sidechain + result.n_hbd_backbone
        result.n_hba_total = result.n_hba_sidechain + result.n_hba_backbone

        # ── Aromatics ──
        for _, _, resname in pocket_resids:
            if resname == "TRP":
                result.n_trp += 1
            elif resname == "TYR":
                result.n_tyr += 1
            elif resname == "PHE":
                result.n_phe += 1
            elif resname == "HIS":
                result.n_his += 1
        result.n_aromatic_total = (result.n_trp + result.n_tyr
                                   + result.n_phe + result.n_his)

        # ── Geometry estimates ──
        # Average residue contributes ~40-60 Å² to pocket SASA
        result.pocket_sasa_estimate_A2 = result.n_pocket_residues * 50.0

        # Burial: deeper pockets have lower ε
        # Rough: >15 residues = deeply buried, <8 = shallow
        if result.n_pocket_residues >= 15:
            result.burial_fraction_estimate = 0.85
        elif result.n_pocket_residues >= 10:
            result.burial_fraction_estimate = 0.70
        else:
            result.burial_fraction_estimate = 0.50

        # ── Energy estimates ──
        # Pocket desolvation: protein surface that becomes buried
        # Favorable for nonpolar, costly for polar
        # Approximate: pocket is ~60% nonpolar, 40% polar
        GAMMA_NP = -0.025  # kJ/mol/Å² (hydrophobic, favorable)
        GAMMA_P = +0.050   # kJ/mol/Å² (polar, costly but partially compensated)
        pocket_np = result.pocket_sasa_estimate_A2 * 0.6
        pocket_p = result.pocket_sasa_estimate_A2 * 0.4
        result.pocket_desolv_kJ = GAMMA_NP * pocket_np + GAMMA_P * pocket_p

        # Preorganization bonus: rigid pockets pay less entropy cost
        # Well-organized active sites (enzymes): -10 to -15 kJ/mol
        # Flexible binding grooves: -3 to -5 kJ/mol
        # Estimate from aromatic content (rigid) vs polar sidechain (flexible)
        n_rigid = result.n_aromatic_total + len(
            [r for r in pocket_resids if r[2] in ("PRO", "ALA", "VAL", "ILE", "LEU")]
        )
        f_rigid = n_rigid / max(result.n_pocket_residues, 1)
        result.preorganization_kJ = -(5.0 + 10.0 * f_rigid)  # -5 to -15

        return result


# ═══════════════════════════════════════════════════════════════════════════
# UC POPULATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def populate_uc_from_pocket(uc, descriptors):
    """Populate UniversalComplex fields from PocketDescriptors.

    Fills protein-side fields that the SMILES-only path cannot compute.
    """
    d = descriptors

    # H-bond inventory
    uc.n_hbond_donors_host = d.n_hbd_total
    uc.n_hbond_acceptors_host = d.n_hba_total

    # Counter-charges
    uc.host_charge = d.net_charge
    uc.n_salt_bridges = d.counter_charge_pairs

    # Cavity
    if d.pocket_sasa_estimate_A2 > 0:
        uc.cavity_volume_A3 = d.pocket_sasa_estimate_A2 * 2.5  # rough V~SASA×depth

    # Metals
    if d.metal_ions:
        uc.metal_formula = d.metal_ions[0][0]  # first metal element

    # Burial
    # Don't override if already set from structural data
    if uc.sasa_buried_A2 == 0 and d.pocket_sasa_estimate_A2 > 0:
        uc.sasa_buried_A2 = d.pocket_sasa_estimate_A2 * 0.5  # ~half interface

    return uc