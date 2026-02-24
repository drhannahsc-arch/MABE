"""
core/universal_schema.py — Sprint 37: Universal Binding Event Schema

One dataclass to represent ANY measured binding event:
  - Metal-ligand coordination (EDTA + Cu²⁺)
  - Host-guest inclusion (β-CD + adamantane)
  - Protein-small molecule (trypsin + benzamidine)
  - MIP-template (imprinted polymer + analyte)
  - Aptamer-target (DNA + small molecule)
  - Synthetic receptor-analyte (calixarene + ammonium)

Design: every field needed for energy computation is explicit.
Guest properties auto-computed from SMILES when RDKit available.
Host properties looked up from HOST_REGISTRY for common hosts.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BindingMode(str, Enum):
    """Primary interaction type — determines which energy terms dominate."""
    METAL_COORDINATION = "metal_coordination"
    HOST_GUEST_INCLUSION = "host_guest_inclusion"
    PROTEIN_LIGAND = "protein_ligand"
    MIP_TEMPLATE = "mip_template"
    APTAMER_TARGET = "aptamer_target"
    SYNTHETIC_RECEPTOR = "synthetic_receptor"
    UNKNOWN = "unknown"


class DataSource(str, Enum):
    """Provenance of the experimental measurement."""
    NIST_SRD46 = "nist_srd46"
    MARTELL_SMITH = "martell_smith"
    IUPAC = "iupac"
    PDBBIND = "pdbbind"
    SUPRABANK = "suprabank"
    BINDINGDB = "bindingdb"
    REKHARSKY_INOUE = "rekharsky_inoue_2007"
    BARROW_ISAACS = "barrow_2015"
    IZATT = "izatt_1991"
    CHEMBL = "chembl"
    MANUAL = "manual"
    OTHER = "other"


@dataclass
class UniversalComplex:
    """A single experimentally measured binding event.

    This is the atomic unit of calibration data. Every field serves
    at least one energy term in the predictor.
    """
    # ── Identity ──────────────────────────────────────────────────────
    name: str                              # "β-CD:adamantane" or "Cu-EDTA"
    binding_mode: str = "unknown"          # BindingMode value
    log_Ka_exp: float = 0.0               # Experimental log Ka (association)
    dg_exp_kj: float = 0.0                # -5.71 * log_Ka at 25°C (auto-computed)

    # ── Conditions ────────────────────────────────────────────────────
    temperature_C: float = 25.0
    ionic_strength_M: float = 0.1
    ph: float = 7.0
    solvent: str = "water"

    # ── HOST / RECEPTOR properties ────────────────────────────────────
    # For metals: host = metal ion. For CD: host = cyclodextrin. For protein: host = protein.
    host_name: str = ""                    # "β-cyclodextrin", "Cu2+", "trypsin"
    host_type: str = ""                    # "metal_ion", "cyclodextrin", "cucurbituril",
                                           # "calixarene", "protein", "mip", "aptamer"
    host_charge: int = 0
    cavity_volume_A3: float = 0.0          # 0 for metals/proteins (not cavity hosts)
    cavity_radius_nm: float = 0.0
    is_macrocyclic: bool = False
    is_cage: bool = False
    n_hbond_donors_host: int = 0           # Available on host binding surface
    n_hbond_acceptors_host: int = 0
    n_aromatic_walls: int = 0              # For calixarenes, cyclophanes
    host_pdb_id: str = ""                  # For protein hosts

    # ── GUEST / LIGAND properties ─────────────────────────────────────
    guest_name: str = ""                   # "adamantane", "EDTA", "benzamidine"
    guest_smiles: str = ""                 # For RDKit auto-compute
    guest_charge: int = 0
    # Auto-computed from SMILES (or manual entry):
    guest_volume_A3: float = 0.0
    guest_sasa_total_A2: float = 0.0
    guest_sasa_nonpolar_A2: float = 0.0
    guest_sasa_polar_A2: float = 0.0
    guest_rotatable_bonds: int = 0
    guest_n_hbond_donors: int = 0
    guest_n_hbond_acceptors: int = 0
    guest_n_aromatic_rings: int = 0
    guest_logP: float = 0.0
    guest_mw: float = 0.0

    # ── METAL-SPECIFIC (active only for metal_coordination mode) ──────
    metal_formula: str = ""                # "Cu2+", "Fe3+", ""
    metal_charge: int = 0
    metal_d_electrons: int = 0
    donor_atoms: list = field(default_factory=list)       # ["N","N","O","O"]
    donor_subtypes: list = field(default_factory=list)     # ["N_amine","N_amine",...]
    chelate_rings: int = 0
    ring_sizes: list = field(default_factory=list)         # [5,5,5]
    denticity: int = 0
    n_ligand_molecules: int = 1             # Number of separate ligand molecules
    donor_type: str = ""                   # "hard","soft","borderline","mixed"

    # ── INTERACTION GEOMETRY (from co-crystal/docking or estimated) ────
    n_hbonds_formed: int = 0               # Host-guest H-bonds
    hbond_types: list = field(default_factory=list)  # ["neutral","charge_assisted"]
    n_pi_contacts: int = 0
    pi_contact_types: list = field(default_factory=list)  # ["parallel_pp","cation_pi"]
    packing_coefficient: float = 0.0       # V_guest / V_cavity
    sasa_buried_A2: float = 0.0            # Estimated buried SASA

    # ── CALIBRATION METADATA ──────────────────────────────────────────
    source: str = "manual"                 # DataSource value
    source_id: str = ""                    # ID within source database
    series_id: str = ""                    # Groups related measurements
    phase: str = ""                        # Which back-solve phase uses this
    holdout: bool = False                  # Reserved for validation
    confidence: str = "medium"             # "high" / "medium" / "low"
    notes: str = ""

    # ── SCAFFOLD (for immobilized systems) ────────────────────────────
    scaffold_type: str = "free"
    geometry: str = "octahedral"           # Metal CN geometry / binding site geometry

    def __post_init__(self):
        """Auto-compute dg_exp from log_Ka if not set."""
        if self.log_Ka_exp != 0.0 and self.dg_exp_kj == 0.0:
            self.dg_exp_kj = -5.71 * self.log_Ka_exp
        if self.dg_exp_kj != 0.0 and self.log_Ka_exp == 0.0:
            self.log_Ka_exp = -self.dg_exp_kj / 5.71
        # Auto-compute packing coefficient
        if self.guest_volume_A3 > 0 and self.cavity_volume_A3 > 0 and self.packing_coefficient == 0:
            self.packing_coefficient = self.guest_volume_A3 / self.cavity_volume_A3

    def is_metal(self):
        return self.binding_mode == "metal_coordination" or self.metal_formula != ""

    def is_host_guest(self):
        return self.binding_mode in ("host_guest_inclusion", "synthetic_receptor")

    def is_protein_ligand(self):
        return self.binding_mode == "protein_ligand"

    def has_smiles(self):
        return self.guest_smiles != ""