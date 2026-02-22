"""
core/host_registry.py — Sprint 37: Pre-annotated host properties

Cavity volumes, radii, H-bond sites, aromatic walls for ~25 common hosts.
Data from crystal structures and published compilations.

Sources:
  - Szejtli, Cyclodextrin Technology (1988)
  - Rekharsky & Inoue, Chem. Rev. 107, 3715 (2007)
  - Barrow et al., Chem. Rev. 115, 12320 (2015)
  - Atwood et al., Comprehensive Supramolecular Chemistry (1996)
  - Mecozzi & Rebek, Chem. Eur. J. 4, 1016 (1998)
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class HostProperties:
    """Pre-computed properties for a known host molecule."""
    name: str
    host_type: str                      # "cyclodextrin", "cucurbituril", etc.
    cavity_volume_A3: float             # Interior cavity volume
    cavity_radius_nm: float             # Approximate cavity radius
    cavity_depth_A: float               # Cavity depth (height)
    is_macrocyclic: bool
    is_cage: bool                       # 3D encapsulation (cryptand, CB)
    n_hbond_donors: int                 # On binding surface / portals
    n_hbond_acceptors: int
    n_aromatic_walls: int               # Aromatic panels in cavity
    host_charge: int                    # Net formal charge
    portal_diameter_nm: float           # 0 if open cavity
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# CYCLODEXTRINS
# Volumes: Szejtli 1988, Connors 1997
# ═══════════════════════════════════════════════════════════════════════════

HOST_REGISTRY = {
    # ── Cyclodextrins ─────────────────────────────────────────────────
    "alpha-CD": HostProperties(
        "α-cyclodextrin", "cyclodextrin",
        cavity_volume_A3=174, cavity_radius_nm=0.235, cavity_depth_A=7.9,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=18, n_hbond_acceptors=30,  # 6 glucose × 3 OH donor, 5 O acceptor
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.47,
        notes="6 glucose units, narrow cavity"),
    "beta-CD": HostProperties(
        "β-cyclodextrin", "cyclodextrin",
        cavity_volume_A3=262, cavity_radius_nm=0.300, cavity_depth_A=7.9,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=21, n_hbond_acceptors=35,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.60,
        notes="7 glucose units, most common CD"),
    "gamma-CD": HostProperties(
        "γ-cyclodextrin", "cyclodextrin",
        cavity_volume_A3=427, cavity_radius_nm=0.375, cavity_depth_A=7.9,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=24, n_hbond_acceptors=40,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.75,
        notes="8 glucose units, large cavity"),
    "HP-beta-CD": HostProperties(
        "HP-β-cyclodextrin", "cyclodextrin",
        cavity_volume_A3=280, cavity_radius_nm=0.310, cavity_depth_A=7.9,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=21, n_hbond_acceptors=42,  # Extra O from hydroxypropyl
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.62,
        notes="Hydroxypropyl-β-CD, better solubility"),

    # ── Cucurbiturils ─────────────────────────────────────────────────
    # Volumes: Barrow et al. 2015, Isaacs group
    "CB5": HostProperties(
        "cucurbit[5]uril", "cucurbituril",
        cavity_volume_A3=82, cavity_radius_nm=0.220, cavity_depth_A=9.1,
        is_macrocyclic=True, is_cage=True,  # Portals restrict entry → cage-like
        n_hbond_donors=0, n_hbond_acceptors=10,  # 5 C=O per portal × 2 portals
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.24,
        notes="5 glycoluril units, very small cavity"),
    "CB6": HostProperties(
        "cucurbit[6]uril", "cucurbituril",
        cavity_volume_A3=164, cavity_radius_nm=0.290, cavity_depth_A=9.1,
        is_macrocyclic=True, is_cage=True,
        n_hbond_donors=0, n_hbond_acceptors=12,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.39,
        notes="6 glycoluril, binds alkylammonium"),
    "CB7": HostProperties(
        "cucurbit[7]uril", "cucurbituril",
        cavity_volume_A3=279, cavity_radius_nm=0.365, cavity_depth_A=9.1,
        is_macrocyclic=True, is_cage=True,
        n_hbond_donors=0, n_hbond_acceptors=14,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.54,
        notes="7 glycoluril, Ka up to 10^15 for ferrocenemethyl-NMe3+"),
    "CB8": HostProperties(
        "cucurbit[8]uril", "cucurbituril",
        cavity_volume_A3=479, cavity_radius_nm=0.440, cavity_depth_A=9.1,
        is_macrocyclic=True, is_cage=True,
        n_hbond_donors=0, n_hbond_acceptors=16,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.69,
        notes="8 glycoluril, can bind ternary complexes"),

    # ── Calixarenes ───────────────────────────────────────────────────
    "calix4arene": HostProperties(
        "calix[4]arene", "calixarene",
        cavity_volume_A3=100, cavity_radius_nm=0.250, cavity_depth_A=5.0,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=4, n_hbond_acceptors=4,  # 4 OH at lower rim
        n_aromatic_walls=4, host_charge=0, portal_diameter_nm=0.30,
        notes="4 phenol units, cation-π binding"),
    "sulfonato-calix4arene": HostProperties(
        "p-sulfonatocalix[4]arene", "calixarene",
        cavity_volume_A3=100, cavity_radius_nm=0.250, cavity_depth_A=5.0,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=4, n_hbond_acceptors=16,  # 4 SO3⁻ + 4 OH
        n_aromatic_walls=4, host_charge=-4, portal_diameter_nm=0.30,
        notes="Water-soluble, strong cation binding"),

    # ── Pillar[n]arenes ───────────────────────────────────────────────
    "pillar5arene": HostProperties(
        "pillar[5]arene", "pillararene",
        cavity_volume_A3=95, cavity_radius_nm=0.230, cavity_depth_A=5.4,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=0, n_hbond_acceptors=10,  # Ether O atoms
        n_aromatic_walls=5, host_charge=0, portal_diameter_nm=0.46,
        notes="Electron-rich tubular cavity"),

    # ── Crown ethers ──────────────────────────────────────────────────
    "12-crown-4": HostProperties(
        "12-crown-4", "crown_ether",
        cavity_volume_A3=30, cavity_radius_nm=0.060, cavity_depth_A=1.5,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=0, n_hbond_acceptors=4,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.0,
        notes="Li+ selective"),
    "15-crown-5": HostProperties(
        "15-crown-5", "crown_ether",
        cavity_volume_A3=50, cavity_radius_nm=0.086, cavity_depth_A=1.5,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=0, n_hbond_acceptors=5,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.0,
        notes="Na+ selective"),
    "18-crown-6": HostProperties(
        "18-crown-6", "crown_ether",
        cavity_volume_A3=80, cavity_radius_nm=0.134, cavity_depth_A=1.5,
        is_macrocyclic=True, is_cage=False,
        n_hbond_donors=0, n_hbond_acceptors=6,
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.0,
        notes="K+ selective, classic size-match"),

    # ── Cryptands ─────────────────────────────────────────────────────
    "cryptand-222": HostProperties(
        "[2.2.2]cryptand", "cryptand",
        cavity_volume_A3=90, cavity_radius_nm=0.140, cavity_depth_A=4.0,
        is_macrocyclic=True, is_cage=True,
        n_hbond_donors=0, n_hbond_acceptors=8,  # 6 O + 2 N
        n_aromatic_walls=0, host_charge=0, portal_diameter_nm=0.20,
        notes="3D K+ binding, cryptate effect"),
}

# Aliases for common name variants
_ALIASES = {
    "α-CD": "alpha-CD", "a-CD": "alpha-CD", "alpha-cyclodextrin": "alpha-CD",
    "β-CD": "beta-CD", "b-CD": "beta-CD", "beta-cyclodextrin": "beta-CD",
    "γ-CD": "gamma-CD", "g-CD": "gamma-CD", "gamma-cyclodextrin": "gamma-CD",
    "HP-β-CD": "HP-beta-CD", "hydroxypropyl-beta-CD": "HP-beta-CD",
    "CB[5]": "CB5", "CB[6]": "CB6", "CB[7]": "CB7", "CB[8]": "CB8",
    "cucurbit[5]uril": "CB5", "cucurbit[6]uril": "CB6",
    "cucurbit[7]uril": "CB7", "cucurbit[8]uril": "CB8",
    "calix[4]arene": "calix4arene",
    "p-sulfonatocalix[4]arene": "sulfonato-calix4arene",
    "pillar[5]arene": "pillar5arene",
    "[2.2.2]cryptand": "cryptand-222",
}


def lookup_host(name):
    """Look up host properties by name. Returns HostProperties or None."""
    key = _ALIASES.get(name, name)
    return HOST_REGISTRY.get(key, None)


def enrich_complex_host(uc):
    """Fill host properties on a UniversalComplex from HOST_REGISTRY.

    Modifies uc in-place. Only fills fields still at default.
    """
    host = lookup_host(uc.host_name)
    if host is None:
        return uc

    if uc.cavity_volume_A3 == 0: uc.cavity_volume_A3 = host.cavity_volume_A3
    if uc.cavity_radius_nm == 0: uc.cavity_radius_nm = host.cavity_radius_nm
    if not uc.is_macrocyclic: uc.is_macrocyclic = host.is_macrocyclic
    if not uc.is_cage: uc.is_cage = host.is_cage
    if uc.n_hbond_donors_host == 0: uc.n_hbond_donors_host = host.n_hbond_donors
    if uc.n_hbond_acceptors_host == 0: uc.n_hbond_acceptors_host = host.n_hbond_acceptors
    if uc.n_aromatic_walls == 0: uc.n_aromatic_walls = host.n_aromatic_walls
    if uc.host_charge == 0: uc.host_charge = host.host_charge
    if uc.host_type == "": uc.host_type = host.host_type

    # Recompute packing coefficient now that cavity volume is known
    if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0:
        uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3

    return uc
