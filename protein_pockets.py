"""
protein_pockets.py -- Active site descriptors for ChEMBL targets.

Sources:
  CA-II:   Krishnamurthy et al., Chem Rev 108, 946 (2008); PDB 1CA2
  Thrombin: Stubbs & Bode, Thromb Res 69, 1 (1993); PDB 1PPB
  Trypsin:  Perona & Craik, Protein Sci 4, 337 (1995); PDB 1TLD
  HIV-PR:  Wlodawer & Vondrasek, Ann Rev Biophys 27, 249 (1998); PDB 1HVR
  COX-2:   Kurumbail et al., Nature 384, 644 (1996); PDB 6COX
  DHFR:    Bolin et al., J Biol Chem 257, 13650 (1982); PDB 1DRF

Pocket volumes from CASTp/fpocket literature.
H-bond sites from co-crystal analysis.
Hydrophobic fraction from buried surface decomposition.
"""

from dataclasses import dataclass


@dataclass
class ProteinPocket:
    """Active site descriptors for energy term estimation."""
    name: str
    pocket_volume_A3: float        # Active site volume (CASTp-style)
    pocket_depth_A: float          # Approximate depth of binding cleft
    n_hbd_pocket: int              # H-bond donors available in pocket
    n_hba_pocket: int              # H-bond acceptors available in pocket
    n_aromatic_residues: int       # Phe, Tyr, Trp, His in binding site
    hydrophobic_fraction: float    # Fraction of pocket surface that's hydrophobic
    net_charge_pocket: int         # Net charge of binding site residues
    sasa_burial_fraction: float    # Typical fraction of ligand SASA buried
    notes: str = ""


PROTEIN_POCKETS = {
    # -- Carbonic Anhydrase II -----------------------------------------
    # Conical pocket ~15A deep, Zn2? at bottom. Hydrophobic wall.
    # Key: His94/96/119 (Zn), Thr199 (gatekeeper HB), Leu198, Val121, Phe131
    "Carbonic anhydrase II": ProteinPocket(
        name="Carbonic anhydrase II",
        pocket_volume_A3=310,       # Conical pocket, medium size
        pocket_depth_A=15.0,
        n_hbd_pocket=4,             # Thr199-OH, Thr200-OH, backbone NH
        n_hba_pocket=6,             # Hisx3, Glu106, Thr199-OG, backbone C=O
        n_aromatic_residues=3,      # Phe131, His94/96/119 (His = aromatic)
        hydrophobic_fraction=0.55,  # Hydrophobic wall: Val121, Leu198, Trp209
        net_charge_pocket=2,        # Zn2? dominant
        sasa_burial_fraction=0.75,  # Deep conical pocket, good burial
    ),

    # -- Thrombin ------------------------------------------------------
    # S1 specificity pocket (Asp189), S2 hydrophobic pocket, catalytic triad
    "Thrombin": ProteinPocket(
        name="Thrombin",
        pocket_volume_A3=450,       # Multi-subsite active site
        pocket_depth_A=12.0,
        n_hbd_pocket=5,             # Gly216-NH, Ser195-OH, backbone NHs
        n_hba_pocket=7,             # Asp189-COO, His57, Ser195, backbone C=Os
        n_aromatic_residues=4,      # Trp215, His57, Tyr228, Phe227
        hydrophobic_fraction=0.50,  # Mixed: S1=charged, S2/S3=hydrophobic
        net_charge_pocket=-1,       # Asp189 dominant
        sasa_burial_fraction=0.65,  # Extended cleft, partial exposure
    ),

    # -- Trypsin -------------------------------------------------------
    # Simple S1 pocket with Asp189 specificity (Arg/Lys selective)
    "Trypsin": ProteinPocket(
        name="Trypsin",
        pocket_volume_A3=350,       # Narrower than thrombin
        pocket_depth_A=10.0,
        n_hbd_pocket=4,             # Gly216-NH, Ser195-OH, Gly193-NH
        n_hba_pocket=5,             # Asp189-COO, His57, Ser195, backbone C=O
        n_aromatic_residues=2,      # His57, Trp215
        hydrophobic_fraction=0.40,  # Charged S1 pocket
        net_charge_pocket=-1,       # Asp189
        sasa_burial_fraction=0.60,  # Shallow groove
    ),

    # -- HIV-1 Protease ------------------------------------------------
    # Homodimeric, large symmetric active site, 2 Asp catalytic
    "HIV-1 protease": ProteinPocket(
        name="HIV-1 protease",
        pocket_volume_A3=650,       # Large: accommodates peptidomimetics
        pocket_depth_A=10.0,
        n_hbd_pocket=6,             # Backbone NHs across both monomers, flap NHs
        n_hba_pocket=8,             # Asp25/25' COO, backbone C=Os, Gly27/27'
        n_aromatic_residues=4,      # Phe53/53', Pro81/81' (near-aromatic)
        hydrophobic_fraction=0.60,  # Large hydrophobic surface, S1/S1' pockets
        net_charge_pocket=-2,       # Asp25 + Asp25'
        sasa_burial_fraction=0.70,  # Flap closure buries ligand well
    ),

    # -- COX-2 ---------------------------------------------------------
    # Long hydrophobic channel + side pocket (COX-2 selectivity)
    "COX-2": ProteinPocket(
        name="COX-2",
        pocket_volume_A3=520,       # Channel + side pocket (larger than COX-1)
        pocket_depth_A=18.0,
        n_hbd_pocket=3,             # Tyr355-OH, Arg120-NH, Ser530-OH
        n_hba_pocket=5,             # Arg120, Glu524, His90, backbone C=Os
        n_aromatic_residues=5,      # Tyr355, Trp387, Phe518, Tyr385, Phe381
        hydrophobic_fraction=0.65,  # Predominantly hydrophobic channel
        net_charge_pocket=0,        # Mixed: Arg120 + Glu524 cancel
        sasa_burial_fraction=0.80,  # Deep channel, excellent burial
    ),

    # -- Dihydrofolate Reductase ---------------------------------------
    # Well-defined pocket for folate/NADPH, polar rim + hydrophobic floor
    "Dihydrofolate reductase": ProteinPocket(
        name="Dihydrofolate reductase",
        pocket_volume_A3=400,
        pocket_depth_A=12.0,
        n_hbd_pocket=5,             # Arg70-NH, backbone NHs, water-mediated
        n_hba_pocket=7,             # Asp27-COO, Glu30, backbone C=Os, Thr136
        n_aromatic_residues=3,      # Phe31, Phe34, Trp24
        hydrophobic_fraction=0.50,  # Mixed: hydrophobic floor, polar rim
        net_charge_pocket=-1,       # Asp27 critical for proton relay
        sasa_burial_fraction=0.70,  # Good burial, folate analog fits well
    ),
}


def annotate_protein_pocket(uc):
    """Populate UniversalComplex host-side fields from pocket descriptors.
    
    Estimates n_hbonds_formed, n_pi_contacts, cavity_volume, and
    adjusts SASA burial for pocket-specific hydrophobicity.
    """
    pocket = PROTEIN_POCKETS.get(uc.host_name)
    if pocket is None:
        return uc

    # Set pocket volume as "cavity" for shape/packing terms
    if uc.cavity_volume_A3 == 0:
        uc.cavity_volume_A3 = pocket.pocket_volume_A3
    if uc.cavity_radius_nm == 0:
        # Approximate: V = (4/3)pir3 -> r = (3V/4pi)^(1/3)
        import math
        r_A = (3 * pocket.pocket_volume_A3 / (4 * math.pi)) ** (1/3)
        uc.cavity_radius_nm = r_A / 10.0

    # Compute packing coefficient now that both volumes are set
    # For protein-ligand: only the buried fraction of ligand volume fills the pocket
    if uc.guest_volume_A3 > 0 and uc.cavity_volume_A3 > 0 and uc.packing_coefficient == 0:
        if uc.binding_mode == "protein_ligand":
            # Effective guest volume = fraction buried in pocket
            effective_vol = uc.guest_volume_A3 * pocket.sasa_burial_fraction
            uc.packing_coefficient = effective_vol / uc.cavity_volume_A3
        else:
            uc.packing_coefficient = uc.guest_volume_A3 / uc.cavity_volume_A3

    # Host H-bond sites
    uc.n_hbond_donors_host = pocket.n_hbd_pocket
    uc.n_hbond_acceptors_host = pocket.n_hba_pocket

    # Estimate H-bonds formed: complementary pairing
    # Guest donors pair with pocket acceptors, guest acceptors pair with pocket donors
    hb_from_donors = min(uc.guest_n_hbond_donors, pocket.n_hba_pocket)
    hb_from_acceptors = min(uc.guest_n_hbond_acceptors, pocket.n_hbd_pocket)
    # Geometric satisfaction rate depends on pocket depth and guest flexibility
    # Deeper pockets -> better geometric match -> higher satisfaction
    # More rotatable bonds -> more conformational search -> lower satisfaction
    n_rotors = uc.guest_rotatable_bonds
    depth_factor = min(1.0, pocket.pocket_depth_A / 15.0)  # Normalize to 1.0
    flex_penalty = max(0.3, 1.0 - 0.04 * n_rotors)  # Flexible guests lose H-bonds
    satisfaction = 0.45 * depth_factor * flex_penalty
    estimated_hbonds = max(1, int(round((hb_from_donors + hb_from_acceptors) * satisfaction)))
    if uc.n_hbonds_formed == 0:
        uc.n_hbonds_formed = estimated_hbonds
        # Classify H-bond types based on charge state
        if uc.guest_charge != 0 or pocket.net_charge_pocket != 0:
            # At least one charge-assisted H-bond
            n_charged = min(abs(uc.guest_charge) + abs(pocket.net_charge_pocket), estimated_hbonds)
            uc.hbond_types = (["charge_assisted"] * n_charged +
                              ["neutral"] * (estimated_hbonds - n_charged))
        else:
            uc.hbond_types = ["neutral"] * estimated_hbonds

    # Estimate pi contacts from aromatic complementarity
    if uc.n_pi_contacts == 0 and pocket.n_aromatic_residues > 0:
        if uc.guest_n_aromatic_rings > 0:
            uc.n_pi_contacts = min(uc.guest_n_aromatic_rings, pocket.n_aromatic_residues)

    # Aromatic walls -> enables pi term in predictor
    if uc.n_aromatic_walls == 0:
        uc.n_aromatic_walls = pocket.n_aromatic_residues

    # Host charge
    if uc.host_charge == 0:
        uc.host_charge = pocket.net_charge_pocket

    # Adjust SASA burial based on pocket-specific fraction
    if uc.guest_sasa_nonpolar_A2 > 0 and uc.sasa_buried_A2 == 0:
        # Modulate by pocket hydrophobic fraction
        uc.sasa_buried_A2 = round(
            uc.guest_sasa_nonpolar_A2 * pocket.sasa_burial_fraction * pocket.hydrophobic_fraction
            + uc.guest_sasa_polar_A2 * pocket.sasa_burial_fraction * (1 - pocket.hydrophobic_fraction),
            1
        )

    return uc
