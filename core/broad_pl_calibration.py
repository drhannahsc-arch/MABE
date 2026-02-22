"""
core/broad_pl_calibration.py -- Sprint 40b: Broad Protein-Ligand Calibration

Trained on BindingDB Articles subset: 14,424 entries across 98 targets.

Model:
  1. Predict with S38 universal params + generic/class pocket descriptors
  2. Apply per-target offset (98 known targets) OR class offset (10 classes)

Performance:
  Known targets (per-target offset): MAE = 1.17 log K
  Unseen targets (class offset):     MAE = 1.46-1.51 log K (cross-validated)
  Unseen targets (global offset):    MAE = 1.52-1.65 log K
"""

import re
import math
from protein_pockets import ProteinPocket


# =========================================================================
# TARGET CLASSIFICATION
# =========================================================================

_CLASS_PATTERNS = {
    'carbonic_anhydrase': r'[Cc]arbonic anhydrase',
    'serine_protease': r'[Cc]oagulation factor|[Pp]rothrombin|[Tt]rypsin|[Ss]erine protease|[Uu]rokinase|[Pp]lasminogen',
    'aspartyl_protease': r'[Gg]ag-Pol|HIV.*protease|[Pp]lasmepsin|[Cc]athepsin D|[Pp]epsin|[Rr]enin',
    'cysteine_protease': r'[Cc]athepsin [BFHKLSZ]|[Pp]rocathepsin|[Cc]aspase|[Cc]alpain',
    'metalloprotease': r'[Mm]etalloprotein|[Cc]ollagenase|[Ss]tromelysin|kDa type IV',
    'kinase': r'[Kk]inase',
    'gpcr': r'(?:opioid|hydroxytryptamine|[Aa]denosine|[Cc]annabinoid|[Nn]ociceptin|[Mm]uscarinic|[Dd]opamine|Mu-type|Delta-type|Kappa).*receptor|receptor.*(?:opioid|hydroxytryptamine|adenosine|cannabinoid|nociceptin|muscarinic|dopamine)',
    'metalloenzyme': r'[Aa]mine oxidase|[Aa]romatase|[Aa]minopeptidase',
    'viral_other': r'[Rr]eplicase|[Gg]enome poly',
    'phosphatase': r'[Pp]hosphatase',
}


def classify_target(name):
    """Classify a protein target by name into a structural class."""
    for cls, pattern in _CLASS_PATTERNS.items():
        if re.search(pattern, name):
            return cls
    return 'other'


# =========================================================================
# CLASS-SPECIFIC POCKET DESCRIPTORS
# =========================================================================

CLASS_POCKETS = {
    'carbonic_anhydrase': ProteinPocket(
        name='carbonic_anhydrase', pocket_volume_A3=310, pocket_depth_A=15.0,
        n_hbd_pocket=4, n_hba_pocket=6, n_aromatic_residues=3,
        hydrophobic_fraction=0.55, net_charge_pocket=2, sasa_burial_fraction=0.75),
    'serine_protease': ProteinPocket(
        name='serine_protease', pocket_volume_A3=400, pocket_depth_A=11.0,
        n_hbd_pocket=5, n_hba_pocket=6, n_aromatic_residues=3,
        hydrophobic_fraction=0.45, net_charge_pocket=-1, sasa_burial_fraction=0.65),
    'aspartyl_protease': ProteinPocket(
        name='aspartyl_protease', pocket_volume_A3=600, pocket_depth_A=10.0,
        n_hbd_pocket=6, n_hba_pocket=8, n_aromatic_residues=4,
        hydrophobic_fraction=0.55, net_charge_pocket=-2, sasa_burial_fraction=0.70),
    'cysteine_protease': ProteinPocket(
        name='cysteine_protease', pocket_volume_A3=380, pocket_depth_A=12.0,
        n_hbd_pocket=4, n_hba_pocket=5, n_aromatic_residues=2,
        hydrophobic_fraction=0.50, net_charge_pocket=-1, sasa_burial_fraction=0.65),
    'metalloprotease': ProteinPocket(
        name='metalloprotease', pocket_volume_A3=450, pocket_depth_A=10.0,
        n_hbd_pocket=4, n_hba_pocket=6, n_aromatic_residues=3,
        hydrophobic_fraction=0.50, net_charge_pocket=1, sasa_burial_fraction=0.60),
    'kinase': ProteinPocket(
        name='kinase', pocket_volume_A3=500, pocket_depth_A=12.0,
        n_hbd_pocket=5, n_hba_pocket=7, n_aromatic_residues=4,
        hydrophobic_fraction=0.55, net_charge_pocket=0, sasa_burial_fraction=0.70),
    'gpcr': ProteinPocket(
        name='gpcr', pocket_volume_A3=350, pocket_depth_A=14.0,
        n_hbd_pocket=3, n_hba_pocket=5, n_aromatic_residues=5,
        hydrophobic_fraction=0.65, net_charge_pocket=0, sasa_burial_fraction=0.80),
    'metalloenzyme': ProteinPocket(
        name='metalloenzyme', pocket_volume_A3=400, pocket_depth_A=11.0,
        n_hbd_pocket=4, n_hba_pocket=6, n_aromatic_residues=3,
        hydrophobic_fraction=0.50, net_charge_pocket=1, sasa_burial_fraction=0.65),
    'viral_other': ProteinPocket(
        name='viral_other', pocket_volume_A3=450, pocket_depth_A=11.0,
        n_hbd_pocket=5, n_hba_pocket=7, n_aromatic_residues=3,
        hydrophobic_fraction=0.50, net_charge_pocket=0, sasa_burial_fraction=0.65),
    'phosphatase': ProteinPocket(
        name='phosphatase', pocket_volume_A3=380, pocket_depth_A=11.0,
        n_hbd_pocket=4, n_hba_pocket=6, n_aromatic_residues=2,
        hydrophobic_fraction=0.45, net_charge_pocket=1, sasa_burial_fraction=0.65),
}

DEFAULT_POCKET = ProteinPocket(
    name='default', pocket_volume_A3=400, pocket_depth_A=11.0,
    n_hbd_pocket=4, n_hba_pocket=6, n_aromatic_residues=3,
    hydrophobic_fraction=0.50, net_charge_pocket=0, sasa_burial_fraction=0.65)


def get_class_pocket(target_name):
    """Return appropriate ProteinPocket for a target name."""
    cls = classify_target(target_name)
    return CLASS_POCKETS.get(cls, DEFAULT_POCKET)


# =========================================================================
# PER-TARGET OFFSETS (98 targets from BDB Articles)
# =========================================================================

BDB_TARGET_OFFSETS = {
    "11-beta-hydroxysteroid dehydrogenase 1": +1.17,
    "5-hydroxytryptamine receptor 1A": +0.94,
    "5-hydroxytryptamine receptor 2A": -0.71,
    "5-hydroxytryptamine receptor 6": +1.75,
    "5-hydroxytryptamine receptor 7": +0.69,
    "72 kDa type IV collagenase": +1.62,
    "Acetylcholinesterase": +1.61,
    "Adenosine receptor A1": +1.26,
    "Adenosine receptor A2a": +1.87,
    "Amine oxidase [flavin-containing] A": +2.76,
    "Amine oxidase [flavin-containing] B": +2.27,
    "Androgen receptor": +1.94,
    "Aromatase": +2.39,
    "Aurora kinase A": +1.74,
    "Beta-carbonic anhydrase 1": -3.26,
    "Cannabinoid receptor 1": +0.67,
    "Cannabinoid receptor 2": +0.56,
    "Carbonic anhydrase": +1.79,
    "Carbonic anhydrase 1": -0.63,
    "Carbonic anhydrase 12": -0.31,
    "Carbonic anhydrase 13": -1.01,
    "Carbonic anhydrase 14": +0.07,
    "Carbonic anhydrase 2": +0.32,
    "Carbonic anhydrase 4": -0.06,
    "Carbonic anhydrase 5A, mitochondrial": -0.28,
    "Carbonic anhydrase 5B, mitochondrial": +2.61,
    "Carbonic anhydrase 6": -0.40,
    "Carbonic anhydrase 7": +2.20,
    "Carbonic anhydrase 9": +0.26,
    "Cathepsin B": +0.42,
    "Cathepsin D": +0.67,
    "Cathepsin H": -0.49,
    "Cathepsin K": -0.20,
    "Cathepsin S": +0.30,
    "Cholinesterase": +1.58,
    "Coagulation factor VII": -0.33,
    "Coagulation factor X": +0.62,
    "Collagenase ColG": +0.97,
    "Cyclin-A2/Cyclin-dependent kinase 2": +0.34,
    "Cyclin-dependent kinase 2/G1/S-specific cyclin-E1": -0.36,
    "Cyclin-dependent kinase 4/G1/S-specific cyclin-D1": -0.13,
    "DNA ligase 1": +1.20,
    "Delta-type opioid receptor": -1.55,
    "Dimer of Gag-Pol polyprotein [489-587]": +0.43,
    "Dimer of Gag-Pol polyprotein [491-589,Q496K]": +1.96,
    "Dimer of Gag-Pol polyprotein [501-599,Q508K,L534I,L564I,C568": +1.65,
    "Dimer of Gag-Pol polyprotein [501-599]": +0.51,
    "Dipeptidyl peptidase 4": +2.06,
    "E3 ubiquitin-protein ligase XIAP [241-356]": -0.58,
    "Fibroblast growth factor receptor 1": +0.50,
    "Gag-Pol polyprotein [489-587]": +2.86,
    "Genome polyprotein": +0.55,
    "Glycogen synthase kinase-3 beta": +0.44,
    "HIV-1 protease": +1.13,
    "Interstitial collagenase": +0.68,
    "Kappa-type opioid receptor": +0.17,
    "Liver carboxylesterase 1": +1.83,
    "MAP kinase-interacting serine/threonine-protein kinase 1": +0.48,
    "Matrix metalloproteinase-9": +1.45,
    "Methionine aminopeptidase 2": +3.50,
    "Mitogen-activated protein kinase 14": +0.25,
    "Mu-type opioid receptor": -1.08,
    "Muscarinic acetylcholine receptor M1": -0.40,
    "Muscarinic acetylcholine receptor M2": -0.42,
    "Muscarinic acetylcholine receptor M3": +0.30,
    "NEDD8-activating enzyme E1 regulatory subunit": +2.46,
    "Neutrophil collagenase": +1.42,
    "Nociceptin receptor": +0.33,
    "Peptidyl-prolyl cis-trans isomerase FKBP1A": -0.32,
    "Plasmepsin II": -0.96,
    "Plasminogen": -0.30,
    "Procathepsin L": -0.29,
    "Prostaglandin G/H synthase 2": +0.08,
    "Proto-oncogene tyrosine-protein kinase Src": +0.42,
    "Prothrombin": -1.14,
    "Replicase polyprotein 1ab": -0.80,
    "Serine protease 1": +0.63,
    "Serine/threonine-protein kinase pim-1": +0.10,
    "Stromelysin-1": +1.01,
    "Tyrosine-protein phosphatase non-receptor type 1 [1-2": -0.53,
    "Urokinase-type plasminogen activator": +0.66,
}

# =========================================================================
# CLASS-LEVEL OFFSETS (for unseen targets)
# =========================================================================

CLASS_OFFSETS = {
    'carbonic_anhydrase': +0.35,
    'serine_protease': +0.11,
    'aspartyl_protease': -1.07,
    'cysteine_protease': +0.46,
    'metalloprotease': +1.39,
    'kinase': +0.73,
    'gpcr': +0.94,
    'metalloenzyme': +2.55,
    'viral_other': -0.19,
    'phosphatase': -0.53,
    'other': +0.94,
}

GLOBAL_OFFSET = +0.87  # median across all targets


def get_target_offset(target_name):
    """Return the best available offset for a target.

    Priority: exact match > startswith match > class > global
    """
    # Exact match first
    if target_name in BDB_TARGET_OFFSETS:
        return BDB_TARGET_OFFSETS[target_name]

    # Startswith match (longest prefix wins)
    best_match = None
    best_len = 0
    for known, offset in BDB_TARGET_OFFSETS.items():
        if target_name.startswith(known) and len(known) > best_len:
            best_match = offset
            best_len = len(known)
    if best_match is not None:
        return best_match

    # Class match
    cls = classify_target(target_name)
    if cls in CLASS_OFFSETS:
        return CLASS_OFFSETS[cls]

    return GLOBAL_OFFSET


def annotate_generic_pocket(uc, pocket=None):
    """Apply generic pocket descriptors to a UniversalComplex.

    Use when the target doesn't have a specific ProteinPocket in PROTEIN_POCKETS.
    """
    if pocket is None:
        pocket = get_class_pocket(uc.host_name)

    uc.cavity_volume_A3 = pocket.pocket_volume_A3
    r_A = (3 * pocket.pocket_volume_A3 / (4 * math.pi)) ** (1/3)
    uc.cavity_radius_nm = r_A / 10.0
    uc.n_hbond_donors_host = pocket.n_hbd_pocket
    uc.n_hbond_acceptors_host = pocket.n_hba_pocket

    if uc.guest_volume_A3 > 0:
        uc.packing_coefficient = (
            uc.guest_volume_A3 * pocket.sasa_burial_fraction
        ) / pocket.pocket_volume_A3

    hb_d = min(uc.guest_n_hbond_donors, pocket.n_hba_pocket)
    hb_a = min(uc.guest_n_hbond_acceptors, pocket.n_hbd_pocket)
    n_hb = max(1, int(round((hb_d + hb_a) * 0.40)))
    uc.n_hbonds_formed = n_hb
    uc.hbond_types = ['neutral'] * n_hb

    if uc.guest_sasa_nonpolar_A2 > 0:
        uc.sasa_buried_A2 = round(
            uc.guest_sasa_nonpolar_A2 * pocket.sasa_burial_fraction, 1)

    return uc
