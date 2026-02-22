"""
core/hbond_classifier.py -- Sprint 40 Chunk C: H-Bond Donor/Acceptor Subtyping

Problem: All neutral H-bonds assigned identical epsilon_hbond_neutral.
A sulfonamide NH (strong, directional pharmacophore for CA-II inhibitors)
gets same energy as an aliphatic amine NH2 (weaker, less directional).
This kills within-target ranking.

Fix: SMARTS-based classification of donor/acceptor functional groups
into strength tiers. Each tier gets its own epsilon parameter in the
predictor, enabling the optimizer to weight pharmacophoric contacts
differently.

Strength tiers (literature ranges from Abraham, Laurence):
  strong_donor:    sulfonamide NH, guanidinium NH, amide NH (6-10 kJ/mol)
  moderate_donor:  hydroxyl, primary amine, carbamate NH (3-6 kJ/mol)
  weak_donor:      aromatic NH, thiol, C-H activated (1-3 kJ/mol)
  strong_acceptor: carboxylate, phosphate, N-oxide (6-10 kJ/mol)
  moderate_acceptor: carbonyl, ether, sulfone (3-6 kJ/mol)
  weak_acceptor:   aromatic N, thioether, nitrile (1-3 kJ/mol)
"""

from rdkit import Chem


# =========================================================================
# DONOR CLASSIFICATION
# =========================================================================

DONOR_RULES = [
    # Strong donors
    {'name': 'sulfonamide_NH', 'smarts': '[NH1]S(=O)=O', 'tier': 'strong_donor'},
    {'name': 'sulfonamide_NH2', 'smarts': '[NH2]S(=O)=O', 'tier': 'strong_donor'},
    {'name': 'amide_NH', 'smarts': '[NH1]C(=O)', 'tier': 'strong_donor'},
    {'name': 'urea_NH', 'smarts': '[NH1]C(=O)[NH]', 'tier': 'strong_donor'},
    {'name': 'guanidine_NH', 'smarts': '[NH]C(=[NH])[NH]', 'tier': 'strong_donor'},

    # Moderate donors
    {'name': 'hydroxyl', 'smarts': '[OX2H1]', 'tier': 'moderate_donor'},
    {'name': 'primary_amine', 'smarts': '[NX3H2;!$([NH2]C=O);!$([NH2]S)]', 'tier': 'moderate_donor'},
    {'name': 'secondary_amine', 'smarts': '[NX3H1;!$([NH1]C=O);!$([NH1]S);!$([nH])]', 'tier': 'moderate_donor'},
    {'name': 'carbamate_NH', 'smarts': '[NH1]C(=O)O', 'tier': 'moderate_donor'},

    # Weak donors
    {'name': 'aromatic_NH', 'smarts': '[nH]', 'tier': 'weak_donor'},
    {'name': 'thiol', 'smarts': '[SX2H1]', 'tier': 'weak_donor'},
]

# =========================================================================
# ACCEPTOR CLASSIFICATION
# =========================================================================

ACCEPTOR_RULES = [
    # Strong acceptors
    {'name': 'carboxylate', 'smarts': '[CX3](=O)[O-]', 'tier': 'strong_acceptor'},
    {'name': 'carboxylic_acid_O', 'smarts': '[CX3](=O)[OX2H1]', 'tier': 'strong_acceptor'},
    {'name': 'phosphate_O', 'smarts': 'P(=O)([O])', 'tier': 'strong_acceptor'},
    {'name': 'sulfonamide_O', 'smarts': 'S(=O)(=O)[NH]', 'tier': 'strong_acceptor'},

    # Moderate acceptors
    {'name': 'carbonyl', 'smarts': '[CX3]=[OX1]', 'tier': 'moderate_acceptor'},
    {'name': 'ether', 'smarts': '[OX2]([CX4])[CX4]', 'tier': 'moderate_acceptor'},
    {'name': 'sulfone_O', 'smarts': '[SX4](=O)(=O)', 'tier': 'moderate_acceptor'},
    {'name': 'hydroxyl_O', 'smarts': '[OX2H1]', 'tier': 'moderate_acceptor'},

    # Weak acceptors
    {'name': 'aromatic_N', 'smarts': '[nX2H0]', 'tier': 'weak_acceptor'},
    {'name': 'nitrile', 'smarts': '[CX2]#[NX1]', 'tier': 'weak_acceptor'},
    {'name': 'thioether', 'smarts': '[SX2]([CX4])[CX4]', 'tier': 'weak_acceptor'},
]

# Compile at import time
_DONOR_COMPILED = []
for rule in DONOR_RULES:
    pat = Chem.MolFromSmarts(rule['smarts'])
    if pat:
        _DONOR_COMPILED.append({'name': rule['name'], 'pattern': pat, 'tier': rule['tier']})

_ACCEPTOR_COMPILED = []
for rule in ACCEPTOR_RULES:
    pat = Chem.MolFromSmarts(rule['smarts'])
    if pat:
        _ACCEPTOR_COMPILED.append({'name': rule['name'], 'pattern': pat, 'tier': rule['tier']})


# =========================================================================
# CLASSIFICATION FUNCTIONS
# =========================================================================

def classify_donors(smiles):
    """Classify H-bond donors by strength tier.

    Returns dict: {tier: count} e.g. {'strong_donor': 1, 'moderate_donor': 2}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    tier_counts = {}
    claimed_atoms = set()

    for rule in _DONOR_COMPILED:
        matches = mol.GetSubstructMatches(rule['pattern'])
        for match in matches:
            # Use first atom (the donor atom) as key
            donor_atom = match[0]
            if donor_atom in claimed_atoms:
                continue
            claimed_atoms.add(donor_atom)
            tier = rule['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return tier_counts


def classify_acceptors(smiles):
    """Classify H-bond acceptors by strength tier.

    Returns dict: {tier: count}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    tier_counts = {}
    claimed_atoms = set()

    for rule in _ACCEPTOR_COMPILED:
        matches = mol.GetSubstructMatches(rule['pattern'])
        for match in matches:
            acceptor_atom = match[0]
            if acceptor_atom in claimed_atoms:
                continue
            claimed_atoms.add(acceptor_atom)
            tier = rule['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return tier_counts


def classify_hbond_groups(smiles):
    """Full H-bond classification.

    Returns:
        (donor_tiers, acceptor_tiers) — both are {tier: count} dicts
    """
    return classify_donors(smiles), classify_acceptors(smiles)


# =========================================================================
# ENRICHMENT: Generate typed hbond_types list for predictor
# =========================================================================

def enrich_hbond_types(uc):
    """Replace generic hbond_types with strength-tiered types.

    Maps guest donor/acceptor tiers against pocket donor/acceptor
    availability to produce a hbond_types list the predictor can dispatch.

    New type vocabulary:
      'strong', 'moderate', 'weak' (replaces 'neutral'/'charge_assisted')

    The predictor needs corresponding epsilon params for each tier.
    """
    if not uc.guest_smiles:
        return uc
    if uc.binding_mode not in ('protein_ligand',):
        return uc  # Only apply to PL for now; HG uses existing types

    donor_tiers, acceptor_tiers = classify_hbond_groups(uc.guest_smiles)

    # Pocket complementary sites
    pocket_acceptors = getattr(uc, 'n_hbond_acceptors_host', 0)
    pocket_donors = getattr(uc, 'n_hbond_donors_host', 0)

    # Build typed H-bond list: each guest donor matched to pocket acceptor
    hbond_types = []

    # Donors -> pocket acceptors (strongest donors matched first)
    remaining_acceptors = pocket_acceptors
    for tier in ['strong_donor', 'moderate_donor', 'weak_donor']:
        n = donor_tiers.get(tier, 0)
        matched = min(n, remaining_acceptors)
        remaining_acceptors -= matched
        # Map donor tier to hbond type
        if tier == 'strong_donor':
            hbond_types.extend(['strong'] * matched)
        elif tier == 'moderate_donor':
            hbond_types.extend(['moderate'] * matched)
        else:
            hbond_types.extend(['weak'] * matched)

    # Acceptors -> pocket donors (strongest acceptors matched first)
    remaining_donors = pocket_donors
    for tier in ['strong_acceptor', 'moderate_acceptor', 'weak_acceptor']:
        n = acceptor_tiers.get(tier, 0)
        matched = min(n, remaining_donors)
        remaining_donors -= matched
        if tier == 'strong_acceptor':
            hbond_types.extend(['strong'] * matched)
        elif tier == 'moderate_acceptor':
            hbond_types.extend(['moderate'] * matched)
        else:
            hbond_types.extend(['weak'] * matched)

    # Satisfaction factor: not all potential H-bonds form
    # Use same ~40% as protein_pockets.py
    import math
    n_potential = len(hbond_types)
    n_formed = max(1, int(round(n_potential * 0.40)))

    # Keep the strongest H-bonds (already sorted strong-first)
    hbond_types = hbond_types[:n_formed]

    uc.n_hbonds_formed = len(hbond_types)
    uc.hbond_types = hbond_types

    return uc
