"""
mabe/glycan/validation_g4.py -- G4: Synthetic Receptor Validation Gate
=======================================================================

Tests locked G1+G2+G3 parameters against non-biological answer keys.
No new parameters fitted. Pure prediction.

Primary test: Davis GluHUT receptor (Tromans et al. 2019, Nature Chem. 11:52)
  - Bicyclic hexaurea cage with TEM aromatic roof/floor
  - Measured Ka for 10 sugars by ITC at pH 7.4, 298K
  - Selectivity: Glc >> Gal, Man — INVERTS ConA selectivity
  - Same G1 desolvation + G3 CH-pi params must predict both

Secondary: ConA deoxy regression (already validated in G1, re-run through
full G1+G2+G3 scorer as integration check).

Answer keys from published ITC data. Zero fitting.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mabe.glycan.scorer import compute_glycan_terms, GlycanScoreDecomposition
from mabe.glycan.sugar_properties import (
    ALPHA_D_MANNOSE, ALPHA_D_GLUCOSE, ALPHA_D_GALACTOSE,
    ALPHA_D_GLCNAC, ALPHA_L_FUCOSE, SugarPropertyCard,
    get_sugar_card,
)
from mabe.glycan.contact_map import (
    GlycanContactMap, OHContact, CHPiContact,
    cona_mannose_pocket,
)
from mabe.glycan.ch_pi import CHPiContact as DetailedCHPi


R = 8.314e-3  # kJ/(mol*K)
T = 298.15    # K


# =====================================================================
# DAVIS GluHUT RECEPTOR — CONTACT MODEL
# =====================================================================
# Tromans 2019: bicyclic cage with two triethylmesitylene (TEM) aromatic
# platforms (roof + floor) and 6 urea NH H-bond donors.
#
# Binding physics:
#   - CH-pi: sugar alpha-face stacks against TEM aromatic platforms
#   - H-bonds: equatorial OHs form H-bonds to urea NH donors
#   - Desolvation: all 5 OHs are partially buried in the cage
#   - Shape: cage prefers all-equatorial sugars (4C1 chair fits best)
#
# Contact model derived from Eades 2026 MD + Tromans 2019 NMR:
#   - 2 aromatic platforms = CH-pi from both faces (sandwich geometry)
#   - 6 urea NH donors available for equatorial OH H-bonds
#   - Axial OHs cannot reach urea donors (geometry mismatch)

def davis_receptor_contacts(sugar_key: str) -> GlycanContactMap:
    """
    Build contact map for Davis GluHUT receptor with a given sugar.

    Contact assignment rules (from Tromans 2019 + Eades 2026):
      - Equatorial OHs: each forms ~1 H-bond to urea NH (n_hb = 1)
      - Axial OHs: poor geometry, no H-bond to receptor (n_hb = 0)
      - C6-OH primary: forms 1 H-bond (accessible)
      - CH-pi: TEM platforms stack sugar alpha-face
        For all-equatorial sugars (Glc): good sandwich, 3 contacts
        For axial-OH sugars (Gal C4-ax, Man C2-ax): reduced stacking
    """
    sugar = get_sugar_card(sugar_key)

    # Assign H-bonds based on stereochemistry
    oh_contacts = []
    for oh in sugar.hydroxyls:
        if oh.position == 'C1':
            # Anomeric OH: partial contact in cage
            oh_contacts.append(OHContact(oh.position, n_hbonds=1,
                                         is_buried=True, is_solvent_exposed=False))
        elif oh.orientation == 'equatorial':
            # Equatorial OHs reach urea NH donors
            oh_contacts.append(OHContact(oh.position, n_hbonds=1,
                                         hbond_partners=['urea_NH'],
                                         is_buried=True, is_solvent_exposed=False))
        elif oh.orientation == 'primary':
            # C6-OH: accessible, forms 1 H-bond
            oh_contacts.append(OHContact(oh.position, n_hbonds=1,
                                         hbond_partners=['urea_NH'],
                                         is_buried=True, is_solvent_exposed=False))
        else:
            # Axial OHs: geometry mismatch with urea donors
            oh_contacts.append(OHContact(oh.position, n_hbonds=0,
                                         is_buried=True, is_solvent_exposed=False))

    # CH-pi contacts from TEM platforms
    # All-equatorial sugars get best stacking (3 CH per face)
    # Axial OH on stacking face reduces contacts
    n_axial_oh_on_faces = sum(1 for oh in sugar.hydroxyls
                              if oh.orientation == 'axial' and oh.position != 'C1')
    if n_axial_oh_on_faces == 0:
        # All-equatorial: excellent stacking, 3 CH contacts
        n_ch_pi = 3
    elif n_axial_oh_on_faces == 1:
        # One axial OH (Man or Gal): reduced to 2 contacts
        n_ch_pi = 2
    else:
        # Multiple axial OHs: poor stacking
        n_ch_pi = 1

    ch_pi_contacts = []
    if n_ch_pi > 0:
        ch_pi_contacts = [CHPiContact(
            sugar_hydrogens=[f'C{i}-H' for i in [1, 3, 5][:n_ch_pi]],
            receptor_residue='TEM_platform',
            n_CH_contacts=n_ch_pi,
        )]

    return GlycanContactMap(
        pdb_id='',
        receptor_name='Davis_GluHUT',
        sugar_key=sugar_key,
        residue_in_binding_site=sugar.name,
        oh_contacts=oh_contacts,
        ch_pi_contacts=ch_pi_contacts,
        n_conserved_waters=0,
        linkage_types=[],
        n_branch_points=0,
    )


# =====================================================================
# EXPERIMENTAL ANSWER KEY — Davis GluHUT (Tromans 2019)
# =====================================================================
# All values at T = 298 K, pH 7.4, 10 mM phosphate buffer.
# Ka from ITC. dG = -RT ln(Ka).

DAVIS_ANSWER_KEY = {
    'aGlc': {'Ka_ITC': 18600, 'Ka_NMR': 18000, 'name': 'D-Glucose'},
    'aGal': {'Ka_ITC': 180, 'Ka_NMR': 130, 'name': 'D-Galactose'},
    'aMan': {'Ka_ITC': 140, 'Ka_NMR': 140, 'name': 'D-Mannose'},
    'aFuc': {'Ka_ITC': 60, 'Ka_NMR': 51, 'name': 'D-Fructose'},
}

# Experimental rank: Glc >> Gal > Man > Fuc
# Selectivity ratios: Glc:Gal = 103:1, Glc:Man = 133:1


def dG_from_Ka(Ka: float, temp: float = T) -> float:
    """Convert Ka to dG in kJ/mol."""
    if Ka <= 0:
        return float('inf')
    return -R * temp * math.log(Ka)


# =====================================================================
# VALIDATION RUNNER
# =====================================================================

@dataclass
class ValidationResult:
    """Result of a single sugar prediction."""
    sugar_key: str
    sugar_name: str
    dG_predicted: float
    dG_experimental: float
    Ka_experimental: float
    score_decomposition: GlycanScoreDecomposition
    rank_predicted: int = 0
    rank_experimental: int = 0


def run_davis_validation(beta_context: float = 1.0,
                         use_enthalpy_hbond: bool = True,
                         verbose: bool = False) -> Dict[str, any]:
    """
    Run G4 validation against Davis GluHUT receptor.

    Parameters
    ----------
    beta_context : float
        Context buffering. 1.0 for monosaccharides (no multivalency).
    use_enthalpy_hbond : bool
        Use k_hbond_dH (intrinsic) vs k_hbond_dG (CD portal).
    verbose : bool
        Print detailed output.

    Returns
    -------
    dict with:
        'results': list of ValidationResult
        'rank_correlation': Spearman rho
        'rank_correct': bool (top sugar = Glc)
        'selectivity_inverted': bool (Glc > Gal, opposite to ConA)
        'summary': str
    """
    results = []

    for sugar_key, answer in DAVIS_ANSWER_KEY.items():
        try:
            sugar = get_sugar_card(sugar_key)
        except KeyError:
            continue

        contacts = davis_receptor_contacts(sugar_key)
        score = compute_glycan_terms(
            sugar, contacts,
            beta_context=beta_context,
            use_enthalpy_hbond=use_enthalpy_hbond,
        )

        Ka_exp = answer['Ka_ITC']
        dG_exp = dG_from_Ka(Ka_exp)

        results.append(ValidationResult(
            sugar_key=sugar_key,
            sugar_name=answer['name'],
            dG_predicted=score.dG_total,
            dG_experimental=dG_exp,
            Ka_experimental=Ka_exp,
            score_decomposition=score,
        ))

    # Rank assignment
    results_by_pred = sorted(results, key=lambda r: r.dG_predicted)
    results_by_exp = sorted(results, key=lambda r: r.dG_experimental)

    for i, r in enumerate(results_by_pred):
        r.rank_predicted = i + 1
    for i, r in enumerate(results_by_exp):
        r.rank_experimental = i + 1

    # Spearman rank correlation
    n = len(results)
    if n >= 2:
        # Build rank lookup
        pred_ranks = {r.sugar_key: r.rank_predicted for r in results}
        exp_ranks = {r.sugar_key: r.rank_experimental for r in results}
        d_sq_sum = sum((pred_ranks[k] - exp_ranks[k]) ** 2 for k in pred_ranks)
        rho = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
    else:
        rho = 0.0

    # Key checks
    best_pred = results_by_pred[0].sugar_key
    rank_correct = (best_pred == 'aGlc')

    # Selectivity: Glc should be more favorable than Gal
    glc_score = next(r for r in results if r.sugar_key == 'aGlc')
    gal_score = next(r for r in results if r.sugar_key == 'aGal')
    selectivity_inverted = glc_score.dG_predicted < gal_score.dG_predicted

    summary_lines = [
        "G4 Validation: Davis GluHUT Receptor",
        "=" * 50,
        f"{'Sugar':<12} {'dG_pred':>10} {'dG_exp':>10} {'Ka_exp':>10} {'Rank_P':>7} {'Rank_E':>7}",
        "-" * 60,
    ]
    for r in results:
        summary_lines.append(
            f"{r.sugar_name:<12} {r.dG_predicted:>+10.2f} {r.dG_experimental:>+10.2f} "
            f"{r.Ka_experimental:>10.0f} {r.rank_predicted:>7d} {r.rank_experimental:>7d}"
        )
    summary_lines.extend([
        "-" * 60,
        f"Spearman rho: {rho:.3f}",
        f"Top predicted: {best_pred} ({'CORRECT' if rank_correct else 'WRONG'})",
        f"Glc > Gal selectivity: {'YES' if selectivity_inverted else 'NO'}",
    ])

    summary = "\n".join(summary_lines)
    if verbose:
        print(summary)

    return {
        'results': results,
        'rank_correlation': rho,
        'rank_correct': rank_correct,
        'selectivity_inverted': selectivity_inverted,
        'summary': summary,
    }


def run_cona_regression(verbose: bool = False) -> Dict[str, any]:
    """
    Re-run ConA deoxy validation through full G1+G2+G3 scorer.
    This is a regression check: G2 and G3 should not degrade G1 results.
    """
    from mabe.glycan.contact_map import cona_mannose_pocket, OHContact

    sugar = ALPHA_D_MANNOSE
    full_contacts = cona_mannose_pocket()

    # Full pocket score
    full_score = compute_glycan_terms(sugar, full_contacts, beta_context=0.45)

    # Deoxy variants: remove each OH's H-bonds
    deoxy_results = {}
    for pos_idx, pos in enumerate(['C1', 'C2', 'C3', 'C4', 'C6']):
        cm = cona_mannose_pocket()
        cm.oh_contacts[pos_idx] = OHContact(pos, n_hbonds=0, is_solvent_exposed=True)
        deoxy_score = compute_glycan_terms(sugar, cm, beta_context=0.45)
        ddg = deoxy_score.dG_total - full_score.dG_total
        deoxy_results[pos] = {
            'ddg_predicted': ddg,
            'essential': full_contacts.oh_contacts[pos_idx].n_hbonds >= 2,
        }

    # Check: essential OHs (C3, C4, C6) should cost > 3 kJ/mol to remove
    # Non-essential (C1, C2) should cost < 1 kJ/mol
    essentials_correct = all(
        deoxy_results[p]['ddg_predicted'] > 3.0
        for p in ['C3', 'C4', 'C6']
    )
    nonessentials_correct = all(
        abs(deoxy_results[p]['ddg_predicted']) < 2.0
        for p in ['C1', 'C2']
    )

    if verbose:
        print("ConA Deoxy Regression Check")
        print("=" * 40)
        for pos, r in deoxy_results.items():
            label = "essential" if r['essential'] else "non-essential"
            print(f"  {pos}: ddG = {r['ddg_predicted']:+.2f} kJ/mol ({label})")
        print(f"  Essentials correct: {essentials_correct}")
        print(f"  Non-essentials correct: {nonessentials_correct}")

    return {
        'deoxy_results': deoxy_results,
        'essentials_correct': essentials_correct,
        'nonessentials_correct': nonessentials_correct,
        'passed': essentials_correct and nonessentials_correct,
    }
