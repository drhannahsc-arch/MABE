"""
test_ranker.py — Integration tests for Layer 3 Realization Ranker.

Two scenarios:
  1. Pb²⁺ capture from acid mine drainage (small pocket, low pH, harsh conditions)
  2. Protein biomarker capture for diagnostics (large pocket, neutral pH, mild)

These exercise all scoring axes and demonstrate epistemic transparency.
"""

import sys
sys.path.insert(0, "/home/claude")

from realization_ranker import (
    rank_realizations,
    print_rankings,
    InteractionGeometrySpec,
    DeploymentConditions,
)
from realization_ranker.epistemic import EpistemicBasis


def test_pb2_amd_capture():
    """
    Scenario 1: Pb²⁺ selective capture from acid mine drainage.

    Layer 2 says: need a ~0.26 nm pocket, 4 donors (2× N_pyridine + 2× O_carboxylate),
    octahedral-ish geometry. Deployment: pH 3.5, 12°C, dissolved O₂, industrial scale.

    Expected: small molecules and chelators should rank high.
    DNA origami should be disqualified (pH too low).
    Proteins should be penalized (pH stress + oxidation).
    """
    print("=" * 72)
    print("SCENARIO 1: Pb²⁺ capture from acid mine drainage")
    print("=" * 72)

    geometry = InteractionGeometrySpec(
        cavity_diameter_nm=0.26,
        donor_count=4,
        donor_types=["N_pyridine", "N_pyridine", "O_carboxylate", "O_carboxylate"],
        donor_distances_nm=[0.28, 0.28, 0.40, 0.40, 0.28, 0.40],
        coordination_geometry="octahedral",
        target_summary="Pb²⁺ selective capture from acid mine drainage",
    )

    conditions = DeploymentConditions(
        temperature_C=12.0,
        pH=3.5,
        oxidants=["dissolved_O2"],
        environment="environmental_water",
        operational_hours=720,       # 30 days continuous
        target_scale="industrial",
        strong_oxidant=False,
    )

    ranked = rank_realizations(geometry, conditions)
    print_rankings(ranked)

    # ─── Assertions ───
    qualified = ranked.qualified_only()
    assert len(qualified) > 0, "At least one realization should qualify"

    # DNA origami should be disqualified at pH 3.5
    dna_origami = [r for r in ranked.realizations if r.realization_type == "DNA_origami"]
    assert dna_origami[0].disqualified, "DNA origami should be disqualified at pH 3.5"

    # Small molecule / chelator should be in top 3 qualified
    top3_types = [r.realization_type for r in qualified[:3]]
    assert any(t in top3_types for t in ("small_molecule", "chelator", "crown_ether")), \
        f"Small molecule or chelator should be top-3. Got: {top3_types}"

    # Every score should have an epistemic basis
    for r in qualified:
        for axis in [r.geometric_fidelity, r.synthetic_accessibility,
                     r.operating_conditions, r.scale_feasibility]:
            assert isinstance(axis.basis, EpistemicBasis), \
                f"Missing epistemic basis for {r.realization_type}"

    # Heuristic scores should have notes
    for r in qualified:
        for axis in [r.geometric_fidelity, r.synthetic_accessibility,
                     r.operating_conditions, r.scale_feasibility]:
            if axis.basis == EpistemicBasis.HEURISTIC_ESTIMATE:
                assert axis.note is not None, \
                    f"Heuristic score missing note for {r.realization_type}"

    print("✓ Scenario 1 assertions passed\n")
    return ranked


def test_protein_biomarker_diagnostic():
    """
    Scenario 2: Protein biomarker capture for blood diagnostic.

    Layer 2 says: need a ~2.0 nm binding surface, 6 donors
    (mixed H-bond donors/acceptors), no specific coordination geometry.
    Deployment: pH 7.4, 37°C, serum environment, diagnostic scale.

    Expected: protein/antibody should rank higher here.
    Small molecules should be penalized (pocket too large).
    MOF and crystal may be disqualified (cavity too large for some).
    """
    print("=" * 72)
    print("SCENARIO 2: Protein biomarker capture for blood diagnostic")
    print("=" * 72)

    geometry = InteractionGeometrySpec(
        cavity_diameter_nm=2.0,
        donor_count=6,
        donor_types=[
            "O_carboxylate", "O_hydroxyl", "N_imidazole",
            "N_amine", "O_carbonyl", "S_thiolate",
        ],
        donor_distances_nm=[1.0, 1.2, 0.8, 1.5, 0.9, 1.1],
        target_summary="Cardiac troponin I capture for point-of-care diagnostic",
    )

    conditions = DeploymentConditions(
        temperature_C=37.0,
        pH=7.4,
        oxidants=[],
        environment="serum",
        operational_hours=1.0,        # 1 hour assay
        target_scale="diagnostic",
        strong_oxidant=False,
    )

    ranked = rank_realizations(geometry, conditions)
    print_rankings(ranked)

    # ─── Assertions ───
    qualified = ranked.qualified_only()
    assert len(qualified) > 0, "At least one realization should qualify"

    # Small molecule chelators should be disqualified (cavity too large)
    chelator = [r for r in ranked.realizations if r.realization_type == "chelator"]
    assert chelator[0].disqualified, \
        "Chelator should be disqualified for 2.0 nm cavity (max 0.6 nm)"

    # Porphyrin should be disqualified (cavity too large)
    porph = [r for r in ranked.realizations if r.realization_type == "porphyrin"]
    assert porph[0].disqualified, \
        "Porphyrin should be disqualified for 2.0 nm cavity"

    # Protein or aptamer should be in top 3
    top3_types = [r.realization_type for r in qualified[:3]]
    assert any(t in top3_types for t in ("protein", "antibody_CDR", "aptamer", "peptide")), \
        f"Protein/aptamer/peptide should be top-3 for biomarker. Got: {top3_types}"

    # Serialization should work
    output = ranked.to_dict()
    assert "realizations" in output
    assert output["n_qualified"] == len(qualified)

    print("✓ Scenario 2 assertions passed\n")
    return ranked


def test_epistemic_transparency():
    """Verify every heuristic score is properly flagged."""
    print("=" * 72)
    print("TEST: Epistemic transparency verification")
    print("=" * 72)

    geometry = InteractionGeometrySpec(
        cavity_diameter_nm=0.3,
        donor_count=4,
        donor_types=["O_carboxylate", "N_amine"],
        target_summary="Generic metal capture",
    )
    conditions = DeploymentConditions()

    ranked = rank_realizations(geometry, conditions)

    heuristic_count = 0
    physics_count = 0
    empirical_count = 0

    for r in ranked.qualified_only():
        for axis in [r.geometric_fidelity, r.synthetic_accessibility,
                     r.operating_conditions, r.scale_feasibility]:
            if axis.basis == EpistemicBasis.HEURISTIC_ESTIMATE:
                heuristic_count += 1
                assert "best guess" in (axis.note or "").lower() or \
                       "heuristic" in (axis.note or "").lower(), \
                    f"Heuristic score for {r.realization_type} missing 'best guess' note"
            elif axis.basis == EpistemicBasis.PHYSICS_DERIVED:
                physics_count += 1
            elif axis.basis == EpistemicBasis.API_EMPIRICAL:
                empirical_count += 1

    total = heuristic_count + physics_count + empirical_count
    print(f"  Physics-derived:    {physics_count}/{total} ({100*physics_count/total:.0f}%)")
    print(f"  API-empirical:      {empirical_count}/{total} ({100*empirical_count/total:.0f}%)")
    print(f"  Heuristic estimate: {heuristic_count}/{total} ({100*heuristic_count/total:.0f}%)")
    print(f"\n✓ All heuristic scores properly flagged\n")


if __name__ == "__main__":
    r1 = test_pb2_amd_capture()
    r2 = test_protein_biomarker_diagnostic()
    test_epistemic_transparency()

    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)