"""
tests/test_ideal_pocket.py — Ideal Pocket Computation Tests
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ideal_pocket import (
    compute_ideal_pocket, ideal_pocket_for_metal, ideal_pocket_mixed,
    ideal_host_cavity, score_deviation,
    IdealPocket, DonorPosition, DeviationReport,
    SHANNON_RADII, DONOR_RADII, PREFERRED_GEOMETRY,
    _geometry_vectors, _parse_donor_string,
)


class TestGeometryVectors:
    def test_linear_2_vectors(self):
        vecs = _geometry_vectors("linear")
        assert len(vecs) == 2

    def test_tetrahedral_4_vectors(self):
        vecs = _geometry_vectors("tetrahedral")
        assert len(vecs) == 4

    def test_octahedral_6_vectors(self):
        vecs = _geometry_vectors("octahedral")
        assert len(vecs) == 6

    def test_square_planar_4_vectors(self):
        vecs = _geometry_vectors("square_planar")
        assert len(vecs) == 4

    def test_unit_vectors(self):
        """All geometry vectors should be unit length."""
        for geom in ["linear", "tetrahedral", "octahedral", "square_planar",
                      "trigonal_bipyramidal", "cubic"]:
            vecs = _geometry_vectors(geom)
            for v in vecs:
                assert abs(np.linalg.norm(v) - 1.0) < 0.01, f"{geom}: {v}"

    def test_tetrahedral_angles(self):
        """Tetrahedral angles should be ~109.5°."""
        vecs = _geometry_vectors("tetrahedral")
        angles = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos_a = np.dot(vecs[i], vecs[j])
                angles.append(math.degrees(math.acos(max(-1, min(1, cos_a)))))
        # All angles should be ~109.5°
        for a in angles:
            assert abs(a - 109.47) < 1.0, f"Tetrahedral angle {a}° != 109.5°"

    def test_octahedral_angles(self):
        """Adjacent octahedral angles should be 90°."""
        vecs = _geometry_vectors("octahedral")
        right_angles = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos_a = abs(np.dot(vecs[i], vecs[j]))
                if cos_a < 0.01:  # ~90°
                    right_angles += 1
        assert right_angles == 12  # 12 pairs at 90° in octahedron

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError):
            _geometry_vectors("dodecahedral")


import math


class TestBondLengths:
    def test_cu2_4n_bond_length(self):
        """Cu²⁺ + N_amine: r = 0.57 + 1.36 = 1.93 Å."""
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, "square_planar")
        assert abs(p.donors[0].bond_length_A - 1.93) < 0.01

    def test_zn2_4n_tetrahedral(self):
        """Zn²⁺ tetrahedral: r = 0.60 + 1.36 = 1.96 Å."""
        p = ideal_pocket_for_metal("Zn2+")
        assert abs(p.donors[0].bond_length_A - 1.96) < 0.01

    def test_pb2_s_thiolate(self):
        """Pb²⁺ + S_thiolate: r = 0.98 + 1.70 = 2.68 Å."""
        p = compute_ideal_pocket("Pb2+", ["S_thiolate"] * 4, "tetrahedral")
        assert abs(p.donors[0].bond_length_A - 2.68) < 0.01

    def test_hg2_linear(self):
        """Hg²⁺ linear: r = 0.69 + 1.70 = 2.39 Å for S donors."""
        p = compute_ideal_pocket("Hg2+", ["S_thiolate", "S_thiolate"], "linear")
        assert abs(p.donors[0].bond_length_A - 2.39) < 0.01


class TestIdealPocketStructure:
    def test_returns_ideal_pocket(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4)
        assert isinstance(p, IdealPocket)

    def test_correct_n_donors(self):
        p = compute_ideal_pocket("Fe3+", ["O_carboxylate"] * 6)
        assert p.n_donors == 6
        assert len(p.donors) == 6

    def test_donor_positions_are_3d(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4)
        for d in p.donors:
            assert len(d.position_A) == 3

    def test_positions_at_correct_distance(self):
        """All donor positions should be at bond_length from origin."""
        p = compute_ideal_pocket("Zn2+", ["N_amine"] * 4, "tetrahedral")
        for d in p.donors:
            dist = np.linalg.norm(d.position_A)
            assert abs(dist - d.bond_length_A) < 0.001

    def test_cavity_volume_positive(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4)
        assert p.cavity_volume_A3 > 0

    def test_ideal_dG_reasonable(self):
        """Net ΔG should be negative for stable complexes."""
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, chelate_rings=2,
                                  macrocyclic=True)
        # With chelate rings + macrocyclic, should be favorable
        assert p.ideal_dG_kJ < 0

    def test_donor_signature(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine", "N_amine",
                                           "O_carboxylate", "O_carboxylate"])
        assert "N_amine" in p.donor_signature
        assert "O_carboxylate" in p.donor_signature


class TestAutoGeometry:
    def test_cu2_prefers_square_planar(self):
        p = ideal_pocket_for_metal("Cu2+", n_donors=4)
        assert p.geometry == "square_planar"

    def test_zn2_prefers_tetrahedral(self):
        p = ideal_pocket_for_metal("Zn2+")
        assert p.geometry == "tetrahedral"

    def test_hg2_prefers_linear(self):
        p = ideal_pocket_for_metal("Hg2+", n_donors=2)
        assert p.geometry == "linear"

    def test_ni2_6_donors_octahedral(self):
        p = compute_ideal_pocket("Ni2+", ["N_amine"] * 6)
        assert p.geometry == "octahedral"


class TestMixedDonors:
    def test_parse_donor_string(self):
        result = _parse_donor_string("2N_amine+2O_carboxylate")
        assert result == ["N_amine", "N_amine", "O_carboxylate", "O_carboxylate"]

    def test_mixed_pocket(self):
        p = ideal_pocket_mixed("Cu2+", "2N_amine+2O_carboxylate")
        assert p.n_donors == 4
        subtypes = [d.donor_subtype for d in p.donors]
        assert subtypes.count("N_amine") == 2
        assert subtypes.count("O_carboxylate") == 2

    def test_mixed_different_bond_lengths(self):
        """N and O donors should have different bond lengths."""
        p = ideal_pocket_mixed("Cu2+", "2N_amine+2O_carboxylate")
        n_lengths = [d.bond_length_A for d in p.donors if d.element == "N"]
        o_lengths = [d.bond_length_A for d in p.donors if d.element == "O"]
        assert n_lengths[0] != o_lengths[0]


class TestHostGuest:
    def test_host_cavity_returns_pocket(self):
        p = ideal_host_cavity(2.76, n_contacts=6, contact_type="O_ether")
        assert isinstance(p, IdealPocket)
        assert p.n_donors == 6

    def test_larger_guest_larger_cavity(self):
        p_small = ideal_host_cavity(2.0)
        p_large = ideal_host_cavity(5.0)
        assert p_large.cavity_radius_A > p_small.cavity_radius_A

    def test_host_donor_positions(self):
        p = ideal_host_cavity(2.76, n_contacts=6)
        for d in p.donors:
            assert np.linalg.norm(d.position_A) > 0


class TestChelateAndMacrocyclic:
    def test_chelate_stabilizes(self):
        p_mono = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, chelate_rings=0)
        p_chel = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, chelate_rings=2)
        assert p_chel.ideal_dG_kJ < p_mono.ideal_dG_kJ

    def test_macrocyclic_stabilizes(self):
        p_open = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, chelate_rings=2,
                                       macrocyclic=False)
        p_macro = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, chelate_rings=2,
                                        macrocyclic=True)
        assert p_macro.ideal_dG_kJ < p_open.ideal_dG_kJ


class TestRigidity:
    def test_coordination_is_preorganized(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4)
        assert p.rigidity_class in ("crystalline", "preorganized")

    def test_host_guest_is_semi_flexible(self):
        p = ideal_host_cavity(3.0, n_contacts=6)
        assert p.rigidity_class in ("semi_flexible", "any")


class TestDeviationScoring:
    def test_perfect_match_high_fidelity(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, "square_planar")
        actual = [d.position_A.copy() for d in p.donors]
        dev = score_deviation(p, actual)
        assert dev.fidelity_score > 0.95
        assert dev.mean_deviation_A < 0.001

    def test_noisy_positions_lower_fidelity(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, "square_planar")
        actual = [d.position_A + np.array([0.2, 0.0, 0.0]) for d in p.donors]
        dev = score_deviation(p, actual)
        assert dev.fidelity_score < 0.5

    def test_missing_donor_penalty(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, "square_planar")
        actual = [d.position_A.copy() for d in p.donors[:3]]  # missing one
        dev = score_deviation(p, actual)
        assert dev.missing_donors == 1
        assert dev.fidelity_score < 0.9

    def test_extra_donor_mild_penalty(self):
        p = compute_ideal_pocket("Cu2+", ["N_amine"] * 4, "square_planar")
        actual = [d.position_A.copy() for d in p.donors]
        actual.append(np.array([0, 0, 3.0]))  # extra
        dev = score_deviation(p, actual)
        assert dev.extra_donors == 1
        assert dev.fidelity_score > 0.8


class TestDatabaseCoverage:
    def test_shannon_radii_coverage(self):
        """Key metals should have Shannon radii."""
        for metal in ["Cu2+", "Zn2+", "Fe3+", "Pb2+", "Hg2+", "Ca2+",
                      "Ni2+", "Co2+", "Cd2+", "Na+", "K+"]:
            assert metal in SHANNON_RADII, f"Missing {metal}"

    def test_donor_radii_coverage(self):
        for donor in ["N_amine", "O_carboxylate", "S_thiolate", "P_phosphine"]:
            assert donor in DONOR_RADII

    def test_preferred_geometry_coverage(self):
        for metal in ["Cu2+", "Zn2+", "Fe3+", "Ni2+", "Hg2+"]:
            assert metal in PREFERRED_GEOMETRY
