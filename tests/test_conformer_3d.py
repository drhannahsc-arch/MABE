"""
tests/test_conformer_3d.py — 3D Conformer Pipeline Tests
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conformer_3d import (
    generate_conformers, find_donors_3d, find_all_donors,
    kabsch_rmsd, align_donors_to_ideal,
    conformer_strain, preorganization_score,
    score_3d, score_3d_batch, print_3d_result,
    Binding3DResult, DonorAtom3D, ConformerResult,
)


class TestConformerGeneration:
    def test_simple_molecule(self):
        mol, energies = generate_conformers("NCCN", n_conformers=5)
        assert mol is not None
        assert len(energies) > 0

    def test_macrocycle(self):
        mol, energies = generate_conformers("C1CNCCNCCNCCNC1", n_conformers=10)
        assert mol is not None
        assert len(energies) > 0

    def test_energies_are_finite(self):
        mol, energies = generate_conformers("NCCN", n_conformers=5)
        for e in energies:
            assert e < 900  # not 999 error value

    def test_invalid_smiles_returns_none(self):
        mol, energies = generate_conformers("INVALID", n_conformers=5)
        assert mol is None
        assert len(energies) == 0

    def test_reproducible(self):
        _, e1 = generate_conformers("NCCN", n_conformers=5, random_seed=42)
        _, e2 = generate_conformers("NCCN", n_conformers=5, random_seed=42)
        assert e1 == e2


class TestDonorLocator:
    def test_find_amines_in_en(self):
        mol, _ = generate_conformers("NCCN", n_conformers=1)
        donors = find_donors_3d(mol, 0, ["N_amine", "N_amine"])
        assert len(donors) == 2
        for d in donors:
            assert d.element == "N"
            assert d.donor_subtype == "N_amine"

    def test_find_mixed_donors_in_glycine(self):
        mol, _ = generate_conformers("NCC(=O)O", n_conformers=1)
        donors = find_donors_3d(mol, 0, ["N_amine", "O_carboxylate"])
        n_found = len(donors)
        assert n_found >= 1  # at least N should be found

    def test_donors_have_3d_positions(self):
        mol, _ = generate_conformers("NCCN", n_conformers=1)
        donors = find_donors_3d(mol, 0, ["N_amine"])
        for d in donors:
            assert len(d.position) == 3
            assert not np.allclose(d.position, [0, 0, 0])

    def test_find_thiol(self):
        mol, _ = generate_conformers("SCC(=O)O", n_conformers=1)
        donors = find_donors_3d(mol, 0, ["S_thiolate"])
        s_donors = [d for d in donors if d.element == "S"]
        assert len(s_donors) >= 1

    def test_find_all_donors_auto(self):
        mol, _ = generate_conformers("NCCN", n_conformers=1)
        donors = find_all_donors(mol, 0)
        assert len(donors) >= 2


class TestKabschAlignment:
    def test_identical_points_zero_rmsd(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = P.copy()
        rmsd, R, t = kabsch_rmsd(P, Q)
        assert rmsd < 1e-6

    def test_translated_points(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = P + np.array([5, 5, 5])
        rmsd, R, t = kabsch_rmsd(P, Q)
        assert rmsd < 1e-6  # Kabsch handles translation

    def test_rotated_points(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # 90° rotation around z
        Q = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        rmsd, R, t = kabsch_rmsd(P, Q)
        assert rmsd < 0.01

    def test_noisy_points_nonzero_rmsd(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = P + np.random.normal(0, 0.3, P.shape)
        rmsd, R, t = kabsch_rmsd(P, Q)
        assert rmsd > 0.01

    def test_align_donors_same_count(self):
        actual = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        ideal = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        rmsd, aligned, assignment = align_donors_to_ideal(actual, ideal)
        assert rmsd < 0.01

    def test_align_donors_different_count(self):
        actual = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        ideal = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        rmsd, aligned, assignment = align_donors_to_ideal(actual, ideal)
        assert rmsd < 999


class TestStrainAndPreorg:
    def test_strain_zero_for_same_energy(self):
        s_kcal, s_kJ = conformer_strain(50.0, 50.0)
        assert s_kcal == 0.0

    def test_strain_positive(self):
        s_kcal, s_kJ = conformer_strain(60.0, 50.0)
        assert s_kcal == 10.0
        assert abs(s_kJ - 41.84) < 0.01

    def test_strain_clamps_negative(self):
        """If binding conf is lower than min, strain = 0."""
        s_kcal, _ = conformer_strain(40.0, 50.0)
        assert s_kcal == 0.0

    def test_preorg_perfect(self):
        assert preorganization_score(0.0) == 1.0

    def test_preorg_decays(self):
        s1 = preorganization_score(0.5)
        s2 = preorganization_score(1.0)
        assert s1 > s2 > 0

    def test_preorg_near_zero_at_large_rmsd(self):
        assert preorganization_score(5.0) < 0.01


class TestScore3D:
    def test_en_returns_result(self):
        r = score_3d("NCCN", "Cu2+", ["N_amine", "N_amine"],
                      n_conformers=10)
        assert isinstance(r, Binding3DResult)
        assert r.n_donors_found == 2
        assert r.fidelity_score > 0

    def test_edta_finds_6_donors(self):
        r = score_3d("OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
                      "Cu2+", ["N_amine"] * 2 + ["O_carboxylate"] * 4,
                      n_conformers=10)
        assert r.n_donors_found == 6

    def test_cyclam_preorganized(self):
        r_cyclam = score_3d("C1CNCCNCCNCCNC1", "Ni2+",
                             ["N_amine"] * 4, n_conformers=10)
        r_edta = score_3d("OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O",
                           "Cu2+", ["N_amine"] * 2 + ["O_carboxylate"] * 4,
                           n_conformers=10)
        # Macrocycle should be more preorganized than flexible EDTA
        assert r_cyclam.preorganization_score > r_edta.preorganization_score

    def test_invalid_smiles_returns_empty(self):
        r = score_3d("INVALID", "Cu2+", ["N_amine"])
        assert r.n_conformers_generated == 0
        assert r.fidelity_score == 0.0

    def test_strain_nonnegative(self):
        r = score_3d("NCCN", "Cu2+", ["N_amine", "N_amine"],
                      n_conformers=10)
        assert r.strain_energy_kcal >= 0
        assert r.strain_energy_kJ >= 0


class TestScore3DBatch:
    def test_batch_returns_sorted(self):
        smiles_list = ["NCCN", "NCCNCCN", "CN"]
        results = score_3d_batch(smiles_list, "Cu2+",
                                  ["N_amine", "N_amine"],
                                  n_conformers=5)
        assert len(results) == 3
        # Should be sorted by fidelity (descending)
        fids = [r.fidelity_score for r in results]
        assert fids == sorted(fids, reverse=True)

    def test_batch_handles_failures(self):
        smiles_list = ["NCCN", "INVALID_SMILES"]
        results = score_3d_batch(smiles_list, "Cu2+",
                                  ["N_amine", "N_amine"],
                                  n_conformers=5)
        assert len(results) == 2
