"""
tests/test_validation_pipeline.py -- Tests for the end-to-end validation pipeline.
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from core.validation_pipeline import (
    validate_glycan, validate_metal, validate_host_guest,
    validate_denovo, run_full_validation, export_json,
    _r2, _mae, _rmse, _spearman,
    ValidationResult,
)


class TestStatistics:

    def test_r2_perfect(self):
        assert abs(_r2([1, 2, 3], [1, 2, 3]) - 1.0) < 0.001

    def test_r2_zero(self):
        assert _r2([1, 2, 3], [2, 2, 2]) < 0.01

    def test_mae(self):
        assert abs(_mae([1, 2, 3], [1.5, 2.5, 3.5]) - 0.5) < 0.001

    def test_rmse(self):
        assert abs(_rmse([1, 2, 3], [1, 2, 3]) - 0.0) < 0.001

    def test_spearman_perfect(self):
        assert abs(_spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 0.001

    def test_spearman_inverse(self):
        assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) < -0.9


class TestGlycanValidation:

    def test_runs(self):
        vr = validate_glycan()
        assert vr.modality == "glycan"
        assert vr.n_scored >= 30

    def test_r2_threshold(self):
        vr = validate_glycan()
        assert vr.r2 > 0.95

    def test_selectivity_all_pass(self):
        vr = validate_glycan()
        assert all(c["correct"] for c in vr.selectivity_checks)

    def test_outlier_flagged(self):
        vr = validate_glycan()
        outlier_names = [f"{o['scaffold']} {o['ligand']}" for o in vr.outliers]
        assert any("DGL" in n and "triMan" in n for n in outlier_names)

    def test_per_scaffold_populated(self):
        vr = validate_glycan()
        assert len(vr.per_scaffold) >= 5


class TestMetalValidation:

    def test_runs(self):
        vr = validate_metal()
        assert vr.modality == "metal"
        assert vr.n_scored > 0

    def test_rank_correlation(self):
        vr = validate_metal()
        assert vr.spearman > 0.7

    def test_irving_williams_mostly_correct(self):
        vr = validate_metal()
        n_pass = sum(1 for c in vr.selectivity_checks if c["correct"])
        assert n_pass >= 3


class TestHostGuestValidation:

    def test_runs(self):
        vr = validate_host_guest()
        assert vr.modality == "host_guest"
        assert vr.n_scored > 0

    def test_mae_reasonable(self):
        vr = validate_host_guest()
        assert vr.mae < 2.0


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not available")
class TestDeNovoValidation:

    def test_runs(self):
        vr = validate_denovo()
        assert vr.modality == "denovo"
        assert vr.n_scored > 0

    def test_grammar_beats_fixed(self):
        vr = validate_denovo()
        grammar_check = [c for c in vr.selectivity_checks
                         if c["scaffold"] == "Glc" and c["preferred"] == "grammar"]
        assert len(grammar_check) == 1
        assert grammar_check[0]["correct"]


class TestFullPipeline:

    def test_runs(self):
        v = run_full_validation(glycan=True, metal=True, host_guest=True, denovo=False)
        assert v.glycan is not None
        assert v.metal is not None
        assert v.host_guest is not None
        assert v.summary != ""

    def test_json_export(self, tmp_path):
        v = run_full_validation(glycan=True, metal=False, host_guest=False, denovo=False)
        path = str(tmp_path / "test_val.json")
        export_json(v, path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert "glycan" in data
        assert data["glycan"]["r2"] > 0.95
