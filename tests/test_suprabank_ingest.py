"""
tests/test_suprabank_ingest.py — Validates SupraBank data ingestion pipeline.

Tests:
  - Data file loading (CB + CD)
  - UC field population
  - Scoring non-regression on known entries
  - PC_opt back-solve result consistency
"""
import pytest
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class TestSupraBankDataLoad:
    """Verify extracted data loads correctly."""

    def test_cb_raw_loads(self):
        path = os.path.join(_ROOT, "data", "suprabank_cb_raw.json")
        assert os.path.exists(path), "suprabank_cb_raw.json not found"
        with open(path) as f:
            data = json.load(f)
        assert data["total_entries"] >= 1300
        assert len(data["entries"]) >= 1300

    def test_cd_raw_loads(self):
        path = os.path.join(_ROOT, "data", "suprabank_cd_raw.json")
        assert os.path.exists(path), "suprabank_cd_raw.json not found"
        with open(path) as f:
            data = json.load(f)
        assert data["total"] >= 390
        assert len(data["entries"]) >= 390

    def test_mol_data_loads(self):
        path = os.path.join(_ROOT, "data", "suprabank_mol_data.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 700
        # Spot check: at least 90% have SMILES
        n_smi = sum(1 for v in data.values() if v)
        assert n_smi / len(data) > 0.9

    def test_cb_entry_structure(self):
        path = os.path.join(_ROOT, "data", "suprabank_cb_raw.json")
        with open(path) as f:
            data = json.load(f)
        e = data["entries"][0]
        assert "guest_name" in e
        assert "host" in e
        assert "logKa" in e
        assert "int_id" in e
        assert "guest_mol_id" in e

    def test_host_distribution(self):
        path = os.path.join(_ROOT, "data", "suprabank_cb_raw.json")
        with open(path) as f:
            data = json.load(f)
        hosts = set(e["host"] for e in data["entries"])
        assert "CB7" in hosts
        assert "CB8" in hosts


class TestSupraBankIngest:
    """Verify ingestion module produces valid UniversalComplex entries."""

    def test_load_cb_entries(self):
        from core.suprabank_ingest import load_suprabank_cb
        entries = load_suprabank_cb(max_entries=20)
        assert len(entries) == 20
        uc = entries[0]
        assert uc.binding_mode == "host_guest_inclusion"
        assert uc.host_name in ("CB5", "CB6", "CB7", "CB8")
        assert uc.cavity_volume_A3 > 0
        assert uc.log_Ka_exp != 0

    def test_load_cd_entries(self):
        from core.suprabank_ingest import load_suprabank_cd
        entries = load_suprabank_cd(max_entries=20)
        assert len(entries) == 20
        uc = entries[0]
        assert uc.host_name in ("alpha-CD", "beta-CD", "gamma-CD")

    def test_smiles_enrichment(self):
        from core.suprabank_ingest import load_suprabank_cb
        entries = load_suprabank_cb(max_entries=50)
        n_smi = sum(1 for uc in entries if uc.guest_smiles)
        # At least some should have SMILES
        assert n_smi > 0


class TestPC_optBackSolve:
    """Verify P28 back-solve result consistency."""

    def test_p28_result_exists(self):
        path = os.path.join(_ROOT, "data", "p28_suprabank_backsolve.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "adopted_value" in data
        pc_opt = data["adopted_value"]
        # Must be within physically reasonable range
        assert 0.4 < pc_opt < 0.7
        # Must be consistent with Rebek at 2-sigma
        assert abs(pc_opt - 0.55) < 0.18  # 2 * 0.09

    def test_p28_methods_converge(self):
        path = os.path.join(_ROOT, "data", "p28_suprabank_backsolve.json")
        with open(path) as f:
            data = json.load(f)
        methods = data["methods"]
        # All methods should agree within 0.15
        values = []
        if "gaussian_envelope" in methods:
            values.append(methods["gaussian_envelope"]["PC_opt"])
        if "top_binder_mean" in methods:
            values.append(methods["top_binder_mean"]["PC_opt"])
        assert len(values) >= 2
        assert max(values) - min(values) < 0.15


class TestScoringDiagnostics:
    """Verify scoring diagnostics are reasonable."""

    def test_diagnostics_exist(self):
        path = os.path.join(_ROOT, "data", "suprabank_scoring_diagnostics.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["scored"] >= 1500

    def test_betacd_reasonable(self):
        """beta-CD should be best-calibrated host."""
        path = os.path.join(_ROOT, "data", "suprabank_scoring_diagnostics.json")
        with open(path) as f:
            data = json.load(f)
        bcd = data["per_host"].get("beta-CD", {})
        assert bcd.get("n", 0) >= 250
        assert bcd.get("MAE", 99) < 2.0  # MAE < 2 logKa units

    def test_cb7_scored(self):
        path = os.path.join(_ROOT, "data", "suprabank_scoring_diagnostics.json")
        with open(path) as f:
            data = json.load(f)
        cb7 = data["per_host"].get("CB7", {})
        assert cb7.get("n", 0) >= 400
