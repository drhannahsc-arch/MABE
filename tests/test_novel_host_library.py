"""
test_novel_host_library.py — Tests for the novel host library.

Split into:
  - Structural tests (no rdkit): run anywhere
  - Integration tests (rdkit required): run on local machine only
"""
import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.novel_host_library import (
    get_host, get_entry, list_hosts, list_names, host_summary,
    NovelHostSpec, HostEntry, _LIBRARY, _ALIASES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data integrity tests (no rdkit)
# ═══════════════════════════════════════════════════════════════════════════

class TestLibraryCompleteness:
    """Verify library has required entries and data quality."""

    def test_minimum_host_count(self):
        """Library has at least 15 hosts."""
        assert len(_LIBRARY) >= 15

    def test_all_categories_present(self):
        for cat in ["MOF", "cage", "zeolite", "synthetic_receptor"]:
            entries = list_hosts(host_type=cat)
            assert len(entries) >= 1, f"Missing category: {cat}"

    def test_all_have_nonzero_cavity_volume(self):
        for name, entry in _LIBRARY.items():
            assert entry.spec.cavity_volume_A3 > 0, \
                f"{name}: zero cavity volume"

    def test_all_have_source_citations(self):
        for name, entry in _LIBRARY.items():
            assert len(entry.source) > 20, \
                f"{name}: missing or short source citation"
            assert "DOI" in entry.source or "doi" in entry.source, \
                f"{name}: source lacks DOI"

    def test_all_have_category(self):
        for name, entry in _LIBRARY.items():
            assert entry.category in (
                "MOF", "cage", "zeolite", "synthetic_receptor", "COF"
            ), f"{name}: invalid category '{entry.category}'"

    def test_rebek_packing_auto_computed(self):
        """max_guest_volume should be auto-set to 0.65 × cavity_volume."""
        for name, entry in _LIBRARY.items():
            s = entry.spec
            if s.max_guest_volume_A3 > 0 and s.cavity_volume_A3 > 0:
                # Either auto (0.65×V) or manually set (smaller)
                assert s.max_guest_volume_A3 <= s.cavity_volume_A3, \
                    f"{name}: max_guest > cavity"

    def test_all_aliases_resolve(self):
        for alias, target in _ALIASES.items():
            assert target in _LIBRARY, \
                f"Alias '{alias}' → '{target}' not in library"

    def test_mof_count(self):
        assert len(list_hosts(host_type="MOF")) >= 5

    def test_cage_count(self):
        assert len(list_hosts(host_type="cage")) >= 3


class TestLookupAPI:
    """Test get_host, get_entry, list_hosts, list_names."""

    def test_get_host_direct(self):
        spec = get_host("HKUST-1")
        assert isinstance(spec, NovelHostSpec)
        assert spec.name == "HKUST-1"
        assert spec.cavity_volume_A3 > 0

    def test_get_host_alias(self):
        spec = get_host("Cu-BTC")
        assert spec.name == "HKUST-1"

    def test_get_host_unknown_raises(self):
        with pytest.raises(KeyError):
            get_host("NotAHost-99")

    def test_get_entry_has_metadata(self):
        entry = get_entry("HKUST-1")
        assert isinstance(entry, HostEntry)
        assert entry.formula != ""
        assert entry.source != ""
        assert len(entry.tags) > 0

    def test_list_hosts_by_type(self):
        mofs = list_hosts(host_type="MOF")
        assert all(e.category == "MOF" for e in mofs)

    def test_list_hosts_by_volume(self):
        small = list_hosts(max_volume=500)
        assert all(e.spec.cavity_volume_A3 <= 500 for e in small)

    def test_list_hosts_by_tags(self):
        co2 = list_hosts(tags=["CO2-capture"])
        assert len(co2) >= 2
        assert all("CO2-capture" in e.tags for e in co2)

    def test_list_hosts_sorted_by_volume(self):
        all_hosts = list_hosts()
        vols = [e.spec.cavity_volume_A3 for e in all_hosts]
        assert vols == sorted(vols)

    def test_list_names(self):
        names = list_names()
        assert "HKUST-1" in names
        assert "CC3" in names
        assert len(names) == len(_LIBRARY)

    def test_list_names_filtered(self):
        cage_names = list_names(host_type="cage")
        assert "CC3" in cage_names
        assert "HKUST-1" not in cage_names

    def test_host_summary_runs(self):
        s = host_summary()
        assert "HKUST-1" in s
        assert "Total:" in s


class TestNovelHostSpecPhysics:
    """Verify NovelHostSpec data class behavior."""

    def test_auto_max_guest_volume(self):
        s = NovelHostSpec(cavity_volume_A3=1000.0)
        assert s.max_guest_volume_A3 == pytest.approx(650.0)

    def test_manual_max_guest_volume_preserved(self):
        s = NovelHostSpec(cavity_volume_A3=1000.0, max_guest_volume_A3=300.0)
        assert s.max_guest_volume_A3 == 300.0

    def test_zero_cavity_no_auto(self):
        s = NovelHostSpec(cavity_volume_A3=0.0)
        assert s.max_guest_volume_A3 == 0.0

    def test_host_type_propagated(self):
        s = NovelHostSpec(host_type="MOF")
        assert s.host_type == "MOF"


class TestPhysicalConsistency:
    """Cross-check data against known physical constraints."""

    def test_hkust1_cavity_in_range(self):
        """HKUST-1 large cage: 600–700 ų (literature consensus)."""
        s = get_host("HKUST-1")
        assert 500 < s.cavity_volume_A3 < 800

    def test_zif8_larger_than_hkust1(self):
        """ZIF-8 cage > HKUST-1 cage (known)."""
        assert get_host("ZIF-8").cavity_volume_A3 > get_host("HKUST-1").cavity_volume_A3

    def test_mil101_largest_mof(self):
        """MIL-101 should be largest MOF in library (mesoporous)."""
        mofs = list_hosts(host_type="MOF")
        largest = max(mofs, key=lambda e: e.spec.cavity_volume_A3)
        assert largest.spec.name == "MIL-101(Cr)"

    def test_cc1_smaller_than_cc3(self):
        """CC1 is the smaller cage in the CC-n series."""
        assert get_host("CC1").cavity_volume_A3 < get_host("CC3").cavity_volume_A3

    def test_uio66_nh2_has_hbond(self):
        """UiO-66-NH2 should declare H-bond capability."""
        s = get_host("UiO-66-NH2")
        assert s.n_hbonds_host >= 1

    def test_uio66_no_hbond(self):
        """UiO-66 (unfunctionalized) should have no H-bonds at portal."""
        s = get_host("UiO-66")
        assert s.n_hbonds_host == 0

    def test_charged_hosts_consistent(self):
        """Hosts with framework charges should be cages or receptors."""
        for name, entry in _LIBRARY.items():
            if entry.spec.host_charge != 0:
                assert entry.category in ("cage", "synthetic_receptor"), \
                    f"{name}: nonzero charge in {entry.category}"

    def test_pore_diameter_vs_volume(self):
        """Sanity: pore diameter should be roughly consistent with volume.
        
        For a sphere: V = (4/3)π(d/2)³ → d = 2(3V/(4π))^(1/3)
        Real pores aren't spheres, so just check d is in same ballpark.
        """
        for name, entry in _LIBRARY.items():
            if entry.pore_diameter_A > 0 and entry.spec.cavity_volume_A3 > 0:
                V = entry.spec.cavity_volume_A3
                d_sphere = 2.0 * (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)
                d_actual = entry.pore_diameter_A
                # Allow 5x disagreement (cylinders, channels, mesopores)
                assert d_actual < d_sphere * 5.0, \
                    f"{name}: d={d_actual} too large for V={V}"


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests (require rdkit + full MABE)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from core.de_novo_generator import generate_for_host as _gfh
    HAS_DENOVOGEN = True
except ImportError:
    HAS_DENOVOGEN = False


@pytest.mark.skipif(not HAS_RDKIT or not HAS_DENOVOGEN,
                    reason="Requires rdkit + de_novo_generator")
class TestDeNovoIntegration:
    """Integration: generate_for_host() with library specs."""

    def test_hkust1_guest_generation(self):
        spec = get_host("HKUST-1")
        r = _gfh(spec, max_candidates=30, max_scored=5)
        assert r.n_scored >= 1
        assert r.mode == "host_guest"

    def test_cc3_guest_generation(self):
        spec = get_host("CC3")
        r = _gfh(spec, max_candidates=30, max_scored=5)
        assert r.n_scored >= 1

    def test_exbox_guest_generation(self):
        spec = get_host("ExBox")
        r = _gfh(spec, max_candidates=30, max_scored=5)
        assert r.n_scored >= 1

    def test_volume_filter_active(self):
        """Small hosts should exclude large guests."""
        spec = get_host("CC1")  # 82 Å³ → max_guest ≈ 53 Å³
        r = _gfh(spec, max_candidates=50, max_scored=10)
        # Should have fewer candidates than a large host
        spec_large = get_host("MIL-101(Cr)")
        r_large = _gfh(spec_large, max_candidates=50, max_scored=10)
        assert r.n_scored <= r_large.n_scored

    def test_alias_generates(self):
        """Alias lookup → spec → generation should work end-to-end."""
        spec = get_host("Cu-BTC")  # alias for HKUST-1
        r = _gfh(spec, max_candidates=20, max_scored=3)
        # r.target is the host name string (extracted from spec)
        target_name = r.target if isinstance(r.target, str) else r.target.name
        assert target_name == "HKUST-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
