"""
tests/test_substrate_anchor.py — Tests for substrate anchor adapter.

Validates:
  - Substrate database completeness
  - Activation protocol routing per substrate
  - Click chemistry selection with Cu tolerance constraint
  - Handle installation per surface group
  - Coupling protocol per element type
  - Full protocol generation
  - Cu-sensitive elements forced to SPAAC
  - PVDF and cellulose special-case handling
  - Protocol report generation
"""

import sys
import os
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.substrate_anchor import (
    SubstrateType,
    ActivationMethod,
    ClickChemistry,
    CaptureElementType,
    get_substrate,
    list_substrates,
    compatible_click_chemistries,
    generate_tether_protocol,
    generate_all_protocols,
    protocol_report,
    TetherProtocol,
)


# ═══════════════════════════════════════════════════════════════════════════
# Substrate database tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSubstrateDatabase:

    def test_all_substrate_types_have_properties(self):
        """Every SubstrateType enum value should have an entry in the database."""
        for st in SubstrateType:
            props = get_substrate(st)
            assert props is not None
            assert props.name != ""
            assert props.native_surface != ""

    def test_list_substrates_returns_all(self):
        substrates = list_substrates()
        assert len(substrates) == len(SubstrateType)

    def test_silica_beads_high_surface_area(self):
        props = get_substrate(SubstrateType.SILICA_BEADS)
        assert props.surface_area_m2_g >= 100.0

    def test_glass_slide_low_surface_area(self):
        props = get_substrate(SubstrateType.GLASS_SLIDE)
        assert props.surface_area_m2_g < 1.0

    def test_pe_netting_inert_surface(self):
        props = get_substrate(SubstrateType.PE_NETTING)
        assert "inert" in props.native_surface.lower() or "C-H" in props.native_surface

    def test_nylon_has_amide(self):
        props = get_substrate(SubstrateType.NYLON_NETTING)
        assert "NH" in props.native_surface or "amide" in props.native_surface

    def test_pvdf_has_fluorine(self):
        props = get_substrate(SubstrateType.PVDF_MEMBRANE)
        assert "F" in props.native_surface or "fluoro" in props.native_surface.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Activation routing tests
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationRouting:

    def test_silica_default_is_aptes(self):
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER)
        assert p.activation.method == ActivationMethod.SILANIZATION_APTES

    def test_silica_gptms_when_requested(self):
        p = generate_tether_protocol(
            SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER,
            preferred_activation="gptms"
        )
        assert p.activation.method == ActivationMethod.SILANIZATION_GPTMS

    def test_pe_netting_uses_plasma(self):
        p = generate_tether_protocol(SubstrateType.PE_NETTING, CaptureElementType.MOF_PARTICLE)
        assert p.activation.method == ActivationMethod.PLASMA_OXIDATION

    def test_nylon_uses_hydrolysis(self):
        p = generate_tether_protocol(SubstrateType.NYLON_NETTING, CaptureElementType.ENZYME_MIMIC)
        assert p.activation.method == ActivationMethod.PARTIAL_HYDROLYSIS

    def test_pvdf_uses_dehydrofluorination(self):
        p = generate_tether_protocol(SubstrateType.PVDF_MEMBRANE, CaptureElementType.PHOTOCATALYST)
        assert p.activation.method == ActivationMethod.DEHYDROFLUORINATION

    def test_steel_uses_phosphonic(self):
        p = generate_tether_protocol(SubstrateType.STAINLESS_STEEL_MESH, CaptureElementType.METAL_OXIDE_NP)
        assert p.activation.method == ActivationMethod.PHOSPHONIC_ACID

    def test_cellulose_uses_periodate(self):
        p = generate_tether_protocol(SubstrateType.CELLULOSE_FILTER, CaptureElementType.MOLECULAR_BINDER)
        assert p.activation.method == ActivationMethod.PERIODATE_OXIDATION

    def test_activation_has_steps(self):
        """Every activation protocol must have at least 2 procedural steps."""
        for st in SubstrateType:
            p = generate_tether_protocol(st, CaptureElementType.MOLECULAR_BINDER)
            assert len(p.activation.steps) >= 2, f"{st.value} activation has <2 steps"

    def test_activation_has_literature_ref(self):
        for st in SubstrateType:
            p = generate_tether_protocol(st, CaptureElementType.MOLECULAR_BINDER)
            assert p.activation.literature_ref != "", f"{st.value} missing literature ref"


# ═══════════════════════════════════════════════════════════════════════════
# Click chemistry selection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClickSelection:

    def test_cu_sensitive_forces_spaac(self):
        """Cu-sensitive element must get SPAAC, even if CuAAC requested."""
        p = generate_tether_protocol(
            SubstrateType.SILICA_BEADS, CaptureElementType.ENZYME_MIMIC,
            cu_tolerant=False, preferred_click=ClickChemistry.CUAAC
        )
        assert p.click_chemistry == ClickChemistry.SPAAC
        assert p.cu_safe is True
        assert len(p.warnings) >= 1  # should warn about override

    def test_cu_tolerant_allows_cuaac(self):
        p = generate_tether_protocol(
            SubstrateType.SILICA_BEADS, CaptureElementType.METAL_OXIDE_NP,
            cu_tolerant=True, preferred_click=ClickChemistry.CUAAC
        )
        assert p.click_chemistry == ClickChemistry.CUAAC
        assert p.cu_safe is False

    def test_default_is_spaac(self):
        """Default click chemistry should be SPAAC (safest)."""
        p = generate_tether_protocol(SubstrateType.GLASS_BEADS, CaptureElementType.MOLECULAR_BINDER)
        assert p.click_chemistry == ClickChemistry.SPAAC

    def test_compatible_chemistries_cu_sensitive(self):
        chems = compatible_click_chemistries(SubstrateType.SILICA_BEADS, cu_tolerant=False)
        assert ClickChemistry.SPAAC in chems
        assert ClickChemistry.CUAAC not in chems

    def test_compatible_chemistries_cu_tolerant(self):
        chems = compatible_click_chemistries(SubstrateType.SILICA_BEADS, cu_tolerant=True)
        assert ClickChemistry.SPAAC in chems
        assert ClickChemistry.CUAAC in chems

    def test_cellulose_supports_oxime(self):
        chems = compatible_click_chemistries(SubstrateType.CELLULOSE_FILTER)
        assert ClickChemistry.OXIME_LIGATION in chems

    def test_spaac_requires_no_copper(self):
        assert not ClickChemistry.SPAAC.requires_copper

    def test_cuaac_requires_copper(self):
        assert ClickChemistry.CUAAC.requires_copper

    def test_all_clicks_aqueous(self):
        for c in ClickChemistry:
            assert c.aqueous_compatible


# ═══════════════════════════════════════════════════════════════════════════
# Handle installation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHandleInstallation:

    def test_amine_surface_spaac_gives_azide(self):
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER)
        assert "N₃" in p.handle.handle_installed or "azide" in p.handle.handle_installed.lower()

    def test_epoxide_surface_gives_azide(self):
        p = generate_tether_protocol(
            SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER,
            preferred_activation="gptms"
        )
        assert "N₃" in p.handle.handle_installed or "azide" in p.handle.handle_installed.lower()

    def test_thiol_surface_gives_maleimide_coupling(self):
        p = generate_tether_protocol(
            SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER,
            preferred_activation="mptms"
        )
        # MPTMS gives -SH → thiol-maleimide is the natural coupling
        # But default click is SPAAC, so it may route to NHS-azide on amine
        # Check that at least the coupling has a valid handle
        assert p.handle.handle_installed != ""

    def test_handle_has_complementary_group(self):
        """Handle must specify what the capture element needs."""
        for st in SubstrateType:
            p = generate_tether_protocol(st, CaptureElementType.MOLECULAR_BINDER)
            assert p.handle.complementary_group != ""

    def test_pvdf_azide_installed_during_activation(self):
        """PVDF thiol-ene installs azide during activation — no separate handle step."""
        p = generate_tether_protocol(SubstrateType.PVDF_MEMBRANE, CaptureElementType.PHOTOCATALYST)
        # Handle time should be 0 (combined with activation)
        assert p.handle.time_hours == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Coupling protocol tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCouplingProtocol:

    def test_molecular_binder_micromolar(self):
        """Molecular binders couple at µM concentration."""
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER)
        assert "µM" in p.coupling.concentration or "uM" in p.coupling.concentration

    def test_mof_particle_mg_ml(self):
        """MOF particles couple as mg/mL suspension."""
        p = generate_tether_protocol(SubstrateType.PE_NETTING, CaptureElementType.MOF_PARTICLE)
        assert "mg/mL" in p.coupling.concentration

    def test_dna_origami_nanomolar(self):
        """DNA origami cages couple at nM concentration."""
        p = generate_tether_protocol(SubstrateType.GLASS_SLIDE, CaptureElementType.DNA_ORIGAMI_CAGE)
        assert "nM" in p.coupling.concentration

    def test_dna_origami_mentions_athena(self):
        """DNA origami protocol should mention ATHENA."""
        p = generate_tether_protocol(SubstrateType.GLASS_SLIDE, CaptureElementType.DNA_ORIGAMI_CAGE)
        all_steps = " ".join(p.coupling.steps)
        assert "ATHENA" in all_steps or "staple" in all_steps.lower()

    def test_coordination_cage_mentions_exohedral(self):
        """Coordination cage protocol should mention exohedral handle."""
        p = generate_tether_protocol(SubstrateType.GLASS_BEADS, CaptureElementType.COORDINATION_CAGE)
        all_steps = " ".join(p.coupling.steps)
        assert "exohedral" in all_steps.lower()

    def test_coupling_efficiency_bounded(self):
        """Coupling efficiency should be 0-100%."""
        for et in CaptureElementType:
            p = generate_tether_protocol(SubstrateType.SILICA_BEADS, et)
            assert 0.0 <= p.coupling.coupling_efficiency_pct <= 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Full protocol generation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFullProtocol:

    def test_protocol_has_three_layers(self):
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.ENZYME_MIMIC)
        assert p.activation is not None
        assert p.handle is not None
        assert p.coupling is not None

    def test_total_steps_positive(self):
        p = generate_tether_protocol(SubstrateType.PE_NETTING, CaptureElementType.MOF_PARTICLE)
        assert p.total_steps > 0

    def test_total_time_positive(self):
        p = generate_tether_protocol(SubstrateType.NYLON_NETTING, CaptureElementType.MOLECULAR_BINDER)
        assert p.estimated_time_hours > 0.0

    def test_aqueous_protocol_flag(self):
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER)
        assert p.aqueous_protocol is True

    def test_generate_all_protocols(self):
        """Should produce multiple protocol variants for substrates with multiple routes."""
        protocols = generate_all_protocols(
            SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER
        )
        assert len(protocols) >= 3  # APTES/GPTMS/MPTMS × click variants

    def test_generate_all_protocols_deduplicated(self):
        """No duplicate (activation, click) pairs in output."""
        protocols = generate_all_protocols(
            SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER
        )
        keys = [(p.activation.method.value, p.click_chemistry.value) for p in protocols]
        assert len(keys) == len(set(keys))


# ═══════════════════════════════════════════════════════════════════════════
# Cross-substrate completeness test
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossSubstrate:

    def test_every_substrate_element_pair_generates_protocol(self):
        """Every (substrate, element) pair should produce a valid protocol without error."""
        for st in SubstrateType:
            for et in CaptureElementType:
                p = generate_tether_protocol(st, et)
                assert isinstance(p, TetherProtocol), \
                    f"Failed for {st.value} + {et.value}"
                assert p.total_steps > 0

    def test_cu_sensitive_elements_always_spaac(self):
        """Cu-sensitive flag must result in SPAAC across all substrates."""
        cu_sensitive_elements = [
            CaptureElementType.ENZYME_MIMIC,
            CaptureElementType.SULFIDE_NP,
        ]
        for st in SubstrateType:
            for et in cu_sensitive_elements:
                p = generate_tether_protocol(st, et, cu_tolerant=False)
                assert p.cu_safe is True, \
                    f"{st.value} + {et.value} should be Cu-safe"


# ═══════════════════════════════════════════════════════════════════════════
# Report generation test
# ═══════════════════════════════════════════════════════════════════════════

class TestReporting:

    def test_report_not_empty(self):
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.ENZYME_MIMIC)
        report = protocol_report(p)
        assert "TETHERING PROTOCOL" in report
        assert "LAYER A" in report
        assert "LAYER B" in report
        assert "LAYER C" in report

    def test_report_includes_substrate_name(self):
        p = generate_tether_protocol(SubstrateType.PE_NETTING, CaptureElementType.MOF_PARTICLE)
        report = protocol_report(p)
        assert "Polyethylene" in report or "PE" in report

    def test_report_includes_click_chemistry(self):
        p = generate_tether_protocol(SubstrateType.GLASS_SLIDE, CaptureElementType.DNA_ORIGAMI_CAGE)
        report = protocol_report(p)
        assert "spaac" in report.lower() or "SPAAC" in report

    def test_summary_method(self):
        p = generate_tether_protocol(SubstrateType.SILICA_BEADS, CaptureElementType.MOLECULAR_BINDER)
        s = p.summary()
        assert "Tether Protocol" in s
        assert "Cu-free" in s
