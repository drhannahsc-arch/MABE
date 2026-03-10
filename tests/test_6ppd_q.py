"""
tests/test_6ppd_q.py — Integration test: 6PPD-Quinone through MABE pipeline.

Tests the full pathway:
    SMILES → pharmacophore → pocket spec → host screening → MIP design

6PPD-Quinone (N-(1,3-dimethylbutyl)-N'-phenyl-p-benzoquinone diimine):
    - Tire rubber antioxidant transformation product
    - Acutely toxic to coho salmon at low ng/L
    - No validated binders exist → MABE test case for environmental remediation

SMILES: CC(C)CC(NC1=CC(=O)C(=CC1=O)NC2=CC=CC=C2)C
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 6PPD-Q canonical SMILES
SMILES_6PPD_Q = "CC(C)CC(NC1=CC(=O)C(=CC1=O)NC2=CC=CC=C2)C"
NAME_6PPD_Q = "6PPD-quinone"


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1: Pharmacophore analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestPharmacophore:
    """Test guest pharmacophore extraction for 6PPD-Q."""

    def test_analyze_guest_runs(self):
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q, name=NAME_6PPD_Q)
        assert pharma is not None
        assert pharma.smiles == SMILES_6PPD_Q
        assert pharma.name == NAME_6PPD_Q

    def test_molecular_properties(self):
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)

        # MW of 6PPD-Q: ~298 g/mol
        assert 290 < pharma.mw < 310, f"MW={pharma.mw}"
        # LogP: moderately lipophilic (~3)
        assert 2.0 < pharma.logP < 5.0, f"LogP={pharma.logP}"
        # Volume: ~250-330 Å³
        assert 200 < pharma.volume_A3 < 400, f"Vol={pharma.volume_A3}"

    def test_hbond_donors_detected(self):
        """6PPD-Q has 2 N-H groups (amine donors)."""
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)
        assert pharma.n_hb_donors >= 2, (
            f"Expected ≥2 H-bond donors (2 NH), got {pharma.n_hb_donors}"
        )

    def test_hbond_acceptors_detected(self):
        """6PPD-Q has 2 quinone C=O groups."""
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)
        assert pharma.n_hb_acceptors >= 2, (
            f"Expected ≥2 H-bond acceptors (2 C=O), got {pharma.n_hb_acceptors}"
        )

    def test_aromatic_ring_detected(self):
        """6PPD-Q has a phenyl ring."""
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)
        assert pharma.n_aromatic_rings >= 1, (
            f"Expected ≥1 aromatic ring (phenyl), got {pharma.n_aromatic_rings}"
        )

    def test_hydrophobic_regions_detected(self):
        """6PPD-Q has an isobutyl chain."""
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)
        assert pharma.n_hydrophobic_centers >= 1, (
            f"Expected ≥1 hydrophobic center (isobutyl), got {pharma.n_hydrophobic_centers}"
        )

    def test_3d_positions_populated(self):
        """Features should have non-zero 3D positions."""
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)
        has_nonzero = any(
            any(abs(c) > 0.01 for c in f.position_A)
            for f in pharma.features
        )
        assert has_nonzero, "All features at origin — 3D embedding failed"

    def test_dominant_interaction_mixed(self):
        """6PPD-Q has both H-bonds and aromatic → mixed or h_bond."""
        from core.small_molecule_target import analyze_guest
        pharma = analyze_guest(SMILES_6PPD_Q)
        assert pharma.dominant_interaction in ("mixed", "h_bond", "hydrophobic"), (
            f"dominant_interaction={pharma.dominant_interaction}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1→2: Pocket spec generation
# ═══════════════════════════════════════════════════════════════════════════

class TestPocketSpec:
    """Test InteractionGeometrySpec generation from 6PPD-Q pharmacophore."""

    def test_pocket_spec_generated(self):
        from core.small_molecule_target import analyze_guest, guest_to_pocket_spec
        pharma = analyze_guest(SMILES_6PPD_Q)
        spec = guest_to_pocket_spec(pharma, application="remediation")
        assert spec is not None
        assert spec.cavity_dimensions.volume_A3 > 0

    def test_cavity_volume_rebek_rule(self):
        """Cavity volume should be ~guest_vol / 0.55 (Rebek rule)."""
        from core.small_molecule_target import analyze_guest, guest_to_pocket_spec
        pharma = analyze_guest(SMILES_6PPD_Q)
        spec = guest_to_pocket_spec(pharma)

        expected_vol = pharma.volume_A3 / 0.55
        actual_vol = spec.cavity_dimensions.volume_A3
        # Allow 5% tolerance
        assert abs(actual_vol - expected_vol) / expected_vol < 0.05, (
            f"Cavity vol {actual_vol:.0f} vs expected {expected_vol:.0f}"
        )

    def test_donor_positions_complement_guest(self):
        """Pocket should have donor positions complementing guest features."""
        from core.small_molecule_target import analyze_guest, guest_to_pocket_spec
        pharma = analyze_guest(SMILES_6PPD_Q)
        spec = guest_to_pocket_spec(pharma)

        # Should have donor positions for guest's H-bond donors AND acceptors
        n_donors = len(spec.donor_positions)
        n_guest_hb_features = pharma.n_hb_donors + pharma.n_hb_acceptors
        assert n_donors >= n_guest_hb_features, (
            f"Expected ≥{n_guest_hb_features} pocket donor positions, got {n_donors}"
        )

    def test_hydrophobic_surfaces_present(self):
        """Pocket should have hydrophobic surfaces for aromatic + alkyl."""
        from core.small_molecule_target import analyze_guest, guest_to_pocket_spec
        pharma = analyze_guest(SMILES_6PPD_Q)
        spec = guest_to_pocket_spec(pharma)
        assert len(spec.hydrophobic_surfaces) >= 1

    def test_application_context_propagates(self):
        from core.small_molecule_target import analyze_guest, guest_to_pocket_spec
        from realization.models import ApplicationContext
        pharma = analyze_guest(SMILES_6PPD_Q)
        spec = guest_to_pocket_spec(pharma, application="remediation")
        assert spec.target_application == ApplicationContext.REMEDIATION


# ═══════════════════════════════════════════════════════════════════════════
# Host screening via unified_scorer_v2
# ═══════════════════════════════════════════════════════════════════════════

class TestHostScreening:
    """Test host-guest scoring of 6PPD-Q against cyclodextrins etc."""

    def test_screen_hosts_runs(self):
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q, name=NAME_6PPD_Q)
        assert len(results) > 0

    def test_all_hosts_scored(self):
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q)
        # Should score all hosts in HOST_DB
        assert len(results) >= 5, f"Only {len(results)} hosts scored"

    def test_results_sorted_descending(self):
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q)
        log_kas = [r.log_Ka_pred for r in results if r.log_Ka_pred > float("-inf")]
        assert log_kas == sorted(log_kas, reverse=True)

    def test_beta_cd_positive_log_ka(self):
        """β-CD should show positive log_Ka for 6PPD-Q (moderate binding)."""
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q, hosts=["beta-CD"])
        assert len(results) == 1
        assert results[0].log_Ka_pred > 0, (
            f"β-CD log_Ka={results[0].log_Ka_pred} — expected positive"
        )

    def test_alpha_cd_size_mismatch(self):
        """α-CD should show poor packing (6PPD-Q too large for α-CD cavity)."""
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q, hosts=["alpha-CD"])
        assert len(results) == 1
        # Packing coefficient >1 means guest too large
        assert results[0].packing_coefficient > 1.0 or results[0].dg_size_mismatch > 0

    def test_gamma_cd_better_than_alpha_cd(self):
        """γ-CD should bind 6PPD-Q more strongly than α-CD (larger cavity)."""
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q, hosts=["alpha-CD", "gamma-CD"])
        by_host = {r.host_key: r for r in results}
        assert by_host["gamma-CD"].log_Ka_pred > by_host["alpha-CD"].log_Ka_pred

    def test_energy_decomposition_populated(self):
        """All energy terms should be populated for β-CD."""
        from core.small_molecule_target import screen_hosts
        results = screen_hosts(SMILES_6PPD_Q, hosts=["beta-CD"])
        r = results[0]
        # At least hydrophobic and cavity dehydration should be nonzero
        assert r.dg_hydrophobic != 0.0
        assert r.dg_cavity_dehydration != 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: MIP adapter
# ═══════════════════════════════════════════════════════════════════════════

class TestMIPAdapter:
    """Test MIP monomer selection for 6PPD-Q."""

    def test_select_monomers_runs(self):
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(SMILES_6PPD_Q, guest_name=NAME_6PPD_Q)
        assert design is not None
        assert design.guest_smiles == SMILES_6PPD_Q

    def test_primary_monomers_selected(self):
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(SMILES_6PPD_Q)
        assert len(design.primary_monomers) >= 2, (
            f"Only {len(design.primary_monomers)} primary monomers"
        )

    def test_quinone_specific_monomers_ranked_high(self):
        """Quinone guests should rank CT/thiol monomers highly."""
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(SMILES_6PPD_Q)
        primary_abbrevs = {m.monomer.abbreviation for m in design.primary_monomers}
        # Should include at least one of: EDOT, Py, MEMA, DMA, oPD
        quinone_targeted = primary_abbrevs & {"EDOT", "Py", "MEMA", "DMA", "oPD"}
        assert len(quinone_targeted) >= 1, (
            f"No quinone-targeted monomer in primary: {primary_abbrevs}"
        )

    def test_diverse_interaction_types(self):
        """Primary monomers should cover different interaction types."""
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(SMILES_6PPD_Q)
        provides_types = {m.monomer.provides for m in design.primary_monomers}
        assert len(provides_types) >= 2, (
            f"Monomer types not diverse: {provides_types}"
        )

    def test_imprinting_factor_positive(self):
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(SMILES_6PPD_Q)
        lo, hi = design.predicted_if_range
        assert lo >= 1.0 and hi > lo

    def test_synthesis_steps_generated(self):
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(SMILES_6PPD_Q)
        assert len(design.synthesis_steps) >= 5

    def test_electrochemical_mode(self):
        """With prefer_electroactive, should select conducting monomers."""
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(
            SMILES_6PPD_Q, prefer_electroactive=True
        )
        has_electroactive = any(
            m.monomer.electroactive for m in design.primary_monomers
        )
        assert has_electroactive, "No electroactive monomer selected"

    def test_click_deployable(self):
        from adapters.mip_adapter import select_monomers_for_guest
        design = select_monomers_for_guest(
            SMILES_6PPD_Q, require_click=True
        )
        assert design.click_deployable
        assert design.recommended_crosslinker == "BA-PEG-DMA"


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: Full pipeline via design_for_guest()
# ═══════════════════════════════════════════════════════════════════════════

class TestDesignForGuest:
    """Test the full pipeline entry point."""

    def test_design_for_guest_runs(self):
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(
            smiles=SMILES_6PPD_Q,
            name=NAME_6PPD_Q,
            application="remediation",
        )
        assert result.pipeline_complete

    def test_all_stages_populated(self):
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(SMILES_6PPD_Q, name=NAME_6PPD_Q)

        assert result.pharmacophore is not None, "Pharmacophore missing"
        assert result.pocket_spec is not None, "Pocket spec missing"
        assert len(result.host_screen) > 0, "Host screen empty"
        assert result.mip_design is not None, "MIP design missing"

    def test_top_host_identified(self):
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(SMILES_6PPD_Q, name=NAME_6PPD_Q)
        assert result.top_host != "", "No top host identified"
        assert result.top_host_log_ka > 0, "Top host log_Ka not positive"

    def test_conditions_propagate(self):
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(
            SMILES_6PPD_Q,
            name=NAME_6PPD_Q,
            conditions={"pH": 6.5, "matrix": "stormwater"},
            application="remediation",
        )
        assert result.conditions["pH"] == 6.5
        assert result.application == "remediation"

    def test_mip_disabled(self):
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(
            SMILES_6PPD_Q, include_mip=False,
        )
        assert result.mip_design is None
        assert result.pipeline_complete

    def test_architecture_layers_preserved(self):
        """Verify no layer-collapsing: pocket spec is realization-agnostic."""
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(SMILES_6PPD_Q, name=NAME_6PPD_Q)

        spec = result.pocket_spec
        # Pocket spec should NOT mention any specific scaffold
        # (no "DNA", "protein", "MOF" in the spec itself)
        spec_str = str(spec)
        for scaffold_word in ["DNA", "protein", "MOF", "polymer", "aptamer"]:
            assert scaffold_word not in spec_str, (
                f"Layer 2 spec mentions scaffold '{scaffold_word}' — architecture violation"
            )


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 5: De novo receptor generation
# ═══════════════════════════════════════════════════════════════════════════

class TestDeNovoGenerator:
    """Test de novo receptor generation for 6PPD-Q."""

    def test_generate_for_guest_runs(self):
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, guest_name=NAME_6PPD_Q,
            max_candidates=100, max_scored=10,
        )
        assert result is not None
        assert result.mode == "receptor"
        assert result.n_enumerated > 0

    def test_candidates_generated(self):
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=100, max_scored=10,
        )
        assert result.n_scored > 0, "No candidates scored"
        assert len(result.candidates) > 0

    def test_candidates_ranked(self):
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=100, max_scored=10,
        )
        composites = [c.composite_score for c in result.candidates]
        assert composites == sorted(composites, reverse=True)

    def test_candidates_have_valid_smiles(self):
        from core.de_novo_generator import generate_for_guest
        from rdkit import Chem
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=100, max_scored=10,
        )
        for c in result.candidates[:5]:
            mol = Chem.MolFromSmiles(c.smiles)
            assert mol is not None, f"Invalid SMILES: {c.smiles}"

    def test_complementarity_scores_positive(self):
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=100, max_scored=10,
        )
        for c in result.candidates:
            assert c.complementarity_score > 0

    def test_receptor_backbones_used(self):
        """Should use receptor-oriented backbones (tweezers, clefts)."""
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=200, max_scored=20,
        )
        backbone_names = {c.backbone_name for c in result.candidates}
        receptor_types = {"glycoluril-clip", "xanthene-tweezer",
                         "isophthalamide", "squaramide-cleft",
                         "pyridine-2,6-diamide", "urea-cleft",
                         "dibenzofuran-tweezer", "Troeger-base"}
        overlap = backbone_names & receptor_types
        assert len(overlap) >= 1, (
            f"No receptor backbones used. Got: {backbone_names}"
        )

    def test_complementary_arms_selected(self):
        """Arms should complement 6PPD-Q's pharmacophore (quinone acceptors → donor arms)."""
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=200, max_scored=20,
        )
        all_arms = set()
        for c in result.candidates:
            all_arms.update(c.arm_names)
        # Should include H-bond donor arms for quinone C=O
        donor_arms = {"catechol", "phenol", "acetohydroxamate",
                      "squaramide", "acetic-acid", "ethanol"}
        overlap = all_arms & donor_arms
        assert len(overlap) >= 1, (
            f"No H-bond donor arms found. Got: {all_arms}"
        )

    def test_sa_scores_reasonable(self):
        from core.de_novo_generator import generate_for_guest
        result = generate_for_guest(
            SMILES_6PPD_Q, max_candidates=100, max_scored=10,
        )
        for c in result.candidates:
            assert 1.0 <= c.sa_score_val <= 10.0

    def test_pipeline_includes_de_novo(self):
        """design_for_guest should include de novo results."""
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(
            SMILES_6PPD_Q, name=NAME_6PPD_Q,
            de_novo_max_candidates=100, de_novo_max_scored=10,
        )
        assert result.de_novo_result is not None
        assert result.de_novo_result.n_scored > 0

    def test_pipeline_de_novo_disabled(self):
        from core.physics_realization_bridge import design_for_guest
        result = design_for_guest(
            SMILES_6PPD_Q, include_de_novo=False,
        )
        assert result.de_novo_result is None
        assert result.pipeline_complete
