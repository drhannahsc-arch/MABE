"""
core/cross_modal_ranker.py — Universal Affinity Score (UAS) Cross-Modal Ranker

Normalizes binding predictions from all material system adapters onto a
common scale: UAS = −log₁₀(Kd_predicted), equivalent to log₁₀(Ka).

Each modality has its own score-to-UAS conversion, calibrated against
known anchor systems with experimentally measured Ka values:

    Host-guest (CD/CB/calix):  UAS = log_Ka_pred directly (already calibrated)
    MIP:                       UAS = log₁₀(IF × Ka_NIP) where Ka_NIP ≈ 30 M⁻¹
    De novo receptor:          UAS = complementarity × scaling from Davis receptors
    DNA origami tertiary:      UAS = −ΔG_total / (RT·ln10) = −ΔG / 5.71
    MOF:                       UAS from cavity match × donor match × anchor
    Coordination cage:         UAS from size match × donor match × anchor
    Porphyrin:                 UAS = estimated_log_Ka_metal (or guest-mode estimate)

Confidence intervals from per-modality calibration RMSE:
    Host-guest: ±1.3 log units (from HG regression MAE)
    MIP: ±1.5 log units (heterogeneous binding)
    De novo: ±2.0 log units (no direct calibration)
    DNA origami: ±2.5 log units (additive module approximation)
    MOF: ±2.0 log units (limited binding data for guest inclusion)
    Cage: ±1.5 log units (Fujita/Nitschke binding data exists)
    Porphyrin: ±1.0 log units (extensive metalloporphyrin data)

Does NOT:
    - Use any cross-modal fitted parameters
    - Claim precision beyond the confidence interval
    - Assume modalities are directly comparable in deployment context
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION ANCHORS
# ═══════════════════════════════════════════════════════════════════════════
# Each modality has anchor systems with known Ka, used to map adapter
# scores to the UAS scale.

# Host-guest: already calibrated. UAS = log_Ka_pred.
# Confidence: MAE 1.31 from HG regression (knowledge/hg_dataset.py)
HG_RMSE = 1.3

# MIP: imprinting factor (IF) relates to selectivity, not absolute Ka.
# Anchor: IF=1 means Ka_MIP ≈ Ka_NIP (non-imprinted polymer).
# Literature: Ka_NIP for organic guests in water ≈ 10-100 M⁻¹ (Sellergren 2001).
# UAS_MIP = log₁₀(IF × Ka_NIP), using Ka_NIP = 30 M⁻¹ as geometric mean.
MIP_KA_NIP = 30.0  # M⁻¹ (baseline for non-imprinted polymer)
MIP_RMSE = 1.5

# De novo small-molecule receptor: complementarity score.
# Anchor: Davis synthetic lectins (Nature Chem 2012) achieve Ka ≈ 10²-10⁴
# with complementarity-driven design. Score ~9.5 (our max) maps to
# ~log Ka 4 (well-designed receptor). Linear mapping.
DENOVO_COMP_TO_LOG_KA = 0.42  # log_Ka ≈ complementarity × 0.42
DENOVO_RMSE = 2.0

# DNA origami tertiary: sum of module interaction energies.
# UAS = −ΔG_total / 5.71 (standard thermodynamic conversion).
# Confidence is wide: additive approximation ignores cooperativity,
# solvation, linker entropy.
DNA_ORIGAMI_RMSE = 2.5

# MOF: composite score is multi-axis (cavity + donor + stability + scalability).
# Anchor: UiO-66-NH₂ with PSM for Pb²⁺ removal: Ka ≈ 10⁴ (literature).
# Composite ~8.7 → log Ka ~4. Scale: UAS ≈ composite × 0.46.
MOF_COMPOSITE_TO_LOG_KA = 0.46
MOF_RMSE = 2.0

# Coordination cage: composite score.
# Anchor: Fujita Pd₂L₄ with adamantane: Ka ≈ 10⁴ in water.
# Composite ~6.7 → log Ka ~4. Scale: UAS ≈ composite × 0.60.
CAGE_COMPOSITE_TO_LOG_KA = 0.60
CAGE_RMSE = 1.5

# Porphyrin: estimated_log_Ka_metal is already in log units.
# For guest-binding mode: composite × scaling.
PORPHYRIN_GUEST_SCALE = 0.45  # composite → log Ka for guest binding
PORPHYRIN_RMSE = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# DEPLOYMENT READINESS SCORING
# ═══════════════════════════════════════════════════════════════════════════

def _deployment_readiness(modality, design=None):
    """Score deployment readiness (0-1).

    Higher = closer to deployable product.
    Based on: synthesis maturity, scalability, cost, regulatory precedent.
    """
    base_scores = {
        "host_guest_CD": 0.95,       # commercial, FDA-approved excipient
        "host_guest_CB": 0.60,       # research stage, limited commercial
        "host_guest_calix": 0.50,    # research stage
        "host_guest_pillar": 0.40,   # early research
        "mip": 0.80,                 # commercial MIP products exist (SPE cartridges)
        "de_novo_receptor": 0.30,    # needs custom synthesis
        "dna_origami": 0.25,         # lab-scale only, expensive
        "mof": 0.70,                 # industrial MOFs exist (BASF Basolite)
        "coordination_cage": 0.20,   # academic only
        "porphyrin_metal": 0.85,     # commercial metalloporphyrins
        "porphyrin_guest": 0.40,     # less precedent for guest binding
    }
    return base_scores.get(modality, 0.3)


def _click_handle_available(modality, design=None):
    """Check if click-chemistry deployment handle is available."""
    if modality == "mip":
        return design.click_deployable if design and hasattr(design, 'click_deployable') else False
    if modality.startswith("host_guest"):
        # CDs: C6-azide well-precedented
        return True
    if modality == "dna_origami":
        return True  # staple extensions are inherently click-compatible
    if modality == "mof":
        # Surface azide/alkyne via PSM
        return design is not None and hasattr(design, 'psm') and len(design.psm) > 0
    if modality == "coordination_cage":
        return False  # not standard yet
    if modality.startswith("porphyrin"):
        return design is not None and hasattr(design, 'has_click_handle') and design.has_click_handle
    if modality == "de_novo_receptor":
        return False  # depends on specific design
    return False


# ═══════════════════════════════════════════════════════════════════════════
# UAS ENTRY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UASEntry:
    """One candidate in the cross-modal comparison."""
    rank: int = 0
    modality: str = ""               # "host_guest_CD", "mip", "mof", etc.
    material_name: str = ""          # "γ-CD", "MEMA+EDOT MIP", "UiO-66-NH₂", etc.

    # Universal Affinity Score
    uas: float = 0.0                 # −log₁₀(Kd) = log₁₀(Ka)
    uas_lower: float = 0.0          # UAS − RMSE (68% CI lower)
    uas_upper: float = 0.0          # UAS + RMSE (68% CI upper)
    confidence: str = ""             # "high", "medium", "low"

    # Selectivity
    selectivity_worst: float = 0.0   # worst-case selectivity ratio
    selective: bool = False          # all interferents < target

    # Deployment
    deployment_readiness: float = 0.0
    click_handle: bool = False
    estimated_cost_usd: float = 0.0
    scale_feasibility: str = ""      # "industrial", "pilot", "lab", "research"

    # Source data
    raw_score: float = 0.0          # original adapter score
    raw_score_type: str = ""        # "log_Ka", "IF", "complementarity", etc.
    notes: str = ""


@dataclass
class CrossModalResult:
    """Complete cross-modal comparison for a guest molecule."""
    guest_name: str
    guest_smiles: str
    n_modalities: int = 0
    n_candidates: int = 0
    entries: list = field(default_factory=list)  # list[UASEntry] sorted by UAS

    # Summary
    best_entry: object = None       # UASEntry
    best_selective: object = None   # best entry that is also selective
    best_deployable: object = None  # best entry with deployment_readiness > 0.5
    best_value: object = None       # best UAS/cost ratio


# ═══════════════════════════════════════════════════════════════════════════
# CORE RANKER
# ═══════════════════════════════════════════════════════════════════════════

def rank_cross_modal(design_result) -> CrossModalResult:
    """Produce cross-modal UAS ranking from a design_for_guest() result.

    Args:
        design_result: GuestDesignResult from design_for_guest()

    Returns:
        CrossModalResult with all candidates on UAS scale.
    """
    r = design_result
    result = CrossModalResult(
        guest_name=r.guest_name,
        guest_smiles=r.guest_smiles,
    )

    entries = []
    modalities = set()

    # ── Host-guest screening ──
    if r.host_screen:
        # Get selectivity data if available
        sel_by_host = {}
        if r.selectivity_result:
            for hs in r.selectivity_result.host_selectivity:
                sel_by_host[hs.host_key] = hs

        for h in r.host_screen:
            if h.log_Ka_pred == float("-inf") or h.log_Ka_pred == float("inf"):
                continue
            if h.feasibility_note and "error" in h.feasibility_note:
                continue

            # Classify sub-modality
            if h.host_key.endswith("-CD"):
                mod = "host_guest_CD"
            elif h.host_key.startswith("CB"):
                mod = "host_guest_CB"
            elif "calix" in h.host_key:
                mod = "host_guest_calix"
            elif "pillar" in h.host_key:
                mod = "host_guest_pillar"
            else:
                mod = "host_guest_other"

            uas = h.log_Ka_pred

            # Selectivity from screening
            sel_data = sel_by_host.get(h.host_key)
            worst_sel = sel_data.worst_ratio if sel_data else 0.0
            is_selective = sel_data.is_selective if sel_data else False

            # Cost: CDs are cheap, CBs are expensive
            if "CD" in h.host_key:
                cost = 5.0  # $5 for CD powder
            elif h.host_key.startswith("CB"):
                cost = 200.0
            else:
                cost = 50.0

            entries.append(UASEntry(
                modality=mod,
                material_name=h.host_key,
                uas=uas,
                uas_lower=uas - HG_RMSE,
                uas_upper=uas + HG_RMSE,
                confidence="high" if abs(uas) < 10 else "medium",
                selectivity_worst=worst_sel,
                selective=is_selective,
                deployment_readiness=_deployment_readiness(mod),
                click_handle=_click_handle_available(mod),
                estimated_cost_usd=cost,
                scale_feasibility="industrial" if "CD" in h.host_key else "lab",
                raw_score=h.log_Ka_pred,
                raw_score_type="log_Ka",
                notes=h.feasibility_note,
            ))
            modalities.add(mod)

    # ── MIP ──
    if r.mip_design:
        mip = r.mip_design
        if_mid = (mip.predicted_if_range[0] + mip.predicted_if_range[1]) / 2
        uas_mip = math.log10(if_mid * MIP_KA_NIP)
        if_lo = mip.predicted_if_range[0]
        if_hi = mip.predicted_if_range[1]
        uas_lo = math.log10(if_lo * MIP_KA_NIP) - MIP_RMSE
        uas_hi = math.log10(if_hi * MIP_KA_NIP) + MIP_RMSE

        # MIP selectivity from selectivity_result
        mip_sel = 1.0
        if r.selectivity_result and r.selectivity_result.mip_selectivity:
            mip_sel = min(r.selectivity_result.mip_selectivity.values())

        monomer_str = "+".join(m.monomer.abbreviation for m in mip.primary_monomers)
        name = f"MIP ({monomer_str})"
        if mip.electrochemical_sensor:
            name += " [electrochemical]"

        entries.append(UASEntry(
            modality="mip",
            material_name=name,
            uas=uas_mip,
            uas_lower=uas_lo,
            uas_upper=uas_hi,
            confidence="medium",
            selectivity_worst=mip_sel,
            selective=mip_sel > 1.0,
            deployment_readiness=_deployment_readiness("mip"),
            click_handle=_click_handle_available("mip", mip),
            estimated_cost_usd=20.0,  # MIP reagents
            scale_feasibility="industrial",
            raw_score=if_mid,
            raw_score_type="imprinting_factor",
        ))
        modalities.add("mip")

    # ── De novo receptors ──
    if r.de_novo_result and r.de_novo_result.candidates:
        for cand in r.de_novo_result.candidates[:3]:  # top 3
            uas_dn = cand.complementarity_score * DENOVO_COMP_TO_LOG_KA

            entries.append(UASEntry(
                modality="de_novo_receptor",
                material_name=cand.name,
                uas=uas_dn,
                uas_lower=uas_dn - DENOVO_RMSE,
                uas_upper=uas_dn + DENOVO_RMSE,
                confidence="low",
                selectivity_worst=0.0,  # not computed for de novo
                selective=False,
                deployment_readiness=_deployment_readiness("de_novo_receptor"),
                click_handle=False,
                estimated_cost_usd=500.0,  # custom synthesis
                scale_feasibility="lab",
                raw_score=cand.complementarity_score,
                raw_score_type="complementarity",
                notes=f"SA={cand.sa_score_val:.1f}",
            ))
        modalities.add("de_novo_receptor")

    # ── DNA origami tertiary ──
    if r.dna_origami_design and r.dna_origami_design.feasibility_grade != "infeasible":
        dna = r.dna_origami_design
        uas_dna = dna.estimated_log_Ka

        entries.append(UASEntry(
            modality="dna_origami",
            material_name=f"DNA origami ({dna.cage.name if dna.cage else 'unknown'})",
            uas=uas_dna,
            uas_lower=uas_dna - DNA_ORIGAMI_RMSE,
            uas_upper=uas_dna + DNA_ORIGAMI_RMSE,
            confidence="low",
            selectivity_worst=0.0,  # not computed
            selective=False,
            deployment_readiness=_deployment_readiness("dna_origami"),
            click_handle=True,
            estimated_cost_usd=dna.estimated_cost_usd,
            scale_feasibility="lab",
            raw_score=dna.total_interaction_energy_kJ,
            raw_score_type="dG_kJ",
            notes=f"{dna.n_modules} modules, {dna.feasibility_grade}",
        ))
        modalities.add("dna_origami")

    # ── MOF ──
    if r.mof_designs:
        for mof in r.mof_designs[:3]:  # top 3
            uas_mof = mof.composite_score * MOF_COMPOSITE_TO_LOG_KA

            psm_str = ""
            if mof.psm:
                psm_str = " + " + "/".join(p.name for p in mof.psm)

            entries.append(UASEntry(
                modality="mof",
                material_name=f"{mof.topology_name}{psm_str}",
                uas=uas_mof,
                uas_lower=uas_mof - MOF_RMSE,
                uas_upper=uas_mof + MOF_RMSE,
                confidence="medium",
                selectivity_worst=0.0,
                selective=False,
                deployment_readiness=_deployment_readiness("mof"),
                click_handle=_click_handle_available("mof", mof),
                estimated_cost_usd=mof.estimated_cost_per_kg * 0.01,  # 10g
                scale_feasibility="industrial" if mof.scalability_score > 0.7 else "pilot",
                raw_score=mof.composite_score,
                raw_score_type="composite",
                notes=f"node={mof.node.name}, SA={mof.topology.surface_area_m2_g}m²/g",
            ))
        modalities.add("mof")

    # ── Coordination cage ──
    if r.cage_designs:
        for cage in r.cage_designs[:3]:  # top 3
            uas_cage = cage.composite_score * CAGE_COMPOSITE_TO_LOG_KA

            entries.append(UASEntry(
                modality="coordination_cage",
                material_name=f"{cage.formula} ({cage.topology.exemplar})",
                uas=uas_cage,
                uas_lower=uas_cage - CAGE_RMSE,
                uas_upper=uas_cage + CAGE_RMSE,
                confidence="medium",
                selectivity_worst=0.0,
                selective=False,
                deployment_readiness=_deployment_readiness("coordination_cage"),
                click_handle=False,
                estimated_cost_usd=cage.estimated_cost_usd,
                scale_feasibility="lab",
                raw_score=cage.composite_score,
                raw_score_type="composite",
                notes=f"charge={cage.total_charge:+d}, {len(cage.endohedral_groups)} endo groups",
            ))
        modalities.add("coordination_cage")

    # ── Porphyrin ──
    if r.porphyrin_designs:
        for porph in r.porphyrin_designs[:3]:
            if porph.estimated_log_Ka_metal > 0:
                uas_p = porph.estimated_log_Ka_metal
                mode = "metal"
            else:
                uas_p = porph.composite_score * PORPHYRIN_GUEST_SCALE
                mode = "guest"

            mod_key = f"porphyrin_{mode}"

            subs = ", ".join(s.name for s in porph.meso_substituents[:2])
            if not subs:
                subs = "unsubstituted"

            entries.append(UASEntry(
                modality=mod_key,
                material_name=f"{porph.macrocycle.name} ({subs})",
                uas=uas_p,
                uas_lower=uas_p - PORPHYRIN_RMSE,
                uas_upper=uas_p + PORPHYRIN_RMSE,
                confidence="high" if mode == "metal" else "medium",
                selectivity_worst=0.0,
                selective=False,
                deployment_readiness=_deployment_readiness(mod_key),
                click_handle=_click_handle_available(mod_key, porph),
                estimated_cost_usd=porph.estimated_cost_usd,
                scale_feasibility="pilot" if mode == "metal" else "lab",
                raw_score=porph.composite_score,
                raw_score_type="log_Ka" if mode == "metal" else "composite",
            ))
        modalities.add(f"porphyrin_{mode}")

    # ── Sort by UAS (descending) and assign ranks ──
    entries.sort(key=lambda e: e.uas, reverse=True)
    for i, e in enumerate(entries):
        e.rank = i + 1

    result.entries = entries
    result.n_modalities = len(modalities)
    result.n_candidates = len(entries)

    # Summary picks
    if entries:
        result.best_entry = entries[0]

        selective = [e for e in entries if e.selective]
        if selective:
            result.best_selective = selective[0]

        deployable = [e for e in entries if e.deployment_readiness > 0.5]
        if deployable:
            result.best_deployable = max(deployable, key=lambda e: e.uas)

        costed = [e for e in entries if e.estimated_cost_usd > 0]
        if costed:
            result.best_value = max(costed, key=lambda e: e.uas / e.estimated_cost_usd)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

def print_cross_modal(result, top_n=20):
    """Print cross-modal comparison table."""
    print(f"\n{'═' * 100}")
    print(f"CROSS-MODAL COMPARISON: {result.guest_name}")
    print(f"{'═' * 100}")
    print(f"  {result.n_candidates} candidates across {result.n_modalities} modalities\n")

    # Header
    print(f"{'Rank':>4s}  {'UAS':>5s}  {'±CI':>5s}  {'Modality':>22s}  "
          f"{'Material':>35s}  {'Sel':>5s}  {'Deploy':>6s}  {'Click':>5s}  "
          f"{'Cost':>7s}  {'Scale':>10s}")
    print(f"{'':>4s}  {'logKa':>5s}  {'':>5s}  {'':>22s}  "
          f"{'':>35s}  {'ratio':>5s}  {'0-1':>6s}  {'':>5s}  "
          f"{'USD':>7s}  {'':>10s}")
    print("─" * 100)

    for e in result.entries[:top_n]:
        sel_str = f"{e.selectivity_worst:.1f}" if e.selectivity_worst > 0 else "—"
        click_str = "✓" if e.click_handle else "—"
        cost_str = f"${e.estimated_cost_usd:.0f}" if e.estimated_cost_usd > 0 else "—"

        print(f"{e.rank:>4d}  {e.uas:>5.1f}  ±{(e.uas_upper - e.uas):>3.1f}  "
              f"{e.modality:>22s}  {e.material_name:>35s}  "
              f"{sel_str:>5s}  {e.deployment_readiness:>5.2f}  {click_str:>5s}  "
              f"{cost_str:>7s}  {e.scale_feasibility:>10s}")

    # Summary
    print(f"\n{'─' * 100}")
    if result.best_entry:
        print(f"  Best overall:    #{result.best_entry.rank} {result.best_entry.material_name} "
              f"(UAS {result.best_entry.uas:.1f})")
    if result.best_selective:
        print(f"  Best selective:  #{result.best_selective.rank} {result.best_selective.material_name} "
              f"(UAS {result.best_selective.uas:.1f}, worst sel {result.best_selective.selectivity_worst:.1f}×)")
    if result.best_deployable:
        print(f"  Best deployable: #{result.best_deployable.rank} {result.best_deployable.material_name} "
              f"(UAS {result.best_deployable.uas:.1f}, readiness {result.best_deployable.deployment_readiness:.0%})")
    if result.best_value:
        print(f"  Best value:      #{result.best_value.rank} {result.best_value.material_name} "
              f"(UAS {result.best_value.uas:.1f}, ${result.best_value.estimated_cost_usd:.0f})")
