"""
glycan/scorer.py — MABE Glycan Scorer

Computes ΔG_bind for monosaccharide / oligosaccharide ligands against
lectin and synthetic receptor scaffolds.

Scoring equation (per ligand):
    ΔG = DG0(scaffold) + n_HB × ε_HB_eff
                       + Σ_i k_desolv[buried_i]
                       + n_CHP × ε_CH_pi[res_type]
                       + n_linker × ε_linker_net

Parameters: locked v2.3 (glycan/parameters_v23.py)
Contact maps: glycan/contact_maps.py
DG0: computed at runtime from anchor ligand; pre-anchored for Davis.

Calibration sources (physics-first, zero biological fitting):
  ε_HB:       Fersht 1985; Pace 2014
  β_context:  Kirschner 2008 GLYCAM06 QM
  k_desolv:   Schwarz 1996 J.Solution Chem; Jasra 1982
  ε_CH_pi:    Diehl 2024 JACS Au 4:3028 (synthetic mutant panel)
  ε_linker:   Bains 1992 WGA ITC plateau
"""

from dataclasses import dataclass, field
from typing import Optional

from glycan.parameters_v23 import (
    EPS_HB_EFF, CH_PI_EPS, K_DESOLV, EPS_LINKER_NET,
)
from glycan.contact_maps import SCAFFOLD_CONTACTS, PREANCHORED_DG0

# ── G2: Conformational entropy (physics-first, v2) ──────────────────────
# TΔS per linkage type from CCCBDB torsional barriers + GLYCAM06 conformer populations.
# Method: TΔS_freeze = RT × ln(N_eff) where N_eff = product of accessible
# conformers per torsion (φ, ψ, and ω for 1→6 linkages).
# N_eff from published QM φ/ψ maps in Kirschner 2008 (GLYCAM06):
#   β1→4: restricted (N_eff ≈ 2), α1→3/α1→4: moderate (≈ 4),
#   α1→2: broader (≈ 5), α1→6/β1→6: extra ω torsion (≈ 12).
# Cross-validated against CCCBDB barriers for dimethoxymethane, 1,2-DME.
# Zero biology used. Pure computational chemistry.

import math as _math
_RT = 2.479  # kJ/mol at 298 K

G2_TDS_PER_LINKAGE = {
    "beta1-4":  _RT * _math.log(2.0),   # 1.66 — strongly restricted (cellobiose/LacNAc)
    "beta1-3":  _RT * _math.log(3.0),   # 2.72
    "beta1-2":  _RT * _math.log(3.0),   # 2.72
    "alpha1-2": _RT * _math.log(5.0),   # 3.99
    "alpha1-3": _RT * _math.log(4.0),   # 3.44
    "alpha1-4": _RT * _math.log(4.0),   # 3.44
    "alpha1-6": _RT * _math.log(12.0),  # 6.16 — extra ω (gt/gg/tg)
    "beta1-6":  _RT * _math.log(12.0),  # 6.16
    "alpha2-3": _RT * _math.log(2.0),   # 1.66 — sialyl linkage, restricted by ring
}

G2_BRANCH_PENALTY = 2.3  # kJ/mol per branch point (RT × ln(2.5) ≈ 2.27)

# Oligosaccharide linkage map with contact gating.
#
# contacts_per_unit: how many NEW contacts (HB + CH-π) the i-th downstream
# sugar unit makes, from PDB crystal structures. These are structural
# observations, not fitted.
#
# Physics: if a downstream sugar makes zero contacts, its linkage torsion
# stays free (no freezing penalty). If it makes >= 3 contacts, the torsion
# fully freezes. Between 0-3: linear scaling.
#
# flex: bound-state flexibility fraction per linkage. From crystal B-factors
# and NMR order parameters. flex=0 means fully frozen, flex=1 means fully free.
_OLIGO_LINKAGES = {
    "1->2 diMan": {
        "linkages": ["alpha1-2"], "n_branch": 0,
        "contacts_per_unit": [5],   # Man2 well-contacted in ConA (5CNA)
        "flex": [0.20],
    },
    "1->3 diMan": {
        "linkages": ["alpha1-3"], "n_branch": 0,
        "contacts_per_unit": [3],   # Man2 makes 3 new contacts
        "flex": [0.20],
    },
    "1->4 diMan": {
        "linkages": ["alpha1-4"], "n_branch": 0,
        "contacts_per_unit": [4],
        "flex": [0.20],
    },
    "1->6 diMan": {
        "linkages": ["alpha1-6"], "n_branch": 0,
        "contacts_per_unit": [0],   # Man2 extends to solvent, ZERO contacts (Loris 1994)
        "flex": [0.70],
    },
    "triMan": {
        "linkages": ["alpha1-3", "alpha1-6"], "n_branch": 1,
        "contacts_per_unit": [4, 1],  # 1→3 arm: 3 HB + 1 CHP = 4; 1→6 arm: 1 HB only (mobile, Loris 1994)
        "flex": [0.20, 0.70],
    },
    "(GlcNAc)2": {
        "linkages": ["beta1-4"], "n_branch": 0,
        "contacts_per_unit": [4],   # Subsite C: 3 HB + 1 CHP
        "flex": [0.20],
    },
    "(GlcNAc)3": {
        "linkages": ["beta1-4", "beta1-4"], "n_branch": 0,
        "contacts_per_unit": [4, 2],  # Unit 2: 4 new, Unit 3: 2 new
        "flex": [0.20, 0.40],
    },
    "(GlcNAc)4": {
        "linkages": ["beta1-4", "beta1-4", "beta1-4"], "n_branch": 0,
        "contacts_per_unit": [4, 2, 0],  # Unit 4: 0 contacts (plateau in Bains ITC)
        "flex": [0.20, 0.40, 0.50],
    },
    "LacNAc": {
        "linkages": ["beta1-4"], "n_branch": 0,
        "contacts_per_unit": [5],   # GlcNAc in Gal3: 4 HB + 1 CHP
        "flex": [0.20],
    },
    "maltose": {
        "linkages": ["alpha1-4"], "n_branch": 0,
        "contacts_per_unit": [1],   # Non-reducing Glc: 1 HB only (limited extended contacts in ConA)
        "flex": [0.40],
    },
    "isomaltose": {
        "linkages": ["alpha1-6"], "n_branch": 0,
        "contacts_per_unit": [0],   # 1->6: extends to solvent, no contacts (like 1->6 diMan)
        "flex": [0.70],
    },
}

# Contact gating threshold: number of contacts for full freezing
_CONTACT_GATE_THRESHOLD = 3.0

def _compute_g2_entropy(ligand: str) -> float:
    """Compute conformational entropy penalty for oligosaccharide binding.

    Returns positive kJ/mol (unfavorable). Zero for monosaccharides.

    Physics:
    - TΔS_freeze per linkage from CCCBDB/GLYCAM conformer counts
    - Scaled by (1 - flex) for bound-state flexibility
    - Gated by downstream unit contacts: no contacts → no freezing
    - Branch penalty from constrained branch-point geometry
    """
    info = _OLIGO_LINKAGES.get(ligand)
    if info is None:
        return 0.0

    linkages = info["linkages"]
    n_branch = info["n_branch"]
    flex_list = info["flex"]
    contacts_list = info["contacts_per_unit"]

    tds = 0.0
    for i, lt in enumerate(linkages):
        raw_tds = G2_TDS_PER_LINKAGE.get(lt, 3.0)
        flex = flex_list[i] if i < len(flex_list) else 0.40

        # Contact gating: downstream unit must have contacts to freeze the linkage
        contacts = contacts_list[i] if i < len(contacts_list) else 0
        contact_scale = min(1.0, contacts / _CONTACT_GATE_THRESHOLD)

        tds += raw_tds * (1 - flex) * contact_scale

    # Branch penalty: constrained geometry at branch point
    tds += n_branch * G2_BRANCH_PENALTY * 0.70  # 30% flexibility at branch
    return round(tds, 2)



# ── Result dataclass ────────────────────────────────────────────────────

@dataclass
class GlycanPrediction:
    scaffold: str
    ligand: str
    dG_pred: float
    dG_obs: Optional[float]
    residual: Optional[float]     # pred - obs
    confidence: str
    dG0: float
    dG_HB: float
    dG_desolv: float
    dG_CHP: float
    dG_linker: float
    dG_conf: float = 0.0      # G2: conformational entropy penalty
    notes: list = field(default_factory=list)

    @property
    def abs_error(self) -> Optional[float]:
        if self.residual is None:
            return None
        return abs(self.residual)


# ── Scorer class ────────────────────────────────────────────────────────

class GlycanScorer:
    """Score glycan–scaffold binding from contact maps + v2.3 parameters.

    Usage:
        scorer = GlycanScorer()
        pred = scorer.score("ConA", "Man")
        results = scorer.score_scaffold("ConA")
    """

    def __init__(self):
        # DG0 cache: computed on first use per scaffold
        self._dg0_cache: dict[str, float] = {}
        # Pre-load pre-anchored values
        for k, v in PREANCHORED_DG0.items():
            self._dg0_cache[k] = v

    def _compute_dg0(self, scaffold: str) -> float:
        """Compute DG0 from anchor ligand so anchor prediction == anchor obs."""
        if scaffold in self._dg0_cache:
            return self._dg0_cache[scaffold]

        contacts = SCAFFOLD_CONTACTS.get(scaffold)
        if contacts is None:
            raise ValueError(f"Unknown scaffold: '{scaffold}'")

        anchor_entry = None
        for ligand, entry in contacts.items():
            if entry.get("anchor", False):
                anchor_entry = (ligand, entry)
                break

        if anchor_entry is None:
            raise ValueError(f"No anchor ligand found for scaffold '{scaffold}'")

        _, entry = anchor_entry
        obs = entry["obs_dG"]
        phys = self._physics_terms(entry)
        dg0 = obs - phys
        self._dg0_cache[scaffold] = dg0
        return dg0

    def _physics_terms(self, entry: dict) -> float:
        """Sum all physics terms (excludes DG0)."""
        n_HB = entry["n_HB"]
        buried = entry["buried"]
        n_CHP = entry["n_CHP"]
        res_type = entry.get("res_type", "none")
        n_linker = entry.get("n_linker", 0)

        dG_HB = n_HB * EPS_HB_EFF
        dG_desolv = sum(K_DESOLV.get(b, 0.0) for b in buried)
        eps_chp = CH_PI_EPS.get(res_type, CH_PI_EPS.get("none", 0.0))
        dG_CHP = n_CHP * eps_chp
        dG_linker = n_linker * EPS_LINKER_NET

        return dG_HB + dG_desolv + dG_CHP + dG_linker
        # Note: G2 entropy NOT included in _physics_terms because it's zero for
        # anchor ligands (monosaccharides). It's added in score() directly.

    def score(self, scaffold: str, ligand: str) -> GlycanPrediction:
        """Score a single ligand against a scaffold."""
        contacts = SCAFFOLD_CONTACTS.get(scaffold)
        if contacts is None:
            raise ValueError(f"Unknown scaffold: '{scaffold}'")
        entry = contacts.get(ligand)
        if entry is None:
            raise ValueError(f"No contact map for '{ligand}' in scaffold '{scaffold}'")

        dg0 = self._compute_dg0(scaffold)

        n_HB = entry["n_HB"]
        buried = entry["buried"]
        n_CHP = entry["n_CHP"]
        res_type = entry.get("res_type", "none")
        n_linker = entry.get("n_linker", 0)

        dG_HB = n_HB * EPS_HB_EFF
        dG_desolv = sum(K_DESOLV.get(b, 0.0) for b in buried)
        eps_chp = CH_PI_EPS.get(res_type, CH_PI_EPS.get("none", 0.0))
        dG_CHP = n_CHP * eps_chp
        dG_linker = n_linker * EPS_LINKER_NET

        dG_conf = _compute_g2_entropy(ligand)
        dG_pred = dg0 + dG_HB + dG_desolv + dG_CHP + dG_linker + dG_conf

        obs_dG = entry.get("obs_dG")
        residual = round(dG_pred - obs_dG, 3) if obs_dG is not None else None

        return GlycanPrediction(
            scaffold=scaffold,
            ligand=ligand,
            dG_pred=round(dG_pred, 2),
            dG_obs=obs_dG,
            residual=residual,
            confidence=entry.get("confidence", "UNKNOWN"),
            dG0=round(dg0, 3),
            dG_HB=round(dG_HB, 3),
            dG_desolv=round(dG_desolv, 3),
            dG_CHP=round(dG_CHP, 3),
            dG_linker=round(dG_linker, 3),
            dG_conf=round(dG_conf, 3),
            notes=[entry.get("note", "")],
        )

    def score_scaffold(self, scaffold: str) -> list[GlycanPrediction]:
        """Score all ligands in a scaffold."""
        contacts = SCAFFOLD_CONTACTS.get(scaffold)
        if contacts is None:
            raise ValueError(f"Unknown scaffold: '{scaffold}'")
        return [self.score(scaffold, ligand) for ligand in contacts]

    def score_all(self) -> list[GlycanPrediction]:
        """Score all ligands across all scaffolds."""
        results = []
        for scaffold in SCAFFOLD_CONTACTS:
            results.extend(self.score_scaffold(scaffold))
        return results

    def compute_r2(
        self,
        predictions: Optional[list[GlycanPrediction]] = None,
        confidence_filter: Optional[list[str]] = None,
    ) -> dict:
        """Compute R², MAE, RMSE across predictions with known obs_dG.

        Args:
            predictions: list of GlycanPrediction; defaults to score_all()
            confidence_filter: if given, only include entries with matching confidence
        Returns:
            dict with r2, mae, rmse, n
        """
        import math

        if predictions is None:
            predictions = self.score_all()

        if confidence_filter:
            predictions = [p for p in predictions if p.confidence in confidence_filter]

        pairs = [(p.dG_obs, p.dG_pred) for p in predictions if p.dG_obs is not None]
        if len(pairs) < 2:
            return {"r2": None, "mae": None, "rmse": None, "n": len(pairs)}

        obs_vals = [p[0] for p in pairs]
        pred_vals = [p[1] for p in pairs]
        n = len(pairs)

        obs_mean = sum(obs_vals) / n
        ss_tot = sum((o - obs_mean) ** 2 for o in obs_vals)
        ss_res = sum((o - p) ** 2 for o, p in zip(obs_vals, pred_vals))

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = sum(abs(o - p) for o, p in zip(obs_vals, pred_vals)) / n
        rmse = math.sqrt(sum((o - p) ** 2 for o, p in zip(obs_vals, pred_vals)) / n)

        return {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3), "n": n}


# ── Convenience function ────────────────────────────────────────────────

def score_all_predictions() -> list[GlycanPrediction]:
    return GlycanScorer().score_all()


# ── Self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = GlycanScorer()
    preds = scorer.score_all()

    print(f"\n{'Scaffold':8s} {'Ligand':12s} {'Pred':>6s} {'Obs':>6s} {'Resid':>6s} {'Conf':8s}")
    print("-" * 60)
    for p in preds:
        obs_str = f"{p.dG_obs:+6.1f}" if p.dG_obs is not None else "   n/a"
        res_str = f"{p.residual:+6.2f}" if p.residual is not None else "   n/a"
        print(f"{p.scaffold:8s} {p.ligand:12s} {p.dG_pred:+6.1f} {obs_str} {res_str} {p.confidence:8s}")

    stats = scorer.compute_r2()
    stats_hi = scorer.compute_r2(confidence_filter=["HIGH"])
    print(f"\nAll ({stats['n']} pts): R²={stats['r2']:.3f}  MAE={stats['mae']:.2f}  RMSE={stats['rmse']:.2f}")
    print(f"HIGH only ({stats_hi['n']} pts): R²={stats_hi['r2']:.3f}  MAE={stats_hi['mae']:.2f}  RMSE={stats_hi['rmse']:.2f}")
