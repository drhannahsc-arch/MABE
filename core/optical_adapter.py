"""
Optical Adapter — Field Paradigm.

Takes a FieldInteractionSpec and designs an optical material solution.
Uses Beer-Lambert law, wavelength-color physics, and known chromophore
properties to predict optical behavior and recommend materials.

SELF-VALIDATING: Predicted colors can be verified by visual inspection.
This is the fastest validation loop in MABE — no wet lab needed for
initial confirmation of the predicted wavelength regime.

Physics connection:
    - Beer-Lambert: A = ε·c·l → concentration from desired absorbance
    - Complementary color: absorbed λ → perceived color
    - Chromophore selection: match target λ from known spectral data
    - Spectral shift prediction: metalation shifts porphyrin Soret band

Data sources:
    - Wavelength-color mapping (physics, Tier 1)
    - Common chromophore properties (textbook, Tier 2)
    - refractiveindex.info (primary literature, Tier 1-equivalent)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from mabe.realization.models import (
    ApplicationContext,
    FabricationSpec,
    FieldInteractionSpec,
    FieldType,
    WavelengthTarget,
    ScaleClass,
)
from mabe.realization.adapters.optical_knowledge import (
    DATA_SOURCES,
    COMMON_CHROMOPHORES,
    CHROMOPHORE_BY_NAME,
    wavelength_to_absorbed_color,
    wavelength_to_perceived_color,
    is_visible,
    beer_lambert_absorbance,
    required_concentration_M,
    find_chromophores_for_wavelength,
    find_chromophores_for_color,
)


# ─────────────────────────────────────────────
# Fabrication spec for optical design
# ─────────────────────────────────────────────

@dataclass
class OpticalFabricationSpec(FabricationSpec):
    """Layer 4 output for an optical material design."""

    # ── Chromophore selection ──
    recommended_chromophore: str = ""
    chromophore_class: str = ""
    lambda_max_nm: float = 0.0
    epsilon_L_mol_cm: float = 0.0
    transition_type: str = ""

    # ── Predicted optical properties ──
    predicted_absorbed_color: str = ""
    predicted_perceived_color: str = ""
    predicted_absorbance_at_1mM: float = 0.0

    # ── Design parameters ──
    required_concentration_M: float = 0.0   # for target absorbance
    recommended_pathlength_cm: float = 1.0
    target_absorbance: float = 1.0

    # ── Self-validation ──
    visual_validation: str = ""     # "Look for [color] solution"
    requires_instrument: bool = False

    # ── Data provenance ──
    data_sources: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────

class OpticalAdapter:
    """
    Designs an optical material/chromophore solution for a FieldInteractionSpec.

    Scores based on:
        - Wavelength match to known chromophores
        - Molar absorptivity (sensitivity)
        - Visibility (is the color in the visible range?)
        - Beer-Lambert feasibility (achievable concentration)
        - Self-validation capability
    """

    system_id = "optical_chromophore"
    supported_spec_types = ["field"]

    def score(self, spec: FieldInteractionSpec) -> dict:
        """Score feasibility and performance of optical approach."""

        if spec.field_type not in (
            FieldType.OPTICAL_ABSORPTION,
            FieldType.OPTICAL_EMISSION,
        ):
            return self._infeasible(
                f"Optical adapter handles absorption/emission, not {spec.field_type.value}"
            )

        # ── Find matching chromophores ──
        candidates = []
        for wt in spec.target_wavelengths:
            matches = find_chromophores_for_wavelength(
                wt.center_nm, wt.bandwidth_nm / 2
            )
            candidates.extend(matches)

        # Also try color-based matching
        if spec.target_color and not candidates:
            candidates = find_chromophores_for_color(spec.target_color)

        if not candidates:
            # No known chromophore — still feasible but lower confidence
            return {
                "feasible": True,
                "wavelength_match_score": 0.0,
                "sensitivity_score": 0.0,
                "visibility_score": 1.0 if any(
                    is_visible(wt.center_nm) for wt in spec.target_wavelengths
                ) else 0.3,
                "composite_score": 0.2,
                "confidence": 0.3,
                "calibration_status": "uncalibrated",
                "notes": ["No known chromophore matches target wavelength. Custom synthesis required."],
                "candidates": [],
            }

        # ── Score best candidate ──
        best = max(candidates, key=lambda c: c.epsilon_L_mol_cm)

        # Wavelength match
        if spec.target_wavelengths:
            target_nm = spec.target_wavelengths[0].center_nm
            delta = abs(best.lambda_max_nm - target_nm)
            wl_score = max(0.0, 1.0 - delta / 50.0)  # linear penalty, 50nm = zero
        else:
            wl_score = 0.5

        # Sensitivity (molar absorptivity)
        # ε > 10^4 is good, ε > 10^5 is excellent
        if best.epsilon_L_mol_cm >= 1e5:
            sens_score = 1.0
        elif best.epsilon_L_mol_cm >= 1e4:
            sens_score = 0.7
        elif best.epsilon_L_mol_cm >= 1e3:
            sens_score = 0.4
        else:
            sens_score = 0.2

        # Visibility
        vis_score = 1.0 if is_visible(best.lambda_max_nm) else 0.3

        # Self-validation bonus: visible range → immediate visual check
        validation_bonus = 0.1 if is_visible(best.lambda_max_nm) else 0.0

        composite = (
            0.35 * wl_score
            + 0.30 * sens_score
            + 0.20 * vis_score
            + 0.15 * 0.8  # general feasibility of optical approach
            + validation_bonus
        )
        composite = min(1.0, composite)

        # Confidence: optical physics is well-understood
        confidence = 0.80
        if wl_score > 0.8:
            confidence = 0.90  # known chromophore, good match
        if best.ph_sensitive and spec.field_type == FieldType.OPTICAL_ABSORPTION:
            confidence -= 0.05  # pH sensitivity adds uncertainty

        return {
            "feasible": True,
            "best_chromophore": best.name,
            "lambda_max_nm": best.lambda_max_nm,
            "epsilon_L_mol_cm": best.epsilon_L_mol_cm,
            "predicted_color": wavelength_to_perceived_color(best.lambda_max_nm),
            "wavelength_match_score": round(wl_score, 3),
            "sensitivity_score": round(sens_score, 3),
            "visibility_score": round(vis_score, 3),
            "composite_score": round(composite, 3),
            "confidence": round(confidence, 2),
            "calibration_status": "physics-derived",
            "candidates": [c.name for c in candidates],
        }

    def design(self, spec: FieldInteractionSpec) -> OpticalFabricationSpec:
        """Produce a fabrication spec for the optical design."""
        score_result = self.score(spec)

        chrom_name = score_result.get("best_chromophore", "")
        chrom = CHROMOPHORE_BY_NAME.get(chrom_name)

        if chrom:
            lambda_max = chrom.lambda_max_nm
            epsilon = chrom.epsilon_L_mol_cm
            perceived = wavelength_to_perceived_color(lambda_max)
            absorbed = wavelength_to_absorbed_color(lambda_max)
            transition = chrom.transition_type
            chrom_class = chrom.chromophore_class

            # Beer-Lambert: concentration for A=1.0 at l=1cm
            target_A = 1.0
            conc = required_concentration_M(target_A, epsilon, 1.0)
            abs_at_1mM = beer_lambert_absorbance(epsilon, 1e-3, 1.0)
        else:
            lambda_max = spec.primary_wavelength_nm
            epsilon = 0.0
            perceived = wavelength_to_perceived_color(lambda_max) if lambda_max > 0 else "unknown"
            absorbed = wavelength_to_absorbed_color(lambda_max) if lambda_max > 0 else "unknown"
            transition = "unknown"
            chrom_class = "unknown"
            conc = 0.0
            abs_at_1mM = 0.0
            target_A = 1.0

        visual_val = (
            f"Solution should appear {perceived}. "
            f"Absorbs {absorbed} light at ~{lambda_max:.0f} nm."
        ) if is_visible(lambda_max) else (
            f"Absorption at {lambda_max:.0f} nm is outside visible range. "
            f"UV-Vis spectrometer required for validation."
        )

        spec_hash = hashlib.md5(
            f"{chrom_name}:{lambda_max}:{spec.field_type.value}".encode()
        ).hexdigest()[:12]

        synthesis_steps = []
        if chrom:
            synthesis_steps = [
                f"Obtain {chrom.name} (commercial or synthesize)",
                f"Prepare stock solution in {chrom.solvent}",
                f"Dilute to target concentration ({conc:.2e} M for A=1.0)",
                f"Verify color visually: expect {perceived} solution",
            ]
            if chrom.ph_sensitive:
                synthesis_steps.insert(2, "Buffer solution to required pH")
        else:
            synthesis_steps = [
                f"Custom chromophore design targeting λ_max ≈ {lambda_max:.0f} nm",
                f"Synthesize candidate compounds",
                f"Measure UV-Vis spectrum to verify absorption wavelength",
            ]

        validation = [
            f"UV-Vis spectrum: verify λ_max within ±5 nm of {lambda_max:.0f} nm",
            f"Beer-Lambert linearity: measure A vs concentration (0.1-10 mM)",
            f"Molar absorptivity: verify ε ≥ {epsilon:.0e} L/(mol·cm)",
        ]
        if is_visible(lambda_max):
            validation.insert(0,
                f"Visual check: solution should appear {perceived}"
            )

        return OpticalFabricationSpec(
            material_system="optical_chromophore",
            geometry_spec_hash=spec_hash,
            predicted_pocket_geometry=None,
            predicted_deviation_from_ideal_A=0.0,
            recommended_chromophore=chrom_name,
            chromophore_class=chrom_class,
            lambda_max_nm=lambda_max,
            epsilon_L_mol_cm=epsilon,
            transition_type=transition,
            predicted_absorbed_color=absorbed,
            predicted_perceived_color=perceived,
            predicted_absorbance_at_1mM=abs_at_1mM,
            required_concentration_M=conc,
            recommended_pathlength_cm=1.0,
            target_absorbance=target_A,
            visual_validation=visual_val,
            requires_instrument=not is_visible(lambda_max),
            synthesis_steps=synthesis_steps,
            validation_experiments=validation,
            estimated_cost_per_unit=self._estimate_cost(chrom),
            estimated_time="1-2 hours (solution preparation + UV-Vis)",
            data_sources=DATA_SOURCES,
        )

    def _infeasible(self, reason: str) -> dict:
        return {
            "feasible": False,
            "reason": reason,
            "wavelength_match_score": 0.0,
            "sensitivity_score": 0.0,
            "visibility_score": 0.0,
            "composite_score": 0.0,
            "confidence": 0.0,
        }

    def _estimate_cost(self, chrom) -> float:
        if chrom is None:
            return 100.0
        if chrom.chromophore_class == "inorganic":
            return 5.0
        elif chrom.chromophore_class == "indicator":
            return 20.0
        elif chrom.chromophore_class == "porphyrin":
            return 50.0
        return 30.0