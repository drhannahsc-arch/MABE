"""
Composite Scoring.

Physics fidelity: 60% (FIXED, non-negotiable).
Implementation: 40% (distributed by application context).

No precedent score. Physics doesn't reward familiarity.
"""

from __future__ import annotations

from mabe.realization.models import ApplicationContext


# Physics fidelity weight is FIXED at 0.60.
PHYSICS_WEIGHT = 0.60
IMPLEMENTATION_WEIGHT = 0.40

# The remaining 0.40 distributed among implementation factors.
# These weights are relative (sum to 1.0 within implementation).
IMPLEMENTATION_PROFILES: dict[ApplicationContext, dict[str, float]] = {
    ApplicationContext.DIAGNOSTIC: {
        "synthetic_accessibility": 0.25,
        "cost": 0.10,
        "scalability": 0.10,
        "operating_conditions": 0.15,
        "reusability": 0.00,
    },
    ApplicationContext.REMEDIATION: {
        "synthetic_accessibility": 0.10,
        "cost": 0.35,
        "scalability": 0.35,
        "operating_conditions": 0.10,
        "reusability": 0.10,
    },
    ApplicationContext.THERAPEUTIC: {
        "synthetic_accessibility": 0.10,
        "cost": 0.10,
        "scalability": 0.10,
        "operating_conditions": 0.35,
        "reusability": 0.00,
    },
    ApplicationContext.RESEARCH: {
        "synthetic_accessibility": 0.35,
        "cost": 0.15,
        "scalability": 0.00,
        "operating_conditions": 0.10,
        "reusability": 0.00,
    },
    ApplicationContext.SEPARATION: {
        "synthetic_accessibility": 0.15,
        "cost": 0.25,
        "scalability": 0.25,
        "operating_conditions": 0.15,
        "reusability": 0.20,
    },
    ApplicationContext.CATALYSIS: {
        "synthetic_accessibility": 0.15,
        "cost": 0.10,
        "scalability": 0.15,
        "operating_conditions": 0.30,
        "reusability": 0.30,
    },
}


def compute_composite(
    physics_fidelity: float,
    synthetic_accessibility: float,
    cost_score: float,
    scalability: float,
    operating_condition_compatibility: float,
    reusability_score: float,
    application: ApplicationContext,
) -> float:
    """
    Weighted composite score.

    Physics always 60%. Implementation 40% distributed by application.
    """
    profile = IMPLEMENTATION_PROFILES.get(
        application,
        IMPLEMENTATION_PROFILES[ApplicationContext.RESEARCH],  # default
    )

    impl_score = (
        profile["synthetic_accessibility"] * synthetic_accessibility
        + profile["cost"] * cost_score
        + profile["scalability"] * scalability
        + profile["operating_conditions"] * operating_condition_compatibility
        + profile["reusability"] * reusability_score
    )

    # Normalize: profile weights should sum to ~1.0 but enforce
    profile_sum = sum(profile.values())
    if profile_sum > 0:
        impl_score /= profile_sum

    return PHYSICS_WEIGHT * physics_fidelity + IMPLEMENTATION_WEIGHT * impl_score
