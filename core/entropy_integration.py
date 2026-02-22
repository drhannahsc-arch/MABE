"""
core/entropy_integration.py - Integrates entropy decomposition into pipeline.
"""
import core.sprint10_integration as s10
from core.entropy import decompose_thermodynamics
from core.thermodynamics import BindingThermodynamics


_orig_rescore = s10.full_physics_rescore


def _entropy_aware_rescore(assemblies, problem):
    """Add entropy decomposition to physics report."""
    assemblies = _orig_rescore(assemblies, problem)
    actual_temp = problem.matrix.temperature_c

    for assembly in assemblies:
        thermo = getattr(assembly, "thermodynamics", None)
        if thermo is None:
            thermo = BindingThermodynamics(
                dG_bind=getattr(assembly, "_dG_bind", 0.0),
                dG_desolv=getattr(assembly, "_dG_desolv", 0.0),
                dG_chelate=getattr(assembly, "_dG_chelate", 0.0),
                dG_preorg=getattr(assembly, "_dG_preorg", 0.0),
                dG_electrostatic=getattr(assembly, "_dG_electrostatic", 0.0),
                dG_net=getattr(assembly, "score_physics", 0.0),
                temperature_k=(actual_temp or 25.0) + 273.15,
            )

        donors = assembly.recognition.donor_atoms if assembly.recognition else []
        is_macro = ("macrocycl" in assembly.recognition.structure.lower()
                     if assembly.recognition and assembly.recognition.structure else False)

        entropy = decompose_thermodynamics(
            thermo, donors=donors, n_donors=len(donors),
            is_macrocyclic=is_macro, actual_temp_c=actual_temp,
        )

        assembly.confidence_reasoning += "\n\nENTROPY DECOMPOSITION:\n" + entropy.summary()

        if actual_temp is not None and entropy.dG_at_target is not None:
            dG_diff = entropy.dG_at_target - entropy.dG_ref
            if dG_diff > 5.0:
                assembly.failure_modes.append(
                    f"Temperature penalty: ΔG worsens by {dG_diff:+.1f} kJ/mol "
                    f"at {actual_temp:.0f}°C vs 25°C"
                )

        if entropy.temperature_sensitivity.startswith("HIGH"):
            assembly.failure_modes.append(
                f"High temperature sensitivity (ΔS = {entropy.dS_total:+.0f} J/(mol·K))"
            )

    return assemblies


s10.full_physics_rescore = _entropy_aware_rescore
