"""
core/novel_material_api.py — Entry Points for Novel Material Binding Prediction
=================================================================================

New constructors for UniversalComplex:
  from_receptor_guest(host_smiles, guest_smiles, ...) → UC
  from_porous_binding(linker, node, topology, guest, ...) → UC

These complement the existing:
  from_smiles(smiles, metal, host) — metal chelation / known hosts
  from_host_guest(hg_entry) — known host database entries
  from_metalloprotein(entry) — protein-metal-ligand

All produce UniversalComplex → predict() → PredictionResult.

No new physics. No new parameters. Routes through existing scorer terms
via data-presence guards.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.universal_schema import UniversalComplex
from core.receptor_descriptor import characterize_receptor
from core.porous_descriptor import characterize_porous


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC RECEPTOR + GUEST
# ═══════════════════════════════════════════════════════════════════════════

def from_receptor_guest(
    host_smiles: str,
    guest_smiles: str = "",
    host_name: str = "",
    log_Ka_exp: float = 0.0,
    temperature_C: float = 25.0,
    pH: float = 7.0,
    source: str = "manual",
) -> UniversalComplex:
    """
    Build a UniversalComplex for a synthetic receptor binding a guest molecule.

    Works for: urea cages, molecular tweezers, macrocyclic receptors,
    calixarene variants, pillar[n]arenes, foldamers, clip molecules.

    Both host and guest described by SMILES. Auto-characterizes:
      - Cavity volume, aromatic walls, H-bond sites (from host SMILES)
      - Guest properties, packing coefficient, complementarity (from guest SMILES)

    Example:
        uc = from_receptor_guest(
            host_smiles="...",  # Davis cage SMILES
            guest_smiles="OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O",  # glucose
            host_name="GluHUT",
            log_Ka_exp=4.27,  # log(18600)
        )
        result = predict(uc)
    """
    # Auto-characterize receptor
    desc = characterize_receptor(host_smiles, guest_smiles, host_name)

    # Build UC
    uc = UniversalComplex(
        name=f"{desc.get('host_name', 'receptor')}:{guest_smiles[:20] if guest_smiles else 'empty'}",
        binding_mode=desc.get('binding_mode', 'synthetic_receptor'),
        log_Ka_exp=log_Ka_exp,
        temperature_C=temperature_C,
        ph=pH,
        source=source,
    )

    # Apply all descriptor fields
    for key, val in desc.items():
        if hasattr(uc, key) and val is not None:
            try:
                setattr(uc, key, val)
            except (AttributeError, TypeError):
                pass

    # Set guest_smiles explicitly (may be overwritten by desc)
    if guest_smiles:
        uc.guest_smiles = guest_smiles

    return uc


# ═══════════════════════════════════════════════════════════════════════════
# POROUS FRAMEWORK + GUEST
# ═══════════════════════════════════════════════════════════════════════════

def from_porous_binding(
    linker_smiles: str = "",
    linker_name: str = "",
    node_type: str = "",
    topology: str = "",
    pore_diameter_A: float = 0.0,
    surface_area_m2g: float = 0.0,
    guest_smiles: str = "",
    framework_name: str = "",
    log_Ka_exp: float = 0.0,
    temperature_C: float = 25.0,
    source: str = "manual",
) -> UniversalComplex:
    """
    Build a UniversalComplex for guest binding in a porous framework.

    Works for: MOFs, COFs, zeolites, porous organic cages.

    Decomposes binding into:
      - Open metal site coordination (→ metal scorer)
      - Pore confinement (→ HG shape/hydrophobic terms)
      - Linker-guest H-bonds and π contacts (→ HG hbond/pi terms)

    Example:
        uc = from_porous_binding(
            linker_name="NH2-BDC",
            node_type="Zr6",
            topology="fcu",
            pore_diameter_A=6.0,
            guest_smiles="O=C=O",  # CO2
            framework_name="UiO-66-NH2",
        )
        result = predict(uc)
    """
    desc = characterize_porous(
        linker_smiles=linker_smiles,
        node_type=node_type,
        topology=topology,
        pore_diameter_A=pore_diameter_A,
        surface_area_m2g=surface_area_m2g,
        guest_smiles=guest_smiles,
        linker_name=linker_name,
        framework_name=framework_name,
    )

    uc = UniversalComplex(
        name=desc.get('host_name', framework_name or 'porous_framework'),
        binding_mode=desc.get('binding_mode', 'porous_framework'),
        log_Ka_exp=log_Ka_exp,
        temperature_C=temperature_C,
        source=source,
    )

    # Apply all descriptor fields
    for key, val in desc.items():
        if hasattr(uc, key) and val is not None:
            try:
                setattr(uc, key, val)
            except (AttributeError, TypeError):
                pass

    # Set guest_smiles explicitly
    if guest_smiles:
        uc.guest_smiles = guest_smiles

    return uc


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: score_receptor_guest / score_porous
# ═══════════════════════════════════════════════════════════════════════════

def score_receptor_guest(host_smiles: str, guest_smiles: str, **kwargs):
    """One-shot: build UC + predict for synthetic receptor."""
    from core.unified_scorer_v2 import predict
    uc = from_receptor_guest(host_smiles, guest_smiles, **kwargs)
    return predict(uc, verbose=kwargs.get('verbose', False))


def score_porous(guest_smiles: str, **kwargs):
    """One-shot: build UC + predict for porous framework."""
    from core.unified_scorer_v2 import predict
    uc = from_porous_binding(guest_smiles=guest_smiles, **kwargs)
    return predict(uc, verbose=kwargs.get('verbose', False))
