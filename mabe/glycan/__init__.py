"""
MABE Glycan Recognition Module
===============================

Physics-based scoring of glycan-binder interactions.
Integrates with the MABE unified scorer — no routing wall.

Submodules:
  - params: G1–G8 parameter constants (non-biological sources)
  - scorer: GlycanScorer class consuming UniversalComplex
  - sugar_properties: SugarPropertyCard generation
  - contact_map: GlycanContactMap from crystal structures
  - descriptors: from_glycan_binding() constructor
"""

from mabe.glycan.params import GLYCAN_PARAMS
from mabe.glycan.scorer import compute_glycan_terms
from mabe.glycan.descriptors import from_glycan_binding
