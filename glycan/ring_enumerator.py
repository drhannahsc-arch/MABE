"""
glycan/ring_enumerator.py -- Re-exports from core.ring_enumerator.

All ring enumeration logic now lives at core level for use by all modalities.
This module preserves backward compatibility for glycan-specific imports.
"""

# Re-export everything from core
from core.ring_enumerator import (  # noqa: F401
    RingSystem,
    DecoratedScaffold,
    PhysicsFilter,
    Decorator,
    DECORATOR_LIBRARY,
    get_catalog,
    get_ring_system,
    list_ring_systems,
    enumerate_decorated,
    enumerate_all_decorated,
    enumerate_physics_filtered,
    get_decorators,
    scaffold_to_backbone,
    decorator_to_arm,
    grammar_backbones,
    grammar_arms,
)
