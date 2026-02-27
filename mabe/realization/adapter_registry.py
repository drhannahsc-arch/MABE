"""
Adapter Registry — maps system_id to concrete adapter classes.

Lazy instantiation: adapters are only created when needed.
Auto-discovery: register_builtins() wires all implemented adapters.

Usage:
    registry = AdapterRegistry()
    registry.register_builtins()
    adapters = registry.instantiate_all()
    ranker = AdapterRanker(adapters)
    result = ranker.rank(spec)
"""

from __future__ import annotations

from typing import Optional, Type

from mabe.realization.adapters.base import RealizationAdapter


class AdapterRegistry:
    """Maps system_id → adapter class. Lazy instantiation."""

    def __init__(self):
        self._registry: dict[str, Type[RealizationAdapter]] = {}
        self._instances: dict[str, RealizationAdapter] = {}

    def register(self, system_id: str, adapter_cls: Type[RealizationAdapter]) -> None:
        """Register an adapter class for a system_id."""
        self._registry[system_id] = adapter_cls

    def get_class(self, system_id: str) -> Optional[Type[RealizationAdapter]]:
        return self._registry.get(system_id)

    def instantiate(self, system_id: str) -> Optional[RealizationAdapter]:
        """Get or create adapter instance."""
        if system_id in self._instances:
            return self._instances[system_id]
        cls = self._registry.get(system_id)
        if cls is None:
            return None
        instance = cls()
        self._instances[system_id] = instance
        return instance

    def instantiate_all(self) -> list[RealizationAdapter]:
        """Instantiate all registered adapters."""
        result = []
        for system_id in self._registry:
            adapter = self.instantiate(system_id)
            if adapter is not None:
                result.append(adapter)
        return result

    def registered_ids(self) -> list[str]:
        return list(self._registry.keys())

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, system_id: str) -> bool:
        return system_id in self._registry


def register_builtins(registry: AdapterRegistry) -> AdapterRegistry:
    """
    Register all implemented adapters.

    Each adapter is imported and registered lazily so missing
    dependencies don't prevent other adapters from loading.
    """
    # ── Class A: Precision Binders ──
    try:
        from mabe.realization.adapters.cyclodextrin_adapter import CyclodextrinAdapter
        registry.register("cyclodextrin", CyclodextrinAdapter)
    except ImportError:
        pass

    try:
        from mabe.realization.adapters.crown_ether_adapter import CrownEtherAdapter
        registry.register("crown_ether", CrownEtherAdapter)
    except ImportError:
        pass

    try:
        from mabe.realization.adapters.porphyrin_adapter import PorphyrinAdapter
        registry.register("porphyrin", PorphyrinAdapter)
    except ImportError:
        pass

    # ── Class D: Bulk Sorbents ──
    try:
        from mabe.realization.adapters.lignin_adapter import FunctionalizedLigninAdapter
        registry.register("functionalized_lignin", FunctionalizedLigninAdapter)
    except ImportError:
        pass

    return registry


# ─────────────────────────────────────────────
# Convenience: global registry
# ─────────────────────────────────────────────

ADAPTER_REGISTRY = AdapterRegistry()
register_builtins(ADAPTER_REGISTRY)
