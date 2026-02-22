"""MABE - Modality-Agnostic Binder Engine"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters.base import ToolRegistry
from adapters.rdkit_adapter import RDKitAdapter
from adapters.dnazyme_adapter import DNAzymeAdapter
from adapters.peptide_adapter import PeptideAdapter
from adapters.aptamer_adapter import AptamerAdapter
from conversation.decomposer_patch import patch_targets
from conversation.interface import run_interactive, run_single_query
patch_targets()
import core.assembly_composer_patch, core.scoring_patch
import core.physics_integration, core.sprint10_integration
import core.protonation_integration, core.lfse_integration
import core.ionic_integration, core.repulsion_integration
import core.entropy_integration

def build_registry():
    registry = ToolRegistry()
    rdkit = RDKitAdapter()
    if rdkit.is_available(): registry.register(rdkit)
    registry.register(DNAzymeAdapter())
    registry.register(PeptideAdapter())
    registry.register(AptamerAdapter())
    return registry

def main():
    registry = build_registry()
    if len(sys.argv) > 1: run_single_query(registry, " ".join(sys.argv[1:]))
    else: run_interactive(registry)

if __name__ == "__main__": main()
