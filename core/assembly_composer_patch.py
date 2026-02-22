"""
core/assembly_composer_patch.py - Patches assembly_composer to use interior_designer_hook.

Monkey-patches the import so assembly_composer calls the hook version
of design_interior that checks self-binding structures first.
"""

import core.assembly_composer as composer
from core.interior_designer_hook import design_interior

# Replace the design_interior reference in the composer module
composer.design_interior = design_interior
