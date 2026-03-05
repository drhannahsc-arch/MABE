"""
MABE Data Adapters
==================
Programmatic access to curated experimental databases for MABE calibration and validation.

Connected sources:
  - SAMPL6/7/9 host-guest ITC data (104 complexes)
  - RefractiveIndex.INFO optical constants (4,164 datasets, 272 materials)
  - ChEMBL binding data API
  - NIST WebBook thermochemistry

Blocked (need domain allowlist update):
  - NIST SRD 46 SQLite (99,000 metal-ligand stability constants)
  - Materials Project API (150,000 inorganic materials)
  - BindingDB host-guest section
  - COD crystal structures
"""

from .sampl_adapter import (
    load_all_sampl,
    sampl_summary,
    to_mabe_format,
    HostGuestMeasurement,
)

from .refractive_index_adapter import (
    RefractiveIndexDB,
    parse_nk_yaml,
    OpticalData,
    STRUCTURAL_COLOR_MATERIALS,
)

from .nist_chembl_adapter import (
    fetch_nist_gas_phase_thermo,
    fetch_nist_ion_energetics,
    chembl_search_molecule,
    chembl_get_activities,
    chembl_get_molecule_info,
    MABE_SPECIES,
)