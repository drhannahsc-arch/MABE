"""
MABE Data Adapter: RefractiveIndex.INFO Optical Constants
==========================================================
Loads n(λ) and k(λ) from the refractiveindex.info YAML database.

Source: github.com/polyanskiy/refractiveindex.info-database
License: CC0 1.0 (public domain)
Paper: Polyanskiy, Scientific Data 11, 94 (2024). DOI: 10.1038/s41597-023-02898-2

Usage:
    db = RefractiveIndexDB("/path/to/refractiveindex.info-database/database")
    n, k = db.get_nk("Ag", "Johnson", wavelength_um=0.55)
    # Or get full spectrum:
    wl, n_arr, k_arr = db.get_spectrum("SiO2", "Malitson")
"""

import os
import re
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpticalData:
    """Optical constants for a single material/source."""
    material: str
    source: str
    wavelength_um: np.ndarray   # wavelength in micrometers
    n: np.ndarray               # real part of refractive index
    k: Optional[np.ndarray]     # imaginary part (extinction coefficient), None if transparent


def parse_nk_yaml(filepath: str) -> Optional[OpticalData]:
    """Parse a refractiveindex.info YAML file containing n,k data."""
    try:
        with open(filepath, "r") as f:
            doc = yaml.safe_load(f)
    except Exception:
        return None

    if not doc or "DATA" not in doc:
        return None

    material = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    source = os.path.splitext(os.path.basename(filepath))[0]

    for data_block in doc["DATA"]:
        dtype = data_block.get("type", "")

        if dtype == "tabulated nk":
            lines = data_block["data"].strip().split("\n")
            wl, n_vals, k_vals = [], [], []
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    wl.append(float(parts[0]))
                    n_vals.append(float(parts[1]))
                    k_vals.append(float(parts[2]))
            if wl:
                return OpticalData(
                    material=material, source=source,
                    wavelength_um=np.array(wl),
                    n=np.array(n_vals),
                    k=np.array(k_vals),
                )

        elif dtype == "tabulated n":
            lines = data_block["data"].strip().split("\n")
            wl, n_vals = [], []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    wl.append(float(parts[0]))
                    n_vals.append(float(parts[1]))
            if wl:
                return OpticalData(
                    material=material, source=source,
                    wavelength_um=np.array(wl),
                    n=np.array(n_vals),
                    k=None,
                )

        elif dtype == "tabulated k":
            # k-only data — store it, n will be None-like
            lines = data_block["data"].strip().split("\n")
            wl, k_vals = [], []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    wl.append(float(parts[0]))
                    k_vals.append(float(parts[1]))
            if wl:
                return OpticalData(
                    material=material, source=source,
                    wavelength_um=np.array(wl),
                    n=np.ones(len(wl)),  # placeholder
                    k=np.array(k_vals),
                )

        elif "formula" in dtype:
            # Sellmeier or Cauchy formula — need to evaluate
            # For now, skip formula-based entries (add later)
            pass

    return None


class RefractiveIndexDB:
    """Interface to the refractiveindex.info database."""

    def __init__(self, catalog_path: str = None, db_root: str = None):
        """
        Initialize with path to catalog-nk.yml and database root.

        Args:
            catalog_path: Path to catalog-nk.yml
            db_root: Root directory of the database (parent of 'main/', 'glass/', etc.)
        """
        self.catalog_path = catalog_path
        self.db_root = db_root
        self._catalog = None
        self._cache = {}

    def _load_catalog(self):
        """Parse the YAML catalog to build material→file index."""
        if self._catalog is not None:
            return
        if not self.catalog_path or not os.path.exists(self.catalog_path):
            self._catalog = {}
            return

        with open(self.catalog_path, "r") as f:
            raw = yaml.safe_load(f)

        self._catalog = {}
        current_shelf = None
        current_book = None
        for entry in raw:
            if "SHELF" in entry:
                current_shelf = entry["SHELF"]
            elif "BOOK" in entry:
                current_book = entry["BOOK"]
            elif "PAGE" in entry:
                page_name = entry["PAGE"]
                data_path = entry.get("data", "")
                if current_book and data_path:
                    key = f"{current_book}/{page_name}"
                    self._catalog[key] = {
                        "shelf": current_shelf,
                        "book": current_book,
                        "page": page_name,
                        "data_path": data_path,
                        "name": entry.get("name", ""),
                    }

    def list_materials(self) -> list[str]:
        """List all available material names (books)."""
        self._load_catalog()
        return sorted(set(v["book"] for v in self._catalog.values()))

    def list_sources(self, material: str) -> list[dict]:
        """List available data sources for a material."""
        self._load_catalog()
        return [
            {"source": v["page"], "name": v["name"], "path": v["data_path"]}
            for v in self._catalog.values()
            if v["book"] == material
        ]

    def load_data(self, material: str, source: str) -> Optional[OpticalData]:
        """Load optical data for a material/source combination."""
        cache_key = f"{material}/{source}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_catalog()
        entry = self._catalog.get(cache_key)
        if not entry or not self.db_root:
            return None

        filepath = os.path.join(self.db_root, entry["data_path"])
        if not os.path.exists(filepath):
            return None

        data = parse_nk_yaml(filepath)
        if data:
            self._cache[cache_key] = data
        return data

    def get_n_at_wavelength(self, material: str, source: str,
                            wavelength_um: float) -> Optional[float]:
        """Get refractive index at a specific wavelength via interpolation."""
        data = self.load_data(material, source)
        if data is None:
            return None
        return float(np.interp(wavelength_um, data.wavelength_um, data.n))

    def get_nk_at_wavelength(self, material: str, source: str,
                              wavelength_um: float) -> tuple[Optional[float], Optional[float]]:
        """Get n and k at a specific wavelength."""
        data = self.load_data(material, source)
        if data is None:
            return None, None
        n = float(np.interp(wavelength_um, data.wavelength_um, data.n))
        k = float(np.interp(wavelength_um, data.wavelength_um, data.k)) if data.k is not None else 0.0
        return n, k


# === Convenience functions for common MABE photonic materials ===

# Materials commonly used in structural color
STRUCTURAL_COLOR_MATERIALS = {
    "polystyrene":  {"material": "PS",    "source": "Sultanova", "n_approx": 1.59},
    "silica":       {"material": "SiO2",  "source": "Malitson",  "n_approx": 1.46},
    "titania":      {"material": "TiO2",  "source": "Devore-o",  "n_approx": 2.49},
    "melanin":      {"material": None,    "source": None,        "n_approx": 1.74},  # not in DB
    "water":        {"material": "H2O",   "source": "Hale",      "n_approx": 1.33},
    "air":          {"material": None,    "source": None,        "n_approx": 1.00},
    "gold":         {"material": "Au",    "source": "Johnson",   "n_approx": None},  # complex
    "silver":       {"material": "Ag",    "source": "Johnson",   "n_approx": None},
    "silicon":      {"material": "Si",    "source": "Aspnes",    "n_approx": 3.48},
    "ZnO":          {"material": "ZnO",   "source": "Bond-o",    "n_approx": 1.96},
}


if __name__ == "__main__":
    import sys
    catalog = sys.argv[1] if len(sys.argv) > 1 else "./data/refractive_index/catalog-nk.yml"

    print("=" * 60)
    print("MABE RefractiveIndex.INFO Adapter")
    print("=" * 60)

    db = RefractiveIndexDB(catalog_path=catalog)
    materials = db.list_materials()
    print(f"Catalog loaded: {len(materials)} materials")
    print(f"First 20: {materials[:20]}")

    # Show what's available for common structural color materials
    print("\n--- Structural Color Materials ---")
    for name, info in STRUCTURAL_COLOR_MATERIALS.items():
        if info["material"]:
            sources = db.list_sources(info["material"])
            print(f"  {name} ({info['material']}): {len(sources)} sources")
            for s in sources[:3]:
                print(f"    {s['source']}: {s['name']}")
        else:
            print(f"  {name}: n ≈ {info['n_approx']} (not in database, use constant)")