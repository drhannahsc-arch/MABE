"""
novel_host_library.py — Preset novel host specifications for de novo guest generation.

Provides a curated library of MOFs, cages, zeolites, and synthetic receptors
with published cavity volumes, portal types, and framework charges. Each entry
returns a NovelHostSpec that can be passed directly to generate_for_host().

Data quality: All cavity volumes from peer-reviewed crystallographic or
computational studies. Sources cited inline with DOI.

Usage:
    from core.novel_host_library import get_host, list_hosts

    # Direct lookup
    spec = get_host("HKUST-1")
    result = generate_for_host(spec)

    # Browse by type
    mofs = list_hosts(host_type="MOF")
    cages = list_hosts(host_type="cage")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# NovelHostSpec — import from de_novo_generator to ensure isinstance() works.
# Falls back to standalone copy only when de_novo_generator is not importable
# (e.g., no rdkit). The standalone copy is structurally identical.
# ═══════════════════════════════════════════════════════════════════════════

try:
    from core.de_novo_generator import NovelHostSpec
except (ImportError, ModuleNotFoundError):
    @dataclass
    class NovelHostSpec:
        """Specification for a novel cavity host not in HOST_REGISTRY.

        Provides minimum properties for the unified scorer's novel host fallback.
        Standalone fallback — used only when de_novo_generator is not importable.
        """
        name: str = "novel_host"
        cavity_volume_A3: float = 0.0
        host_charge: int = 0
        n_hbonds_host: int = 0
        portal_type: str = "neutral"
        host_type: str = "novel_cavity"
        max_guest_volume_A3: float = 0.0

        def __post_init__(self):
            if self.max_guest_volume_A3 <= 0 and self.cavity_volume_A3 > 0:
                self.max_guest_volume_A3 = self.cavity_volume_A3 * 0.65


@dataclass
class HostEntry:
    """Extended host specification with metadata for the library."""
    spec: NovelHostSpec
    category: str             # "MOF", "cage", "zeolite", "synthetic_receptor", "COF"
    formula: str = ""         # e.g. "Cu3(BTC)2" for HKUST-1
    topology: str = ""        # e.g. "tbo", "fcu"
    pore_diameter_A: float = 0.0      # largest included sphere diameter
    surface_area_m2g: float = 0.0     # BET surface area
    source: str = ""          # citation with DOI
    notes: str = ""
    tags: List[str] = field(default_factory=list)  # e.g. ["CO2-capture", "water-stable"]


# ═══════════════════════════════════════════════════════════════════════════
# HOST LIBRARY
# ═══════════════════════════════════════════════════════════════════════════

_LIBRARY: Dict[str, HostEntry] = {}


def _add(entry: HostEntry):
    """Register a host entry in the library."""
    _LIBRARY[entry.spec.name] = entry
    # Also register common aliases
    return entry


# ─── MOFs ─────────────────────────────────────────────────────────────────

_add(HostEntry(
    spec=NovelHostSpec(
        name="HKUST-1",
        cavity_volume_A3=636.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="MOF",
    ),
    category="MOF",
    formula="Cu3(BTC)2",
    topology="tbo",
    pore_diameter_A=12.0,
    surface_area_m2g=1500.0,
    source="Chui et al. Science 1999, 283, 1148. DOI:10.1126/science.283.5405.1148. "
           "Cavity volume: Mason et al. Chem. Sci. 2014, 5, 32. DOI:10.1039/C3SC52633J",
    notes="Three pore types: large cage (12 Å), small tetrahedral (5 Å), "
          "small triangular (3.5 Å). 636 Å³ is the large cage.",
    tags=["CO2-capture", "open-metal-site", "water-sensitive"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="UiO-66",
        cavity_volume_A3=753.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="MOF",
    ),
    category="MOF",
    formula="Zr6O4(OH)4(BDC)6",
    topology="fcu",
    pore_diameter_A=11.0,
    surface_area_m2g=1200.0,
    source="Cavka et al. JACS 2008, 130, 13850. DOI:10.1021/ja8057953. "
           "Cavity volume: Yang et al. Chem. Rev. 2012, 112, 1162. DOI:10.1021/cr200190s",
    notes="Two cage types: octahedral (11 Å) and tetrahedral (8 Å). "
          "753 Å³ is the octahedral cage. Exceptionally water-stable.",
    tags=["water-stable", "thermal-stable", "Zr-cluster"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="UiO-66-NH2",
        cavity_volume_A3=905.0,
        host_charge=0,
        n_hbonds_host=1,
        portal_type="amine",
        host_type="MOF",
    ),
    category="MOF",
    formula="Zr6O4(OH)4(NH2-BDC)6",
    topology="fcu",
    pore_diameter_A=11.0,
    surface_area_m2g=1100.0,
    source="Kandiah et al. Chem. Mater. 2010, 22, 6632. DOI:10.1021/cm102601v. "
           "CO2 affinity: Cmarik et al. Langmuir 2012, 28, 15606. DOI:10.1021/la3035352",
    notes="Amine-functionalized UiO-66. Enhanced CO2 affinity via amine-CO2 H-bond. "
          "905 Å³ larger than UiO-66 due to amino group pushing pore walls.",
    tags=["CO2-capture", "water-stable", "functionalized", "H-bond-donor"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="ZIF-8",
        cavity_volume_A3=2465.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="MOF",
    ),
    category="MOF",
    formula="Zn(mIm)2",
    topology="sod",
    pore_diameter_A=11.6,
    surface_area_m2g=1630.0,
    source="Park et al. PNAS 2006, 103, 10186. DOI:10.1073/pnas.0602439103. "
           "Pore volume: Fairen-Jimenez et al. JACS 2011, 133, 8900. DOI:10.1021/ja202154j",
    notes="Sodalite topology. Large cage (11.6 Å) with narrow 6-ring windows (3.4 Å). "
          "Gate-opening effect allows larger molecules through flexible aperture.",
    tags=["gas-separation", "flexible-aperture", "hydrophobic"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="MOF-5",
        cavity_volume_A3=2587.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="MOF",
    ),
    category="MOF",
    formula="Zn4O(BDC)3",
    topology="pcu",
    pore_diameter_A=15.1,
    surface_area_m2g=3800.0,
    source="Li et al. Nature 1999, 402, 276. DOI:10.1038/46248. "
           "Pore volume: Kaye et al. JACS 2007, 129, 14176. DOI:10.1021/ja076877g",
    notes="IRMOF-1. One of the first MOFs. Very large pore, moderate stability. "
          "Sensitive to moisture.",
    tags=["H2-storage", "large-pore", "moisture-sensitive"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="MIL-101(Cr)",
        cavity_volume_A3=12700.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="MOF",
    ),
    category="MOF",
    formula="Cr3F(H2O)2O(BDC)3",
    topology="mtn",
    pore_diameter_A=34.0,
    surface_area_m2g=4100.0,
    source="Férey et al. Science 2005, 309, 2040. DOI:10.1126/science.1116275. "
           "Pore volume: Llewellyn et al. Langmuir 2008, 24, 7245. DOI:10.1021/la800227x",
    notes="Mesoporous MOF. Two cage types: 29 Å and 34 Å. 12700 Å³ is the larger cage. "
          "Water-stable, used in drug delivery and catalysis.",
    tags=["mesoporous", "water-stable", "drug-delivery", "catalysis"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="MOF-74(Zn)",
        cavity_volume_A3=350.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="MOF",
        max_guest_volume_A3=150.0,
    ),
    category="MOF",
    formula="Zn2(DOBDC)",
    topology="etb",
    pore_diameter_A=10.8,
    surface_area_m2g=816.0,
    source="Rosi et al. JACS 2005, 127, 1504. DOI:10.1021/ja0451230. "
           "CO2 data: Caskey et al. JACS 2008, 130, 10870. DOI:10.1021/ja8036096",
    notes="1D hexagonal channels, not cages. Volume per adsorption site ~350 Å³. "
          "Open Zn²⁺ sites for strong guest coordination.",
    tags=["open-metal-site", "CO2-capture", "1D-channel"],
))

# ─── Organic Cages ────────────────────────────────────────────────────────

_add(HostEntry(
    spec=NovelHostSpec(
        name="CC3",
        cavity_volume_A3=409.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="cage",
    ),
    category="cage",
    formula="[4+6] imine cage",
    topology="tetrahedral",
    pore_diameter_A=5.8,
    surface_area_m2g=409.0,
    source="Tozawa et al. Nature Mater. 2009, 8, 973. DOI:10.1038/nmat2545. "
           "Cavity: Hasell & Cooper, Nature Rev. Mater. 2016, 1, 16053. DOI:10.1038/natrevmats.2016.53",
    notes="Covalent cage CC3: tetrahedral [4+6] imine condensation. "
          "Intrinsic porosity in both crystal and amorphous phases.",
    tags=["porous-organic-cage", "shape-selective", "chiral-separation"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="CC1",
        cavity_volume_A3=82.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="cage",
    ),
    category="cage",
    formula="[2+3] imine cage",
    topology="trigonal-prismatic",
    pore_diameter_A=3.6,
    source="Hasell & Cooper, Nature Rev. Mater. 2016, 1, 16053. DOI:10.1038/natrevmats.2016.53",
    notes="Smallest member of the CC-n series. Small cavity, good for "
          "small gas molecule capture.",
    tags=["porous-organic-cage", "gas-capture"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="FeCage-1",
        cavity_volume_A3=1340.0,
        host_charge=8,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="cage",
    ),
    category="cage",
    formula="Fe8L6",
    topology="cubic",
    pore_diameter_A=13.0,
    source="Rizzuto & Nitschke, Nature Chem. 2017, 9, 903. DOI:10.1038/nchem.2758",
    notes="Self-assembled metal-organic cage. Fe⁸⁺ total framework charge. "
          "Large internal cavity for molecular encapsulation.",
    tags=["metal-organic-cage", "self-assembly", "encapsulation"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="Pd12-L24-sphere",
        cavity_volume_A3=900.0,
        host_charge=24,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="cage",
    ),
    category="cage",
    formula="[Pd12(L)24]24+",
    topology="cuboctahedral",
    pore_diameter_A=12.0,
    source="Fujita et al. Nature 2012, 489, 304. DOI:10.1038/nature11440. "
           "Cavity: Harris et al. Chem. Commun. 2013, 49, 6703. DOI:10.1039/c3cc43191f",
    notes="Fujita M₁₂L₂₄ sphere. Very large internal cavity. "
          "High framework charge (+24) from Pd²⁺ ions.",
    tags=["Fujita-cage", "self-assembly", "large-cavity"],
))

# ─── Zeolites ─────────────────────────────────────────────────────────────

_add(HostEntry(
    spec=NovelHostSpec(
        name="ZSM-5",
        cavity_volume_A3=135.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="zeolite",
        max_guest_volume_A3=90.0,
    ),
    category="zeolite",
    formula="NanAlnSi96-nO192",
    topology="MFI",
    pore_diameter_A=5.5,
    source="Olson et al. J. Phys. Chem. 1981, 85, 2238. DOI:10.1021/j150615a020. "
           "Channel dimensions: IZA Structure Database (iza-structure.org/databases/)",
    notes="10-ring zeolite. Two intersecting channel systems (5.3×5.6 Å and "
          "5.1×5.5 Å). 135 Å³ is the channel intersection void.",
    tags=["catalysis", "petroleum", "shape-selective"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="FAU-Y",
        cavity_volume_A3=3000.0,
        host_charge=0,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="zeolite",
    ),
    category="zeolite",
    formula="Na56[Al56Si136O384]",
    topology="FAU",
    pore_diameter_A=13.0,
    surface_area_m2g=900.0,
    source="Baerlocher et al. Atlas of Zeolite Framework Types, 6th Ed. "
           "IZA-SC, DOI:10.1016/B978-0-444-53064-2.X5186-X",
    notes="Faujasite supercage. Very large by zeolite standards. "
          "12-ring windows (7.4 Å aperture).",
    tags=["FCC-catalysis", "large-pore", "supercage"],
))

# ─── Synthetic Receptors ──────────────────────────────────────────────────

_add(HostEntry(
    spec=NovelHostSpec(
        name="Rebek-softball",
        cavity_volume_A3=400.0,
        host_charge=0,
        n_hbonds_host=8,
        portal_type="neutral",
        host_type="synthetic_receptor",
    ),
    category="synthetic_receptor",
    formula="self-complementary glycoluril dimer",
    pore_diameter_A=9.0,
    source="Rebek J. Angew. Chem. Int. Ed. 2005, 44, 2068. DOI:10.1002/anie.200462839",
    notes="H-bonded capsule. Self-assembles in organic solvent. "
          "~400 Å³ cavity, 55% packing rule.",
    tags=["capsule", "H-bond-assembly", "organic-solvent"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="Gibb-octa-acid",
        cavity_volume_A3=350.0,
        host_charge=-8,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="synthetic_receptor",
    ),
    category="synthetic_receptor",
    formula="OA cavitand dimer",
    pore_diameter_A=8.0,
    source="Gibb CLD, Gibb BC. JACS 2004, 126, 11408. DOI:10.1021/ja0475611. "
           "SAMPL data: Yin et al. JCTC 2017, 13, 2375. DOI:10.1021/acs.jctc.7b00264",
    notes="Deep-cavity cavitand. Dimerizes in water. Used in SAMPL blind challenges. "
          "High negative charge from 8 carboxylates.",
    tags=["aqueous", "SAMPL-benchmark", "cavitand", "anionic"],
))

_add(HostEntry(
    spec=NovelHostSpec(
        name="ExBox",
        cavity_volume_A3=309.0,
        host_charge=4,
        n_hbonds_host=0,
        portal_type="neutral",
        host_type="synthetic_receptor",
    ),
    category="synthetic_receptor",
    formula="Ex^2Box4+",
    pore_diameter_A=7.0,
    source="Barnes et al. JACS 2013, 135, 183. DOI:10.1021/ja307360n",
    notes="Extended bipyridinium box. Binds PAHs by π-stacking. "
          "+4 charge from 4 pyridinium nitrogen.",
    tags=["pi-stacking", "PAH-binding", "cationic", "Stoddart"],
))


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP API
# ═══════════════════════════════════════════════════════════════════════════

# Aliases for common names
_ALIASES = {
    "Cu-BTC": "HKUST-1",
    "MOF-199": "HKUST-1",
    "IRMOF-1": "MOF-5",
    "Zn-MOF-74": "MOF-74(Zn)",
    "CPO-27-Zn": "MOF-74(Zn)",
    "ZIF8": "ZIF-8",
    "UiO66": "UiO-66",
    "UiO66-NH2": "UiO-66-NH2",
    "MIL101": "MIL-101(Cr)",
    "MIL-101": "MIL-101(Cr)",
    "Y-zeolite": "FAU-Y",
    "zeolite-Y": "FAU-Y",
    "faujasite": "FAU-Y",
    "OA": "Gibb-octa-acid",
    "octa-acid": "Gibb-octa-acid",
    "softball": "Rebek-softball",
    "ExBox4+": "ExBox",
}


def get_host(name: str) -> NovelHostSpec:
    """Look up a novel host by name or alias.

    Parameters
    ----------
    name : str
        Host name (e.g. "HKUST-1", "CC3", "ZSM-5") or common alias.

    Returns
    -------
    NovelHostSpec
        Ready to pass to generate_for_host().

    Raises
    ------
    KeyError
        If name not found in library or aliases.
    """
    resolved = _ALIASES.get(name, name)
    if resolved not in _LIBRARY:
        avail = sorted(_LIBRARY.keys())
        raise KeyError(
            f"Unknown host '{name}'. Available: {avail}. "
            f"Aliases: {sorted(_ALIASES.keys())}"
        )
    return _LIBRARY[resolved].spec


def get_entry(name: str) -> HostEntry:
    """Look up full HostEntry with metadata."""
    resolved = _ALIASES.get(name, name)
    if resolved not in _LIBRARY:
        raise KeyError(f"Unknown host '{name}'.")
    return _LIBRARY[resolved]


def list_hosts(host_type: Optional[str] = None,
               tags: Optional[List[str]] = None,
               min_volume: float = 0.0,
               max_volume: float = float("inf")) -> List[HostEntry]:
    """List hosts matching filters.

    Parameters
    ----------
    host_type : str, optional
        Filter by category: "MOF", "cage", "zeolite", "synthetic_receptor".
    tags : list of str, optional
        Require ALL listed tags.
    min_volume, max_volume : float
        Cavity volume range in ų.

    Returns
    -------
    List[HostEntry]
        Matching entries, sorted by cavity volume.
    """
    results = []
    for entry in _LIBRARY.values():
        if host_type and entry.category != host_type:
            continue
        vol = entry.spec.cavity_volume_A3
        if vol < min_volume or vol > max_volume:
            continue
        if tags and not all(t in entry.tags for t in tags):
            continue
        results.append(entry)
    return sorted(results, key=lambda e: e.spec.cavity_volume_A3)


def list_names(host_type: Optional[str] = None) -> List[str]:
    """List all host names, optionally filtered by type."""
    return [e.spec.name for e in list_hosts(host_type=host_type)]


def host_summary() -> str:
    """Human-readable summary of the library."""
    lines = []
    lines.append("=" * 72)
    lines.append("Novel Host Library — MABE De Novo Guest Generation")
    lines.append("=" * 72)
    for cat in ["MOF", "cage", "zeolite", "synthetic_receptor"]:
        entries = list_hosts(host_type=cat)
        if not entries:
            continue
        lines.append(f"\n{'─'*3} {cat.upper()} ({len(entries)}) {'─'*50}")
        for e in entries:
            s = e.spec
            tags_str = ", ".join(e.tags[:3]) if e.tags else ""
            lines.append(
                f"  {s.name:20s}  V={s.cavity_volume_A3:>8.0f} ų  "
                f"d={e.pore_diameter_A:>5.1f} Å  "
                f"q={s.host_charge:+d}  [{tags_str}]"
            )
    lines.append(f"\nTotal: {len(_LIBRARY)} hosts, "
                 f"{len(_ALIASES)} aliases")
    return "\n".join(lines)


if __name__ == "__main__":
    print(host_summary())
