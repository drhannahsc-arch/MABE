"""
core/metal_expansion.py -- Sprint 40: Expanded Metal Calibration Library

Additional well-characterized metal-ligand stability constants from:
  - Martell & Smith, Critical Stability Constants (Plenum, 1974-1989)
  - IUPAC Stability Constants Database
  - Smith & Martell, Critical Stability Constants Supplement (1989)

Focus on systematic series that isolate specific physics:
  1. Ammonia stepwise: isolates N_amine exchange energy per donor
  2. Acetate/oxalate: isolates O_carboxylate exchange energy
  3. Glycinate across metals: mixed N/O, Irving-Williams ordering
  4. Chloride complexes: weak monodentate reference
  5. Hydroxide: strong O_hydroxyl reference
  6. Ethylenediamine stepwise: chelate ring effect per ring
  7. Macrocyclic: cyclam/cyclen for preorganization

All at 25C, I=0.1M unless noted. log K = overall stability constant (beta).
"""

from core.universal_schema import UniversalComplex


def build_metal_expansion():
    """Return list of additional UniversalComplex entries for metal calibration."""
    entries = []

    def _add(name, metal, metal_charge, guest, donors, donor_subs,
             log_K, dent, chelate_rings=0, ring_sizes=None, geom="octahedral",
             source="Martell & Smith"):
        uc = UniversalComplex(
            name=name,
            binding_mode="metal_coordination",
            log_Ka_exp=log_K,
            dg_exp_kj=-log_K * 5.71,
            temperature_C=25.0,
            ionic_strength_M=0.1,
            ph=7.0,
            solvent="water",
            host_name=metal,
            host_type="metal_ion",
            host_charge=metal_charge,
            guest_name=guest,
            metal_formula=metal,
            metal_charge=metal_charge,
            donor_atoms=donors,
            donor_subtypes=donor_subs,
            chelate_rings=chelate_rings,
            ring_sizes=ring_sizes or [],
            denticity=dent,
            donor_type=_classify_donor_type(donor_subs),
            source=source,
            phase="Phase1-5",
            confidence="high",
            scaffold_type="free",
            geometry=geom,
        )
        entries.append(uc)

    # ==================================================================
    # 1. AMMONIA SERIES — isolates N_amine exchange energy
    # log beta_n for M(NH3)_n. Using cumulative (overall) constants.
    # ==================================================================

    # Cu2+ + nNH3 (n=1..4, max stable in solution)
    _add("Cu-NH3_1", "Cu2+", 2, "NH3_1", ["N"], ["N_amine"], 4.0, 1)
    _add("Cu-NH3_2", "Cu2+", 2, "NH3_2", ["N","N"], ["N_amine"]*2, 7.5, 2)
    _add("Cu-NH3_4", "Cu2+", 2, "NH3_4", ["N"]*4, ["N_amine"]*4, 12.6, 4, geom="square_planar")

    # Ni2+ + nNH3
    _add("Ni-NH3_1", "Ni2+", 2, "NH3_1", ["N"], ["N_amine"], 2.8, 1)
    _add("Ni-NH3_2", "Ni2+", 2, "NH3_2", ["N","N"], ["N_amine"]*2, 5.0, 2)
    _add("Ni-NH3_3", "Ni2+", 2, "NH3_3", ["N"]*3, ["N_amine"]*3, 6.8, 3)
    # (already have Ni-NH3_6=8.6 in seed library)

    # Zn2+ + nNH3
    _add("Zn-NH3_1", "Zn2+", 2, "NH3_1", ["N"], ["N_amine"], 2.2, 1)
    _add("Zn-NH3_2", "Zn2+", 2, "NH3_2", ["N","N"], ["N_amine"]*2, 4.5, 2)
    _add("Zn-NH3_4", "Zn2+", 2, "NH3_4", ["N"]*4, ["N_amine"]*4, 9.1, 4, geom="tetrahedral")

    # Co2+ + nNH3
    _add("Co-NH3_1", "Co2+", 2, "NH3_1", ["N"], ["N_amine"], 2.1, 1)
    _add("Co-NH3_2", "Co2+", 2, "NH3_2", ["N","N"], ["N_amine"]*2, 3.7, 2)
    _add("Co-NH3_4", "Co2+", 2, "NH3_4", ["N"]*4, ["N_amine"]*4, 5.6, 4)

    # Cd2+ + nNH3 (soft metal, borderline)
    _add("Cd-NH3_1", "Cd2+", 2, "NH3_1", ["N"], ["N_amine"], 2.6, 1)
    _add("Cd-NH3_2", "Cd2+", 2, "NH3_2", ["N","N"], ["N_amine"]*2, 4.7, 2)
    _add("Cd-NH3_4", "Cd2+", 2, "NH3_4", ["N"]*4, ["N_amine"]*4, 7.1, 4)

    # ==================================================================
    # 2. ACETATE SERIES — isolates O_carboxylate monodentate
    # ==================================================================

    _add("Cu-acetate", "Cu2+", 2, "acetate", ["O"], ["O_carboxylate"], 2.2, 1)
    _add("Ni-acetate", "Ni2+", 2, "acetate", ["O"], ["O_carboxylate"], 1.4, 1)
    _add("Zn-acetate", "Zn2+", 2, "acetate", ["O"], ["O_carboxylate"], 1.6, 1)
    _add("Co-acetate", "Co2+", 2, "acetate", ["O"], ["O_carboxylate"], 1.5, 1)
    _add("Pb-acetate", "Pb2+", 2, "acetate", ["O"], ["O_carboxylate"], 2.7, 1)
    _add("Cd-acetate", "Cd2+", 2, "acetate", ["O"], ["O_carboxylate"], 1.9, 1)
    _add("Mn-acetate", "Mn2+", 2, "acetate", ["O"], ["O_carboxylate"], 1.4, 1)
    _add("Ca-acetate", "Ca2+", 2, "acetate", ["O"], ["O_carboxylate"], 1.2, 1)
    _add("Fe3-acetate", "Fe3+", 3, "acetate", ["O"], ["O_carboxylate"], 3.4, 1)

    # ==================================================================
    # 3. OXALATE — bidentate O,O chelator
    # ==================================================================

    _add("Cu-oxalate", "Cu2+", 2, "oxalate", ["O","O"], ["O_carboxylate"]*2, 6.2, 2,
         chelate_rings=1, ring_sizes=[5])
    _add("Ni-oxalate", "Ni2+", 2, "oxalate", ["O","O"], ["O_carboxylate"]*2, 5.2, 2,
         chelate_rings=1, ring_sizes=[5])
    _add("Zn-oxalate", "Zn2+", 2, "oxalate", ["O","O"], ["O_carboxylate"]*2, 4.9, 2,
         chelate_rings=1, ring_sizes=[5])
    _add("Fe3-oxalate", "Fe3+", 3, "oxalate", ["O","O"], ["O_carboxylate"]*2, 9.4, 2,
         chelate_rings=1, ring_sizes=[5])
    _add("Pb-oxalate", "Pb2+", 2, "oxalate", ["O","O"], ["O_carboxylate"]*2, 4.9, 2,
         chelate_rings=1, ring_sizes=[5])
    _add("Ca-oxalate", "Ca2+", 2, "oxalate", ["O","O"], ["O_carboxylate"]*2, 3.0, 2,
         chelate_rings=1, ring_sizes=[5])
    _add("Mn-oxalate", "Mn2+", 2, "oxalate", ["O","O"], ["O_carboxylate"]*2, 3.9, 2,
         chelate_rings=1, ring_sizes=[5])

    # ==================================================================
    # 4. GLYCINATE across Irving-Williams series
    # Mixed N_amine + O_carboxylate — tests donor mixing
    # ==================================================================

    _add("Cu-glycinate2", "Cu2+", 2, "glycinate2",
         ["N","N","O","O"], ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
         15.1, 4, chelate_rings=2, ring_sizes=[5,5], geom="square_planar")
    # Already have Cu-glycine2 and Zn-glycine2 in seed library
    _add("Ni-glycinate2", "Ni2+", 2, "glycinate2",
         ["N","N","O","O"], ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
         11.1, 4, chelate_rings=2, ring_sizes=[5,5])
    _add("Co-glycinate2", "Co2+", 2, "glycinate2",
         ["N","N","O","O"], ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
         9.2, 4, chelate_rings=2, ring_sizes=[5,5])
    _add("Mn-glycinate2", "Mn2+", 2, "glycinate2",
         ["N","N","O","O"], ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
         5.5, 4, chelate_rings=2, ring_sizes=[5,5])
    _add("Cd-glycinate2", "Cd2+", 2, "glycinate2",
         ["N","N","O","O"], ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
         8.0, 4, chelate_rings=2, ring_sizes=[5,5])
    _add("Pb-glycinate2", "Pb2+", 2, "glycinate2",
         ["N","N","O","O"], ["N_amine","N_amine","O_carboxylate","O_carboxylate"],
         8.5, 4, chelate_rings=2, ring_sizes=[5,5])

    # ==================================================================
    # 5. ETHYLENEDIAMINE (en) stepwise — isolates chelate ring effect
    # Already have en2(Cu) and en3(Co,Ni,Zn) in seeds
    # ==================================================================

    _add("Cu-en1", "Cu2+", 2, "en1",
         ["N","N"], ["N_amine","N_amine"],
         10.5, 2, chelate_rings=1, ring_sizes=[5])
    _add("Ni-en1", "Ni2+", 2, "en1",
         ["N","N"], ["N_amine","N_amine"],
         7.5, 2, chelate_rings=1, ring_sizes=[5])
    _add("Ni-en2", "Ni2+", 2, "en2",
         ["N"]*4, ["N_amine"]*4,
         13.9, 4, chelate_rings=2, ring_sizes=[5,5])
    _add("Zn-en1", "Zn2+", 2, "en1",
         ["N","N"], ["N_amine","N_amine"],
         5.7, 2, chelate_rings=1, ring_sizes=[5])
    _add("Zn-en2", "Zn2+", 2, "en2",
         ["N"]*4, ["N_amine"]*4,
         10.6, 4, chelate_rings=2, ring_sizes=[5,5])
    _add("Co-en1", "Co2+", 2, "en1",
         ["N","N"], ["N_amine","N_amine"],
         5.9, 2, chelate_rings=1, ring_sizes=[5])
    _add("Co-en2", "Co2+", 2, "en2",
         ["N"]*4, ["N_amine"]*4,
         10.7, 4, chelate_rings=2, ring_sizes=[5,5])

    # ==================================================================
    # 6. PYRIDINE series — isolates N_pyridine
    # ==================================================================

    _add("Cu-pyridine2", "Cu2+", 2, "pyridine2",
         ["N","N"], ["N_pyridine","N_pyridine"], 4.4, 2)
    _add("Ni-pyridine2", "Ni2+", 2, "pyridine2",
         ["N","N"], ["N_pyridine","N_pyridine"], 3.5, 2)
    _add("Zn-pyridine2", "Zn2+", 2, "pyridine2",
         ["N","N"], ["N_pyridine","N_pyridine"], 2.4, 2)

    # ==================================================================
    # 7. IMIDAZOLE — isolates N_imidazole (His mimic)
    # ==================================================================

    _add("Cu-imidazole4", "Cu2+", 2, "imidazole4",
         ["N"]*4, ["N_imidazole"]*4, 15.2, 4, geom="square_planar")
    _add("Ni-imidazole4", "Ni2+", 2, "imidazole4",
         ["N"]*4, ["N_imidazole"]*4, 10.0, 4)
    _add("Zn-imidazole4", "Zn2+", 2, "imidazole4",
         ["N"]*4, ["N_imidazole"]*4, 10.2, 4, geom="tetrahedral")
    _add("Co-imidazole4", "Co2+", 2, "imidazole4",
         ["N"]*4, ["N_imidazole"]*4, 9.3, 4)

    # ==================================================================
    # 8. HYDROXIDE — strong O_hydroxyl
    # ==================================================================

    _add("Fe3-OH", "Fe3+", 3, "hydroxide", ["O"], ["O_hydroxyl"], 11.8, 1)
    _add("Al-OH", "Al3+", 3, "hydroxide", ["O"], ["O_hydroxyl"], 9.0, 1)
    _add("Cu-OH", "Cu2+", 2, "hydroxide", ["O"], ["O_hydroxyl"], 6.3, 1)
    _add("Pb-OH", "Pb2+", 2, "hydroxide", ["O"], ["O_hydroxyl"], 6.3, 1)
    _add("Zn-OH", "Zn2+", 2, "hydroxide", ["O"], ["O_hydroxyl"], 5.0, 1)

    # ==================================================================
    # 9. THIOL (cysteine-like) — extends S_thiolate series
    # ==================================================================

    _add("Hg-cysteine", "Hg2+", 2, "cysteine_S",
         ["S"], ["S_thiolate"], 14.4, 1)
    _add("Ag-cysteine", "Ag+", 1, "cysteine_S",
         ["S"], ["S_thiolate"], 11.9, 1)
    _add("Cu-cysteine", "Cu2+", 2, "cysteine_S",
         ["S","N","O"], ["S_thiolate","N_amine","O_carboxylate"],
         10.2, 3, chelate_rings=2, ring_sizes=[5,5])

    # ==================================================================
    # 10. MACROCYCLIC — cyclam (1,4,8,11-tetraazacyclotetradecane)
    # ==================================================================

    _add("Cu-cyclam", "Cu2+", 2, "cyclam",
         ["N"]*4, ["N_amine"]*4, 27.2, 4,
         chelate_rings=4, ring_sizes=[5,6,5,6], geom="square_planar",
         source="IUPAC")
    _add("Ni-cyclam", "Ni2+", 2, "cyclam",
         ["N"]*4, ["N_amine"]*4, 22.2, 4,
         chelate_rings=4, ring_sizes=[5,6,5,6],
         source="IUPAC")
    _add("Zn-cyclam", "Zn2+", 2, "cyclam",
         ["N"]*4, ["N_amine"]*4, 15.5, 4,
         chelate_rings=4, ring_sizes=[5,6,5,6], geom="tetrahedral",
         source="IUPAC")

    return entries


def _classify_donor_type(subtypes):
    """Classify overall donor type from subtypes list."""
    types = set(s.split("_")[0] for s in subtypes)
    if len(types) == 1:
        t = list(types)[0]
        if t == "N":
            return "nitrogen"
        elif t == "O":
            return "oxygen"
        elif t == "S":
            return "sulfur"
    return "mixed"
