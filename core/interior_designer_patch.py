"""
core/interior_designer_patch.py - Extensions for physics-up structures.

Some structures ARE the binder:
- MIP: cavity IS the recognition (no separate chemistry needed)
- Zeolite: framework charge IS the ion exchange site
- LDH: interlayer charge IS the anion binder

For these, interior design means tuning the STRUCTURE, not decorating it.
"""

from core.assembly import RecognitionChemistry, InteriorDesign, InteriorSite, StructuralConstraint
from core.problem import Problem


def design_self_binding_interior(structure: StructuralConstraint,
                                  problem: Problem) -> InteriorDesign | None:
    """
    For structures that ARE the binder, design the interior without
    external recognition chemistry.
    Returns None if structure needs external recognition.
    """
    target = problem.target

    # ── MIP: the polymer IS the binder ────────────────────────────
    if structure.type == "mip":
        donor_atoms = []
        monomers = []
        if target.electronic and target.electronic.hardness_softness:
            hsab = target.electronic.hardness_softness
            if hsab == "hard":
                donor_atoms = ["O", "N"]
                monomers.append("methacrylic acid (O donor) + vinylpyridine (N donor)")
            elif hsab == "soft":
                donor_atoms = ["S", "N"]
                monomers.append("allylthiourea (S donor) + vinylimidazole (N donor)")
            else:
                donor_atoms = ["O", "N", "S"]
                monomers.append("methacrylic acid + vinylpyridine + allylthiourea (mixed donors)")
        else:
            donor_atoms = ["O", "N"]
            monomers.append("methacrylic acid + vinylpyridine (default)")

        return InteriorDesign(
            sites=[InteriorSite(
                recognition=RecognitionChemistry(
                    name=f"MIP cavity templated on {target.identity}",
                    type="imprinted_cavity",
                    donor_atoms=donor_atoms,
                    donor_type=target.electronic.hardness_softness if target.electronic else "unknown",
                    structure=f"Functional monomers: {monomers[0]}. Cross-linker: EGDMA. Template: {target.formula}",
                    kd_um=None,
                    source_tool="structural_library",
                    notes=(
                        f"The polymer cavity IS the binder. Molded around {target.formula}. "
                        f"Shape + charge + donor complementarity at sub-angstrom scale. "
                        f"No separate recognition chemistry needed."
                    ),
                ),
                copies=1,
                position_description="bulk polymer — every cavity is a binding site",
                attachment_chemistry="Monomers self-organize around template during polymerization",
            )],
            design_level="tertiary",
            total_binding_sites=1,
            unique_recognition_types=1,
            avidity_factor=1.0,
            cooperativity_note=(
                "MIP cavity provides simultaneous shape, charge, and donor atom complementarity "
                "— equivalent to a protein binding pocket but in synthetic polymer. "
                "Each cavity is an independent recognition site."
            ),
            geometric_match=f"Cavity geometry templated directly on {target.formula} — perfect geometric match by definition.",
            design_rationale=(
                f"Molecularly imprinted polymer templated on {target.formula}. "
                f"No external recognition chemistry needed — the cavity IS the binder. "
                f"Functional monomers {monomers[0]} arranged by target during polymerization. "
                f"Cross-linked with EGDMA. Template removed by washing."
            ),
        )

    # ── Zeolite: framework IS the ion exchanger ───────────────────
    if structure.type == "zeolite":
        charge = target.charge if target.charge else 2.0
        if charge > 0:
            return InteriorDesign(
                sites=[InteriorSite(
                    recognition=RecognitionChemistry(
                        name=f"Zeolite framework site for {target.identity}",
                        type="framework_site",
                        donor_atoms=["O"],
                        donor_type="hard",
                        structure=f"Si-O-Al framework negative charge site. Exchanges {target.formula} for Na+/K+.",
                        kd_um=None,
                        source_tool="structural_library",
                        notes=(
                            f"Zeolite framework has permanent negative charge from Al3+ substituting Si4+. "
                            f"Charge-compensating cations ({target.formula}) sit in framework cavities. "
                            f"Ion exchange selectivity from cavity size + charge density."
                        ),
                    ),
                    copies=structure.max_interior_sites,
                    position_description="framework cation exchange sites in sodalite/supercages",
                    attachment_chemistry="Inherent — no functionalization needed",
                )],
                design_level="composite",
                total_binding_sites=structure.max_interior_sites,
                unique_recognition_types=1,
                avidity_factor=float(structure.max_interior_sites) ** 0.5,
                cooperativity_note=(
                    f"Multiple framework sites in confined channels create cooperative capture. "
                    f"Once {target.formula} enters pore, sequential binding to channel sites. "
                    f"Molecular sieving at {structure.pore_size_nm} nm excludes larger competitors."
                ),
                geometric_match=(
                    f"Zeolite pore {structure.pore_size_nm} nm. "
                    f"Target hydrated radius determines fit — "
                    f"ions too large are physically excluded."
                ),
                kinetic_trapping=(
                    f"Angstrom-precision channels ({structure.pore_size_nm} nm) create strong kinetic trapping. "
                    f"Once dehydrated and inside, target encounters dense field of framework sites."
                ),
                design_rationale=(
                    f"Zeolite {structure.name} — the framework IS the ion exchanger. "
                    f"No external recognition chemistry needed. "
                    f"Al3+ substituting Si4+ creates permanent negative charge. "
                    f"Cation exchange selectivity from cavity size + charge density. "
                    f"Industrial track record, dirt cheap, extreme stability."
                ),
            )
        else:
            return None  # zeolites are cation exchangers — anions need LDH

    # ── LDH: interlayer IS the anion binder ───────────────────────
    if structure.type == "ldh":
        charge = target.charge if target.charge else 0.0
        if charge < 0:
            return InteriorDesign(
                sites=[InteriorSite(
                    recognition=RecognitionChemistry(
                        name=f"LDH interlayer site for {target.identity}",
                        type="interlayer_exchange",
                        donor_atoms=["electrostatic"],
                        donor_type="hard",
                        structure=f"Mg-Al brucite layer positive charge. Interlayer anion exchange for {target.formula}.",
                        kd_um=None,
                        source_tool="structural_library",
                        notes=(
                            f"Layered double hydroxide: positively charged brucite-like layers. "
                            f"Interlayer gallery contains exchangeable anions. "
                            f"{target.formula} displaces NO3-/Cl- from gallery by ion exchange. "
                            f"The LDH IS the binder for anionic targets."
                        ),
                    ),
                    copies=structure.max_interior_sites,
                    position_description="interlayer gallery between brucite sheets",
                    attachment_chemistry="Ion exchange — no functionalization needed",
                )],
                design_level="composite",
                total_binding_sites=structure.max_interior_sites,
                unique_recognition_types=1,
                avidity_factor=float(structure.max_interior_sites) ** 0.4,
                cooperativity_note=(
                    f"Dense interlayer sites create cooperative anion capture. "
                    f"Positively charged layers exclude cations — intrinsic selectivity."
                ),
                geometric_match=(
                    f"Interlayer gallery ~{structure.pore_size_nm} nm accommodates "
                    f"flat oxyanions (arsenate, chromate, selenite, phosphate)."
                ),
                design_rationale=(
                    f"Layered double hydroxide — the structure IS the anion binder. "
                    f"No external recognition chemistry needed. "
                    f"Cation exclusion by electrostatic repulsion from positive layers. "
                    f"Cheap ($3/g), scalable, nontoxic. "
                    f"Proven for arsenate, chromate, selenite, phosphate removal."
                ),
            )
        else:
            return None  # LDH is for anions

    # ── Mesoporous silica with organosilane functionalization ──────
    if structure.type == "mesoporous_silica":
        hsab = target.electronic.hardness_softness if target.electronic else None
        if hsab == "soft":
            silane = "3-mercaptopropyltrimethoxysilane (MPTMS)"
            donors = ["S"]
            dtype = "soft"
        elif hsab == "hard":
            silane = "3-aminopropyltriethoxysilane (APTES) + iminodiacetic acid"
            donors = ["N", "O"]
            dtype = "hard"
        else:
            silane = "APTES + MPTMS mixed functionalization"
            donors = ["N", "O", "S"]
            dtype = "mixed"

        return InteriorDesign(
            sites=[InteriorSite(
                recognition=RecognitionChemistry(
                    name=f"Organosilane-functionalized pore for {target.identity}",
                    type="silane_chelator",
                    donor_atoms=donors,
                    donor_type=dtype,
                    structure=f"Interior pore walls functionalized with {silane}.",
                    kd_um=None,
                    source_tool="structural_library",
                    notes=(
                        f"Mesoporous silica channels lined with organosilane chelators. "
                        f"1000+ m2/g surface area = dense binding field. "
                        f"Silane bonded covalently to Si-O wall — no leaching."
                    ),
                ),
                copies=structure.max_interior_sites,
                position_description=f"channel walls, {structure.pore_size_nm} nm pore diameter",
                attachment_chemistry=f"Covalent grafting: {silane} onto silica surface",
            )],
            design_level="composite",
            total_binding_sites=structure.max_interior_sites,
            unique_recognition_types=1,
            avidity_factor=float(min(structure.max_interior_sites, 20)) ** 0.6,
            cooperativity_note=(
                f"Dense chelator field inside {structure.pore_size_nm} nm channels. "
                f"Target encounters many binding sites in confined geometry."
            ),
            geometric_match=(
                f"Channel diameter {structure.pore_size_nm} nm — target must partially dehydrate to enter. "
                f"Selectivity from both pore size and chelator chemistry."
            ),
            kinetic_trapping=(
                f"Long channels ({structure.geometry}) create extended residence time. "
                f"Target diffuses through dense binding field — high probability of capture."
            ),
            design_rationale=(
                f"Mesoporous silica {structure.name} + {silane} interior lining. "
                f"Si-O backbone survives pH {structure.ph_stable_range[0]}-{structure.ph_stable_range[1]}. "
                f"Massive surface area, cheap, scalable, no degradation products."
            ),
        )

    # Not a self-binding structure — return None, let interior_designer handle it
    return None
