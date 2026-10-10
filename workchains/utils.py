import os
import yaml
from aiida.orm import load_code
from pymatgen.core.structure import Composition, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetryUndeterminedError
from uvsib.codes.utils import get_element_entries, get_structures_from_mpdb_by_composition
from uvsib.db.utils import query_structure, add_structures
from uvsib.workflows import settings


EHULL_SCAN = settings.EHULL_SCAN
DFT_FUNC = settings.DFT_FUNC

matcher = StructureMatcher(
    ltol=0.3,
    stol=0.5,
    angle_tol=7,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
    primitive_cell=True,
)

def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

_workchains_dir = os.path.join(settings.uvsib_directory,"workchains")
_ZPE_ALL: dict    = _read_yaml(os.path.join(_workchains_dir, "zpe_corrections.yaml"))

def load_zpe(reaction):
    """Return the ZPE correction dict (species name → ZPE correction (eV)) for the given reaction type ('oer', 'co2rr', 'noxrr')"""
    try:
        return dict(_ZPE_ALL[reaction])
    except:
        return None

def che_overpotential(steps, local_energy, zpe, equilibrium_potential,
                      reduction=True, n_electrons=None, pin_total=False):
    """Free-energy diagram and thermodynamic overpotential for a CHE pathway.

    This is the shared core for every electrochemical reaction calculator
    (CO2RR / NOXRR / NRR / ORR / HER / CER); ``oer.py`` keeps its own
    energy-conserving closure (which ``pin_total`` below now generalises).

    Thermodynamic pin (``pin_total=True``, requires ``n_electrons``)
    ----------------------------------------------------------------
    Exactly as ``oer.py`` closes its O2-release step on
    ``G_OER_TOTAL = 4 * 1.23 = n_e * U_eq``, this anchors the *integrated* free
    energy of the whole pathway to the experimental total and back-solves the
    final (gas-product release) elementary step:

        G_total = (-1 if reduction else +1) * n_electrons * U_eq      # eV
        dG_last = G_total - sum(dG_other_steps)

    ``G_total`` is the experimental overall reaction free energy at U = 0, set
    by the literature equilibrium potential ``U_eq`` (the audited per-pathway
    values in co2rr.py / noxrr.py, derived from standard formation free
    energies). Pinning removes the MLIP gas-phase reference error from the
    closed step and forces the correct overall thermodynamics, so an ideal
    one-(H+ + e-)-per-step catalyst gives eta = 0. It does NOT remove gas-phase
    reference error that sits in a *non-closed* step (e.g. the CO2 / NO reactant
    in step 1) -- that residual still needs the per-molecule electronic
    correction. The non-electrochemical "nominal" pathways (n_electrons == 0)
    are left unpinned.

    ``steps`` is the list of elementary-reaction dictionaries from the
    pathway definition. Each dict maps a species to its **signed
    stoichiometric coefficient in that single elementary reaction**
    (products +, reactants -). A leading empty ``{}`` reference marker is
    tolerated and ignored. CHE bookkeeping for the proton/ion couple:

        * one (H+ + e-) consumed (reduction step)  -> 'H2':  -1/2
        * one (H+ + e-) released (oxidation step)   -> 'H2':  +1/2
        * a released gas molecule (H2O, NH3, ...)   -> +1
        * a consumed gas reactant (NO, CO2, ...)    -> -1

    Each step free energy is evaluated **directly** from its own dict:

        dG_i = sum_q (E[q] + ZPE[q]) * coeff_q

    The previous implementation instead summed each dict into a ``dga`` list
    and then took consecutive differences ``dga[1:] - dga[:-1]``. Because the
    dicts are *incremental* reactions (each references the previous
    intermediate), differencing them double-subtracts the shared intermediate;
    with raw total energies on the MLIP scale (H2O ~ -2080 eV) the uncancelled
    atomic energies blow the overpotential up to thousands of volts. Summing
    each dict on its own keeps every step bounded and physical.

    The thermodynamic (limiting-potential) overpotential, with the
    potential-determining step PDS = max_i dG_i (eV) at U = 0:

        reduction:  eta = PDS + U_eq   (drive at U < U_eq; ideal catalyst -> 0)
        oxidation:  eta = PDS - U_eq   (drive at U > U_eq; ideal catalyst -> 0)

    Returns ``(overpotential, dg_steps, dg_cumulative)`` where ``dg_steps`` are
    the per-step free energies at U = 0 V and ``dg_cumulative`` is their
    running sum ``[0, dG1, dG1+dG2, ...]`` at U = 0 V (matching oer.py).
    """
    dg_steps = [
        sum((local_energy[q] + zpe[q]) * coeff for q, coeff in r.items())
        for r in steps if r          # skip the {} reference-state marker
    ]

    # OER-style thermodynamic pin (see docstring). Skipped when n_electrons is
    # None/0, i.e. the non-electrochemical nominal pathways.
    if pin_total and n_electrons:
        sign = -1.0 if reduction else 1.0
        g_total = sign * n_electrons * equilibrium_potential
        dg_steps[-1] = g_total - sum(dg_steps[:-1])

    pds = max(dg_steps)
    overpotential = (pds + equilibrium_potential) if reduction \
        else (pds - equilibrium_potential)

    dg_cumulative, running = [0.0], 0.0
    for d in dg_steps:
        running += d
        dg_cumulative.append(running)

    return overpotential, dg_steps, dg_cumulative

def get_primitive_cell(struct_dict):
    """Refine a structure dictionary into its primitive cell"""
    structure = Structure.from_dict(struct_dict)

    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.05, angle_tolerance=5)
    except SymmetryUndeterminedError:
        print('Symmetry reduction failed')
        print(structure)
        return structure

    try:
        prim_struct = sga.get_primitive_standard_structure()
    except:
        prim_struct = sga.find_primitive() or structure

    return prim_struct

def get_output_as_entry(wch):
    """Extract structures and energies from an ML energy prediction calculation"""
    entries = []
    output_dict = wch.outputs.output_dict
    for indx, struct in enumerate(output_dict["structures"]):
        entries.append(ComputedStructureEntry(
            structure = Structure.from_dict(struct),
            energy = output_dict["energies"][indx])
        )
    return entries


def split_relax_output(wch, n_main):
    """Split one bundled relax job into ``(main_entries, ref_entries)``.

    Used when the elemental reference structures are appended after the main
    (generated / CSP) structures in a single relax CalcJob, so the MLIP is
    loaded once instead of in a separate reference job. The relax runner emits
    ``indices`` (the original input position of each converged structure), so
    inputs ``[0, n_main)`` are the main structures and ``[n_main, ...)`` the
    appended references. Splitting on the input index -- not the output
    position -- is robust to relax.py dropping non-converged structures.
    Falls back to output order if ``indices`` is absent (legacy jobs).
    """
    output_dict = wch.outputs.output_dict.get_dict()
    structs = output_dict.get("structures", [])
    energies = output_dict.get("energies", [])
    indices = output_dict.get("indices")
    if not indices or len(indices) != len(structs):
        indices = list(range(len(structs)))
    main, refs = [], []
    for struct, energy, idx in zip(structs, energies, indices):
        entry = ComputedStructureEntry(
            structure=Structure.from_dict(struct), energy=energy)
        (refs if idx >= n_main else main).append(entry)
    return main, refs

def unique_low_energy_chemsys(chemical_system, entries, ehull, min_n_return=None, element_entries=None):
    """Select the unique lowest-energy structures for a given chemical system.

    ``element_entries`` are the elemental hull endpoints to add. Pass the
    on-method MLIP references (``element_reference_entries``); when ``None`` the
    bundled DFT references are used (``DFT_FUNC`` selects GGA/r2SCAN) -- legacy
    behaviour, kept for backward compatibility.

    ``min_n_return``: if fewer than this many unique structures are within
    ``ehull``, pad with the next-lowest unique ones above it.
    """
    entries = list(entries)
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(element_entries if element_entries is not None
                       else get_element_entries(elements, DFT_FUNC))
    pd = PhaseDiagram(entries)

    # sorted by e_above_hull so each duplicate group keeps its lowest-energy
    # copy and padding (min_n_return) takes the next most stable structures
    candidates = [(entry, pd.get_e_above_hull(entry)) for entry in pd.entries
                  if entry.composition.chemical_system == chemical_system]
    candidates.sort(key=lambda x: x[1])

    stable_entries = []
    existing_structs = []

    for entry, eh in candidates:
        if eh > ehull and len(stable_entries) >= (min_n_return or 0):
            break

        prim_struct = get_primitive_cell(entry.structure.as_dict())

        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        stable_entries.append(entry)
    return stable_entries

def unique_low_energy_comp(chemical_formula, entries, ehull, min_n_return=None, element_entries=None):
    """Select the lowest-energy unique structures for a given chemical formula.

    ``element_entries``: elemental hull endpoints to add (the on-method MLIP
    references from ``element_reference_entries``); ``None`` -> bundled DFT
    references (legacy).
    """
    entries = list(entries)
    chemical_system = Composition(chemical_formula).chemical_system
    if "-" in chemical_system:
        elements = chemical_system.split("-")
        entries.extend(element_entries if element_entries is not None
                       else get_element_entries(elements, DFT_FUNC))
    pd = PhaseDiagram(entries)

    candidates = []
    existing_structs = []

    stable_entries = []
    ehulls = []

    for en in pd.entries:
        if en.composition.reduced_formula != chemical_formula:
            continue

        prim_struct = get_primitive_cell(en.structure.as_dict())
        if any(matcher.fit(prim_struct, s) for s in existing_structs):
            continue

        existing_structs.append(prim_struct)
        candidates.append((en, pd.get_e_above_hull(en)))

    candidates.sort(key=lambda x: x[1])

    for en, eh in candidates:
        if eh <= ehull:
            stable_entries.append(en)
            ehulls.append(eh)

    if min_n_return:
        if len(stable_entries) < min_n_return:
            for en, eh in candidates[len(stable_entries):min_n_return]:
                stable_entries.append(en)
                ehulls.append(eh)
    return stable_entries, ehulls

def add_from_mpdb(chemical_formula):
    """Add missing stable and experimental structures and their reference elements from MPDB."""
    results = query_structure({"composition": chemical_formula}, method = "DFT")
    existing_sources = {result.source for result in results}

    missing_stb = "MPDB_stb" not in existing_sources
    missing_exp = "MPDB_exp" not in existing_sources

    exp_structures = []
    stb_structures = []
    if missing_stb or missing_exp:
        stb_structures, exp_structures = (
            get_structures_from_mpdb_by_composition(
                chemical_formula,
                EHULL_SCAN,
            )
        )
    if missing_stb and stb_structures:
        structures = [(get_primitive_cell(structure).as_dict(), None) for structure, _ in stb_structures ]
        mp_ids = [mp_id for _, mp_id in stb_structures]
        add_structures("MPDB_stb", "DFT", structures, mp_ids=mp_ids)
    if missing_exp and exp_structures:
        structures = [(get_primitive_cell(structure).as_dict(), None) for structure, _ in exp_structures ]
        mp_ids = [mp_id for _, mp_id in exp_structures]
        add_structures("MPDB_exp", "DFT", structures, mp_ids=mp_ids)
    # add reference structures (elements)
    elements = Composition(chemical_formula).chemical_system.split("-")
    missing_el = []
    for el in elements:
        rows = query_structure({"chemsys": el}, method="DFT") or []

        if not any(row.source == "MPDB_ref" for row in rows):
            missing_el.append(el)

    el_entries = get_element_entries(missing_el, DFT_FUNC)
    el_structures = [(e.structure.as_dict(), None) for e in el_entries]
    add_structures("MPDB_ref", "DFT", el_structures)

def get_code(model_key):
    """
    Helper to fetch builder.code, model_path, and device 
    """
    return load_code(settings.configs["codes"][model_key]["code_string"])

def get_model_device(ML_model):
    """Return (model_path, device) for the given ML model."""
    path_to_pretrained_models = settings.configs["models"]["path_to_pretrained_models"]
    model = settings.configs["models"][ML_model]
    model_path = os.path.join(path_to_pretrained_models, model)
    device = settings.configs["codes"][ML_model]["job_script"]["device"]
    return model, model_path, device

def get_ref_entries(chemical_formula, ML_model):
    el_entries = []
    missing = []
    elements = Composition(chemical_formula).chemical_system.split("-")
    for el in elements:
        rows = query_structure({"chemsys": el}, method = ML_model, source = "MPDB_ref")
        if rows:
            row = rows[0]
            struct = Structure.from_dict(row.structure)
            el_entries.append(
                    ComputedStructureEntry(
                    structure=struct,
                    energy=row.energy,
                    data={"struct_uuid": row.structure_uuid})
            )
        else:
            missing.append(el)
    if missing:
        el_entries.extend(get_element_entries(missing, DFT_FUNC))
    return el_entries, missing
