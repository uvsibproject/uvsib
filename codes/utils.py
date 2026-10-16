import os
import json
from ase.constraints import FixAtoms
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from mp_api.client import MPRester
from ase import Atoms
from uvsib.workflows import settings


def get_cmdline(job_info):
    """Construct command line"""
    cmdline = []
    cmdline.append(f"--ML_model={job_info['ML_model']}")
    cmdline.append(f"--model={job_info['model_name']}")
    cmdline.append(f"--model_path={job_info['model_path']}")
    cmdline.append(f"--task_name={job_info['model_head']}")
    cmdline.append(f"--device={job_info['device']}")

    job_type = job_info['job_type']

    if job_type == 'relax':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}"]
        )
    elif job_type == 'hopping':
        cmdline.extend([
            f"--mh_steps={job_info['mh_steps']}"
        ])
    elif job_type == 'face_build':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--max_miller_idx={job_info['max_miller_idx']}",
            f"--max_num_surf={job_info['max_num_surf']}"]
        )
    elif job_type == 'adsorbates':
        cmdline.extend([
            f"--slab_energy={job_info['slab_energy']}",
            f"--fmax={job_info['fmax']}",
            f"--max_steps={job_info['max_steps']}",
            f"--reaction={job_info['reaction']}",
            f"--pathway={job_info['pathway']}",
            f"--no-validate"]
        )
    elif job_type == 'akmc':
        cmdline.extend([
            f"--fmax={job_info['fmax']}",
            f"--dimer_fmax={job_info['dimer_fmax']}",
            f"--relax_steps={job_info['relax_steps']}",
            f"--dimer_steps={job_info['dimer_steps']}",
            f"--searches_per_minimum={job_info['searches_per_minimum']}",
            f"--temperature={job_info['temperature']}",
            f"--prefactor={job_info['prefactor']}",
            f"--dimer_displacement={job_info['dimer_displacement']}",
            f"--product_displacement={job_info['product_displacement']}",
            f"--maxstep={job_info['maxstep']}",
            f"--seed={job_info['seed']}"]
        )
    return cmdline

def get_element_entries(chemsys_list, functional):
    if functional == "GGA":
        file = os.path.join(settings.uvsib_directory, 'codes', 'files', 'gga_ggau_entries.json')
    else:
        file = os.path.join(settings.uvsib_directory, 'codes', 'files', 'r2scan_entries.json')
    with open(file, "r") as f:
        entries = json.load(f)
    output_entries = []
    for entry in entries:
        cse = ComputedStructureEntry.from_dict(entry["entries"][functional])
        if cse.composition.chemical_system in chemsys_list:
            output_entries.append(cse)
    return output_entries

def pmg_to_ase(structure):
    """
    Convert a pymatgen Structure to an ASE Atoms object
    """
    symbols = [str(site.specie) for site in structure.sites]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    pbc = [True, True, True]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    return atoms

def ase_to_pmg(atoms):
    """
    Convert an ASE Atoms object to a pymatgen Structure 
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)

    selective_dynamics = [[True, True, True] for _ in atoms]

    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            for idx in constraint.index:
                selective_dynamics[idx] = [False, False, False]
    return Structure(
        lattice=Lattice(lattice),
        species=symbols,
        coords=frac_coords,
        coords_are_cartesian=False,
        site_properties={"selective_dynamics": selective_dynamics},
    )

def get_structures_from_mpdb_by_composition(chemical_formula, e_hull=0.1):
    """Get stable structures from the MPDB.

    Parameters
    ----------
    chemical_formula : str
    e_hull : float
        Maximum energy above hull in eV/atom.

    Returns
    -------
    stable_structures, exp_structures : list[tuple[dict, str]]
        Two lists of ``(structure_dict, mp_id)`` pairs, where ``mp_id`` is the
        Materials Project material id (e.g. ``"mp-1234"``) the structure came
        from. ``stable_structures`` holds the theoretical entries, and
        ``exp_structures`` the experimentally observed ones.
    """
    stable_structures = []
    exp_structures = []
    api_key = settings.api_key

    search_kwargs = {
        "formula": chemical_formula,
        "energy_above_hull": (0, e_hull),
        "fields": ["material_id", "structure", "energy_above_hull", "theoretical"],
    }

    with MPRester(api_key, use_document_model=False) as mpr:
        summaries = mpr.materials.summary.search(**search_kwargs)

    if not summaries:
        return [], []

    for summary in summaries:
        raw_id = summary.get("material_id")
        mp_id = str(raw_id) if raw_id is not None else None
        structure = summary["structure"]
        struct_dict = structure.as_dict() if hasattr(structure, "as_dict") else structure
        if summary.get("theoretical"):
            stable_structures.append((struct_dict, mp_id))
        else:
            exp_structures.append((struct_dict, mp_id))

    return stable_structures, exp_structures
