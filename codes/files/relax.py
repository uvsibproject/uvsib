import json
import argparse
from pymatgen.core import Lattice, Structure
from ase import Atoms
from ase.optimize import FIRE
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.filters import FrechetCellFilter


def ase_to_pmg(atoms):
    """
    Convert an ASE Atoms object to a pymatgen Structure
    """
    lattice = atoms.cell.array.tolist()
    symbols = atoms.get_chemical_symbols()
    frac_coords = atoms.get_scaled_positions().tolist()
    lattice_obj = Lattice(lattice)
    return Structure(lattice_obj, symbols, frac_coords, coords_are_cartesian=False)

def pmg_to_ase(pmg_structure):
    """
    Convert a pymatgen Structure to an ASE Atoms object
    """
    scaled_positions = pmg_structure.frac_coords
    symbols = [str(site.specie) for site in pmg_structure.sites]
    cell = pmg_structure.lattice.matrix
    return Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)

def is_garbage(atoms, max_cell_component=100.0):
    """
    Sanity check for blown-up cells: a variable-cell relaxation can "converge"
    with fragments pulled apart in vacuum (zero force/stress), often with an
    unphysically low MLIP energy. Flags any lattice-vector Cartesian component
    larger than ``max_cell_component`` (Angstrom).
    """
    return any(abs(v) > max_cell_component for vec in atoms.cell.array for v in vec)

def relax_structures(calc, fmax, max_steps):
    """
    Relax a list of ASE Atoms objects 
    """
    with open("input_structures.json", "r") as f:
        structure_list = json.load(f)

    relaxed_structures = []
    energies = []
    epas = []
    indices = []          # original input position of each converged structure
    num_failed = 0
    for i, structure in enumerate(structure_list):
        atoms = pmg_to_ase(Structure.from_dict(structure))
        atoms.calc = calc
        cell_filter = FrechetCellFilter(atoms)

        opt = BFGSLineSearch(cell_filter, logfile="opt.log")
        try:
            opt.run(fmax=fmax, steps=max_steps)
        except RuntimeError:
            # BFGSLineSearch's Wolfe-condition line search can fail to find a
            # valid step on some structures; FIRE has no line search and is
            # more robust, so fall back to it and continue from where the
            # atoms currently are.
            opt = FIRE(cell_filter, logfile="opt.log")
            opt.run(fmax=fmax, steps=max_steps)

        # blown-up cells are dropped and counted as failed, like non-converged ones
        if opt.converged and not is_garbage(atoms):
            energy = float(atoms.get_potential_energy())
            energies.append(energy)
            pmg_structure = ase_to_pmg(atoms)
            relaxed_structures.append(pmg_structure.as_dict())
            epas.append(energy/len(pmg_structure.sites))
            indices.append(i)
        else:
            num_failed += 1

    # ``indices`` lets a caller that bundled several structure groups into one
    # relax job (e.g. generated/CSP structures + appended elemental references)
    # map each surviving output back to its input position -- robust to the
    # non-converged structures dropped above. See utils.split_relax_output.
    to_dump = {'structures': relaxed_structures, 'energies': energies,
               'epas': epas, 'indices': indices}

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)

    with open('total.txt', 'w') as f:
        f.write(str(len(structure_list)))

    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    relax_structures(calc, args.fmax, args.max_steps)
