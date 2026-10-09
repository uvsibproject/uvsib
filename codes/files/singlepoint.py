"""Single-point energy / force / stress evaluation of fixed geometries.

Used by EhullUncertaintyWorkChain: a committee MLIP re-scores the structures
the primary MLIP relaxed, WITHOUT relaxing them, so every model's energy refers
to the same geometry. Shipped as ``aiida.py`` next to ``_calculators.py`` like
``relax.py``; the job runs in the committee model's own code / environment.

Input (``input_structures.json``)
    [{"uuid": <structure uuid>, "structure": <pymatgen Structure.as_dict()>,
      "index": <input position>}, ...]

Output (``output.json``)
    {
      "results": [{"uuid", "index", "energy", "epa",
                   "max_force",      # max |F_i|, eV/A
                   "force_sq_mean",  # sum_i |F_i|^2 / N, eV^2/A^2
                   "stress_voigt",   # [xx, yy, zz, yz, xz, xy], GPa (None if unavailable)
                   "max_stress",     # max |sigma_voigt|, GPa (None if unavailable)
                   "pressure"},      # -trace(sigma)/3, GPa (None if unavailable)
                  ...],
      "failed": [{"uuid", "index", "reason"}, ...]
    }

The uuid/index are echoed per record, so attribution never relies on output
position (failed structures are dropped from ``results``).
"""

import json
import argparse
import numpy as np
from pymatgen.core import Structure
from ase import Atoms

EV_A3_TO_GPA = 160.21766208


def pmg_to_ase(pmg_structure):
    """Convert a pymatgen Structure to an ASE Atoms object"""
    scaled_positions = pmg_structure.frac_coords
    symbols = [str(site.specie) for site in pmg_structure.sites]
    cell = pmg_structure.lattice.matrix
    return Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)


def evaluate(atoms):
    """Energy, max force and stress of ``atoms`` with its attached calculator."""
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError("non-finite energy")
    forces = atoms.get_forces()
    force_norms = np.linalg.norm(forces, axis=1)
    max_force = float(np.max(force_norms)) if len(forces) else 0.0
    force_sq_mean = float(np.mean(force_norms ** 2)) if len(forces) else 0.0
    try:
        stress = np.asarray(atoms.get_stress(voigt=True)) * EV_A3_TO_GPA
        stress_voigt = [float(x) for x in stress]
        max_stress = float(np.max(np.abs(stress)))
        pressure = float(-np.mean(stress[:3]))
    except Exception:  # noqa: BLE001 -- calculator without stress support
        stress_voigt = max_stress = pressure = None
    return {
        "energy": energy,
        "epa": energy / len(atoms),
        "max_force": max_force,
        "force_sq_mean": force_sq_mean,
        "stress_voigt": stress_voigt,
        "max_stress": max_stress,
        "pressure": pressure,
    }


def run_singlepoints(calc):
    """Evaluate every input structure; one failure never stops the batch."""
    with open("input_structures.json", "r") as f:
        items = json.load(f)

    results = []
    failed = []
    for pos, item in enumerate(items):
        uuid = item.get("uuid")
        index = item.get("index", pos)
        try:
            atoms = pmg_to_ase(Structure.from_dict(item["structure"]))
            atoms.calc = calc
            record = evaluate(atoms)
            record.update({"uuid": uuid, "index": index})
            results.append(record)
        except Exception as e:  # noqa: BLE001 -- log + continue per structure
            failed.append({"uuid": uuid, "index": index,
                           "reason": f"{type(e).__name__}: {e}"})
            print(f"Error evaluating structure {index} (uuid {uuid}): {e}")

    with open("output.json", "w") as f:
        json.dump({"results": results, "failed": failed}, f)

    with open("total.txt", "w") as f:
        f.write(str(len(items)))

    with open("failed.txt", "w") as f:
        f.write(str(len(failed)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    run_singlepoints(calc)
