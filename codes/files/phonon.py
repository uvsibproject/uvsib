"""Phonon / finite-T free-energy runner (compute-node side).

Staged to the remote folder as ``aiida.py`` by ``PhononCalculation`` together
with ``_calculators.py``; an ``input_structures.json`` list is staged alongside.

For each requested structure it does the sub-step the synthesizability stage
needs to go from a 0 K hull to a finite-T one:

    tight relax  ->  phonopy finite displacements  ->  MLIP forces
                 ->  force constants  ->  harmonic F_vib(T)
                 ->  QHA over a few volumes  ->  Gibbs free energy G(T)

and writes, per structure, the relaxed static energy and the per-atom free-energy
correction ``G(T) - E0`` on a temperature grid. The synthesizability workchain
adds that correction to each phase's 0 K energy to build the finite-T hull.

IMPORTANT: phonons are second derivatives of the PES, so use a *conservative*
MLIP (forces = -grad E): MACE or MatterSim. Direct-force models (eqV2/UMA direct
heads) give asymmetric force constants and are not suitable here.

Input schema (``input_structures.json``): a list of ``{"uuid": str,
"structure": <pymatgen Structure.as_dict()>}``.
Output (``output.json``): ``{"results": [ ... per structure ... ], "params": {...}}``.
"""

import json
import argparse

import numpy as np
from ase import Atoms
from ase.optimize import BFGSLineSearch
from ase.filters import FrechetCellFilter
from pymatgen.core import Structure

EV_PER_KJMOL = 1.0 / 96.48533212  # eV per (kJ/mol)
R_J_PER_K_MOL = 8.314462618       # gas constant; phonopy entropy [J/K/mol] / R = kB/(formula cell)


# --------------------------------------------------------------------------- #
# structure <-> ASE / phonopy helpers
# --------------------------------------------------------------------------- #
def to_ase(struct_dict):
    s = Structure.from_dict(struct_dict)
    cell, frac = s.lattice.matrix, s.frac_coords
    if np.linalg.det(cell) < 0:          # left-handed -> swap a,b (phonopy needs det>0)
        cell, frac = cell[[1, 0, 2]], frac[:, [1, 0, 2]]
    return Atoms(symbols=[str(site.specie) for site in s],
                 scaled_positions=frac, cell=cell, pbc=True)


def _phonopy_atoms(atoms):
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(symbols=list(atoms.get_chemical_symbols()),
                        cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())


def _supercell_matrix(atoms, min_length):
    """Diagonal supercell so every cell vector is at least ``min_length`` (capped)."""
    lengths = atoms.cell.lengths()
    n = [int(min(3, max(1, np.ceil(min_length / l)))) for l in lengths]
    return np.diag(n)


# --------------------------------------------------------------------------- #
# relaxation
# --------------------------------------------------------------------------- #
def tight_relax(atoms, calc, fmax, steps):
    """Relax cell + positions to a tight force/stress minimum."""
    atoms = atoms.copy()
    atoms.calc = calc
    opt = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms, bool(opt.converged())


def relax_positions(atoms, calc, fmax, steps):
    """Relax internal positions at fixed cell (for a QHA volume point)."""
    atoms = atoms.copy()
    atoms.calc = calc
    opt = BFGSLineSearch(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms


# --------------------------------------------------------------------------- #
# harmonic free energy at one volume
# --------------------------------------------------------------------------- #
def harmonic_free_energy(atoms, calc, params):
    """Harmonic F_vib(T) and S_vib(T) per atom at the given (fixed) cell.

    Returns ``(temperatures, F_per_atom [eV], S_per_atom [kB], freq_stats,
    n_prim_atoms, static_energy_per_atom)``; ``freq_stats`` holds the
    dynamical-stability summary of the mesh (see ``_frequency_stats``).
    """
    from phonopy import Phonopy

    atoms = atoms.copy()
    atoms.calc = calc
    e_static = float(atoms.get_potential_energy()) / len(atoms)

    smat = _supercell_matrix(atoms, params["min_supercell_length"])
    phonon = Phonopy(_phonopy_atoms(atoms), supercell_matrix=smat,
                     primitive_matrix="auto")
    phonon.generate_displacements(distance=params["displacement"])

    forces = []
    for sc in phonon.supercells_with_displacements:
        a = Atoms(symbols=list(sc.symbols), cell=sc.cell,
                  scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    phonon.forces = forces
    phonon.produce_force_constants()

    mesh = params["mesh"]
    phonon.run_mesh([mesh, mesh, mesh])
    phonon.run_thermal_properties(
        t_min=params["t_min"], t_max=params["t_max"], t_step=params["t_step"])
    tp = phonon.get_thermal_properties_dict()

    n_prim = len(phonon.primitive)
    temps = np.asarray(tp["temperatures"], dtype=float)
    # phonopy free_energy is kJ/mol per primitive cell -> eV/atom
    f_per_atom = np.asarray(tp["free_energy"], dtype=float) * EV_PER_KJMOL / n_prim
    # phonopy entropy is J/K/mol per primitive cell -> kB/atom (S/R/n_prim)
    s_per_atom = np.asarray(tp["entropy"], dtype=float) / (R_J_PER_K_MOL * n_prim)

    stats = _frequency_stats(phonon.get_mesh_dict(), params["imaginary_tol_THz"])
    stats["supercell_matrix"] = [int(n) for n in np.diag(smat)]
    stats["n_atoms_supercell"] = len(phonon.supercell)
    stats["n_displacements"] = len(phonon.supercells_with_displacements)

    return temps, f_per_atom, s_per_atom, stats, n_prim, e_static


def _frequency_stats(mesh_dict, tol):
    """Dynamical-stability summary of a phonopy mesh.

    Imaginary modes show up as negative frequencies. The three acoustic modes
    at Gamma are exactly zero up to numerical noise, so a Gamma q-point (if the
    mesh contains one) is left out; elsewhere a mode below ``-tol`` THz counts.
    """
    qpoints = np.asarray(mesh_dict["qpoints"], dtype=float)
    freqs = np.asarray(mesh_dict["frequencies"], dtype=float)   # (n_q, n_bands)
    off_gamma = np.linalg.norm(qpoints, axis=1) > 1e-8
    if not off_gamma.any():
        off_gamma[:] = True
    f = freqs[off_gamma]
    n_imag = int(np.sum(np.any(f < -tol, axis=1)))
    return {
        "imaginary": bool(n_imag > 0),
        "min_frequency_THz": round(float(np.min(f)), 4),
        "max_frequency_THz": round(float(np.max(freqs)), 4),
        "n_imaginary_qpoints": n_imag,
        "n_qpoints": int(off_gamma.sum()),
    }


# --------------------------------------------------------------------------- #
# QHA: free energy minimised over volume at each T
# --------------------------------------------------------------------------- #
def qha_free_energy(atoms0, calc, params):
    """Gibbs free energy per atom G(T) via QHA over a few volumes.

    Relaxes the input, then for each volume scale relaxes internal positions,
    computes E_static(V) + F_vib(V,T), and minimises over V at each T (P=0 ->
    G ~= min_V F). With a single volume this reduces to the harmonic G = E0 + F_vib.
    """
    relaxed, converged = tight_relax(atoms0, calc, params["fmax"], params["max_steps"])

    scales = params["volume_scales"]
    Vs, E0s, Fmats, Smats, stats_all = [], [], [], [], []
    temps = None
    cell0 = relaxed.cell.array.copy()

    for s in scales:
        a = relaxed.copy()
        a.set_cell(cell0 * (s ** (1.0 / 3.0)), scale_atoms=True)
        if len(scales) > 1:
            a = relax_positions(a, calc, params["fmax"], params["max_steps"])
        temps, f_per_atom, s_per_atom, stats, _, e_static = harmonic_free_energy(a, calc, params)
        Vs.append(a.get_volume() / len(a))   # volume per atom
        E0s.append(e_static)
        Fmats.append(f_per_atom)
        Smats.append(s_per_atom)
        stats_all.append(stats)

    Vs = np.asarray(Vs)
    E0s = np.asarray(E0s)
    Fmats = np.asarray(Fmats)            # shape (n_vol, n_T), per atom
    F_total = E0s[:, None] + Fmats        # Helmholtz F(V,T) per atom

    if len(scales) == 1:
        g_per_atom = F_total[0]
        v_eq = np.full_like(temps, Vs[0])
    else:
        g_per_atom = np.empty_like(temps)
        v_eq = np.empty_like(temps)
        for j in range(len(temps)):
            g_per_atom[j], v_eq[j] = _min_over_volume(Vs, F_total[:, j])

    # equilibrium static energy at T=0 (or V0) for the correction reference
    ref_idx = int(np.argmin(np.abs(Vs - np.median(Vs))))
    e0_ref = float(E0s[ref_idx])
    correction = g_per_atom - e0_ref
    # harmonic vibrational entropy at the (near-equilibrium) reference volume --
    # the quantity Fultz et al. measure as the vibrational entropy of ordering
    s_ref = Smats[ref_idx]
    # stability is judged at the reference (equilibrium) volume; the QHA
    # volumes only get a flag, since compressed/expanded cells may soften
    stats_ref = stats_all[ref_idx]
    return {
        "temperatures": temps.tolist(),
        "E0_eV_per_atom": round(e0_ref, 6),
        "G_qha_eV_per_atom": [round(float(x), 6) for x in g_per_atom],
        "free_energy_correction_eV_per_atom": [round(float(x), 6) for x in correction],
        "entropy_kB_per_atom": [round(float(x), 6) for x in s_ref],
        "volume_per_atom_A3": [round(float(x), 4) for x in v_eq],
        "imaginary_modes": stats_ref["imaginary"],
        "imaginary_modes_any_volume": any(st["imaginary"] for st in stats_all),
        "min_frequency_THz": stats_ref["min_frequency_THz"],
        "max_frequency_THz": stats_ref["max_frequency_THz"],
        "n_imaginary_qpoints": stats_ref["n_imaginary_qpoints"],
        "n_qpoints": stats_ref["n_qpoints"],
        "supercell_matrix": stats_ref["supercell_matrix"],
        "n_atoms_supercell": stats_ref["n_atoms_supercell"],
        "n_displacements": stats_ref["n_displacements"],
        "n_atoms": len(relaxed),
        "relax_converged": converged,
        "n_volumes": len(scales),
    }


def _min_over_volume(V, F):
    """Quadratic-EOS minimum of F(V); clamps to sampled range if the vertex escapes."""
    a, b, c = np.polyfit(V, F, 2)
    if a > 0:
        v_star = -b / (2 * a)
        if V.min() <= v_star <= V.max():
            return float(a * v_star ** 2 + b * v_star + c), float(v_star)
    k = int(np.argmin(F))
    return float(F[k]), float(V[k])


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(structures, calc, params):
    results = []
    n_failed = 0
    for item in structures:
        uuid = item.get("uuid")
        try:
            atoms = to_ase(item["structure"])
            out = qha_free_energy(atoms, calc, params)
            out["uuid"] = uuid
            out["formula"] = Structure.from_dict(item["structure"]).composition.reduced_formula
            results.append(out)
        except Exception as exc:  # a single bad structure must not sink the job
            n_failed += 1
            results.append({"uuid": uuid, "error": f"{type(exc).__name__}: {exc}"})
    return results, n_failed


def _parse_args():
    p = argparse.ArgumentParser(description="MLIP phonon / QHA free-energy runner")
    p.add_argument("--ML_model", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--task_name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fmax", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--min_supercell_length", type=float, default=8.0)
    p.add_argument("--displacement", type=float, default=0.01)
    p.add_argument("--mesh", type=int, default=20)
    p.add_argument("--t_min", type=float, default=0.0)
    p.add_argument("--t_max", type=float, default=1500.0)
    p.add_argument("--t_step", type=float, default=50.0)
    p.add_argument("--volume_scales", type=str, default="0.97,0.99,1.0,1.01,1.03")
    p.add_argument("--imaginary_tol_THz", type=float, default=0.1)
    return p.parse_args()


def main():
    args = _parse_args()
    params = {
        "fmax": args.fmax, "max_steps": args.max_steps,
        "min_supercell_length": args.min_supercell_length,
        "displacement": args.displacement, "mesh": args.mesh,
        "t_min": args.t_min, "t_max": args.t_max, "t_step": args.t_step,
        "volume_scales": [float(x) for x in args.volume_scales.split(",")],
        "imaginary_tol_THz": args.imaginary_tol_THz,
    }
    with open("input_structures.json", "r", encoding="utf-8") as f:
        structures = json.load(f)

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    results, n_failed = run(structures, calc, params)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "params": params}, f)
    with open("total.txt", "w") as f:
        f.write(str(len(structures)))
    with open("failed.txt", "w") as f:
        f.write(str(n_failed))


if __name__ == "__main__":
    main()
