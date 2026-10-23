import os, json
import argparse
import signal
import time
import traceback
from ase import Atoms
from ase.io import read, write
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from minimahopping.minhop import Minimahopping


# ---------------------------------------------------------------------------
# Caps and metal-safe defaults (kept here in the runner; the MinimaHopping
# source tree is left untouched).
#
# Why these exist: on close-packed metals the MH escape loop (minhop._escape)
# is the only unbounded loop in the algorithm.  Its exit test relies on the
# OMFP fingerprint distinguishing near-degenerate minima, which it often cannot
# for metallic environments -> the loop keeps re-running a full MD + geometry
# optimization, each of which can run to md_max_steps / opt_max_steps (10000 by
# default).  A handful of failed escapes then blows a >12 h walltime and SLURM
# kills the job before any result is written.
#
# Three independent guards are applied, all from the runner:
#   1. hard wall-clock watchdog (SIGALRM) -> stop cleanly and salvage the
#      accepted minima written so far, so the job never gets SIGTERM-killed;
#   2. low per-escape cost: small md_max_steps / opt_max_steps;
#   3. metal-appropriate fingerprint so escapes actually succeed
#      (lower fingerprint_threshold, add p-orbitals, wider width_cutoff).
#
# fmax and the wall-clock budget come from the workflow (--fmax from
# input.yaml MinimaHopping.fmax, --max_seconds from config.yaml
# codes.MinimaHopping.job_script.time minus 30 min). The remaining knobs are
# overridable via an environment variable so the values can be tuned per
# element/walltime without editing code (set them e.g. through the AiiDA
# code's `prepend_text` / computer environment_variables).
# ---------------------------------------------------------------------------

def _envf(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default

def _envi(name, default):
    try:
        return int(float(os.environ[name]))
    except (KeyError, ValueError):
        return default


# Total wall-clock the runner may use.  MUST be < the SLURM walltime of the job
# so the salvage step can write output.json before the scheduler SIGTERMs us.
# Normally set by --max_seconds (SLURM walltime - 30 min); this value is only
# the fallback for a manual run without --max_seconds.
MH_MAX_SECONDS = _envi("MH_MAX_SECONDS", 10000)
# Time reserved at the end for stopping MH and writing output.json.
MH_SALVAGE_MARGIN = _envi("MH_SALVAGE_MARGIN", 180)


def mh_parameters(fmax, max_seconds):
    """MinimaHopping kwargs tuned to terminate on metals and to bound cost."""
    return dict(
        # --- cost caps per escape iteration (default 10000 each) -------------
        md_max_steps=_envi("MH_MD_MAX_STEPS", 200),
        opt_max_steps=_envi("MH_OPT_MAX_STEPS", 200),
        mdmin=_envi("MH_MDMIN", 2),          # stop MD after 2 minima (default 2; was 5)
        fmax=fmax,
        dt0=_envf("MH_DT0", 0.08),
        # --- make the escape test resolve metallic minima -------------------
        # energies of distinct metallic minima are tiny: keep the energy gate
        # tight so the fingerprint actually gets used, and lower the fingerprint
        # threshold so close-packed minima count as "different".
        energy_threshold=_envf("MH_ENERGY_THRESHOLD", 0.002),
        fingerprint_threshold=_envf("MH_FP_THRESHOLD", 0.02),
        n_S_orbitals=1,
        n_P_orbitals=_envi("MH_N_P_ORBITALS", 1),   # richer OMFP than s-only (default 0)
        width_cutoff=_envf("MH_WIDTH_CUTOFF", 6.0),  # more neighbour context (default 4.0)
        T0=_envf("MH_T0", 1000.0),
        symprec=0.05,
        # --- in-loop budget (checked between hops); SIGALRM is the backstop --
        run_time=_format_runtime(max(max_seconds - MH_SALVAGE_MARGIN - 60, 60)),
        # --- keep the run lean and self-contained ---------------------------
        use_MPI=False,
        new_start=True,
        verbose_output=False,
        collect_md_data=False,
        write_graph_output=False,
    )


def _format_runtime(seconds):
    """Seconds -> MinimaHopping run_time string 'd-hh:mm:ss'."""
    seconds = int(max(seconds, 0))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return "{:d}-{:02d}:{:02d}:{:02d}".format(days, hours, minutes, secs)


class _WalltimeReached(Exception):
    """Raised by the SIGALRM watchdog when the wall-clock budget is hit."""


def _alarm_handler(signum, frame):
    raise _WalltimeReached()


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


def run_mh(calc, mh_steps, fmax, max_seconds):
    """Run MinimaHopping under a hard wall-clock watchdog.

    The accepted minima are appended to output/accepted_minima.extxyz by
    MinimaHopping itself (and flushed per acceptance), so whatever has been
    found is on disk even if we stop early.  This function therefore never
    lets the job die by SLURM SIGTERM: when the budget is reached (or any
    error occurs) it stops MH and returns, and the caller salvages the file.
    """
    with open("initial_structure.json", "r") as f:
        struct = json.load(f)
    pmg_structure = Structure.from_dict(struct)
    init_conf = pmg_to_ase(pmg_structure)
    init_conf.calc = calc

    params = mh_parameters(fmax, max_seconds)
    budget = max(max_seconds - MH_SALVAGE_MARGIN, 1)
    print("MH runner: wall-clock budget {:d}s (alarm), params: {}".format(
        budget, {k: params[k] for k in
                 ("md_max_steps", "opt_max_steps", "mdmin", "fmax",
                  "energy_threshold", "fingerprint_threshold",
                  "n_P_orbitals", "width_cutoff", "run_time")}),
        flush=True)

    # NOTE: do not pre-create 'output/' here -- MinimaHopping creates output/,
    # output/restart/ and minima/ together, and skips restart/ if output/
    # already exists (which would crash on the first params.json write).
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(budget)
    t0 = time.time()
    try:
        with Minimahopping(init_conf, **params) as mh:
            try:
                mh(totalsteps=int(mh_steps))
            except _WalltimeReached:
                print("MH runner: wall-clock budget reached after {:.0f}s; "
                      "stopping and salvaging accepted minima.".format(time.time() - t0),
                      flush=True)
    except _WalltimeReached:
        # alarm fired during context enter/exit
        print("MH runner: wall-clock budget reached during setup/teardown.", flush=True)
    except Exception:
        # any MH failure: still fall through to salvage so the AiiDA job
        # completes with whatever was found instead of hard-failing.
        print("MH runner: stopped early on exception:", flush=True)
        traceback.print_exc()
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--mh_steps", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float, default=_envf("MH_FMAX", 0.05))
    parser.add_argument("--max_seconds", type=int, default=MH_MAX_SECONDS)
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(ml_model=args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    run_mh(calc, args.mh_steps, args.fmax, args.max_seconds)

    # Salvage: read whatever MinimaHopping accepted (flushed incrementally).
    # accepted_minima[0] is the initial relaxed structure -> skip with [1:].
    try:
        accepted_minima = read('output/accepted_minima.extxyz', index=':')
    except Exception:
        accepted_minima = []

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=7, scale=True,
                               attempt_supercell=False, allow_subset=False, primitive_cell=True)

    existing_structs = []
    structures = []
    energies = []
    for atoms in accepted_minima[1:]:
        pmg_struct = ase_to_pmg(atoms)
        energy = float(atoms.get_potential_energy())

        sga = SpacegroupAnalyzer(pmg_struct, symprec=0.05, angle_tolerance=5)

        try:
            prim_struct = sga.get_primitive_standard_structure()
        except:
            prim_struct = sga.find_primitive() or pmg_struct

        if any(matcher.fit(prim_struct, existing) for existing in existing_structs):
            continue

        existing_structs.append(prim_struct)
        structures.append(prim_struct.as_dict())
        energies.append(energy * (prim_struct.num_sites/pmg_struct.num_sites))

    to_dump = {'structures': structures, 'energies': energies}

    with open('output.json', 'w') as f:
        json.dump(to_dump, f)

    print("MH runner: wrote output.json with {:d} unique accepted minima.".format(len(structures)),
          flush=True)
