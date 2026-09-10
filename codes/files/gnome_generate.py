"""GNoME-style generative runner (compute-node side).

Shipped to the remote folder as ``aiida.py`` by ``GNoMECalculation`` together
with ``_saps.py`` (the symmetry-aware partial-substitution generator),
``refine.py`` (charge-neutral + primitive-cell dedup helpers, reused from the
MatterGen path) and ``_calculators.py`` (the ASE calculator factory, reused for
the GNN screen). A ``templates.json`` pool of seed crystals is staged alongside.

Pipeline (mirrors GNoME's discovery funnel — generate, then GNN-screen):

  1. SAPS generation from the template pool toward the requested target
     (a chemical system for ``gen``; an explicit formula for ``csp``).
  2. charge-neutrality filter + primitive-cell de-duplication.
  3. optional cheap energy screen with an MLIP single point (``--screen``),
     keeping the lowest energy-per-atom candidates per composition. This stands
     in for GNoME's formation-energy GNN; point ``--screen`` at the real GNoME
     checkpoint once a ``make_calculator`` branch is added for it.

The output contract is identical to MatterGen's: ``output.json`` is a list of
pymatgen structure dicts, consumed unchanged by ``GNoMEParser``.
"""

import json
import argparse
from collections import defaultdict

from pymatgen.core import Structure

from _saps import saps_generate
from refine import select_charge_neutral, refine_and_filter_structures


def screen_by_energy(struct_dicts, args):
    """Keep the lowest energy-per-atom candidates per composition via an MLIP.

    Returns the input unchanged when ``--screen none`` (lets the generation step
    be exercised on a CPU node without a model).
    """
    if not struct_dicts or args.screen.lower() == "none":
        return struct_dicts

    from ase import Atoms
    from _calculators import make_calculator

    calc = make_calculator(args.screen, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    scored = []
    for sd in struct_dicts:
        s = Structure.from_dict(sd)
        atoms = Atoms(symbols=[str(site.specie) for site in s],
                      scaled_positions=s.frac_coords,
                      cell=s.lattice.matrix, pbc=True)
        atoms.calc = calc
        try:
            epa = float(atoms.get_potential_energy()) / len(s)
        except Exception:
            continue
        scored.append((epa, sd, s.composition.reduced_formula))

    by_comp = defaultdict(list)
    for epa, sd, rf in scored:
        by_comp[rf].append((epa, sd))

    per_comp = max(1, args.keep // max(1, len(by_comp)))
    kept = []
    for rf, lst in by_comp.items():
        lst.sort(key=lambda x: x[0])
        kept.extend(sd for _, sd in lst[:per_comp])
    return kept[:args.keep]


def main():
    parser = argparse.ArgumentParser(description="GNoME-style SAPS generator")
    parser.add_argument("--mode", required=True, choices=["gen", "csp"])
    parser.add_argument("--target", required=True,
                        help="chemical system 'Y-Ru-O' (gen) or formula 'Y2Ru2O7' (csp)")
    parser.add_argument("--templates", default="templates.json")
    parser.add_argument("--n_max", type=int, default=200)
    parser.add_argument("--max_per_template", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--partial", type=int, default=2)
    parser.add_argument("--keep", type=int, default=60)
    # GNN screen (reuses the relaxer calculator factory)
    parser.add_argument("--screen", default="none",
                        help="MLIP tag for the energy screen (MACE/uPET/UMA/MatterSim/SevenNet/GRACE) or 'none'")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.templates, "r", encoding="utf-8") as f:
        templates = json.load(f)

    structures = saps_generate(
        templates, args.target, args.mode,
        n_max=args.n_max, max_per_template=args.max_per_template,
        threshold=args.threshold, partial=args.partial,
    )

    struct_dicts = [s.as_dict() for s in structures]
    struct_dicts = select_charge_neutral(struct_dicts)
    struct_dicts = refine_and_filter_structures(struct_dicts)
    struct_dicts = screen_by_energy(struct_dicts, args)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(struct_dicts, f)


if __name__ == "__main__":
    main()
