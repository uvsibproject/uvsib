% Post-relaxation sanity checks for ML-relaxed adsorbates
% codes/files/adsorbates.py
% 2026-05-26

# Why

`codes/files/adsorbates.py` generates adsorbate-on-slab structures and relaxes
them with a machine-learning interatomic potential (MACE, uPET, MatterSim,
UMA, ...). ML relaxers can extrapolate into regions they were never trained
on and return *converged* geometries that are physically wrong. Because the
energy is then stored under a label like `*COOH`, a bad relaxation silently
corrupts the CHE free-energy ladder downstream.

Composition is preserved by relaxation, so composition checks are useless.
The failure modes that matter are geometric / topological:

1. **Dissociation / fragmentation** -- `*COOH` loses its H, `*OOH` breaks its
   O-O bond. The recorded energy no longer belongs to the intended species.
   This is the dominant silent corruptor.
2. **Desorption** -- the adsorbate floats off the slab; the "adsorption
   energy" is really `E(gas molecule) + E(slab)`.
3. **Explosion / atom overlap** -- the model drives atoms on top of each other
   or blows the structure apart.
4. **Isomerisation** -- still intact and bound, but rearranged to a different
   connectivity than the label implies (`*OCHO` vs `*COOH`).
5. **Slab reconstruction** -- the slab itself melts or ejects atoms, so the
   clean-slab reference no longer cancels.

# The check, in layers

Validation runs in `validate_relaxed_adsorbate()` after each adsorbate
relaxes. Cheap filters run first and bail early. Layers 0-2 are **on by
default**; layers 3-4 are **opt-in**.

| Layer | Catches | Default | Cost |
|-------|---------|---------|------|
| 0  finite energy + atom overlap | NaN/inf energy, explosion/overlap | on  | trivial |
| 1  surface binding              | desorption                        | on  | cheap   |
| 2  molecular identity (graph)   | dissociation, isomerisation       | on  | cheap   |
| 3  slab integrity               | slab reconstruction               | off | cheap   |
| 4  energy outliers (ensemble)   | subtle ML failures               | off | medium  |

## Layer 0 -- finite energy and overlap

The relaxed energy must be finite (not NaN/inf), and `has_reasonable_distances`
(the same routine used to screen initial placements) must pass on the relaxed
geometry: no pair of light atoms closer than `0.5 * (r_cov_i + r_cov_j)`.

## Layer 1 -- surface binding

At least one adsorbate atom must lie within `bind_tol * (r_cov_i + r_cov_j)` of
at least one slab atom, using minimum-image (PBC-aware) distances. If nothing
is within bonding range, the adsorbate has desorbed and the adsorption energy
is meaningless. Default `bind_tol = 1.25`.

## Layer 2 -- molecular identity

The intended species is known from `ads.properties['adsorbate']`, so we compare
the relaxed adsorbate's intramolecular bond graph to a reference graph for that
species:

- **2a Fragmentation:** the relaxed adsorbate must be a single connected
  component (every adsorbate in the library is one connected molecule). A
  fragment that breaks off creates a second component. This is the robust core
  check and depends only on the relaxed geometry.
- **2b Isomerisation:** the relaxed graph must be isomorphic (element-aware,
  via `networkx.is_isomorphic`) to the reference graph -- catching
  rearrangements that keep the molecule in one piece.

Bonds are covalent-radii based: an edge connects i, j when their separation is
within `graph_tol * (r_cov_i + r_cov_j)`. The relaxed graph uses minimum-image
distances so it is robust to a molecule wrapping across a cell boundary.

### Calibrating `graph_tol`

The reference graphs are built from the (crude, unrelaxed) library geometries
in `adsorbates.py`. The tolerance has to be loose enough that every reference
is a single connected component, but tight enough not to invent spurious 1,3
bonds. Scanning all 38 bundled adsorbates:

- at `1.20`, `*CHOH` is disconnected (its placeholder C-H sits at 1.295 A);
- at `1.25`, **all 38 references are connected** and no spurious bonds appear;
- spurious 1,3 bonds first appear at `>= 1.30` (in `*OCCO`, `*ONNO`).

So the default is **`graph_tol = 1.25`**, on the plateau. Both the reference
and the relaxed adsorbate use the same value (constant
`_ADSORBATE_BOND_TOL`), which is what makes the comparison meaningful.

## Layer 3 -- slab integrity (opt-in)

When `check_slab_integrity=True`, the slab atoms of the relaxed structure are
compared to the clean relaxed slab (already computed in the same loop). If the
maximum minimum-image displacement of any slab atom exceeds `slab_max_disp`
(default 1.5 A), the slab has reconstructed and the structure is rejected.
The threshold is deliberately loose so that normal adsorption-induced
relaxation (~0.3-0.5 A) does not trip it.

## Layer 4 -- energy outliers (opt-in, ensemble)

When `check_energy_outliers=True`, after all sites on a slab are relaxed, the
adsorbate energies are pooled **per species across sites**. A set is rejected
if any of its adsorbates deviates from the species median by more than
`energy_mad_factor` robust standard deviations (`1.4826 * MAD`, default factor
5). This catches subtle ML failures that pass every geometric check but give an
absurd energy. It needs at least 4 sites per species to form a band.

# How it is wired in

- **Identifying the adsorbate atoms.** `generate_adsorbed_structures` appends
  the adsorbate after the slab, so the adsorbate is the last `n_ads` atoms.
  `n_ads` is read from the reference graph's node count. As a safety net, the
  validator checks that the last `n_ads` atoms have the expected element
  multiset; if not, it skips layers 1-3 (returns ok) rather than risk a false
  rejection from a bad index guess.
- **Reference graphs.** `generate_adsorbed_structures` returns
  `expected_graphs` (a dict `{adsorbate_name: networkx.Graph}`) built once per
  run from the reference Molecule geometries (with the `X` binding dummy
  stripped).
- **Hook point.** In `run_relaxation`, immediately after the convergence check
  and energy read, a failing structure is treated exactly like a relaxation
  failure: its set is dropped and the reason recorded. The all-or-nothing set
  logic is unchanged.

## rejected.json

Every run writes `rejected.json` (retrieved alongside `output.json`,
`total.txt`, `failed.txt`) -- a list of records:

```json
[
  {"adsorbate": "*COOH", "site_type": "bridge",
   "ads_coord": [...], "repeat": [1, 1, 1],
   "reason": "dissociation (adsorbate fragmented)"}
]
```

It is an empty list when nothing was rejected. Nothing is ever dropped
silently -- if numbers look thin, read `rejected.json` to see why.

## CLI flags

`adsorbates.py` (the runtime script) accepts:

| Flag | Effect |
|------|--------|
| `--no-validate`            | turn OFF layers 0-2 (default: on) |
| `--check-slab-integrity`   | turn ON layer 3 |
| `--check-energy-outliers`  | turn ON layer 4 |
| `--bind_tol`, `--graph_tol`, `--slab_max_disp`, `--energy_mad_factor` | tolerances (see below) |

## input.yaml

In the AiiDA workflow the checks are controlled from the `adsorbates:` block of
`input.yaml`; `get_cmdline` (`codes/utils.py`) turns the keys into the flags
above. **Validation is OFF unless `validate: true` is set** -- the workflow
passes `--no-validate` otherwise.

```yaml
adsorbates:
  validate: true                # layers 0-2 (default false)
  check_slab_integrity: false   # layer 3
  check_energy_outliers: false  # layer 4
  bind_tol: 1.25                # optional tolerances; omitted -> script defaults
  graph_tol: 1.25
  slab_max_disp: 1.5
  energy_mad_factor: 5.0
```

The MLIP parsers copy `rejected.json` into the calculation's `output_dict`
(key `rejected`), and `AdsorbatesWorkChain` reports the number of rejected
sets per surface with their adsorbates and reasons.

# Tolerances and tuning

| Parameter | Default | Layer | Meaning |
|-----------|---------|-------|---------|
| `bind_tol`          | 1.25 | 1 | bond cutoff factor for adsorbate-slab binding |
| `graph_tol`         | 1.25 | 2 | bond cutoff factor for the identity graph |
| `slab_max_disp`     | 1.5  | 3 | max slab-atom displacement (A) before "reconstructed" |
| `energy_mad_factor` | 5.0  | 4 | robust-sigma band half-width for outliers |

Start lenient: the goal is to catch *garbage*, not to enforce tight geometry.
Tighten only if junk slips through.

# Caveats

- **Metal bonding** is poorly described by covalent radii. That is why layer 2
  graphs only the adsorbate's *internal* (C/H/N/O/Cl) bonds, and layer 1 treats
  surface binding as a looser distance test rather than a graph edge.
- **Bidentate / multi-atom binding** is fine: the internal graph (e.g. the two
  C-O and one C-H bonds of `*OCHO`) is unchanged whether the molecule binds
  through one atom or two.
- **Crude reference geometries.** A few polyatomic coupling intermediates
  (`*OCCO`, `*ONNO`, `*CCHO`) have placeholder geometries with borderline
  bonds. If layer 2b false-rejects one of these, it is logged in
  `rejected.json` with a clear reason and can be tuned via `graph_tol` -- the
  fragmentation check (2a) is unaffected and still catches dissociation.
- **Broken reference geometries.** If a reference adsorbate is not one
  connected molecule at `graph_tol` (a bond in the library geometry is longer
  than the cutoff), layer 2 is skipped for that adsorbate, a warning is printed
  and the name is listed under `identity_check_skipped` in the job summary
  (shown in the pipeline report). Otherwise every intact adsorbate would be
  rejected. This happened for the OER `*OH` (O-H 1.28 A, fixed to 0.97 A on
  2026-10-23).
- **Intended water release etc.** happens *across* pathway steps as separate
  species, never within a single adsorbate relaxation, so "intact" is the
  correct per-structure criterion.
- **Gas-phase references** are not validated here (out of scope); a future
  extension could apply the same identity check to the relaxed
  molecular_references to guard against a reference molecule dissociating.

# Extending

To add a check, write it as another early-return branch in
`validate_relaxed_adsorbate` (per-structure) or as a post-pass like
`_flag_energy_outliers` (ensemble), gate it behind a `run_relaxation` flag if
it is not cheap-and-safe enough to be on by default, and make sure failures
land in `rejected.json` with a descriptive reason.
