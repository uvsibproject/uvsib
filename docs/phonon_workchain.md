# PhononScreenWorkChain

MLIP phonons and dynamical stability for the **predicted** bulks of a composition.

## Where it runs

`MainWorkChain` step `step_status["phonon"]`, right after the phase-diagram stages
(`pd_ml`, `pd_verification`) and before the optical screen. It is skipped when

- `phonon.enabled` is false (it is on by default),
- the run is an SQS run,
- the step is already `Done` for the composition, or
- there is no ML bulk selection.

A sibling `MainWorkChain` that finds the step `Running` waits, as for the other
composition-level steps. The step is **mandatory** when enabled: if any phonon
calculation fails (a whole `PhononWorkChain` job, a single structure inside it,
or a missing code), the step and the composition are marked `Failed` and the
`MainWorkChain` stops (exit 300). Results that did succeed are still stored, and
with `skip_existing` a rerun only redoes the failed structures. Structures are
not filtered on the result.

## What it computes

1. Reads `DBComposition.stable_struct["ml_selection"]` (the bulks that passed the
   E_above_hull screen) and keeps only structures whose `DBStructureVersion.source`
   is `csp` or `generated` (`PREDICTED_SOURCES`). Materials Project structures
   (`MPDB_stb`, `MPDB_exp`, `MPDB_ref`) are skipped. With `skip_existing` (default),
   bulks that already have an `ok` `DBPhonon` row for the model are skipped too.
2. Submits one `PhononWorkChain` (`codes/phonon/`) per `chunk_size` structures, in
   parallel. The runner `codes/files/phonon.py`:
   - tight-relaxes cell and positions (`fmax`),
   - builds a diagonal supercell with every vector at least `min_supercell_length`
     (at most 3× per axis) and runs phonopy finite displacements with the MLIP,
   - computes the phonon mesh, thermal properties and, with
     `harmonic_only: false`, QHA over `volume_scales`.
3. Writes one `DBPhonon` row per bulk.

A structure is **dynamically stable** when no mode at the reference volume is
below `-imaginary_tol_THz` at any q-point other than Γ (the acoustic modes at Γ
are zero up to numerical noise).

Phonons are second derivatives of the energy, so the model must be
**conservative** (forces = −∇E): e.g. SevenNet, MACE or MatterSim. Do not use
direct-force heads.

## Configuration

`input.yaml` (everything except `enabled` is optional):

```yaml
phonon:
  enabled: true                # default true
  model: SevenNet              # default: bulk_relax.model
  head: omat24                 # default: bulk_relax.head (if model == bulk_relax.model)
  include_above_threshold: true
  skip_existing: true
  harmonic_only: true          # false -> QHA over volume_scales (~n_volumes x cost)
  volume_scales: [0.97, 0.99, 1.0, 1.01, 1.03]
  fmax: 0.001
  max_steps: 500
  min_supercell_length: 8.0
  displacement: 0.01
  mesh: 20
  t_min: 0.0
  t_max: 1500.0
  t_step: 50.0
  imaginary_tol_THz: 0.1
  chunk_size: 10
```

`config.yaml`: an optional dedicated `Phonon` code block (phonons need a longer
walltime than relaxations). Without it, the code and job script of `phonon.model`
are used.

```yaml
codes:
  Phonon:
    code_string: sevennet@rosi_gpuv100   # an env with phonopy + the MLIP
    job_script:
      device: cuda
      nodes: 1
      ntasks: 1
      cpus: 12
      time: 86400
      exclusive: False
```

The code's environment needs `phonopy` in addition to the MLIP package.

## Storage: `db_phonon`

One row per `(structure_uuid, phonon_model)`; a rerun overwrites it. Create the
table once with `run_dir/create_tables.py`.

| Column | Meaning |
|---|---|
| `status` | `ok` / `failed` |
| `source` | `csp` / `generated` |
| `dynamically_stable` | no imaginary modes at the reference volume |
| `min_frequency_THz` | lowest mode off Γ (negative = imaginary) |
| `n_imaginary_qpoints` | q-points with at least one imaginary mode |
| `e0` | static energy after the tight relax (eV/atom) |
| `free_energy_300K` | G(300 K) − E0 (eV/atom) |
| `ehull` | snapshot from the ML selection |
| `attributes` | temperature grid, G(T), F_vib(T), S(T), V(T), supercell, run parameters, error |

## Report

`pipeline_report.phonon_for_bulk()` feeds the "Lattice dynamics" section and the
"Phonons" entry in the pipeline metadata. Materials Project bulks are listed as
not phononed.
