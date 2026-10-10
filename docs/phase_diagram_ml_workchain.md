# PhaseDiagramMLWorkChain maintainer notes

This document explains how `PhaseDiagramMLWorkChain` works and what future
coders should check before changing it. The implementation lives in
`workchains/phase_diagram.py` and is registered as the AiiDA workflow entry
point `phasediagram`.

The short version: this work chain collects or creates bulk structures for a
target formula, makes sure the relevant chemical systems have ML-relaxed
entries, builds an ML phase diagram, and stores the UUIDs of the selected
low-energy structures in `DBComposition.stable_struct["ml_uuid_list"]`.

## Source map

| Code | Role |
|---|---|
| `workchains/phase_diagram.py` | Orchestrates MPDB relaxation, CSP, generation, data wait, and final stable-structure storage. |
| `workchains/mpdb_ml.py` | Adds MPDB structures and relaxes MPDB/reference structures with the selected ML model. |
| `workchains/csp.py` | Runs CSP for the exact target composition, relaxes candidates, runs minima hopping, and stores source `csp` structures. |
| `workchains/gen.py` | Generates missing chemical systems, relaxes candidates, filters them, and marks `DBChemsys.gen_structures = "Ready"`. |
| `workchains/utils.py` | Shared hull filtering, reference entry lookup, MPDB import, structure matching, code/model lookup. |
| `workchains/pythonjob_inputs.py` | Small polling functions used by `PythonJob`, especially `is_data_available`. |
| `workflows/settings.py` | Reads `input.yaml` / `config.yaml` and exposes global toggles and thresholds. |

## Inputs

`PhaseDiagramMLWorkChain` has three inputs:

| Input | AiiDA type | Meaning |
|---|---|---|
| `chemical_formula` | `Str` | Exact target formula, for example `Y2Ru2O7`. The final stored structures must have this reduced formula. |
| `chemical_systems` | `List` | Chemical systems that are not ready yet and need generated structures, for example `["O-Ru", "O-Y", "O-Ru-Y"]`. |
| `ml_bulk_model` | `Str` | ML relaxation backend used as the energy method for the phase diagram. It must match a registered workflow entry point such as `mace`, `mattersim`, `upet`, or `uma` after lowercasing. |

`MainWorkChain._construct_pd_ml_builder()` passes these values into
`PhaseDiagramMLWorkChain`. `ml_bulk_model` normally comes from
`settings.inputs["bulk_relax"]["model"]`.

## Outline

The work chain outline is:

```text
setup
if should_run_mpdb_ml:
    mpdb_ml
    inspect_mpdb_ml
if should_run_csp:
    csp_calcs
    inspect_csp_cals
prepare_gen
if should_run_gen:
    gen_calcs
    inspect_gen_calcs
wait_for_data
check_pythonjob
store_stable_structs
final_report
```

The no-DFT light-harvesting screen (`OpticalScreenWorkChain`) is **not** part of
this work chain: it is a mandatory `MainWorkChain` stage (`optical_screen`) that
runs on this work chain's ML bulk selection right before the surface builder.
See `docs/optical_screen_workchain.md`.

The child workflows are submitted through `WorkflowFactory`:

| Called workflow | Entry point | Builder method |
|---|---|---|
| `MPDBMLWorkChain` | `mpdbml` | `_construct_mpdb_ml_builder()` |
| `CSPWorkChain` | `csp` | `_construct_csp_builder()` |
| `GeneratorWorkChain` | `gen` | `_construct_gen_builder()` |
| `aiida_pythonjob.PythonJob` | direct class | `wait_for_data()` |

## Stage behavior

### Setup

`setup()` copies the three inputs into `self.ctx` and reports whether any new
chemical systems were provided. No database writes happen here.

### MPDB ML relaxation

`should_run_mpdb_ml()` calls `_has_mpdb_ml_structures()`.

The MPDB branch is skipped only when the database already has at least one
structure for the target composition with:

```text
method = ml_bulk_model
source in {"MPDB_stb", "MPDB_exp"}
```

If not skipped, `mpdb_ml()` submits `MPDBMLWorkChain` with:

| Builder field | Value |
|---|---|
| `chemical_formula` | current target formula |
| `ml_bulk_model` | selected bulk ML model |

`MPDBMLWorkChain` then:

1. Calls `add_from_mpdb(chemical_formula)`.
2. Finds DFT `MPDB_stb` / `MPDB_exp` structures for the target formula.
3. Finds elemental `MPDB_ref` structures that do not yet have this ML method.
4. Relaxes all collected structures using `WorkflowFactory(ml_bulk_model.lower())`.
5. Stores new `DBStructureVersion` rows with `method = ml_bulk_model`.

Important detail: elemental reference structures are included because final
phase-diagram energies should use ML-relaxed elemental endpoints where
possible. If ML reference rows are missing and cannot be created, the MPDB
child workflow exits with `ERROR_MISSING_REFERENCE_STRUCTURES`.

### CSP

`should_run_csp()` skips CSP when either:

| Condition | Effect |
|---|---|
| `settings._SKIP_CSP` is true | No CSP child workflow is launched. |
| `query_structure({"composition": chemical_formula}, method=ml_bulk_model, source="csp")` returns rows | Existing CSP results for this formula/model are reused. |

If not skipped, `csp_calcs()` submits `CSPWorkChain` with:

| Builder field | Value |
|---|---|
| `chemical_formula` | current target formula |
| `n_csp` | `settings.inputs["MatterGen_CSP"]["num_runs"]` |
| `n_mh` | `settings.inputs["MinimaHopping"]["num_runs"]` |
| `ml_bulk_model` | selected bulk ML model |

`CSPWorkChain` can call these lower-level workflows:

| Lower-level workflow | Entry point | When used |
|---|---|---|
| MatterGen CSP | `mattergen.csp` | When `settings.MATTERGEN_CSP_ENABLED` is true and the formula has 20 atoms or fewer. |
| GNoME SAPS CSP | `gnome.csp` | When `settings.GNOME_PARALLEL` is true. |
| DiffCSP CSP | `diffcsp.csp` | When `settings.DIFFCSP_ENABLED` is true (default). See [diffcsp_generation.md](diffcsp_generation.md). |
| ML relax backend | `ml_bulk_model.lower()` | Always after CSP structures are collected. |
| Minima hopping | `minimahopping` | Runs from sampled low-energy CSP candidates. |

The CSP child stores final structures with source `csp` and method
`ml_bulk_model`. The parent only checks whether the child finished OK.

If `CSPWorkChain` fails, `inspect_csp_cals()` removes `DBChemsys` rows for the
requested `chemical_systems` whose `gen_structures` field is still empty. This
cleanup allows later retries to recreate those rows.

### Generator

`prepare_gen()` runs before the generator predicate. When `_SKIP_GEN` is true,
it marks each requested `DBChemsys` row as `gen_structures = "Ready"`. If any
requested row is missing, it returns `ERROR_NO_CHEMSYS_FOUND`.

`should_run_gen()` is now a boolean-only predicate. It skips generation when:

| Condition | Effect |
|---|---|
| `settings._SKIP_GEN` is true | `prepare_gen()` already marked requested rows ready, so no generator child is launched. |
| `chemical_systems` is empty | Nothing is generated. |

If not skipped, `gen_calcs()` submits `GeneratorWorkChain` with:

| Builder field | Value |
|---|---|
| `chemical_formula` | current target formula |
| `chemical_systems` | the input list of missing systems |
| `ml_bulk_model` | selected bulk ML model |

`GeneratorWorkChain` can call:

| Lower-level workflow | Entry point | When used |
|---|---|---|
| MatterGen generation | `mattergen.base` | When `settings.MATTERGEN_GEN_ENABLED` is true (default). |
| GNoME SAPS generation | `gnome.base` | When `settings.GNOME_PARALLEL` is true. |
| ML relax backend | `ml_bulk_model.lower()` | Per chemical system after generated structures are collected. |

For each chemical system, generated structures are filtered by
`unique_low_energy_chemsys(...)`, stored with source `generated`, and the
matching `DBChemsys` row is marked `Ready`.

`GeneratorWorkChain.store_ml_energies()` now checks that the matching
`DBChemsys` row exists before updating it. If the row is missing, the child
returns `ERROR_NO_CHEMSYS_FOUND`.

As in the CSP failure path, a failed generator child causes cleanup of
`DBChemsys` rows whose `gen_structures` field is still empty.

### Data wait

`wait_for_data()` submits a `PythonJob` running
`is_data_available(chemical_systems=all_chemical_systems)`, where
`all_chemical_systems` comes from `get_chemical_systems(chemical_formula)`.

`is_data_available()` checks that every relevant `DBChemsys` row has:

```text
gen_structures = "Ready"
```

It polls every 60 seconds and times out after 36,000 seconds, or 10 hours.

Current behavior to remember: `_SKIP_GEN` bypasses the readiness wait because
`prepare_gen()` marks the requested missing chemical systems ready. `_SKIP_CSP`
does not bypass this wait. `check_pythonjob()` simply returns when no `pyjob`
was submitted.

### Final stable-structure storage

`store_stable_structs()` is the main output stage.

It:

1. Calls `get_entries_from_db(chemical_formula, ml_bulk_model)`.
2. Reads all `DBStructureVersion` rows whose `chemsys` is in the chemical
   space returned by `get_chemical_systems(chemical_formula)` and whose
   `method` equals `ml_bulk_model`.
3. Ignores rows with source `MPDB_ref`; reference entries are added separately.
4. Calls `get_ref_entries(chemical_formula, ml_bulk_model)`.
5. Calls `unique_low_energy_comp(chemical_formula, entries, EHULL_ML,
   min_n_return=1, element_entries=ref_entries)`.
6. Builds a metadata record for each selected UUID, including its
   energy-above-hull and whether it was selected above the threshold.
7. Merges the selected UUIDs and metadata into `DBComposition.stable_struct`.

`unique_low_energy_comp()` builds a pymatgen `PhaseDiagram`, considers entries
whose reduced formula equals `chemical_formula`, removes obviously invalid
structures with lattice-vector components larger than 100 Angstrom, reduces
structures to primitive cells where possible, removes duplicates with the
shared `StructureMatcher`, and keeps structures with energy above hull less
than or equal to `EHULL_ML`.

Because `min_n_return=1` is passed, the function will return at least one
candidate if any candidate exists, even if that candidate is above `EHULL_ML`.
If this fallback happens, the work chain reports a warning with the selected
UUID, `ehull`, and threshold. If no candidate exists at all, the parent exits
with `ERROR_NO_STRUCTURES_FOUND`.

The final `stable_struct` update preserves existing keys and adds or replaces:

```python
{
    "ml_uuid_list": [...],
    "ml_selection": [
        {
            "uuid": "...",
            "ehull": 0.0,
            "selected_above_threshold": False,
        },
    ],
    "ml_ehull_threshold": EHULL_ML,
    "ml_bulk_model": ml_bulk_model,
}
```

Before writing this field, the work chain now checks that the target
`DBComposition` row exists. If it does not, it returns
`ERROR_NO_COMPOSITION_FOUND`.

## Important parameters

These settings most directly affect the scientific result:

| Parameter | Used in | Meaning |
|---|---|---|
| `settings.inputs["bulk_relax"]["model"]` | `MainWorkChain`, passed as `ml_bulk_model` | Main ML potential used for phase-diagram energies. |
| `settings.inputs["bulk_relax"]["head"]` | MPDB, CSP, Generator ML relax builders | Model head/task for relaxation. |
| `settings.inputs["bulk_relax"]["fmax"]` | MPDB, CSP, Generator ML relax builders | Force convergence threshold. |
| `settings.inputs["bulk_relax"]["max_steps"]` | MPDB, CSP, Generator ML relax builders | Relaxation step limit. |
| `settings.EHULL_ML` | CSP, Generator, final storage | Energy-above-hull threshold for keeping low-energy structures. |
| `settings.EHULL_SCAN` | `add_from_mpdb()` | DFT MPDB hull window used when importing MPDB structures. |
| `settings.DFT_FUNC` | reference fallback utilities | Chooses bundled DFT elemental references when ML references are unavailable. |

Generation and CSP volume/cost parameters:

| Parameter | Used in | Meaning |
|---|---|---|
| `MatterGen_CSP.num_runs` | parent builder for `CSPWorkChain` | Number of independent MatterGen CSP jobs. |
| `MatterGen_CSP.batch_size` | `mattergen.csp` job info | Number of CSP samples per batch. |
| `MatterGen_CSP.num_batches` | `mattergen.csp` job info | Number of CSP batches. |
| `MinimaHopping.num_runs` | parent builder for `CSPWorkChain` | Maximum number of low-energy CSP seeds sent to minima hopping. |
| `MinimaHopping.model` | minima hopping builder | ML model used inside minima hopping. It can differ from `ml_bulk_model`. |
| `MinimaHopping.head` | minima hopping builder | Model head for minima hopping. |
| `MinimaHopping.mh_steps` | minima hopping builder | Length of the minima hopping search. |
| `MatterGen_generate.energy_above_hull` | `mattergen.base` job info | MatterGen conditioning target for generated structures. |
| `MatterGen_generate.batch_size` | `mattergen.base` job info | Number of generated structures per batch. |
| `MatterGen_generate.num_batches` | `mattergen.base` job info | Number of generation batches. |
| `mattergen.csp_enabled` | CSP | Enables the `mattergen.csp` branch through `settings.MATTERGEN_CSP_ENABLED` (default `False`). |
| `mattergen.gen_enabled` | Generator | Enables the `mattergen.base` branch through `settings.MATTERGEN_GEN_ENABLED` (default `True`). |
| `diffcsp.enabled` | CSP only | Enables the `diffcsp.csp` branch through `settings.DIFFCSP_ENABLED` (default `True`). Not wired into Generator. |
| `gnome.enabled` | CSP and Generator | Enables GNoME/SAPS branches through `settings.GNOME_PARALLEL`. |
| `GNoME_CSP.*` | `gnome.csp` job info | Candidate generation and optional screening settings for CSP. |
| `GNoME_generate.*` | `gnome.base` job info | Candidate generation and optional screening settings for missing chemical systems. |
| `DiffCSP_CSP.*` | `diffcsp.csp` job info | Candidate count, batch size, and step size for DiffCSP CSP sampling; see [diffcsp_generation.md](diffcsp_generation.md). |

Infrastructure parameters:

| Parameter | Used in | Meaning |
|---|---|---|
| `settings.configs["codes"][model_key]["code_string"]` | `get_code()` | AiiDA code selected for each backend. |
| `settings.configs["models"]["path_to_pretrained_models"]` | `get_model_device()` | Base directory for ML model files. |
| `settings.configs["models"][model_key]` | `get_model_device()` | Model/checkpoint name. |
| `settings.configs["codes"][model_key]["job_script"]["device"]` | `get_model_device()` | Device string passed to the backend. |

## Database side effects

`PhaseDiagramMLWorkChain` itself writes only the final stable UUID list and
some skip-mode readiness fields. Most structure writes happen inside child
workflows.

| Table | Field / row type | Writer |
|---|---|---|
| `DBStructureVersion` | source `MPDB_stb`, `MPDB_exp`, `MPDB_ref`, method `DFT` | `add_from_mpdb()` inside `MPDBMLWorkChain.setup()` |
| `DBStructureVersion` | source `MPDB_stb`, `MPDB_exp`, `MPDB_ref`, method `ml_bulk_model` | `MPDBMLWorkChain.store_ml_energies()` |
| `DBStructureVersion` | source `csp`, method `ml_bulk_model` | `CSPWorkChain.final_step()` |
| `DBStructureVersion` | source `generated`, method `ml_bulk_model` | `GeneratorWorkChain.store_ml_energies()` |
| `DBChemsys` | `gen_structures = "Ready"` | `GeneratorWorkChain.store_ml_energies()` or parent skip-gen path |
| `DBComposition` | `stable_struct["ml_uuid_list"]`, `stable_struct["ml_selection"]`, `stable_struct["ml_ehull_threshold"]`, `stable_struct["ml_bulk_model"]` | `PhaseDiagramMLWorkChain.store_stable_structs()` |

The parent checks that the `DBComposition` row for `chemical_formula` exists
before updating `stable_struct`. Missing rows return `ERROR_NO_COMPOSITION_FOUND`.
`GeneratorWorkChain` performs the same explicit check before marking a
`DBChemsys` row ready.

## Exit codes

Parent exit codes:

| Code | Name | Typical cause |
|---|---|---|
| 300 | `ERROR_CALCULATION_FAILED` | Child workflow failed, data wait failed, or phase-diagram entries could not be loaded. |
| 301 | `ERROR_NO_STRUCTURES_FOUND` | No candidate for the target formula could be selected in final storage. |
| 302 | `ERROR_NO_CHEMSYS_FOUND` | `_SKIP_GEN` path expected a `DBChemsys` row that does not exist. |
| 303 | `ERROR_NO_COMPOSITION_FOUND` | Final storage could not find the target `DBComposition` row. |

Child workflow failures are collapsed to parent `ERROR_CALCULATION_FAILED`.
The parent has more specific exits for `_SKIP_GEN` missing rows, missing final
structures, and missing `DBComposition` rows during final storage.

## Practical change notes

- Treat `ml_bulk_model` as the method contract for the whole ML phase diagram.
  MPDB-derived structures, CSP structures, generated structures, and elemental
  references should use the same method whenever possible. Mixed-method
  references can shift hull energies.
- `get_entries_from_db()` intentionally loads all rows in the target chemical
  space for `method == ml_bulk_model`, except `MPDB_ref`. Final filtering then
  selects candidates with the exact requested reduced formula. Changing that
  query changes which entries support the convex hull.
- `get_ref_entries()` can still report missing ML elemental references and use
  fallback references. If strict method consistency is required, make missing
  ML references a hard failure before calling `unique_low_energy_comp()`.
- `unique_low_energy_comp()` and `unique_low_energy_chemsys()` append reference
  entries to a shallow copy of the input list. Do not rely on these helpers to
  mutate caller-owned lists; do not mutate returned entry objects in-place
  unless every caller can tolerate that.
- `min_n_return=1` allows final storage to keep one above-threshold candidate
  when no structure is within `EHULL_ML`. Downstream code should inspect
  `stable_struct["ml_selection"][...]["selected_above_threshold"]` and `ehull`
  rather than assuming every UUID is within the threshold.
- `_SKIP_GEN` is the only skip flag that bypasses the readiness wait. This is
  safe only because `prepare_gen()` marks the requested missing chemical
  systems ready first. Keep `should_run_gen()` boolean-only; validation and
  side effects belong in `prepare_gen()`.
- Failure cleanup deletes `DBChemsys` rows whose `gen_structures` field is
  empty after CSP or Generator failure. This supports retries, but it becomes
  risky if `DBChemsys` starts carrying metadata that should survive failed
  attempts.
- `MPDBMLWorkChain` stores `MPDB_ref` versions with `on_conflict="ignore"` and
  non-reference MPDB versions with `on_conflict="error"`. Re-running after
  partial writes can therefore fail for non-reference structures.
- `stable_struct` is merged rather than replaced. Preserve this behavior if
  later stages add their own keys to the same JSON field.
- `PhaseDiagramMLWorkChain` imports `get_code` and `get_model_device` but does
  not use them directly. They are used by child workflows; the parent imports
  can be removed if no external code relies on them.
- The final `DBComposition` write and generated `DBChemsys` update have
  explicit missing-row exits. Other helpers still contain broad exceptions and
  guarded first-row indexing; more explicit failure modes would make retries
  and debugging easier.

## Suggested future improvements

1. Add unit tests for the three gate methods: `should_run_mpdb_ml()`,
   `should_run_csp()`, and `should_run_gen()`.
2. Add an integration-style test for `store_stable_structs()` using a small
   fake database fixture with elemental refs, generated entries, and duplicate
   structures.
3. Replace broad `except:` blocks with targeted exceptions and reports that
   include the failed formula or chemical system.
4. Consider moving DBChemsys cleanup into a helper with tests before adding more
   fields to that table.
5. Add stricter method-consistency validation for elemental references. The
   code still allows `get_ref_entries()` to report missing ML references and
   supply fallback references.
6. Add provenance metadata recording which child branches ran: MPDB, CSP,
   MatterGen, GNoME, minima hopping, and the exact model/checkpoint used.
