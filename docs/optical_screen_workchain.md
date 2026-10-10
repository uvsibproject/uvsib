# OpticalScreenWorkChain / ElectronicWorkChain maintainer notes

This document explains the **no-DFT light-harvesting screen**: how
`OpticalScreenWorkChain` and its `ElectronicWorkChain` sub-workflow work, which
parameters control them, how the result is stored, and how it is surfaced in
the pipeline report.

The short version: after `PhaseDiagramMLWorkChain` has chosen the ML bulk
selection for a composition, the `optical_screen` stage of `MainWorkChain`
(right before the surface builder) predicts each selected bulk's **band
gap** (pretrained ML property models, no SCF), derives absolute **band-edge
positions** from an empirical relation, computes a photocatalytic **straddle
verdict** for every implemented reaction/pathway, and writes all of it to
`DBStructureVersion.band_info`. `pipeline_report.py` then renders it per bulk.

It is **on by default** (`settings.OPTICAL_SCREEN_ENABLED`; disable with
`optical_screen: {enabled: false}`) and **mandatory** when enabled: a failure
marks the step and the composition `Failed` and stops the `MainWorkChain`.

## Source map

| Code | Role |
|---|---|
| `workchains/optical_screen.py` | `OpticalScreenWorkChain` — reads the ML bulk selection, submits one `ElectronicWorkChain`, adds the straddle block, writes `band_info`. Entry point `opticalscreen`. |
| `codes/electronic/workchain.py` | `ElectronicWorkChain(BaseRestartWorkChain)` — runs `ElectronicCalculation` with automatic restarts. Entry point `electronic` (in `aiida.workflows`). |
| `codes/electronic/calculation.py` | `ElectronicCalculation(CalcJob)` — stages `electronic.py`, retrieves `output.json`. Entry point `electronic` (in `aiida.calculations`). |
| `codes/electronic/parser.py` | `ElectronicParser` — `output.json` → `output_dict`. Entry point `electronic_parser`. |
| `codes/files/electronic.py` | The staged runner. Predicts the gap (matgl MEGNet multi-fidelity; ALIGNN if importable) and the Butler–Ginley / Mulliken band edges. **No DFT, no SCF.** |
| `workchains/redox_couples.py` | Maps `(reaction, reaction_path)` → `{u_red, u_ox, role}` and provides `straddle_verdict()`. Single source of truth: the CHE calculators' own `equilibrium_potential`. |
| `db/utils.py` | `update_structure_band_info()` — sets the `band_info` **column** of one `DBStructureVersion`. |
| `workchains/main.py` | Hosts the `optical_screen` stage (before `surface_builder`) and `_construct_optical_screen_builder()`. |
| `workflows/workflows.py` | `optical_screen` is in `_SHARED_STEP_KEYS`, so `reset_orphaned_compositions()` clears a stale `Running`. |
| `workchains/pipeline_report.py` | `electronic_for_bulk()`, `bulk_straddle()`, `plot_band_alignment()`, and the "Light Harvesting" report section. |
| `workflows/settings.py` | `OPTICAL_SCREEN_ENABLED`; reads the `optical_screen:` block of `input.yaml`. |

## Why no DFT

The active chain (`phase diagram → surface builder → adsorbates → AKMC`)
computes catalytic activity but never touches light absorption. This screen
fills that gap without adding a DFT stage:

- **Gap**: matgl **MEGNet multi-fidelity** is trained jointly on PBE / GLLB-SC /
  HSE / SCAN, so it can be queried at an experiment-like fidelity; MBJ-trained
  ALIGNN (optional cross-check) is likewise far closer to experiment than PBE.
  Inference is seconds on CPU for a handful of structures.
- **Band edges**: the empirical **Butler–Ginley / Mulliken** relation

  ```text
  E_CB(V vs NHE) = chi - E_e - 0.5 * E_g
  E_VB           = E_CB + E_g
  ```

  with `E_e = 4.5` eV and `chi` the stoichiometry-weighted geometric-mean
  Mulliken electronegativity of the constituents (`chi_i = (IE_i + EA_i) / 2`,
  both from pymatgen). At pH 0 the NHE and RHE scales coincide and, because
  oxide edges shift ~Nernstian, the RHE-scale numbers are ~pH-independent — the
  scale the straddle test uses. Expect **±0.3–0.5 eV** on the edge positions;
  this is a filter/ranking input, not a quantitative prediction.

## Inputs

`OpticalScreenWorkChain` has one direct AiiDA input:

| Input | AiiDA type | Meaning |
|---|---|---|
| `chemical_formula` | `Str` | Composition whose ML bulk selection should be screened. |

`MainWorkChain._construct_optical_screen_builder()` passes only this.
Everything else is read from `settings.inputs["optical_screen"]`.

`ElectronicWorkChain` inputs (built by `OpticalScreenWorkChain.run_screen()`):

| Input | AiiDA type | Meaning |
|---|---|---|
| `input_structures` | `List` | `[{"uuid": <bulk uuid>, "structure": <pymatgen Structure.as_dict()>}, ...]` |
| `code` | `Code` | `get_code("Electronic")` → `configs["codes"]["Electronic"]["code_string"]` |
| `job_info` | `Dict` | `models`, `megnet_fidelity`, `gap_min`, `gap_max`, `pH` |
| `local_label` | `Str` | Human label for the CalcJob. |

## Configuration

`input.yaml` (all keys optional; block absent → screen disabled):

```yaml
optical_screen:
  enabled: true
  models: [megnet_mfi]        # gap models; see codes/files/electronic.py
  megnet_fidelity: 2          # 0 PBE, 1 GLLB-SC, 2 HSE, 3 SCAN
  gap_min: 1.4               # visible-light window (eV) -- label only
  gap_max: 3.1
  pH: 0.0                    # pH the RHE-scale edges are reported at
  straddle_margin: 0.2       # required head-room per band edge (V)
  gate_surface_builder: false
```

`config.yaml` needs an `Electronic` code whose environment provides the gap
models (`pip install matgl`; optionally `alignn` + `jarvis-tools`). matgl
downloads its own weights, so no `path_to_pretrained_models` entry is needed. A
CPU node is sufficient.

```yaml
codes:
  Electronic:
    code_string: electronic@<computer>
    job_script:
      device: cpu
      nodes: 1
      ntasks: 1
      cpus: 4
      time: 3600
      exclusive: False
```

| Model key | Meaning |
|---|---|
| `megnet_mfi` | matgl ≥ 4 `MEGNet-BandGap-mfi-MP-2019.4.1` (1.x name `MEGNet-MP-2019.4.1-BandGap-mfi` as fallback), queried at `megnet_fidelity`. The workhorse. |
| `alignn_pbe` | ALIGNN JARVIS `mp_gappbe_alignn` (optional). |
| `alignn_mbj` | ALIGNN JARVIS `jv_mbj_bandgap_alignn` (optional; MBJ ≈ experiment). |

Unknown or unimportable models are dropped. If **no** gap model is importable
the job still succeeds with `status = "unavailable"` and `gap_eV = null`
(no edges, no straddle).

## Outline

### OpticalScreenWorkChain

```text
setup
run_screen
inspect_screen
store_results
final_report
```

| Stage | Behavior |
|---|---|
| `setup` | Reads `optical_screen` config; loads `DBComposition.stable_struct["ml_uuid_list"]` and each matching `DBStructureVersion` (`method == ml_bulk_model`). Exits `ERROR_NO_STRUCTURES_FOUND` if the selection is empty. |
| `run_screen` | Submits one `ElectronicWorkChain` for the whole selection. |
| `inspect_screen` | Reads `output_dict`; `ERROR_CALCULATION_FAILED` if the sub-workflow failed. Logs a warning if `status == "unavailable"`. |
| `store_results` | For each result: if band edges exist, adds `band_info["straddle"]` (all reactions/pathways) and `band_info["straddle_margin_V"]`; writes via `update_structure_band_info(uuid, ml_bulk_model, band_info)`. |
| `final_report` | Reports how many `band_info` rows were written. |

### ElectronicWorkChain

Standard `BaseRestartWorkChain` outline (`setup → while(should_run_process)(run_process, inspect_process) → results`), identical in shape to `MACEWorkChain` / `SQSWorkChain`. `setup()` builds `self.ctx.inputs` with the staged CLI from `get_cmdline(job_info)`.

## Staged runner (`codes/files/electronic.py`)

**Input** (`input_structures.json`, via the `file` namespace):

```json
[{"uuid": "<bulk uuid>", "structure": "<pymatgen Structure.as_dict()>"}, ...]
```

**CLI**:

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `megnet_mfi` | comma list of gap-model keys |
| `--megnet_fidelity` | `2` | matgl mfi index (0 PBE, 1 GLLB-SC, 2 HSE, 3 SCAN) |
| `--gap_min` / `--gap_max` | `1.4` / `3.1` | visible-light window (eV); label only |
| `--pH` | `0.0` | pH the RHE-scale edges are reported at |

**Output** (`output.json` → `output_dict`):

```json
{
  "results": [{"uuid": "...", "band_info": { ... }}, ...],
  "config":  {"models_requested": [...], "models_used": [...],
              "megnet_fidelity": 2, "pH": 0.0, "gap_window_eV": [1.4, 3.1]},
  "status":  "ok" | "unavailable"
}
```

### `band_info` schema

Written by the runner, then extended by `OpticalScreenWorkChain.store_results`
(the `straddle*` keys). Stored verbatim in `DBStructureVersion.band_info`.

| Key | Meaning |
|---|---|
| `screen` | `"ml_no_dft"` |
| `screen_version` | integer, currently `1` |
| `gap_eV` | ensemble-mean gap (`null` if no model) |
| `gap_std_eV` | spread across models (0.0 for a single model) |
| `gap_values_eV` | `{model_key: gap}` for every model that succeeded |
| `gap_models` | list of model keys used |
| `megnet_fidelity` / `megnet_fidelity_label` | e.g. `2` / `"HSE"` |
| `direct_gap` | `true` / `false` / `null` (currently always `null`; hook for a classifier) |
| `mulliken_electronegativity_eV` | `chi` used for the edges |
| `band_edges_vs_rhe_V` | `{cb, vb, pH, method, E_e_eV}` (`null` if no gap / no IE data) |
| `band_edges_vs_vacuum_eV` | `{cb, vb}` |
| `absorption` | `{regime, absorbs_visible, onset_nm, window_eV}` — `regime ∈ {metallic, narrow-gap, visible, uv}` |
| `notes` | list of strings (e.g. why edges were skipped) |
| `straddle` | `{REACTION: {pathway: verdict}}` — see below (added by the workchain) |
| `straddle_margin_V` | the margin used |

`verdict` (from `redox_couples.straddle_verdict`, plus `role`/`label`):

```json
{
  "u_red": -0.11, "u_ox": 1.23,
  "margin_required_V": 0.2,
  "min_gap_eV": 1.74,
  "margin_reduction_V": 0.49,   // head-room: u_red - E_CB
  "margin_oxidation_V": 0.57,   // head-room: E_VB - u_ox
  "straddles": true,
  "role": "reduction",          // material's role in a solar-fuel cell
  "label": "CO2RR:co_to_co (u_red=-0.11 V)  vs  O2/H2O"
}
```

## Redox couples and the straddle test

For a solar-fuel material the fuel-forming half-reaction is the one under test
and the partner is almost always **water oxidation** (`u_ox = 1.23` V vs RHE).
OER and CER are the **oxidation photo-anode** case: the material does the
oxidation and the partner is **hydrogen evolution** (`u_red = 0`).

Straddle condition (V vs RHE axis; more negative = higher electron energy):

```text
E_CB <= u_red - margin     AND     E_VB >= u_ox + margin
=> required gap  E_g >= (u_ox - u_red) + 2 * margin
```

`redox_couples.all_couples()` builds `{reaction: {pathway: couple}}` by pulling
each pathway's `equilibrium_potential` from `co2rr.py` / `noxrr.py` / `nrr.py` /
`orr.py` / `her.py` / `cer.py`. OER has no `*_PATHWAYS` dict, so its single
route is keyed `"default"` — matching `check_valid()` in
`workflows/workflows.py`. `couple_for(reaction, reaction_path)` is tolerant of
the `default` / `none` / `""` spellings the frontend may send.

Because it imports the reaction modules, `redox_couples` transitively loads
`uvsib.workflows.settings` (an AiiDA profile). Call it only from a
profile-loaded context — the same constraint `pipeline_report.step_labels()`
already documents.

## Storage

`update_structure_band_info(structure_uuid, method, band_info, source=None)`
sets the `band_info` **JSONB column** (not `attributes`) of the
`DBStructureVersion` selected by `(structure_uuid, method)`. `method` is the
composition's `ml_bulk_model`, i.e. the version `PhaseDiagramMLWorkChain`
ranked — the same row `bulk_candidates()` and `SurfaceBuilderWorkChain` read.
Overwrites any previous value; returns `False` if no matching version exists.

## MainWorkChain integration

A shared (once-per-composition) stage with its own
`DBComposition.step_status["optical_screen"]` key, placed after the
phase-diagram / verification / SQS stages and before `surface_builder`:

```text
...
if_(should_run_sqs)(...)
if_(should_run_optical_screen)(
    while_(should_wait_optical_screen)(wait_sleep, check_pythonjob_sleep)
    optical_screen
    inspect_optical_screen
)
if_(should_run_surface_builder)(...)
...
```

| Method | Behavior |
|---|---|
| `should_run_optical_screen` | `False` if `settings.OPTICAL_SCREEN_ENABLED` is off, the step is already `Done`, or `stable_struct["ml_uuid_list"]` is empty (e.g. SQS-only runs) -- a skip, not a failure. Not affected by `soft_stop.before_surface_builder`. |
| `should_wait_optical_screen` | Waits while a sibling `MainWorkChain` (another reaction on the same composition) has the step `Running`. |
| `optical_screen` | Re-checks `Done`, sets the step `Running`, submits `OpticalScreenWorkChain`. A submit error (e.g. no `Electronic` code) sets the step and composition `Failed` and exits 300. |
| `inspect_optical_screen` | **Mandatory**: if the sub-workflow did not finish OK, sets the step and composition `Failed` and exits 300 -- the surface builder and everything after it do not run. Otherwise sets the step `Done`. |

To re-screen a composition (e.g. after a model fix), delete or reset its
`step_status["optical_screen"]` and resubmit; only this stage re-runs.

Note: `status == "unavailable"` from the runner (no gap model importable) is
**not** a failure of `OpticalScreenWorkChain`; the step finishes `Done` with
null gaps. Check `band_info["notes"]` / the job stdout.

## Optional: gating the surface builder

`optical_screen.gate_surface_builder: true` (default **false**) makes
`SurfaceBuilderWorkChain.get_struct_uuid()` drop bulks that the screen did not
mark as visible-light-absorbing (`band_info["absorption"]["absorbs_visible"]`),
via the `_light_absorbing()` helper.

This gate is **reaction-agnostic** (a gap-window filter), *not* the per-reaction
straddle test: the surface builder runs once per composition, before the
per-reaction fan-out, so it cannot know which reaction's couple to apply. The
per-reaction straddle verdict stays in the report. If the gate would remove
every bulk (or `band_info` is not yet present) it falls back to the full ML
selection rather than starving the pipeline.

## pipeline_report integration

`pipeline_report.py` is per `(composition, reaction, reaction_path)`.

| Function | Role |
|---|---|
| `electronic_for_bulk(chemical_formula)` | `{structure_uuid: band_info}` for the `ml_bulk_model` version. DB-only, no profile. Empty when the screen was not run. |
| `bulk_straddle(electronic, reaction, reaction_path)` | The stored verdict for this reaction/path out of one bulk's `band_info`, tolerant of the OER spellings. Pure dict access. |
| `_reaction_couple(reaction, reaction_path)` | Lazily imports `redox_couples.couple_for` (profile-loading, like `step_labels`); `None` on failure. |
| `plot_band_alignment(summaries, reaction, reaction_path)` | One CB→VB bar per bulk on an inverted V-vs-RHE axis, with the reduction/oxidation lines; green if it straddles with margin. `None` if no bulk has edges. |

`summarize()` attaches `s["electronic"]` to every bulk summary; it therefore
flows into `raw_data.json` unchanged. `render_html_report()` adds:

- an **Executive Summary** tile "Light-viable" — bulks that straddle this
  reaction's couple with margin (`—` if the screen did not run);
- **Gap (eV)** and **Straddle** columns in the Stable Bulk Structures table;
- a **"Light Harvesting — ML screen (no DFT)"** section (only when the screen
  ran): the alignment plot plus a per-bulk table — gap ± spread, character,
  `E_CB` / `E_VB` (V vs RHE), required minimum gap, reduction/oxidation
  head-room — with a caption stating the method and the ±0.3–0.5 eV edge
  uncertainty;
- a **Pipeline Metadata** field "Light Screen (no DFT)" — models + fidelity.

## Exit codes

### OpticalScreenWorkChain

| Code | Name | Cause |
|---|---|---|
| 300 | `ERROR_CALCULATION_FAILED` | The `ElectronicWorkChain` sub-workflow failed. |
| 301 | `ERROR_NO_STRUCTURES_FOUND` | No ML bulk selection to screen. |

### ElectronicWorkChain

| Code | Name | Cause |
|---|---|---|
| 400 | `ERROR_MAX_RESTARTS_EXCEEDED` | `BaseRestartWorkChain` gave up. |

### ElectronicCalculation

| Code | Name | Cause |
|---|---|---|
| 100 | `ERROR_MISSING_OUTPUT` | `output.json` not retrieved. |
| 200 | `ERROR_NO_RETRIEVED_FOLDER` | Retrieved folder inaccessible. |
| 303 | `ERROR_OUTPUT_INCOMPLETE` | `output.json` present but malformed. |

Any of these makes `OpticalScreenWorkChain` not finish OK, which fails the
`MainWorkChain` stage (`MainWorkChain` exit 300, `step_status["optical_screen"] = "Failed"`).

## Practical change notes

- **Method contract.** `band_info` is written on the `ml_bulk_model` version.
  If `PhaseDiagramMLWorkChain` ever stores the selection under a different
  method, update `_selected_bulks()` (optical_screen) and `electronic_for_bulk()`
  (pipeline_report) together.
- **Straddle keys.** `store_results` keys `band_info["straddle"]` by uppercase
  reaction and lowercase pathway, `"default"` for OER — the same normalization
  as `DBSurfaceMLAdsorbate.reaction` / `.reaction_path`. `bulk_straddle()`
  relies on this. Keep `redox_couples.all_couples()` and `check_valid()` in
  sync; both already derive pathway names from the `*_PATHWAYS` dicts.
- **Equilibrium potentials.** They live only in the reaction modules. Do not
  hard-code them in `redox_couples.py`.
- **Model API drift.** `megnet_gap()` tries both `state_attr` and `state_feats`
  kwargs before the positional fallback. ALIGNN's `pretrained` API is
  version-dependent and any failure silently drops that model — check the job's
  stdout (`[electronic] model ... unavailable`) if a cross-check is missing.
- **Edge uncertainty.** Butler–Ginley is systematically off (~0.3–0.5 eV) for
  some classes (d⁰ oxides, chalcogenides). A per-anion-class empirical
  correction, or a small residual model, would go in
  `codes/files/electronic.py::build_band_info` (fit against JARVIS band-edge /
  IP–EA data, still no DFT).
- **`direct_gap`** is a stored `null` today. Wiring an ALIGNN direct/indirect
  classifier only needs `build_band_info` to fill it; the report column already
  renders it.
- **Mandatory by design.** `MainWorkChain.inspect_optical_screen` stops the
  pipeline on failure, so the surface builder (and its optional optical gate)
  never runs on un-screened bulks.
