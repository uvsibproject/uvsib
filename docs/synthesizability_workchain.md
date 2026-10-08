# SynthesizabilityScreenWorkChain / CSLLMWorkChain maintainer notes

This document explains the **CSLLM synthesizability screen**: what it predicts,
how it is wired into the pipeline, which parameters control it, where results
are stored, and how they appear in the pipeline report.

The short version: after `PhaseDiagramMLWorkChain` has chosen the ML bulk
selection for a composition (the bulks that passed the E_above_hull screen),
and after the optical screen, this branch runs
[CSLLM](https://github.com/szl666/CSLLM) (Song et al., *Nat. Commun.* **16**,
6530 (2025)) on every selected bulk and stores, per bulk:

- **P(synthesizable)** with an entropy-based uncertainty and a display label;
- the predicted **synthesis method** (`solid_state` / `solution` /
  `solid_state&solution`) with class probabilities;
- ranked **precursor sets**, each element-checked against the target.

It is **opt-in** (`settings.SYNTHESIZABILITY_ENABLED`, default off) and
**advisory**: it never removes a structure, and a failure never fails the
phase diagram.

## Source map

| Code | Role |
|---|---|
| `workchains/synthesizability_screen.py` | `SynthesizabilityScreenWorkChain` — reads the ML bulk selection, submits one `CSLLMWorkChain`, applies the label thresholds, writes `DBSynthesizability` rows. Entry point `synthesizabilityscreen`. |
| `codes/csllm/workchain.py` | `CSLLMWorkChain(BaseRestartWorkChain)` — runs `CSLLMCalculation` with automatic restarts. Entry point `csllm` (in `aiida.workflows`). |
| `codes/csllm/calculation.py` | `CSLLMCalculation(CalcJob)` — stages `csllm.py`, retrieves `output.json`. Entry point `csllm` (in `aiida.calculations`). |
| `codes/csllm/parser.py` | `CSLLMParser` — `output.json` → `output_dict`. Entry point `csllm_parser`. |
| `codes/files/csllm.py` | The staged runner: loads the LLaMA3-8B base once, attaches the three CSLLM LoRA adapters, and scores or generates. |
| `db/tables.py` | `DBSynthesizability` — one row per `(structure_uuid, synthesizability_model)`. |
| `db/utils.py` | `upsert_synthesizability()` — a rerun replaces the previous row. |
| `workchains/phase_diagram.py` | Hosts the `if_(should_run_synthesizability)` branch after the optical screen. |
| `workchains/pipeline_report.py` | `synthesizability_for_bulk()` and the "Synthesizability" report section. |
| `workflows/settings.py` | `SYNTHESIZABILITY_ENABLED`; reads the `synthesizability:` block of `input.yaml`. |

## Where it runs and on what

```text
PhaseDiagramMLWorkChain
  ... store_stable_structs            # E_above_hull screen -> stable_struct.ml_selection
  if optical_screen:  OpticalScreenWorkChain
  if synthesizability: SynthesizabilityScreenWorkChain   # sequential, after the optical screen
  final_report
```

Input bulks are `DBComposition.stable_struct["ml_selection"]`, read from the
`DBStructureVersion` with `method == ml_bulk_model` (the same source as
SurfaceBuilderWorkChain and the optical screen). `ml_selection` can contain a
bulk **above** `EHULL_ML` when no bulk is below it (`min_n_return=1`);
`include_above_threshold` (default `true`) scores it anyway, and the report
flags it.

Not covered: compositions whose `pd_ml` step was already `Done` before the
screen was enabled (see *Backfill*), and SQS runs (they skip `pd_ml`).

## How CSLLM is called

The Hugging Face repository holds **LoRA adapters** (r = 8 on `q_proj`/`v_proj`)
over one base model, so the job loads the base once (~16 GB in bf16/fp16) and
switches adapters per task.

Prompts are the **training** format from CSLLM's `material_str.py` /
`cons_data.py` (LMFlow `text_only`), not the chat template in `gui.py`:

```text
Input: Can this material structure be synthesized "<material string>"? \n Output: True|False
Input: How can this material structure be synthesized "<reduced formula>"? \n Output: <method>
Input: How can this material structure be synthesized "<reduced formula>"? \n Output: ['A', 'B', ...]
```

- **Material string**: `"<spg> |a,b,c,α,β,γ| (El-<mult><letter>[x y z])->..."`,
  built with pyxtal exactly as in CSLLM. MLIP-relaxed cells can carry symmetry
  noise, so the tolerance falls back `symprec → 0.05 → 0.1`; the tolerance
  actually used is stored in `attributes.symmetry_tol`.
- **Synthesizability and method are scored, not parsed**: the sequence
  probability of each allowed answer is computed directly and normalized
  (`label_probabilities`). `attributes.label_mass` (and
  `method_label_mass`) is the probability mass that landed on valid answers
  before normalization. Values near 1 mean the model answers in its trained
  format; a low value means the prompt or structure is off-distribution and
  the score should not be trusted.
- **Uncertainty** = binary entropy of P(synthesizable), in bits (0 = certain,
  1 = coin flip).
- **Precursors**: beam search returns `precursor_num_sets` lists. Each is parsed
  and checked: every target element must come from a precursor (O/N/H may
  come from the atmosphere), and foreign elements must be volatile (O/C/H/N).
  This is an element check, **not** a stoichiometric balance.
- **Method and precursors depend only on the formula**, so they are computed
  once per formula and shared by all polymorphs.
- **Domain**: CSLLM's precursor model was validated on binary and ternary
  compounds. `in_domain = n_elements <= precursor_max_elements`; the report
  marks other compositions "precursors out of domain".

## Configuration

`input.yaml`:

```yaml
synthesizability:
  enabled: true
  include_above_threshold: true
  score_threshold: 0.5          # label only
  uncertainty_threshold: 0.8    # bits; above -> "uncertain"
  predict_method: true
  predict_precursors: true
  precursor_num_sets: 3
  precursor_max_elements: 3
  symprec: 0.01
  dtype: auto                   # bf16 if the GPU supports it, else fp16
```

`config.yaml`:

```yaml
models:
  CSLLM: csllm                  # dir under path_to_pretrained_models
codes:
  CSLLM:
    code_string: csllm@rosi_gpua100
    job_script: {device: cuda, nodes: 1, ntasks: 1, cpus: 8, time: 3600, exclusive: False}
```

The thresholds only set the label. The raw score and uncertainty are always
stored, so you can change a threshold later without rerunning the model.

## Storage: `db_synthesizability`

| Column | Content |
|---|---|
| `structure_uuid` | FK → `db_structure.uuid`; join to versions, surfaces, adsorbates |
| `composition` | DBComposition formula |
| `synthesizability_model` | `"CSLLM"` |
| `synthesizability_score` | P(synthesizable) |
| `synthesizability_label` | `synthesizable` / `not synthesizable` / `uncertain` |
| `synthesizability_uncertainty` | binary entropy, bits |
| `predicted_synthesis_method` | top method label |
| `predicted_precursors` | ranked list of `{precursors, beam_score, raw, element_consistent, missing_elements, extra_elements, unparsable}` |
| `in_domain` | precursor model's validated scope |
| `ehull` | E_above_hull snapshot from `ml_selection` |
| `attributes` | material string, space group, label masses, method probabilities, thresholds, run config, notes |

A new table needs no migration: `run_dir/create_tables.py::init_db()` creates
missing tables on the next run. Rerunning the screen overwrites the row for
the same `(structure_uuid, model)`.

Example of ranking candidates together:

```sql
SELECT s.composition, s.structure_uuid, s.ehull, s.synthesizability_score,
       s.predicted_synthesis_method, MIN(a.eta) AS best_eta
FROM db_synthesizability s
LEFT JOIN db_surface_ml_adsorbate a ON a.structure_uuid = s.structure_uuid
GROUP BY s.id
ORDER BY s.synthesizability_score DESC, best_eta;
```

## Report

`pipeline_report.summarize()` adds `summary["synthesizability"]` (a list, one
item per model). `render_html_report()` adds a **Synthesizability** section
after Light Harvesting: bulk, E_hull, model, P(synth.), verdict, uncertainty,
method (hover for all class probabilities), and the top precursor set with its
element check plus the alternative sets. The sidebar shows which model ran.
`raw_data.json` carries everything. The section is absent if the screen did
not run.

## Deployment (rosi4)

1. Weights (~16 GB needed; skip the `checkpoint-*` folders and the LLaMA-7B files):

   ```bash
   cd /bigdata/casus/fwuk/mirho50/pretrained_models
   git lfs install
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/zhilong777/csllm
   cd csllm
   git lfs pull --include="llama3-8bf-hf/*,synthesis_llm_llama3/*,method_llm_llama3/*,precursor_llm_llama3/*" \
                --exclude="*/checkpoint-*/*"
   ```

2. Environment:

   ```bash
   python -m venv /bigdata/casus/fwuk/mirho50/venv_csllm
   source /bigdata/casus/fwuk/mirho50/venv_csllm/bin/activate
   pip install torch transformers peft accelerate pyxtal pymatgen
   ```

3. AiiDA code: `verdi code create core.code.installed --config
   aiid_computers_and_codes/code_csllm_rosi_gpua100.yaml` (same pattern as the
   `electronic` code). Then set `synthesizability.enabled: true`.

4. **Validate before trusting the numbers.** Run the runner by hand on ~50
   entries of CSLLM's `data/sym_classify/test/test_sym_15012.json`. Accuracy
   should be close to the published value, and `label_mass` should be near 1.
   If not, the prompt format or the base-model files do not match what the
   adapters were trained on.

## Backfill

Compositions whose `pd_ml` was already `Done` never reach this branch.
Submit the screen directly for them:

```python
from aiida.engine import submit
from aiida.orm import Str
from aiida.plugins import WorkflowFactory
W = WorkflowFactory("synthesizabilityscreen")
for formula in ["SnO2", ...]:
    submit(W, chemical_formula=Str(formula))
```

Then regenerate the report. The `pipeline_report` step is skipped once it is
`Done`, so either clear `step_status.pipeline_report.<reaction>.<path>` or call
`pipeline_report.render_html_report()` directly.

## Licensing

The CSLLM code is MIT and the Hugging Face weights are Apache-2.0. The base
model is subject to the Meta Llama 3 license.
