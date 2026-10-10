# Building `venv_electronic` on rosi4

Remote Python environment for the AiiDA **`electronic`** code
([`code_electronic_rosi_gpua100.yaml`](../../aiid_computers_and_codes/code_electronic_rosi_gpua100.yaml)).
That code's `prepend_text` does nothing but:

```
source /bigdata/casus/fwuk/mirho50/venv_electronic/bin/activate
```

and then `/bigdata/casus/fwuk/mirho50/bin/run_aiida_python` runs `python -u aiida.py`,
where `aiida.py` is [`uvsib/codes/files/electronic.py`](../codes/files/electronic.py)
staged verbatim. So this venv only has to satisfy that one script:

* `matgl` — MEGNet multi-fidelity band-gap model (`megnet_mfi`, the workhorse)
* `alignn` + `jarvis-tools` — ALIGNN cross-check (`alignn_pbe`, `alignn_mbj`)
* `pymatgen` — structure I/O and Mulliken electronegativity

**Host:** `rosi4.fz-rossendorf.de`
**Path:** `/bigdata/casus/fwuk/mirho50/venv_electronic`
**Builder:** `uv` (`~/.local/bin/uv`, ≥ 0.8)

---

## 1. Why these versions

| package | pin | reason |
|---|---|---|
| `matgl` | `==4.1.0` | PyG backend. **Do not use matgl 1.1.3** (the previous pin): with it, `MEGNet-MP-2019.4.1-BandGap-mfi` gives unphysical gaps — MgO 0.27 eV at HSE fidelity (exp 7.8), SnO₂/KTaO₃/CeO₂ ≈ 0 eV, HSE < PBE, and gaps jumping by >2 eV for a 0.05 Å lattice change. matgl 4.1.0 serves the same model as `MEGNet-BandGap-mfi-MP-2019.4.1` and behaves (see §6). |
| `alignn` | `==2024.5.27` | Still has `mp_gappbe_alignn` / `jv_mbj_bandgap_alignn` and the classic `get_prediction`. 2025+/2026 releases are ALIGNN2 (`ALIGNN2_MODELS`, new API) and a broken legacy shim. Needs DGL. |
| `dgl` | `==2.4.0` | Required by alignn 2024.5.27 (matgl 4.x no longer uses it). Last DGL line; CPU wheel from `https://data.dgl.ai/wheels/torch-2.4/repo.html`; requires `torch<=2.4.0`. |
| `torch` | `==2.4.0+cpu` | Highest torch dgl 2.4.0 accepts. **CPU on purpose** — the `gpu-a100` node driver is CUDA 12.2, too old for recent cu torch; inference on a handful of structures is sub-second on CPU. |
| `numpy` | `1.26.x` (resolved) | dgl is built against the NumPy 1.x ABI. |
| `pymatgen` | latest ok | Only `Structure` + `Element.ionization_energy/electron_affinity` are used. |

Python: **3.12** (matgl 4.x needs ≥ 3.11).

`electronic.py` loads the MEGNet model by its matgl ≥ 2 name first and falls back to
the 1.x name, and records the model + matgl version in `output.json`
(`config.megnet_model`), so results from the broken stack can be told apart.

---

## 2. Build

`uv` venvs are **not relocatable** — the absolute path is baked into `bin/activate*`
and every `bin/` console-script shebang. **Create it at the final path.** If you ever
build elsewhere and move it:
`grep -rlI OLDPATH venv_electronic/ | xargs sed -i 's#OLDPATH#NEWPATH#g'`.

```bash
ssh rosi4.fz-rossendorf.de
cd /bigdata/casus/fwuk/mirho50

# keep the current one until the new one is verified
mv venv_electronic venv_electronic.old.$(date +%Y%m%d)

export UV_HTTP_TIMEOUT=180
V=/bigdata/casus/fwuk/mirho50/venv_electronic
~/.local/bin/uv venv --python 3.12 $V

~/.local/bin/uv pip install \
  --python $V/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --find-links https://data.dgl.ai/wheels/torch-2.4/repo.html \
  "torch==2.4.0+cpu" "dgl==2.4.0" \
  "matgl==4.1.0" "alignn==2024.5.27" \
  jarvis-tools pymatgen
```

The install copies a few GB into `/bigdata` (uv warns it can't hardlink across
filesystems — expected, ignore). Write a lockfile beside the venv:
`uv pip freeze --python $V/bin/python > venv_electronic.freeze.pinned_YYYYMMDD.txt`.

---

## 3. Pre-stage the pretrained models (mandatory)

Compute nodes cannot download at run time.

**MEGNet** — matgl 4.x fetches `materialyze/MEGNet-BandGap-mfi-MP-2019.4.1` from the
Hugging Face Hub into `~/.cache/matgl/` (HF cache layout: `models--materialyze--…`).
Populate it once from the login node; afterwards it loads with `HF_HUB_OFFLINE=1`:

```bash
source $V/bin/activate
python -c "import matgl; matgl.load_model('MEGNet-BandGap-mfi-MP-2019.4.1')"
```

(The old `~/.cache/matgl/MEGNet-MP-2019.4.1-BandGap-mfi/` folder is the matgl 1.x
copy; matgl 4.x ignores it.)

**ALIGNN** caches its zip next to the package. The baked figshare URL
(`figshare.com/ndownloader/...`) returns HTTP 202 with an empty body; the
`ndownloader.figshare.com` host works. Copy them from a previous venv or download:

```bash
AP=$V/lib/python3.12/site-packages/alignn
curl -sSL --fail -o "$AP/mp_gappbe_alignn.zip"     https://ndownloader.figshare.com/files/31458814
curl -sSL --fail -o "$AP/jv_mbj_bandgap_alignn.zip" https://ndownloader.figshare.com/files/31458694
```

If a **partial/empty** `*.zip` is already there (a failed earlier run), delete it
first — the download branch only fires when the file is absent.

---

## 4. Verify

```bash
RUN=/bigdata/casus/fwuk/mirho50/aiida_calculations/uvsib/2e/d3/a6a7-49e0-4557-b2dd-d799ed3f6fb4
T=/tmp/electronic_itest; rm -rf $T; mkdir -p $T; cd $T
cp "$RUN/aiida.py" "$RUN/input_structures.json" .   # or any real staged inputs
cp <repo>/uvsib/codes/files/electronic.py aiida.py   # the current script, if newer
source /bigdata/casus/fwuk/mirho50/venv_electronic/bin/activate
HF_HUB_OFFLINE=1 /bigdata/casus/fwuk/mirho50/bin/run_aiida_python \
  --models=megnet_mfi,alignn_mbj --megnet_fidelity=2 --gap_min=0.4 --gap_max=3.1 --pH=0.0
python -c "import json; d=json.load(open('output.json')); \
print(d['status'], d['config']['models_used'], d['config'].get('megnet_model')); \
[print(r['uuid'][:8], r['band_info']['gap_values_eV']) for r in d['results']]"
```

Expect `ok ['megnet_mfi', 'alignn_mbj'] MEGNet-BandGap-mfi-MP-2019.4.1 (matgl 4.1.0)`.
For that run (three ZnO bulks) both models give ≈ 1.9–2.5 eV; the matgl 1.1.3 stack
gave 0.18–0.93 eV.

Also check activation itself (catches the non-relocatable-venv trap):

```bash
bash -c "source /bigdata/casus/fwuk/mirho50/venv_electronic/bin/activate; command -v python && python --version"
```

Then remove `venv_electronic.old.*` once happy. No AiiDA code-node change is
needed — the YAML `source`s the same path.

---

## 5. Known-harmless noise

* `pytorch-lightning` may appear in the freeze next to `lightning`; unused, harmless.
* No CUDA / `torch.cuda.is_available() == False` — intentional (see §1).

---

## 6. Calibration reference (2026-10-10)

27 textbook solids (experimental lattice constants, approximate experimental gaps).
MAE vs experiment, eV:

| model | all 27 | non-magnetic, gap ≤ 4.5 eV (17) | TM oxides (9) |
|---|---|---|---|
| megnet HSE, matgl 1.1.3 (old stack) | 2.22 | 1.43 | 2.15 |
| megnet HSE, matgl 4.1.0 | 0.99 | 0.70 | 1.15 |
| megnet GLLB-SC, matgl 4.1.0 | 0.94 | 0.78 | 1.15 |
| alignn_mbj | 1.06 | 0.65 | 1.88 |

`alignn_mbj` collapses to ≈ 0 eV for Cu₂O, NiO, MnO and is poor for Fe₂O₃; no model
here handles magnetic TM oxides reliably. Script and raw results:
`/bigdata/casus/fwuk/mirho50/gap_calib_claude/`.

---

## 7. If you must upgrade later

* Bumping `alignn` into ALIGNN2 territory means rewriting `alignn_gap()` against
  `alignn.pretrained.ALIGNN2_MODELS` and the new predict entrypoint; the gap model
  keys also change (e.g. `optb88vdw_bandgap_radius`, `mbj_bandgap_radius`,
  `snumat_Band_gap_HSE_radius`). That would also drop DGL and unpin torch.
* `dgl` publishes no Linux wheels beyond the 2.4 line on `data.dgl.ai`; while
  alignn 2024.5.27 is in the venv, torch stays at ≤ 2.4.0.
* `alignn_gap()` already tolerates both scalar and 1-element-list returns from
  `get_prediction` (see [`electronic.py`](../codes/files/electronic.py)).
