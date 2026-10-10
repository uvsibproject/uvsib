import os
import yaml
from aiida.orm import Group
from aiida.manage.configuration import load_profile

load_profile()

this_directory = os.path.abspath(os.path.dirname(__file__))
uvsib_directory = os.path.split(this_directory)[0]

run_folder_group = Group.collection.get(label='uvsib_run_folder')
run_dir = run_folder_group.nodes[0].value

with open(os.path.join(run_dir, 'input.yaml'), 'r', encoding='utf8') as fhandle:
    inputs = yaml.safe_load(fhandle)

with open(os.path.join(run_dir, 'config.yaml'), 'r', encoding='utf8') as fhandle:
    configs = yaml.safe_load(fhandle)

api_key = configs['MP_API_KEY']['api_key']

MAX_NUM_BULK = 5
MAX_NUM_SURF = 4
MAX_NUM_ADS = 2

EHULL_ML = 0.1
EHULL_SCAN = 0.1
DFT_FUNC = 'GGA'  # GGA/r2SCAN

# Run the GNoME (SAPS) generator in parallel with MatterGen in the gen + csp
# paths. Opt-in via input.yaml (`gnome: {enabled: true}`); defaults off so
# existing runs without the block are unaffected.
GNOME_PARALLEL = bool(inputs.get('gnome', {}).get('enabled', False))

# Run MatterGen's de-novo generator in the gen path (GeneratorWorkChain).
# Configured in input.yaml under `mattergen: {gen_enabled: ...}`; defaults ON,
# so MatterGen is the default gen-path generator for runs without a
# `mattergen`/`gnome` block (DiffCSP does not participate in the gen path --
# see MATTERGEN_CSP_ENABLED/DIFFCSP_ENABLED below). Turning this off requires
# gnome.enabled on instead; at least one must be enabled for the gen path.
MATTERGEN_GEN_ENABLED = bool(inputs.get('mattergen', {}).get('gen_enabled', True))

# Run MatterGen's CSP mode in the csp path (CSPWorkChain), parallel to
# GNoME/DiffCSP CSP if those are also enabled. Configured in input.yaml under
# `mattergen: {csp_enabled: ...}`; defaults OFF, since DiffCSP (below) is the
# default CSP-path generator. Independent from MATTERGEN_GEN_ENABLED -- e.g.
# MatterGen can generate de-novo without also running its CSP mode, or vice
# versa.
MATTERGEN_CSP_ENABLED = bool(inputs.get('mattergen', {}).get('csp_enabled', False))

# Run the DiffCSP generator (https://github.com/jiaor17/DiffCSP,
# scripts/sample.py -- composition-conditioned structure prediction) in the
# csp path (CSPWorkChain), parallel to MatterGen/GNoME CSP if those are also
# enabled. Configured in input.yaml under `diffcsp:`; defaults ON, so DiffCSP
# is the default CSP-path generator for runs without a
# `diffcsp`/`mattergen`/`gnome` block. See DiffCSP_CSP for its parameters.
# DiffCSP is not wired into the gen path (GeneratorWorkChain) -- its
# generation task there would need to be unconditioned on chemical system
# (scripts/generation.py), which was a worse fit; see docs/diffcsp_generation.md.
DIFFCSP_ENABLED = bool(inputs.get('diffcsp', {}).get('enabled', True))

# Run adaptive kinetic Monte Carlo after adsorbate screening. This is opt-in
# because dimer searches are much more expensive than the CHE adsorbate pass.
AKMC_ENABLED = bool(inputs.get('akmc', {}).get('enabled', False))

# No-DFT electronic / light-harvesting screen (ML band gap + Butler-Ginley band
# edges + photocatalytic straddle test) as a MainWorkChain step
# (step_status["optical_screen"]) after the phase-diagram / SQS stages and
# before the surface builder; it also runs under the surface soft stop. On by
# default; disable via input.yaml (`optical_screen: {enabled: false}`). Needs a dedicated `Electronic` code (see
# config.yaml) whose environment provides the pretrained gap models (matgl,
# optionally alignn). Mandatory when enabled: a failure (including a missing
# `Electronic` code) marks the step and composition Failed and stops the
# MainWorkChain (exit 300).
# `optical_screen.gate_surface_builder: true` additionally restricts the bulks
# handed to SurfaceBuilderWorkChain to those predicted to absorb visible light
# (reaction-agnostic gap window).
OPTICAL_SCREEN_ENABLED = bool(inputs.get('optical_screen', {}).get('enabled', True))

# CSLLM synthesizability / synthesis-method / precursor prediction
# (SynthesizabilityScreenWorkChain) as a PhaseDiagramMLWorkChain branch, run
# after the ML bulk selection on the bulks that passed the E_above_hull screen.
# It never filters structures. On by default; disable via input.yaml
# (`synthesizability: {enabled: false}`). Needs a dedicated GPU `CSLLM` code
# and the CSLLM weights (see config.yaml and docs/synthesizability_workchain.md).
# Mandatory when enabled: a failure (including a missing `CSLLM` code) fails
# PhaseDiagramMLWorkChain (exit 305).
SYNTHESIZABILITY_ENABLED = bool(inputs.get('synthesizability', {}).get('enabled', True))

# Soft stop: gracefully end the MainWorkChain after the generation/phase-diagram
# stages, before the surface builder (and adsorbates) start.
# Opt-in via input.yaml (`soft_stop: {before_surface_builder: true}`); absent or
# false -> the full pipeline runs as before.
SOFT_STOP_BEFORE_SURFACE = bool(inputs.get('soft_stop', {}).get('before_surface_builder', False))

# E_above_hull uncertainty of the primary bulk MLIP (bulk_relax.model)
# (EhullUncertaintyWorkChain) as an advisory PhaseDiagramMLWorkChain branch,
# run right after the ML bulk selection. A committee of other MLIPs re-scores
# the primary-relaxed structures by single-point energies (no relaxation), one
# job per model on that model's own code; the spread of the per-model E_hull is
# stored in DBComposition.stable_struct["ml_uncertainty"]. Opt-in via
# input.yaml; default OFF. Committee energies are stored as DBStructureVersion
# rows with method = <committee model name>.
#
#   uncertainty:
#     enabled: true
#     committee:
#       - {model: UMA,       head: omat}
#       - {model: MatterSim, head: null}
#     ehull_window: 0.2          # eV/atom: competing phases re-evaluated (primary hull)
#     force_constant: 10.0       # eV/A^2: k of the harmonic relaxation-energy estimate
#     shear_modulus: 50.0        # GPa: G of the harmonic relaxation-energy estimate
#     relax_energy_floor: 0.003  # eV/atom: geometry shifts below this never flag
#     chunk_size: 500            # structures per single-point job
#
# "geometry disagreement" flag: the estimated energy a committee model would
# gain by relaxing (forces + deviatoric stress; see
# workchains/ehull_uncertainty.py) could shift its E_hull by more than
# max(spread, relax_energy_floor).
#
# Surface-energy and adsorption / overpotential uncertainty (advisory
# SurfaceUncertaintyWorkChain after SurfaceBuilderWorkChain and
# AdsorptionUncertaintyWorkChain after AdsorbatesWorkChain): the same
# single-point committee idea, on the slabs / adsorbate systems relaxed by
# face_build.model / adsorbates.model. Results go to the attributes of the
# DBSurface / DBSurfaceMLAdsorbate rows. Enabled independently of the bulk
# block above; force_constant is shared.
#
#   uncertainty:
#     surface:
#       enabled: true
#       committee:                 # used for surface energies AND adsorption
#         - {model: UMA,  head: oc22}
#         - {model: MACE, head: oc20_usemppbe}
#       gamma_floor: 0.002         # eV/A^2: geometry shifts below this never flag
#     adsorption:
#       enabled: true              # uses surface.committee
#       eta_tolerance: 0.10        # V: std(eta) above this -> "uncertain"
#       eta_floor: 0.05            # V: geometry shifts below this never flag
_uncertainty = inputs.get('uncertainty', {}) or {}
UNCERTAINTY_ENABLED = bool(_uncertainty.get('enabled', False))
UNCERTAINTY_COMMITTEE = [
    {"model": str(m["model"]), "head": m.get("head")} if isinstance(m, dict)
    else {"model": str(m), "head": None}
    for m in (_uncertainty.get('committee') or [])
]
UNCERTAINTY_EHULL_WINDOW = float(_uncertainty.get('ehull_window', 0.2))
UNCERTAINTY_FORCE_CONSTANT = float(_uncertainty.get('force_constant', 10.0))
UNCERTAINTY_SHEAR_MODULUS = float(_uncertainty.get('shear_modulus', 50.0))
UNCERTAINTY_RELAX_ENERGY_FLOOR = float(_uncertainty.get('relax_energy_floor', 0.003))
UNCERTAINTY_CHUNK_SIZE = int(_uncertainty.get('chunk_size', 500))


def _committee(block):
    return [
        {"model": str(m["model"]), "head": m.get("head")} if isinstance(m, dict)
        else {"model": str(m), "head": None}
        for m in (block.get('committee') or [])
    ]


_surface_uncertainty = _uncertainty.get('surface', {}) or {}
_adsorption_uncertainty = _uncertainty.get('adsorption', {}) or {}
SURFACE_UNCERTAINTY_ENABLED = bool(_surface_uncertainty.get('enabled', False))
SURFACE_UNCERTAINTY_COMMITTEE = _committee(_surface_uncertainty)
SURFACE_UNCERTAINTY_GAMMA_FLOOR = float(_surface_uncertainty.get('gamma_floor', 0.002))
ADSORPTION_UNCERTAINTY_ENABLED = bool(_adsorption_uncertainty.get('enabled', False))
ADSORPTION_UNCERTAINTY_ETA_TOLERANCE = float(_adsorption_uncertainty.get('eta_tolerance', 0.10))
ADSORPTION_UNCERTAINTY_ETA_FLOOR = float(_adsorption_uncertainty.get('eta_floor', 0.05))


def _check_committee(label, committee, primaries):
    """Fail early on an invalid committee: a committee model equal to a
    primary model of that stage, or listed twice, would collide with that
    model's stored results (bulk: DBStructureVersion rows with method = model
    name; surfaces / adsorption: per-model entries in the row attributes).
    Model names map case-insensitively onto the same AiiDA plugin, so they are
    compared case-insensitively."""
    names = [m["model"] for m in committee]
    if not names:
        raise ValueError(f"{label} is enabled but its committee is empty.")
    lowered = [n.lower() for n in names]
    for key, primary in primaries.items():
        if str(primary).lower() in lowered:
            raise ValueError(f"{label} committee must not contain the primary model "
                             f"{key}='{primary}'.")
    duplicates = sorted({n for n in names if lowered.count(n.lower()) > 1})
    if duplicates:
        raise ValueError(f"{label} committee lists model(s) more than once: {duplicates}.")
    missing = [n for n in names
               if n not in configs.get('codes', {}) or n not in configs.get('models', {})]
    if missing:
        raise ValueError(f"{label} committee model(s) {missing} have no 'codes' / "
                         "'models' entry in config.yaml.")


def _check_uncertainty_committee():
    if UNCERTAINTY_ENABLED:
        _check_committee("uncertainty", UNCERTAINTY_COMMITTEE,
                         {"bulk_relax.model": inputs['bulk_relax']['model']})
    if SURFACE_UNCERTAINTY_ENABLED or ADSORPTION_UNCERTAINTY_ENABLED:
        _check_committee("uncertainty.surface", SURFACE_UNCERTAINTY_COMMITTEE,
                         {"face_build.model": inputs['face_build']['model'],
                          "adsorbates.model": inputs['adsorbates']['model']})


_check_uncertainty_committee()

_PD_VERIFICATION = False
_SKIP_CSP = False
_SKIP_GEN = False

code_folder_path =  os.path.join(uvsib_directory, 'codes')
files_path = os.path.join(code_folder_path, 'files')
molecular_reference_files = os.path.join(code_folder_path, 'files', 'molecular_references')

# Where MainWorkChain.pipeline_report writes generated figures + report.html
# (via uvsib.workchains.pipeline_report.report()/render_html_report()), and the
# public URL prefix at which the backend serves that directory. Both are
# machine/deployment-specific, so they are defined in run_dir/run.py
# (REPORTS_DIR, REPORTS_URL_PREFIX) and passed through AiiDA groups, like
# run_dir above.
REPORTS_DIR = Group.collection.get(label='uvsib_reports_dir').nodes[0].value
REPORTS_URL_PREFIX = Group.collection.get(label='uvsib_reports_url_prefix').nodes[0].value
