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

# Classify all generated structures by synthesizability (thermo + reaction +
# PU) as a MainWorkChain stage after the phase diagram. Configured in input.yaml
# under `synthesizability:`; enabled by default (post-processing only, no jobs).
SYNTH_ENABLED = bool(inputs.get('synthesizability', {}).get('enabled', False))

# Run adaptive kinetic Monte Carlo after adsorbate screening. This is opt-in
# because dimer searches are much more expensive than the CHE adsorbate pass.
AKMC_ENABLED = bool(inputs.get('akmc', {}).get('enabled', True))

# No-DFT electronic / light-harvesting screen (ML band gap + Butler-Ginley band
# edges + photocatalytic straddle test) as a PhaseDiagramMLWorkChain branch,
# right after the ML bulk selection. Opt-in via input.yaml (`optical_screen:
# {enabled: true}`) because it needs a dedicated `Electronic` code (see
# config.yaml) whose environment provides the pretrained gap models (matgl,
# optionally alignn). Default OFF, so runs without the block are unaffected.
# `optical_screen.gate_surface_builder: true` additionally restricts the bulks
# handed to SurfaceBuilderWorkChain to those predicted to absorb visible light
# (reaction-agnostic gap window); default off.
OPTICAL_SCREEN_ENABLED = bool(inputs.get('optical_screen', {}).get('enabled', False))

# Soft stop: gracefully end the MainWorkChain after the generation/phase-diagram/
# synthesizability stages, before the surface builder (and adsorbates) start.
# Opt-in via input.yaml (`soft_stop: {before_surface_builder: true}`); absent or
# false -> the full pipeline runs as before.
SOFT_STOP_BEFORE_SURFACE = bool(inputs.get('soft_stop', {}).get('before_surface_builder', False))

_PD_VERIFICATION = False
_SKIP_CSP = False
_SKIP_GEN = False

code_folder_path =  os.path.join(uvsib_directory, 'codes')
files_path = os.path.join(code_folder_path, 'files')
molecular_reference_files = os.path.join(code_folder_path, 'files', 'molecular_references')

# Where MainWorkChain.pipeline_report writes generated figures + report.html
# (via uvsib.workchains.pipeline_report.report()/render_html_report()).
# Sibling to the uvsib package directory (NOT run_dir, which is a
# per-run/per-machine AiiDA-tracked directory) so reports always land in one
# fixed, predictable place next to the code -- e.g.
# /data/hossein/platform/reports if uvsib lives in /data/hossein/platform/uvsib.
REPORTS_DIR = "/opt/uvsib/backend/uploads/public/result/" #os.path.join(os.path.dirname(uvsib_directory), 'reports')

# Public URL prefix at which the backend (app/main.py) serves the contents of
# REPORTS_DIR. REPORTS_DIR is the backend's <UPLOAD_ROOT>/public/result/, and its
# StaticFiles mount maps "/uploads" -> "<UPLOAD_ROOT>/public", so a report
# written to REPORTS_DIR/<folder>/report.html is served at
# REPORTS_URL_PREFIX/<folder>/report.html. Keep the two in sync.
REPORTS_URL_PREFIX = "/uploads/result"
