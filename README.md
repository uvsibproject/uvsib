# UvSiB — Conversion of Solar Energy into Fuels

**UvSiB** is an open, workflow-driven platform for accelerating the discovery and evaluation of photocatalytic materials for solar-fuel production.

The project combines high-throughput atomistic simulation, machine-learning models, materials databases, and automated scientific workflows. Its long-term goal is to give researchers in materials science, chemistry, and physics a user-friendly way to design virtual experiments, screen candidate materials, and identify promising photocatalysts without requiring specialist expertise in artificial intelligence or workflow automation.

> **Project status:** Active research software under development. APIs, workflows, database schemas, and installation procedures may change.

## Why UvSiB?

Solar energy can, in principle, drive the production of fuels and fundamental chemicals from abundant feedstocks such as water and carbon dioxide. Practical photocatalysts, however, must satisfy several requirements simultaneously:

- suitable light absorption and electronic structure;
- efficient charge separation and transport;
- favorable surface reaction energetics;
- chemical and structural stability;
- low cost, scalability, and synthesizability.

Experimental trial-and-error across the enormous chemical and structural search space is expensive and slow. UvSiB addresses this challenge by connecting materials generation, machine-learning screening, first-principles verification, surface modelling, and reaction analysis in reproducible computational workflows.

## Project vision

UvSiB is being developed as an open-access virtual materials-design environment with three central capabilities:

1. **High-throughput simulation** — automated calculations over large sets of candidate materials.
2. **Machine-learning-assisted discovery and analysis** — faster structure generation, relaxation, screening, and mapping between structures and properties.
3. **Composable scientific workflows** — reusable computational building blocks that can be combined into virtual experiments and executed with minimal manual intervention.

The platform is intended to support the discovery of materials for solar-fuel reactions, including water splitting and carbon-dioxide conversion, and to provide reusable data, models, and workflows to the wider community.

## Repository scope

This repository contains the computational backend and workflow components of UvSiB. The current codebase includes:

### Materials generation and exploration

- crystal-structure prediction and generation workflows;
- integrations for **MatterGen**, **GNoME**, and **DiffCSP**;
- minima-hopping structure search.

### Machine-learning interatomic potentials

AiiDA calculation, parser, and workchain integrations are provided for several machine-learning models, including:

- MatterSim;
- MACE;
- uPET;
- UMA;
- SevenNet.

These models can be used for rapid structure relaxation and screening before more expensive first-principles verification.

### Thermodynamic stability

- machine-learning phase-diagram construction and convex-hull analysis;
- DFT phase-diagram verification;
- precursor / synthesis-route search.

### Electronic structure and light harvesting

- band-alignment workflows with hybrid-functional (HSE) band edges and core-level referencing;
- a no-DFT light-harvesting screen (`OpticalScreenWorkChain` / `ElectronicWorkChain`) that predicts the
  band gap from pretrained ML property models (matgl MEGNet multi-fidelity, optional ALIGNN) and the
  absolute band-edge positions from the empirical Butler–Ginley / Mulliken-electronegativity relation;
- a photocatalytic band-edge *straddle* test against the target reaction redox couples (`redox_couples`).

### Surfaces and photocatalytic reactions

- surface and slab construction and selection (`SurfaceBuilderWorkChain`);
- adsorption-site and adsorbate generation (ML and DFT variants);
- reaction-path and free-energy data models;
- workflows for catalytic-reaction analysis (HER, OER, ORR, CER, CO2RR, NRR, NOxRR);
- adaptive kinetic Monte Carlo (`AKMCWorkChain`) using MLIP dimer / saddle searches;
- an end-to-end pipeline report that aggregates stability, light harvesting, surfaces, and activity per composition.

## Software architecture

UvSiB is structured as an **AiiDA plugin**. The main components are:

```text
uvsib/
├── codes/          # AiiDA calculations, parsers, workchains, and executable templates
│                   #   (mattergen, gnome, diffcsp, mattersim, mace, upet, uma,
│                   #    minimahopping, electronic, precursor_search, vasp)
├── workchains/     # Higher-level scientific workflows
├── workflows/      # Workflow orchestration and settings
├── db/             # SQLAlchemy database models and utility functions
├── docs/           # Developer and installation notes
├── setup.py
└── setup.json      # Package metadata, dependencies, and AiiDA entry points
```

### Registered workflows

The `aiida.workflows` entry points currently include:

| Entry point | Purpose |
|---|---|
| `mattergen.base`, `mattergen.csp`, `gnome.base`, `gnome.csp`, `diffcsp.csp` | Generative structure generation and composition-conditioned CSP |
| `csp`, `gen` | Crystal-structure prediction and de-novo generation orchestration |
| `mattersim`, `mace`, `upet`, `uma` | ML-interatomic-potential relaxation / single points |
| `minimahopping` | Minima-hopping structure search |
| `mpdbml` | ML relaxation of Materials Project database structures |
| `phasediagram` | ML phase-diagram construction and stable-structure selection |
| `opticalscreen`, `electronic` | No-DFT band-gap / band-edge light-harvesting screen |
| `bandalignment` | DFT (PBE + HSE) band alignment |
| `surfacebuilder` | Slab construction, relaxation, and surface selection |
| `adsorbates` | Adsorbate generation and reaction free-energy analysis |
| `akmc` | Adaptive kinetic Monte Carlo with MLIP saddle searches |

The `aiida.calculations` and `aiida.parsers` groups register the matching low-level codes, including
`electronic` (ML band gap / band edges) and `precursor_search`.

## Requirements

### Core package (`setup.json` → `install_requires`)

Installed automatically with `pip install -e .`:

- **Python** 3.10 or newer;
- **AiiDA** (`aiida-core`, pulled in via the plugins below) and `aiida-pythonjob==0.4.8`;
- `aiida-vasp==4.1.0` — DFT CalcJobs (band alignment, DFT phase-diagram verification, DFT adsorbates);
- `aiida-submission-controller==0.1.2` — high-throughput submission;
- `ase` — atomic structures, optimizers, and the MLIP calculator interface;
- `pymatgen` — structure I/O, symmetry, surfaces, phase diagrams, electronegativities;
- `mp-api` — Materials Project database client.

Also imported by the workchains and expected in the daemon environment: `numpy`, `scipy`,
`sqlalchemy` (the UvSiB database models), `pyyaml`, and `matplotlib` (only for the pipeline
report / plotting helpers, imported lazily).

### Per-workflow / remote-code packages

Most heavy calculations run on remote computers, each with its **own isolated virtual environment**
holding only what that one runner (`codes/files/*.py`) needs. These are **not** declared in `setup.json`.

**Band-gap / light-harvesting screen** (`electronic` code — see `docs/venv_electronic_build.md`):

- `matgl==1.1.3` — MEGNet multi-fidelity band-gap model (`megnet_mfi`, the workhorse; PBE / GLLB-SC / HSE / SCAN fidelities);
- `dgl==2.1.0` — DGL backend required by matgl 1.x;
- `torch==2.2.0` (CPU build) and `torchdata==0.7.1`;
- `numpy<2` (dgl 2.1.0 is built against the NumPy 1.x ABI);
- `lightning==2.2.5`, `pydantic`, `pydantic-settings`, `pyparsing<3`;
- `alignn==2024.5.27` + `jarvis-tools` — optional ALIGNN (`alignn_pbe`, `alignn_mbj`) cross-check;
- `pymatgen`, `ase` — structure I/O and Mulliken-electronegativity band edges.

  The pretrained models (`MEGNet-MP-2019.4.1-BandGap-mfi`, optional ALIGNN zips) must be **pre-staged**
  into the cache; compute nodes cannot download them at run time.

**ML interatomic potentials** (one env per backend): `mattersim`, `mace-torch` (MACE), `upet` (uPET),
`fairchem` (UMA), `sevenn` (SevenNet) — each with its own `torch` build; plus `minimahopping` for the
minima-hopping runner.

**Generative models**: `mattergen`, the GNoME SAPS runner (pymatgen only), and `diffcsp` (its upstream
repo checkout, `torch`, `pytorch-lightning`, `hydra-core`, …).

**Precursor search**: an external containerized agent (optionally literature APIs such as
Crossref / OpenAlex / Semantic Scholar); no fixed Python dependency in this repo.

Individual workflows may also require external DFT codes, trained model checkpoints, pseudopotentials,
PostgreSQL/database access, scheduler configuration, API keys, and AiiDA computer/code registrations.

## Installation

Clone the repository and install it in an isolated Python environment:

```bash
git clone https://github.com/hmhoseini/uvsib.git
cd uvsib

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm that AiiDA can discover the plugin entry points:

```bash
verdi plugin list aiida.calculations
verdi plugin list aiida.workflows
verdi plugin list aiida.parsers
```

Before running workflows, configure:

1. an AiiDA profile;
2. one or more local or remote computers;
3. the required simulation codes and ML executables;
4. one isolated virtual environment per remote code (the ML potentials, generative models, the
   `electronic` band-gap runner, …), with the pretrained model files pre-staged into their
   caches — see `docs/venv_electronic_build.md` for the worked example;
5. PostgreSQL/database access where needed;
6. API keys and project-specific settings.

Because the repository is under active development, consult the source code and files in `docs/` for workflow-specific configuration.

## Reproducibility and provenance

AiiDA records calculation inputs, outputs, codes, computational resources, and workflow dependencies as a provenance graph. UvSiB builds on this capability so that multi-stage materials-discovery campaigns can be inspected, reproduced, restarted, and extended.

The separate UvSiB database stores domain-level entities and curated results needed by the platform, while AiiDA retains detailed computational provenance.

## Roadmap

Planned development directions include:

- a user-friendly open web interface;
- reusable end-to-end photocatalyst-screening workflows;
- expanded initial datasets and pretrained models;
- improved workflow composition and monitoring;
- broader support for water splitting and carbon-dioxide conversion;
- stronger links between calculated candidates and experimental validation;
- documentation, examples, tests, and reproducible demonstration cases.

## Funding and project period

UvSiB is conducted at the **Center for Advanced Systems Understanding (CASUS)**, an institute of the **Helmholtz-Zentrum Dresden-Rossendorf (HZDR)**.

The project is supported through the European Union's **Just Transition Fund (JTF)** and with participation from the Free State of Saxony under the *Forschung InfraProNet 2021–2027* funding framework.

## License

The package metadata identifies this repository as released under the **MIT License**. Add a root-level `LICENSE` file to make the licensing terms explicit for users and contributors.

## Acknowledgment

When presenting or reusing UvSiB, please acknowledge CASUS at HZDR, the European Union's Just Transition Fund, and the Free State of Saxony.
