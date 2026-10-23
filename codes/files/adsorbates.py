from __future__ import annotations
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Callable
from ase.data import atomic_numbers, covalent_radii
from ase import Atoms
from ase.io import jsonio
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.surface import Slab
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from scipy.spatial.distance import pdist, squareform

ch3oh_ref = Structure.from_file("ch3oh.vasp")
ch4_ref = Structure.from_file("ch4.vasp")
co2_ref = Structure.from_file("co2.vasp")
n2o_ref = Structure.from_file("n2o.vasp")
nh3_ref = Structure.from_file("nh3.vasp")
n2_ref = Structure.from_file("n2.vasp")
h2_ref = Structure.from_file("h2.vasp")
h2o_ref = Structure.from_file("h2o.vasp")
cl2_ref = Structure.from_file("cl2.vasp")
o2_ref = Structure.from_file("o2.vasp")
h2o2_ref = Structure.from_file("h2o2.vasp")
hcooh_ref = Structure.from_file("hcooh.vasp")
c2h4_ref = Structure.from_file("c2h4.vasp")
no_ref = Structure.from_file("no.vasp")
no2_ref = Structure.from_file("no2.vasp")
# NO3 is built as a neutral planar (D3h) radical. As a stand-in for the NO3-
# reactant it carries no charge/solvation correction -- the same approximation
# the former hard-coded references.yaml value used.
no3_ref = Structure.from_file("no3.vasp")

# Registry of gas-phase reference structures, keyed by the same name used
# in `ReactionStep.released` and (after `_ADS_TO_GAS` translation) by the
# pathway's initial reactant. `_pathway_required_refs` selects the subset
# that actually needs to be relaxed for a given reaction/pathway.
_GAS_REF_REGISTRY: Dict[str, Structure] = {
    "H2": h2_ref,
    "H2O": h2o_ref,
    "N2": n2_ref,
    "NH3": nh3_ref,
    "N2O": n2o_ref,
    "CO2": co2_ref,
    "CH4": ch4_ref,
    "CH3OH": ch3oh_ref,
    "Cl2": cl2_ref,
    "O2": o2_ref,
    "H2O2": h2o2_ref,
    "HCOOH": hcooh_ref,
    "C2H4": c2h4_ref,
    "NO": no_ref,
    "NO2": no2_ref,
    "NO3": no3_ref,
}

# Map *_ads adsorbate names (which appear as the first ReactionStep.reactant
# for many pathways) to the corresponding gas-phase reference name.
_ADS_TO_GAS: Dict[str, str] = {
    "CO2_ads": "CO2",
    "N2_ads": "N2",
    "O2_ads": "O2",
}


def _pathway_required_refs(reaction: str, pathway_obj) -> List[str]:
    """Return the gas-phase reference names this pathway needs computed.

    Always includes H2 and H2O when the pathway transfers protons (CHE
    reference for the H+/e- couple). Adds any species declared in
    `ReactionStep.released` and any gas-phase reactant the pathway starts
    from (via `_ADS_TO_GAS`). Special-cased for OER, which has no pathway
    dataclass.

    Parameters
    ----------
    reaction : str
        Reaction key passed to `generate_adsorbed_structures`.
    pathway_obj : ReactionPathway | None
        Pathway object returned by the corresponding `generate_*_adsorbates`
        helper. `None` is allowed for OER.

    Returns
    -------
    list[str]
        Sorted list of gas-phase reference names, all guaranteed to be keys
        of `_GAS_REF_REGISTRY`.
    """
    needed: set[str] = set()

    if reaction == "OER":
        # 4 H+/e- transfers; product is O2 from 2 H2O.
        return sorted({"H2", "H2O", "O2"})

    if pathway_obj is None:
        return ["H2", "H2O"]

    if any(s.protons != 0 for s in pathway_obj.steps):
        needed.update(["H2", "H2O"])

    if pathway_obj.steps:
        first = pathway_obj.steps[0].reactant
        gas = _ADS_TO_GAS.get(first, first)
        if gas in _GAS_REF_REGISTRY:
            needed.add(gas)

    for step in pathway_obj.steps:
        for sp in step.released:
            if sp in _GAS_REF_REGISTRY:
                needed.add(sp)
            elif sp == "O":
                # Released atomic O is referenced to 1/2 O2 downstream
                # (calculate_noxrr_overpotential sets E[O] = E[O2] / 2), so the
                # O2 reference must be computed for any O-releasing pathway.
                needed.add("O2")

    return sorted(needed)


def _reference_molecule_count(atoms: Atoms, ref_name: str) -> int:
    """Number of whole `ref_name` molecules packed into a reference cell.

    The molecular_references/*.vasp cells hold several copies of a molecule
    (e.g. h2o.vasp = 8 H2O, co2.vasp = 4 CO2, ch3oh.vasp = 1 CH3OH). The CHE
    bookkeeping in the workchains expects *per-molecule* gas-phase energies, so
    the relaxed cell energy is divided by this count before it is stored.
    """
    from ase.formula import Formula
    atoms_per_molecule = len(Formula(ref_name))
    return max(1, round(len(atoms) / atoms_per_molecule))


def _create_adsorbate_with_dummy(species: List[str],
                                 coords: List[List[float]],
                                 properties: Dict = None,
                                 height: float = 2) -> Molecule:
    """
    Create a Pymatgen Molecule with a DummySpecies binding atom.

    Parameters
    ----------
    species : list of str
        Species symbols
    coords : list of list of float
        Coordinates relative to binding site [in Angstroms].
    height : float, optional
        Length of X-Molecule bond (default: 2.0 Å).

    Returns
    -------
    Molecule
        Pymatgen Molecule with DummySpecies "X" at index 0 [0, 0, 0].
    """
    shift = np.array([0, 0, height])
    shifted_coords = [list(np.array(c) + shift) for c in coords]
    mol = Molecule(
        [DummySpecies("X")] + species,
        [[0.0, 0.0, 0.0]] + shifted_coords,
    )
    mol.properties = properties or {}
    return mol


def has_reasonable_distances(atoms: Atoms, scale: float = 0.5) -> bool:
    """
    Check if interatomic distances are physically reasonable.
    Only checks distances involving H, C, N, or O atoms.
    Parameters
    ----------
    atoms : ASE Atoms object
    scale : float, optional
        Scaling factor for minimum allowed distance (default: 0.5).
        min_distance = scale * (covalent_radius_i + covalent_radius_j)
    Returns
    -------
    bool
        True if all checked distances are >= min_distance, False otherwise.
    """
    CHECK_ELEMENTS = {1, 6, 7, 8}
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    n = len(atoms)

    for i in range(n):
        Zi = numbers[i]
        for j in range(i + 1, n):
            Zj = numbers[j]

            # Skip if neither atom is a CHECK_ELEMENT
            if Zi not in CHECK_ELEMENTS and Zj not in CHECK_ELEMENTS:
                continue

            d = np.linalg.norm(positions[i] - positions[j])
            r_min = scale * (covalent_radii[Zi] + covalent_radii[Zj])

            if d < r_min:
                return False
    return True


# ---------------------------------------------------------------------------
# Post-relaxation sanity checks for ML-relaxed adsorbates.
# See docs/sanity_checks.md for the full rationale and failure taxonomy.
# Layers 0-2 (finite energy, atom overlap, surface binding, molecular
# identity) run by default; slab-integrity (layer 3) and energy-outlier
# (layer 4) are opt-in via run_relaxation flags.
# ---------------------------------------------------------------------------

# Bond cutoff used both to build the reference adsorbate graphs and to graph
# the relaxed adsorbate. A bond connects i, j when their separation is within
# _ADSORBATE_BOND_TOL * (r_cov_i + r_cov_j). Both sides MUST use the same value.
# Calibrated at 1.25: at this value every bundled adsorbate's reference graph is
# a single connected component, while spurious 1,3 bonds (e.g. in *OCCO/*ONNO)
# only appear at >= 1.30. See docs/sanity_checks.md.
_ADSORBATE_BOND_TOL = 1.25


@dataclass
class AdsorbateValidation:
    """Outcome of validate_relaxed_adsorbate (ok + human-readable reason)."""
    ok: bool
    reason: str = ""


def _bond_graph(numbers, positions, tol: float):
    """Element-labelled covalent-bond graph for an isolated (non-periodic) set
    of atoms. Used for the reference adsorbate geometries."""
    import networkx as nx
    g = nx.Graph()
    for i, z in enumerate(numbers):
        g.add_node(i, z=int(z))
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(np.asarray(positions[i]) - np.asarray(positions[j])))
            if d <= tol * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]]):
                g.add_edge(i, j)
    return g


def _subset_bond_graph(atoms: Atoms, idx, tol: float):
    """Element-labelled covalent-bond graph for a subset of a periodic Atoms
    object, using minimum-image distances (robust to wrapping across cell
    boundaries)."""
    import networkx as nx
    numbers = atoms.get_atomic_numbers()
    g = nx.Graph()
    for k, i in enumerate(idx):
        g.add_node(k, z=int(numbers[i]))
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            d = float(atoms.get_distance(i, j, mic=True))
            if d <= tol * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]]):
                g.add_edge(a, b)
    return g


def _adsorbate_reference_graph(ads_molecule, tol: float = _ADSORBATE_BOND_TOL):
    """Intramolecular bond graph of a reference adsorbate Molecule.

    The DummySpecies 'X' binding marker is stripped first; the remaining atoms
    are graphed exactly as the relaxed adsorbate is, so the two can be compared
    by element-aware graph isomorphism.
    """
    numbers, positions = [], []
    for site in ads_molecule:
        sym = str(site.specie.symbol)
        if sym == "X":
            continue
        numbers.append(atomic_numbers[sym])
        positions.append(np.asarray(site.coords))
    return _bond_graph(numbers, positions, tol)


def validate_relaxed_adsorbate(atoms: Atoms, n_ads: int, expected_graph,
                               energy, *, graph_tol: float = _ADSORBATE_BOND_TOL,
                               bind_tol: float = 1.25, overlap_scale: float = 0.5,
                               clean_slab_atoms: Atoms = None,
                               slab_max_disp: float = 1.5) -> AdsorbateValidation:
    """Validate a single relaxed adsorbate-on-slab structure.

    The adsorbate is assumed to be the LAST ``n_ads`` atoms of ``atoms`` (this
    is how generate_adsorbed_structures appends it). Returns early with ok=True
    (skipping identity/binding) if that assumption cannot be verified, so a
    bad index guess never causes a false rejection.

    Layers
    ------
    0  finite energy; no atom overlap / explosion (has_reasonable_distances)
    1  surface binding: an adsorbate atom is within bonding range of the slab
    2  molecular identity: adsorbate's intramolecular bond graph is isomorphic
       (element-aware) to the reference -- catches dissociation/isomerisation
    3  (only if clean_slab_atoms given) slab has not reconstructed
    """
    import networkx as nx

    # --- Layer 0: finite energy + no overlap/explosion ---
    if energy is None or not np.isfinite(energy):
        return AdsorbateValidation(False, "non-finite energy")
    if not has_reasonable_distances(atoms, scale=overlap_scale):
        return AdsorbateValidation(False, "atoms too close (overlap/explosion)")

    n_total = len(atoms)
    if not (0 < n_ads <= n_total):
        return AdsorbateValidation(True, "")  # cannot locate adsorbate; skip 1-3

    numbers = atoms.get_atomic_numbers()
    ads_idx = list(range(n_total - n_ads, n_total))
    slab_idx = list(range(0, n_total - n_ads))

    # Safety: the last n_ads atoms must match the expected element multiset,
    # otherwise the append-order assumption is wrong -- skip rather than reject.
    exp_z = sorted(int(d['z']) for _, d in expected_graph.nodes(data=True))
    got_z = sorted(int(numbers[i]) for i in ads_idx)
    if exp_z != got_z:
        return AdsorbateValidation(True, "")

    # --- Layer 1: surface binding (PBC-aware) ---
    if slab_idx:
        bound = False
        for i in ads_idx:
            d = atoms.get_distances(i, slab_idx, mic=True)
            rsum = np.array([covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
                             for j in slab_idx])
            if np.any(d <= bind_tol * rsum):
                bound = True
                break
        if not bound:
            return AdsorbateValidation(False, "desorbed (no adsorbate-slab bond)")

    # --- Layer 2: molecular identity (intramolecular bond graph) ---
    got_graph = _subset_bond_graph(atoms, ads_idx, graph_tol)
    # 2a: fragmentation -- an intact adsorbate is a single connected component.
    if got_graph.number_of_nodes() > 1 and not nx.is_connected(got_graph):
        return AdsorbateValidation(False, "dissociation (adsorbate fragmented)")
    # 2b: isomerisation -- topology must match the reference (element-aware).
    node_match = nx.algorithms.isomorphism.numerical_node_match('z', -1)
    if not nx.is_isomorphic(got_graph, expected_graph, node_match=node_match):
        return AdsorbateValidation(
            False, "molecular identity changed (isomerisation)")

    # --- Layer 3 (opt-in): slab integrity ---
    if clean_slab_atoms is not None and slab_idx:
        from ase.geometry import find_mic
        n_slab = len(slab_idx)
        if len(clean_slab_atoms) == n_slab:
            disp = atoms.get_positions()[slab_idx] - clean_slab_atoms.get_positions()
            mic_disp, _ = find_mic(disp, atoms.cell, atoms.pbc)
            max_disp = float(np.linalg.norm(mic_disp, axis=1).max())
            if max_disp > slab_max_disp:
                return AdsorbateValidation(
                    False, f"slab reconstructed (max displacement {max_disp:.2f} A)")

    return AdsorbateValidation(True, "")


def _flag_energy_outliers(relaxed_sets, model_key, factor: float):
    """Layer 4 (opt-in): drop sets whose adsorbate energy is a MAD outlier.

    For each adsorbate species, energies are pooled across all sites on this
    slab; a set is rejected if any of its adsorbates deviates from the species
    median by more than ``factor`` robust standard deviations (1.4826*MAD).
    Returns (kept_sets, reject_records).
    """
    from collections import defaultdict
    decoded = [[jsonio.decode(x) for x in s["structures"]] for s in relaxed_sets]

    per_name = defaultdict(list)
    for structs in decoded:
        for a in structs:
            nm = a.info.get('adsorbate', '')
            if nm.startswith('*') and nm != '*':
                e = a.info.get(model_key)
                if e is not None:
                    per_name[nm].append(e)

    stats = {}
    for nm, es in per_name.items():
        arr = np.asarray(es, dtype=float)
        if len(arr) < 4:  # too few sites for a robust band
            continue
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) or 1e-9
        stats[nm] = (med, mad)

    kept, rejects = [], []
    for s, structs in zip(relaxed_sets, decoded):
        reason = None
        for a in structs:
            nm = a.info.get('adsorbate', '')
            if nm in stats:
                med, mad = stats[nm]
                e = a.info.get(model_key)
                if e is not None and abs(e - med) > factor * 1.4826 * mad:
                    reason = (f"energy outlier for {nm}: {e:.3f} eV "
                              f"(median {med:.3f}, MAD {mad:.3f})")
                    break
        if reason:
            rejects.append({"site_type": s["site_type"], "ads_coord": s["ads_coord"],
                            "repeat": s["repeat"], "reason": reason})
        else:
            kept.append(s)
    return kept, rejects


def pmg_to_ase(pmg_structure):
    """Convert a pymatgen Structure to an ASE Atoms object"""
    scaled_positions = pmg_structure.frac_coords
    symbols = [str(site.specie) for site in pmg_structure.sites]
    cell = pmg_structure.lattice.matrix

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True
    )

    if "selective_dynamics" in pmg_structure.site_properties:
        sd = pmg_structure.site_properties["selective_dynamics"]
        mask = [not any(flags) for flags in sd]
        if any(mask):
            atoms.set_constraint(FixAtoms(mask=mask))
    return atoms


def average_minimum_distance_structure(structure):
    """"Compute the average nearest-neighbor distance"""
    positions = structure.cart_coords
    dist_matrix = squareform(pdist(positions))
    ma = np.ma.masked_equal(dist_matrix, 0.0, copy=False)
    minimums = np.min(ma, axis=0)
    return minimums.mean()


def get_adsorption_sites(slab_pmg: Structure,
                         positions: List[str] = ['ontop', 'bridge', 'hollow']) -> tuple:
    """
    Get adsorption sites using pymatgen's AdsorbateSiteFinder.

    Parameters
    ----------
    slab_pmg : pymatgen Structure
    positions : list of str, optional
        Site types to find: 'ontop', 'bridge', 'hollow', or combinations (default: all three).
    Returns
    -------
    tuple
        (sites_dict, asf) where:
        - sites_dict: Dictionary with site information for each site type
        - asf: AdsorbateSiteFinder instance for further operations
    """
    # Height criteria for selection of surface sites
    h = 1.7 * average_minimum_distance_structure(slab_pmg)
    asf = AdsorbateSiteFinder(slab_pmg, selective_dynamics=True, height=h)

    sites_dict = asf.find_adsorption_sites(
        distance=0,
        put_inside=True,
        symm_reduce=0.1,
        near_reduce=0.1,
        positions=positions,
        no_obtuse_hollow=True
    )

    return sites_dict, asf


# ==============================================================================
# Reaction pathway data model (shared by all reaction types)
# ==============================================================================
@dataclass
class ReactionStep:
    """One elementary step in a reaction pathway.

    Attributes:
        reactant: Name of the surface intermediate before this step.
        product: Name of the surface intermediate after this step.
        step_type: ``'electrochemical'`` (H⁺ + e⁻ transfer) or ``'chemical'`` (no charge).
        electrons: Number of electrons transferred (negative = gained).
        protons: Number of protons transferred.
        released: Species desorbed/released (e.g., ``['H2O']``, ``['CO']``).
        notes: Free-text annotation (e.g., rate-limiting, competing mechanisms).
    """
    reactant: str
    product: str
    step_type: str = "electrochemical"
    electrons: int = -1
    protons: int = 1
    released: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ReactionPathway:
    """A named reaction pathway with ordered elementary steps.

    Attributes:
        name: Unique identifier (used by :func:`get_pathway`).
        description: Human-readable summary.
        steps: Ordered list of :class:`ReactionStep` objects.
        selectivity_metals: Metals where this pathway dominates in DFT/expt.
        overall_reaction: Balanced overall equation string.
    """
    name: str
    description: str
    steps: List[ReactionStep]
    selectivity_metals: List[str] = field(default_factory=list)
    overall_reaction: str = ""

    @property
    def intermediates(self) -> List[str]:
        """All surface intermediates in pathway order (no duplicates)."""
        seen: List[str] = []
        for step in self.steps:
            if step.reactant not in seen:
                seen.append(step.reactant)
        last = self.steps[-1].product
        if last not in seen:
            seen.append(last)
        return seen


def generate_oer_adsorbates():
    """Return a list of Pymatgen Molecule objects for the OER reaction"""

    X = DummySpecies("X")

    adsorbates = []

    # *O
    mol = Molecule(
        [X, "O"],
        [(0, 0, 0), (0, 0, 2.0)],
    )
    mol.properties = {"adsorbate": "*O", "energy": 0}
    adsorbates.append(mol)

    # *OH
    mol = Molecule(
        [X, "O", "H"],
        [(0, 0, 0), (0, 0, 2.0), (0.1, 0.9, 2.9)],
    )
    mol.properties = {"adsorbate": "*OH", "energy": 0}
    adsorbates.append(mol)

    # *OOH
    mol = Molecule(
        [X, "O", "O", "H"],
        [(0, 0, 0), (0, 0, 2.0), (1.2, -0.2, 2.8), (0.8, -0.2, 3.7)],
    )
    mol.properties = {"adsorbate": "*OOH", "energy": 0}
    adsorbates.append(mol)

    return adsorbates


def generate_co2rr_adsorbates(pathway_name: str) -> tuple:
    """CO2 electroreduction reaction pathways on metal surfaces.

    Provides a literature-grounded library of CO2RR intermediates and reaction
    pathways (CHE model, Nørskov group and follow-up work) that can be placed on
    *any* surface slab.

    Pathways implemented
    --------------------
    - 'co2_to_co'    : CO2 → *COOH → *CO → CO(g)            (Au, Ag, Zn)
    - 'co2_to_hcooh' : CO2 → *OCHO → HCOOH(aq)              (formate, Pd, In)
    - 'co_to_ch4'    : *CO → *CHO → *CHOH → *CH2 → *CH3 → CH4(g)  (Cu)
    - 'co_to_ch3oh'  : *CO → *CHO → *CHOH → *CH2OH → CH3OH(g)     (Cu)
    - 'co2_to_ch4'   : CO2 → CH4 full pathway on Cu
    - 'co2_to_ch3oh' : CO2 → CH3OH full pathway on Cu
    - 'co2_to_c2h4'  : 2 *CO → *OCCO → … → C2H4(g)          (Cu C–C coupling)

    Each intermediate is a Pymatgen Molecule (X is binding atom at index 0)

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Peterson et al. *Energy Environ. Sci.* **3**, 1311 (2010).
    Kuhl et al. *J. Am. Chem. Soc.* **136**, 14107 (2014).
    Montoya et al. *ChemSusChem* **8**, 2180 (2015).
    Goodpaster et al. *J. Phys. Chem. Lett.* **7**, 1471 (2016).
    """

    # =========================================================================
    # Adsorbate geometry library (Pymatgen version)
    # Each molecule includes a DummySpecies "X" at [0, 0, 0]
    # =========================================================================

    def _co():
        """*CO — C-down (atop binding on most metals)."""
        return _create_adsorbate_with_dummy(
            ["C", "O"],
            [[0, 0, 0], [0, 0, 1.15]],
            properties={"adsorbate": "*CO"}
        )

    def _cooh():
        """*COOH (carboxyl) — C-down, bidentate capable.
        Planar: C at origin, C=O pointing up, C-OH in-plane.
        """
        return _create_adsorbate_with_dummy(
            ["C", "O", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.22],
                [1.10, 0.00, -0.60],
                [1.94, 0.00, -0.12],
            ],
            properties={"adsorbate": "*COOH"}
        )

    def _ocho():
        """*OCHO (formate) — O-down bidentate or monodentate.
        Monodentate: O at origin; formate oriented upright.
        """
        return _create_adsorbate_with_dummy(
            ["O", "C", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.30],
                [0.00, 1.10, 1.85],
                [0.00, -0.95, 1.98],
            ],
            properties={"adsorbate": "*OCHO"}
        )

    def _co2_ads():
        """*CO2 — weakly adsorbed, bent (activated), O-C-O ~125°."""
        angle_rad = np.radians(125.0 / 2)
        d = 1.20
        return _create_adsorbate_with_dummy(
            ["C", "O", "O"],
            [
                [0.00, 0.00, 0.00],
                [float(d * np.sin(angle_rad)), 0.0, float(d * np.cos(angle_rad))],
                [float(-d * np.sin(angle_rad)), 0.0, float(d * np.cos(angle_rad))]
            ],
            properties={"adsorbate": "*CO2"}
        )

    def _cho():
        """*CHO (formyl) — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.09, 0.63],
                [1.05, 0.00, 0.85],
            ],
            properties={"adsorbate": "*CHO"}
        )

    def _choh():
        """*CHOH — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.09, 0.70],
                [1.23, 0.00, 0.65],
                [1.85, 0.00, 1.38],
            ],
            properties={"adsorbate": "*CHOH"}
        )

    def _ch2o():
        """*CH2O (formaldehyde adsorbed) — C-down, η1 mode."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.94, 0.00, 0.59],
                [-0.94, 0.00, 0.59],
                [0.00, 1.10, 0.60],
            ],
            properties={"adsorbate": "*CH2O"}
        )

    def _ch2oh():
        """*CH2OH — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.95, 0.55, 0.62],
                [-0.95, 0.55, 0.62],
                [0.00, -1.22, 0.62],
                [0.00, -1.90, 1.35],
            ],
            properties={"adsorbate": "*CH2OH"}
        )

    def _ch():
        """*CH — C-down (hollow site preferred)."""
        return _create_adsorbate_with_dummy(
            ["C", "H"],
            [[0, 0, 0], [0, 0, 1.09]],
            properties={"adsorbate": "*CH"}
        )

    def _ch2():
        """*CH2 — C-down."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.94, 0.00, 0.70],
                [-0.94, 0.00, 0.70],
            ],
            properties={"adsorbate": "*CH2"}
        )

    def _ch3():
        """*CH3 — C-down (atop)."""
        return _create_adsorbate_with_dummy(
            ["C", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.03, 0.70],
                [0.89, -0.51, 0.70],
                [-0.89, -0.51, 0.70],
            ],
            properties={"adsorbate": "*CH3"}
        )

    def _oh():
        """*OH — O-down."""
        return _create_adsorbate_with_dummy(
            ["O", "H"],
            [[0, 0, 0], [0, 0, 0.97]],
            properties={"adsorbate": "*OH"}
        )

    def _h():
        """*H — single hydrogen."""
        return _create_adsorbate_with_dummy(
            ["H"],
            [[0, 0, 0]],
            properties={"adsorbate": "*H"}
        )

    def _occo():
        """*OCCO (oxalyl, CO dimer) — first O binds surface."""
        return _create_adsorbate_with_dummy(
            ["O", "C", "C", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.25],
                [0.00, 1.35, 1.25],
                [0.00, 1.35, 2.50],
            ],
            properties={"adsorbate": "*OCCO"}
        )

    def _ccho():
        """*CCHO — C-down, C2 species."""
        return _create_adsorbate_with_dummy(
            ["C", "C", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.35, 0.65],
                [0.00, 2.40, 1.30],
                [0.00, 1.35, 1.76],
            ],
            properties={"adsorbate": "*CCHO"}
        )

    def _c2h4():
        """C2H4 (ethylene) — di-σ mode, C-C bridging; C1 at origin (mol_index=0)."""
        return _create_adsorbate_with_dummy(
            ["C", "C", "H", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [1.34, 0.00, 0.00],
                [-0.56, 0.92, 0.60],
                [-0.56, -0.92, 0.60],
                [1.90, 0.92, 0.60],
                [1.90, -0.92, 0.60],
            ],
            properties={"adsorbate": "C2H4"}
        )

    # Registry: name → factory function
    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "CO2_ads": _co2_ads,
        "COOH": _cooh,
        "OCHO": _ocho,
        "CO": _co,
        "CHO": _cho,
        "CHOH": _choh,
        "CH2O": _ch2o,
        "CH2OH": _ch2oh,
        "CH": _ch,
        "CH2": _ch2,
        "CH3": _ch3,
        "OH": _oh,
        "H": _h,
        "OCCO": _occo,
        "CCHO": _ccho,
        "C2H4_ads": _c2h4,
    }

    def get_adsorbate(name: str) -> Molecule:
        """Return a fresh copy of the named adsorbate geometry.
        Args:
            name: Key from the adsorbate registry.
        Returns:
            Pymatgen Molecule with X at index 0
        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    # =========================================================================
    # Reaction pathway data
    # =========================================================================
    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    _reg(ReactionPathway(
        name="co2_to_co",
        description="CO2 reduction to CO via carboxyl intermediate",
        overall_reaction="CO2 + H+ + e- → CO + H2O",
        selectivity_metals=["Au", "Ag", "Zn"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting on most metals",
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="CO desorbs on weak-binding metals (Au, Ag)",
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_hcooh",
        description="CO2 reduction to formic acid via formate intermediate",
        overall_reaction="CO2 + 2(H+ + e-) → HCOOH",
        selectivity_metals=["Pd", "In", "Sn"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="OCHO",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O-bound formate; competing with COOH on Pd/In",
            ),
            ReactionStep(
                reactant="OCHO", product="CO2_ads",  # product released
                step_type="electrochemical", electrons=-1, protons=1,
                released=["HCOOH"],
                notes="HCOOH desorbs; surface returns to clean",
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co_to_ch4",
        description="Further reduction of *CO to methane (Langmuir–Hinshelwood)",
        overall_reaction="*CO + 6(H+ + e-) → CH4 + H2O",
        selectivity_metals=["Cu"],
        steps=[
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting on Cu",
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH",
                step_type="chemical", electrons=0, protons=0,
                released=["H2O"],
                notes="water elimination; CH occupies hollow site",
            ),
            ReactionStep(
                reactant="CH", product="CH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2", product="CH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH3", product="CO",  # product = bare surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH4"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co_to_ch3oh",
        description="Further reduction of *CO to methanol",
        overall_reaction="*CO + 4(H+ + e-) → CH3OH",
        selectivity_metals=["Cu", "Mo"],
        steps=[
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH2OH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2OH", product="CO",  # bare surface
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH3OH"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_ch4",
        description="Full CO2 → CH4 pathway on Cu",
        overall_reaction="CO2 + 8(H+ + e-) → CH4 + 2 H2O",
        selectivity_metals=["Cu"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting on Cu",
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH",
                step_type="chemical", electrons=0, protons=0,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CH", product="CH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2", product="CH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH3", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH4"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_ch3oh",
        description="Full CO2 → CH3OH pathway on Cu/Mo",
        overall_reaction="CO2 + 6(H+ + e-) → CH3OH + H2O",
        selectivity_metals=["Cu", "Mo"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CO", product="CHO",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHO", product="CHOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CHOH", product="CH2OH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CH2OH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["CH3OH"],
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="co2_to_c2h4",
        description="CO2 → C2H4 via CO dimerisation on Cu",
        overall_reaction="2 CO2 + 12(H+ + e-) → C2H4 + 4 H2O",
        selectivity_metals=["Cu"],
        steps=[
            ReactionStep(
                reactant="CO2_ads", product="COOH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="COOH", product="CO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="CO", product="OCCO",
                step_type="chemical", electrons=0, protons=0,
                notes="C–C coupling of two *CO; rate-determining on Cu(100)",
            ),
            ReactionStep(
                reactant="OCCO", product="CCHO",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="CCHO", product="C2H4_ads",
                step_type="chemical", electrons=0, protons=0,
                notes="multiple H+/e- transfers collapsed; treated as chemical for pathway bookkeeping",
            ),
            ReactionStep(
                reactant="C2H4_ads", product="CO",
                step_type="chemical", electrons=0, protons=0,
                released=["C2H4"],
            ),
        ],
    ))

    def get_co2rr_pathway(name: str) -> ReactionPathway:
        """Return the named CO2RR :class:`ReactionPathway`.
        Args:
            name: One of the CO2RR pathway keys.
        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown CO2RR pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    skip_if_not_registered = {"CO"}  # TODO not clear

    pathway = get_co2rr_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        # Skip "bare surface" markers that aren't registered adsorbates
        if name in skip_if_not_registered and name not in _ADSORBATE_REGISTRY:
            continue
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_adsorbate(name)

    return pathway, adsorbates


def generate_noxrr_adsorbates(pathway_name: str) -> tuple:
    """NOx electroreduction reaction pathways on metal surfaces.

    Provides a library of NOx reduction intermediates and reaction pathways
    (CHE model) that can be placed on *any* surface slab.

    NOx covered: NO, NO₂, NO₃⁻ as starting species.

    Pathways implemented
    --------------------
    - 'no_dissociative' : *NO → *N + *O → N₂(g)              (Ru, Rh, Ir)
    - 'no_to_nh3_noh'   : *NO → *NOH → *N → *NH₂ → NH₃       (Cu, Fe)
    - 'no_to_nh3_nhoh'  : *NO → *NOH → *NHOH → *NH₂OH → NH₃  (Cu, hydroxylamine route)
    - 'no_to_n2o'       : 2*NO → *ONNO → N₂O + *O            (Pt, Pd automotive)
    - 'no2_to_no'       : *NO₂ → *NO + *O                     (prereduction step)
    - 'no3_to_nh3'      : *NO₃ → *NO₂ → *NO → … → NH₃        (eNO3RR, Cu)
    - 'no3_to_n2'       : *NO₃ → *NO₂ → *NO → *N → N₂        (eNO3RR, Ru)

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Gao et al. *Nat. Chem.* **9**, 547 (2017).
    Liu et al. *Nat. Commun.* **12**, 5797 (2021).
    Wang et al. *J. Am. Chem. Soc.* **142**, 5702 (2020).
    van 't Veer et al. *J. Phys. Chem. C* **124**, 22 (2020).
    Pérez-Ramírez & López *Nat. Catal.* **2**, 971 (2019).
    """

    # =========================================================================
    # Adsorbate geometry library (Pymatgen version with DummySpecies)
    # =========================================================================

    def _no():
        """*NO — N-down (preferred on most transition metals)."""
        return _create_adsorbate_with_dummy(
            ["N", "O"],
            [[0, 0, 0], [0, 0, 1.15]],
            properties={"adsorbate": "*NO"}
        )

    def _no2():
        """*NO₂ — N-down, bent (O–N–O ≈ 115°)."""
        half = np.radians(115.0 / 2)
        d = 1.20
        return _create_adsorbate_with_dummy(
            ["N", "O", "O"],
            [
                [0.00, 0.00, 0.00],
                [d * np.sin(half), 0, d * np.cos(half)],
                [-d * np.sin(half), 0, d * np.cos(half)],
            ],
            properties={"adsorbate": "*NO2"}
        )

    def _no3():
        """*NO₃ — N-down, planar nitrate (D₃ₕ, N–O = 1.24 Å) slightly tilted up."""
        d = 1.24
        h = 0.40
        return _create_adsorbate_with_dummy(
            ["N", "O", "O", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, d, h],
                [d * np.sin(np.radians(120)), -d * 0.5, h],
                [-d * np.sin(np.radians(120)), -d * 0.5, h],
            ],
            properties={"adsorbate": "*NO3"}
        )

    def _noh():
        """*NOH — N-down, O–H bond (first H on O)."""
        return _create_adsorbate_with_dummy(
            ["N", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.20],
                [0.90, 0.00, 1.70],
            ],
            properties={"adsorbate": "*NOH"}
        )

    def _hno():
        """*HNO — N-down, N–H bond (first H on N)."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.01, 0.42],
                [1.10, 0.00, 0.75],
            ],
            properties={"adsorbate": "*HNO"}
        )

    def _n2o2():
        """*ONNO (cis-hyponitrite dimer, N-down) — O=N–N=O bridge species."""
        return _create_adsorbate_with_dummy(
            ["N", "O", "N", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.18],
                [1.30, 0.00, 0.00],
                [1.30, 0.00, 1.18],
            ],
            properties={"adsorbate": "*ONNO"}
        )

    def _n2o():
        """*N₂O — N-down, linear (N≡N–O, terminal N binds)."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "O"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.13],
                [0.00, 0.00, 2.27],
            ],
            properties={"adsorbate": "*N2O"}
        )

    def _n_ads():
        """*N — atomic nitrogen (hollow site preferred)."""
        return _create_adsorbate_with_dummy(
            ["N"],
            [[0, 0, 0]],
            properties={"adsorbate": "*N"}
        )

    def _nh():
        """*NH — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H"],
            [[0, 0, 0], [0, 0, 1.01]],
            properties={"adsorbate": "*NH"}
        )

    def _nh2():
        """*NH₂ — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.82, 0.00, 0.56],
                [-0.82, 0.00, 0.56],
            ],
            properties={"adsorbate": "*NH2"}
        )

    def _nh3():
        """*NH₃ — N-down (tetrahedral)."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.94, 0.34],
                [0.82, -0.47, 0.34],
                [-0.82, -0.47, 0.34],
            ],
            properties={"adsorbate": "*NH3"}
        )

    def _nhoh():
        """*NHOH — N-down, both N–H and O–H bonds present."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.95, 0.45],
                [1.25, 0.00, 0.65],
                [1.85, 0.00, 1.40],
            ],
            properties={"adsorbate": "*NHOH"}
        )

    def _nh2oh():
        """*NH₂OH (hydroxylamine) — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.95, 0.45],
                [-0.95, -0.35, 0.45],
                [1.22, 0.00, 0.75],
                [1.82, 0.00, 1.50],
            ],
            properties={"adsorbate": "*NH2OH"}
        )

    def _o_ads():
        """*O — atomic oxygen."""
        return _create_adsorbate_with_dummy(
            ["O"],
            [[0, 0, 0]],
            properties={"adsorbate": "*O"}
        )

    def _oh():
        """*OH — O-down."""
        return _create_adsorbate_with_dummy(
            ["O", "H"],
            [[0, 0, 0], [0, 0, 0.97]],
            properties={"adsorbate": "*OH"}
        )

    def _h2o():
        """*H₂O — O-down (physisorbed; typically desorbs above 200 K)."""
        return _create_adsorbate_with_dummy(
            ["O", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.76, 0.00, 0.59],
                [-0.76, 0.00, 0.59],
            ],
            properties={"adsorbate": "*H2O"}
        )

    # Registry
    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "NO": _no,
        "NO2": _no2,
        "NO3": _no3,
        "NOH": _noh,
        "HNO": _hno,
        "N2O2": _n2o2,
        "N2O": _n2o,
        "N": _n_ads,
        "NH": _nh,
        "NH2": _nh2,
        "NH3": _nh3,
        "NHOH": _nhoh,
        "NH2OH": _nh2oh,
        "O": _o_ads,
        "OH": _oh,
        "H2O": _h2o,
    }

    def get_noxrr_adsorbate(name: str) -> Molecule:
        """Return a fresh copy of the named NOx-reduction adsorbate.

        Args:
            name: Key from the adsorbate registry.
        Returns:
            Pymatgen Molecule with X at index 0
        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown NOx adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    # =========================================================================
    # Reaction pathway data (using global ReactionPathway and ReactionStep)
    # =========================================================================
    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    # 1 ── Dissociative NO reduction → N₂ (Ru, Rh, Ir catalytic)
    _reg(ReactionPathway(
        name="no_dissociative",
        description="Dissociative NO reduction to N₂ via N coupling",
        overall_reaction="2 NO → N₂ + 2 O*",
        selectivity_metals=["Ru", "Rh", "Ir", "Pd"],
        steps=[
            ReactionStep(
                reactant="NO", product="N",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="N–O bond scission; rate-limiting on Rh/Pd",
            ),
            ReactionStep(
                reactant="N", product="N2O2",
                step_type="chemical", electrons=0, protons=0,
                notes="N + N Langmuir–Hinshelwood coupling (2 *N sites)",
            ),
            ReactionStep(
                reactant="N2O2", product="N",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["N2"],
                notes="N₂ desorption",
            ),
        ],
    ))

    # 2 ── Electrochemical NO → NH₃ via NOH route (N–O bond breaks early)
    _reg(ReactionPathway(
        name="no_to_nh3_noh",
        description="Electrochemical NO reduction to NH₃ via *NOH → *N pathway",
        overall_reaction="NO + 5(H+ + e-) → NH3 + H2O",
        selectivity_metals=["Cu", "Fe", "Co"],
        steps=[
            ReactionStep(
                reactant="NO", product="NOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O protonation; competing with *HNO on many metals",
            ),
            ReactionStep(
                reactant="NOH", product="N",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="N–O cleavage after O protonation",
            ),
            ReactionStep(
                reactant="N", product="NH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="N",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
                notes="NH₃ desorption",
            ),
        ],
    ))

    # 3 ── Electrochemical NO → NH₃ via hydroxylamine route (*NHOH → *NH₂OH)
    _reg(ReactionPathway(
        name="no_to_nh3_nhoh",
        description="Electrochemical NO reduction to NH₃ via hydroxylamine (*NH₂OH)",
        overall_reaction="NO + 5(H+ + e-) → NH3 + H2O",
        selectivity_metals=["Cu", "Pt", "Ag"],
        steps=[
            ReactionStep(
                reactant="NO", product="NOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O protonation; first step common to both routes",
            ),
            ReactionStep(
                reactant="NOH", product="NHOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="N protonation giving N,O-dihydroxyl; potential-limiting on Cu",
            ),
            ReactionStep(
                reactant="NHOH", product="NH2OH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="N protonation to hydroxylamine",
            ),
            ReactionStep(
                reactant="NH2OH", product="NH2",
                step_type="chemical", electrons=0, protons=0,
                released=["H2O"],
                notes="N–O bond cleavage / H₂O elimination",
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="N",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
                notes="NH₃ desorption",
            ),
        ],
    ))

    # 4 ── Catalytic NO → N₂O (automotive, Pt/Pd/Rh three-way catalyst)
    _reg(ReactionPathway(
        name="no_to_n2o",
        description="Catalytic NO dimerisation to N₂O (Langmuir–Hinshelwood)",
        overall_reaction="2 NO → N₂O + O*",
        selectivity_metals=["Pt", "Pd", "Rh"],
        steps=[
            ReactionStep(
                reactant="NO", product="N2O2",
                step_type="chemical", electrons=0, protons=0,
                notes="2*NO → *ONNO dimer; kinetically facile on Pt(111)",
            ),
            ReactionStep(
                reactant="N2O2", product="N2O",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="asymmetric N–O cleavage of dimer leaving *N₂O + *O",
            ),
            ReactionStep(
                reactant="N2O", product="NO",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["N2O"],
                notes="N₂O desorption (or further reduction on Rh)",
            ),
        ],
    ))

    # 5 ── NO₂ → NO (prereduction / first step of NO₂ reduction)
    _reg(ReactionPathway(
        name="no2_to_no",
        description="NO₂ reduction to NO (dissociative O removal)",
        overall_reaction="NO₂ + 2(H+ + e-) → NO + H₂O",
        selectivity_metals=["Pt", "Cu", "Ru", "Fe"],
        steps=[
            ReactionStep(
                reactant="NO2", product="NO",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="N–O scission; *O then removed by H (next steps) or stays",
            ),
            ReactionStep(
                reactant="O", product="OH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="*O hydrogenation",
            ),
            ReactionStep(
                reactant="OH", product="NO",  # bare surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="*OH hydrogenation → H₂O desorption",
            ),
        ],
    ))

    # 6 ── eNO3RR: NO₃⁻ → NH₃ (full electrochemical, Cu selectivity)
    _reg(ReactionPathway(
        name="no3_to_nh3",
        description="Electrochemical NO₃⁻ reduction to NH₃ (eNO3RR) via NOH route",
        overall_reaction="NO3- + 9(H+ + e-) → NH3 + 3 H2O",
        selectivity_metals=["Cu", "Fe", "Co", "Ru"],
        steps=[
            ReactionStep(
                reactant="NO3", product="NO2",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="first reduction; often fast on Cu",
            ),
            ReactionStep(
                reactant="NO2", product="NO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="NO", product="NOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="potential-limiting step on many metals",
            ),
            ReactionStep(
                reactant="NOH", product="N",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="N", product="NH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="NO3",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
            ),
        ],
    ))

    # 7 ── eNO3RR: NO₃⁻ → N₂ (dissociative, Ru selectivity)
    _reg(ReactionPathway(
        name="no3_to_n2",
        description="Electrochemical NO₃⁻ reduction to N₂ (dissociative pathway, Ru)",
        overall_reaction="2 NO3- + 12(H+ + e-) → N2 + 6 H2O",
        selectivity_metals=["Ru", "Rh", "Ir"],
        steps=[
            ReactionStep(
                reactant="NO3", product="NO2",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="NO2", product="NO",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
            ),
            ReactionStep(
                reactant="NO", product="N",
                step_type="chemical", electrons=0, protons=0,
                released=["O"],
                notes="N–O bond scission; rate-limiting on Ru",
            ),
            ReactionStep(
                reactant="N", product="N2O2",
                step_type="chemical", electrons=0, protons=0,
                notes="N + N coupling (2 sites)",
            ),
            ReactionStep(
                reactant="N2O2", product="NO3",  # bare surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["N2"],
            ),
        ],
    ))

    def get_noxrr_pathway(name: str) -> ReactionPathway:
        """Return the named NOxRR :class:`ReactionPathway`.
        Args:
            name: One of the NOxRR pathway keys.
        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown NOxRR pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    pathway = get_noxrr_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_noxrr_adsorbate(name)

    return pathway, adsorbates


def generate_cer_adsorbates(pathway_name: str) -> tuple:
    """Chlorine evolution reaction pathways on metal / oxide surfaces.

    Provides a literature-grounded library of CER intermediates and reaction
    pathways (CHE model with the Cl⁻/Cl₂ couple as reference, E° = 1.36 V vs SHE
    in 1 M HCl) that can be placed on *any* surface slab.

    Pathways implemented
    --------------------
    - 'volmer_tafel'      : * + Cl⁻ → *Cl, then 2*Cl → Cl₂(g)      (metallic Pt, Ru)
    - 'volmer_heyrovsky'  : * + Cl⁻ → *Cl, then *Cl + Cl⁻ → Cl₂(g) (RuO₂, IrO₂)
    - 'krishtalik'        : *O + Cl⁻ → *OCl, then *OCl + Cl⁻ → Cl₂(g) + *O
                            (O-covered oxide route on RuO₂(110), Co₃O₄, …)

    All elementary steps consume Cl⁻ rather than H⁺; the CHE-equivalent reference
    is the Cl⁻/Cl₂ couple. Each pathway is two electrons total to balance
    2 Cl⁻ → Cl₂ + 2 e⁻ (Tafel is the chemical recombination step and carries no
    electron).

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and
        adsorbates_dict maps intermediate names to Molecule objects.

    Raises
    ------
    KeyError
        If pathway_name is not registered.
    """

    # =========================================================================
    # Adsorbate geometry library (Pymatgen version with DummySpecies)
    # =========================================================================

    def _cl():
        """*Cl — atop, atomic chlorine (Volmer/Heyrovsky intermediate)."""
        return _create_adsorbate_with_dummy(
            ["Cl"],
            [[0, 0, 0]],
            properties={"adsorbate": "*Cl"},
        )

    def _ocl():
        """*OCl — O-down, Cl pointing up (Krishtalik intermediate on oxide)."""
        return _create_adsorbate_with_dummy(
            ["O", "Cl"],
            [[0.00, 0.00, 0.00], [0.00, 0.00, 1.70]],
            properties={"adsorbate": "*OCl"},
        )

    def _o_ads():
        """*O — atomic oxygen (Krishtalik starting state on O-covered oxide)."""
        return _create_adsorbate_with_dummy(
            ["O"],
            [[0, 0, 0]],
            properties={"adsorbate": "*O"},
        )

    # Registry
    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "Cl": _cl,
        "OCl": _ocl,
        "O": _o_ads,
    }

    def get_cer_adsorbate(name: str) -> Molecule:
        """Return a fresh copy of the named CER adsorbate.

        Args:
            name: Key from the adsorbate registry.
        Returns:
            Pymatgen Molecule with X at index 0.
        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown CER adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    # =========================================================================
    # Reaction pathway data
    # =========================================================================
    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    # 1 ── Volmer + Tafel: electrochemical adsorption + chemical recombination
    _reg(ReactionPathway(
        name="volmer_tafel",
        description="CER via Volmer adsorption and Tafel recombination of two *Cl",
        overall_reaction="2 Cl- → Cl2 + 2 e-",
        selectivity_metals=["Pt", "Ru", "Pd"],
        steps=[
            ReactionStep(
                reactant="Cl", product="Cl",
                step_type="electrochemical", electrons=-1, protons=0,
                notes="Volmer: * + Cl- → *Cl + e- (CHE w.r.t. Cl-/Cl2 couple)",
            ),
            ReactionStep(
                reactant="Cl", product="Cl",  # bare-surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["Cl2"],
                notes="Tafel: 2 *Cl → Cl2(g) + 2*; recombination is chemical "
                      "(no electron transfer)",
            ),
        ],
    ))

    # 2 ── Volmer + Heyrovsky: electrochemical adsorption + electrochemical desorption
    _reg(ReactionPathway(
        name="volmer_heyrovsky",
        description="CER via Volmer adsorption and Heyrovsky electrochemical desorption",
        overall_reaction="2 Cl- → Cl2 + 2 e-",
        selectivity_metals=["RuO2", "IrO2", "Ru", "Ir"],
        steps=[
            ReactionStep(
                reactant="Cl", product="Cl",
                step_type="electrochemical", electrons=-1, protons=0,
                notes="Volmer: * + Cl- → *Cl + e-",
            ),
            ReactionStep(
                reactant="Cl", product="Cl",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=0,
                released=["Cl2"],
                notes="Heyrovsky: *Cl + Cl- → Cl2(g) + * + e-; "
                      "potential-limiting on RuO2(110) / IrO2(110)",
            ),
        ],
    ))

    # 3 ── Krishtalik: oxide route via *OCl intermediate (O-covered active site)
    _reg(ReactionPathway(
        name="krishtalik",
        description="CER on O-covered oxide surface via *OCl intermediate",
        overall_reaction="2 Cl- → Cl2 + 2 e- (active site stays *O between turnovers)",
        selectivity_metals=["RuO2", "IrO2", "Co3O4"],
        steps=[
            ReactionStep(
                reactant="O", product="OCl",
                step_type="electrochemical", electrons=-1, protons=0,
                notes="Volmer-like: *O + Cl- → *OCl + e- (Cl attaches to surface O)",
            ),
            ReactionStep(
                reactant="OCl", product="O",  # active site regenerated
                step_type="electrochemical", electrons=-1, protons=0,
                released=["Cl2"],
                notes="Heyrovsky-like: *OCl + Cl- → Cl2(g) + *O + e-; "
                      "active site returns to *O, ready for next turnover",
            ),
        ],
    ))

    def get_cer_pathway(name: str) -> ReactionPathway:
        """Return the named CER :class:`ReactionPathway`.

        Args:
            name: One of the CER pathway keys.
        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown CER pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    pathway = get_cer_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_cer_adsorbate(name)

    return pathway, adsorbates


def generate_her_adsorbates(pathway_name: str) -> tuple:
    """Hydrogen evolution reaction pathways on metal surfaces.

    The single-intermediate workhorse of (photo)electrocatalysis: H* is the
    sole surface species, and its binding strength sets the HER activity
    (Sabatier volcano, |ΔG_H*| ≈ 0 at the optimum).

    Pathways implemented
    --------------------
    - 'volmer_tafel'     : * + H⁺ + e⁻ → *H, then 2*H → H₂(g) + 2*  (Pt, MoS₂)
    - 'volmer_heyrovsky' : * + H⁺ + e⁻ → *H, then *H + H⁺ + e⁻ → H₂(g) + *

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Nørskov et al. *J. Electrochem. Soc.* **152**, J23 (2005).
    Greeley et al. *Nat. Mater.* **5**, 909 (2006).
    Conway & Tilak *Electrochim. Acta* **47**, 3571 (2002).
    Skúlason et al. *J. Phys. Chem. C* **114**, 18182 (2010).
    """

    def _h():
        """*H — single hydrogen (hollow site preferred on most metals)."""
        return _create_adsorbate_with_dummy(
            ["H"],
            [[0, 0, 0]],
            properties={"adsorbate": "*H"},
        )

    _ADSORBATE_REGISTRY: Dict[str, Callable] = {"H": _h}

    def get_her_adsorbate(name: str) -> Molecule:
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown HER adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    _reg(ReactionPathway(
        name="volmer_tafel",
        description="HER via Volmer adsorption and Tafel recombination",
        overall_reaction="2 H+ + 2 e- → H2",
        selectivity_metals=["Pt", "Pd", "Ir"],
        steps=[
            ReactionStep(
                reactant="H", product="H",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="Volmer: * + H+ + e- → *H",
            ),
            ReactionStep(
                reactant="H", product="H",  # bare-surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["H2"],
                notes="Tafel: 2 *H → H2(g) + 2*; chemical recombination",
            ),
        ],
    ))

    _reg(ReactionPathway(
        name="volmer_heyrovsky",
        description="HER via Volmer adsorption and Heyrovsky electrochemical desorption",
        overall_reaction="2 H+ + 2 e- → H2",
        selectivity_metals=["MoS2", "Ni-Mo", "Au", "Ag"],
        steps=[
            ReactionStep(
                reactant="H", product="H",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="Volmer: * + H+ + e- → *H",
            ),
            ReactionStep(
                reactant="H", product="H",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2"],
                notes="Heyrovsky: *H + H+ + e- → H2(g) + *",
            ),
        ],
    ))

    def get_her_pathway(name: str) -> ReactionPathway:
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown HER pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    pathway = get_her_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_her_adsorbate(name)

    return pathway, adsorbates


def generate_orr_adsorbates(pathway_name: str) -> tuple:
    """Oxygen reduction reaction pathways on metal / oxide surfaces.

    The cathodic counterpart of OER. Same scaling relations between *OH and
    *OOH (≈ 3.2 eV gap) constrain the maximum 4e⁻ ORR activity to a
    theoretical overpotential of ~0.3–0.4 V.

    Pathways implemented
    --------------------
    - '4e_associative'  : O₂ → *O₂ → *OOH → *O → *OH → H₂O           (Pt, Pd)
    - '4e_dissociative' : O₂ → 2*O → *OH → H₂O                       (Pt(111) at low T)
    - '2e_to_h2o2'      : O₂ → *O₂ → *OOH → H₂O₂(aq)                 (Au, Hg, single-atom Co/N–C)

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Nørskov et al. *J. Phys. Chem. B* **108**, 17886 (2004).
    Stamenkovic et al. *Nat. Mater.* **6**, 241 (2007).
    Greeley et al. *Nat. Chem.* **1**, 552 (2009).
    Siahrostami et al. *Nat. Mater.* **12**, 1137 (2013).
    Kulkarni et al. *Chem. Rev.* **118**, 2302 (2018).
    """

    def _o2():
        """*O₂ — end-on (Pauling, superoxide-like, O-O = 1.30 Å)."""
        return _create_adsorbate_with_dummy(
            ["O", "O"],
            [[0.00, 0.00, 0.00], [0.00, 0.00, 1.30]],
            properties={"adsorbate": "*O2"},
        )

    def _ooh():
        """*OOH — O-down, peroxyl. Geometry matches the OER intermediate."""
        return _create_adsorbate_with_dummy(
            ["O", "O", "H"],
            [
                [0.00, 0.00, 0.00],
                [1.20, -0.20, 0.80],
                [1.94, -0.20, 1.32],
            ],
            properties={"adsorbate": "*OOH"},
        )

    def _o_ads():
        """*O — atomic oxygen (hollow / bridge preferred)."""
        return _create_adsorbate_with_dummy(
            ["O"],
            [[0, 0, 0]],
            properties={"adsorbate": "*O"},
        )

    def _oh():
        """*OH — O-down."""
        return _create_adsorbate_with_dummy(
            ["O", "H"],
            [[0, 0, 0], [0, 0, 0.97]],
            properties={"adsorbate": "*OH"},
        )

    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "O2_ads": _o2,
        "OOH": _ooh,
        "O": _o_ads,
        "OH": _oh,
    }

    def get_orr_adsorbate(name: str) -> Molecule:
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown ORR adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    # 1 ── 4e⁻ associative (Nørskov, dominant on Pt-group metals)
    _reg(ReactionPathway(
        name="4e_associative",
        description="4e⁻ ORR via *O2 → *OOH → *O → *OH → H2O (associative)",
        overall_reaction="O2 + 4(H+ + e-) → 2 H2O",
        selectivity_metals=["Pt", "Pd", "Pt3Ni", "Ir"],
        steps=[
            ReactionStep(
                reactant="O2_ads", product="OOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O2 protonation; first electron transfer",
            ),
            ReactionStep(
                reactant="OOH", product="O",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="O-O cleavage upon protonation; H2O released",
            ),
            ReactionStep(
                reactant="O", product="OH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="*O protonation; *OH-vs-*OOH scaling sets the overpotential",
            ),
            ReactionStep(
                reactant="OH", product="O2_ads",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="*OH protonation; H2O desorbs and site returns to bare",
            ),
        ],
    ))

    # 2 ── 4e⁻ dissociative (direct O-O scission first, then protonation)
    _reg(ReactionPathway(
        name="4e_dissociative",
        description="4e⁻ ORR via O2 dissociation to 2*O then protonation",
        overall_reaction="O2 + 4(H+ + e-) → 2 H2O",
        selectivity_metals=["Pt(111)", "Ru", "Rh"],
        steps=[
            ReactionStep(
                reactant="O2_ads", product="O",
                step_type="chemical", electrons=0, protons=0,
                notes="O-O scission to 2 *O (single-site bookkeeping)",
            ),
            ReactionStep(
                reactant="O", product="OH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="*O protonation",
            ),
            ReactionStep(
                reactant="OH", product="O2_ads",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O"],
                notes="*OH protonation → H2O desorption",
            ),
        ],
    ))

    # 3 ── 2e⁻ ORR to hydrogen peroxide (Au, Hg, single-atom catalysts)
    _reg(ReactionPathway(
        name="2e_to_h2o2",
        description="2e⁻ ORR producing H2O2 via *OOH (no O-O bond cleavage)",
        overall_reaction="O2 + 2(H+ + e-) → H2O2",
        selectivity_metals=["Au", "Hg", "Co-N4", "Fe-N4"],
        steps=[
            ReactionStep(
                reactant="O2_ads", product="OOH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="O2 protonation; first electron transfer",
            ),
            ReactionStep(
                reactant="OOH", product="O2_ads",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["H2O2"],
                notes="*OOH protonation; H2O2 desorbs without O-O cleavage",
            ),
        ],
    ))

    def get_orr_pathway(name: str) -> ReactionPathway:
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown ORR pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    pathway = get_orr_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_orr_adsorbate(name)

    return pathway, adsorbates


def generate_nrr_adsorbates(pathway_name: str) -> tuple:
    """Nitrogen reduction reaction pathways on metal / single-atom surfaces.

    Electrochemical analog of Haber–Bosch. Activity is strongly limited by
    competition with HER and by the weakness of *N2 binding on most metals.

    Pathways implemented
    --------------------
    - 'distal'        : *N2 → *NNH → *NNH₂ → *N + NH₃ → *NH → *NH₂ → NH₃
                        (one N hydrogenated and released first, then the other)
    - 'alternating'   : *N2 → *NNH → *NHNH → *NHNH₂ → *NH₂NH₂ → 2 NH₃
                        (H atoms alternate between the two N atoms)
    - 'dissociative'  : N2 → 2*N → 2*NH → 2*NH₂ → 2 NH₃
                        (Haber–Bosch-like; needs very strong *N binding, rare
                        electrochemically)

    Returns
    -------
    tuple
        (pathway, adsorbates_dict) where pathway is a ReactionPathway and
        adsorbates_dict maps intermediate names to Molecule objects.

    References
    ----------
    Skúlason et al. *Phys. Chem. Chem. Phys.* **14**, 1235 (2012).
    Montoya et al. *ChemSusChem* **8**, 2180 (2015).
    Singh et al. *ACS Catal.* **7**, 706 (2017).
    Andersen et al. *Nature* **570**, 504 (2019).
    Suryanto et al. *Nat. Catal.* **2**, 290 (2019).
    """

    def _n2_ads():
        """*N2 — end-on (η¹), N-N ≈ 1.13 Å (slightly elongated from gas-phase 1.10 Å)."""
        return _create_adsorbate_with_dummy(
            ["N", "N"],
            [[0.00, 0.00, 0.00], [0.00, 0.00, 1.13]],
            properties={"adsorbate": "*N2"},
        )

    def _nnh():
        """*NNH — end-on, terminal H on far N (diazenyl)."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.20],
                [0.90, 0.00, 1.70],
            ],
            properties={"adsorbate": "*NNH"},
        )

    def _nnh2():
        """*NNH2 — end-on, two H on the distal N (hydrazido)."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.25],
                [0.95, 0.00, 1.65],
                [-0.95, 0.00, 1.65],
            ],
            properties={"adsorbate": "*NNH2"},
        )

    def _nhnh():
        """*NHNH — alternating, one H on each N (trans-diazene-like)."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.30],
                [0.95, 0.00, -0.50],
                [-0.95, 0.00, 1.80],
            ],
            properties={"adsorbate": "*NHNH"},
        )

    def _nhnh2():
        """*NHNH2 — alternating, one H on near N, two H on distal N."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.40],
                [0.95, 0.00, -0.45],
                [0.95, 0.00, 1.85],
                [-0.95, 0.00, 1.85],
            ],
            properties={"adsorbate": "*NHNH2"},
        )

    def _n2h4():
        """*N2H4 (hydrazine adsorbed, near-N atop) — two H on each N, gauche."""
        return _create_adsorbate_with_dummy(
            ["N", "N", "H", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.45],
                [0.94, 0.00, -0.40],
                [-0.94, 0.00, -0.40],
                [0.94, 0.00, 1.85],
                [-0.94, 0.00, 1.85],
            ],
            properties={"adsorbate": "*N2H4"},
        )

    def _n_ads():
        """*N — atomic nitrogen (hollow preferred)."""
        return _create_adsorbate_with_dummy(
            ["N"],
            [[0, 0, 0]],
            properties={"adsorbate": "*N"},
        )

    def _nh():
        """*NH — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H"],
            [[0, 0, 0], [0, 0, 1.01]],
            properties={"adsorbate": "*NH"},
        )

    def _nh2():
        """*NH2 — N-down."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.82, 0.00, 0.56],
                [-0.82, 0.00, 0.56],
            ],
            properties={"adsorbate": "*NH2"},
        )

    def _nh3():
        """*NH3 — N-down (tetrahedral)."""
        return _create_adsorbate_with_dummy(
            ["N", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.94, 0.34],
                [0.82, -0.47, 0.34],
                [-0.82, -0.47, 0.34],
            ],
            properties={"adsorbate": "*NH3"},
        )

    _ADSORBATE_REGISTRY: Dict[str, Callable] = {
        "N2_ads": _n2_ads,
        "NNH": _nnh,
        "NNH2": _nnh2,
        "NHNH": _nhnh,
        "NHNH2": _nhnh2,
        "N2H4": _n2h4,
        "N": _n_ads,
        "NH": _nh,
        "NH2": _nh2,
        "NH3": _nh3,
    }

    def get_nrr_adsorbate(name: str) -> Molecule:
        if name not in _ADSORBATE_REGISTRY:
            raise KeyError(
                f"Unknown NRR adsorbate '{name}'. "
                f"Available: {sorted(_ADSORBATE_REGISTRY)}"
            )
        return _ADSORBATE_REGISTRY[name]()

    _PATHWAYS: dict[str, ReactionPathway] = {}

    def _reg(pw: ReactionPathway) -> ReactionPathway:
        _PATHWAYS[pw.name] = pw
        return pw

    # 1 ── Distal mechanism (one N hydrogenated and released first)
    _reg(ReactionPathway(
        name="distal",
        description="NRR via distal mechanism: distal N fully hydrogenated and released as NH3 first",
        overall_reaction="N2 + 6(H+ + e-) → 2 NH3",
        selectivity_metals=["Ru", "Mo", "Re", "Mo-N3"],
        steps=[
            ReactionStep(
                reactant="N2_ads", product="NNH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="N2 protonation; often potential-limiting",
            ),
            ReactionStep(
                reactant="NNH", product="NNH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NNH2", product="N",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["NH3"],
                notes="N-N bond cleavage; first NH3 desorbs, *N remains",
            ),
            ReactionStep(
                reactant="N", product="NH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH2", product="N2_ads",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["NH3"],
                notes="second NH3 desorbs; site available for next N2",
            ),
        ],
    ))

    # 2 ── Alternating mechanism (H alternates between the two N atoms)
    _reg(ReactionPathway(
        name="alternating",
        description="NRR via alternating mechanism: H atoms alternate between the two N atoms",
        overall_reaction="N2 + 6(H+ + e-) → 2 NH3",
        selectivity_metals=["Fe", "Co", "Fe-N4"],
        steps=[
            ReactionStep(
                reactant="N2_ads", product="NNH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NNH", product="NHNH",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="next H goes to the near N rather than distal",
            ),
            ReactionStep(
                reactant="NHNH", product="NHNH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NHNH2", product="N2H4",
                step_type="electrochemical", electrons=-1, protons=1,
                notes="fourth proton; surface-bound hydrazine *N2H4",
            ),
            ReactionStep(
                reactant="N2H4", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
                released=["NH3"],
                notes="N-N cleavage with first NH3 release",
            ),
            ReactionStep(
                reactant="NH2", product="NH3",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH3", product="N2_ads",  # bare-surface marker
                step_type="chemical", electrons=0, protons=0,
                released=["NH3"],
                notes="second NH3 desorbs",
            ),
        ],
    ))

    # 3 ── Dissociative (Haber-Bosch-like, needs strong *N binding)
    _reg(ReactionPathway(
        name="dissociative",
        description="NRR via initial N2 dissociation then sequential protonation of *N",
        overall_reaction="N2 + 6(H+ + e-) → 2 NH3",
        selectivity_metals=["Fe", "Ru", "Os"],
        steps=[
            ReactionStep(
                reactant="N2_ads", product="N",
                step_type="chemical", electrons=0, protons=0,
                notes="N-N bond scission; very high activation barrier",
            ),
            ReactionStep(
                reactant="N", product="NH",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH", product="NH2",
                step_type="electrochemical", electrons=-1, protons=1,
            ),
            ReactionStep(
                reactant="NH2", product="N2_ads",  # bare-surface marker
                step_type="electrochemical", electrons=-1, protons=1,
                released=["NH3"],
                notes="single-site bookkeeping; full 2-NH3 turnover spans two sites",
            ),
        ],
    ))

    def get_nrr_pathway(name: str) -> ReactionPathway:
        if name not in _PATHWAYS:
            raise KeyError(
                f"Unknown NRR pathway '{name}'. "
                f"Available: {sorted(_PATHWAYS)}"
            )
        return _PATHWAYS[name]

    pathway = get_nrr_pathway(pathway_name)
    adsorbates = {}

    for name in pathway.intermediates:
        if name in _ADSORBATE_REGISTRY:
            adsorbates[name] = get_nrr_adsorbate(name)

    return pathway, adsorbates


def get_multipliers(slab_pmg):
    a = slab_pmg.lattice.a
    b = slab_pmg.lattice.b
    if a > b:
        return [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 2, 1)]
    return [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 1)]


def generate_adsorbed_structures(reaction: str, pathway_name: str = "") -> dict:
    """Generate surface + adsorbate structures for specified reaction.

    Parameters
    ----------
    reaction : str
        Reaction type: 'OER', 'CO2RR', 'NOXRR', 'CER', 'HER', 'ORR', or 'NRR'.
    pathway_name : str, optional
        Pathway name (required for everything except OER).

    Returns
    -------
    dict
        Dict of adsorption sets, where each set is a list of ASE Atoms objects
        with site_type, ads_coord, and adsorbate info in the Atoms.info dict.

    Raises
    ------
    ValueError
        If reaction type is unknown or pathway_name missing for certain reactions.
    FileNotFoundError
        If input_structures.json is not found.
    """
    with open('input_structures.json', 'r') as f:
        data = json.load(f)

    slab_pmg = Slab.from_dict(data[0])
    slab_pmg.sort()

    # Generate adsorbates based on reaction type and pathway
    pathway_obj = None
    if reaction == "OER":
        adsorbates = generate_oer_adsorbates()
    elif reaction == "CO2RR":
        pathway_obj, ads_dict = generate_co2rr_adsorbates(pathway_name)
        adsorbates = list(ads_dict.values())
    elif reaction == "NOXRR":
        pathway_obj, ads_dict = generate_noxrr_adsorbates(pathway_name)
        adsorbates = list(ads_dict.values())
    elif reaction == "CER":
        pathway_obj, ads_dict = generate_cer_adsorbates(pathway_name)
        adsorbates = list(ads_dict.values())
    elif reaction == "HER":
        pathway_obj, ads_dict = generate_her_adsorbates(pathway_name)
        adsorbates = list(ads_dict.values())
    elif reaction == "ORR":
        pathway_obj, ads_dict = generate_orr_adsorbates(pathway_name)
        adsorbates = list(ads_dict.values())
    elif reaction == "NRR":
        pathway_obj, ads_dict = generate_nrr_adsorbates(pathway_name)
        adsorbates = list(ads_dict.values())
    else:
        raise ValueError(
            f"Unknown reaction: {reaction}. "
            f"Expected one of: OER, CO2RR, NOXRR, CER, HER, ORR, NRR")

    # Build the gas-phase reference set this pathway actually needs. These are
    # returned alongside `adsorption_sets` and relaxed once at the top of
    # `run_relaxation`, then attached per-set just like the clean slab.
    gas_refs = {}
    for ref_name in _pathway_required_refs(reaction, pathway_obj):
        atoms = pmg_to_ase(_GAS_REF_REGISTRY[ref_name])
        atoms.info['adsorbate'] = ref_name
        gas_refs[ref_name] = atoms

    # Get adsorption sites from slab
    sites_dict, asf = get_adsorption_sites(slab_pmg)

    # Build slab + adsorbate
    site_types = ['ontop', 'bridge', 'hollow']
    adsorption_sets = {}
    # site counts for the per-surface diagnostics: every (repeat, site) tried,
    # and those dropped because an adsorbate could not be placed without overlap
    n_sites = 0
    n_placement_failed = 0
    multipliers = [(1,1,1), (1,2,1)] #get_multipliers(slab_pmg) #TODO only for test
    for idx, repeat in enumerate(multipliers):
        clean_slab_pmg = asf.slab.copy()
        clean_slab_pmg.make_supercell(repeat)
        clean_slab = pmg_to_ase(clean_slab_pmg)
        clean_slab.info['adsorbate'] = "*"
        adsorption_sets[idx] = {"clean_slab": clean_slab, "adsorb_set": []}
        for site_type in site_types:
            sites = sites_dict.get(site_type, [])
            for ads_coord in sites:
                adsorb_set = {
                    "site_type": site_type,
                    "ads_coord": ads_coord,
                    "repeat": repeat,
                    "structures": []
                }

                for ads in adsorbates:
                    ads_slab = asf.add_adsorbate(ads, ads_coord, repeat=repeat, translate=False, reorient=True)
                    ads_slab.remove_species("X")

                    ase_struct = pmg_to_ase(ads_slab)
                    ase_struct.info['adsorbate'] = ads.properties['adsorbate']

                    if not has_reasonable_distances(ase_struct):
                        break
                    adsorb_set["structures"].append(ase_struct)

                # Gas-phase references are computed once in run_relaxation and
                # attached per-set there, so per-site duplication is not needed.
                n_sites += 1
                if len(adsorb_set["structures"]) == len(adsorbates):
                    adsorption_sets[idx]["adsorb_set"].append(adsorb_set)
                else:
                    n_placement_failed += 1

    # Reference intramolecular bond graphs, keyed by adsorbate name, for the
    # post-relaxation identity check in run_relaxation (see docs/sanity_checks.md).
    expected_graphs = {
        ads.properties['adsorbate']: _adsorbate_reference_graph(ads)
        for ads in adsorbates
    }

    return {"sets": adsorption_sets, "gas_refs": gas_refs,
            "expected_graphs": expected_graphs,
            "placement": {"n_sites": n_sites, "n_placement_failed": n_placement_failed}}


def run_relaxation(ml_model: str, calc, fmax: float, max_steps: int,
                   reaction: str, pathway: str,
                   validate_adsorbates: bool = True,
                   check_slab_integrity: bool = False,
                   check_energy_outliers: bool = False,
                   bind_tol: float = 1.25,
                   graph_tol: float = _ADSORBATE_BOND_TOL,
                   slab_max_disp: float = 1.5,
                   energy_mad_factor: float = 5.0,
                   **relaxation_kwargs) -> dict:
    """Run geometry relaxation on adsorbate structures.

    Parameters
    ----------
    ml_model : str
        ML model name (used for output key naming).
    calc : Calculator
        ASE calculator object (MACE, UPET, MatterSim).
    fmax : float
        Force convergence criterion (eV/Å).
    max_steps : int
        Maximum relaxation steps.
    reaction : str
        Reaction type: 'OER', 'CO2RR', 'NOXRR', 'CER', 'HER', 'ORR', or 'NRR'.
    pathway : str
        Pathway name (required for everything except OER, which takes "").
    validate_adsorbates : bool, optional
        Run the default post-relaxation sanity checks (layers 0-2: finite
        energy, atom overlap, surface binding, molecular identity). A relaxed
        adsorbate that fails is treated exactly like a relaxation failure and
        its set is dropped; the reason is recorded in rejected.json. Default True.
    check_slab_integrity : bool, optional
        Opt-in layer 3: also reject a set if its slab reconstructs (max slab
        displacement vs the clean relaxed slab exceeds ``slab_max_disp``).
    check_energy_outliers : bool, optional
        Opt-in layer 4: after relaxation, drop sets whose adsorbate energy is a
        MAD outlier across the sites computed on this slab.
    bind_tol, graph_tol, slab_max_disp, energy_mad_factor : float, optional
        Tolerances for the binding (layer 1), identity (layer 2), slab-integrity
        (layer 3) and energy-outlier (layer 4) checks respectively.

    Notes
    -----
    Gas-phase reference structures are determined from the requested pathway
    via `_pathway_required_refs`, relaxed once at the top of this routine, and
    appended in JSON-encoded form to every relaxed_set's `structures` list
    alongside `clean_slab`. Downstream analysis routines find them by the
    `info['adsorbate']` key (e.g. 'H2O', 'O2', 'Cl2') in the same way they
    find '*' (clean slab) and surface intermediates ('*O', '*OH', …).

    Adsorbate validation is documented in docs/sanity_checks.md. Rejected
    structures (with reasons) are written to rejected.json.
    """
    relaxed_sets = []
    rejected = []
    num_failed = 0
    total_number = 0
    # per-set outcome counts for the per-surface diagnostics (output.json "summary")
    n_relax_error = 0
    n_not_converged = 0
    n_rejected = 0
    model_key = f'{ml_model.lower()}_energy'

    result = generate_adsorbed_structures(reaction, pathway)
    adsorption_sets = result["sets"]
    gas_refs = result["gas_refs"]
    expected_graphs = result.get("expected_graphs", {})

    # Relax each pathway-required gas reference once. These energies are
    # appended (in the same JSON-encoded form) to every relaxed_set below, so
    # downstream analysis routines find them by their `info['adsorbate']` key
    # (e.g. 'H2O', 'O2') alongside '*' (clean slab) and the surface
    # intermediates ('*O', '*OH', ...).
    relaxed_refs_json = []
    for name, atoms in gas_refs.items():
        atoms.calc = calc
        relax = BFGSLineSearch(atoms, maxstep=0.1, logfile='opt.log')
        try:
            relax.run(fmax=fmax, steps=max_steps)
        except Exception as e:
            raise RuntimeError(f"Cannot proceed without gas-phase reference '{name}': {e}")
        if not relax.converged:
            raise RuntimeError(f"Gas-phase reference '{name}' did not converge in {max_steps} steps")

        # Store the PER-MOLECULE energy: the reference cells pack several
        # molecules (e.g. 8 H2O, 4 CO2), but the CHE bookkeeping downstream
        # consumes one molecule per stoichiometric unit.
        n_molecules = _reference_molecule_count(atoms, name)
        atoms.info[model_key] = atoms.get_potential_energy() / n_molecules
        relaxed_refs_json.append(jsonio.encode(atoms))
        total_number += 1

    for set_data in adsorption_sets.values():
        # Relax clean slab once per repeat configuration
        clean_slab = set_data['clean_slab']
        clean_slab.calc = calc
        relax_clean = BFGSLineSearch(clean_slab, maxstep=0.1, logfile='opt.log')

        try:
            relax_clean.run(fmax=fmax, steps=max_steps)
            clean_slab_energy = clean_slab.get_potential_energy()
            clean_slab.info[model_key] = clean_slab_energy
        except Exception as e:
            raise RuntimeError(f"Cannot proceed without clean surface reference: {e}")

        # Relax adsorbate structures
        for adsorb_set in set_data["adsorb_set"]:
            site_type = adsorb_set["site_type"]
            ads_coords = adsorb_set["ads_coord"]
            relaxed_structures = []
            for adsorbed in adsorb_set["structures"]:
                adsorbed.calc = calc
                relax = BFGSLineSearch(adsorbed, maxstep=0.1, logfile='opt.log')
                try:
                    relax.run(fmax=fmax, steps=max_steps)
                except Exception as e:
                    print(f"Warning: Relaxation failed for {adsorbed.info.get('adsorbate', 'unknown')}: {e}")
                    num_failed += 1
                    n_relax_error += 1
                    break

                if not relax.converged:
                    num_failed += 1
                    n_not_converged += 1
                    break

                ads_energy = adsorbed.get_potential_energy()
                adsorbed.info[model_key] = ads_energy

                # Post-relaxation sanity checks (layers 0-2 by default, +3 opt-in).
                # A failure is treated like a relaxation failure: the set is
                # dropped and the reason logged to rejected.json.
                if validate_adsorbates:
                    name = adsorbed.info.get('adsorbate')
                    exp_graph = expected_graphs.get(name)
                    if exp_graph is not None:
                        verdict = validate_relaxed_adsorbate(
                            adsorbed, exp_graph.number_of_nodes(), exp_graph,
                            ads_energy, graph_tol=graph_tol, bind_tol=bind_tol,
                            clean_slab_atoms=(clean_slab if check_slab_integrity else None),
                            slab_max_disp=slab_max_disp)
                        if not verdict.ok:
                            rejected.append({
                                "adsorbate": name,
                                "site_type": site_type,
                                "ads_coord": ads_coords.tolist(),
                                "repeat": list(adsorb_set["repeat"]),
                                "reason": verdict.reason})
                            num_failed += 1
                            n_rejected += 1
                            break

                relaxed_structures.append(jsonio.encode(adsorbed))
                total_number += 1

            # Only add complete relaxed sets with all adsorbates successfully relaxed
            if len(relaxed_structures) == len(adsorb_set["structures"]):
                relaxed_set = {
                    "site_type": site_type,
                    "ads_coord": ads_coords.tolist(),
                    "repeat": adsorb_set["repeat"],
                    "structures": (relaxed_structures
                                   + relaxed_refs_json
                                   + [jsonio.encode(clean_slab)])}
                relaxed_sets.append(relaxed_set)

    # Layer 4 (opt-in): ensemble energy-outlier rejection across sites.
    n_outliers = 0
    if check_energy_outliers and relaxed_sets:
        relaxed_sets, outlier_rejects = _flag_energy_outliers(relaxed_sets, model_key, energy_mad_factor)
        rejected.extend(outlier_rejects)
        num_failed += len(outlier_rejects)
        n_outliers = len(outlier_rejects)

    # Write output files
    summary = dict(result.get("placement", {}))
    summary.update({
        "n_sets": sum(len(sd["adsorb_set"]) for sd in adsorption_sets.values()),
        "n_relax_error": n_relax_error,
        "n_not_converged": n_not_converged,
        "n_rejected": n_rejected,
        "n_outliers": n_outliers,
        "n_relaxed": len(relaxed_sets),
        "validate": bool(validate_adsorbates),
    })
    output = {'structures': relaxed_sets, 'summary': summary}
    with open('output.json', 'w') as f:
        json.dump(output, f)

    with open('total.txt', 'w') as f:
        f.write(str(total_number))

    with open('failed.txt', 'w') as f:
        f.write(str(num_failed))

    # Structures rejected by the sanity checks, with reasons (see
    # docs/sanity_checks.md). Empty list when nothing was rejected.
    with open('rejected.json', 'w') as f:
        json.dump(rejected, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--slab_energy", type=float)
    parser.add_argument("--reaction", type=str)
    parser.add_argument("--pathway", type=str)
    # Post-relaxation sanity checks (see docs/sanity_checks.md).
    parser.add_argument("--no-validate", action="store_true",
                        help="disable default adsorbate validation (layers 0-2)")
    parser.add_argument("--check-slab-integrity", action="store_true",
                        help="opt-in layer 3: reject sets where the slab reconstructs")
    parser.add_argument("--check-energy-outliers", action="store_true",
                        help="opt-in layer 4: reject MAD-outlier adsorption energies")
    parser.add_argument("--bind_tol", type=float, default=1.25,
                        help="layer 1: surface-binding distance factor x (r_cov_i + r_cov_j)")
    parser.add_argument("--graph_tol", type=float, default=_ADSORBATE_BOND_TOL,
                        help="layer 2: bond-graph distance factor x (r_cov_i + r_cov_j)")
    parser.add_argument("--slab_max_disp", type=float, default=1.5,
                        help="layer 3: max slab-atom displacement (A)")
    parser.add_argument("--energy_mad_factor", type=float, default=5.0,
                        help="layer 4: outlier cut in robust standard deviations")
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model, model_path=args.model_path,
                           device=args.device, task_name=args.task_name)

    run_relaxation(ml_model=args.ML_model, calc=calc, fmax=args.fmax, max_steps=args.max_steps,
                   reaction=args.reaction, pathway=args.pathway,
                   validate_adsorbates=not args.no_validate,
                   check_slab_integrity=args.check_slab_integrity,
                   check_energy_outliers=args.check_energy_outliers,
                   bind_tol=args.bind_tol, graph_tol=args.graph_tol,
                   slab_max_disp=args.slab_max_disp,
                   energy_mad_factor=args.energy_mad_factor)
