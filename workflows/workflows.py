from datetime import datetime, timedelta, timezone
from sqlalchemy import or_, text
from pymatgen.core.structure import Composition
from aiida.orm import QueryBuilder, WorkChainNode
from uvsib.db.tables import DBFrontend, DBChemsys, DBComposition
from uvsib.db.session import get_session
from uvsib.db.utils import (update_row, add_row, get_chemical_systems, query_by_columns,
                            update_step_status_path)
from uvsib.workchains.submit import submit_mainworkchain
from uvsib.workchains.phase_diagram import cleanup_failed_systems


def check_valid(reaction, reaction_path):
    # BATTERY is the bulk (deintercalation) pathway: the "path" is the working
    # ion, and the run skips surface builder + adsorbates (see docs/batteries.md)
    from uvsib.workchains.cer import CER_PATHWAYS
    from uvsib.workchains.co2rr import CO2RR_PATHWAYS
    from uvsib.workchains.her import HER_PATHWAYS
    from uvsib.workchains.noxrr import NOXRR_PATHWAYS
    from uvsib.workchains.nrr import NRR_PATHWAYS
    from uvsib.workchains.orr import ORR_PATHWAYS
    # Path lists come from the single source of truth (the *_PATHWAYS dict of
    # each workchain) so this gate cannot drift out of sync again — a stale
    # hand-copied list here is what blocked ORR/HER/NRR/CER and the newer
    # CO2RR chains. OER has no pathway dict; 'default' is its only route.
    implemented_reactions = {'OER': ['4e'],
                             'HER': sorted(HER_PATHWAYS),
                             'ORR': sorted(ORR_PATHWAYS),
                             'CER': sorted(CER_PATHWAYS),
                             'NRR': sorted(NRR_PATHWAYS),
                             'CO2RR': sorted(CO2RR_PATHWAYS),
                             'NOXRR': sorted(NOXRR_PATHWAYS)}
    # Normalize case so 'battery'/'Li'/'li' all work, and so one canonical
    # spelling reaches the DB rows, step_status keys, and workchain labels.
    reaction = reaction.strip().upper()
    if reaction not in implemented_reactions:
        raise NotImplementedError(f"Reaction {reaction} not implemented.")
    if reaction == 'BATTERY':
        reaction_path = reaction_path.strip().capitalize()   # ion symbol: li -> Li
    else:
        reaction_path = reaction_path.strip().lower()
    if reaction_path not in implemented_reactions[reaction]:
        raise NotImplementedError(f"Path {reaction_path} not implemented for {reaction}.")
    return reaction, reaction_path

def add_from_frontend(dict_from_frontend_list):
    """Process frontend submissions and update the database accordingly."""
    # Self-heal state left "Running" forever by a crashed/killed workflow
    # (see reset_orphaned_chemsys/reset_orphaned_compositions) before the
    # active/ran_before checks below rely on it, so a retry isn't blocked by
    # a crash the periodic sweep hasn't gotten to yet.
    reset_orphaned_chemsys()
    reset_orphaned_compositions()

    # Count reactions per composition so we only gate (wait) when there are
    # follower reactions that would benefit from reusing the shared steps.
    reactions_per_comp = {}
    for e in dict_from_frontend_list:
        comp = Composition(e["chemical_formula"]).reduced_formula
        reactions_per_comp[comp] = reactions_per_comp.get(comp, 0) + 1

    for entry in dict_from_frontend_list:
        chemical_formula = Composition(entry["chemical_formula"]).reduced_formula
        user = entry["user"]
        reaction = entry["reaction"]
        reaction_path = entry["reaction_path"]

        retry = entry["retry"] if "retry" in entry else False

        if "similarities" in entry:
            similars = entry['similarities']
        else:
            similars = {}

        # The SQS request payload (parent structure, sublattices, composition
        # grid, surfaces, defects) rides through verbatim; {} means a normal
        # (non-SQS) submission. submit_mainworkchain wraps it in an aiida Dict.
        sqs = entry.get("sqs", {})

        check_valid(reaction, reaction_path)

        existing_frontend_rows = query_by_columns(DBFrontend,{"composition": chemical_formula})
        user_already_exists = any(row.username == user for row in existing_frontend_rows)

        if not user_already_exists:
            add_row(DBFrontend, {
                "username": user,
                "composition": chemical_formula,
                "reaction": reaction,
                "reaction_path": reaction_path}
            )

        # check if a composition is already processed
        existing_composition = query_by_columns(DBComposition, {"composition": chemical_formula})
        if not existing_composition:
            add_row(DBComposition, {"composition": chemical_formula})

        # only new chemical systems
        _, new_chemsys = get_chemical_systems(chemical_formula)
        for chemsys in new_chemsys:
            add_row(DBChemsys, {"chemsys": chemsys})

        # (a) already in flight -> an active MainWorkChain carries this label;
        # (b) failed + no retry  -> a terminated MainWorkChain carries this label.
        # Label must match launch_calculations.get_inputs_and_processclass_from_extras.
        label = f"CatalystChain {reaction}:{reaction_path} on {chemical_formula}"
        try:
            active = QueryBuilder().append(
                WorkChainNode,
                filters={"label": label,
                         "attributes.process_state": {"in": ["created", "running", "waiting"]}},
            ).count()
        except Exception:
            active = 0
        if active:
            continue
        if not retry:
            ran_before = QueryBuilder().append(
                WorkChainNode,
                filters={"label": label,
                         "attributes.process_state": {"in": ["finished", "excepted", "killed"]}},
            ).count()
            if ran_before:        # ran before without a result row -> failed
                continue

        submit_mainworkchain(chemical_formula=chemical_formula, chemical_systems=new_chemsys,
                             reaction=reaction, reaction_path=reaction_path,
                             similarities=similars, sqs=sqs)
#        update_dbfrontend()

def reset_orphaned_chemsys():
    """Delete DBChemsys rows stuck not-"Ready" from a crashed/killed workflow.

    A MainWorkChain/PhaseDiagramMLWorkChain that dies (OOM, daemon restart,
    manual kill) never reaches its inspect_* step, so gen_structures stays
    unset and the row blocks is_data_available() (workchains/pythonjob_inputs.py)
    for every future submission needing that chemsys, for the full 10 h
    timeout. Only chemsys not claimed by a still-active workflow are removed,
    so a run that is genuinely in progress is left untouched; the row gets
    re-created and regenerated the next time it's requested.
    """
    with get_session() as session:
        stuck_rows = session.query(DBChemsys).filter(
            or_(DBChemsys.gen_structures.is_(None), DBChemsys.gen_structures != "Ready")
        ).all()
        stuck_chemsys = [row.chemsys for row in stuck_rows]

    if not stuck_chemsys:
        return []

    active_chemsys = set()
    active_nodes = QueryBuilder().append(
        WorkChainNode,
        filters={"attributes.process_state": {"in": ["created", "running", "waiting"]}},
    ).all()
    for (node,) in active_nodes:
        try:
            active_chemsys.update(node.inputs.chemical_systems.get_list())
        except AttributeError:
            continue

    orphaned = [chemsys for chemsys in stuck_chemsys if chemsys not in active_chemsys]
    cleanup_failed_systems(orphaned)
    return orphaned

_ORPHAN_GRACE_PERIOD = timedelta(minutes=15)

# Shared/"pioneer" steps: one per composition, owned by whichever reaction gets
# there first (see the 'pioneered' comment in add_from_frontend).
_SHARED_STEP_KEYS = ["pd_ml", "pd_verification", "synthesizability", "sqs", "surface_builder"]
# Per-(reaction, reaction_path) steps: step_status[key][reaction][reaction_path].
_PER_REACTION_STEP_KEYS = ["adsorbates", "akmc", "pipeline_report"]

def _stuck_step_paths(step_status):
    """Yield step_status paths (lists of keys) currently marked "Running"."""
    step_status = step_status or {}
    for key in _SHARED_STEP_KEYS:
        if step_status.get(key) == "Running":
            yield [key]
    for key in _PER_REACTION_STEP_KEYS:
        for reaction, per_reaction in (step_status.get(key) or {}).items():
            for reaction_path, state in (per_reaction or {}).items():
                if state == "Running":
                    yield [key, reaction, reaction_path]

def reset_orphaned_compositions():
    """Fail DBComposition status/step_status stuck "Running" from a
    crashed/killed workflow.

    Mirrors reset_orphaned_chemsys(), but a killed MainWorkChain leaves
    "Running" behind in TWO places, not one: the top-level ``status`` column,
    and ``step_status`` -- which records the same state per shared step
    (pd_ml, pd_verification, synthesizability, sqs, surface_builder) and per
    (reaction, reaction_path) for adsorbates/akmc/pipeline_report.
    should_wait_*() (main.py) makes any sibling WorkChain on the same
    composition -- including a fresh resubmission of the very same
    (reaction, reaction_path) -- loop in an unbounded while_(wait_sleep) as
    long as the step it needs reads "Running", regardless of whether
    ``status`` itself got fixed. So fixing only ``status`` leaves that hang
    in place; both need repairing together.

    - Composition with no active WorkChainNode at all: every "Running" leaf
      in step_status (shared or per-reaction) is unambiguously orphaned, and
      ``status`` is reset to "Failed" -- what the graceful failure path would
      have set, so add_from_frontend's existing "active"/"ran_before" retry
      logic picks it back up on the next submission.
    - Composition WITH an active WorkChainNode (a healthy sibling reaction is
      still running): shared/pioneer keys are left alone since we can't tell
      which sibling owns them, but a per-reaction leaf is still reset if ITS
      OWN (composition, reaction, reaction_path) has no active node, since
      that key is written only by the workflow with matching inputs.

    mtime gates candidates so a row already fresh isn't touched. Because
    mtime is a row-level timestamp shared by all sibling reactions on a
    composition, a busy composition can keep it "recent" even while one
    sibling's own leaf is stale; this only delays cleanup of that leaf, it
    never resets something that is still genuinely in progress.
    """
    cutoff = datetime.now(timezone.utc) - _ORPHAN_GRACE_PERIOD
    with get_session() as session:
        candidate_rows = session.execute(text("""
            SELECT uuid, composition, status, step_status
            FROM db_composition
            WHERE mtime < :cutoff
              AND (status = 'Running' OR step_status::text LIKE '%"Running"%')
        """), {"cutoff": cutoff}).fetchall()

    if not candidate_rows:
        return []

    active_compositions = set()
    active_pairs = set()
    active_nodes = QueryBuilder().append(
        WorkChainNode,
        filters={"attributes.process_state": {"in": ["created", "running", "waiting"]}},
    ).all()
    for (node,) in active_nodes:
        try:
            formula = node.inputs.chemical_formula.value
        except AttributeError:
            continue
        active_compositions.add(formula)
        try:
            active_pairs.add((formula, node.inputs.reaction.value, node.inputs.reaction_path.value))
        except AttributeError:
            pass

    reset_compositions = []
    for row_uuid, composition, status, step_status in candidate_rows:
        composition_inactive = composition not in active_compositions
        touched = False

        for path in _stuck_step_paths(step_status):
            if len(path) == 1:
                if not composition_inactive:
                    continue
            else:
                _, reaction, reaction_path = path
                if (composition, reaction, reaction_path) in active_pairs:
                    continue
            update_step_status_path(DBComposition, row_uuid, path, "Failed")
            touched = True

        if status == "Running" and composition_inactive:
            update_row(DBComposition, row_uuid, {"status": "Failed"})
            touched = True

        if touched:
            reset_compositions.append(composition)

    return reset_compositions

def update_dbfrontend():
    """Updateing DBFrontend status"""
    for status in ["Pending", "Created", "Running", "Failed"]:
        db_fe_rows = query_by_columns(DBFrontend, {"status": status})
        for fe_row in db_fe_rows:
            db_c_row = query_by_columns(DBComposition,{"composition": fe_row.composition})[0]
            update_row(DBFrontend, fe_row.uuid,{"status": db_c_row.status, "step_status": db_c_row.step_status})
