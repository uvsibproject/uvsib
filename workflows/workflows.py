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
from uvsib.workflows import settings


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
    reaction = reaction.strip().upper()
    if reaction not in implemented_reactions:
        raise NotImplementedError(f"Reaction {reaction} not implemented.")
    reaction_path = reaction_path.strip().lower()
    if reaction_path not in implemented_reactions[reaction]:
        raise NotImplementedError(f"Path {reaction_path} not implemented for {reaction}.")
    return reaction, reaction_path

def add_from_frontend(dict_from_frontend_list):
    """Process frontend submissions and update the database accordingly."""
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
        reaction = entry["reaction"]
        reaction_path = entry["reaction_path"]

        retry = entry["retry"] if "retry" in entry else False

        if "similarities" in entry:
            similars = entry['similarities']
        else:
            similars = {}

        sqs = entry.get("sqs", {})

        check_valid(reaction, reaction_path)

        # db_frontend is owned by the backend: it writes the submission rows this
        # loop consumes, and platform only writes progress back via
        # update_dbfrontend(). Every entry here already has its row, so there is
        # nothing to insert.

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

    # Surface workflow progress back to the frontend / backend API for every
    # existing row (this cycle's new rows included). update_dbfrontend() is the
    # ONLY writer of db_frontend progress, so the periodic sweep / driver must
    # call it too -- not only here -- for queries that keep running between
    # submission cycles.
    update_dbfrontend()

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

def _project_step_status(comp_step_status, reaction, reaction_path):
    """Slice a DBComposition.step_status down to one (reaction, reaction_path).

    db_frontend has one row per (composition, reaction, reaction_path) while
    DBComposition has one row per composition, so a frontend row must reflect
    only the shared/pioneer step flags plus its OWN per-reaction leaves -- not
    the whole composition's step_status, which also holds every sibling
    reaction's state.
    """
    ss = comp_step_status or {}
    projected = {}
    for key in _SHARED_STEP_KEYS:
        state = ss.get(key)
        if isinstance(state, str):
            projected[key] = state
    for key in _PER_REACTION_STEP_KEYS:
        leaf = ((ss.get(key) or {}).get(reaction) or {}).get(reaction_path)
        if isinstance(leaf, str):
            projected[key] = leaf
    return projected


def _derive_frontend_status(projected):
    """Flat query status (Pending/Running/Done/Failed) from a projected
    step_status. Reads only the projected slice, never DBComposition.status --
    one sibling reaction's inspect step sets that column for the whole
    composition and it would leak across reactions.
    """
    states = set(projected.values())
    if "Failed" in states:
        return "Failed"
    if projected.get("pipeline_report") == "Done":
        return "Done"
    # Soft-stop runs end gracefully before surface builder / adsorbates /
    # pipeline_report, so "every step that ran is Done" is terminal there.
    if settings.SOFT_STOP_BEFORE_SURFACE and projected and states == {"Done"}:
        return "Done"
    if "Running" in states or "Done" in states:
        return "Running"
    return "Pending"


def update_dbfrontend():
    """Reconcile every non-terminal db_frontend row against its DBComposition.

    This is the only path that surfaces workflow progress back to the frontend /
    backend API. Run it on a timer as well as once per add_from_frontend cycle.

    - Each row is updated from its OWN projected slice of the composition's
      step_status (see _project_step_status), so two reactions on the same
      composition report independently.
    - A row whose composition has no DBComposition row yet is left untouched
      (still "Pending").
    - Only rows whose derived (status, step_status, result) actually changed are
      written, so repeated runs don't churn mtime.
    - A row that raises for any reason is skipped, never fatal to the sweep.
    """
    with get_session() as session:
        frontend_rows = (
            session.query(DBFrontend)
            .filter(or_(DBFrontend.status.is_(None), DBFrontend.status != "Done"))
            .all()
        )
        if not frontend_rows:
            return
        compositions = {
            row.composition: row for row in session.query(DBComposition).all()
        }

        updates = []
        for fe_row in frontend_rows:
            try:
                key = Composition(fe_row.composition).reduced_formula
                comp_row = compositions.get(key)
                if comp_row is None:                 # not picked up by a workflow yet
                    continue

                projected = _project_step_status(
                    comp_row.step_status, fe_row.reaction, fe_row.reaction_path
                )
                new_status = _derive_frontend_status(projected)
                # The report URL is recorded on DBComposition.attributes by
                # MainWorkChain.pipeline_report() when it writes the file, so we
                # copy it verbatim rather than re-deriving the path convention.
                stored_report = (
                    ((comp_row.attributes or {}).get("reports") or {})
                    .get(fe_row.reaction, {})
                    .get(fe_row.reaction_path)
                )
                new_result = stored_report or fe_row.result

                if (new_status, projected, new_result) == (
                    fe_row.status, fe_row.step_status, fe_row.result
                ):
                    continue
                updates.append((fe_row.uuid, {
                    "status": new_status,
                    "step_status": projected,
                    "result": new_result,
                }))
            except Exception as exc:
                print(f"update_dbfrontend: skipped {getattr(fe_row, 'uuid', '?')}: {exc}")

    for row_uuid, values in updates:
        try:
            update_row(DBFrontend, row_uuid, values)
        except Exception as exc:
            print(f"update_dbfrontend: failed to write {row_uuid}: {exc}")
