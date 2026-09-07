import re
from itertools import combinations
from sqlalchemy import inspect, delete, text
from pymatgen.core import Composition, Structure
from uvsib.db.session import get_session
from uvsib.db.tables import (DBChemsys, DBStructure, DBStructureVersion, DBSurface,
                             DBSurfaceMLAdsorbate, DBAkmcEvent)


def add_surface_ml_adsorbate(existing_uuid, surf_id, surface_miller_index, comp, react, react_path, site_type, ads_coord, repeat, e, dG_steps, dG_cumulative, ad_set):
    """Store a new DBSurfaceAdsorbate row corresponding to a given DBStructure UUID and surface ID"""
    with get_session() as session:
        adsorb = DBSurfaceMLAdsorbate(
                structure_uuid=existing_uuid,
                surface_id=surf_id,
                surface_miller_index=surface_miller_index,
                composition=comp,
                reaction=react,
                reaction_path=react_path,
                site_type=site_type,
                ads_coord=ads_coord,
                repeat=repeat,
                eta=e,
                dG_steps=dG_steps,
                dG_cumulative=dG_cumulative,
                adsorb_set=ad_set
        )
        session.add(adsorb)
        session.commit()
    return True

def add_akmc_events(events):
    """Bulk-insert AKMC dimer-search events as DBAkmcEvent rows.

    Parameters
    ----------
    events : list of dict
        Each dict supplies the DBAkmcEvent columns (source_row_id, energies,
        barrier, rate, kmc_selected, atoms_json blobs, etc).
    """
    if not events:
        return []
    with get_session() as session:
        db_events = [DBAkmcEvent(**event) for event in events]
        session.add_all(db_events)
        session.commit()
    return db_events


def merge_row_attributes(table_class, row_id, new_attributes):
    """Merge ``new_attributes`` into ``table_class.attributes`` (JSONB) for the
    row with the given integer id, preserving unrelated existing keys."""
    with get_session() as session:
        row = session.query(table_class).filter_by(id=row_id).first()
        if row is None:
            return False
        merged = dict(row.attributes or {})
        merged.update(new_attributes)
        row.attributes = merged
        session.commit()
    return True


def add_slab(existing_uuid, comp, slab_dict, head=None, formation_energy=None):
    with get_session() as session:
        slab = DBSurface(
            structure_uuid=existing_uuid,
            composition=comp,
            slab=slab_dict,
            formation_energy=formation_energy,
            attributes={"model_head": head} if head else None,
        )

        session.add(slab)
        session.commit()
    return True

####################################

def add_structures(
        source,
        method,
        structure_energy_pairs,
        attributes={},
        mp_ids=None
    ):
    """Add new structures and associated energies to the database.

    ``mp_ids``, when given, is a list of Materials Project ids parallel to
    ``structure_energy_pairs`` (use ``None`` for entries with no id). It is
    stored on both the ``DBStructure`` and ``DBStructureVersion`` rows and is
    only meaningful when ``source`` is an MPDB source.
    """
    with get_session() as session:
        db_structures = []
        db_versions = []

        for idx, (struct_dict, energy) in enumerate(structure_energy_pairs):
            struct = Structure.from_dict(struct_dict)
            composition = struct.composition.reduced_formula
            chemical_system = Composition(composition).chemical_system
            mp_id = mp_ids[idx] if mp_ids is not None else None

            db_structure = DBStructure(
                composition=composition,
                chemsys=chemical_system,
                mp_id=mp_id,
                attributes=attributes
            )
            session.add(db_structure)
            session.flush()

            db_version = DBStructureVersion(
                structure_uuid=db_structure.uuid,
                composition=composition,
                chemsys=chemical_system,
                method=method,
                source=source,
                mp_id=mp_id,
                structure=struct_dict,
                energy=energy,
            )

            db_versions.append(db_version)
            db_structures.append(db_structure)

        session.add_all(db_versions)
        session.commit()

def add_version_to_existing_structure(
        existing_uuid,
        struct_dict,
        method,
        add_attributes,
        on_conflict = "error"):
    """
    Add a new version to an existing structure in the database.

    Parameters
    ----------
    existing_uuid : str
        UUID of the existing structure.
    method : str
        Method used to generate this version (e.g., "DFT", "MACE").
    add_attributes : dict
        Dictionary containing any additional attributes for DBStructureVersion.
    on_conflict : {"error", "ignore", "override"}, optional
        How to handle conflicts if a version with the same structure_uuid and
        method already exists.
        - "error": raise an exception (default)
        - "ignore": do nothing and return the existing version
        - "override": update the existing version with new attributes
    """
    with get_session() as session:
        existing_version = (
            session.query(DBStructureVersion)
            .filter_by(structure_uuid=existing_uuid, method=method)
            .first()
        )

        if existing_version:
            if on_conflict == "error":
                return False
            if on_conflict == "ignore":
                return True
            if on_conflict == "override":
                for key, value in add_attributes.items():
                    setattr(existing_version, key, value)
                session.commit()
                return True

        # If no conflict, add a new version

        struct = Structure.from_dict(struct_dict)
        composition = struct.composition.reduced_formula
        chemical_system = Composition(composition).chemical_system

        db_version = DBStructureVersion(
            structure_uuid=existing_uuid,
            structure=struct_dict,
            composition=composition,
            chemsys=chemical_system,
            method=method,
            **add_attributes
        )
        session.add(db_version)
        session.commit()
    return True

def delete_structure(structure_filters: dict, **version_filters):
    """
    Delete DBStructureVersion entries (and possibly their parent DBStructure)
    
    Parameters
    ----------
    structure_filters : dict
        Filters for DBStructure (e.g., {'uuid': '...'})
    version_filters : dict
        Filters for DBStructureVersion (e.g., version=1, status='draft')
    
    Returns
    -------
    int
        Number of DBStructureVersion rows deleted
    """
    deleted_count = 0

    with get_session() as session:
        query = session.query(DBStructureVersion).join(DBStructure)

        # Apply filters on DBStructure
        for attr, value in structure_filters.items():
            query = query.filter(getattr(DBStructure, attr) == value)

        # Apply filters on DBStructureVersion
        for attr, value in version_filters.items():
            query = query.filter(getattr(DBStructureVersion, attr) == value)

        versions_to_delete = query.all()

        if versions_to_delete:
            # Track affected structures
            affected_structure_uuids = {v.structure_uuid for v in versions_to_delete}

            # Delete versions safely
            for v in versions_to_delete:
                session.delete(v)
                deleted_count += 1

            session.flush()

            # Delete parent structures if they have no remaining versions
            for uuid in affected_structure_uuids:
                remaining = (
                    session.query(DBStructureVersion)
                    .filter(DBStructureVersion.structure_uuid == uuid)
                    .count()
                )
                if remaining == 0:
                    session.query(DBStructure).filter(DBStructure.uuid == uuid).delete()

        session.commit()

def query_structure(structure_filters, **version_filters):
    """
    Query DBStructureVersion joined with DBStructure
    
    structure_filters: Dict of column names and values for DBStructure (e.g., {'uuid': '...'})
    version_filters: Keyword arguments for DBStructureVersion filters
    """
    with get_session() as session:
        query = session.query(DBStructureVersion).join(DBStructure)

        # Apply filters on DBStructure
        for attr, value in structure_filters.items():
            query = query.filter(getattr(DBStructure, attr) == value)

        # Apply filters on DBStructureVersion
        for attr, value in version_filters.items():
            query = query.filter(getattr(DBStructureVersion, attr) == value)

        return query.all()

def query_structureversions_by_attributes(**filters):
    """Query DBStructureVersion with flexible filters (method, energy, etc)"""
    with get_session() as session:
        query = session.query(DBStructureVersion)
        for attr, value in filters.items():
            query = query.filter(getattr(DBStructureVersion, attr) == value)
        return query.all()

####################################

def update_version_attributes(structure_uuid, method, new_attributes, source=None):
    """Merge ``new_attributes`` into a DBStructureVersion.attributes JSONB.

    Selects the version by (structure_uuid, method[, source]); returns False if
    no matching version exists. Used to attach synthesizability scores to the
    generated structure versions without adding a duplicate version row.
    """
    with get_session() as session:
        query = session.query(DBStructureVersion).filter_by(
            structure_uuid=structure_uuid, method=method)
        if source is not None:
            query = query.filter_by(source=source)
        version = query.first()
        if version is None:
            return False
        merged = dict(version.attributes or {})
        merged.update(new_attributes)
        version.attributes = merged
        session.commit()
    return True


def update_structure_band_info(structure_uuid, method, band_info, source=None):
    """Set the ``band_info`` JSONB **column** (not ``attributes``) of the
    DBStructureVersion selected by ``(structure_uuid, method[, source])``.

    Used by OpticalScreenWorkChain to attach the no-DFT band gap / band-edge /
    straddle screen to the version PhaseDiagramMLWorkChain ranked
    (``method == ml_bulk_model``). Overwrites any previous value. Returns
    ``False`` if no matching version exists.
    """
    with get_session() as session:
        query = session.query(DBStructureVersion).filter_by(
            structure_uuid=structure_uuid, method=method)
        if source is not None:
            query = query.filter_by(source=source)
        version = query.first()
        if version is None:
            return False
        version.band_info = band_info
        session.commit()
    return True


def query_by_columns(table_class, filters):
    """
    Query a given table for rows matching all column-value pairs in "filters"
    Args:
        table_name (str): Name of the table
        filters (dict): Dictionary of column names and their expected values
    Returns:
        list of rows
    """
    table_name = table_class.__tablename__
    where_clauses = [f"{col} = :{col}" for col in filters]
    query_str = f"""
        SELECT *
        FROM {table_name}
        WHERE {" AND ".join(where_clauses)}
    """
    query = text(query_str)
    with get_session() as session:
        result = session.execute(query, filters).fetchall()
    return result

####################################

def get_chemical_systems(chemical_formula):
    """Given a chemical formula, return either all or new chemical systems"""
    comp = Composition(chemical_formula)
    elements = sorted(el.symbol for el in comp.elements)

    chemical_systems = []
    new_chemsys = []

    for n in range(1, len(elements) + 1):
        for combo in combinations(elements, n):
            subsystem = "-".join(combo)
            chemical_systems.append(subsystem)

    for subsystem in chemical_systems:
        result = query_by_columns(DBChemsys,{"chemsys": subsystem})
        if result:
            continue
        else:
            new_chemsys.append(subsystem)
    return chemical_systems, new_chemsys

####################################

def update_row(table_class, uuid_value, columns_values):
    """
    Update a row in a table by uuid (overwrite existing data)
    Args:
        table_class: table class (declarative)
        uuid_value: UUID value (either UUID object or str)
        columns_values: dict of column_name and value to update
    """
    # Remove uuid from update fields to avoid updating primary key
    update_values = dict(columns_values)

    stmt = (
        table_class.__table__
        .update()
        .where(table_class.uuid == uuid_value)
        .values(**update_values)
    )
    with get_session() as session:
        session.execute(stmt)
        session.commit()


_SAFE_STEP_KEY = re.compile(r"^[A-Za-z0-9_:.\-]+$")
_SAFE_JSON_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def update_json_path(table_class, uuid_value, column, path, value):
    """Atomically set a nested key in a JSONB ``column`` of the row picked by
    ``uuid``.

    Sets ``column`` at ``path`` (a list of keys, e.g.
    ``["adsorbates", reaction, reaction_path]``) to the string ``value``,
    creating any missing intermediate objects and PRESERVING sibling keys at
    every level. Implemented as a single nested ``jsonb_set`` UPDATE, so it is
    safe under concurrent updates of the SAME row by parallel sibling
    workchains -- each writes only its own leaf instead of overwriting the whole
    column (which a read-modify-write of the JSONB would do). A NULL column is
    treated as ``{}``.

    ``column`` and the keys are restricted to a safe identifier charset;
    ``value`` must be a string.
    """
    if not path:
        raise ValueError("path must be a non-empty list of keys")
    if not _SAFE_JSON_COLUMN.match(column):
        raise ValueError(f"unsafe column name: {column!r}")
    for key in path:
        if not _SAFE_STEP_KEY.match(key):
            raise ValueError(f"unsafe {column} key: {key!r}")
    if not isinstance(value, str):
        raise ValueError("value must be a string")

    # Build the nested jsonb_set expression from the innermost key outward, so
    # each level coalesces the existing sub-object (or '{}') -> missing parents
    # are created and existing siblings are kept.
    expr = "to_jsonb(CAST(:val AS text))"
    for i in range(len(path) - 1, -1, -1):
        leaf_arr = "'{" + path[i] + "}'::text[]"
        parent = path[:i]
        if parent:
            parent_arr = "'{" + ",".join(parent) + "}'::text[]"
            existing = f"COALESCE({column} #> {parent_arr}, '{{}}'::jsonb)"
        else:
            existing = f"COALESCE({column}, '{{}}'::jsonb)"
        expr = f"jsonb_set({existing}, {leaf_arr}, {expr}, true)"

    query = text(
        f"UPDATE {table_class.__tablename__} "
        f"SET {column} = {expr} WHERE uuid = :uuid"
    )
    with get_session() as session:
        session.execute(query, {"uuid": str(uuid_value), "val": value})
        session.commit()


def update_step_status_path(table_class, uuid_value, path, value):
    """Atomically set a nested key in the ``step_status`` JSONB column
    (thin wrapper around :func:`update_json_path`; step states used here are
    'Running'/'Done'/'Failed')."""
    update_json_path(table_class, uuid_value, "step_status", path, value)

def add_row(table_class, rows_data):
    """
    Add one or more rows to a table

    Parameters:
    - table_class
    - rows_data: list of dictionaries (each dict is one row), or a single dict

    Returns:
    - List of newly added row objects
    """
    if isinstance(rows_data, dict):
        rows_data = [rows_data]  # Allow single dict input

    new_rows = [table_class(**row_data) for row_data in rows_data]

    with get_session() as session:
        session.add_all(new_rows)
        session.commit()
    return new_rows

def delete_row(table_class, row):
    """
    Parameters:
    - table_class: SQLAlchemy model class (e.g., DBFrontend)
    - row to be deleted
    """
    with get_session() as session:
        stmt = delete(table_class).where(table_class.uuid == row.uuid)
        session.execute(stmt)
        session.commit()

def get_table_data(session, table_class):
    """
    Fetch all rows from the given table class and return as a list of dictionaries
    Args:
        Session: SQLAlchemy session factory
        table_class: SQLAlchemy ORM model class
    """
    try:
        rows = session.query(table_class).all()
        columns = [column.name for column in inspect(table_class).columns]
        return [{col: getattr(row, col) for col in columns} for row in rows]
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

def delete_all_rows(session, table_class):
    """
    Deletes all rows from the given SQLAlchemy table class
    Args:
        Session: SQLAlchemy session factory
        table_class: SQLAlchemy ORM model class
    """
    try:
        deleted_count = session.query(table_class).delete()
        session.commit()
        print(f"Deleted {deleted_count} rows from '{table_class.__tablename__}'")
    except Exception as e:
        session.rollback()
        print(f"Error deleting rows: {e}")
    finally:
        session.close()

def print_all_rows(session, table_class):
    """Reflect the table by name and print all rows"""
    rows = get_table_data(session, table_class)
    if rows:
        print(f"\nRows from table '{table_class.__table__}':")
        for row in rows:
            print(row)
