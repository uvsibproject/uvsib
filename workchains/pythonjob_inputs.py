def is_data_available(chemical_systems, timeout=36000):
    """Check if data for a phase diagram is available.
    Returns True once no chemical system is still being generated, otherwise
    sleeps. Ready, Empty and Failed systems are all finished (Empty/Failed are
    recorded as hull gaps by PhaseDiagramMLWorkChain), and so is a system with
    no DBChemsys row, since nothing will generate it. Timeout 10 h.
    """
    import time
    from uvsib.db.tables import DBChemsys
    from uvsib.db.utils import query_by_columns, GEN_TERMINAL

    tstart = time.time()
    while True:
        pending = []
        for chemical_system in chemical_systems:
            try:
                result = query_by_columns(DBChemsys,{"chemsys": chemical_system})
            except:
                return {"moveon": False}
            if result and result[0].gen_structures not in GEN_TERMINAL:
                pending.append(chemical_system)

        if not pending:
            return {"moveon": True}
        if time.time() - tstart > timeout:
            return {"moveon": False}
        time.sleep(60)

def wait_sleep(timeout=360):
    """Sleep"""
    import time
    tstart = time.time()
    while True:
        if time.time() - tstart > timeout:
            return
        time.sleep(60)
