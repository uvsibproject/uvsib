import uuid
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, UniqueConstraint, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, DOUBLE_PRECISION
from sqlalchemy.sql import func
from uvsib.db.db_url import DB_URL
from uvsib.db.session import Base


class DBChemsys(Base):
    """Chemical system table"""
    __tablename__ = "db_chemsys"

    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    chemsys = Column(String, nullable=False)
    model = Column(String, nullable=True)
    gen_structures = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("chemsys", name="_list_formula_uc"),
    )

class DBComposition(Base):
    """Composition table """
    __tablename__ = "db_composition"

    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    composition = Column(String, nullable=False)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=False, default=dict)
    stable_struct = Column(JSONB, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<DBComposition(uuid={self.uuid}, composition={self.composition})>"

class DBStructure(Base):
    __tablename__ = "db_structure"
    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    composition = Column(String, nullable=False)
    chemsys = Column(String, nullable=False)
    mp_id = Column(String, nullable=True)  # Materials Project id, set when source is MPDB
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

class DBStructureVersion(Base):
    __tablename__ = 'db_structure_version'

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    structure = Column(JSONB, nullable=False)
    composition = Column(String, nullable=False)
    chemsys = Column(String, nullable=False)
    source = Column(String, nullable=False)
    mp_id = Column(String, nullable=True)  # Materials Project id, set when source is MPDB
    method = Column(String, nullable=False)
    energy = Column(DOUBLE_PRECISION, nullable=True)
    ehull = Column(DOUBLE_PRECISION, nullable=True)
    vasprun_str = Column(Text, nullable=True)
    band_info = Column(JSONB, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

class DBSurface(Base):
    __tablename__ = "db_surface"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    composition = Column(String, nullable=True)
    slab = Column(JSONB, nullable=False) # structure & energy
    formation_energy = Column(DOUBLE_PRECISION, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBSurfaceMLAdsorbate(Base):
    __tablename__ = "db_surface_ml_adsorbate"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(UUID(as_uuid=True), ForeignKey("db_structure.uuid", ondelete="CASCADE"), nullable=False)
    surface_id = Column(Integer, ForeignKey("db_surface.id", ondelete="CASCADE"), nullable=False)
    surface_miller_index = Column(JSONB, nullable=False)
    composition = Column(String, nullable=True)
    reaction = Column(String, nullable=False)
    reaction_path = Column(String, nullable=False)
    site_type = Column(String, nullable=False)
    ads_coord = Column(JSONB, nullable=False)
    repeat = Column(Text, nullable=False)
    eta = Column(DOUBLE_PRECISION, nullable=False)
    dG_steps = Column(JSONB, nullable=False)
    dG_cumulative = Column(JSONB, nullable=False)
    adsorb_set = Column(JSONB, nullable=False) # structures & energies
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

class DBAkmcEvent(Base):
    """One dimer-search event discovered while running AKMCWorkChain on a
    DBSurfaceMLAdsorbate minimum. One row per event (not one blob per source
    row), so barrier/rate/kmc_selected are indexable SQL columns instead of
    JSONB buried in an unrelated row's attributes."""
    __tablename__ = "db_akmc_event"

    id = Column(Integer, primary_key=True)
    source_row_id = Column(Integer, ForeignKey("db_surface_ml_adsorbate.id", ondelete="CASCADE"), nullable=False)
    composition = Column(String, nullable=True)
    reaction = Column(String, nullable=True)
    reaction_path = Column(String, nullable=True)
    source_adsorbate = Column(String, nullable=True)
    source_site_type = Column(String, nullable=True)
    source_eta = Column(DOUBLE_PRECISION, nullable=True)
    attempt = Column(Integer, nullable=False)
    initial_energy = Column(DOUBLE_PRECISION, nullable=False)
    saddle_energy = Column(DOUBLE_PRECISION, nullable=False)
    final_energy = Column(DOUBLE_PRECISION, nullable=False)
    barrier = Column(DOUBLE_PRECISION, nullable=False)
    reverse_barrier = Column(DOUBLE_PRECISION, nullable=False)
    rate = Column(DOUBLE_PRECISION, nullable=False)
    temperature = Column(DOUBLE_PRECISION, nullable=False)
    prefactor = Column(DOUBLE_PRECISION, nullable=False)
    seed = Column(Integer, nullable=True)
    kmc_selected = Column(Boolean, nullable=False, default=False)
    kmc_time_increment = Column(DOUBLE_PRECISION, nullable=True)
    final_state_key = Column(JSONB, nullable=True)
    initial_atoms_json = Column(Text, nullable=False)
    saddle_atoms_json = Column(Text, nullable=False)
    final_atoms_json = Column(Text, nullable=False)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<DBAkmcEvent(id={self.id}, source_row_id={self.source_row_id}, barrier={self.barrier})>"

class DBFrontend(Base):
    """Frontend table"""
    __tablename__ = 'db_frontend'
    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    username = Column(String, nullable=True)
    user_id = Column(UUID, nullable=True)
    composition = Column(String, nullable=True)
    reaction = Column(String, nullable=True)
    reaction_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=True)
    result = Column(String, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<DBFrontend(uuid={self.uuid}, username={self.username})>"

#class DBSimilarities(Base):
#    __tablename__ = "db_similarities"
#    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
#    similarity = Column(Float, nullable=True)
#    composition = Column(String, nullable=True)
#    csp_structure = Column(JSONB, nullable=True)
#    chemical_system = Column(String, nullable=True)
#    reference_structure = Column(JSONB, nullable=True)
#    reference_material_id = Column(String, nullable=True)
#    mtime = Column(DateTime(timezone=True), onupdate=func.now())
#    ctime = Column(DateTime(timezone=True), server_default=func.now())

class DBSynthesizability(Base):
    """Synthesizability prediction for one ML-selected bulk by one model
    (written by SynthesizabilityScreenWorkChain). One row per
    (structure_uuid, synthesizability_model), so several models can score the
    same bulk side by side; joins to DBStructureVersion / DBSurface /
    DBSurfaceMLAdsorbate on structure_uuid. Advisory only -- nothing in the
    pipeline filters on it."""
    __tablename__ = "db_synthesizability"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(UUID(as_uuid=True), ForeignKey("db_structure.uuid", ondelete="CASCADE"), nullable=False)
    composition = Column(String, nullable=False)
    synthesizability_model = Column(String, nullable=False)
    synthesizability_score = Column(DOUBLE_PRECISION, nullable=True)        # P(synthesizable)
    synthesizability_label = Column(String, nullable=True)                  # synthesizable / not synthesizable / uncertain
    synthesizability_uncertainty = Column(DOUBLE_PRECISION, nullable=True)  # binary entropy of the score, bits
    predicted_synthesis_method = Column(String, nullable=True)
    predicted_precursors = Column(JSONB, nullable=True)                     # ranked precursor sets
    in_domain = Column(Boolean, nullable=True)                              # precursor model's validated scope
    ehull = Column(DOUBLE_PRECISION, nullable=True)                         # snapshot from the ML selection
    attributes = Column(JSONB, nullable=True)                               # raw model output + config
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("structure_uuid", "synthesizability_model", name="_synthesizability_struct_model_uc"),
    )

if __name__ == "__main__":
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)
