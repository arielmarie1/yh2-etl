from psycopg2.extras import execute_values
from sqlalchemy import create_engine, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
import pandas as pd
import logging
import psycopg2 as pg2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# CREATE TABLES
class Scenario(Base):
    """Table for scenario metadata that links Timeseries and Plan data"""
    __tablename__ = 'scenario'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    duration: Mapped[str] = mapped_column(String(50), nullable=False)
    scenario: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Relationships
    timeseries_data = relationship("Timeseries", back_populates="scenario_info")
    plan_data = relationship("Plan", back_populates="scenario_info")
    
    def __repr__(self):
        return f'<Scenario {self.scenario}_{self.duration}_{self.location}_{self.constraint}_{self.amount}>'


class Timeseries(Base):
    """Table for time series data from CSV processing"""
    __tablename__ = 'timeseries'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    time: Mapped[str] = mapped_column(DateTime, nullable=False)
    component: Mapped[str] = mapped_column(String(100), nullable=False)
    zone: Mapped[str] = mapped_column(String(100), nullable=False)
    parameter: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Foreign key to Scenario
    scenario_id: Mapped[int] = mapped_column(Integer, ForeignKey('scenario.id'), nullable=False)
    scenario_info = relationship("Scenario", back_populates="timeseries_data")

    def __repr__(self):
        return f'<Timeseries {self.component}_{self.zone}.{self.parameter} at {self.time}>'


class Plan(Base):
    """Table for plan data linked to scenarios"""
    __tablename__ = 'plan'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(500), nullable=True)
    zone: Mapped[str] = mapped_column(String(500), nullable=True)
    indicator: Mapped[str] = mapped_column(String(500), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=True)
    unit: Mapped[str] = mapped_column(String(50), nullable=True)
    alias: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Foreign key to Scenario
    scenario_id: Mapped[int] = mapped_column(Integer, ForeignKey('scenario.id'), nullable=False)
    scenario_info = relationship("Scenario", back_populates="plan_data")
    
    def __repr__(self):
        return f'<Plan {self.component}_{self.indicator}_{self.alias}>'


class DatabaseManager:
    """Manages database operations using SQLAlchemy"""
    
    def __init__(self, database_uri: str):
        """
        Initialize database manager
        
        Args:
            database_uri (str): Database connection URI (e.g., 'postgresql://user:pass@host:port/db')
        """
        self.engine = create_engine(database_uri, echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    @staticmethod
    def ensure_db_config(cfg: dict):
        missing = [k for k, v in cfg.items() if v in (None, "")]
        if missing:
            raise RuntimeError(f"Missing DB config for: {', '.join(missing)}. "
                               f"Check your .env (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS).")

    @staticmethod
    def ensure_database_exists(cfg: dict, default_db: str = "postgres", verbose: bool = True) \
            -> bool:

        conn = pg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["username"],
            password=cfg["password"],
            dbname=default_db,
        )

        # Check database exists
        try:
            conn.autocommit = True

            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (cfg["database"],),)
                exists = cur.fetchone() is not None

                if exists:
                    if verbose:
                        print(f"✓ Database '{cfg["database"]}' already exists.")
                    return True

                if verbose:
                    print(f"✖ Database '{cfg['database']}' does not exist yet. Creating it...")
                cur.execute(f'CREATE DATABASE "{cfg['database']}";')
                if verbose:
                    print(f"Database '{cfg['database']}' created.")
                return False
        finally:
            conn.close()

    def create_tables(self):
        """Create all tables defined in the models"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    def get_or_create_scenario(self, metadata: dict, session) -> int:
        """Get existing scenario or create new one, return scenario ID"""
        scenario = session.query(Scenario).filter_by(
            duration=metadata['Duration'],
            scenario=metadata['Scenario']
        ).first()

        if scenario:
            logger.info(f"Found existing scenario: {scenario}")
            return scenario.id
        else:
            new_scenario = Scenario(
                duration=metadata['Duration'],
                scenario=metadata['Scenario']
            )
            session.add(new_scenario)
            session.commit()
            logger.info(f"Created new scenario: {new_scenario}")
            return new_scenario.id

    def _build_scenario_ids(self, session, df: pd.DataFrame) -> dict:
        """Return { (Duration, Scenario): scenario_id } for df."""
        scenario_ids = {}
        keys = ["Duration", "Scenario"]
        for _, row in df.drop_duplicates(subset=keys).iterrows():
            key = (row["Duration"], row["Scenario"])
            scenario_ids[key] = self.get_or_create_scenario(
                {
                    "Duration": row["Duration"],
                    "Scenario": row["Scenario"]
                },
                session,
            )
        return scenario_ids

    def _scenario_ids_with_existing_data(self, session, table) -> set[int]:
        return {row[0] for row in session.query(table.scenario_id).distinct().all()}

    def _bulk_insert_execute_values(self, table, rows, columns, page_size=50000):
        """
        Fast bulk insert using psycopg2.extras.execute_values.
        `rows`: list[tuple] aligned to `columns`.
        """
        if not rows:
            return
        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                template = "(" + ",".join(["%s"] * len(columns)) + ")"
                sql = f"INSERT INTO {table.name} ({', '.join(columns)}) VALUES %s"
                execute_values(cur, sql, rows, template=template, page_size=page_size)
            raw_conn.commit()
        finally:
            raw_conn.close()

    def _delete_existing_for_scenarios(self, session, table, scenario_ids: set):
        """Delete all rows in `table` for the provided scenario IDs."""
        from sqlalchemy import delete
        if not scenario_ids:
            return
        session.execute(delete(table).where(table.scenario_id.in_(list(scenario_ids))))
        session.commit()

    def insert_plan_data(self, df: pd.DataFrame, batch_size: int = 5000, mode: str = "replace"):
        """
        Inserts rows into Plan.

        Expects df columns:
        Model, Indicator, Value, Unit, Alias,
        Duration, Scenario
        """
        session = self.Session()
        try:
            if "Value" in df.columns:
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

            total_rows = len(df)
            logger.info(f"Preparing to insert {total_rows} plan rows (mode={mode})...")

            # Resolve scenario IDs
            scenario_ids_map = self._build_scenario_ids(session, df)
            involved_ids = set(scenario_ids_map.values())

            # REPLACE mode: delete existing rows for these scenarios
            if mode == "replace":
                logger.info(f"Deleting existing plan rows for {len(involved_ids)} scenario(s)...")
                self._delete_existing_for_scenarios(session, Plan, involved_ids)
            elif mode == "skip":
                existing = self._scenario_ids_with_existing_data(session, Plan)
                keep_ids = involved_ids - existing
                if not keep_ids:
                    logger.info(f"All scenarios have already been added to the database; skipping insert")
                    return
                keys = ["Duration", "Scenario"]
                df = df[df.apply(lambda r: scenario_ids_map[(r["Duration"], r["Scenario"])] in keep_ids, axis=1)]
                logger.info(f"Skipping existing scenarios. Inserting rows for {len(keep_ids)} new scenario(s).")

            # IMPORTANT: recompute total_rows if you filtered df in 'skip' mode
            total_rows = len(df)

            # Columns in the target table, and rows as tuples aligned to those columns
            cols = ["component", "zone", "indicator", "value", "unit", "alias", "scenario_id"]

            # Build all rows (tuples) once; itertuples is faster than iterrows
            rows_all = []
            for r in df.itertuples(index=False):
                sid = scenario_ids_map[(r.Duration, r.Scenario)]
                rows_all.append((
                    str(r.Component),
                    str(r.Zone),
                    str(r.Indicator),
                    None if pd.isna(r.Value) else float(r.Value),
                    None if ("Unit" not in df.columns or pd.isna(r.Unit)) else str(r.Unit),
                    None if ("Alias" not in df.columns or pd.isna(r.Alias)) else str(r.Alias),
                    int(sid),
                ))

            # Chunk + bulk insert
            for i in range(0, len(rows_all), batch_size):
                chunk = rows_all[i:i + batch_size]
                self._bulk_insert_execute_values(Plan.__table__, chunk, cols, page_size=batch_size)
                logger.info(f"Inserted plan rows {i + 1}–{i + len(chunk)}")

            logger.info("Plan insert complete.")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert plan data: {e}")
            raise
        finally:
            session.close()

    def insert_timeseries_data(self, df: pd.DataFrame, batch_size: int = 50000, mode: str = "replace"):
        """
        Inserts rows into Timeseries.

        Expects df columns:
        Time (float), Component, Zone, Parameter, Value,
        Duration, Scenario
        """
        session = self.Session()
        try:
            # Coerce numeric columns
            if "Value" in df.columns:
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

            total_rows = len(df)
            logger.info(f"Preparing to insert {len(df)} timeseries rows (mode={mode})...")

            # Resolve scenario IDs
            scenario_ids_map = self._build_scenario_ids(session, df)
            involved_ids = set(scenario_ids_map.values())

            # REPLACE mode: delete existing rows for these scenarios
            if mode == "replace":
                logger.info(f"Deleting existing timeseries rows for {len(involved_ids)} scenario(s)...")
                self._delete_existing_for_scenarios(session, Timeseries, involved_ids)
            elif mode == "skip":
                existing = self._scenario_ids_with_existing_data(session, Timeseries)
                keep_ids = involved_ids - existing
                if not keep_ids:
                    logger.info("All scenarios have already been added to the database; skipping insert")
                    return
                # filter df to only scenarios that don't already exist
                keys = ["Duration", "Scenario"]
                df = df[df.apply(lambda r: scenario_ids_map[(r["Duration"], r["Scenario"])] in keep_ids, axis=1)]
                logger.info(f"Skipping existing scenarios. Inserting rows for {len(keep_ids)} new scenario(s).")

            # Prepare rows for bulk insert
            cols = ["time", "component", "zone", "parameter", "value", "scenario_id"]
            rows_all = [
                (
                    row.Time,  # leave as is if already datetime
                    str(row.Component),
                    str(row.Zone),
                    str(row.Parameter),
                    None if pd.isna(row.Value) else float(row.Value),
                    int(scenario_ids_map[(row.Duration, row.Scenario)])
                )
                for row in df.itertuples(index=False)
            ]

            # Perform bulk insert
            for i in range(0, len(rows_all), batch_size):
                chunk = rows_all[i:i + batch_size]
                self._bulk_insert_execute_values(Timeseries.__table__, chunk, cols, page_size=batch_size)
                logger.info(f"Inserted timeseries rows {i + 1}–{i + len(chunk)}")

            logger.info("Timeseries insert complete.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert timeseries data: {e}")
            raise
        finally:
            session.close()

    def close(self):
        """Close engine (not strictly required, SQLAlchemy handles cleanup)"""
        self.engine.dispose()
        logger.info("Database connection closed")


def create_connection_string(host: str, port: int, database: str, username: str, password: str) -> str:
    """Create PostgreSQL connection string"""
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"


if __name__ == "__main__":
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'yh2_files',
        'username': 'your_username',
        'password': 'your_password'
    }

    conn_string = create_connection_string(**DB_CONFIG)
    db_manager = DatabaseManager(conn_string)

    try:
        db_manager.create_tables()
    finally:
        db_manager.close()
