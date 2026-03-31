import os
import logging
from sqlalchemy import create_engine, String, Float, Integer, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_models.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Build engine kwargs depending on the DB backend
_is_sqlite = "sqlite" in DATABASE_URL

if _is_sqlite:
    _connect_args: dict = {"check_same_thread": False}
    _engine_kwargs: dict = {}
else:
    _connect_args = {
        "connect_timeout": 10,
        "sslmode": "require",
    }
    _engine_kwargs = {
        "pool_pre_ping": True,      # recycle stale connections automatically
        "pool_recycle": 300,        # recycle connections every 5 min
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    **_engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SavedModel(Base):
    """Stores trained agent checkpoints with training statistics."""
    __tablename__ = "saved_models"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    data: Mapped[bytes] = mapped_column()  # serialized agent stored as binary blob

    # Training statistics
    algorithm: Mapped[str] = mapped_column(String, default="PPO")
    pseudo_epsilon: Mapped[float] = mapped_column(Float, default=1.0)
    ppo_updates: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    total_episodes: Mapped[int] = mapped_column(Integer, default=0)
    avg_reward: Mapped[float] = mapped_column(Float, default=0.0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 to 1.0
    environment_config: Mapped[str] = mapped_column(String, default="{}")


def _ensure_saved_models_columns():
    """Add missing columns for existing DBs without requiring manual migrations."""
    insp = inspect(engine)
    if "saved_models" not in insp.get_table_names():
        return

    existing = {c["name"] for c in insp.get_columns("saved_models")}
    statements: list[str] = []

    if "algorithm" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN algorithm VARCHAR DEFAULT 'PPO'")
    if "pseudo_epsilon" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN pseudo_epsilon FLOAT DEFAULT 1.0")
    if "ppo_updates" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN ppo_updates INTEGER DEFAULT 0")
    if "total_steps" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN total_steps INTEGER DEFAULT 0")
    if "total_episodes" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN total_episodes INTEGER DEFAULT 0")
    if "avg_reward" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN avg_reward FLOAT DEFAULT 0.0")
    if "success_rate" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN success_rate FLOAT DEFAULT 0.0")
    if "environment_config" not in existing:
        statements.append("ALTER TABLE saved_models ADD COLUMN environment_config VARCHAR DEFAULT '{}'")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    logger.info("saved_models schema upgraded with %d missing column(s).", len(statements))
def init_db():
    """Create tables.  Non-fatal on failure so the HTTP server can still start."""
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_saved_models_columns()
        logger.info("Database tables created / verified.")
    except Exception as exc:
        logger.warning("init_db failed (will retry on first request): %s", exc)
