"""SQLAlchemy async engine and session factory."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
from loguru import logger
from backend.config import get_settings

_engine = None
_session_factory = None


async def init_db():
    """Initialise the async DB engine. Degrades gracefully when the DB is unreachable."""
    global _engine, _session_factory
    settings = get_settings()
    try:
        # pool_size / max_overflow are NOT valid for async engines using asyncpg.
        # Use NullPool in test/dev environments; in production set via DATABASE_URL params.
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            poolclass=NullPool,  # Each connection is opened/closed individually — safe for async
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Smoke-test the connection
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.success("Database connected.")

        # Optional dev convenience: create ORM tables without running Alembic.
        if settings.DB_AUTO_CREATE:
            from backend.db.models import Base
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("DB_AUTO_CREATE enabled — ORM tables ensured.")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Running without DB.")
        _engine = None
        _session_factory = None


async def get_session():
    """Yield an async DB session, or yield None when DB is unavailable."""
    if _session_factory is None:
        yield None
        return
    async with _session_factory() as session:
        yield session


def get_engine():
    return _engine
