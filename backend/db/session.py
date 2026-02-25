"""SQLAlchemy async engine and session factory."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from loguru import logger
from backend.config import get_settings

_engine = None
_session_factory = None


async def init_db():
    global _engine, _session_factory
    settings = get_settings()
    try:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DEBUG,
            pool_size=5,
            max_overflow=10,
        )
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Test connection
        async with _engine.begin() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        logger.success("Database connected.")
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Running without DB.")
        _engine = None
        _session_factory = None


async def get_session():
    if _session_factory is None:
        yield None
        return
    async with _session_factory() as session:
        yield session


def get_engine():
    return _engine
