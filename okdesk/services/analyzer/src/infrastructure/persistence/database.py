"""Database connection and session management."""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ...config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_size=settings.db_pool_size,
    max_overflow=20,
)

# Create async session factory
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for SQLAlchemy models
Base = declarative_base()


async def get_db_session() -> AsyncSession:
    """
    Get database session.

    Yields:
        AsyncSession instance
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
