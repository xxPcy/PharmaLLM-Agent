import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from pharma_agent.db import Base
from pharma_agent.settings import Settings

config = context.config
target_metadata = Base.metadata


def do_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def online():
    engine = create_async_engine(Settings().database_url)
    async with engine.connect() as connection:
        await connection.run_sync(do_migrations)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=Settings().database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())
