# alembic/env.py

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.database import get_base_metadata
import app.models

config = context.config

fileConfig(config.config_file_name)

target_metadata = get_base_metadata()

url = os.getenv('DATABASE_URL')
if not url:
    raise ValueError("DATABASE_URL is not set in environment variables.")

def run_migrations_offline():
    """Run migrations in offline mode."""
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in online mode."""
    connectable = engine_from_config(
        {
            'sqlalchemy.url': url,
            'sqlalchemy.connect_args': {"options": "-c client_encoding=utf8"}
        },
        prefix='sqlalchemy.',
        poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
