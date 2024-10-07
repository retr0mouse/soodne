# alembic/env.py

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Добавляем путь к корню проекта в sys.path для корректного импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем метаданные базы данных
from app.database.database import get_base_metadata
import app.models  # Импортируйте все ваши модели здесь

# Получаем конфигурацию Alembic
config = context.config

# Читаем настройки логирования из файла конфигурации
fileConfig(config.config_file_name)

# Устанавливаем метаданные для автогенерации
target_metadata = get_base_metadata()

# Получаем URL базы данных из переменных окружения
url = os.getenv('DATABASE_URL')
if not url:
    raise ValueError("DATABASE_URL не установлена в переменных окружения.")

def run_migrations_offline():
    """Запуск миграций в офлайн-режиме."""
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Запуск миграций в онлайн-режиме."""
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
