# app/core/logger.py

import logging
import logging.config
import os

# Загрузка конфигурации логирования из файла logging.conf
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'logging.conf'), disable_existing_loggers=False)

# Создание логгера для использования в других модулях
logger = logging.getLogger(__name__)
