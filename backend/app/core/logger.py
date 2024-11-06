import logging
import logging.config
import os

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'logging.conf'), disable_existing_loggers=False)

logger = logging.getLogger(__name__)
