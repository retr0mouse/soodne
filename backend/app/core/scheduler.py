from apscheduler.schedulers.background import BackgroundScheduler
from app.core.jobs import scheduled_job
import logging

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

scheduler.add_job(scheduled_job, 'cron', hour=1, id='scheduled_job')
