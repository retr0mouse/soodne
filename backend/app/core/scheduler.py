from apscheduler.schedulers.background import BackgroundScheduler
from app.scraper.scraper import scrape_store_products
from app.ai.matcher import run_ai_matching

scheduler = BackgroundScheduler()

def scheduled_job():
    scrape_store_products()
    run_ai_matching()

scheduler.add_job(scheduled_job, 'cron', hour=1)