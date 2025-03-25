from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app import schemas
from app.api import deps
from app.core.jobs import scheduled_job
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/run-parsing")
async def run_parsing():
    try:
        from app.scraper.scraper import scrape_store_products
        scrape_store_products()
        return {"message": "Parsing started successfully!"}
    except Exception as e:
        logger.error(f"Error starting parsing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error while starting parsing: {str(e)}")

@router.post("/run-scheduled-job", response_model=schemas.JobStatus)
def run_scheduled_job(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(scheduled_job)
        return schemas.JobStatus(message="Scheduled task started successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while starting scheduled task: {str(e)}")

@router.get("/match-all", response_model=schemas.JobStatus)
def match_all_products(db: Session = Depends(deps.get_db)):
    try:
        from app.ai.matcher import run_matching
        result = run_matching(db)
        if isinstance(result, dict):
            return schemas.JobStatus(
                message=f"AI matching completed: {result['matched']} matched, {result['created']} created out of {result['processed']} processed"
            )
        return schemas.JobStatus(message="AI matching process completed successfully!")
    except Exception as e:
        logger.error(f"Error during AI matching: {str(e)}", exc_info=True)
        db.rollback()
        return schemas.JobStatus(
            message=f"Error during AI matching: {str(e)}",
            status="error"
        )