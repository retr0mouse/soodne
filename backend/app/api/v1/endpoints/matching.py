from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from app import schemas
from app.services import product_service, product_matching_log_service
from app.ai.matcher import match_products
from app.api import deps
from app.core.jobs import scheduled_job
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/match", response_model=schemas.ProductMatchingLog)
def match_two_products(
        product_id1: int,
        product_id2: int,
        db: Session = Depends(deps.get_db)
):
    product1 = product_service.get(db, product_id=product_id1)
    product2 = product_service.get(db, product_id=product_id2)
    if not product1 or not product2:
        raise HTTPException(status_code=404, detail="Product not found")

    matched, confidence = match_products(product1, product2)

    if matched:
        product1.matching_status = schemas.MatchingStatusEnum.matched
        product2.matching_status = schemas.MatchingStatusEnum.matched
        db.commit()

        log = schemas.ProductMatchingLogCreate(
            product_id1=product_id1,
            product_id2=product_id2,
            confidence_score=confidence,
            matched_by="api_matcher"
        )
        matching_log = product_matching_log_service.create(db, log=log)
        return matching_log
    else:
        return schemas.ProductMatchingLog(
            log_id=0,
            product_id1=product_id1,
            product_id2=product_id2,
            confidence_score=confidence,
            matched_at=None,
            matched_by="api_matcher"
        )

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