from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from app import schemas
from app.services import product_store_data_service, product_matching_log_service
from app.ai.matcher import match_products
from app.api import deps
from app.core.jobs import scheduled_job

router = APIRouter()

@router.post("/match", response_model=schemas.ProductMatchingLog)
def match_two_products(
        product_store_id1: int,
        product_store_id2: int,
        db: Session = Depends(deps.get_db)
):
    psd1 = product_store_data_service.get(db, product_store_id=product_store_id1)
    psd2 = product_store_data_service.get(db, product_store_id=product_store_id2)
    if not psd1 or not psd2:
        raise HTTPException(status_code=404, detail="ProductStoreData not found")

    matched, confidence = match_products(psd1, psd2)

    if matched:
        psd1.matching_status = schemas.MatchingStatusEnum.matched
        psd2.matching_status = schemas.MatchingStatusEnum.matched
        db.commit()

        log = schemas.ProductMatchingLogCreate(
            product_store_id=product_store_id1,
            product_id=psd2.product_id,
            confidence_score=confidence,
            matched_by="api_matcher"
        )
        matching_log = product_matching_log_service.create(db, log=log)
        return matching_log
    else:
        return schemas.ProductMatchingLog(
            log_id=0,
            product_store_id=product_store_id1,
            product_id=None,
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
        raise HTTPException(status_code=500, detail=f"Error while starting parsing: {str(e)}")

@router.post("/run-scheduled-job", response_model=schemas.JobStatus)
def run_scheduled_job(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(scheduled_job)
        return schemas.JobStatus(message="Scheduled task started successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while starting scheduled task: {str(e)}")