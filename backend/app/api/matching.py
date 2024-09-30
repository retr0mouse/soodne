from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Tuple
from .. import schemas, crud
from ..database import SessionLocal
from ..ai.matcher import match_products

router = APIRouter(
    prefix="/matching",
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/match", response_model=schemas.ProductMatchingLog)
def match_two_products(product_store_id1: int, product_store_id2: int, db: Session = Depends(get_db)):
    psd1 = crud.get_product_store_data(db, product_store_id=product_store_id1)
    psd2 = crud.get_product_store_data(db, product_store_id=product_store_id2)
    if not psd1 or not psd2:
        raise HTTPException(status_code=404, detail="One or both ProductStoreData not found")
    
    name1 = psd1.store_product_name or psd1.product.name
    name2 = psd2.store_product_name or psd2.product.name

    matched, confidence = match_products(name1, name2)
    
    if matched:
        psd1.matching_status = schemas.MatchingStatusEnum.matched
        psd2.matching_status = schemas.MatchingStatusEnum.matched
        db.commit()
        
        log = schemas.ProductMatchingLogCreate(
            product_store_id=product_store_id1,
            product_id=psd2.product_id,
            confidence_score=confidence,
            matched_by="simple_matcher"
        )
        return crud.create_product_matching_log(db=db, log=log)
    else:
        return schemas.ProductMatchingLog(
            log_id=0, 
            product_store_id=product_store_id1, 
            product_id=None, 
            confidence_score=confidence, 
            matched_at=None, 
            matched_by="simple_matcher"
        )
